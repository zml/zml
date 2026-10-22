const std = @import("std");

const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const constants = @import("../constants.zig");
const TensorStore = @import("TensorStore.zig");
const buffered_loader = @import("buffered_loader.zig");
const direct_loader = @import("direct_loader.zig");
const dma_calibration = @import("dma_calibration.zig");

pub const SubmitError = direct_loader.Loader.SubmitError || buffered_loader.Loader.SubmitError;
pub const AwaitError = direct_loader.Loader.AwaitError || buffered_loader.Loader.AwaitError;

/// The largest source request, and so the largest DMA block: a request is
/// a whole number of blocks, at least one.
pub const max_read_request_size: usize = 32 * 1024 * 1024;

/// `Loader.Options` as `Loader.init` checked them, the checkpoint resolved
/// to its VFS backend's read hints. See `Loader.Options` for each field.
pub const Options = struct {
    read_parallelism: ?usize = null,
    load_profile: VFS.LoadProfile = .local,
    dma_block_sizes: []const usize = &dma_calibration.default_block_sizes,
    direct_io: VFS.DirectIo = .auto,

    /// Concurrent source reads by latency class, from the recorded
    /// width sweeps. Local sources knee at 12 to 16 on every host
    /// measured: the B70 peaks at 12 to 16 and loses 11% at 24, one MI300X
    /// prefers 12, the RTX 5090 host 16 over 12, and the GB300 plateau from 12
    /// to 48 has 16 within 5% of its best rung, so 16 is the width with the
    /// smallest regret. Remote sources plateau from 24 to 32: hf:// reads 32
    /// and 64 alike and AWS was flat from 24, while pinned memory grows with
    /// the width (32 x 32 MiB requests pin 1.5 GiB).
    const local_read_parallelism: usize = 16;
    const high_latency_read_parallelism: usize = 32;

    /// The width the backend uses: the option, or the profile's default.
    pub fn readWidth(self: Options) usize {
        return self.read_parallelism orelse if (self.load_profile.high_latency) high_latency_read_parallelism else local_read_parallelism;
    }
};

/// One destination, its caller-owned output and the source ranges that fill
/// it. A backend copies each spec and its parts during submission; the
/// sources and the output stay alive until await.
pub const LoadSpec = struct {
    /// The destination's global shape, which the output buffer keeps. Its
    /// sharding places the bytes on the devices; a shape without a mesh is
    /// replicated over every device (`placedShape`).
    shape: Shape,
    /// The memory space of each device that receives the bytes, from the
    /// source's `TensorStore` binding. The front end only submits kinds
    /// `supportsMemory` accepts.
    memory: Memory.Kind = .default,
    /// Written whole once the destination's buffers exist, and not touched
    /// before: a failure leaves it as it was or holding those buffers.
    output: *Buffer,
    /// What fills the destination's logical bytes, in ascending, disjoint
    /// destination order (`checkParts`): one whole source for a plain
    /// tensor, one range per source for a composite one.
    parts: []const LoadPart,
    /// Whether bytes no part covers are zero. Without it the parts must
    /// cover every byte, so a source of the wrong size is
    /// `error.SourceSizeMismatch`, not a silently padded tensor.
    zero_fill: bool = false,
};

/// `width` bytes of `source` from `src_offset` (relative to the tensor's
/// data) land at `dst_offset` of the destination's logical bytes: the
/// row-major bytes of its packed shape, before the sharding splits them.
/// With `dims` the part is strided: that run repeats for every index of
/// `dims`, each step adding its strides to both offsets (`Lattice`).
pub const LoadPart = struct {
    source: *safetensors.Tensor,
    src_offset: u64 = 0,
    dst_offset: u64 = 0,
    width: u64,
    /// Outermost first; empty for one contiguous run.
    dims: []const Dim = &.{},

    pub const Dim = TensorStore.Part.Dim;

    /// The whole of `source`, at the start of the destination.
    pub fn whole(source: *safetensors.Tensor) LoadPart {
        return .{ .source = source, .width = source.byteSize() };
    }

    pub fn lattice(self: LoadPart) Lattice {
        return .{ .src_offset = self.src_offset, .dst_offset = self.dst_offset, .width = self.width, .dims = self.dims };
    }
};

/// The most dimensions a part repeats over: `Source` gives each axis at
/// most two.
pub const max_part_dims = 2 * constants.MAX_RANK;

/// A strided part's byte moves: the run of `width` bytes at `src_offset`
/// and `dst_offset`, and one more for every index of `dims` (outermost
/// first), each offset moved by the index times its strides. A zero source
/// stride repeats the same bytes. Runs are numbered row-major over `dims`.
pub const Lattice = struct {
    src_offset: u64,
    dst_offset: u64,
    width: u64,
    dims: []const LoadPart.Dim = &.{},

    pub const Run = struct { src: u64, dst: u64 };

    pub fn runs(self: Lattice) u64 {
        var n: u64 = 1;
        for (self.dims) |d| n *= d.count;
        return n;
    }

    pub fn at(self: Lattice, index: u64) Run {
        var rest = index;
        var run: Run = .{ .src = self.src_offset, .dst = self.dst_offset };
        var d = self.dims.len;
        while (d > 0) {
            d -= 1;
            const i = rest % self.dims[d].count;
            rest /= self.dims[d].count;
            run.src += i * self.dims[d].src_stride;
            run.dst += i * self.dims[d].dst_stride;
        }
        return run;
    }

    /// The source and destination bytes the runs span from their first
    /// offsets, null when a count is zero or the bounds overflow.
    pub fn spans(self: Lattice) ?struct { src: u64, dst: u64 } {
        var src = self.width;
        var dst = self.width;
        for (self.dims) |d| {
            if (d.count == 0) return null;
            src = std.math.add(u64, std.math.mul(u64, d.count - 1, d.src_stride) catch return null, src) catch return null;
            dst = std.math.add(u64, std.math.mul(u64, d.count - 1, d.dst_stride) catch return null, dst) catch return null;
        }
        return .{ .src = src, .dst = dst };
    }

    /// Whether each dimension steps past everything inside it on the
    /// destination side: runs land in index order without overlapping.
    pub fn dstOrdered(self: Lattice) bool {
        return ordered(self.dims, self.width, .dst, false);
    }

    /// Whether the source runs come in index order without overlapping,
    /// gaps allowed: the runs of an index range lie between the first's
    /// start and the last's end.
    pub fn srcOrdered(self: Lattice) bool {
        return ordered(self.dims, self.width, .src, false);
    }

    /// Whether the source runs are back to back in index order: an index
    /// range reads one contiguous source range, each byte once.
    pub fn srcTiled(self: Lattice) bool {
        return ordered(self.dims, self.width, .src, true);
    }

    fn ordered(dims: []const LoadPart.Dim, width: u64, comptime side: enum { src, dst }, exact: bool) bool {
        var extent = width;
        var d = dims.len;
        while (d > 0) {
            d -= 1;
            if (dims[d].count < 2) continue;
            const stride = if (side == .src) dims[d].src_stride else dims[d].dst_stride;
            if (if (exact) stride != extent else stride < extent) return false;
            extent += (dims[d].count - 1) * stride;
        }
        return true;
    }

    /// The first run, in index order, that ends after destination offset
    /// `offset`; `runs()` when none does. Needs `dstOrdered`.
    pub fn firstEndingAfter(self: Lattice, offset: u64) u64 {
        var low: u64 = 0;
        var high = self.runs();
        while (low < high) {
            const middle = low + (high - low) / 2;
            if (self.at(middle).dst + self.width > offset) high = middle else low = middle + 1;
        }
        return low;
    }

    /// The first run that starts at destination offset `offset` or later;
    /// `runs()` when none does. Needs `dstOrdered`.
    pub fn firstStartingFrom(self: Lattice, offset: u64) u64 {
        var low: u64 = 0;
        var high = self.runs();
        while (low < high) {
            const middle = low + (high - low) / 2;
            if (self.at(middle).dst >= offset) high = middle else low = middle + 1;
        }
        return low;
    }

    /// Whether a run has a byte in the destination range `[start, end)`.
    pub fn hasRunIn(self: Lattice, start: u64, end: u64) bool {
        const first = self.firstEndingAfter(start);
        return first < self.runs() and self.at(first).dst < end;
    }

    /// The destination bytes of the runs inside `[start, end)`.
    pub fn coveredIn(self: Lattice, start: u64, end: u64) u64 {
        const first = self.firstEndingAfter(start);
        const last = self.firstStartingFrom(end);
        if (first >= last) return 0;
        var covered = (last - first) * self.width;
        const head = self.at(first).dst;
        if (head < start) covered -= start - head;
        const tail = self.at(last - 1).dst + self.width;
        if (tail > end) covered -= tail - end;
        return covered;
    }

    /// Runs `first` to `end` (exclusive), in index order, without a
    /// division per run.
    pub const Cursor = struct {
        lattice: Lattice,
        index: u64,
        end: u64,
        counters: [max_part_dims]u64,
        run: Run,

        pub fn init(lattice: Lattice, first: u64, end: u64) Cursor {
            std.debug.assert(lattice.dims.len <= max_part_dims);
            var cursor: Cursor = .{ .lattice = lattice, .index = first, .end = end, .counters = undefined, .run = lattice.at(first) };
            var rest = first;
            var d = lattice.dims.len;
            while (d > 0) {
                d -= 1;
                cursor.counters[d] = rest % lattice.dims[d].count;
                rest /= lattice.dims[d].count;
            }
            return cursor;
        }

        pub fn next(self: *Cursor) ?Run {
            if (self.index >= self.end) return null;
            const current = self.run;
            self.index += 1;
            if (self.index < self.end) {
                const dims = self.lattice.dims;
                var d = dims.len;
                while (d > 0) {
                    d -= 1;
                    self.counters[d] += 1;
                    self.run.src +%= dims[d].src_stride;
                    self.run.dst +%= dims[d].dst_stride;
                    if (self.counters[d] < dims[d].count) break;
                    self.counters[d] = 0;
                    self.run.src -%= dims[d].count * dims[d].src_stride;
                    self.run.dst -%= dims[d].count * dims[d].dst_stride;
                }
            }
            return current;
        }
    };

    /// Copies runs `first` to `end` (exclusive): the `width` bytes at
    /// `src + run.src` to `dst + run.dst`, `src` and `dst` being the
    /// addresses the offsets count from (wrapping). Whole rows of the two
    /// innermost dimensions go through nested loops, and a run of 1, 2, 4,
    /// 8 or 16 bytes is one load and one store: a swizzle's 4-byte runs
    /// cost a few cycles each instead of a cursor step and a `memcpy`.
    pub fn copyRuns(self: Lattice, first: u64, end: u64, dst: usize, src: usize) void {
        switch (self.width) {
            inline 1, 2, 4, 8, 16 => |width| self.copyRunsOf(width, first, end, dst, src),
            else => self.copyRunsOf(0, first, end, dst, src),
        }
    }

    /// `copyRuns` for runs of `width` bytes, or of `self.width` when zero.
    fn copyRunsOf(self: Lattice, comptime width: usize, first: u64, end: u64, dst: usize, src: usize) void {
        if (first >= end) return;
        const inner = self.dims[self.dims.len - @min(self.dims.len, 2) ..];
        var row: u64 = 1;
        for (inner) |d| row *= d.count;
        const rows_start = std.math.divCeil(u64, first, row) catch unreachable;
        const rows_end = end / row;
        if (rows_start >= rows_end) return self.copyEach(width, first, end, dst, src);
        self.copyEach(width, first, rows_start * row, dst, src);
        const outer: Lattice = .{ .src_offset = self.src_offset, .dst_offset = self.dst_offset, .width = self.width, .dims = self.dims[0 .. self.dims.len - inner.len] };
        var rows: Cursor = .init(outer, rows_start, rows_end);
        while (rows.next()) |start| {
            const row_dst = dst +% start.dst;
            const row_src = src +% start.src;
            switch (inner.len) {
                0 => copyRun(width, self.width, row_dst, row_src),
                1 => copyLine(width, self.width, inner[0], row_dst, row_src),
                else => {
                    var line_dst = row_dst;
                    var line_src = row_src;
                    for (0..inner[0].count) |_| {
                        copyLine(width, self.width, inner[1], line_dst, line_src);
                        line_dst +%= inner[0].dst_stride;
                        line_src +%= inner[0].src_stride;
                    }
                },
            }
        }
        self.copyEach(width, rows_end * row, end, dst, src);
    }

    fn copyEach(self: Lattice, comptime width: usize, first: u64, end: u64, dst: usize, src: usize) void {
        if (first >= end) return;
        var cursor: Cursor = .init(self, first, end);
        while (cursor.next()) |run| copyRun(width, self.width, dst +% run.dst, src +% run.src);
    }

    inline fn copyLine(comptime width: usize, len: u64, d: LoadPart.Dim, dst: usize, src: usize) void {
        var to = dst;
        var from = src;
        for (0..d.count) |_| {
            copyRun(width, len, to, from);
            to +%= d.dst_stride;
            from +%= d.src_stride;
        }
    }

    inline fn copyRun(comptime width: usize, len: u64, dst: usize, src: usize) void {
        if (width == 0) {
            const n: usize = @intCast(len);
            @memcpy(@as([*]u8, @ptrFromInt(dst))[0..n], @as([*]const u8, @ptrFromInt(src))[0..n]);
        } else {
            const Word = @Int(.unsigned, 8 * width);
            @as(*align(1) Word, @ptrFromInt(dst)).* = @as(*align(1) const Word, @ptrFromInt(src)).*;
        }
    }
};

pub const PartsError = error{ SourceSizeMismatch, InvalidLayout, UnsupportedLayout };

/// What both backends rely on from a spec's parts: there is at least one,
/// each is non-empty and inside its source, they come in ascending
/// destination order without overlapping, and they stay inside the
/// destination's `destination_bytes`. Without `zero_fill` they must cover
/// every destination byte; a plain tensor whose source is larger or smaller
/// than its shape fails there, with `SourceSizeMismatch`. Strided parts must
/// land in index order (`Lattice.dstOrdered`). Where many of them
/// interleave, their runs are listed and sorted when there are few enough
/// (`allocator` holds them while checking); otherwise two interleaved
/// parts are proven disjoint only when they repeat with the same strides
/// (`overlaps`), other pairs are `UnsupportedLayout`. Every other violation
/// is `InvalidLayout`.
pub fn checkParts(allocator: std.mem.Allocator, parts: []const LoadPart, destination_bytes: u64, zero_fill: bool) PartsError!void {
    if (parts.len == 0) return error.InvalidLayout;
    for (parts) |part| {
        if (part.dims.len != 0) return checkStridedParts(allocator, parts, destination_bytes, zero_fill);
    }
    var end: u64 = 0;
    var covered: u64 = 0;
    for (parts) |part| {
        const source_end = std.math.add(u64, part.src_offset, part.width) catch return error.InvalidLayout;
        const part_end = std.math.add(u64, part.dst_offset, part.width) catch return error.InvalidLayout;
        if (part.width == 0 or source_end > part.source.byteSize() or part.dst_offset < end) return error.InvalidLayout;
        end = part_end;
        covered += part.width;
    }
    if (!zero_fill and covered != destination_bytes) return error.SourceSizeMismatch;
    if (end > destination_bytes) return error.InvalidLayout;
}

fn checkStridedParts(allocator: std.mem.Allocator, parts: []const LoadPart, destination_bytes: u64, zero_fill: bool) PartsError!void {
    var covered: u64 = 0;
    var previous_start: u64 = 0;
    for (parts) |part| {
        const lattice = part.lattice();
        if (part.width == 0 or part.dims.len > max_part_dims or !lattice.dstOrdered()) return error.InvalidLayout;
        const spans = lattice.spans() orelse return error.InvalidLayout;
        const source_end = std.math.add(u64, part.src_offset, spans.src) catch return error.InvalidLayout;
        const part_end = std.math.add(u64, part.dst_offset, spans.dst) catch return error.InvalidLayout;
        if (source_end > part.source.byteSize() or part_end > destination_bytes or part.dst_offset < previous_start) return error.InvalidLayout;
        previous_start = part.dst_offset;
        covered += lattice.runs() * part.width;
    }
    if (covered > destination_bytes) return error.InvalidLayout;
    if (!zero_fill and covered != destination_bytes) return error.SourceSizeMismatch;
    // Parts are sorted by their first byte; a part's bytes stay before its
    // span's end, so only parts of one cluster, whose spans meet, can
    // overlap.
    var start: usize = 0;
    while (start < parts.len) {
        var end = parts[start].dst_offset + parts[start].lattice().spans().?.dst;
        var runs = parts[start].lattice().runs();
        var stop = start + 1;
        while (stop < parts.len and parts[stop].dst_offset < end) : (stop += 1) {
            end = @max(end, parts[stop].dst_offset + parts[stop].lattice().spans().?.dst);
            runs += parts[stop].lattice().runs();
        }
        try checkCluster(allocator, parts[start..stop], runs);
        start = stop;
    }
}

/// The most runs `checkCluster` lists to sort them.
const max_listed_runs = 1 << 21;

/// Whether the parts of a cluster, `runs` runs in all, are disjoint. A few
/// parts (an interleave of members) are checked pair by pair; many that
/// interleave (a swizzle of a column-sharded grid, each part spanning the
/// tensor) would make that quadratic, so their runs are sorted instead when
/// they are few enough to list.
fn checkCluster(allocator: std.mem.Allocator, cluster: []const LoadPart, runs: u64) PartsError!void {
    if (cluster.len < 2) return;
    const Range = struct { start: u64, end: u64 };
    if (cluster.len > 8 and runs <= max_listed_runs) {
        if (allocator.alloc(Range, @intCast(runs))) |ranges| {
            defer allocator.free(ranges);
            var count: usize = 0;
            for (cluster) |part| {
                var cursor: Lattice.Cursor = .init(part.lattice(), 0, part.lattice().runs());
                while (cursor.next()) |run| {
                    ranges[count] = .{ .start = run.dst, .end = run.dst + part.width };
                    count += 1;
                }
            }
            std.mem.sort(Range, ranges, {}, struct {
                fn lessThan(_: void, lhs: Range, rhs: Range) bool {
                    return lhs.start < rhs.start;
                }
            }.lessThan);
            for (ranges[1..], ranges[0 .. ranges.len - 1]) |range, previous| {
                if (range.start < previous.end) return error.InvalidLayout;
            }
            return;
        } else |_| {}
    }
    for (cluster, 0..) |part, index| {
        const end = part.dst_offset + part.lattice().spans().?.dst;
        for (cluster[index + 1 ..]) |other| {
            if (other.dst_offset >= end) break;
            if (try overlaps(part, other)) return error.InvalidLayout;
        }
    }
}

/// Whether two strided parts share a destination byte. Exact when one has
/// few runs: each is looked for among the other's (`Lattice.hasRunIn`).
/// Otherwise exact when they repeat with the same strides once their
/// touching runs are merged, or else as they are (`Footprint.overlaps`);
/// any other pair is `UnsupportedLayout`.
fn overlaps(part: LoadPart, other: LoadPart) error{UnsupportedLayout}!bool {
    const few = 1024;
    for ([2]LoadPart{ part, other }, [2]LoadPart{ other, part }) |a, b| {
        const footprint: Footprint = .of(a, true);
        if (footprint.lattice().runs() > few) continue;
        const against: Footprint = .of(b, true);
        var cursor: Lattice.Cursor = .init(footprint.lattice(), 0, footprint.lattice().runs());
        while (cursor.next()) |run| {
            if (against.lattice().hasRunIn(run.dst, run.dst + footprint.width)) return true;
        }
        return false;
    }
    const merged = Footprint.of(part, true).overlaps(Footprint.of(other, true)) catch null;
    return merged orelse Footprint.of(part, false).overlaps(Footprint.of(other, false));
}

/// A part's destination bytes alone, its dimensions of one step dropped,
/// and, when `merge`, those whose runs touch merged into wider runs.
const Footprint = struct {
    start: u64,
    width: u64,
    dims: [max_part_dims]LoadPart.Dim = undefined,
    len: usize = 0,

    fn of(part: LoadPart, merge: bool) Footprint {
        var footprint: Footprint = .{ .start = part.dst_offset, .width = part.width };
        var d = part.dims.len;
        while (d > 0) {
            d -= 1;
            const dim = part.dims[d];
            if (dim.count < 2) continue;
            if (!merge) {
                footprint.dims[footprint.len] = .{ .count = dim.count, .src_stride = 0, .dst_stride = dim.dst_stride };
                footprint.len += 1;
                continue;
            }
            if (footprint.len == 0 and dim.dst_stride == footprint.width) {
                footprint.width *= dim.count;
                continue;
            }
            if (footprint.len != 0) {
                const inner = &footprint.dims[footprint.len - 1];
                if (dim.dst_stride == inner.count * inner.dst_stride) {
                    inner.count *= dim.count;
                    continue;
                }
            }
            footprint.dims[footprint.len] = .{ .count = dim.count, .src_stride = 0, .dst_stride = dim.dst_stride };
            footprint.len += 1;
        }
        std.mem.reverse(LoadPart.Dim, footprint.dims[0..footprint.len]);
        return footprint;
    }

    fn lattice(self: *const Footprint) Lattice {
        return .{ .src_offset = 0, .dst_offset = self.start, .width = self.width, .dims = self.dims[0..self.len] };
    }

    /// Whether some byte is in both: exact for a contiguous footprint
    /// against any, and for two that repeat with the same strides, counts
    /// and widths aside (an interleave of members, however many dimensions
    /// it has).
    fn overlaps(self: Footprint, other: Footprint) error{UnsupportedLayout}!bool {
        if (self.len == 0) return other.lattice().hasRunIn(self.start, self.start + self.width);
        if (other.len == 0) return self.lattice().hasRunIn(other.start, other.start + other.width);
        if (self.len != other.len) return error.UnsupportedLayout;
        for (self.dims[0..self.len], other.dims[0..other.len]) |a, b| {
            if (a.dst_stride != b.dst_stride) return error.UnsupportedLayout;
        }
        // Run i of self and run j of other overlap when the gap between
        // their starts, diff + sum((j - i) * stride), lies in
        // (-other.width, self.width). Each difference m = j - i lies in
        // (-n_self, n_other) on its dimension, any of which some i and j
        // give, so the search is for such an m.
        var search: OverlapSearch = .{ .self = &self, .other = &other };
        const diff: i128 = @as(i128, other.start) - @as(i128, self.start);
        return search.find(0, -@as(i128, other.width) - diff, @as(i128, self.width) - diff);
    }

    /// A depth-first search for steps `m` with `sum(m * stride)` in an open
    /// interval, outermost dimension first. The dimensions are ordered
    /// (`Lattice.dstOrdered`), so the inner ones reach less than a stride
    /// of the outer one and each level has few candidates; the search gives
    /// up past `budget` of them.
    const OverlapSearch = struct {
        self: *const Footprint,
        other: *const Footprint,
        budget: u32 = 1 << 16,

        fn find(search: *OverlapSearch, d: usize, low: i128, high: i128) error{UnsupportedLayout}!bool {
            const len = search.self.len;
            if (d == len) return low < 0 and 0 < high;
            // What dimensions d + 1 on can add, below and above zero.
            var below: i128 = 0;
            var above: i128 = 0;
            for (d + 1..len) |k| {
                const stride: i128 = search.self.dims[k].dst_stride;
                below += (@as(i128, search.self.dims[k].count) - 1) * stride;
                above += (@as(i128, search.other.dims[k].count) - 1) * stride;
            }
            const stride: i128 = search.self.dims[d].dst_stride;
            const first = @max(1 - @as(i128, search.self.dims[d].count), (std.math.divFloor(i128, low - above, stride) catch unreachable) + 1);
            const last = @min(@as(i128, search.other.dims[d].count) - 1, (std.math.divCeil(i128, high + below, stride) catch unreachable) - 1);
            var m = first;
            while (m <= last) : (m += 1) {
                if (search.budget == 0) return error.UnsupportedLayout;
                search.budget -= 1;
                if (try search.find(d + 1, low - m * stride, high - m * stride)) return true;
            }
            return false;
        }
    };
};

/// `shape` with the platform's replicated mesh when it names none, which is
/// how a buffer places an unsharded shape (`PjrtBufferMetadata.init`): every
/// placement the loader computes goes through this, so the bytes it counts,
/// the devices it allocates on and the spans it writes agree.
pub fn placedShape(platform: *const Platform, shape: Shape) Shape {
    var placed = shape;
    if (placed._sharding.mesh == null) placed._sharding.mesh = platform.replicated_mesh;
    return placed;
}

/// Whether the loader can deliver into `kind` on `platform`: every device
/// must expose that memory space, and host-pinned delivery needs the direct
/// pipeline, as on master, where TPU, neuron, metal and furiosa refused it.
/// CPU takes the direct pipeline here, so it accepts `.host_pinned` (its
/// plugin resolves every kind to its one memory space).
pub fn supportsMemory(platform: *const Platform, kind: Memory.Kind) bool {
    if (kind == .default) return true;
    switch (platform.target) {
        .cuda, .rocm, .oneapi, .cpu => {},
        .tpu, .neuron, .metal, .furiosa => if (kind == .host_pinned) return false,
    }
    for (platform.devices) |*device| {
        if (device.memory(kind) == null) return false;
    }
    return true;
}

/// Whether bytes delivered into `kind` take device memory, which program
/// admission measures. Host memory spaces never count against a device.
pub fn countsAgainstDevice(kind: Memory.Kind) bool {
    return switch (kind) {
        .default, .device => true,
        .host_pinned, .host_unpinned => false,
    };
}

/// The transfer implementation selected for the platform.
pub const Backend = union(enum) {
    direct: *direct_loader.Loader,
    buffered: *buffered_loader.Loader,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) !Backend {
        // Transfer-manager support and host pinning are independent. CPU
        // implements byte-range transfers from ordinary pages but not DmaMap,
        // so it can share coalescing and bounded blocks without DMA support.
        // The whole-tensor buffered path lacked those benefits: on one B70,
        // interleaved HF Qwen3.5-4B loads took 10.9/16.3 s direct versus
        // 48.3/47.6 s buffered; coalescing
        // reduced roughly 900 source calls to 278. Both coalescing and blind
        // width growth changed, and network results varied between days.
        // TPU symbols alone did not prove a working transfer implementation;
        // TPU, neuron, metal and furiosa keep the buffered path pending
        // runtime checks.
        return switch (platform.target) {
            .cuda, .rocm, .oneapi, .cpu => .{
                .direct = try direct_loader.Loader.create(
                    allocator,
                    io,
                    platform,
                    opts,
                ),
            },
            .tpu, .neuron, .metal, .furiosa => .{
                .buffered = try buffered_loader.Loader.create(
                    allocator,
                    io,
                    platform,
                    opts.readWidth(),
                    opts.load_profile,
                ),
            },
        };
    }

    pub fn submit(self: Backend, specs: []const LoadSpec, progress: std.Progress.Node) SubmitError!Submission {
        return switch (self) {
            .direct => |direct| .{ .direct = .{
                .loader = direct,
                .batch = try direct.submit(specs, progress),
            } },
            .buffered => |buffered| .{ .buffered = .{
                .loader = buffered,
                .batch = try buffered.submit(specs, progress),
            } },
        };
    }

    /// Fails the backend unless it already failed: queued work is dropped,
    /// work in flight drains, and every later await returns the first error.
    pub fn abort(self: Backend) void {
        switch (self) {
            .direct => |direct| direct.abort(),
            .buffered => |buffered| buffered.abort(),
        }
    }

    pub fn destroy(self: Backend) void {
        switch (self) {
            .direct => |direct| direct.destroy(),
            .buffered => |buffered| buffered.destroy(),
        }
    }
};

pub const Submission = union(enum) {
    direct: struct { loader: *direct_loader.Loader, batch: *direct_loader.Batch },
    buffered: struct { loader: *buffered_loader.Loader, batch: *buffered_loader.Batch },

    /// Waits for and retires the batch. Its pointer is dangling afterwards.
    pub fn await(self: Submission) AwaitError!void {
        return switch (self) {
            .direct => |direct| direct.loader.awaitBatch(direct.batch),
            .buffered => |buffered| buffered.loader.awaitBatch(buffered.batch),
        };
    }
};

/// A random lattice whose runs land in order (`Lattice.dstOrdered`), its
/// source strides anything, zero (a broadcast) included.
fn randomLattice(random: std.Random, dims: []LoadPart.Dim) Lattice {
    const width = random.intRangeAtMost(u64, 1, 17);
    var extent = width;
    var d = dims.len;
    while (d > 0) {
        d -= 1;
        dims[d] = .{
            .count = random.intRangeAtMost(u64, 1, 5),
            .src_stride = random.intRangeAtMost(u64, 0, 40),
            .dst_stride = extent + random.intRangeAtMost(u64, 0, 3) * random.intRangeAtMost(u64, 0, 4),
        };
        extent = dims[d].dst_stride * (dims[d].count - 1) + extent;
    }
    return .{ .src_offset = random.intRangeAtMost(u64, 0, 9), .dst_offset = random.intRangeAtMost(u64, 0, 9), .width = width, .dims = dims };
}

test "copyRuns lands every run of a range where the cursor does" {
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    var source: [4096]u8 = undefined;
    for (&source, 0..) |*byte, i| byte.* = @truncate(i * 7 + 1);
    var expected: [4096]u8 = undefined;
    var actual: [4096]u8 = undefined;
    var dims_buffer: [5]LoadPart.Dim = undefined;
    for (0..2000) |_| {
        var lattice = randomLattice(random, dims_buffer[0..random.intRangeAtMost(usize, 0, dims_buffer.len)]);
        // The widths `copyRuns` loads as one word, and others.
        if (random.boolean()) lattice.width = ([_]u64{ 1, 2, 4, 8, 16 })[random.uintLessThan(usize, 5)];
        const spans = lattice.spans().?;
        if (lattice.src_offset + spans.src > source.len or lattice.dst_offset + spans.dst > expected.len or !lattice.dstOrdered()) continue;
        const runs = lattice.runs();
        const first = random.uintAtMost(u64, runs);
        const end = random.intRangeAtMost(u64, first, runs);
        @memset(&expected, 0);
        @memset(&actual, 0);
        var cursor: Lattice.Cursor = .init(lattice, first, end);
        while (cursor.next()) |run| {
            const width: usize = @intCast(lattice.width);
            @memcpy(expected[@intCast(run.dst)..][0..width], source[@intCast(run.src)..][0..width]);
        }
        lattice.copyRuns(first, end, @intFromPtr(&actual), @intFromPtr(&source));
        try std.testing.expectEqualSlices(u8, &expected, &actual);
    }
}

test "footprints with the same strides overlap exactly when a byte is in both" {
    var prng: std.Random.DefaultPrng = .init(0xf00d);
    const random = prng.random();
    var dims_buffer: [4]LoadPart.Dim = undefined;
    var overlapping: usize = 0;
    var disjoint: usize = 0;
    for (0..3000) |_| {
        const shape = randomLattice(random, dims_buffer[0..random.intRangeAtMost(usize, 1, dims_buffer.len)]);
        var self: Footprint = .{ .start = random.intRangeAtMost(u64, 0, 40), .width = random.intRangeAtMost(u64, 1, shape.width), .len = shape.dims.len };
        var other: Footprint = .{ .start = random.intRangeAtMost(u64, 0, 40), .width = random.intRangeAtMost(u64, 1, shape.width), .len = shape.dims.len };
        for (shape.dims, 0..) |d, k| {
            self.dims[k] = .{ .count = random.intRangeAtMost(u64, 1, d.count), .src_stride = 0, .dst_stride = d.dst_stride };
            other.dims[k] = .{ .count = random.intRangeAtMost(u64, 1, d.count), .src_stride = 0, .dst_stride = d.dst_stride };
        }
        var bytes: [1024]u2 = @splat(0);
        var marked = true;
        for ([_]*const Footprint{ &self, &other }, [_]u2{ 1, 2 }) |footprint, bit| {
            var cursor: Lattice.Cursor = .init(footprint.lattice(), 0, footprint.lattice().runs());
            while (cursor.next()) |run| {
                if (run.dst + footprint.width > bytes.len) marked = false else for (bytes[@intCast(run.dst)..][0..@intCast(footprint.width)]) |*byte| byte.* |= bit;
            }
        }
        if (!marked) continue;
        const expected = std.mem.indexOfScalar(u2, &bytes, 3) != null;
        if (expected) overlapping += 1 else disjoint += 1;
        try std.testing.expectEqual(expected, try self.overlaps(other));
        try std.testing.expectEqual(expected, try other.overlaps(self));
    }
    try std.testing.expect(overlapping > 100 and disjoint > 100);
}

test "strided parts overlap exactly when a byte is in both, whatever their runs merge into" {
    var prng: std.Random.DefaultPrng = .init(0xbeef);
    const random = prng.random();
    var source: safetensors.Tensor = undefined;
    var shape_dims: [4]LoadPart.Dim = undefined;
    var dims: [2][4]LoadPart.Dim = undefined;
    var decided: usize = 0;
    for (0..3000) |_| {
        const shape = randomLattice(random, shape_dims[0..random.intRangeAtMost(usize, 1, shape_dims.len)]);
        var parts: [2]LoadPart = undefined;
        for (&parts, &dims) |*part, *part_dims| {
            for (shape.dims, part_dims[0..shape.dims.len]) |d, *own| own.* = .{ .count = random.intRangeAtMost(u64, 1, d.count), .src_stride = 0, .dst_stride = d.dst_stride };
            part.* = .{ .source = &source, .dst_offset = random.intRangeAtMost(u64, 0, 40), .width = random.intRangeAtMost(u64, 1, shape.width), .dims = part_dims[0..shape.dims.len] };
        }
        var bytes: [1024]u2 = @splat(0);
        var marked = true;
        for (parts, [_]u2{ 1, 2 }) |part, bit| {
            var cursor: Lattice.Cursor = .init(part.lattice(), 0, part.lattice().runs());
            while (cursor.next()) |run| {
                if (run.dst + part.width > bytes.len) marked = false else for (bytes[@intCast(run.dst)..][0..@intCast(part.width)]) |*byte| byte.* |= bit;
            }
        }
        if (!marked) continue;
        const expected = std.mem.indexOfScalar(u2, &bytes, 3) != null;
        try std.testing.expectEqual(expected, try overlaps(parts[0], parts[1]));
        decided += 1;
    }
    try std.testing.expect(decided > 1000);
}

test "many interleaved strided parts are checked by sorting their runs" {
    const allocator = std.testing.allocator;
    var prng: std.Random.DefaultPrng = .init(0xc0de);
    const random = prng.random();
    var source: safetensors.Tensor = .{ .file_uri = "", .name = "source", .shape = Shape.init(.{1 << 16}, .u8), .offset = 0 };
    const destination_bytes = 4096;
    var dims: [16][2]LoadPart.Dim = undefined;
    var outcomes: [2]usize = .{ 0, 0 };
    for (0..400) |_| {
        // A grid's columns, each part one of them with a stride of its
        // own, some columns shifted onto their neighbours'.
        const count = random.intRangeAtMost(usize, 9, dims.len);
        var parts: [16]LoadPart = undefined;
        for (parts[0..count], dims[0..count], 0..) |*part, *part_dims, i| {
            const row = 4 * count + random.uintLessThan(u64, 2) * 4 * count;
            part_dims.* = .{
                .{ .count = random.intRangeAtMost(u64, 1, 3), .src_stride = 0, .dst_stride = 8 * row },
                .{ .count = random.intRangeAtMost(u64, 1, 8), .src_stride = 1, .dst_stride = row },
            };
            const shift = if (random.uintLessThan(u8, 8) == 0) random.uintLessThan(u64, 4) else 0;
            part.* = .{ .source = &source, .dst_offset = 4 * i + shift, .width = 4, .dims = part_dims };
        }
        std.mem.sort(LoadPart, parts[0..count], {}, struct {
            fn lessThan(_: void, lhs: LoadPart, rhs: LoadPart) bool {
                return lhs.dst_offset < rhs.dst_offset;
            }
        }.lessThan);
        var bytes: [destination_bytes]u8 = @splat(0);
        var overlapping = false;
        for (parts[0..count]) |part| {
            var cursor: Lattice.Cursor = .init(part.lattice(), 0, part.lattice().runs());
            while (cursor.next()) |run| for (bytes[@intCast(run.dst)..][0..4]) |*byte| {
                if (byte.* != 0) overlapping = true;
                byte.* = 1;
            };
        }
        if (checkParts(allocator, parts[0..count], destination_bytes, true)) {
            try std.testing.expect(!overlapping);
        } else |err| {
            try std.testing.expect(overlapping);
            try std.testing.expectEqual(error.InvalidLayout, err);
        }
        outcomes[@intFromBool(overlapping)] += 1;
    }
    try std.testing.expect(outcomes[0] > 50 and outcomes[1] > 50);
}
