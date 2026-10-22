const std = @import("std");

const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const constants = @import("../constants.zig");
const buffered_loader = @import("buffered_loader.zig");
const direct_loader = @import("direct_loader.zig");
const dma_calibration = @import("dma_calibration.zig");

pub const SubmitError = direct_loader.Loader.SubmitError || buffered_loader.Loader.SubmitError;
pub const AwaitError = direct_loader.Loader.AwaitError || buffered_loader.Loader.AwaitError;

/// The largest source request, and so the largest DMA block: a request is
/// a whole number of blocks, at least one.
pub const max_read_request_size: usize = 32 * 1024 * 1024;

/// What the direct backend widens the reads of a file read directly to:
/// the VFS's alignment when `io` is a VFS and `direct_io` is not `off`, else
/// 0, every read exact. The VFS decides per local file whether it is read
/// directly (`VFS.useDirectIo`).
pub fn readAlignment(io: std.Io, direct_io: VFS.DirectIo) usize {
    if (direct_io == .off or VFS.fromIo(io) == null) return 0;
    return VFS.direct_io_alignment;
}

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
/// it, as the front end checked them (`checkParts`): backends trust them. A
/// backend copies each spec, its parts and their dimensions during
/// submission; the sources and the output stay alive until await.
pub const LoadSpec = struct {
    /// The destination's global shape, which the output buffer keeps. Its
    /// sharding places the bytes on the devices; a shape without a mesh is
    /// replicated over every device (`Platform.placedShape`).
    shape: Shape,
    /// The memory space of each device that receives the bytes, from the
    /// source's `TensorStore` binding. The front end only submits kinds
    /// `supportsMemory` accepts.
    memory: Memory.Kind = .default,
    /// Written whole once the destination's buffers exist, and not touched
    /// before: a failure leaves it as it was or holding those buffers.
    output: *Buffer,
    /// What fills the destination's logical bytes, in ascending, disjoint
    /// destination order; every byte no part covers is zero.
    parts: []const LoadPart,
};

/// `width` bytes of `source` from `src_offset` (relative to the tensor's
/// data) land at `dst_offset` of the destination's logical bytes: the
/// row-major bytes of its packed shape, before the sharding splits them.
/// With `dims` the part is strided: that run repeats for every index of
/// `dims`, each step adding its strides to both offsets (`Lattice`).
pub const LoadPart = struct {
    source: *const safetensors.Tensor,
    src_offset: u64 = 0,
    dst_offset: u64 = 0,
    width: u64,
    /// Outermost first; empty for one contiguous run.
    dims: []const Dim = &.{},

    /// A step of the part's runs: `count` of them, each moving the source
    /// and destination offsets by their strides. A zero `src_stride`
    /// repeats the same bytes.
    pub const Dim = struct {
        count: u64,
        src_stride: u64,
        dst_stride: u64,
    };

    /// The whole of `source`, at the start of the destination.
    pub fn whole(source: *const safetensors.Tensor) LoadPart {
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

/// Whether `parts` keep the `LoadSpec` contract for a destination of
/// `destination_bytes`: there is at least one, each is non-empty and inside
/// its source and the destination, and they come in ascending destination
/// order, contiguous ones without overlapping. Strided parts must land in
/// index order (`Lattice.dstOrdered`); interleaving ones are not checked
/// against each other, since only `Source` makes them, disjoint by
/// construction.
pub fn checkParts(parts: []const LoadPart, destination_bytes: u64) bool {
    if (parts.len == 0) return false;
    for (parts) |part| {
        if (part.dims.len != 0) return checkStridedParts(parts, destination_bytes);
    }
    var end: u64 = 0;
    for (parts) |part| {
        const source_end = std.math.add(u64, part.src_offset, part.width) catch return false;
        if (part.width == 0 or source_end > part.source.byteSize() or part.dst_offset < end) return false;
        end = std.math.add(u64, part.dst_offset, part.width) catch return false;
    }
    return end <= destination_bytes;
}

fn checkStridedParts(parts: []const LoadPart, destination_bytes: u64) bool {
    var covered: u64 = 0;
    var previous_start: u64 = 0;
    for (parts) |part| {
        const lattice = part.lattice();
        if (part.width == 0 or part.dims.len > max_part_dims or !lattice.dstOrdered()) return false;
        const spans = lattice.spans() orelse return false;
        const source_end = std.math.add(u64, part.src_offset, spans.src) catch return false;
        const part_end = std.math.add(u64, part.dst_offset, spans.dst) catch return false;
        if (source_end > part.source.byteSize() or part_end > destination_bytes or part.dst_offset < previous_start) return false;
        previous_start = part.dst_offset;
        covered += lattice.runs() * part.width;
    }
    return covered <= destination_bytes;
}

/// Whether `platform` loads through the direct backend. Transfer-manager
/// support and host pinning are independent. CPU implements byte-range
/// transfers from ordinary pages but not DmaMap, so it can share coalescing
/// and bounded blocks without DMA support. The whole-tensor buffered path
/// lacked those benefits: on one B70, interleaved HF Qwen3.5-4B loads took
/// 10.9/16.3 s direct versus 48.3/47.6 s buffered; coalescing reduced
/// roughly 900 source calls to 278. Both coalescing and blind width growth
/// changed, and network results varied between days. TPU symbols alone did
/// not prove a working transfer implementation; TPU, neuron, metal and
/// furiosa keep the buffered path pending runtime checks.
fn loadsDirect(platform: *const Platform) bool {
    return switch (platform.target) {
        .cuda, .rocm, .oneapi, .cpu => true,
        .tpu, .neuron, .metal, .furiosa => false,
    };
}

/// Whether the loader can deliver into `kind` on `platform`: every device
/// must expose that memory space, and host-pinned delivery needs the direct
/// pipeline, as on master, where TPU, neuron, metal and furiosa refused it.
/// CPU takes the direct pipeline here, so it accepts `.host_pinned` (its
/// plugin resolves every kind to its one memory space).
pub fn supportsMemory(platform: *const Platform, kind: Memory.Kind) bool {
    if (kind == .default) return true;
    if (kind == .host_pinned and !loadsDirect(platform)) return false;
    for (platform.devices) |*device| {
        if (device.memory(kind) == null) return false;
    }
    return true;
}

/// The transfer implementation selected for the platform (`loadsDirect`).
pub const Backend = union(enum) {
    direct: *direct_loader.Loader,
    buffered: *buffered_loader.Loader,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) !Backend {
        return if (loadsDirect(platform))
            .{ .direct = try direct_loader.Loader.create(allocator, io, platform, opts) }
        else
            .{ .buffered = try buffered_loader.Loader.create(allocator, io, platform, opts) };
    }

    pub fn submit(self: Backend, specs: []const LoadSpec, progress: std.Progress.Node) SubmitError!Submission {
        return switch (self) {
            inline else => |loader, tag| @unionInit(Submission, @tagName(tag), try loader.submit(specs, progress)),
        };
    }

    /// Waits for and retires `submission`, which this backend published.
    /// Its batch is dangling afterwards.
    pub fn await(self: Backend, submission: Submission) AwaitError!void {
        return switch (self) {
            inline else => |loader, tag| loader.awaitBatch(@field(submission, @tagName(tag))),
        };
    }

    /// Fails the backend unless it already failed: queued work is dropped,
    /// work in flight drains, and every later await returns the first error.
    pub fn abort(self: Backend) void {
        switch (self) {
            inline else => |loader| loader.abort(),
        }
    }

    pub fn destroy(self: Backend) void {
        switch (self) {
            inline else => |loader| loader.destroy(),
        }
    }
};

/// A published batch of the backend that published it.
pub const Submission = union(enum) {
    direct: *direct_loader.Batch,
    buffered: *buffered_loader.Batch,
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

test "checkParts refuses parts outside their source or destination, out of order or overlapping" {
    // A source of 16 bytes into a destination of 32.
    const source: safetensors.Tensor = .{ .file_uri = "memory", .name = "s", .shape = .init(.{16}, .u8), .offset = 0 };
    // Runs of 4 bytes, every other 4 bytes of the destination.
    const every_other: []const LoadPart.Dim = &.{.{ .count = 4, .src_stride = 4, .dst_stride = 8 }};
    const refused = [_][]const LoadPart{
        &.{},
        &.{.{ .source = &source, .width = 0 }},
        // Past the source, past the destination, and an overflow.
        &.{.{ .source = &source, .src_offset = 8, .width = 9 }},
        &.{.{ .source = &source, .dst_offset = 24, .width = 9 }},
        &.{.{ .source = &source, .src_offset = std.math.maxInt(u64), .width = 1 }},
        // Overlapping, and descending.
        &.{ .{ .source = &source, .width = 8 }, .{ .source = &source, .dst_offset = 4, .width = 8 } },
        &.{ .{ .source = &source, .dst_offset = 8, .width = 4 }, .{ .source = &source, .width = 4 } },
        // Strided: runs that overlap on the destination, a dimension of no
        // step, past the source, past the destination, and descending.
        &.{.{ .source = &source, .width = 4, .dims = &.{.{ .count = 2, .src_stride = 4, .dst_stride = 2 }} }},
        &.{.{ .source = &source, .width = 4, .dims = &.{.{ .count = 0, .src_stride = 4, .dst_stride = 8 }} }},
        &.{.{ .source = &source, .src_offset = 4, .width = 4, .dims = every_other }},
        &.{.{ .source = &source, .dst_offset = 8, .width = 4, .dims = every_other }},
        &.{ .{ .source = &source, .dst_offset = 4, .width = 4, .dims = every_other }, .{ .source = &source, .width = 4, .dims = every_other } },
    };
    for (refused) |parts| try std.testing.expect(!checkParts(parts, 32));
    const accepted = [_][]const LoadPart{
        // Back to back.
        &.{ .{ .source = &source, .width = 16 }, .{ .source = &source, .dst_offset = 16, .width = 16 } },
        // Two strided parts interleaving their runs.
        &.{ .{ .source = &source, .width = 4, .dims = every_other }, .{ .source = &source, .dst_offset = 4, .width = 4, .dims = every_other } },
        // A contiguous part, then a strided one.
        &.{ .{ .source = &source, .width = 8 }, .{ .source = &source, .dst_offset = 8, .width = 4, .dims = &.{.{ .count = 2, .src_stride = 4, .dst_stride = 8 }} } },
    };
    for (accepted) |parts| try std.testing.expect(checkParts(parts, 32));
}
