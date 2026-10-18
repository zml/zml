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
const limits = @import("limits.zig");

pub const SubmitError = direct_loader.Loader.SubmitError || buffered_loader.Loader.SubmitError;
pub const AwaitError = direct_loader.Loader.AwaitError || buffered_loader.Loader.AwaitError;

pub const Options = struct {
    pub const auto: Options = .{};

    /// Concurrent source reads, at most `limits.max_read_parallelism`.
    /// Null takes the profile's default (`limits.defaultReadParallelism`:
    /// 16 locally, 32 on a high-latency source), clipped to one less than
    /// what the pre-grown pinned set holds. Fixed for the load: a source
    /// that rate limits holds the VFS instead (`vfs/request.zig`).
    read_parallelism: ?usize = null,
    /// Model-wide source tuning prepared from the VFS path. The default
    /// is the no-VFS local profile; prepare one with `VFS.loadProfile`
    /// for a VFS path.
    load_profile: VFS.LoadProfile = .local,
    /// Calibrate transfer sizing during initialization. Ignored by buffered
    /// backends; CPU uses the default sizing without measurement.
    dma: dma_calibration.Options = .{},
    /// Direct I/O for local source files, decided per file by the
    /// VFS that opens it: `auto` reads a file past the page cache when it
    /// is mostly not cached at the first decision, `on` whenever the
    /// filesystem allows it, `off` never. The planner widens a direct file's
    /// reads to the profile's alignment, at most two alignment units per
    /// request. A direct read never fills the page cache, so under
    /// `auto` a cold file stays cold and comes from the disk on every
    /// load; on a host whose warm buffered reads beat its disk, a model
    /// loaded repeatedly is better served by `off`. Nothing changes for
    /// a profile without alignment.
    /// Warm replicated Llama-3.1-8B on eight MI300X took ~1.10 s buffered but
    /// ~4.53 s forced direct from a slow storage extent; on four GB300
    /// with four NVMe drives in RAID0, direct took ~0.30 s versus ~0.32 s
    /// warm buffered. These are loader times, not disk-only rates.
    /// Residency alone cannot predict which path wins. The loader retains
    /// each open file, so its decision is not remeasured on every submission
    /// if cache residency changes.
    direct_io: VFS.DirectIo = .auto,

    /// The width the backend uses: the option, or the profile default.
    pub fn readWidth(self: Options) usize {
        return self.read_parallelism orelse limits.defaultReadParallelism(self.load_profile.high_latency);
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
};

pub const PartsError = error{ SourceSizeMismatch, InvalidLayout, UnsupportedLayout };

/// What both backends rely on from a spec's parts: there is at least one,
/// each is non-empty and inside its source, they come in ascending
/// destination order without overlapping, and they stay inside the
/// destination's `destination_bytes`. Without `zero_fill` they must cover
/// every destination byte; a plain tensor whose source is larger or smaller
/// than its shape fails there, with `SourceSizeMismatch`. Strided parts must
/// land in index order (`Lattice.dstOrdered`); two of them whose runs
/// interleave are proven disjoint only when they repeat with the same
/// stride, other pairs are `UnsupportedLayout`. Every other violation is
/// `InvalidLayout`.
pub fn checkParts(parts: []const LoadPart, destination_bytes: u64, zero_fill: bool) PartsError!void {
    if (parts.len == 0) return error.InvalidLayout;
    for (parts) |part| {
        if (part.dims.len != 0) return checkStridedParts(parts, destination_bytes, zero_fill);
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

fn checkStridedParts(parts: []const LoadPart, destination_bytes: u64, zero_fill: bool) PartsError!void {
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
    // span's end, so only parts whose spans meet can overlap.
    for (parts, 0..) |part, index| {
        const end = part.dst_offset + part.lattice().spans().?.dst;
        for (parts[index + 1 ..]) |other| {
            if (other.dst_offset >= end) break;
            if (try Footprint.of(part).overlaps(Footprint.of(other))) return error.InvalidLayout;
        }
    }
}

/// A part's destination bytes alone: dimensions whose runs touch merged
/// into wider runs.
const Footprint = struct {
    start: u64,
    width: u64,
    dims: [max_part_dims]LoadPart.Dim = undefined,
    len: usize = 0,

    fn of(part: LoadPart) Footprint {
        var footprint: Footprint = .{ .start = part.dst_offset, .width = part.width };
        var d = part.dims.len;
        while (d > 0) {
            d -= 1;
            const dim = part.dims[d];
            if (dim.count < 2) continue;
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
    /// against any, and for two that repeat once with the same stride.
    fn overlaps(self: Footprint, other: Footprint) error{UnsupportedLayout}!bool {
        if (self.len == 0) return other.lattice().hasRunIn(self.start, self.start + self.width);
        if (other.len == 0) return self.lattice().hasRunIn(other.start, other.start + other.width);
        if (self.len != 1 or other.len != 1 or self.dims[0].dst_stride != other.dims[0].dst_stride) return error.UnsupportedLayout;
        // Run i of self and run j of other overlap when the gap between
        // their starts, diff + (j - i) * stride, lies in (-other.width,
        // self.width): at most two values of m = j - i, each of which
        // needs i in [0, n_self) with j in [0, n_other).
        const stride: i128 = self.dims[0].dst_stride;
        const diff: i128 = @as(i128, other.start) - @as(i128, self.start);
        const self_count: i128 = self.dims[0].count;
        const other_count: i128 = other.dims[0].count;
        const lowest = std.math.divCeil(i128, 1 - @as(i128, other.width) - diff, stride) catch unreachable;
        const highest = std.math.divFloor(i128, @as(i128, self.width) - 1 - diff, stride) catch unreachable;
        var m = lowest;
        while (m <= highest) : (m += 1) {
            if (@max(0, -m) < @min(self_count, other_count - m)) return true;
        }
        return false;
    }
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

/// Whether bytes delivered into `kind` take device memory, which loadExecute
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

    pub fn submit(self: Backend, specs: []const LoadSpec, progress: ?*std.Progress.Node) SubmitError!Submission {
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
