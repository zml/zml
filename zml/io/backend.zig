//! The seam between the front end (loader.zig) and its backends: the
//! `LoadSpec` contract they trust, its parts' byte runs (`Lattice`), the
//! source request size (`requestSize`), and which backend and memory kinds
//! a platform takes (`loadsDirect`, `supportsMemory`).
const std = @import("std");

const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Memory = @import("TensorStore.zig").Memory;
const constants = @import("../constants.zig");
const buffered_loader = @import("buffered_loader.zig");
const direct_loader = @import("direct_loader.zig");
const dma_calibration = @import("dma_calibration.zig");

/// The largest source request, and so the largest DMA block: a request is
/// a whole number of blocks, at least one.
pub const max_read_request_size: usize = 32 * 1024 * 1024;

/// The source request for DMA blocks of `block_size`: the read chunk
/// rounded up to whole blocks, one block at least. Null unless the block is
/// a nonzero multiple of `alignment` (`readAlignment`: reads land in
/// blocks), and the request is at most `max_read_request_size` and, with an
/// alignment, longer than two units, so a read widened by a unit at each
/// end fits.
pub fn requestSize(read_chunk_size: usize, block_size: usize, alignment: usize) ?usize {
    if (block_size == 0 or block_size % @max(alignment, 1) != 0) return null;
    // Bounded before the rounding, which then cannot overflow.
    const least = @max(read_chunk_size, block_size);
    if (least > max_read_request_size) return null;
    const request = std.mem.alignForwardAnyAlign(usize, least, block_size);
    if (request > max_read_request_size) return null;
    if (alignment != 0 and request <= 2 * alignment) return null;
    return request;
}

/// What a backend widens the reads of a file read directly to: the VFS's
/// alignment when `io` is a VFS and `direct_io` is not `off`, else 0, every
/// read exact. The VFS decides per local file whether it is read directly
/// (`VFS.useDirectIo`).
pub fn readAlignment(io: std.Io, direct_io: VFS.DirectIo) usize {
    if (direct_io == .off or VFS.fromIo(io) == null) return 0;
    return VFS.direct_io_alignment;
}

/// The loader's options as its front end checked them, for the backends.
pub const Options = struct {
    /// Source reads in flight at once.
    read_parallelism: usize,
    /// The checkpoint's VFS backend's read hints.
    read_hints: VFS.ReadHints = .local,
    /// The DMA block, else one calibration picks.
    dma_block_size: ?usize = null,
    /// Whether local files are read directly (`VFS.useDirectIo`).
    direct_io: VFS.DirectIo = .auto,

    /// The blocks calibration may select: `dma_block_size`, else
    /// `dma_calibration.default_block_sizes`.
    pub fn dmaBlockSizes(self: *const Options) []const usize {
        return if (self.dma_block_size) |*block_size| block_size[0..1] else &dma_calibration.default_block_sizes;
    }
};

/// One destination, its caller-owned output and the source ranges that fill
/// it, as the front end checked them (`checkParts`): backends trust them.
/// Everything a spec names (its parts, their dimensions, the sources, the
/// output) stays alive until its await; backends borrow it.
pub const LoadSpec = struct {
    /// The destination's global shape, which the output buffer keeps. Its
    /// sharding places the bytes on the devices; a shape without a mesh is
    /// replicated over every device (`Platform.placedShape`).
    shape: Shape,
    /// Where the bytes go, from the source's `TensorStore` binding. The
    /// front end only submits kinds `supportsMemory` accepts, and a host
    /// kind only with contiguous parts (`Lattice.dims` empty).
    memory: Memory = .default,
    /// Written whole once the destination's buffers exist, and not touched
    /// before: a failure leaves it as it was or holding those buffers.
    output: *Buffer,
    /// What fills the destination's logical bytes, in ascending, disjoint
    /// destination order; every byte no part covers is zero.
    parts: []const LoadPart,
};

/// The bytes of `source` that `lattice` moves: its source offsets relative
/// to the tensor's data, its destination offsets into the destination's
/// logical bytes, the row-major bytes of its packed shape before the
/// sharding splits them.
pub const LoadPart = struct {
    source: *const safetensors.Tensor,
    lattice: Lattice,

    /// The whole of `source`, at the start of the destination.
    pub fn whole(source: *const safetensors.Tensor) LoadPart {
        return .{ .source = source, .lattice = .{ .width = source.byteSize() } };
    }
};

/// The most dimensions a part repeats over: `Source` gives each axis at
/// most two.
pub const max_part_dims = 2 * constants.MAX_RANK;

/// A part's byte moves: the run of `width` bytes at `src_offset` and
/// `dst_offset`, and, for a strided part, one more for every index of
/// `dims` (outermost first), each offset moved by the index times its
/// strides. A zero source stride repeats the same bytes. Runs are numbered
/// row-major over `dims`.
pub const Lattice = struct {
    src_offset: u64 = 0,
    dst_offset: u64 = 0,
    width: u64,
    /// Empty for one contiguous run.
    dims: []const Dim = &.{},

    /// A step of the runs: `count` of them, each moving the source and
    /// destination offsets by their strides.
    pub const Dim = struct {
        count: u64,
        src_stride: u64,
        dst_stride: u64,
    };

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
    /// Needs `spans` not null.
    pub fn dstOrdered(self: Lattice) bool {
        var extent = self.width;
        var d = self.dims.len;
        while (d > 0) {
            d -= 1;
            if (self.dims[d].count < 2) continue;
            if (self.dims[d].dst_stride < extent) return false;
            extent += (self.dims[d].count - 1) * self.dims[d].dst_stride;
        }
        return true;
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
    /// addresses the offsets count from (wrapping).
    pub fn copyRuns(self: Lattice, first: u64, end: u64, dst: usize, src: usize) void {
        if (first >= end) return;
        const width: usize = @intCast(self.width);
        var cursor: Cursor = .init(self, first, end);
        while (cursor.next()) |run| {
            @memcpy(@as([*]u8, @ptrFromInt(dst +% run.dst))[0..width], @as([*]const u8, @ptrFromInt(src +% run.src))[0..width]);
        }
    }
};

/// Whether `parts` keep the `LoadSpec` contract for a destination of
/// `destination_bytes`: there is at least one, each is non-empty and inside
/// its source and the destination, and they come in ascending destination
/// order, each starting at or after the previous contiguous part's end or
/// strided part's start. Each part's bytes end in their file at least
/// `VFS.direct_io_alignment` below `maxInt(u64)`, so a read of them widened
/// to the alignment does not overflow. Strided parts must land in index
/// order (`Lattice.dstOrdered`); interleaving ones are only checked against
/// each other by the bytes they cover together, since only `Source` makes
/// them, disjoint by construction.
pub fn checkParts(parts: []const LoadPart, destination_bytes: u64) bool {
    if (parts.len == 0) return false;
    var floor: u64 = 0;
    var covered: u64 = 0;
    for (parts) |part| {
        const lattice = part.lattice;
        if (lattice.width == 0 or lattice.dims.len > max_part_dims) return false;
        const spans = lattice.spans() orelse return false;
        if (!lattice.dstOrdered()) return false;
        const source_end = std.math.add(u64, lattice.src_offset, spans.src) catch return false;
        const file_end = std.math.add(u64, part.source.offset, source_end) catch return false;
        const end = std.math.add(u64, lattice.dst_offset, spans.dst) catch return false;
        if (source_end > part.source.byteSize() or file_end > std.math.maxInt(u64) - VFS.direct_io_alignment) return false;
        if (end > destination_bytes or lattice.dst_offset < floor) return false;
        floor = if (lattice.dims.len == 0) end else lattice.dst_offset;
        covered +|= lattice.runs() * lattice.width;
    }
    return covered <= destination_bytes;
}

/// Whether `platform` loads through the direct backend. Transfer-manager
/// support and host pinning are independent. CPU implements byte-range
/// transfers from ordinary pages but not DmaMap, so it shares coalescing
/// and bounded blocks without DMA support. The whole-tensor buffered path
/// has neither: on one B70, interleaved HF Qwen3.5-4B loads took 10.9 to
/// 16.3 s direct against about 48 s buffered. Furiosa stages in its
/// plugin's `.host_pinned` buffers (SDK dma-buf memory, which the SDK
/// transfers without pinning pages per request). TPU, neuron and metal
/// stay buffered until their transfer path is checked on hardware.
pub fn loadsDirect(platform: *const Platform) bool {
    return switch (platform.target) {
        .cuda, .rocm, .oneapi, .cpu, .furiosa => true,
        .tpu, .neuron, .metal => false,
    };
}

/// Whether the loader can deliver into `kind` on `platform`: device memory,
/// on every platform. The host kinds need a backend that writes host
/// memory in place.
pub fn supportsMemory(platform: *const Platform, kind: Memory) bool {
    _ = platform;
    return kind == .default;
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

    /// Publishes `specs`, and completes one item of `progress` per spec.
    /// Fails with what the backend detected in the files, host memory and
    /// PJRT, logged there (a plain allocation failure is `OutOfMemory`,
    /// unlogged).
    pub fn submit(self: Backend, specs: []const LoadSpec, progress: std.Progress.Node) !Submission {
        return switch (self) {
            inline else => |loader, tag| @unionInit(Submission, @tagName(tag), try loader.submit(specs, progress)),
        };
    }

    /// Waits for and retires `submission`, which this backend published.
    /// Its batch is dangling afterwards.
    pub fn await(self: Backend, submission: Submission) !void {
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
fn randomLattice(random: std.Random, dims: []Lattice.Dim) Lattice {
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

test "the cursor walks a range's runs where at places them, and copyRuns copies them" {
    var prng: std.Random.DefaultPrng = .init(0x5eed);
    const random = prng.random();
    var source: [4096]u8 = undefined;
    for (&source, 0..) |*byte, i| byte.* = @truncate(i * 7 + 1);
    var expected: [4096]u8 = undefined;
    var actual: [4096]u8 = undefined;
    var dims_buffer: [5]Lattice.Dim = undefined;
    for (0..2000) |_| {
        const lattice = randomLattice(random, dims_buffer[0..random.intRangeAtMost(usize, 0, dims_buffer.len)]);
        const spans = lattice.spans().?;
        if (lattice.src_offset + spans.src > source.len or lattice.dst_offset + spans.dst > expected.len) continue;
        try std.testing.expect(lattice.dstOrdered());
        const runs = lattice.runs();
        const first = random.uintAtMost(u64, runs);
        const end = random.intRangeAtMost(u64, first, runs);
        @memset(&expected, 0);
        @memset(&actual, 0);
        var cursor: Lattice.Cursor = .init(lattice, first, end);
        for (first..end) |index| {
            const run = lattice.at(index);
            try std.testing.expectEqual(run, cursor.next().?);
            const width: usize = @intCast(lattice.width);
            @memcpy(expected[@intCast(run.dst)..][0..width], source[@intCast(run.src)..][0..width]);
        }
        try std.testing.expectEqual(null, cursor.next());
        lattice.copyRuns(first, end, @intFromPtr(&actual), @intFromPtr(&source));
        try std.testing.expectEqualSlices(u8, &expected, &actual);
    }
}

test "checkParts refuses parts outside their source or destination, out of order or overlapping" {
    // A source of 16 bytes into a destination of 32.
    const source: safetensors.Tensor = .{ .file_uri = "memory", .name = "s", .shape = .init(.{16}, .u8), .offset = 0 };
    // Sixteen bytes ending at the largest file offset a read widened to the
    // alignment can reach, and one byte later.
    const last: safetensors.Tensor = .{ .file_uri = "memory", .name = "s", .shape = .init(.{16}, .u8), .offset = std.math.maxInt(u64) - VFS.direct_io_alignment - 16 };
    const past_last: safetensors.Tensor = .{ .file_uri = "memory", .name = "s", .shape = .init(.{16}, .u8), .offset = last.offset + 1 };
    // Runs of 4 bytes, every other 4 bytes of the destination.
    const every_other: []const Lattice.Dim = &.{.{ .count = 4, .src_stride = 4, .dst_stride = 8 }};
    const refused = [_][]const LoadPart{
        &.{},
        &.{.{ .source = &source, .lattice = .{ .width = 0 } }},
        // Past the source, past the destination, an overflow, and past the
        // largest file offset.
        &.{.{ .source = &source, .lattice = .{ .src_offset = 8, .width = 9 } }},
        &.{.{ .source = &source, .lattice = .{ .dst_offset = 24, .width = 9 } }},
        &.{.{ .source = &source, .lattice = .{ .src_offset = std.math.maxInt(u64), .width = 1 } }},
        &.{.whole(&past_last)},
        // Overlapping, and descending.
        &.{ .{ .source = &source, .lattice = .{ .width = 8 } }, .{ .source = &source, .lattice = .{ .dst_offset = 4, .width = 8 } } },
        &.{ .{ .source = &source, .lattice = .{ .dst_offset = 8, .width = 4 } }, .{ .source = &source, .lattice = .{ .width = 4 } } },
        // Strided: runs that overlap on the destination, a dimension of no
        // step, strides whose span overflows, past the source, past the
        // destination, and descending.
        &.{.{ .source = &source, .lattice = .{ .width = 4, .dims = &.{.{ .count = 2, .src_stride = 4, .dst_stride = 2 }} } }},
        &.{.{ .source = &source, .lattice = .{ .width = 4, .dims = &.{.{ .count = 0, .src_stride = 4, .dst_stride = 8 }} } }},
        &.{.{ .source = &source, .lattice = .{ .width = 4, .dims = &.{.{ .count = 1 << 40, .src_stride = 0, .dst_stride = 1 << 40 }} } }},
        &.{.{ .source = &source, .lattice = .{ .src_offset = 4, .width = 4, .dims = every_other } }},
        &.{.{ .source = &source, .lattice = .{ .dst_offset = 8, .width = 4, .dims = every_other } }},
        &.{ .{ .source = &source, .lattice = .{ .dst_offset = 4, .width = 4, .dims = every_other } }, .{ .source = &source, .lattice = .{ .width = 4, .dims = every_other } } },
        // Interleaving strided parts that cover more than the destination,
        // and contiguous parts overlapping beside a strided one.
        &.{ .{ .source = &source, .lattice = .{ .width = 4, .dims = every_other } }, .{ .source = &source, .lattice = .{ .width = 4, .dims = every_other } }, .{ .source = &source, .lattice = .{ .dst_offset = 4, .width = 4, .dims = every_other } } },
        &.{ .{ .source = &source, .lattice = .{ .width = 8 } }, .{ .source = &source, .lattice = .{ .dst_offset = 4, .width = 8 } }, .{ .source = &source, .lattice = .{ .dst_offset = 16, .width = 4, .dims = &.{.{ .count = 2, .src_stride = 4, .dst_stride = 8 }} } } },
    };
    for (refused) |parts| try std.testing.expect(!checkParts(parts, 32));
    const accepted = [_][]const LoadPart{
        // Back to back.
        &.{ .{ .source = &source, .lattice = .{ .width = 16 } }, .{ .source = &source, .lattice = .{ .dst_offset = 16, .width = 16 } } },
        // Two strided parts interleaving their runs.
        &.{ .{ .source = &source, .lattice = .{ .width = 4, .dims = every_other } }, .{ .source = &source, .lattice = .{ .dst_offset = 4, .width = 4, .dims = every_other } } },
        // A contiguous part, then a strided one.
        &.{ .{ .source = &source, .lattice = .{ .width = 8 } }, .{ .source = &source, .lattice = .{ .dst_offset = 8, .width = 4, .dims = &.{.{ .count = 2, .src_stride = 4, .dst_stride = 8 }} } } },
        // A broadcast: one source run repeated over the destination.
        &.{.{ .source = &source, .lattice = .{ .width = 8, .dims = &.{.{ .count = 4, .src_stride = 0, .dst_stride = 8 }} } }},
        // Ending at the largest file offset.
        &.{.whole(&last)},
    };
    for (accepted) |parts| try std.testing.expect(checkParts(parts, 32));
}

test "requestSize rounds the read chunk up to whole blocks of whole alignment units" {
    const mib: usize = 1 << 20;
    // One block at least, the chunk rounded up to whole blocks, and a
    // rounded request past the largest refused.
    try std.testing.expectEqual(16 * mib, requestSize(8 * mib, 16 * mib, 4096));
    try std.testing.expectEqual(9 * mib, requestSize(8 * mib, 3 * mib, 4096));
    try std.testing.expectEqual(null, requestSize(32 * mib, 24 * mib, 0));
    // Without an alignment any nonzero block; with one, only its multiples.
    try std.testing.expectEqual(8 * mib + 2048, requestSize(8 * mib, 8 * mib + 2048, 0));
    try std.testing.expectEqual(null, requestSize(8 * mib, 8 * mib + 2048, 4096));
    try std.testing.expectEqual(null, requestSize(8 * mib, 0, 0));
    // A request of two alignment units cannot hold a read widened at both
    // ends.
    try std.testing.expectEqual(null, requestSize(4096, 8192, 4096));
    try std.testing.expectEqual(3 * 4096, requestSize(4096, 3 * 4096, 4096));
}

test "the options pin a DMA block or leave calibration its candidates, and only a VFS widens reads" {
    const pinned: Options = .{ .read_parallelism = 1, .dma_block_size = 4 << 20 };
    try std.testing.expectEqualSlices(usize, &.{4 << 20}, pinned.dmaBlockSizes());
    const calibrated: Options = .{ .read_parallelism = 1 };
    try std.testing.expectEqualSlices(usize, &dma_calibration.default_block_sizes, calibrated.dmaBlockSizes());

    var filesystem = try VFS.init(std.testing.allocator, std.testing.io);
    defer filesystem.deinit();
    try std.testing.expectEqual(VFS.direct_io_alignment, readAlignment(filesystem.io(), .auto));
    try std.testing.expectEqual(0, readAlignment(filesystem.io(), .off));
    try std.testing.expectEqual(0, readAlignment(std.testing.io, .on));
}
