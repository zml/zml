const std = @import("std");

const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
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
pub const LoadPart = struct {
    source: *safetensors.Tensor,
    src_offset: u64 = 0,
    dst_offset: u64 = 0,
    width: u64,

    /// The whole of `source`, at the start of the destination.
    pub fn whole(source: *safetensors.Tensor) LoadPart {
        return .{ .source = source, .width = source.byteSize() };
    }
};

pub const PartsError = error{ SourceSizeMismatch, InvalidLayout };

/// What both backends rely on from a spec's parts: there is at least one,
/// each is a non-empty range inside its source, they come in ascending destination order without
/// overlapping, and they stay inside the destination's `destination_bytes`.
/// Without `zero_fill` they must cover every destination byte; a plain
/// tensor whose source is larger or smaller than its shape fails there,
/// with `SourceSizeMismatch`. Every other violation is `InvalidLayout`.
pub fn checkParts(parts: []const LoadPart, destination_bytes: u64, zero_fill: bool) PartsError!void {
    if (parts.len == 0) return error.InvalidLayout;
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
