const std = @import("std");

const VFS = @import("vfs");
const Buffer = @import("../buffer.zig").Buffer;
const mem = @import("../mem.zig");
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const buffered_loader = @import("buffered_loader.zig");
const direct_loader = @import("direct_loader.zig");
const dma_calibration = @import("dma_calibration.zig");
const limits = @import("limits.zig");

pub const InitError = direct_loader.Loader.InitError || buffered_loader.Loader.InitError;
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

/// One resolved source placement and its caller-owned output. A backend copies
/// each spec during submission; its source and output stay alive until await.
pub const LoadSpec = struct {
    source: *safetensors.Tensor,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
};

/// The transfer implementation selected for the platform.
pub const Backend = union(enum) {
    direct: *direct_loader.Loader,
    buffered: *buffered_loader.Loader,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) InitError!Backend {
        // Transfer-manager support and host pinning are independent. CPU
        // implements byte-range transfers from ordinary pages but not DmaMap,
        // so it can share coalescing and bounded blocks without DMA support.
        // The whole-tensor buffered path lacked those benefits: on one B70,
        // interleaved HF Qwen3.5-4B loads took 10.9/16.3 s direct versus
        // 48.3/47.6 s buffered; coalescing
        // reduced roughly 900 source calls to 278. Both coalescing and blind
        // width growth changed, and network results varied between days.
        // TPU symbols alone did not prove a working transfer implementation;
        // TPU, neuron and metal keep the buffered path pending runtime checks.
        return switch (platform.target) {
            .cuda, .rocm, .oneapi, .cpu => .{ .direct = try direct_loader.Loader.create(allocator, io, platform, opts) },
            .tpu, .neuron, .metal => initBuffered(allocator, io, platform, opts.readWidth(), opts.load_profile),
        };
    }

    pub fn initBuffered(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        read_parallelism: usize,
        load_profile: VFS.LoadProfile,
    ) buffered_loader.Loader.InitError!Backend {
        return .{ .buffered = try buffered_loader.Loader.create(
            allocator,
            io,
            platform,
            read_parallelism,
            load_profile,
        ) };
    }

    pub fn calibration(self: Backend) ?dma_calibration.Result {
        return switch (self) {
            .direct => |direct| direct.calibration,
            .buffered => null,
        };
    }

    /// Device bytes the direct backend allocated for outputs so far, per
    /// `platform.devices` index. Only that backend counts; the front end
    /// never asks the buffered one, which it keeps serial.
    pub fn allocatedBytesPerDevice(self: Backend, out: []u64) void {
        for (out, self.direct.allocated_bytes) |*bytes, *counter| bytes.* = counter.load(.acquire);
    }

    pub fn submit(self: Backend, specs: []const LoadSpec, progress: ?*std.Progress.Node) SubmitError!Submission {
        return switch (self) {
            .direct => |direct| .{ .direct = .{ .loader = direct, .batch = try direct.submit(specs, progress) } },
            .buffered => |buffered| .{ .buffered = .{ .loader = buffered, .batch = try buffered.submit(specs, progress) } },
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
