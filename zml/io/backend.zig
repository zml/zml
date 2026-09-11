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

pub const Config = struct {
    /// Concurrent source reads; the front end resolved the profile default.
    read_parallelism: usize,
    load_profile: VFS.LoadProfile,
    dma: dma_calibration.Options,
    max_host_bytes: usize,
    /// Direct I/O for local source files. The direct backend widens the
    /// reads of a file its VFS reads directly to the profile's alignment;
    /// buffered backends ignore it.
    direct_io: VFS.DirectIo,
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
        config: Config,
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
        // TPU, neuron and metal keep the buffered path pending runtime checks.
        return switch (platform.target) {
            .cuda, .rocm, .oneapi, .cpu => .{ .direct = try direct_loader.Loader.create(allocator, io, platform, config) },
            .tpu, .neuron, .metal => initBuffered(allocator, io, platform, config.read_parallelism, config.load_profile),
        };
    }

    pub fn initBuffered(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        read_parallelism: usize,
        load_profile: VFS.LoadProfile,
    ) !Backend {
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

    pub fn submit(self: Backend, specs: []const LoadSpec, progress: ?*std.Progress.Node) !Submission {
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
    pub fn await(self: Submission) !void {
        return switch (self) {
            .direct => |direct| direct.loader.awaitBatch(direct.batch),
            .buffered => |buffered| buffered.loader.awaitBatch(buffered.batch),
        };
    }
};
