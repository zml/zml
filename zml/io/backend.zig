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

pub const Parallelism = @import("source_concurrency.zig").Parallelism;

pub const Config = struct {
    read_parallelism: Parallelism,
    load_profile: VFS.LoadProfile,
    dma: dma_calibration.Options,
    max_host_bytes: usize,
    numa: mem.dma.NumaPlacement,
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
        return switch (platform.target) {
            .cuda, .rocm, .oneapi, .cpu => .{ .direct = try direct_loader.Loader.create(allocator, io, platform, config) },
            .tpu, .neuron, .metal => initBuffered(allocator, io, platform, config.read_parallelism, config.load_profile),
        };
    }

    pub fn initBuffered(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        read_parallelism: Parallelism,
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

    pub fn calibration(self: Backend) ?dma_calibration.Calibration {
        return switch (self) {
            .direct => |direct| direct.calibration,
            .buffered => null,
        };
    }

    pub fn bytesLoaded(self: Backend) usize {
        return switch (self) {
            .direct => |direct| direct.bytes_loaded.load(.acquire),
            .buffered => |buffered| buffered.bytes_loaded.load(.acquire),
        };
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

    pub fn isDone(self: Submission) bool {
        return switch (self) {
            .direct => |direct| direct.batch.done.isSet(),
            .buffered => |buffered| buffered.batch.done.isSet(),
        };
    }

    /// Waits for and retires the batch. Its pointer is dangling afterwards.
    pub fn await(self: Submission) !void {
        return switch (self) {
            .direct => |direct| direct.loader.awaitBatch(direct.batch),
            .buffered => |buffered| buffered.loader.awaitBatch(buffered.batch),
        };
    }

    pub fn commitBytes(self: Submission, logical_bytes: usize) void {
        switch (self) {
            .direct => |direct| direct.loader.commitBytes(logical_bytes),
            .buffered => |buffered| buffered.loader.commitBytes(logical_bytes),
        }
    }
};
