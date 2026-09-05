const std = @import("std");
const VFS = @import("vfs");
const Buffer = @import("../buffer.zig").Buffer;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const dma = @import("dma_calibration.zig");
const mem = @import("../mem.zig");
const limits = @import("limits.zig");

pub const Parallelism = @import("source_concurrency.zig").Parallelism;

pub const Options = struct {
    pub const auto: Options = .{};

    /// Concurrent positional source requests.
    read_parallelism: Parallelism = .{ .adaptive = .{
        .initial = 12,
        .maximum = limits.max_read_parallelism,
    } },
    /// Model-wide source tuning prepared from the VFS path. The default is
    /// generic for callers that do not have an explicit VFS profile.
    load_profile: VFS.LoadProfile = .default,
    /// Calibrate transfer sizing during initialization. Ignored by buffered
    /// backends; CPU uses the default sizing without measurement.
    dma: dma.Options = .{},
    /// Upper bound for the direct backend's host arenas, not a growth target.
    max_host_bytes: usize = 16 * 1024 * 1024 * 1024,
    numa: mem.dma.NumaPlacement = .memory_nodes,
    progress: ?*std.Progress.Node = null,
};

/// Backend contract: one resolved source placement and its caller-owned output.
/// A backend copies the specs during submission; sources and outputs stay alive
/// until that submission has been awaited.
pub const LoadSpec = struct {
    source: *safetensors.Tensor,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
};
