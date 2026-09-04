const std = @import("std");
const VFS = @import("vfs");
const Sharding = @import("../Sharding.zig");
const dma = @import("dma_calibration.zig");
const limits = @import("limits.zig");

pub const Parallelism = union(enum) {
    adaptive: Adaptive,
    fixed: usize,

    pub const Adaptive = struct {
        initial: usize,
        maximum: usize,
    };

    pub fn initial(self: Parallelism) usize {
        return switch (self) {
            .adaptive => |adaptive| adaptive.initial,
            .fixed => |fixed| fixed,
        };
    }

    pub fn maximum(self: Parallelism) usize {
        return switch (self) {
            .adaptive => |adaptive| adaptive.maximum,
            .fixed => |fixed| fixed,
        };
    }

    pub fn isAdaptive(self: Parallelism) bool {
        return switch (self) {
            .adaptive => true,
            .fixed => false,
        };
    }
};

pub const LoaderOptions = struct {
    pub const auto: LoaderOptions = .{};

    /// Concurrent positional source requests.
    read_parallelism: Parallelism = .{ .adaptive = .{
        .initial = 12,
        .maximum = limits.max_read_parallelism,
    } },
    /// Model-wide source tuning prepared from the VFS path. The default is
    /// generic for callers that do not have an explicit VFS profile.
    load_profile: VFS.LoadProfile = .default,
    /// Reuses an existing calibration on direct-transfer platforms. When
    /// absent, the loader calibrates and owns its DMA settings. Supplied
    /// settings must outlive the loader and may be used by only one loader.
    dma: ?*?dma.Settings = null,
    shardings: []const Sharding = &.{},
    progress: ?*std.Progress.Node = null,
};
