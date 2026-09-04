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
    /// Required by direct-transfer platforms. The settings and their platform
    /// must outlive the loader, and the settings may be used by only one loader
    /// at a time.
    dma: ?*dma.DmaPlatformSettings = null,
    shardings: []const Sharding = &.{},
    progress: ?*std.Progress.Node = null,
};
