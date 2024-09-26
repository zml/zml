const builtin = @import("builtin");
const std = @import("std");

const aio = @import("aio.zig");
const meta = @import("meta.zig");
const module = @import("module.zig");
const pjrt = @import("pjrtx.zig");
const pjrt_core = @import("pjrt");
const log = std.log.scoped(.zml);

pub const Target = enum {
    cpu,
    cuda,
    rocm,
    tpu,
};

pub const available_targets = switch (builtin.os.tag) {
    .macos => [_]Target{
        .cpu,
    },
    .linux => [_]Target{
        .cpu,
        .cuda,
        .rocm,
        .tpu,
    },
    else => [_]Target{},
};

pub const CompilationOptions = struct {
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    cache_location: ?[]const u8 = null,
    sharding_enabled: bool = true,
    sharding_axes: std.BoundedArray([*:0]const u8, 8) = .{},
};

pub const Platform = struct {
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,
    compilation_options: CompilationOptions = .{},

    pub const MAX_NUM_DEVICES: u8 = 8;

    pub fn init(target: Target, api: *const pjrt.Api) !Platform {
        const pjrt_client = try pjrt.Client.init(api, &.{});
        const true_num_devices = pjrt_client.getAddressableDevices(api).len;
        if (true_num_devices > MAX_NUM_DEVICES) {
            log.warn("platform {} got {} devices, but ZML only support up to {} devices. Some devices won't be used.", .{ target, true_num_devices, MAX_NUM_DEVICES });
        }
        return .{
            .target = target,
            .pjrt_api = api,
            .pjrt_client = pjrt_client,
            .compilation_options = .{},
        };
    }

    pub fn getDevices(self: Platform) []const *const pjrt_core.Device {
        const all_devices = self.pjrt_client.getAddressableDevices(self.pjrt_api);
        if (all_devices.len > MAX_NUM_DEVICES) {
            return all_devices[0..MAX_NUM_DEVICES];
        }
        return all_devices;
    }

    pub fn numDevices(self: Platform) u8 {
        return @intCast(self.getDevices().len);
    }

    pub fn withCompilationOptions(self: Platform, opts: CompilationOptions) Platform {
        var res = self;
        res.compilation_options = opts;
        return res;
    }

    pub fn deinit(self: *Platform) void {
        self.pjrt_client.deinit(self.pjrt_api);
    }

    /// Returns the Profiler for this API.
    /// Not all platform have a profiling api, for those the profiler object will do nothing.
    /// Platforms with known profiler extensions: cuda, xpu
    pub fn getProfiler(self: Platform, options: pjrt_core.Profiler.Options) pjrt_core.Profiler {
        return self.pjrt_client.getProfiler(self.pjrt_api, options);
    }
};
