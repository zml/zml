const builtin = @import("builtin");
const std = @import("std");

const pjrt = @import("pjrt");
const asynk = @import("async");

const meta = @import("meta.zig");
const module = @import("module.zig");
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
    sharding_enabled: bool = false,
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

    pub fn getDevices(self: Platform) []const *const pjrt.Device {
        const all_devices = self.pjrt_client.getAddressableDevices(self.pjrt_api);
        if (all_devices.len > MAX_NUM_DEVICES) {
            return all_devices[0..MAX_NUM_DEVICES];
        }
        return all_devices;
    }

    pub const Sharding = struct { num_replicas: u8, num_partitions: u8 };

    pub fn sharding(self: Platform) Sharding {
        // replicas run the same function but with different inputs,
        // while partitions contribute to one evaluation over a shared input.
        // Inside an inference process, we generally don't want replicas,
        // as it's best to fully isolate replicas on different processes.
        // For now we hardcode num_replicas = 1.
        const num_devices: u8 = @intCast(self.getDevices().len);
        return if (self.compilation_options.sharding_enabled)
            .{ .num_replicas = 1, .num_partitions = num_devices }
        else
            .{ .num_replicas = 1, .num_partitions = 1 };
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
    pub fn getProfiler(self: Platform, options: pjrt.Profiler.Options) pjrt.Profiler {
        return self.pjrt_client.getProfiler(self.pjrt_api, options);
    }

    /// Suspend the current co-routine while awaiting for a pjrt event to be over.
    pub fn awaitEvent(self: Platform, event: *pjrt.Event) !void {
        defer event.deinit(self.pjrt_api);
        // If we aren't in a coroutine just use the normal blocking api.
        if (!asynk.inCoro()) {
            return try event.await_(self.pjrt_api);
        }

        var ctx = struct {
            err: ?*pjrt.Error = null,
            notif: asynk.Notification,
            ready: bool = false,
        }{
            .notif = try asynk.Notification.init(),
        };
        defer ctx.notif.deinit();

        try event.onReady(self.pjrt_api, &(struct {
            fn call(err: ?*pjrt.Error, user_arg: ?*anyopaque) callconv(.C) void {
                const ctx_: *@TypeOf(ctx) = @ptrCast(@alignCast(user_arg.?));
                ctx_.err = err;
                @atomicStore(bool, &ctx_.ready, true, .seq_cst);
                ctx_.notif.notify() catch @panic("Unable to notify");
            }
        }.call), &ctx);
        // Suspend
        try ctx.notif.wait();
        if (ctx.err) |e| {
            defer e.deinit(self.pjrt_api);
            return e.getCode(self.pjrt_api).toApiError();
        }
    }
};
