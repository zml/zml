const asynk = @import("async");
const builtin = @import("builtin");
const runtimes = @import("runtimes");
const std = @import("std");
const stdx = @import("stdx");

const meta = @import("meta.zig");
const module = @import("module.zig");
const pjrt = @import("pjrtx.zig");

const log = std.log.scoped(.zml);

pub const Target = runtimes.Platform;

pub const available_targets = std.enums.values(Target);

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

    pub const MAX_NUM_DEVICES: u8 = 32;
    pub const CreateOptions = _CreateOptions;

    pub fn init(target: Target, api: *const pjrt.Api, options: CreateOptions) !Platform {
        var named_values_buf: [16]pjrt.NamedValue = undefined;
        const pjrt_client = try pjrt.Client.init(api, options.toNamedValues(target, &named_values_buf));
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
};

const _CreateOptions = struct {
    cpu: void = {},
    cuda: ?Cuda = null,
    rocm: void = {},
    tpu: void = {},
    neuron: void = {},

    pub const Cuda = struct {
        allocator: Allocator = .default,
        // preallocate and memory fraction are only used by the bfc allocator
        preallocate: bool = true,
        memory_fraction: ?f32 = null,
        collective_memory_size_mb: ?u16 = null,
        // TODO support all of https://github.com/openxla/xla/blob/3d31c48c719d331d432132b3e0c2c5ce52650675/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L76-L86

        pub const Allocator = enum {
            default, // the client chose the best option
            bfc, // "Best-Fit with Coalescing" algorithm
            cuda_async, // use cudaMallocAsync
            platform, // the platform default eg cuMalloc
        };

        pub fn writeNamedValues(self: Cuda, values: *std.ArrayListUnmanaged(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(pjrt.NamedValue.from("allocator", self.allocator));
            values.appendAssumeCapacity(pjrt.NamedValue.from("preallocate", self.preallocate));
            if (self.memory_fraction) |memory_fraction| {
                values.appendAssumeCapacity(pjrt.NamedValue.from("memory_fraction", memory_fraction));
            }
            if (self.collective_memory_size_mb) |collective_memory_size_mb| {
                const collective = @as(i64, collective_memory_size_mb) * 1024 * 1024;
                values.appendAssumeCapacity(pjrt.NamedValue.from("collective_memory_size", collective));
            }
        }
    };

    pub fn toNamedValues(self: _CreateOptions, target: Target, out: []pjrt.NamedValue) []pjrt.NamedValue {
        var values = std.ArrayListUnmanaged(pjrt.NamedValue).fromOwnedSlice(out);
        values.shrinkRetainingCapacity(0);
        switch (target) {
            inline else => |t| {
                const options = @field(self, @tagName(t));
                if (@TypeOf(options) == void) return &.{};

                if (options) |opt| opt.writeNamedValues(&values);
            },
        }
        return values.items;
    }
};

comptime {
    for (std.meta.fields(Target)) |target_field| {
        stdx.debug.assertComptime(@hasField(_CreateOptions, target_field.name), "CreateOptions doesn't list target {s}", target_field.name);
    }
}
