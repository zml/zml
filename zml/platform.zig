const std = @import("std");

const runtimes = @import("runtimes");
pub const Target = runtimes.Platform;
const stdx = @import("stdx");

const pjrt = @import("pjrtx.zig");

const log = std.log.scoped(.zml);

pub const available_targets = std.enums.values(Target);

test {
    std.testing.refAllDecls(@This());
}

pub const CompilationOptions = struct {
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    sharding_enabled: bool = false,
    sharding_axes: stdx.BoundedArray([*:0]const u8, 8) = .{},
    device_memory_size: u64 = 0,
};

pub const Platform = struct {
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,

    // This make the pjrt struct quite fat, but is only used during compilation.
    // TODO: Reconsider having it here, and maybe pass explicitly to compile,
    // or create an intermediary struct:
    // `const comp = platform.compiler(compile_opts); const exe = comp.compile(...);`
    compilation_options: CompilationOptions = .{},

    pub const MAX_NUM_DEVICES: u8 = if (runtimes.isEnabled(.tpu)) 32 else 8;
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

    pub fn memoryForDevice(platform: Platform, memory: pjrt.Memory.Kind, device: *const pjrt.Device) *const pjrt.Memory {
        const memory_target: pjrt.Memory.Kind = switch (memory) {
            .host_unpinned => switch (platform.target) {
                // Cuda doesn't have host_unpinned.
                // ROCm doesn't seem to have it either.
                // TODO(gwenzek): investigate why it was not forced before.
                .cuda, .rocm => .host_pinned,
                else => .host_unpinned,
            },
            inline else => |t| t,
        };
        // TODO measure the cost of this and consider caching.
        const device_memories = device.addressableMemories(platform.pjrt_api);
        for (device_memories) |m| {
            if (memory_target == m.kind(platform.pjrt_api)) {
                return m;
            }
        }
        log.err("Platform {t} doesn't have memory {t}", .{ platform.target, memory });
        @panic("Memory kind not found");
    }

    test memoryForDevice {
        const zml = @import("zml.zig");
        const platform = zml.testing.env();
        const memory_fields = @typeInfo(pjrt.Memory.Kind).@"enum".fields;
        inline for (memory_fields) |field| {
            for (platform.getDevices()) |dev| {
                _ = platform.memoryForDevice(@field(pjrt.Memory.Kind, field.name), dev);
            }
        }
    }

    pub fn memoryStats(platform: Platform, device_id: usize) pjrt.MemoryStats {
        if (platform.target == .cpu) return .zeroes;

        const device = platform.getDevices()[device_id];
        return device.memoryStats(platform.pjrt_api) catch .zeroes;
    }
};

const _CreateOptions = struct {
    cpu: Cpu = .{ .device_count = 4 },

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the Cuda driver
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    cuda: Cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    rocm: struct {} = .{},
    tpu: struct {} = .{},
    neuron: struct {} = .{},

    pub const Cpu = struct {
        device_count: u32,

        fn writeNamedValues(self: Cpu, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(pjrt.NamedValue.from("cpu_device_count", @as(i64, self.device_count)));
        }
    };

    pub const Cuda = struct {
        allocator: Allocator = .{ .bfc = .{} },
        // TODO support all of https://github.com/openxla/xla/blob/3d31c48c719d331d432132b3e0c2c5ce52650675/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L76-L86
        // visible_devices: []const i64 = &.{},
        // node_id
        // num_nodes
        // enable_mock_nccl
        // mock_gpu_topology

        pub const Allocator = union(enum) {
            /// "Best-Fit with Coalescing" algorithm
            bfc: Options,
            /// use cudaMallocAsync
            async: Options,
            /// use raw cuMalloc
            platform,

            pub const Options = struct {
                preallocate: bool = true,
                memory_fraction: f32 = 0.90,
                collective_memory_size_mb: u32 = 0,
            };
        };

        fn writeNamedValues(self: Cuda, values: *std.ArrayList(pjrt.NamedValue)) void {
            switch (self.allocator) {
                .platform => {
                    values.appendAssumeCapacity(.fromString("allocator", "platform"));
                },
                .bfc, .async => |opt| {
                    values.appendAssumeCapacity(.fromString("allocator", switch (self.allocator) {
                        .bfc => "bfc",
                        .async => "cuda_async",
                        .platform => unreachable,
                    }));
                    values.appendAssumeCapacity(.from("preallocate", opt.preallocate));
                    if (opt.memory_fraction > 0) {
                        values.appendAssumeCapacity(.from("memory_fraction", opt.memory_fraction));
                    }
                    if (opt.collective_memory_size_mb > 0) {
                        const collective = @as(i64, opt.collective_memory_size_mb) * 1024 * 1024;
                        values.appendAssumeCapacity(.from("collective_memory_size", collective));
                    }
                },
            }
        }
    };

    pub fn toNamedValues(self: _CreateOptions, target: Target, out: []pjrt.NamedValue) []pjrt.NamedValue {
        var values = std.ArrayList(pjrt.NamedValue).fromOwnedSlice(out);
        values.shrinkRetainingCapacity(0);
        switch (target) {
            .cpu => self.cpu.writeNamedValues(&values),
            .cuda => self.cuda.writeNamedValues(&values),
            inline else => |t| {
                stdx.debug.assertComptime(@hasField(_CreateOptions, @tagName(t)), "zml.platform.CreateOptions doesn't list target {s}", .{@tagName(t)});
                const options = @field(self, @tagName(t));
                stdx.debug.assertComptime(@sizeOf(@TypeOf(options)) == 0, "zml.platform.CreateOptions.{s} is discarded", .{@tagName(t)});
            },
        }
        return values.items;
    }
};
