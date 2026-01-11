const std = @import("std");

const c = @import("c");
const pjrt = @import("pjrt");
const pjrtx = @import("pjrt");
const runtimes = @import("runtimes");
pub const Target = runtimes.Platform;
const stdx = @import("stdx");

const Exe = @import("exe.zig").Exe;
const zml = @import("zml.zig");

const log = std.log.scoped(.zml);

pub const CompilationOptions = struct {
    xla_dump_to: ?[]const u8 = null,
    xla_dump_fusion_visualization: bool = false,
    xla_dump_hlo_pass_re: ?[]const u8 = null,
    sharding_enabled: bool = false,
    sharding_axes: stdx.BoundedArray([*:0]const u8, 8) = .{},
    device_memory_size: u64 = 0,
};

fn StaticPlatformMap(comptime E: type, comptime T: type) type {
    const tag_count = @typeInfo(E).@"enum".fields.len;
    var struct_field_names: [tag_count][]const u8 = undefined;
    var struct_field_types: [tag_count]type = @splat(T);
    const is_optional = @typeInfo(T) == .optional;
    const default_value_ptr: ?*const anyopaque = if (is_optional) @ptrCast(&@as(T, null)) else null;
    var struct_field_attrs: [tag_count]std.builtin.Type.StructField.Attributes = @splat(.{ .default_value_ptr = default_value_ptr });
    inline for (@typeInfo(E).@"enum".fields, 0..) |f, i| struct_field_names[i] = f.name;
    return @Struct(.auto, null, &struct_field_names, &struct_field_types, &struct_field_attrs);
}

var api_map: StaticPlatformMap(Target, ?*const pjrt.Api) = .{};

fn disableXlaLogs() void {
    // https://deepreg.readthedocs.io/en/latest/docs/logging.html#tensorflow-logging
    const TF_CPP_LOG_LEVEL = struct {
        const DEBUG = "0";
        const INFO = "1";
        const WARNING = "2";
        const ERROR = "3";
    };
    _ = c.setenv(
        "TF_CPP_MIN_LOG_LEVEL",
        std.posix.getenv("TF_CPP_MIN_LOG_LEVEL") orelse TF_CPP_LOG_LEVEL.ERROR,
        1,
    );
}

fn loadOrGetApi(target: Target, io: std.Io) !*const pjrt.Api {
    return switch (target) {
        inline else => |tag| @field(api_map, @tagName(tag)) orelse b: {
            disableXlaLogs();
            const api = try runtimes.load(tag, io);
            @field(api_map, @tagName(tag)) = api;
            break :b api;
        },
    };
}

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

    pub const Device = struct {
        pub const Iterator = struct {
            api: *const pjrt.Api,
            devices: []const *const pjrt.Device,
            current: usize = 0,

            pub fn next(self: *Iterator) ?Device {
                if (self.current >= self.devices.len) {
                    return null;
                }
                defer self.current += 1;
                return .{
                    .api = self.api,
                    .device = self.devices[self.current],
                };
            }
        };

        api: *const pjrt.Api,
        device: *const pjrt.Device,

        pub fn format(self: Device, writer: *std.Io.Writer) std.Io.Writer.Error!void {
            const description = self.device.getDescription(self.api);
            try writer.print("{s} ({s})", .{ description.getKind(self.api), description.debugString(self.api) });
        }
    };

    pub fn init(target: Target, io: std.Io, options: CreateOptions) !Platform {
        const api = try loadOrGetApi(target, io);

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

    pub fn auto(io: std.Io, options: CreateOptions) !Platform {
        const ordered_targets: []const Target = &.{
            .tpu,
            .neuron,
            .rocm,
            .cuda,
            .cpu,
        };
        return for (ordered_targets) |target| {
            break init(target, io, options) catch continue;
        } else error.Unavailable;
    }

    pub fn devicesIterator(self: Platform) Device.Iterator {
        return .{
            .api = self.pjrt_api,
            .devices = self.getDevices(),
        };
    }

    pub fn formatWithDevices(self: Platform, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Platform: {s}\n", .{@tagName(self.target)});
        const devices = self.getDevices();
        try writer.print("Devices total={d}\n", .{devices.len});
        for (devices, 0..) |device, i| {
            const description = device.getDescription(self.pjrt_api);
            try writer.print("\t#{d}: {s}\n", .{ i, description.getKind(self.pjrt_api) });
        }
    }

    pub fn fmtVerbose(self: Platform) std.fmt.Alt(Platform, formatWithDevices) {
        return .{ .data = self };
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

    pub fn compile(
        self: Platform,
        allocator: std.mem.Allocator,
        io: std.Io,
        model_: anytype,
        comptime func: std.meta.DeclEnum(@TypeOf(model_)),
        args: stdx.meta.Tail(stdx.meta.FnArgs(@field(@TypeOf(model_), @tagName(func)))),
    ) !Exe {
        return self.compileFn(
            allocator,
            io,
            @field(@TypeOf(model_), @tagName(func)),
            .{model_} ++ args,
        );
    }

    pub fn compileModel(self: Platform, allocator: std.mem.Allocator, io: std.Io, comptime func: anytype, model: stdx.meta.Head(stdx.meta.FnArgs(func)), args: stdx.meta.Tail(stdx.meta.FnArgs(func))) !Exe {
        return self.compileFn(allocator, io, func, .{model} ++ args);
    }

    pub fn compileFn(self: Platform, allocator: std.mem.Allocator, io: std.Io, comptime func: anytype, args: stdx.meta.FnArgs(func)) !Exe {
        return zml.module.compile(allocator, io, func, args, self);
    }

    pub fn format(self: @This(), writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s} {{ ", .{@tagName(self.target)});
        const devices = self.getDevices();
        for (0..devices.len) |i| {
            const description = devices[i].getDescription(self.pjrt_api);
            try writer.print("{s}(\"{s}\")", .{ description.toString(self.pjrt_api), description.getKind(self.pjrt_api) });
            if (i < devices.len - 1) try writer.writeAll(", ");
        }
        try writer.writeAll(" }");
    }
};

pub const CreateOptions = struct {
    cpu: Cpu = .{ .device_count = 4 },

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the Cuda driver
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    cuda: Cuda = .{ .allocator = .{ .async = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    rocm: struct {} = .{},
    tpu: struct {} = .{},
    neuron: struct {} = .{},

    pub const Cpu = struct {
        device_count: u32,

        fn writeNamedValues(self: Cpu, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(.init(.int64, "cpu_device_count", self.device_count));
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
                collective_memory_size_mb: i64 = 0,
            };
        };

        fn writeNamedValues(self: Cuda, values: *std.ArrayList(pjrt.NamedValue)) void {
            switch (self.allocator) {
                .platform => {
                    values.appendAssumeCapacity(.init(.string, "allocator", "platform"));
                },
                .bfc, .async => |opt| {
                    values.appendAssumeCapacity(.init(.string, "allocator", switch (self.allocator) {
                        .bfc => "bfc",
                        .async => "cuda_async",
                        .platform => unreachable,
                    }));
                    values.appendAssumeCapacity(.init(.bool, "preallocate", opt.preallocate));
                    if (opt.memory_fraction > 0) {
                        values.appendAssumeCapacity(.init(.float, "memory_fraction", opt.memory_fraction));
                    }
                    if (opt.collective_memory_size_mb > 0) {
                        values.appendAssumeCapacity(.init(.int64, "collective_memory_size", opt.collective_memory_size_mb * 1024 * 1024));
                    }
                },
            }
        }
    };

    pub fn toNamedValues(self: CreateOptions, target: Target, out: []pjrt.NamedValue) []pjrt.NamedValue {
        var values = std.ArrayList(pjrt.NamedValue).fromOwnedSlice(out);
        values.shrinkRetainingCapacity(0);
        switch (target) {
            .cpu => self.cpu.writeNamedValues(&values),
            .cuda => self.cuda.writeNamedValues(&values),
            inline else => |t| {
                stdx.debug.assertComptime(@hasField(CreateOptions, @tagName(t)), "zml.platform.CreateOptions doesn't list target {s}", .{@tagName(t)});
                const options = @field(self, @tagName(t));
                stdx.debug.assertComptime(@sizeOf(@TypeOf(options)) == 0, "zml.platform.CreateOptions.{s} is discarded", .{@tagName(t)});
            },
        }
        return values.items;
    }
};
