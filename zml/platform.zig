const std = @import("std");

const c = @import("c");
const pjrt = @import("pjrt");
const pjrtx = @import("pjrt");
const platforms = @import("platforms");
pub const Target = platforms.Platform;
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

fn loadOrGetApi(allocator: std.mem.Allocator, io: std.Io, target: Target) !*const pjrt.Api {
    return switch (target) {
        inline else => |tag| @field(api_map, @tagName(tag)) orelse b: {
            disableXlaLogs();
            const api = try platforms.load(allocator, io, tag);
            @field(api_map, @tagName(tag)) = api;
            break :b api;
        },
    };
}

pub const Memory = struct {
    pub const Kind = enum {
        default,
        host_unpinned,
        host_pinned,
        device,
    };

    pjrt_memory: *const pjrt.Memory,
    platform: *const Platform,
    addressable_by_devices: []*const Device,

    fn init(allocator: std.mem.Allocator, pjrt_memory: *const pjrt.Memory, platform: *Platform, all_devices: []const Device) !Memory {
        const pjrt_addressable_by_devices = pjrt_memory.addressableByDevices(platform.pjrt_api);
        const addressable_by_devices = try allocator.alloc(*const Device, pjrt_addressable_by_devices.len);
        for (pjrt_addressable_by_devices, addressable_by_devices) |pjrt_device, *addressable_by_device| {
            addressable_by_device.* = &all_devices[pjrt_device.localHardwareId(platform.pjrt_api)];
        }

        return .{
            .pjrt_memory = pjrt_memory,
            .platform = platform,
            .addressable_by_devices = addressable_by_devices,
        };
    }

    pub fn kind(self: Memory) []const u8 {
        return self.pjrt_memory.kind_(self.platform.pjrt_api);
    }

    pub fn isOfKind(self: Memory, kind_: Kind) bool {
        switch (self.platform.target) {
            .cuda, .rocm => {
                const zml_kind: Memory.Kind = switch (self.kind().len) {
                    "device".len => .device,
                    "pinned_host".len => .host_pinned,
                    "unpinned_host".len => .host_unpinned,
                    else => std.debug.panic("unknown memory {s}", .{self.kind()}),
                };
                return zml_kind == kind_;
            },
            .cpu, .tpu, .neuron => return true,
        }
    }

    fn deinit(self: *Memory, allocator: std.mem.Allocator) void {
        allocator.free(self.addressable_devices);
    }
};

pub const Device = struct {
    platform: *const Platform,
    pjrt_device: *const pjrt.Device,
    pjrt_desc: *const pjrt.DeviceDescription,
    addressable_memories: []*const Memory,

    fn init(allocator: std.mem.Allocator, pjrt_device_: *const pjrt.Device, platform: *const Platform, all_addressable_memories: []const Memory) !Device {
        const pjrt_addressable_memories = pjrt_device_.addressableMemories(platform.pjrt_api);
        const addressable_memories = try allocator.alloc(*const Memory, pjrt_addressable_memories.len);
        for (pjrt_addressable_memories, addressable_memories) |pjrt_memory, *addressable_memory| {
            addressable_memory.* = &all_addressable_memories[pjrt_memory.id(platform.pjrt_api)];
        }

        return .{
            .platform = platform,
            .pjrt_device = pjrt_device_,
            .pjrt_desc = pjrt_device_.getDescription(platform.pjrt_api),
            .addressable_memories = addressable_memories,
        };
    }

    fn deinit(self: *Device, allocator: std.mem.Allocator) void {
        allocator.free(self.addressable_memories);
    }

    pub fn id(self: Device) i32 {
        return self.pjrt_desc.id(self.platform.pjrt_api);
    }

    pub fn processIndex(self: Device) i32 {
        return self.pjrt_desc.processIndex(self.platform.pjrt_api);
    }

    pub fn localHardwareId(self: Device) i32 {
        return self.device.localHardwareId(self.platform.pjrt_api);
    }

    pub fn kind(self: Device) []const u8 {
        return self.pjrt_desc.kind(self.platform.pjrt_api);
    }

    pub fn debugString(self: Device) []const u8 {
        return self.pjrt_desc.debugString(self.platform.pjrt_api);
    }

    pub fn toString(self: Device) []const u8 {
        return self.pjrt_desc.toString(self.platform.pjrt_api);
    }

    pub fn memoryStats(self: Device) pjrt.Device.MemoryStats {
        return self.pjrt_device.memoryStats(self.platform.pjrt_api) catch .zeroes;
    }

    pub fn format(self: Device, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s} ({s})", .{
            self.pjrt_desc.kind(self.platform.pjrt_api),
            self.pjrt_desc.debugString(self.platform.pjrt_api),
        });
    }

    pub fn memory(self: Device, memory_kind: Memory.Kind) *const Memory {
        if (memory_kind == .default) {
            const mem = self.pjrt_device.defaultMemory(self.platform.pjrt_api);
            return &self.platform.memories[mem.id(self.platform.pjrt_api)];
        }

        for (self.addressable_memories) |mem| {
            if (mem.isOfKind(memory_kind)) {
                return mem;
            }
        }
        unreachable;
    }
};

pub const Platform = struct {
    arena_state: std.heap.ArenaAllocator.State = .{},
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,
    devices: []const Device,
    memories: []const Memory,

    // This make the pjrt struct quite fat, but is only used during compilation.
    // TODO: Reconsider having it here, and maybe pass explicitly to compile,
    // or create an intermediary struct:
    // `const comp = platform.compiler(compile_opts); const exe = comp.compile(...);`
    compilation_options: CompilationOptions = .{},

    pub const MAX_NUM_DEVICES: u8 = if (platforms.isEnabled(.tpu)) 32 else 8;

    pub fn init(allocator: std.mem.Allocator, io: std.Io, target: Target, options: CreateOptions) !*Platform {
        const api = try loadOrGetApi(allocator, io, target);

        var named_values_buf: [16]pjrt.NamedValue = undefined;
        const pjrt_client = try pjrt.Client.init(api, options.toNamedValues(target, &named_values_buf));
        const pjrt_devices = pjrt_client.addressableDevices(api);
        if (pjrt_devices.len > MAX_NUM_DEVICES) {
            log.warn("platform {} got {} devices, but ZML only support up to {} devices. Some devices won't be used.", .{ target, pjrt_devices.len, MAX_NUM_DEVICES });
        }

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const devices = try arena.allocator().alloc(Device, pjrt_devices.len);

        const pjrt_memories = pjrt_client.addressableMemories(api);
        const memories = try arena.allocator().alloc(Memory, pjrt_memories.len);

        const platform = try arena.allocator().create(Platform);
        platform.* = .{
            .target = target,
            .pjrt_api = api,
            .pjrt_client = pjrt_client,
            .compilation_options = .{},
            .devices = devices,
            .memories = memories,
        };
        defer platform.arena_state = arena.state;

        {
            for (pjrt_devices, devices) |pjrt_device, *platform_device| {
                platform_device.* = try .init(arena.allocator(), pjrt_device, platform, memories);
            }
            for (pjrt_memories, memories) |pjrt_memory, *platform_memory| {
                platform_memory.* = try .init(arena.allocator(), pjrt_memory, platform, devices);
            }
        }

        switch (target) {
            .cuda => {
                zml.attention.flashattn.load(arena.allocator(), io) catch {
                    log.warn("Failed to load flashattn", .{});
                };
                zml.attention.flashattn.register(platform) catch {
                    log.warn("Failed to register flashattn custom call", .{});
                };
            },
            else => {},
        }

        return platform;
    }

    pub fn auto(allocator: std.mem.Allocator, io: std.Io, options: CreateOptions) !*Platform {
        const ordered_targets: []const Target = &.{
            .tpu,
            .neuron,
            .rocm,
            .cuda,
            .cpu,
        };
        return for (ordered_targets) |target| {
            break init(allocator, io, target, options) catch continue;
        } else error.Unavailable;
    }

    pub fn formatWithDevices(self: *const Platform, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const tee = "├─ ";
        const line = "│  ";
        const langle = "└─ ";

        try writer.print("platform: {s}\n", .{@tagName(self.target)});
        try writer.print("version: {f}\n", .{self.pjrt_api.version()});
        try writer.print("stablehlo_version: {?s}\n", .{self.pjrt_api.stablehloCurrentVersion()});
        try writer.print("extensions:\n", .{});
        {
            var it = self.pjrt_api.extensions();
            while (it.next()) |ext| {
                try writer.print("{s}{s}\n", .{ if (it.current != null) tee else langle, @tagName(ext) });
            }
        }

        try writer.print("devices:\n", .{});
        for (self.devices, 0..) |device, i| {
            const is_last_device = i < self.devices.len - 1;
            try writer.print("{s}{f}\n", .{ if (is_last_device) tee else langle, device });
            for (device.addressable_memories, 0..) |mem, j| {
                try writer.print("{s}{s}memory: {s}\n", .{
                    if (is_last_device) line else "   ",
                    if (j < device.addressable_memories.len - 1) tee else langle,
                    mem.kind(),
                });
            }
        }
    }

    pub fn fmtVerbose(self: *const Platform) std.fmt.Alt(*const Platform, formatWithDevices) {
        return .{ .data = self };
    }

    pub const Sharding = struct { num_replicas: u8, num_partitions: u8 };

    pub fn sharding(self: *const Platform) Sharding {
        // replicas run the same function but with different inputs,
        // while partitions contribute to one evaluation over a shared input.
        // Inside an inference process, we generally don't want replicas,
        // as it's best to fully isolate replicas on different processes.
        // For now we hardcode num_replicas = 1.
        const num_devices: u8 = @intCast(self.devices.len);
        return if (self.compilation_options.sharding_enabled)
            .{ .num_replicas = 1, .num_partitions = num_devices }
        else
            .{ .num_replicas = 1, .num_partitions = 1 };
    }

    pub fn deinit(self: *Platform, allocator: std.mem.Allocator) void {
        self.pjrt_client.deinit(self.pjrt_api);
        self.arena_state.promote(allocator).deinit();
    }

    pub fn compile(
        self: *const Platform,
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

    pub fn compileModel(self: *const Platform, allocator: std.mem.Allocator, io: std.Io, comptime func: anytype, model: stdx.meta.Head(stdx.meta.FnArgs(func)), args: stdx.meta.Tail(stdx.meta.FnArgs(func))) !Exe {
        return self.compileFn(allocator, io, func, .{model} ++ args);
    }

    pub fn compileFn(self: *const Platform, allocator: std.mem.Allocator, io: std.Io, comptime func: anytype, args: stdx.meta.FnArgs(func)) !Exe {
        return zml.module.compile(allocator, io, func, args, self);
    }

    pub fn format(self: *const Platform, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s} {{ ", .{@tagName(self.target)});
        for (self.devices(), 0..) |device, i| {
            try writer.print("{s}(\"{s}\")", .{ device.toString(), device.kind() });
            if (i < self.devices.len - 1) try writer.writeAll(", ");
        }
        try writer.writeAll(" }");
    }

    pub fn memoryKind(self: *const Platform, kind: Memory.Kind) []const u8 {
        for (self.memories) |mem| {
            if (mem.isOfKind(kind)) {
                return mem.kind();
            }
        }
        unreachable;
    }
};

pub const CreateOptions = struct {
    cpu: Cpu = .{ .device_count = 4 },

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the Cuda driver
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    cuda: Cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.95 } } },
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

// TODO(Corendos): Consider moving that in its own file if its size increase too much.
pub const cuda = struct {
    pub fn tryGetComputeCapabilities(platform: *const zml.Platform, device: *const pjrt.Device) ?[]const u8 {
        stdx.debug.assert(platform.target == .cuda, "tryGetComputeCapabilities expects .cuda platform, got {}", .{platform.target});
        const description = device.getDescription(platform.pjrt_api);

        const attributes = description.attributes(platform.pjrt_api);
        return for (attributes) |attr| {
            if (std.mem.eql(u8, attr.name(), "compute_capability")) {
                break attr.value().string;
            }
        } else null;
    }
};
