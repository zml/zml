const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const pjrt = @import("pjrt");
const platforms = @import("platforms");
pub const Target = platforms.Platform;
const stdx = @import("stdx");

const attention = @import("attention.zig");
const constants = @import("constants.zig");
const Exe = @import("exe.zig").Exe;
const pjrtx = @import("pjrtx.zig");
const profiler_ = @import("profiling/profiler.zig");
const Sharding = @import("Sharding.zig");
const zml = @import("zml.zig");

const log = std.log.scoped(.zml);

var api_map: std.enums.EnumArray(Target, ?*const pjrt.Api) = .initFill(null);

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
        std.c.getenv("TF_CPP_MIN_LOG_LEVEL") orelse TF_CPP_LOG_LEVEL.ERROR,
        1,
    );
}

fn validateDeviceCount(target: Target, num_devices: usize) !void {
    if (num_devices == 0) {
        log.err("The selected platform requires at least 1 device, got {}", .{num_devices});
        return error.MissingDevices;
    }
    switch (target) {
        .cpu, .cuda, .rocm, .tpu, .neuron, .metal, .oneapi => {
            if (!std.math.isPowerOfTwo(num_devices)) {
                log.err("Platform {} requires a power-of-two device count, got {}", .{ target, num_devices });
                return error.InvalidDeviceCount;
            }
        },
    }
}

fn loadOrGetApi(allocator: std.mem.Allocator, io: std.Io, target: Target) !*const pjrt.Api {
    return switch (target) {
        inline else => |tag| api_map.get(tag) orelse b: {
            disableXlaLogs();
            const api = try platforms.load(allocator, io, tag);
            api_map.set(tag, api);
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

    fn init(allocator: std.mem.Allocator, pjrt_memory: *const pjrt.Memory, platform: *Platform) !Memory {
        const pjrt_addressable_by_devices = pjrt_memory.addressableByDevices(platform.pjrt_api);
        const addressable_by_devices = try allocator.alloc(*const Device, pjrt_addressable_by_devices.len);

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
            .cuda, .rocm, .oneapi, .tpu => {
                const zml_kind: Memory.Kind = switch (self.kind().len) {
                    "device".len => .device,
                    "pinned_host".len => .host_pinned,
                    "unpinned_host".len => .host_unpinned,
                    else => std.debug.panic("unknown memory {s}", .{self.kind()}),
                };
                return zml_kind == kind_;
            },
            .cpu, .neuron, .metal => return true,
        }
    }

    fn deinit(self: *Memory, allocator: std.mem.Allocator) void {
        allocator.free(self.addressable_devices);
    }

    fn populateAddressableByDevices(self: *Memory) void {
        const pjrt_addressable_by_devices = self.pjrt_memory.addressableByDevices(self.platform.pjrt_api);
        for (pjrt_addressable_by_devices, self.addressable_by_devices) |pjrt_device, *addressable_by_device| {
            addressable_by_device.* = self.platform.deviceFromPjrt(pjrt_device);
        }
    }
};

pub const Device = struct {
    platform: *const Platform,
    pjrt_device: *const pjrt.Device,
    pjrt_desc: *const pjrt.DeviceDescription,
    addressable_memories: []*const Memory,
    memory_by_kind: std.EnumArray(Memory.Kind, ?*const Memory),

    fn init(allocator: std.mem.Allocator, pjrt_device_: *const pjrt.Device, platform: *const Platform) !Device {
        const pjrt_addressable_memories = pjrt_device_.addressableMemories(platform.pjrt_api);
        const addressable_memories = try allocator.alloc(*const Memory, pjrt_addressable_memories.len);
        for (pjrt_addressable_memories, addressable_memories) |pjrt_memory, *addressable_memory| {
            addressable_memory.* = platform.memoryFromPjrt(pjrt_memory);
        }

        // Cache memory lookups since they are expensive
        const default_memory: *const Memory = resolveDefaultMemory(pjrt_device_, platform, addressable_memories);
        const memory_by_kind: std.EnumArray(Memory.Kind, ?*const Memory) = .init(.{
            .default = default_memory,
            .device = resolveMemory(addressable_memories, .device),
            .host_pinned = resolveMemory(addressable_memories, .host_pinned),
            .host_unpinned = resolveMemory(addressable_memories, .host_unpinned),
        });

        return .{
            .platform = platform,
            .pjrt_device = pjrt_device_,
            .pjrt_desc = pjrt_device_.getDescription(platform.pjrt_api),
            .addressable_memories = addressable_memories,
            .memory_by_kind = memory_by_kind,
        };
    }

    fn deinit(self: *Device, allocator: std.mem.Allocator) void {
        allocator.free(self.addressable_memories);
    }

    fn resolveDefaultMemory(pjrt_device_: *const pjrt.Device, platform: *const Platform, addressable_memories: []*const Memory) *const Memory {
        const pjrt_memory = pjrt_device_.defaultMemory(platform.pjrt_api);
        for (addressable_memories) |mem| {
            if (mem.pjrt_memory == pjrt_memory) return mem;
        }
        return platform.memoryFromPjrt(pjrt_memory);
    }

    fn resolveMemory(addressable_memories: []*const Memory, memory_kind: Memory.Kind) ?*const Memory {
        std.debug.assert(memory_kind != .default);

        for (addressable_memories) |mem| {
            if (mem.isOfKind(memory_kind)) {
                return mem;
            }
        }
        return null;
    }

    pub fn id(self: Device) u32 {
        return @intCast(self.pjrt_desc.id(self.platform.pjrt_api));
    }

    pub fn processIndex(self: Device) i32 {
        return self.pjrt_desc.processIndex(self.platform.pjrt_api);
    }

    pub fn localHardwareId(self: Device) i32 {
        return @intCast(self.pjrt_device.localHardwareId(self.platform.pjrt_api));
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

    pub fn memory(self: *const Device, memory_kind: Memory.Kind) ?*const Memory {
        return self.memory_by_kind.values[@intFromEnum(memory_kind)];
    }
};

fn platformDeviceSortId(target: Target, device: Device) usize {
    return switch (target) {
        .neuron => @intCast(device.localHardwareId()),
        .cuda, .rocm, .tpu, .cpu, .oneapi, .metal => device.id(),
    };
}

fn sortDevicesById(target: Target, devices: []Device) void {
    const Context = struct {
        target: Target,

        fn lessThan(ctx: @This(), lhs: Device, rhs: Device) bool {
            return platformDeviceSortId(ctx.target, lhs) < platformDeviceSortId(ctx.target, rhs);
        }
    };

    std.mem.sort(Device, devices, Context{ .target = target }, Context.lessThan);

    if (builtin.mode == .Debug) {
        for (devices, 0..) |device, expected_id| {
            std.debug.assert(platformDeviceSortId(target, device) == expected_id);
        }
    }
}

pub const Platform = struct {
    arena: std.heap.ArenaAllocator,
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,
    devices: []const Device,
    memories: []const Memory,
    physical_mesh: zml.Sharding.PhysicalMesh,
    replicated_sharding: zml.Sharding,
    shardings: std.StringArrayHashMapUnmanaged(zml.Sharding),

    pub const MAX_NUM_DEVICES: u16 = if (platforms.isEnabled(.tpu)) 64 else 32;

    pub fn init(allocator: std.mem.Allocator, io: std.Io, target: Target, options: CreateOptions) !*Platform {
        const api = try loadOrGetApi(allocator, io, target);

        var named_values_buf: [16]pjrt.NamedValue = undefined;
        const pjrt_client = try pjrt.Client.init(api, options.toNamedValues(target, &named_values_buf));
        const pjrt_devices = pjrt_client.addressableDevices(api);
        try validateDeviceCount(target, pjrt_devices.len);
        if (pjrt_devices.len > MAX_NUM_DEVICES) {
            log.warn("platform {} got {} devices, but ZML only support up to {} devices. Some devices won't be used.", .{ target, pjrt_devices.len, MAX_NUM_DEVICES });
        }

        const pjrt_memories = pjrt_client.addressableMemories(api);

        // Note: Platform is a self-owning struct. It contains the arena that created it in the first place
        // But it does mean we have to be careful to pass the arena state that contains the node
        const platform: *Platform = platform: {
            var initial_arena = std.heap.ArenaAllocator.init(allocator);
            errdefer initial_arena.deinit();

            var initial_allocator = initial_arena.allocator();
            // Pre-heat the arena, to avoid fragmentation.
            initial_allocator.free(try initial_allocator.alloc(u8, 8 * 1024));

            const platform = try initial_allocator.create(Platform);
            platform.* = .{
                .arena = initial_arena,
                .target = target,
                .pjrt_api = api,
                .pjrt_client = pjrt_client,
                .shardings = .empty,
                // set below
                .devices = undefined,
                .memories = undefined,
                .physical_mesh = undefined,
                .replicated_sharding = undefined,
            };
            break :platform platform;
        };

        const arena = platform.arena.allocator();
        errdefer platform.arena.deinit();
        try platform.shardings.ensureTotalCapacity(arena, 8);

        {
            const devices = try arena.alloc(Device, pjrt_devices.len);
            platform.devices = devices;
            const memories = try arena.alloc(Memory, pjrt_memories.len);
            platform.memories = memories;

            // TODO: part of the complication here is that we layout the data in spaghetti mode,
            // where devices and memories point to each other and also point to the platform.
            for (pjrt_memories, memories) |pjrt_memory, *platform_memory| {
                platform_memory.* = try .init(arena, pjrt_memory, platform);
            }
            for (pjrt_devices, devices) |pjrt_device, *platform_device| {
                platform_device.* = try .init(arena, pjrt_device, platform);
            }
            sortDevicesById(target, devices);
            for (memories) |*platform_memory| {
                platform_memory.populateAddressableByDevices();
            }

            platform.physical_mesh = try switch (options.physical_mesh) {
                .auto => zml.Sharding.PhysicalMesh.auto(arena, target, devices),
                .custom => |builder| builder(arena, target, devices),
            };
            platform.replicated_sharding = try platform.registerSharding("replicated", .mesh(.{ .x = .high_bandwidth }));
        }

        switch (target) {
            .cuda => {
                zml.attention.flashattn.load(arena, io) catch {
                    log.warn("Failed to load flashattn", .{});
                };
                zml.attention.flashattn.register(platform) catch {
                    log.warn("Failed to register flashattn custom call", .{});
                };
            },
            else => {},
        }

        platform.registerFfi(
            .{
                .name = "zml$print",
                .handler = printCallback,
                .traits = .{ .command_buffer_compatible = false },
            },
        ) catch |err| {
            log.warn("Failed to register FFI custom call \"zml$print\", error: {}", .{err});
        };

        return platform;
    }

    pub fn auto(allocator: std.mem.Allocator, io: std.Io, options: CreateOptions) !*Platform {
        const ordered_targets: []const Target = &.{
            .tpu,
            .neuron,
            .rocm,
            .cuda,
            .oneapi,
            .metal,
            .cpu,
        };
        return for (ordered_targets) |target| {
            break init(allocator, io, target, options) catch continue;
        } else error.Unavailable;
    }

    pub fn formatWithAttributes(self: *const Platform, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        const tee = "├─ ";
        const line = "│  ";
        const langle = "└─ ";

        try writer.print("platform: {s}\n", .{@tagName(self.target)});
        try writer.print("version: {f}\n", .{self.pjrt_api.version()});

        try writer.print("extensions:\n", .{});
        {
            var it = self.pjrt_api.extensions();
            while (it.next()) |ext| {
                try writer.print("{s}{s}\n", .{ if (it.current != null) tee else langle, @tagName(ext) });
            }
        }

        try writer.print("plugin attributes:\n", .{});
        {
            const attributes = self.pjrt_api.pluginAttributes();
            if (attributes.len == 0) {
                try writer.print("{s}(none)\n", .{langle});
            } else {
                for (attributes, 0..) |attr, i| {
                    const is_last_attr = i == attributes.len - 1;
                    try writer.print("{s}{s}", .{ if (is_last_attr) langle else tee, attr.name() });
                    switch (attr.value()) {
                        .string => |v| try writer.print(": \"{s}\"", .{v}),
                        .int64 => |v| try writer.print(": {d}", .{v}),
                        .int64list => |v| {
                            for (v, 0..) |item, j| {
                                if (j == 0) {
                                    try writer.print(": {d}", .{item});
                                } else {
                                    try writer.print(".{d}", .{item});
                                }
                            }
                        },
                        .float => |v| try writer.print(": {d}", .{v}),
                        .bool => |v| try writer.print(": {}", .{v}),
                    }
                    try writer.writeAll("\n");
                }
            }
        }

        try writer.print("devices:\n", .{});
        for (self.devices, 0..) |device, i| {
            const is_last_device = i == self.devices.len - 1;
            const child_indent = if (is_last_device) "   " else line;

            try writer.print("{s}{f}\n", .{ if (is_last_device) langle else tee, device });

            {
                const device_attrs = device.pjrt_desc.attributes(self.pjrt_api);
                if (device_attrs.len > 0) {
                    var last_name: ?[]const u8 = null;
                    var remaining: usize = device_attrs.len;

                    while (remaining > 0) : (remaining -= 1) {
                        var next_index: ?usize = null;
                        for (device_attrs, 0..) |attr, j| {
                            const name = attr.name();
                            if (last_name) |last| {
                                if (std.mem.order(u8, name, last) != .gt) continue;
                            }
                            if (next_index) |ni| {
                                if (std.mem.order(u8, name, device_attrs[ni].name()) == .lt) {
                                    next_index = j;
                                }
                            } else {
                                next_index = j;
                            }
                        }

                        if (next_index == null) break;
                        const attr = device_attrs[next_index.?];

                        try writer.print("{s}{s}{s}", .{ child_indent, tee, attr.name() });
                        switch (attr.value()) {
                            .string => |v| try writer.print(": \"{s}\"", .{v}),
                            .int64 => |v| try writer.print(": {d}", .{v}),
                            .int64list => |v| try writer.print(": {any}", .{v}),
                            .float => |v| try writer.print(": {d}", .{v}),
                            .bool => |v| try writer.print(": {}", .{v}),
                        }
                        try writer.writeAll("\n");

                        last_name = attr.name();
                    }
                }
            }

            try writer.print("{s}{s}memories:\n", .{ child_indent, langle });
            {
                const memory_indent = "   ";
                if (device.addressable_memories.len == 0) {
                    try writer.print("{s}{s}{s}(none)\n", .{ child_indent, memory_indent, langle });
                } else {
                    for (device.addressable_memories, 0..) |mem, j| {
                        const is_last_mem = j == device.addressable_memories.len - 1;
                        try writer.print("{s}{s}{s}memory: {s}\n", .{
                            child_indent,
                            memory_indent,
                            if (is_last_mem) langle else tee,
                            mem.pjrt_memory.debugString(self.pjrt_api),
                        });
                    }
                }
            }
        }
    }

    pub fn fmtVerbose(self: *const Platform) std.fmt.Alt(*const Platform, formatWithAttributes) {
        return .{ .data = self };
    }

    pub fn deinit(self: *Platform, allocator: std.mem.Allocator, io: std.Io) void {
        _ = io;
        _ = allocator;
        self.physical_mesh.deinit(self.arena.allocator());
        self.pjrt_client.deinit(self.pjrt_api);
        self.arena.deinit();
    }

    pub fn compile(
        self: *const Platform,
        allocator: std.mem.Allocator,
        io: std.Io,
        model_: anytype,
        comptime func: std.meta.DeclEnum(@TypeOf(model_)),
        args: stdx.meta.Tail(
            std.meta.ArgsTuple(@TypeOf(@field(@TypeOf(model_), @tagName(func)))),
        ),
        opts: zml.module.CompilationOptions,
    ) !Exe {
        return self.compileFn(
            allocator,
            io,
            @field(@TypeOf(model_), @tagName(func)),
            .{model_} ++ args,
            opts,
        );
    }

    pub fn compileModel(
        self: *const Platform,
        allocator: std.mem.Allocator,
        io: std.Io,
        comptime func: anytype,
        model: stdx.meta.Head(std.meta.ArgsTuple(@TypeOf(func))),
        args: stdx.meta.Tail(std.meta.ArgsTuple(@TypeOf(func))),
        opts: zml.module.CompilationOptions,
    ) !Exe {
        return self.compileFn(allocator, io, func, .{model} ++ args, opts);
    }

    pub fn compileFn(
        self: *const Platform,
        allocator: std.mem.Allocator,
        io: std.Io,
        comptime func: anytype,
        args: std.meta.ArgsTuple(@TypeOf(func)),
        opts: zml.module.CompilationOptions,
    ) !Exe {
        return zml.module.compile(allocator, io, func, args, self, opts);
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

    pub const FfiRegistration = struct {
        name: []const u8,
        handler: *const pjrt.ffi.Handler,
        traits: pjrt.ffi.HandlerTraits = .{ .command_buffer_compatible = false },
        platform_name: ?[]const u8 = null,
    };

    pub fn registerFfi(self: *const zml.Platform, registration: FfiRegistration) !void {
        const platform_name = registration.platform_name orelse self.pjrt_client.platformName(self.pjrt_api);
        if (self.pjrt_api.ffi()) |ffi| {
            try ffi.register(self.pjrt_api, registration.name, platform_name, registration.handler, registration.traits);
        } else {
            log.warn("PJRT FFI extension not available for {s}", .{@tagName(self.target)});
        }
    }

    pub const Profiler = profiler_.Profiler;
    pub const ProfilerOptions = profiler_.ProfilerOptions;

    pub fn profiler(self: *const Platform, allocator: std.mem.Allocator, io: std.Io, options: ProfilerOptions) !Profiler {
        return try profiler_.profiler(self.pjrt_api, allocator, io, options);
    }

    /// Create a Sharding based on the given logical mesh and the default strategy.
    /// Memory is owned by the platform, making it safe to copy around.
    pub fn registerSharding(platform: *Platform, name: []const u8, logical: Sharding.LogicalMesh) error{OutOfMemory}!Sharding {
        return platform.registerShardingWithStrategy(
            name,
            logical,
            .suggest(logical, &platform.physical_mesh),
        ) catch |err| switch (err) {
            error.InvalidPhysicalMesh, error.InvalidStrategy, error.InvalidPhysicalAxis => {
                std.debug.panic("ZML failed to create a valid sharding for logical mesh: {f}\nand physical_mesh: {f}\nPlease report this bug.", .{ logical, platform.physical_mesh });
            },
            error.OutOfMemory => |e| return e,
        };
    }

    /// Create a Sharding based on the given logical mesh and a strategy.
    /// Memory is owned by the platform, making it safe to copy around.
    pub fn registerShardingWithStrategy(platform: *Platform, name: []const u8, logical: Sharding.LogicalMesh, strategy: Sharding.Strategy) !Sharding {
        const arena = platform.arena.allocator();
        const entry = try platform.shardings.getOrPut(arena, name);
        if (entry.found_existing) {
            std.debug.panic("Another sharding already exists with this name: {s}", .{name});
        }

        const owned_name = try arena.dupe(u8, name);
        const owned_data = try arena.create(Sharding.Data);
        owned_data.* = try .init(owned_name, &platform.physical_mesh, logical, strategy);
        const sharding: Sharding = .{ .data = owned_data };
        entry.key_ptr.* = owned_name;
        entry.value_ptr.* = sharding;

        return sharding;
    }

    fn memoryFromPjrt(self: *const Platform, pjrt_memory: *const pjrt.Memory) *const Memory {
        for (self.memories) |*mem| {
            if (mem.pjrt_memory == pjrt_memory) return mem;
        }
        unreachable;
    }

    fn deviceFromPjrt(self: *const Platform, pjrt_device: *const pjrt.Device) *const Device {
        for (self.devices) |*device| {
            if (device.pjrt_device == pjrt_device) return device;
        }
        unreachable;
    }

    pub inline fn defaultMemoryLayout(platform: *const Platform, dims: []const i64, dtype: zml.DataType) pjrt.MemoryLayout {
        // inline cause `default` is a huge ass struct allocated on the stack,
        // and toMemoryLayout returns slices into it.
        // There is probably a better way of doing this,
        // but given it's compiled out (except for TPU), I'm not gonna care for now.
        return switch (platform.target) {
            .tpu => {
                if (comptime !platforms.isEnabled(.tpu)) unreachable;
                const element_type = pjrtx.bufferTypeFromDtype(dtype);
                const default = platform.pjrt_client.defaultMemoryLayout(platform.pjrt_api, element_type, dims) catch @panic("Failed to get default memory layout");
                return default.toMemoryLayout();
            },
            .cuda, .rocm, .neuron, .oneapi, .cpu, .metal => .{
                // If this is the default layout on the platform, there is no point calling PJRT
                .tiled = .{
                    .minor_to_major = constants.minorToMajor(@intCast(dims.len)),
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
        };
    }
};

pub const CreateOptions = struct {
    pub const CreatePhysicalMeshFn = *const fn (
        allocator: std.mem.Allocator,
        target: Target,
        devices: []const Device,
    ) anyerror!zml.Sharding.PhysicalMesh;

    pub const PhysicalMesh = union(enum) {
        auto,
        custom: CreatePhysicalMeshFn,
    };

    physical_mesh: PhysicalMesh = .auto,
    cpu: Cpu = .{ .device_count = 4 },

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the platform driver/runtime.
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    xla_gpu: XlaGpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    tpu: struct {} = .{},
    neuron: struct {} = .{},
    oneapi: struct {} = .{},
    metal: Metal = .{},

    pub const Cpu = struct {
        device_count: u32,

        fn writeNamedValues(self: Cpu, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(.init(.int64, "cpu_device_count", self.device_count));
        }
    };

    /// Metal goes through the same XLA GPU plugin as CUDA and ROCm, but it is
    /// not a discrete GPU: its memory is the machine's memory, so preallocation
    /// is off by default.
    pub const Metal = struct {
        preallocate: bool = false,
        memory_fraction: f32 = 0.90,

        fn writeNamedValues(self: Metal, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(.init(.string, "allocator", "bfc"));
            values.appendAssumeCapacity(.init(.bool, "preallocate", self.preallocate));
            if (self.memory_fraction > 0) {
                values.appendAssumeCapacity(.init(.float, "memory_fraction", self.memory_fraction));
            }
        }
    };

    pub const XlaGpu = struct {
        allocator: Allocator = .{ .bfc = .{} },
        /// The PJRT C API still exposes this under legacy
        /// `use_tfrt_gpu_client` name.
        gpu_async_dispatch: bool = true,
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

        fn writeNamedValues(self: XlaGpu, target: Target, values: *std.ArrayList(pjrt.NamedValue)) void {
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
            switch (target) {
                .cuda => values.appendAssumeCapacity(.init(.bool, "use_tfrt_gpu_client", self.gpu_async_dispatch)),
                else => {},
            }
        }
    };

    pub fn toNamedValues(self: CreateOptions, target: Target, out: []pjrt.NamedValue) []pjrt.NamedValue {
        var values = std.ArrayList(pjrt.NamedValue).fromOwnedSlice(out);
        values.shrinkRetainingCapacity(0);
        switch (target) {
            .cpu => self.cpu.writeNamedValues(&values),
            .cuda, .rocm, .oneapi => self.xla_gpu.writeNamedValues(target, &values),
            .metal => self.metal.writeNamedValues(&values),
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

fn dataTypeFromFfiDataType(ffi_dt: pjrt.ffi.DataType) zml.DataType {
    return switch (ffi_dt) {
        .bool => .bool,
        .i8 => .i8,
        .i16 => .i16,
        .i32 => .i32,
        .i64 => .i64,
        .u8 => .u8,
        .u16 => .u16,
        .u32 => .u32,
        .u64 => .u64,
        .f16 => .f16,
        .f32 => .f32,
        .f64 => .f64,
        .bf16 => .bf16,
        .c64 => .c64,
        .c128 => .c128,
        .f8e5m2 => .f8e5m2,
        .f8e4m3fn => .f8e4m3fn,
        .f8e4m3b11fnuz => .f8e4m3b11fnuz,
        .f8e5m2fnuz => .f8e5m2fnuz,
        .f8e4m3fnuz => .f8e4m3fnuz,
        else => unreachable,
    };
}

fn shapeFromFfiBuffer(buffer: *const pjrt.ffi.Buffer) zml.Shape {
    return .init(buffer.dims(), dataTypeFromFfiDataType(buffer.dtype));
}

fn getScalarAttributeAs(comptime T: type, call_frame: *pjrt.ffi.CallFrame, attribute_name: []const u8) ?T {
    const attribute = call_frame.attrs.getByName(.scalar, attribute_name) orelse return null;
    return attribute.get(T);
}

fn printCallback(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
    return printCallbackInner(call_frame) catch |e| b: {
        log.err("Error in print callback: {}", .{e});
        break :b pjrt.ffi.Error.create(call_frame.api, .unknown, "Unknown");
    };
}

fn printCallbackInner(call_frame: *pjrt.ffi.CallFrame) !?*pjrt.ffi.Error {
    if (call_frame.registeringHook()) return null;

    const pjrt_api: *pjrt.Api = @ptrFromInt(getScalarAttributeAs(u64, call_frame, "pjrt_api").?);
    const pjrt_client: *pjrt.Client = @ptrFromInt(getScalarAttributeAs(u64, call_frame, "pjrt_client").?);

    const device_ordinal: usize = @intCast(try call_frame.ctx.getDeviceOrdinal(call_frame.api));

    const buffer = call_frame.args.buffers()[0];
    const shape = shapeFromFfiBuffer(buffer);

    // NOTE(Corentin): This is a hack. We take the first non device memory, hoping that it's host visible,
    // and copy the buffer there to read it on the CPU and print it.
    const device = pjrt_client.devices(pjrt_api)[device_ordinal];
    const addressable_memories = device.addressableMemories(pjrt_api);
    const first_non_device_memory = for (addressable_memories) |memory| {
        if (!std.mem.eql(u8, memory.kind_(pjrt_api), "device")) break memory;
    } else return error.MemoryNotFound;

    var pjrt_buffer = try pjrt_client.createViewOfDeviceBuffer(pjrt_api, .{
        .data = buffer.data,
        .dims = shape.dims(),
        .element_type = pjrtx.bufferTypeFromDtype(shape.dtype()),
        .device = device,
        .layout = .{
            .tiled = .{
                .minor_to_major = zml.constants.minorToMajor(shape.rank()),
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
    });

    pjrt_buffer = try pjrt_buffer.copyToMemory(pjrt_api, first_non_device_memory);
    try pjrt_buffer.readyEvent(pjrt_api).awaitRaw(pjrt_api);

    const host_visible_data: [*]u8 = @ptrCast(@alignCast(try pjrt_buffer.opaqueDeviceMemoryDataPointer(pjrt_api)));

    const slice: zml.Slice = .init(shape, host_visible_data[0..shape.byteSize()]);
    const name = call_frame.attrs.getByName(.string, "name").?.slice();

    std.debug.print("{s} {f} [device={d}]: {d}\n", .{ name, slice.shape, device_ordinal, slice });

    return null;
}

test "platform defaultMemoryLayout is boring" {
    const platform = zml.testing.env();

    const shapes = [_][]const i64{
        &.{4096},
        &.{ 4096, 4096 },
        &.{ 4096, 4096, 4096 },
    };
    for (shapes) |dims| {
        // Checks that the PJRT client always return the same thing than `platform.defaultMemoryLayout`
        // This allows to bypass the PJRT calls and the string of pjrt_client.defaultMemoryLayout.
        const default_layout = try platform.pjrt_client.defaultMemoryLayout(platform.pjrt_api, pjrtx.bufferTypeFromDtype(.f32), dims);
        const mem_layout = default_layout.toMemoryLayout();

        // Note: I'm not just calling platform.defaultMemoryLayout because I'm investigating
        // wether TPU requires its special branch.
        try std.testing.expectEqualDeep(mem_layout, pjrt.MemoryLayout{
            .tiled = .{
                .minor_to_major = constants.minorToMajor(@intCast(dims.len)),
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        });
    }
}
