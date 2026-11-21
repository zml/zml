const std = @import("std");
const builtin = @import("builtin");

const c = @import("c");
const constants = @import("zml/constants");
const pjrt = @import("pjrt");
const profiler_ = @import("zml/profiling").profiler;
const stdx = @import("stdx");
const tenstorrent_platform = @import("platforms/tenstorrent");
pub const Target = @import("platforms").Platform;

const Exe = @import("exe.zig").Exe;
const flashinfer_cutlass_contract = @import("flashinfer_cutlass/contract.zig");
const flashinfer_cutlass_runtime = @import("flashinfer_cutlass/runtime.zig");
const pjrtx = @import("pjrtx.zig");
const Sharding = @import("Sharding.zig");
const zml = @import("core.zig");

const log = std.log.scoped(.zml);

var api_map: std.enums.EnumArray(Target, ?*const pjrt.Api) = .initFill(null);

/// Tenstorrent clients still alive when the process exits. A client that is
/// never destroyed leaves the cards' Ethernet cores in fabric mode, and every
/// later open of the cards times out until they are reset; `zml.testing`'s
/// platform, for one, is never deinit'ed. Clients left are destroyed when the
/// main thread exits normally (return from main, `std.process.exit`), by a
/// thread-local destructor of the main thread: glibc's `exit` runs those
/// before any `atexit` handler, and so before the destructors of the plugin's
/// C++ statics (an `atexit` release crashed in tt-metal's GraphTracker, already
/// destroyed). A crash or a fatal signal still leaves the cards to a reset.
const exit_release = struct {
    const Entry = struct { api: *const pjrt.Api, client: *pjrt.Client };

    var mutex: std.atomic.Mutex = .unlocked;
    var entries: stdx.BoundedArray(Entry, 32) = .empty;
    var registered = false;

    /// glibc's registration of thread-local destructors, as C++ `thread_local` uses it.
    extern "c" fn __cxa_thread_atexit_impl(func: *const fn (?*anyopaque) callconv(.c) void, obj: ?*anyopaque, dso_symbol: ?*anyopaque) c_int;

    fn track(api: *const pjrt.Api, client: *pjrt.Client) void {
        if (comptime builtin.os.tag != .linux) return;
        lock();
        defer mutex.unlock();
        if (!registered) {
            // Another thread's destructors run when that thread ends, not at exit.
            if (std.os.linux.gettid() != std.os.linux.getpid()) {
                log.warn("Tenstorrent client created off the main thread: deinit its platform before the process ends, or the cards stay unusable until a reset", .{});
                return;
            }
            if (__cxa_thread_atexit_impl(releaseAll, null, @ptrCast(&registered)) != 0) {
                log.warn("Cannot release Tenstorrent clients at exit: deinit every platform before the process ends", .{});
                return;
            }
            registered = true;
        }
        entries.append(.{ .api = api, .client = client }) catch {
            log.warn("More than {} live Tenstorrent clients: the others are not released at exit", .{entries.capacity()});
        };
    }

    fn untrack(client: *pjrt.Client) void {
        if (comptime builtin.os.tag != .linux) return;
        lock();
        defer mutex.unlock();
        for (entries.constSlice(), 0..) |entry, i| {
            if (entry.client == client) {
                _ = entries.swapRemove(i);
                return;
            }
        }
    }

    fn releaseAll(_: ?*anyopaque) callconv(.c) void {
        lock();
        defer mutex.unlock();
        for (entries.constSlice()) |entry| {
            log.info("Closing a Tenstorrent client left open at exit", .{});
            entry.client.deinit(entry.api);
        }
        entries.clear();
    }

    fn lock() void {
        while (!mutex.tryLock()) std.atomic.spinLoopHint();
    }
};

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
        .cpu, .cuda, .rocm, .tpu, .neuron, .metal, .oneapi, .furiosa => {
            if (!std.math.isPowerOfTwo(num_devices)) {
                log.err("Platform {} requires a power-of-two device count, got {}", .{ target, num_devices });
                return error.InvalidDeviceCount;
            }
        },
        // The backend partitions over any view of the cabling, e.g. three cards in a line.
        .tenstorrent => {},
    }
}

fn loadOrGetApi(allocator: std.mem.Allocator, io: std.Io, target: Target, options: CreateOptions) !*const pjrt.Api {
    return switch (target) {
        inline else => |tgt| api_map.get(tgt) orelse b: {
            disableXlaLogs();
            const api = if (tgt == .tenstorrent and options.tenstorrent.runtime == .fake)
                // The device-free runtime needs no card on this host.
                try tenstorrent_platform.loadPlugin(allocator, io)
            else
                try tgt.load(allocator, io);
            api_map.set(tgt, api);
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
            .cuda, .rocm, .oneapi, .tpu, .furiosa, .tenstorrent => {
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

    /// NUMA node closest to the device, when the plugin reports one.
    pub fn numaNode(self: Device) ?u32 {
        const attributes = self.pjrt_device.attributes(self.platform.pjrt_api) catch return null;
        defer attributes.deinit();

        const value = attributes.get("numa_node") orelse return null;

        return switch (value) {
            .int64 => |node| std.math.cast(u32, node),
            else => null,
        };
    }

    pub fn format(self: Device, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("{s} ({s})", .{
            self.pjrt_desc.kind(self.platform.pjrt_api),
            self.pjrt_desc.debugString(self.platform.pjrt_api),
        });
    }

    pub fn memory(self: *const Device, memory_kind: Memory.Kind) ?*const Memory {
        return self.memory_by_kind.values[@backingInt(memory_kind)];
    }
};

fn platformDeviceSortId(target: Target, device: Device) usize {
    return switch (target) {
        .neuron => @intCast(device.localHardwareId()),
        .cuda, .rocm, .tpu, .cpu, .oneapi, .metal, .furiosa, .tenstorrent => device.id(),
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

    if (builtin.mode == .debug) {
        for (devices, 0..) |device, expected_id| {
            std.debug.assert(platformDeviceSortId(target, device) == expected_id);
        }
    }
}

// State union tagged on target platform to handle related resources
pub const State = union(Target) {
    cpu: void,
    cuda: CudaState,
    rocm: void,
    tpu: void,
    neuron: void,
    oneapi: void,
    metal: void,
    furiosa: void,
    tenstorrent: void,

    pub const CudaState = struct {
        flashinfer_cutlass_moe_runners: ?*flashinfer_cutlass_runtime.Runners = null,

        fn deinit(self: *CudaState) void {
            if (self.flashinfer_cutlass_moe_runners) |runners| {
                runners.deinit();
                self.flashinfer_cutlass_moe_runners = null;
            }
        }
    };

    pub fn init(target: Target) State {
        return switch (target) {
            .cpu => .{ .cpu = {} },
            .cuda => .{ .cuda = .{} },
            .rocm => .{ .rocm = {} },
            .tpu => .{ .tpu = {} },
            .neuron => .{ .neuron = {} },
            .oneapi => .{ .oneapi = {} },
            .metal => .{ .metal = {} },
            .furiosa => .{ .furiosa = {} },
            .tenstorrent => .{ .tenstorrent = {} },
        };
    }

    pub fn deinit(self: *State) void {
        switch (self.*) {
            .cuda => |*cuda_state| cuda_state.deinit(),
            else => {},
        }
    }
};

pub const Platform = struct {
    arena: std.heap.ArenaAllocator,
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,
    state: State,
    devices: []const Device,
    memories: []const Memory,
    physical_mesh: zml.Sharding.PhysicalMesh,
    meshes: std.StringArrayHashMapUnmanaged(*const zml.Mesh),
    replicated_mesh: *const zml.Mesh,
    io_impl: IoImpl,
    /// See `CreateOptions.pjrt_program_dump_dir`.
    pjrt_program_dump_dir: ?[]const u8 = null,

    pub const MAX_NUM_DEVICES: u16 = if (Target.tpu.isEnabled()) 64 else 32;

    pub const IoImpl = enum {
        threaded,
        zio,
    };

    pub fn init(allocator: std.mem.Allocator, io: std.Io, target: Target, options: CreateOptions) !*Platform {
        const api = try loadOrGetApi(allocator, io, target, options);

        var create_options = options;
        var defaults_arena: std.heap.ArenaAllocator = .init(allocator);
        defer defaults_arena.deinit();
        if (target == .tenstorrent) {
            create_options.tenstorrent = try options.tenstorrent.withDefaultPaths(defaults_arena.allocator(), io);
        }

        var named_values_buf: [16]pjrt.NamedValue = undefined;
        const pjrt_client = try pjrt.Client.init(api, create_options.toNamedValues(target, &named_values_buf));
        // A client may hold hardware (Tenstorrent cards stay unusable until a
        // reset if their client is never destroyed): release it on every failure below.
        errdefer pjrt_client.deinit(api);
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
                .state = State.init(target),
                .meshes = .empty,
                .replicated_mesh = undefined,
                // set below
                .devices = undefined,
                .memories = undefined,
                .physical_mesh = undefined,
                .io_impl = options.io_impl,
                .pjrt_program_dump_dir = if (options.pjrt_program_dump_dir) |dir| try initial_allocator.dupe(u8, dir) else null,
            };
            break :platform platform;
        };

        const arena = platform.arena.allocator();
        errdefer platform.arena.deinit();
        try platform.meshes.ensureTotalCapacity(arena, 8);

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
            platform.replicated_mesh = try platform.registerMesh("replicated", .mesh(.{ .all = .high_bandwidth }));
        }

        switch (target) {
            .cuda => {
                cuda.initFlashinferCutlass(platform, io) catch |err| {
                    log.warn("Failed to initialize Flashinfer CUTLASS MoE: {}", .{err});
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

        if ((comptime Target.tenstorrent.isEnabled()) and target == .tenstorrent) exit_release.track(api, pjrt_client);
        return platform;
    }

    /// Returns the first platform of this build that has devices on this host.
    /// A platform whose plugin is present but fails to start is reported: a
    /// Tenstorrent failure is returned, other platforms are skipped after a warning.
    pub fn auto(allocator: std.mem.Allocator, io: std.Io, options: CreateOptions) !*Platform {
        const ordered_targets: []const Target = &.{
            .tpu,
            .neuron,
            .rocm,
            .cuda,
            .oneapi,
            .metal,
            .furiosa,
            .tenstorrent,
            .cpu,
        };
        return for (ordered_targets) |target| {
            break init(allocator, io, target, options) catch |err| switch (err) {
                error.Unavailable => continue,
                else => {
                    // Cards that are present but cannot be opened (held by
                    // another process, wedged, misconfigured) must not turn
                    // a run into a silent CPU run.
                    if (target == .tenstorrent) {
                        log.err("Platform {} has devices on this host but failed to start: {}", .{ target, err });
                        return err;
                    }
                    log.warn("Platform {} failed to start, trying the next one: {}", .{ target, err });
                    continue;
                },
            };
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
        if (comptime Target.cuda.isEnabled()) {
            self.state.deinit();
        }
        self.physical_mesh.deinit(self.arena.allocator());
        if ((comptime Target.tenstorrent.isEnabled()) and self.target == .tenstorrent) exit_release.untrack(self.pjrt_client);
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
        opts: zml.Compiler.Options,
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
        opts: zml.Compiler.Options,
    ) !Exe {
        return self.compileFn(allocator, io, func, .{model} ++ args, opts);
    }

    pub fn compileFn(
        self: *const Platform,
        allocator: std.mem.Allocator,
        io: std.Io,
        comptime func: anytype,
        args: std.meta.ArgsTuple(@TypeOf(func)),
        opts: zml.Compiler.Options,
    ) !Exe {
        return zml.Compiler.compile(allocator, io, self, func, args, opts);
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

    /// Create a Mesh based on the given logical mesh and the default strategy.
    /// Memory is owned by the platform, making it safe to copy around.
    pub fn registerMesh(platform: *Platform, name: []const u8, logical: Sharding.LogicalMesh) error{OutOfMemory}!*const Sharding.Mesh {
        return platform.registerMeshWithStrategy(
            name,
            logical,
            .suggest(logical, &platform.physical_mesh),
        ) catch |err| switch (err) {
            error.InvalidPhysicalMesh, error.InvalidStrategy, error.InvalidPhysicalAxis, error.OverlappingLogicalAxes => {
                std.debug.panic("ZML failed to create a valid mesh for logical mesh: {f}\nand physical_mesh: {f}\nPlease report this bug.", .{ logical, platform.physical_mesh });
            },
            error.InvalidPartialBinding, error.InvalidLogicalDimension, error.IncompatibleLogicalDimension => {
                std.debug.panic("ZML cannot satisfy the dimensions of logical mesh {f} on physical mesh {f}", .{ logical, platform.physical_mesh });
            },
            error.OutOfMemory => |e| return e,
        };
    }

    /// Create a Mesh based on the given logical mesh and a strategy.
    /// Memory is owned by the platform, making it safe to copy around.
    pub fn registerMeshWithStrategy(platform: *Platform, name: []const u8, logical: Sharding.LogicalMesh, strategy: Sharding.Strategy) !*const Sharding.Mesh {
        const arena = platform.arena.allocator();
        const entry = try platform.meshes.getOrPut(arena, name);
        if (entry.found_existing) {
            std.debug.panic("Another mesh already exists with this name: {s}", .{name});
        }

        const owned_name = try arena.dupe(u8, name);
        const owned_mesh = try arena.create(Sharding.Mesh);
        owned_mesh.* = try .init(owned_name, &platform.physical_mesh, logical, strategy);
        entry.key_ptr.* = owned_name;
        entry.value_ptr.* = owned_mesh;

        return owned_mesh;
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

    pub inline fn defaultMemoryLayout(platform: *const Platform, dims: []const i64, dtype: zml.DataType) !pjrt.MemoryLayout {
        // inline cause `default` is a huge ass struct allocated on the stack,
        // and toMemoryLayout returns slices into it.
        // There is probably a better way of doing this,
        // The queried layout can include backend-specific tiling.
        return switch (platform.target) {
            .tpu, .furiosa, .tenstorrent => {
                if (comptime !Target.tpu.isEnabled() and !Target.furiosa.isEnabled() and !Target.tenstorrent.isEnabled()) unreachable;
                const element_type = pjrtx.bufferTypeFromDtype(dtype);
                // A plugin may hold only some types (Tenstorrent's M0: f32, bf16, i32,
                // u32, u16, u8 and bool): the buffer that asks fails with its error.
                const default = platform.pjrt_client.defaultMemoryLayout(platform.pjrt_api, element_type, dims) catch |err| {
                    log.warn("{} has no default memory layout for {s} {any}: {}", .{ platform.target, @tagName(dtype), dims, err });
                    return err;
                };
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

    /// Number of execution cores (such as GPU streaming multiprocessors or TPU tensor cores) available on a specific hardware device
    pub fn coreCountPerDevice(platform: *const zml.Platform) ?u32 {
        const attribute = platform.devices[0].pjrt_desc.attribute(platform.pjrt_api, "core_count") orelse return null;
        return if (attribute.int64 > 0) std.math.cast(u32, attribute.int64) else null;
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
    rocm: XlaGpu = .{ .allocator = .{ .vmm = .{ .memory_fraction = 0.90 } } },
    cuda: XlaGpu = .{ .allocator = .{ .vmm = .{ .memory_fraction = 0.90 } } },
    tpu: struct {} = .{},
    neuron: struct {} = .{},
    furiosa: Furiosa = .{},
    tenstorrent: Tenstorrent = .{},
    oneapi: XlaGpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    metal: XlaGpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    io_impl: Platform.IoImpl = .threaded,
    /// Debugging: every compile on this platform writes the program it hands to
    /// PJRT under this directory before compiling it (see `pjrtx.ProgramDump`),
    /// named after `Compiler.Options.program_name`.
    pjrt_program_dump_dir: ?[]const u8 = null,

    pub const Furiosa = struct {
        pe_count: u8 = 8,

        fn writeNamedValues(self: Furiosa, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(.init(.int64, "pe_count", self.pe_count));
        }
    };

    /// Client options of the Tenstorrent plugin (`tt_*` named values).
    pub const Tenstorrent = struct {
        runtime: Runtime = .ttnn_eager,
        /// tt-umd cluster descriptor (YAML) to take the system from instead
        /// of discovering the cards; required by the device-free runtime.
        cluster_descriptor: ?[]const u8 = null,
        /// Forces the logical view, e.g. `&.{ 2, 2 }`; empty lets the plugin choose.
        mesh_shape: []const i64 = &.{},
        /// Data-parallel and tensor-parallel degrees the program will use,
        /// `&.{ dp, tp }`, for the plugin to choose a view that suits them.
        mesh_hint: []const i64 = &.{},
        /// Comma-separated board ids (device attribute `tt_board_id`) the
        /// client may use; null uses every reachable card.
        visible_chips: ?[]const u8 = null,
        /// tt-metal runtime root; defaults to the sandbox's.
        runtime_root: ?[]const u8 = null,
        /// Writable kernel JIT cache, reused across processes; defaults to
        /// `$XDG_CACHE_HOME/zml/tenstorrent`, else `$HOME/.cache/zml/tenstorrent`.
        cache_dir: ?[]const u8 = null,
        /// tt-metal logs; the plugin defaults to `<cache_dir>/logs`.
        logs_dir: ?[]const u8 = null,
        /// DRAM left to tt-metal per chip (kernel binaries, TTNN temporaries).
        dram_reserve_bytes: ?i64 = null,
        /// tt-metal trace region per chip, fixed when the mesh opens.
        trace_region_bytes: ?i64 = null,
        /// XLA-owned memory per device (host memory for the device-free runtime).
        arena_bytes: ?i64 = null,

        pub const Runtime = enum {
            /// The cards of this host, through TTNN.
            ttnn_eager,
            /// No card: compiles against `cluster_descriptor` and runs on the host.
            fake,
        };

        /// Fills the directories the card runtime needs and that only the
        /// loader knows: the sandbox's runtime root and a persistent cache.
        /// Strings are allocated in `allocator`.
        pub fn withDefaultPaths(self: Tenstorrent, allocator: std.mem.Allocator, io: std.Io) !Tenstorrent {
            var resolved = self;
            if (self.runtime == .fake) return resolved;

            if (resolved.runtime_root == null) {
                resolved.runtime_root = try tenstorrent_platform.runtimeRoot(allocator, io);
            }
            if (resolved.cache_dir == null) {
                resolved.cache_dir = if (std.c.getenv("XDG_CACHE_HOME")) |cache_home|
                    try std.Io.Dir.path.join(allocator, &.{ std.mem.span(cache_home), "zml", "tenstorrent" })
                else if (std.c.getenv("HOME")) |home|
                    try std.Io.Dir.path.join(allocator, &.{ std.mem.span(home), ".cache", "zml", "tenstorrent" })
                else {
                    log.err("Tenstorrent needs a writable JIT cache: set CreateOptions.tenstorrent.cache_dir, $XDG_CACHE_HOME or $HOME", .{});
                    return error.MissingCacheDir;
                };
            }
            return resolved;
        }

        fn writeNamedValues(self: Tenstorrent, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(.init(.string, "tt_runtime", @tagName(self.runtime)));
            if (self.cluster_descriptor) |path| values.appendAssumeCapacity(.init(.string, "tt_cluster_descriptor", path));
            if (self.mesh_shape.len > 0) values.appendAssumeCapacity(.init(.int64list, "tt_mesh_shape", self.mesh_shape));
            if (self.mesh_hint.len > 0) values.appendAssumeCapacity(.init(.int64list, "tt_mesh_hint", self.mesh_hint));
            if (self.visible_chips) |chips| values.appendAssumeCapacity(.init(.string, "tt_visible_chips", chips));
            if (self.runtime_root) |path| values.appendAssumeCapacity(.init(.string, "tt_runtime_root", path));
            if (self.cache_dir) |path| values.appendAssumeCapacity(.init(.string, "tt_cache_dir", path));
            if (self.logs_dir) |path| values.appendAssumeCapacity(.init(.string, "tt_logs_dir", path));
            if (self.dram_reserve_bytes) |bytes| values.appendAssumeCapacity(.init(.int64, "tt_dram_reserve_bytes", bytes));
            if (self.trace_region_bytes) |bytes| values.appendAssumeCapacity(.init(.int64, "tt_trace_region_bytes", bytes));
            if (self.arena_bytes) |bytes| values.appendAssumeCapacity(.init(.int64, "tt_arena_bytes", bytes));
        }
    };

    pub const Cpu = struct {
        device_count: u32,

        fn writeNamedValues(self: Cpu, values: *std.ArrayList(pjrt.NamedValue)) void {
            values.appendAssumeCapacity(.init(.int64, "cpu_device_count", self.device_count));
        }
    };

    pub const XlaGpu = struct {
        allocator: Allocator = .{ .bfc = .{} },
        /// The PJRT C API still exposes this under legacy
        /// `use_tfrt_gpu_client` name.
        gpu_async_dispatch: bool = false,
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
            /// use virtual memory allocator
            vmm: VmmOptions,
            /// use raw cuMalloc
            platform,

            pub const Options = struct {
                preallocate: bool = true,
                memory_fraction: f32 = 0.90,
                collective_memory_size_mb: i64 = 0,
            };

            pub const VmmOptions = struct {
                memory_fraction: f32 = 0.90,
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
                        .platform, .vmm => unreachable,
                    }));
                    values.appendAssumeCapacity(.init(.bool, "preallocate", opt.preallocate));
                    if (opt.memory_fraction > 0) {
                        values.appendAssumeCapacity(.init(.float, "memory_fraction", opt.memory_fraction));
                    }
                    if (opt.collective_memory_size_mb > 0) {
                        values.appendAssumeCapacity(.init(.int64, "collective_memory_size", opt.collective_memory_size_mb * 1024 * 1024));
                    }
                },
                .vmm => |opt| {
                    values.appendAssumeCapacity(.init(.string, "allocator", "vmm"));
                    if (opt.memory_fraction > 0) {
                        values.appendAssumeCapacity(.init(.float, "memory_fraction", opt.memory_fraction));
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
            .furiosa => self.furiosa.writeNamedValues(&values),
            .tenstorrent => self.tenstorrent.writeNamedValues(&values),
            .cuda => self.cuda.writeNamedValues(target, &values),
            .rocm => self.rocm.writeNamedValues(target, &values),
            .oneapi => self.oneapi.writeNamedValues(target, &values),
            .metal => self.metal.writeNamedValues(target, &values),
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
    pub const ComputeCapability = struct {
        major: u8,
        minor: u8,

        pub fn parse(text: []const u8) ?ComputeCapability {
            var parts = std.mem.splitScalar(u8, text, '.');
            return .{
                .major = std.fmt.parseInt(u8, parts.first(), 10) catch return null,
                .minor = std.fmt.parseInt(u8, parts.next() orelse "0", 10) catch return null,
            };
        }

        pub fn eql(self: ComputeCapability, other: ComputeCapability) bool {
            return self.major == other.major and self.minor == other.minor;
        }

        pub fn atLeast(self: ComputeCapability, other: ComputeCapability) bool {
            return self.major > other.major or (self.major == other.major and self.minor >= other.minor);
        }

        pub fn sm(self: ComputeCapability) u16 {
            return @as(u16, self.major) * 10 + self.minor;
        }
    };

    pub fn computeCapability(platform: *const zml.Platform) ?ComputeCapability {
        if (platform.target != .cuda) return null;
        const devices = platform.pjrt_client.devices(platform.pjrt_api);
        if (devices.len == 0) return null;

        const attributes = devices[0].getDescription(platform.pjrt_api).attributes(platform.pjrt_api);
        return for (attributes) |attr| {
            if (std.mem.eql(u8, attr.name(), "compute_capability")) {
                break ComputeCapability.parse(attr.value().string);
            }
        } else null;
    }

    fn initFlashinferCutlass(platform: *Platform, io: std.Io) !void {
        if (comptime !Target.cuda.isEnabled()) return error.UnsupportedPlatform;
        const cc = computeCapability(platform) orelse return error.UnsupportedArchitecture;

        const allocator = platform.arena.allocator();
        const runners = try allocator.create(flashinfer_cutlass_runtime.Runners);
        errdefer allocator.destroy(runners);

        try flashinfer_cutlass_runtime.load(allocator, io, cc.sm());

        runners.* = .{};
        errdefer runners.deinit();

        try platform.registerFfi(.{
            .name = flashinfer_cutlass_contract.name,
            .handler = flashinfer_cutlass_runtime.handler,
            .traits = .{ .command_buffer_compatible = true },
            .platform_name = if (flashinfer_cutlass_contract.compute_on_host) "Host" else null,
        });

        platform.state.cuda.flashinfer_cutlass_moe_runners = runners;
    }
};

pub const rocm = struct {
    pub const ComputeCapability = enum {
        /// CDNA1: MI100.
        gfx908,
        /// CDNA2: MI200 series.
        gfx90a,
        /// CDNA3: MI300 series.
        gfx942,
        /// CDNA4: MI350 series.
        gfx950,
        /// RDNA2: RX 6000 series.
        gfx1030,
        /// RDNA3 (Navi 31): RX 7900 series.
        gfx1100,
        /// RDNA3 (Navi 32): RX 7800 and RX 7700 series.
        gfx1101,
        /// RDNA3 (Navi 33): RX 7600 series.
        gfx1102,
        /// RDNA3 APU: Phoenix.
        gfx1103,
        /// RDNA3.5: newer Ryzen AI APUs.
        gfx1150,
        /// RDNA4 (Navi 44): RX 9060 family.
        gfx1200,
        /// RDNA4 (Navi 48): RX 9070 family.
        gfx1201,

        pub const Architecture = enum {
            cdna1,
            cdna2,
            cdna3,
            cdna4,
            rdna2,
            rdna3,
            rdna3_5,
            rdna4,
        };

        pub fn architecture(self: ComputeCapability) Architecture {
            return switch (self) {
                .gfx908 => .cdna1,
                .gfx90a => .cdna2,
                .gfx942 => .cdna3,
                .gfx950 => .cdna4,
                .gfx1030 => .rdna2,
                .gfx1100, .gfx1101, .gfx1102, .gfx1103 => .rdna3,
                .gfx1150 => .rdna3_5,
                .gfx1200, .gfx1201 => .rdna4,
            };
        }
    };

    /// Assumes homogeneous devices.
    pub fn computeCapability(platform: *const zml.Platform) ?ComputeCapability {
        stdx.debug.assert(platform.target == .rocm, "computeCapability expects .rocm platform, got {}", .{platform.target});
        const device = platform.pjrt_client.devices(platform.pjrt_api)[0];
        const description = device.getDescription(platform.pjrt_api);

        const attributes = description.attributes(platform.pjrt_api);
        return for (attributes) |attr| {
            if (std.mem.eql(u8, attr.name(), "compute_capability")) {
                break std.meta.stringToEnum(ComputeCapability, std.mem.sliceTo(attr.value().string, ':'));
            }
        } else null;
    }
};

pub const tenstorrent = struct {
    pub const MAX_VIEW_RANK = Sharding.MAX_MESH_RANK;

    pub const AxisKind = enum {
        /// An axis of extent 1, as the view of a single chip.
        singleton,
        /// Chips linked to their neighbours, open ends.
        line,
        /// Chips linked to their neighbours, wrap-around link cabled.
        ring,
    };

    pub const Axis = struct {
        extent: u32,
        kind: AxisKind,
        links_per_hop: u32,
    };

    /// The logical view the plugin serves, identical on every device.
    pub const View = struct {
        axes: stdx.BoundedArray(Axis, MAX_VIEW_RANK) = .empty,

        /// Parses `tt_view_shape`, `tt_view_axis_kinds` ("line,ring") and `tt_view_axis_links`.
        pub fn parse(shape: []const i64, kinds: []const u8, links: []const i64) !View {
            if (shape.len == 0 or shape.len > MAX_VIEW_RANK or links.len != shape.len) return error.InvalidTenstorrentView;
            var view: View = .{};
            var kind_names = std.mem.splitScalar(u8, kinds, ',');
            for (shape, links) |extent, link_count| {
                const kind_name = kind_names.next() orelse return error.InvalidTenstorrentView;
                const kind = std.meta.stringToEnum(AxisKind, kind_name) orelse return error.InvalidTenstorrentView;
                if (extent < 1 or link_count < 0) return error.InvalidTenstorrentView;
                if ((kind == .singleton) != (extent == 1)) return error.InvalidTenstorrentView;
                view.axes.appendAssumeCapacity(.{
                    .extent = std.math.cast(u32, extent) orelse return error.InvalidTenstorrentView,
                    .kind = kind,
                    .links_per_hop = std.math.cast(u32, link_count) orelse return error.InvalidTenstorrentView,
                });
            }
            if (kind_names.next() != null) return error.InvalidTenstorrentView;
            return view;
        }

        pub fn eql(a: View, b: View) bool {
            if (a.axes.len != b.axes.len) return false;
            for (a.axes.constSlice(), b.axes.constSlice()) |x, y| {
                if (!std.meta.eql(x, y)) return false;
            }
            return true;
        }

        pub fn deviceCount(self: View) usize {
            var count: usize = 1;
            for (self.axes.constSlice()) |ax| count *= ax.extent;
            return count;
        }
    };

    /// What the plugin tells about one device, beyond its PJRT id.
    pub const DeviceInfo = struct {
        /// Position on the non-singleton axes of `view`, `[0]` for a single chip.
        coords: stdx.BoundedArray(u32, MAX_VIEW_RANK),
        view: View,
        /// Board id of the chip, as `CreateOptions.tenstorrent.visible_chips` takes it.
        board_id: u64,
        /// DRAM bandwidth, whole GB/s.
        dram_gbps: i64,
        /// Latency floor of an all-reduce over the whole view, in microseconds.
        ccl_base_us: f64,

        pub fn init(platform: *const Platform, device: Device) !DeviceInfo {
            stdx.debug.assert(platform.target == .tenstorrent, "tenstorrent.DeviceInfo expects a Tenstorrent platform, got {}", .{platform.target});
            const api = platform.pjrt_api;
            const desc = device.pjrt_desc;
            const coords = try int64List(desc.attribute(api, "coords"));
            var info: DeviceInfo = .{
                .coords = .empty,
                .view = try .parse(
                    try int64List(desc.attribute(api, "tt_view_shape")),
                    try string(desc.attribute(api, "tt_view_axis_kinds")),
                    try int64List(desc.attribute(api, "tt_view_axis_links")),
                ),
                // PJRT has no unsigned attribute: the plugin keeps the bit pattern.
                .board_id = @bitCast(try int64(desc.attribute(api, "tt_board_id"))),
                .dram_gbps = try int64(desc.attribute(api, "tt_dram_gbps")),
                // The PJRT C API carries no float attribute, hence a decimal string.
                .ccl_base_us = std.fmt.parseFloat(f64, try string(desc.attribute(api, "tt_ccl_base_us"))) catch return error.InvalidDeviceAttribute,
            };
            if (coords.len == 0 or coords.len > MAX_VIEW_RANK) return error.InvalidDeviceCoords;
            for (coords) |coord| {
                info.coords.appendAssumeCapacity(std.math.cast(u32, coord) orelse return error.InvalidDeviceCoords);
            }
            return info;
        }

        fn int64(value: ?pjrt.NamedValue.Value) !i64 {
            const v = value orelse return error.MissingDeviceAttribute;
            return if (v == .int64) v.int64 else error.InvalidDeviceAttribute;
        }

        fn int64List(value: ?pjrt.NamedValue.Value) ![]const i64 {
            const v = value orelse return error.MissingDeviceAttribute;
            return if (v == .int64list) v.int64list else error.InvalidDeviceAttribute;
        }

        fn string(value: ?pjrt.NamedValue.Value) ![]const u8 {
            const v = value orelse return error.MissingDeviceAttribute;
            return if (v == .string) v.string else error.InvalidDeviceAttribute;
        }
    };

    test "View.parse reads the plugin's view attributes" {
        const one: View = try .parse(&.{1}, "singleton", &.{0});
        try std.testing.expectEqual(1, one.deviceCount());
        try std.testing.expectEqual(AxisKind.singleton, one.axes.get(0).kind);

        const grid: View = try .parse(&.{ 2, 4 }, "line,ring", &.{ 2, 4 });
        try std.testing.expectEqual(8, grid.deviceCount());
        try std.testing.expectEqual(Axis{ .extent = 4, .kind = .ring, .links_per_hop = 4 }, grid.axes.get(1));
        try std.testing.expect(grid.eql(try .parse(&.{ 2, 4 }, "line,ring", &.{ 2, 4 })));
        try std.testing.expect(!grid.eql(try .parse(&.{ 2, 4 }, "line,line", &.{ 2, 4 })));

        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{ 2, 2 }, "line", &.{ 2, 2 }));
        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{2}, "line,line", &.{2}));
        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{2}, "torus", &.{2}));
        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{1}, "line", &.{2}));
        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{3}, "singleton", &.{2}));
        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{}, "", &.{}));
        try std.testing.expectError(error.InvalidTenstorrentView, View.parse(&.{ 2, 2, 2, 2, 2 }, "line,line,line,line,line", &.{ 2, 2, 2, 2, 2 }));
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

    const buffers = call_frame.args.buffers();
    if (buffers.len <= 0) return null;

    const pjrt_api: *pjrt.Api = @ptrFromInt(getScalarAttributeAs(u64, call_frame, "pjrt_api").?);
    const pjrt_client: *pjrt.Client = @ptrFromInt(getScalarAttributeAs(u64, call_frame, "pjrt_client").?);

    const device_ordinal: usize = @intCast(try call_frame.ctx.getDeviceOrdinal(call_frame.api));

    const buffer = buffers[0];
    const shape = shapeFromFfiBuffer(buffer);

    // NOTE(Corentin): This is a hack. We take the first non device memory, hoping that it's host visible,
    // and copy the buffer there to read it on the CPU and print it.
    const device = pjrt_client.devices(pjrt_api)[device_ordinal];
    const addressable_memories = device.addressableMemories(pjrt_api);
    const first_non_device_memory = for (addressable_memories) |memory| {
        if (!std.mem.eql(u8, memory.kind_(pjrt_api), "device")) break memory;
    } else return error.MemoryNotFound;

    var pjrt_buffer = try pjrt_client.createViewOfDeviceBuffer(pjrt_api, .{
        .device_buffer_ptr = buffer.data,
        .dims = shape.dims(),
        .element_type = pjrtx.bufferTypeFromDtype(shape.dtype()),
        .device = device,
        .layout = .{
            .tiled = .{
                .minor_to_major = constants.minorToMajor(shape.rank()),
                .tile_dims = &.{},
                .tile_dims_sizes = &.{},
            },
        },
    });
    defer pjrt_buffer.deinit(pjrt_api);

    pjrt_buffer = try pjrt_buffer.copyToMemory(pjrt_api, first_non_device_memory);

    const event = pjrt_buffer.readyEvent(pjrt_api);
    defer event.deinit(pjrt_api);
    try event.awaitRaw(pjrt_api);

    const host_visible_data: [*]u8 = @ptrCast(@alignCast(try pjrt_buffer.opaqueDeviceMemoryDataPointer(pjrt_api)));

    const slice: zml.Slice = .init(shape, host_visible_data[0..shape.byteSize()]);
    const name = call_frame.attrs.getByName(.string, "name").?.slice();

    std.debug.print("{s} [device={d}]: {d}\n", .{ name, device_ordinal, slice });

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

        if (platform.target == .furiosa or platform.target == .tenstorrent) {
            try std.testing.expectEqualDeep(mem_layout, try platform.defaultMemoryLayout(dims, .f32));
            continue;
        }

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
