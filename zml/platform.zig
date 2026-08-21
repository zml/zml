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

const ExecutionTimerRegistration = struct {
    type_id: ?pjrt.ffi.TypeId = null,
    start_handler_registered: bool = false,
    stop_handler_registered: bool = false,
};

var execution_timer_registrations: std.enums.EnumArray(Target, ExecutionTimerRegistration) = .initFill(.{});

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

// State union tagged on target platform to handle related resources
pub const State = union(Target) {
    cpu: void,
    cuda: CudaState,
    rocm: void,
    tpu: void,
    neuron: void,
    oneapi: void,
    metal: void,

    pub const CudaState = struct {
        fi_cutlass_moe_runners: ?*zml.moe.cutlass_flashinfer.Runners = null,

        fn deinit(self: *CudaState) void {
            if (self.fi_cutlass_moe_runners) |runners| {
                runners.deinit();
                self.fi_cutlass_moe_runners = null;
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
        };
    }

    pub fn deinit(self: *State) void {
        switch (self.*) {
            .cuda => |*cuda_state| cuda_state.deinit(),
            else => {},
        }
    }
};

threadlocal var autotune_nesting_depth: usize = 0;

fn executionTimerAvailableFor(target: Target, has_type_id: bool, has_event_provider: bool, has_ffi: bool) bool {
    return switch (target) {
        .cuda, .rocm => has_type_id and has_event_provider and has_ffi,
        else => false,
    };
}

test "execution timer availability requires a supported fully registered backend" {
    try std.testing.expect(executionTimerAvailableFor(.cuda, true, true, true));
    try std.testing.expect(executionTimerAvailableFor(.rocm, true, true, true));
    try std.testing.expect(!executionTimerAvailableFor(.cpu, true, true, true));
    try std.testing.expect(!executionTimerAvailableFor(.cuda, false, true, true));
    try std.testing.expect(!executionTimerAvailableFor(.cuda, true, false, true));
    try std.testing.expect(!executionTimerAvailableFor(.cuda, true, true, false));
}

pub const AutotuneState = struct {
    const CachedResult = struct {
        candidate_index: usize,
        candidate_count: usize,
        median_ns: i96,
        mad_ns: i96,
        repetitions: usize,
        sample_count: usize,
        compiled_count: usize,
        rejected_count: usize,
        options: zml.AutotuneOptions,

        fn init(result: anytype, candidate_count: usize, options: zml.AutotuneOptions) CachedResult {
            return .{
                .candidate_index = result.candidate_index,
                .candidate_count = candidate_count,
                .median_ns = result.median.nanoseconds,
                .mad_ns = result.mad.nanoseconds,
                .repetitions = result.repetitions,
                .sample_count = result.sample_count,
                .compiled_count = result.compiled_count,
                .rejected_count = result.rejected_count,
                .options = options,
            };
        }

        fn toResult(self: CachedResult, configs: anytype) zml.AutotuneResult(zml.AutotuneConfigType(@TypeOf(configs))) {
            const Config = zml.AutotuneConfigType(@TypeOf(configs));
            const config_slice: []const Config = configs;
            std.debug.assert(self.candidate_index < config_slice.len);
            return .{
                .config = config_slice[self.candidate_index],
                .source = .cache,
                .candidate_index = self.candidate_index,
                .median = .fromNanoseconds(self.median_ns),
                .mad = .fromNanoseconds(self.mad_ns),
                .repetitions = self.repetitions,
                .sample_count = self.sample_count,
                .compiled_count = self.compiled_count,
                .rejected_count = self.rejected_count,
            };
        }
    };

    enabled: bool,
    mutex: std.Io.Mutex = .init,
    arena: std.heap.ArenaAllocator,
    cache: std.StringHashMapUnmanaged(CachedResult) = .empty,

    fn init(allocator: std.mem.Allocator, enabled: bool) AutotuneState {
        return .{
            .enabled = enabled,
            .arena = .init(allocator),
        };
    }

    fn deinit(self: *AutotuneState) void {
        self.cache.deinit(self.arena.allocator());
        self.arena.deinit();
    }

    fn autotune(
        self: *AutotuneState,
        allocator: std.mem.Allocator,
        io: std.Io,
        cache_key: []const u8,
        ctx: anytype,
        configs: anytype,
        comptime compile_fn: anytype,
        comptime measure_fn: anytype,
        comptime deinit_fn: anytype,
        options: zml.AutotuneOptions,
    ) zml.AutotuneError!zml.AutotuneResult(zml.AutotuneConfigType(@TypeOf(configs))) {
        const Config = zml.AutotuneConfigType(@TypeOf(configs));
        const config_slice: []const Config = configs;
        if (config_slice.len == 0) return error.NoConfigurations;

        if (!self.enabled) {
            return .{
                .config = config_slice[0],
                .source = .disabled,
                .candidate_index = 0,
                .median = .zero,
                .mad = .zero,
                .repetitions = 0,
                .sample_count = 0,
                .compiled_count = 0,
                .rejected_count = 0,
            };
        }

        if (autotune_nesting_depth != 0) {
            return .{
                .config = config_slice[0],
                .source = .nested,
                .candidate_index = 0,
                .median = .zero,
                .mad = .zero,
                .repetitions = 0,
                .sample_count = 0,
                .compiled_count = 0,
                .rejected_count = 0,
            };
        }

        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        autotune_nesting_depth += 1;
        defer autotune_nesting_depth -= 1;

        const cached = self.cache.getPtr(cache_key);
        if (cached) |result| {
            if (result.candidate_count == config_slice.len and
                result.candidate_index < config_slice.len and
                std.meta.eql(result.options, options))
            {
                return result.toResult(config_slice);
            }
        }

        const result = try zml.autotune(
            allocator,
            ctx,
            config_slice,
            compile_fn,
            measure_fn,
            deinit_fn,
            options,
        );
        const cache_result: CachedResult = .init(result, config_slice.len, options);

        if (cached) |entry| {
            entry.* = cache_result;
        } else {
            const cache_allocator = self.arena.allocator();
            const owned_key = try cache_allocator.dupe(u8, cache_key);
            errdefer cache_allocator.free(owned_key);
            try self.cache.put(cache_allocator, owned_key, cache_result);
        }
        return result;
    }
};

test "autotune cache arena uses the supplied allocator" {
    const allocator = std.testing.allocator;
    var state: AutotuneState = .init(allocator, true);
    defer state.deinit();

    try std.testing.expect(state.arena.child_allocator.ptr == allocator.ptr);
    try std.testing.expect(state.arena.child_allocator.vtable == allocator.vtable);
}

const PlatformAutotuneTestConfig = struct {
    id: u8,
    marker: u8,
};

const PlatformAutotuneTestProgram = struct {
    duration_ns: i96,
};

const PlatformAutotuneTestContext = struct {
    io: std.Io,
    compile_delay: std.Io.Duration = .zero,
    compile_count: std.atomic.Value(usize) = .init(0),
    measure_count: std.atomic.Value(usize) = .init(0),
    deinit_count: std.atomic.Value(usize) = .init(0),
    active_compiles: std.atomic.Value(usize) = .init(0),
    compile_overlap: std.atomic.Value(bool) = .init(false),

    fn compile(self: *PlatformAutotuneTestContext, config: PlatformAutotuneTestConfig) std.Io.Cancelable!PlatformAutotuneTestProgram {
        _ = self.compile_count.fetchAdd(1, .seq_cst);
        const previous_active = self.active_compiles.fetchAdd(1, .seq_cst);
        defer _ = self.active_compiles.fetchSub(1, .seq_cst);
        if (previous_active != 0) self.compile_overlap.store(true, .seq_cst);
        if (self.compile_delay.nanoseconds != 0) try self.io.sleep(self.compile_delay, .awake);
        return .{ .duration_ns = if (config.id == 0) 20 else 10 };
    }

    fn measure(self: *PlatformAutotuneTestContext, program: *PlatformAutotuneTestProgram, repetitions: usize) error{}!std.Io.Duration {
        _ = self.measure_count.fetchAdd(1, .seq_cst);
        return .fromNanoseconds(program.duration_ns * @as(i96, @intCast(repetitions)));
    }

    fn deinit(self: *PlatformAutotuneTestContext, _: *PlatformAutotuneTestProgram) void {
        _ = self.deinit_count.fetchAdd(1, .seq_cst);
    }
};

fn platformAutotuneTestOptions() zml.AutotuneOptions {
    return .{
        .warmup_rounds = 0,
        .initial_samples = 1,
        .max_samples = 1,
        .target_sample_duration = .fromNanoseconds(1),
        .max_sample_duration = .fromNanoseconds(1_000),
        .max_repetitions = 1,
        .tie_threshold = 0,
    };
}

fn runConcurrentPlatformAutotuneTest(
    state: *AutotuneState,
    ctx: *PlatformAutotuneTestContext,
    ready: *std.atomic.Value(usize),
    key: []const u8,
    configs: *const [1]PlatformAutotuneTestConfig,
    source: *zml.AutotuneSource,
) !void {
    _ = ready.fetchAdd(1, .seq_cst);
    while (ready.load(.seq_cst) != 2) {
        try std.testing.io.sleep(.fromMicroseconds(100), .awake);
    }
    const result = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        key,
        ctx,
        configs,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        platformAutotuneTestOptions(),
    );
    source.* = result.source;
}

test "platform autotune caches an owned key and reconstructs the current config" {
    var parent_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer parent_arena.deinit();
    var state: AutotuneState = .init(parent_arena.allocator(), true);
    defer state.deinit();

    var ctx: PlatformAutotuneTestContext = .{ .io = std.testing.io };
    const first_configs = [_]PlatformAutotuneTestConfig{
        .{ .id = 0, .marker = 10 },
        .{ .id = 1, .marker = 11 },
    };
    var key = "shared".*;
    const first = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        &key,
        &ctx,
        &first_configs,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        platformAutotuneTestOptions(),
    );
    try std.testing.expectEqual(zml.AutotuneSource.tuned, first.source);
    try std.testing.expectEqual(@as(u8, 1), first.config.id);

    key[0] = 'x';
    const second_configs = [_]PlatformAutotuneTestConfig{
        .{ .id = 0, .marker = 20 },
        .{ .id = 1, .marker = 21 },
    };
    const second = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "shared",
        &ctx,
        &second_configs,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        platformAutotuneTestOptions(),
    );
    try std.testing.expectEqual(zml.AutotuneSource.cache, second.source);
    try std.testing.expectEqual(@as(u8, 21), second.config.marker);
    try std.testing.expectEqual(@as(usize, 2), ctx.compile_count.load(.seq_cst));

    const changed_candidates = [_]PlatformAutotuneTestConfig{.{ .id = 0, .marker = 30 }};
    const retuned = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "shared",
        &ctx,
        &changed_candidates,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        platformAutotuneTestOptions(),
    );
    try std.testing.expectEqual(zml.AutotuneSource.tuned, retuned.source);
    try std.testing.expectEqual(@as(usize, 3), ctx.compile_count.load(.seq_cst));

    var changed_options = platformAutotuneTestOptions();
    changed_options.shuffle_seed = 1;
    const retuned_for_options = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "shared",
        &ctx,
        &changed_candidates,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        changed_options,
    );
    try std.testing.expectEqual(zml.AutotuneSource.tuned, retuned_for_options.source);
    try std.testing.expectEqual(@as(usize, 4), ctx.compile_count.load(.seq_cst));

    const cached_for_options = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "shared",
        &ctx,
        &changed_candidates,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        changed_options,
    );
    try std.testing.expectEqual(zml.AutotuneSource.cache, cached_for_options.source);
    try std.testing.expectEqual(@as(usize, 4), ctx.compile_count.load(.seq_cst));
}

test "disabled platform autotune selects the first config without callbacks" {
    var parent_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer parent_arena.deinit();
    var state: AutotuneState = .init(parent_arena.allocator(), false);
    defer state.deinit();

    var ctx: PlatformAutotuneTestContext = .{ .io = std.testing.io };
    const configs = [_]PlatformAutotuneTestConfig{
        .{ .id = 0, .marker = 10 },
        .{ .id = 1, .marker = 11 },
    };
    const result = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "disabled",
        &ctx,
        &configs,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        .{ .initial_samples = 0 },
    );
    try std.testing.expectEqual(zml.AutotuneSource.disabled, result.source);
    try std.testing.expectEqual(configs[0], result.config);
    try std.testing.expectEqual(@as(usize, 0), ctx.compile_count.load(.seq_cst));
    try std.testing.expectEqual(@as(usize, 0), ctx.measure_count.load(.seq_cst));
    try std.testing.expectEqual(@as(usize, 0), ctx.deinit_count.load(.seq_cst));

    const empty: [0]PlatformAutotuneTestConfig = .{};
    try std.testing.expectError(error.NoConfigurations, state.autotune(
        std.testing.allocator,
        std.testing.io,
        "empty",
        &ctx,
        &empty,
        PlatformAutotuneTestContext.compile,
        PlatformAutotuneTestContext.measure,
        PlatformAutotuneTestContext.deinit,
        .{},
    ));
}

test "recursive platform autotune selects the first config without nested callbacks" {
    const Program = struct {
        duration_ns: i96,
    };
    const Context = struct {
        state: *AutotuneState,
        outer_compile_count: usize = 0,
        inner_compile_count: usize = 0,
        inner_measure_count: usize = 0,
        inner_deinit_count: usize = 0,
        nested_source: zml.AutotuneSource = .disabled,
        nested_config: u8 = 0,
        nested_repetitions: usize = undefined,

        fn outerCompile(self: *@This(), _: u8) zml.AutotuneError!Program {
            self.outer_compile_count += 1;
            const inner_configs = [_]u8{ 7, 8 };
            const result = try self.state.autotune(
                std.testing.allocator,
                std.testing.io,
                "inner",
                self,
                &inner_configs,
                innerCompile,
                innerMeasure,
                innerDeinit,
                platformAutotuneTestOptions(),
            );
            self.nested_source = result.source;
            self.nested_config = result.config;
            self.nested_repetitions = result.repetitions;
            return .{ .duration_ns = 10 };
        }

        fn innerCompile(self: *@This(), _: u8) error{}!Program {
            self.inner_compile_count += 1;
            return .{ .duration_ns = 1 };
        }

        fn innerMeasure(self: *@This(), program: *Program, repetitions: usize) error{}!std.Io.Duration {
            self.inner_measure_count += 1;
            return .fromNanoseconds(program.duration_ns * @as(i96, @intCast(repetitions)));
        }

        fn innerDeinit(self: *@This(), _: *Program) void {
            self.inner_deinit_count += 1;
        }

        fn outerMeasure(_: *@This(), program: *Program, repetitions: usize) error{}!std.Io.Duration {
            return .fromNanoseconds(program.duration_ns * @as(i96, @intCast(repetitions)));
        }

        fn outerDeinit(_: *@This(), _: *Program) void {}
    };

    var parent_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer parent_arena.deinit();
    var state: AutotuneState = .init(parent_arena.allocator(), true);
    defer state.deinit();

    var ctx: Context = .{ .state = &state };
    const outer_configs = [_]u8{1};
    const result = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "outer",
        &ctx,
        &outer_configs,
        Context.outerCompile,
        Context.outerMeasure,
        Context.outerDeinit,
        platformAutotuneTestOptions(),
    );
    try std.testing.expectEqual(zml.AutotuneSource.tuned, result.source);
    try std.testing.expectEqual(zml.AutotuneSource.nested, ctx.nested_source);
    try std.testing.expectEqual(@as(u8, 7), ctx.nested_config);
    try std.testing.expectEqual(@as(usize, 0), ctx.nested_repetitions);
    try std.testing.expectEqual(@as(usize, 0), ctx.inner_compile_count);
    try std.testing.expectEqual(@as(usize, 0), ctx.inner_measure_count);
    try std.testing.expectEqual(@as(usize, 0), ctx.inner_deinit_count);

    const cached = try state.autotune(
        std.testing.allocator,
        std.testing.io,
        "outer",
        &ctx,
        &outer_configs,
        Context.outerCompile,
        Context.outerMeasure,
        Context.outerDeinit,
        platformAutotuneTestOptions(),
    );
    try std.testing.expectEqual(zml.AutotuneSource.cache, cached.source);
    try std.testing.expectEqual(@as(usize, 1), ctx.outer_compile_count);
}

test "platform autotune serializes concurrent cache misses" {
    var parent_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer parent_arena.deinit();
    var state: AutotuneState = .init(parent_arena.allocator(), true);
    defer state.deinit();

    var ctx: PlatformAutotuneTestContext = .{
        .io = std.testing.io,
        .compile_delay = .fromMilliseconds(5),
    };
    var ready: std.atomic.Value(usize) = .init(0);
    const configs = [_]PlatformAutotuneTestConfig{.{ .id = 0, .marker = 0 }};
    var first_source: zml.AutotuneSource = .disabled;
    var second_source: zml.AutotuneSource = .disabled;
    var first = try std.Io.concurrent(std.testing.io, runConcurrentPlatformAutotuneTest, .{ &state, &ctx, &ready, "first", &configs, &first_source });
    defer _ = first.cancel(std.testing.io) catch {};
    var second = try std.Io.concurrent(std.testing.io, runConcurrentPlatformAutotuneTest, .{ &state, &ctx, &ready, "second", &configs, &second_source });
    defer _ = second.cancel(std.testing.io) catch {};
    try first.await(std.testing.io);
    try second.await(std.testing.io);

    try std.testing.expectEqual(zml.AutotuneSource.tuned, first_source);
    try std.testing.expectEqual(zml.AutotuneSource.tuned, second_source);
    try std.testing.expectEqual(@as(usize, 2), ctx.compile_count.load(.seq_cst));
    try std.testing.expect(!ctx.compile_overlap.load(.seq_cst));
}

test "concurrent platform autotune requests share a same-key result" {
    var parent_arena: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer parent_arena.deinit();
    var state: AutotuneState = .init(parent_arena.allocator(), true);
    defer state.deinit();

    var ctx: PlatformAutotuneTestContext = .{
        .io = std.testing.io,
        .compile_delay = .fromMilliseconds(5),
    };
    var ready: std.atomic.Value(usize) = .init(0);
    const configs = [_]PlatformAutotuneTestConfig{.{ .id = 0, .marker = 0 }};
    var first_source: zml.AutotuneSource = .disabled;
    var second_source: zml.AutotuneSource = .disabled;

    var first = try std.Io.concurrent(std.testing.io, runConcurrentPlatformAutotuneTest, .{ &state, &ctx, &ready, "same", &configs, &first_source });
    defer _ = first.cancel(std.testing.io) catch {};
    var second = try std.Io.concurrent(std.testing.io, runConcurrentPlatformAutotuneTest, .{ &state, &ctx, &ready, "same", &configs, &second_source });
    defer _ = second.cancel(std.testing.io) catch {};
    try first.await(std.testing.io);
    try second.await(std.testing.io);

    try std.testing.expect((first_source == .tuned and second_source == .cache) or
        (first_source == .cache and second_source == .tuned));
    try std.testing.expectEqual(@as(usize, 1), ctx.compile_count.load(.seq_cst));
    try std.testing.expect(!ctx.compile_overlap.load(.seq_cst));
}

pub const Platform = struct {
    arena: std.heap.ArenaAllocator,
    autotune_state: *AutotuneState = undefined,
    target: Target,
    pjrt_api: *const pjrt.Api,
    pjrt_client: *pjrt.Client,
    execution_timer_type_id: ?pjrt.ffi.TypeId = null,
    state: State,
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
                .autotune_state = undefined,
                .target = target,
                .pjrt_api = api,
                .pjrt_client = pjrt_client,
                .execution_timer_type_id = null,
                .state = State.init(target),
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
        const autotune_state = try arena.create(AutotuneState);
        // Cache allocations can happen while concurrent compiles use the main
        // platform arena. Back the nested cache arena with the original
        // allocator, whose lifetime already has to cover the Platform, and
        // deinitialize it before the platform arena.
        autotune_state.* = .init(allocator, options.autotune);
        errdefer autotune_state.deinit();
        platform.autotune_state = autotune_state;
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
                if (zml.moe.cutlass_flashinfer.load(arena, io, platform)) {
                    zml.moe.cutlass_flashinfer.register(platform) catch |err| {
                        log.warn(
                            "Failed to register FlashInfer CUTLASS MoE custom calls: {}",
                            .{err},
                        );
                    };
                } else |err| {
                    log.warn("Failed to load FlashInfer CUTLASS MoE: {}", .{err});
                }
            },
            else => {},
        }

        switch (target) {
            .cuda, .rocm => registerExecutionTimer(platform) catch |err| {
                log.warn("Failed to register native execution timer: {}", .{err});
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
        self.autotune_state.deinit();
        if (comptime platforms.isEnabled(.cuda)) {
            self.state.deinit();
        }
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

    /// Serializes autotuning on this platform and reuses successful results for
    /// identical keys. `cache_key` must identify both the workload and the
    /// ordered candidate set; tuning options are compared internally. When
    /// autotuning is disabled, option validation is intentionally bypassed and
    /// candidate zero is returned. Same-thread recursive calls from tuning
    /// callbacks select candidate zero without tuning. Callbacks must not offload a
    /// recursive autotune call to another thread and wait for it: the platform
    /// lock is intentionally non-reentrant across threads.
    pub fn autotune(
        self: *const Platform,
        allocator: std.mem.Allocator,
        io: std.Io,
        cache_key: []const u8,
        ctx: anytype,
        configs: anytype,
        comptime compile_fn: anytype,
        comptime measure_fn: anytype,
        comptime deinit_fn: anytype,
        options: zml.AutotuneOptions,
    ) zml.AutotuneError!zml.AutotuneResult(zml.AutotuneConfigType(@TypeOf(configs))) {
        return self.autotune_state.autotune(
            allocator,
            io,
            cache_key,
            ctx,
            configs,
            compile_fn,
            measure_fn,
            deinit_fn,
            options,
        );
    }

    pub fn autotuneEnabled(self: *const Platform) bool {
        return self.autotune_state.enabled;
    }

    pub fn executionTimerAvailable(self: *const Platform) bool {
        return switch (self.target) {
            .cuda, .rocm => executionTimerAvailableFor(
                self.target,
                self.execution_timer_type_id != null,
                platforms.eventProvider(self.target) != null,
                self.pjrt_api.ffi() != null,
            ),
            else => false,
        };
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
    autotune: bool = true,
    cpu: Cpu = .{ .device_count = 4 },

    // bump memory fraction from XLA defaults of 75% to 90%.
    // Even on a 8GB GPU it should leave enough space for the platform driver/runtime.
    // https://github.com/openxla/xla/blob/3e87afa11a865cf91137522492918ad18bfe5b7c/xla/pjrt/plugin/xla_gpu/xla_gpu_allocator_config.h#L25-L60
    xla_gpu: XlaGpu = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.90 } } },
    tpu: struct {} = .{},
    neuron: struct {} = .{},
    oneapi: struct {} = .{},
    metal: struct {} = .{},

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
            .cuda, .rocm, .oneapi, .metal => self.xla_gpu.writeNamedValues(target, &values),
            inline else => |t| {
                stdx.debug.assertComptime(@hasField(CreateOptions, @tagName(t)), "zml.platform.CreateOptions doesn't list target {s}", .{@tagName(t)});
                const options = @field(self, @tagName(t));
                stdx.debug.assertComptime(@sizeOf(@TypeOf(options)) == 0, "zml.platform.CreateOptions.{s} is discarded", .{@tagName(t)});
            },
        }
        return values.items;
    }
};

test "CreateOptions enables autotune without forwarding it to PJRT" {
    try std.testing.expect((CreateOptions{}).autotune);

    var enabled_buffer: [16]pjrt.NamedValue = undefined;
    var disabled_buffer: [16]pjrt.NamedValue = undefined;
    const enabled_values = (CreateOptions{}).toNamedValues(.cpu, &enabled_buffer);
    const disabled_values = (CreateOptions{ .autotune = false }).toNamedValues(.cpu, &disabled_buffer);
    try std.testing.expectEqual(enabled_values.len, disabled_values.len);
    for (enabled_values, disabled_values) |enabled, disabled| {
        try std.testing.expectEqualStrings(enabled.name(), disabled.name());
        try std.testing.expectEqual(enabled.kind(), disabled.kind());
        try std.testing.expect(std.meta.eql(enabled.value(), disabled.value()));
    }
}

const ExecutionTimerState = struct {
    const Phase = enum(u8) {
        idle,
        resetting,
        armed,
        executing,
        measured,
        reading,
    };

    const SlotStatus = enum {
        empty,
        started,
        reported,
        failed,
    };

    const Slot = struct {
        device_ordinal: i32,
        generation: u64 = 0,
        status: SlotStatus = .empty,
        start_event: ?platforms.gpu_event.Event = null,
        duration_ns: i96 = 0,
    };

    provider: *const platforms.gpu_event.Provider,
    slots: []Slot,
    generation: u64 = 0,
    phase: std.atomic.Value(u8) = .init(@intFromEnum(Phase.idle)),

    fn transition(self: *ExecutionTimerState, from: Phase, to: Phase) bool {
        return self.phase.cmpxchgStrong(
            @intFromEnum(from),
            @intFromEnum(to),
            .acq_rel,
            .acquire,
        ) == null;
    }

    fn isPhase(self: *const ExecutionTimerState, phase: Phase) bool {
        return self.phase.load(.acquire) == @intFromEnum(phase);
    }

    fn beginExecution(opaque_state: *anyopaque) bool {
        const self: *ExecutionTimerState = @ptrCast(@alignCast(opaque_state));
        return self.transition(.armed, .executing);
    }

    fn endExecution(opaque_state: *anyopaque, succeeded: bool) void {
        const self: *ExecutionTimerState = @ptrCast(@alignCast(opaque_state));
        if (succeeded) {
            std.debug.assert(self.isPhase(.executing));
            self.phase.store(@intFromEnum(Phase.measured), .release);
        } else {
            self.cleanupSlots();
            self.phase.store(@intFromEnum(Phase.idle), .release);
        }
    }

    fn slotForOrdinal(self: *ExecutionTimerState, ordinal: i32) ?*Slot {
        for (self.slots) |*slot| {
            if (slot.device_ordinal == ordinal) return slot;
        }
        return null;
    }

    fn destroyEvent(self: *ExecutionTimerState, event: platforms.gpu_event.Event) void {
        switch (self.provider.destroy(event)) {
            .ok => {},
            .err => |failure| log.warn(
                "Failed to destroy {s} timing event ({s}, status {}): {s}",
                .{ @tagName(self.provider.backend), @tagName(failure.operation), failure.status, self.provider.errorString(failure) },
            ),
        }
    }

    fn cleanupSlots(self: *ExecutionTimerState) void {
        for (self.slots) |*slot| {
            if (slot.start_event) |event| self.destroyEvent(event);
            slot.start_event = null;
            slot.status = .empty;
        }
    }

    fn cleanup(self: *ExecutionTimerState) void {
        self.cleanupSlots();
        self.phase.store(@intFromEnum(Phase.idle), .release);
    }
};

/// Measures the device critical path of an entry function compiled with
/// `CompilationOptions.execution_timing = .device`.
///
/// An instrumented executable is serial-only: call `reset`, execute it once
/// with waiting enabled, then call `read`. `measureCall` implements that cycle
/// for a reusable `Exe.Arguments`/`Exe.Results` pair and sums multiple runs.
/// Custom kernels that launch work on auxiliary streams must join that work
/// back to the FFI stream before producing their outputs, otherwise the stop
/// marker cannot include it in the measured interval.
pub const ExecutionTimer = struct {
    pub const Error = std.mem.Allocator.Error || pjrt.ApiError || error{
        UnsupportedPlatform,
        TimerUnavailable,
        InvalidState,
        ExecutionInProgress,
        MissingDeviceMeasurement,
        InvalidRepetitionCount,
        DurationOverflow,
    };

    exe: *Exe,
    state: *ExecutionTimerState,

    pub fn attach(exe: *Exe) Error!ExecutionTimer {
        if (exe.platform.target != .cuda and exe.platform.target != .rocm) {
            return error.UnsupportedPlatform;
        }
        const type_id = exe.platform.execution_timer_type_id orelse return error.TimerUnavailable;
        _ = platforms.eventProvider(exe.platform.target) orelse return error.TimerUnavailable;

        if (exe.execution_timer_state) |opaque_state| {
            return .{
                .exe = exe,
                .state = @ptrCast(@alignCast(opaque_state)),
            };
        }

        const ffi = exe.platform.pjrt_api.ffi() orelse return error.TimerUnavailable;
        var owns_context = false;
        const context = exe.context orelse context: {
            owns_context = true;
            break :context try exe.platform.pjrt_api.createExecuteContext();
        };
        errdefer if (owns_context) context.deinit(exe.platform.pjrt_api);

        const assigned_devices = exe.exe.addressableDevices(exe.platform.pjrt_api);
        if (exe.num_devices == 0 or assigned_devices.len != exe.num_devices) {
            return error.InvalidState;
        }

        const allocator = exe.arena.allocator();
        const state = try allocator.create(ExecutionTimerState);
        const slots = try allocator.alloc(ExecutionTimerState.Slot, exe.num_devices);
        for (slots, assigned_devices) |*slot, pjrt_device| {
            const device = exe.platform.deviceFromPjrt(pjrt_device);
            slot.* = .{ .device_ordinal = device.localHardwareId() };
        }
        state.* = .{
            .provider = platforms.eventProvider(exe.platform.target).?,
            .slots = slots,
        };

        try ffi.addUserData(exe.platform.pjrt_api, context, .{
            .type_id = type_id.type_id,
            .user_data = state,
        });

        exe.context = context;
        exe.execution_timer_state = state;
        exe.execution_timer_deinit = destroyExecutionTimerState;
        exe.execution_timer_begin = ExecutionTimerState.beginExecution;
        exe.execution_timer_end = ExecutionTimerState.endExecution;
        return .{ .exe = exe, .state = state };
    }

    /// Advances the measurement generation and arms exactly one execution.
    pub fn reset(self: *ExecutionTimer) Error!void {
        if (!self.state.transition(.idle, .resetting)) return error.ExecutionInProgress;
        self.state.generation +%= 1;
        for (self.state.slots) |*slot| {
            if (slot.start_event) |event| self.state.destroyEvent(event);
            slot.* = .{
                .device_ordinal = slot.device_ordinal,
                .generation = self.state.generation,
            };
        }
        self.state.phase.store(@intFromEnum(ExecutionTimerState.Phase.armed), .release);
    }

    /// Returns the slowest device duration for the current generation.
    pub fn read(self: *ExecutionTimer) Error!std.Io.Duration {
        if (!self.state.transition(.measured, .reading)) return error.InvalidState;
        defer self.state.phase.store(@intFromEnum(ExecutionTimerState.Phase.idle), .release);
        var maximum_ns: i96 = 0;
        for (self.state.slots) |slot| {
            if (slot.generation != self.state.generation or slot.status != .reported) {
                return error.MissingDeviceMeasurement;
            }
            maximum_ns = @max(maximum_ns, slot.duration_ns);
        }
        return .fromNanoseconds(maximum_ns);
    }

    /// Native timing adapter for preallocated executable storage. PJRT ready
    /// events are awaited so execution and marker failures reject the sample.
    /// For sub-resolution, non-capturable programs, benchmark a compiled body
    /// that batches the kernel internally rather than adding a delay kernel.
    pub fn measureCall(
        self: *ExecutionTimer,
        io: std.Io,
        arguments: Exe.Arguments,
        results: *Exe.Results,
        repetitions: usize,
    ) Error!std.Io.Duration {
        if (repetitions == 0) return error.InvalidRepetitionCount;
        var total_ns: i96 = 0;
        for (0..repetitions) |_| {
            try self.reset();
            self.exe.tryCallOpts(io, arguments, results, .{
                .wait = true,
                .allow_input_donation = false,
            }) catch |err| {
                results.releaseBuffers();
                return err;
            };
            const duration = self.read() catch |err| {
                results.releaseBuffers();
                return err;
            };
            results.releaseBuffers();
            if (duration.nanoseconds > std.math.maxInt(i96) - total_ns) return error.DurationOverflow;
            total_ns += duration.nanoseconds;
        }
        return .fromNanoseconds(total_ns);
    }
};

fn destroyExecutionTimerState(opaque_state: *anyopaque) void {
    const state: *ExecutionTimerState = @ptrCast(@alignCast(opaque_state));
    state.cleanup();
}

fn executionTimerTypeId(comptime target: Target) ?pjrt.ffi.TypeId {
    return execution_timer_registrations.get(target).type_id;
}

fn executionTimerFfiError(call_frame: *pjrt.ffi.CallFrame, code: pjrt.ffi.ErrorCode, message: []const u8) *pjrt.ffi.Error {
    return pjrt.ffi.Error.create(call_frame.api, code, message);
}

fn executionTimerProviderError(
    call_frame: *pjrt.ffi.CallFrame,
    provider: *const platforms.gpu_event.Provider,
    failure: platforms.gpu_event.Failure,
) *pjrt.ffi.Error {
    log.err(
        "{s} execution timer failed during {s} (status {}): {s}",
        .{ @tagName(provider.backend), @tagName(failure.operation), failure.status, provider.errorString(failure) },
    );
    return executionTimerFfiError(call_frame, .internal, provider.errorString(failure));
}

fn ExecutionTimerHandler(comptime target: Target) type {
    const expected_backend: platforms.gpu_event.Backend = switch (target) {
        .cuda => .cuda,
        .rocm => .rocm,
        else => @compileError("native execution timing only supports CUDA and ROCm"),
    };

    return struct {
        fn state(call_frame: *pjrt.ffi.CallFrame) !*ExecutionTimerState {
            if (@intFromPtr(call_frame.ctx) == 0) return error.NotFound;
            const type_id = executionTimerTypeId(target) orelse return error.NotFound;
            const opaque_state = try call_frame.ctx.getContext(type_id, call_frame.api);
            const timer_state: *ExecutionTimerState = @ptrCast(@alignCast(opaque_state));
            if (timer_state.provider.backend != expected_backend) return error.InvalidArgument;
            return timer_state;
        }

        fn stateAndSlot(call_frame: *pjrt.ffi.CallFrame) !struct { *ExecutionTimerState, *ExecutionTimerState.Slot } {
            const timer_state = try state(call_frame);
            if (!timer_state.isPhase(.executing)) return error.FailedPrecondition;
            const ordinal = try call_frame.ctx.getDeviceOrdinal(call_frame.api);
            const slot = timer_state.slotForOrdinal(ordinal) orelse return error.OutOfRange;
            if (slot.generation != timer_state.generation) return error.FailedPrecondition;
            return .{ timer_state, slot };
        }

        pub fn start(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
            if (call_frame.registeringHook()) return null;
            const timer_state, const slot = stateAndSlot(call_frame) catch |err| {
                return executionTimerFfiError(call_frame, .failed_precondition, @errorName(err));
            };
            if (slot.status != .empty) {
                return executionTimerFfiError(call_frame, .failed_precondition, "concurrent or repeated execution of an instrumented executable");
            }
            if (call_frame.results.buffers().len != 1) {
                return executionTimerFfiError(call_frame, .invalid_argument, "autotune start expects one marker result");
            }
            const stream = call_frame.tryStream() catch |err| {
                return executionTimerFfiError(call_frame, .failed_precondition, @errorName(err));
            };

            const start_event = switch (timer_state.provider.create(stream)) {
                .ok => |event| event,
                .err => |failure| return executionTimerProviderError(call_frame, timer_state.provider, failure),
            };
            var keep_event = false;
            defer if (!keep_event) timer_state.destroyEvent(start_event);

            const marker: *anyopaque = @ptrCast(call_frame.results.buffers()[0].data);
            switch (timer_state.provider.markerInitAsync(marker, stream)) {
                .ok => {},
                .err => |failure| {
                    slot.status = .failed;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            }
            switch (timer_state.provider.record(start_event, stream)) {
                .ok => {},
                .err => |failure| {
                    slot.status = .failed;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            }

            slot.start_event = start_event;
            slot.status = .started;
            keep_event = true;
            return null;
        }

        pub fn stop(call_frame: *pjrt.ffi.CallFrame) callconv(.c) ?*pjrt.ffi.Error {
            if (call_frame.registeringHook()) return null;
            const timer_state, const slot = stateAndSlot(call_frame) catch |err| {
                return executionTimerFfiError(call_frame, .failed_precondition, @errorName(err));
            };
            if (slot.status != .started or slot.start_event == null) {
                return executionTimerFfiError(call_frame, .failed_precondition, "autotune stop observed without a matching start");
            }
            const start_event = slot.start_event.?;
            const stream = call_frame.tryStream() catch |err| {
                slot.status = .failed;
                timer_state.destroyEvent(start_event);
                slot.start_event = null;
                return executionTimerFfiError(call_frame, .failed_precondition, @errorName(err));
            };
            if (call_frame.results.buffers().len != 1) {
                slot.status = .failed;
                timer_state.destroyEvent(start_event);
                slot.start_event = null;
                return executionTimerFfiError(call_frame, .invalid_argument, "autotune stop expects one marker result");
            }

            const stop_event = switch (timer_state.provider.create(stream)) {
                .ok => |event| event,
                .err => |failure| {
                    slot.status = .failed;
                    timer_state.destroyEvent(start_event);
                    slot.start_event = null;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            };
            var stop_event_alive = true;
            var start_event_alive = true;
            defer {
                if (stop_event_alive) timer_state.destroyEvent(stop_event);
                if (start_event_alive) timer_state.destroyEvent(start_event);
                slot.start_event = null;
            }

            switch (timer_state.provider.record(stop_event, stream)) {
                .ok => {},
                .err => |failure| {
                    slot.status = .failed;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            }
            switch (timer_state.provider.sync(stop_event)) {
                .ok => {},
                .err => |failure| {
                    slot.status = .failed;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            }
            const milliseconds = switch (timer_state.provider.elapsedMs(start_event, stop_event)) {
                .ok => |duration| duration,
                .err => |failure| {
                    slot.status = .failed;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            };
            if (!std.math.isFinite(milliseconds) or milliseconds < 0) {
                slot.status = .failed;
                return executionTimerFfiError(call_frame, .data_loss, "GPU event provider returned an invalid duration");
            }

            var destroy_failure: ?platforms.gpu_event.Failure = null;
            switch (timer_state.provider.destroy(stop_event)) {
                .ok => {},
                .err => |failure| destroy_failure = failure,
            }
            stop_event_alive = false;
            switch (timer_state.provider.destroy(start_event)) {
                .ok => {},
                .err => |failure| if (destroy_failure == null) {
                    destroy_failure = failure;
                },
            }
            start_event_alive = false;
            slot.start_event = null;
            if (destroy_failure) |failure| {
                slot.status = .failed;
                return executionTimerProviderError(call_frame, timer_state.provider, failure);
            }

            // The stop result exists only to carry a non-replicated sharding,
            // ensuring one side-effecting marker call is emitted per device.
            // Initialize it after the stop event so this bookkeeping is not
            // included in the measured interval.
            const marker: *anyopaque = @ptrCast(call_frame.results.buffers()[0].data);
            switch (timer_state.provider.markerInitAsync(marker, stream)) {
                .ok => {},
                .err => |failure| {
                    slot.status = .failed;
                    return executionTimerProviderError(call_frame, timer_state.provider, failure);
                },
            }

            slot.duration_ns = @intFromFloat(@round(@as(f64, milliseconds) * std.time.ns_per_ms));
            slot.status = .reported;
            return null;
        }
    };
}

fn registerExecutionTimer(platform: *Platform) !void {
    _ = platforms.eventProvider(platform.target) orelse return error.Unavailable;
    const ffi = platform.pjrt_api.ffi() orelse return error.Unavailable;
    const registration = execution_timer_registrations.getPtr(platform.target);
    const type_id = registration.type_id orelse type_id: {
        const type_info = (pjrt.Ffi.TypeInfo{}).toCStruct();
        const registered_type_id = try ffi.registerTypeId(
            platform.pjrt_api,
            switch (platform.target) {
                .cuda => "zml.ExecutionTimer.cuda",
                .rocm => "zml.ExecutionTimer.rocm",
                else => unreachable,
            },
            &type_info,
        );
        // XLA's type registry is process-global and registering a name twice
        // fails. Preserve this successful step even if a handler fails below,
        // so a later Platform instance can retry only the missing handlers.
        registration.type_id = registered_type_id;
        break :type_id registered_type_id;
    };

    switch (platform.target) {
        .cuda => {
            if (!registration.start_handler_registered) {
                try platform.registerFfi(.{
                    .name = "zml$autotune_start",
                    .handler = ExecutionTimerHandler(.cuda).start,
                    .traits = .{ .command_buffer_compatible = false },
                });
                registration.start_handler_registered = true;
            }
            if (!registration.stop_handler_registered) {
                try platform.registerFfi(.{
                    .name = "zml$autotune_stop",
                    .handler = ExecutionTimerHandler(.cuda).stop,
                    .traits = .{ .command_buffer_compatible = false },
                });
                registration.stop_handler_registered = true;
            }
        },
        .rocm => {
            if (!registration.start_handler_registered) {
                try platform.registerFfi(.{
                    .name = "zml$autotune_start",
                    .handler = ExecutionTimerHandler(.rocm).start,
                    .traits = .{ .command_buffer_compatible = false },
                });
                registration.start_handler_registered = true;
            }
            if (!registration.stop_handler_registered) {
                try platform.registerFfi(.{
                    .name = "zml$autotune_stop",
                    .handler = ExecutionTimerHandler(.rocm).stop,
                    .traits = .{ .command_buffer_compatible = false },
                });
                registration.stop_handler_registered = true;
            }
        },
        else => unreachable,
    }

    platform.execution_timer_type_id = type_id;
}

test "ExecutionTimer returns the distributed critical path" {
    var slots = [_]ExecutionTimerState.Slot{
        .{ .device_ordinal = 2 },
        .{ .device_ordinal = 7 },
    };
    var state: ExecutionTimerState = .{
        .provider = undefined,
        .slots = &slots,
    };
    var timer: ExecutionTimer = .{ .exe = undefined, .state = &state };

    try timer.reset();
    try std.testing.expect(ExecutionTimerState.beginExecution(&state));
    slots[0].status = .reported;
    slots[0].duration_ns = 125;
    slots[1].status = .reported;
    slots[1].duration_ns = 400;
    ExecutionTimerState.endExecution(&state, true);
    try std.testing.expectEqual(@as(i96, 400), (try timer.read()).nanoseconds);

    try timer.reset();
    try std.testing.expect(ExecutionTimerState.beginExecution(&state));
    slots[0].status = .reported;
    slots[0].duration_ns = 10;
    ExecutionTimerState.endExecution(&state, true);
    try std.testing.expectError(error.MissingDeviceMeasurement, timer.read());
}

test "ExecutionTimer atomically rejects concurrent execution" {
    var slots = [_]ExecutionTimerState.Slot{.{ .device_ordinal = 0 }};
    var state: ExecutionTimerState = .{
        .provider = undefined,
        .slots = &slots,
    };
    var timer: ExecutionTimer = .{ .exe = undefined, .state = &state };
    try timer.reset();
    try std.testing.expect(ExecutionTimerState.beginExecution(&state));
    try std.testing.expect(!ExecutionTimerState.beginExecution(&state));
    try std.testing.expectError(error.ExecutionInProgress, timer.reset());
    ExecutionTimerState.endExecution(&state, false);
    try timer.reset();
}

test "instrumented GPU executable requires an attached timer" {
    const platform = zml.testing.env();
    if (platform.target != .cuda and platform.target != .rocm) return error.SkipZigTest;

    const Forward = struct {
        fn forward(input: zml.Tensor) zml.Tensor {
            return input.addConstant(1).reuseBuffer(input);
        }
    };
    const input_tensor: zml.Tensor = .init(.{1 << 18}, .f32);
    var exe = try zml.module.compile(
        std.testing.allocator,
        std.testing.io,
        Forward.forward,
        .{input_tensor},
        platform,
        // The autotune integration below exercises default Shardy; use GSPMD
        // here while also validating repeated execution of a donated input.
        .{ .execution_timing = .device, .partitioner = .gspmd },
    );
    defer exe.deinit();

    var input = try zml.Buffer.uninitialized(std.testing.io, platform, input_tensor.shape(), .replicated, .{});
    defer input.deinit();
    var arguments = try exe.args(std.testing.allocator);
    defer arguments.deinit(std.testing.allocator);
    arguments.set(.{input});
    var results = try exe.results(std.testing.allocator);
    defer results.deinit(std.testing.allocator);

    var rejected = false;
    exe.tryCallOpts(std.testing.io, arguments, &results, .{ .wait = true }) catch {
        rejected = true;
    };
    results.releaseBuffers();
    try std.testing.expect(rejected);

    var timer = try ExecutionTimer.attach(&exe);
    const duration = try timer.measureCall(std.testing.io, arguments, &results, 2);
    try std.testing.expect(duration.nanoseconds > 0);
}

test "GPU autotune selects a known faster instrumented program" {
    const platform = zml.testing.env();
    if (platform.target != .cuda and platform.target != .rocm) return error.SkipZigTest;

    const Config = enum { slow, fast };
    const Fast = struct {
        fn forward(input: zml.Tensor) zml.Tensor {
            return input.addConstant(1);
        }
    };
    const Slow = struct {
        fn forward(input: zml.Tensor) zml.Tensor {
            var value = input;
            inline for (0..16) |_| value = value.tanh();
            return value;
        }
    };
    const Program = struct {
        exe: Exe,
        arguments: Exe.Arguments,
        results: Exe.Results,
        timer: ?ExecutionTimer = null,
    };
    const Context = struct {
        platform: *const Platform,
        input_tensor: zml.Tensor,
        input: zml.Buffer,

        fn compile(self: *@This(), config: Config) !Program {
            var exe = switch (config) {
                .slow => try zml.module.compile(
                    std.testing.allocator,
                    std.testing.io,
                    Slow.forward,
                    .{self.input_tensor},
                    self.platform,
                    .{ .execution_timing = .device },
                ),
                .fast => try zml.module.compile(
                    std.testing.allocator,
                    std.testing.io,
                    Fast.forward,
                    .{self.input_tensor},
                    self.platform,
                    .{ .execution_timing = .device },
                ),
            };
            errdefer exe.deinit();
            var arguments = try exe.args(std.testing.allocator);
            errdefer arguments.deinit(std.testing.allocator);
            arguments.set(.{self.input});
            const results = try exe.results(std.testing.allocator);
            return .{ .exe = exe, .arguments = arguments, .results = results };
        }

        fn measure(_: *@This(), program: *Program, repetitions: usize) !std.Io.Duration {
            const timer = if (program.timer) |*timer| timer else timer: {
                program.timer = try ExecutionTimer.attach(&program.exe);
                break :timer &program.timer.?;
            };
            return timer.measureCall(std.testing.io, program.arguments, &program.results, repetitions);
        }

        fn deinit(_: *@This(), program: *Program) void {
            program.results.deinit(std.testing.allocator);
            program.arguments.deinit(std.testing.allocator);
            program.exe.deinit();
        }
    };

    const input_tensor: zml.Tensor = .init(.{1 << 20}, .f32);
    var ctx: Context = .{
        .platform = platform,
        .input_tensor = input_tensor,
        .input = try .uninitialized(std.testing.io, platform, input_tensor.shape(), .replicated, .{}),
    };
    defer ctx.input.deinit();

    const result = try zml.autotune(
        std.testing.allocator,
        &ctx,
        &[_]Config{ .slow, .fast },
        Context.compile,
        Context.measure,
        Context.deinit,
        .{
            .warmup_rounds = 2,
            .initial_samples = 5,
            .max_samples = 9,
            .target_sample_duration = .fromMilliseconds(2),
            .max_sample_duration = .fromMilliseconds(50),
        },
    );
    try std.testing.expectEqual(Config.fast, result.config);
    try std.testing.expect(result.median.nanoseconds > 0);
}

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
