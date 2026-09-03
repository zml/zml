const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const mem = @import("../mem.zig");
const pjrtx = @import("../pjrtx.zig");
const platform_mod = @import("../platform.zig");
const Memory = platform_mod.Memory;
const Platform = platform_mod.Platform;
const limits = @import("limits.zig");

const log = std.log.scoped(.@"zml/io");

const max_load_dma_parallelism = limits.max_dma_parallelism;
const max_load_read_request_size = limits.max_read_request_size;
const maximumCoalescedJobBlocks = limits.maximumCoalescedJobBlocks;

pub const default_dma_benchmark_block_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
};

const dma_benchmark_repeats = 3;

const DmaBenchmarkPhase = enum {
    block,
    block_confirmation,
};

const DmaBenchmarkSample = struct {
    phase: DmaBenchmarkPhase,
    device_index: usize,
    block_size: usize,
    parallelism: usize,
    repeat: usize = 0,
    bytes: u64,
    transfers: u64,
    elapsed_ns: u64,
    total_latency_ns: u64,

    pub fn bytesPerSecond(self: DmaBenchmarkSample) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn averageLatencyNs(self: DmaBenchmarkSample) f64 {
        if (self.transfers == 0) return 0;
        return @as(f64, @floatFromInt(self.total_latency_ns)) /
            @as(f64, @floatFromInt(self.transfers));
    }
};

const DeviceDmaRecommendation = struct {
    device_index: usize,
    device_id: u32,
    dma_block_size: usize,
    dma_parallelism: usize,
    measured_bytes_per_second: f64,
    average_latency_ns: f64,
    windows: usize = 0,
};

/// Immutable DMA settings shared by every device participating in one load.
/// The slices are owned by the enclosing platform settings.
pub const DmaLoadConfig = struct {
    device_numa_nodes: []const ?usize,
    block_size: usize,
    max_in_flight_per_device: usize,
    max_mapped_bytes: usize,
};

pub fn requiredDmaWorkspaceBytes(config: DmaLoadConfig) !usize {
    const maximum_request_blocks = try maximumCoalescedJobBlocks(
        max_load_read_request_size,
        config.block_size,
    );
    var required_blocks: usize = 0;
    if (config.device_numa_nodes[0] == null) {
        const feed_blocks = std.math.mul(
            usize,
            config.device_numa_nodes.len,
            config.max_in_flight_per_device,
        ) catch return error.InvalidDmaLoadConfig;
        required_blocks = @max(feed_blocks, maximum_request_blocks);
    } else {
        for (config.device_numa_nodes, 0..) |maybe_node, index| {
            const node = maybe_node.?;
            var seen = false;
            for (config.device_numa_nodes[0..index]) |previous| {
                if (previous.? == node) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            var device_count: usize = 0;
            for (config.device_numa_nodes) |candidate| {
                if (candidate.? == node) device_count += 1;
            }
            const feed_blocks = std.math.mul(
                usize,
                device_count,
                config.max_in_flight_per_device,
            ) catch return error.InvalidDmaLoadConfig;
            required_blocks = std.math.add(
                usize,
                required_blocks,
                @max(feed_blocks, maximum_request_blocks),
            ) catch return error.InvalidDmaLoadConfig;
        }
    }
    return std.math.mul(usize, required_blocks, config.block_size) catch
        error.InvalidDmaLoadConfig;
}

fn validateDmaLoadConfig(config: DmaLoadConfig) !void {
    if (config.device_numa_nodes.len == 0 or config.device_numa_nodes.len > 64 or
        config.block_size == 0 or config.max_in_flight_per_device == 0 or
        config.max_in_flight_per_device > max_load_dma_parallelism or
        config.block_size > max_load_read_request_size or
        config.max_mapped_bytes < config.block_size)
        return error.InvalidDmaLoadConfig;
    var known_numa_nodes: usize = 0;
    for (config.device_numa_nodes) |maybe_node| {
        if (maybe_node) |node| {
            known_numa_nodes += 1;
            if (node >= DmaBenchmarkNumaAllocator.max_nodes)
                return error.InvalidDmaLoadConfig;
        }
    }
    if (known_numa_nodes != 0 and known_numa_nodes != config.device_numa_nodes.len)
        return error.InvalidDmaLoadConfig;
    if (known_numa_nodes != 0 and builtin.os.tag != .linux)
        return error.DmaBenchmarkNumaUnsupported;
    if (try requiredDmaWorkspaceBytes(config) > config.max_mapped_bytes)
        return error.InvalidDmaLoadConfig;
}

fn dupeDmaLoadConfig(allocator: std.mem.Allocator, config: DmaLoadConfig) !DmaLoadConfig {
    try validateDmaLoadConfig(config);
    const nodes = try allocator.dupe(?usize, config.device_numa_nodes);
    return .{
        .device_numa_nodes = nodes,
        .block_size = config.block_size,
        .max_in_flight_per_device = config.max_in_flight_per_device,
        .max_mapped_bytes = config.max_mapped_bytes,
    };
}

fn freeDmaLoadConfig(allocator: std.mem.Allocator, config: DmaLoadConfig) void {
    allocator.free(config.device_numa_nodes);
}

test "DMA load config validates uniform caps, topology, and workspace budget" {
    const valid: DmaLoadConfig = .{
        .device_numa_nodes = &.{ null, null },
        .block_size = 4 * 1024 * 1024,
        .max_in_flight_per_device = 8,
        .max_mapped_bytes = 64 * 1024 * 1024,
    };
    try validateDmaLoadConfig(valid);

    var invalid = valid;
    invalid.device_numa_nodes = &.{ 0, null };
    try std.testing.expectError(error.InvalidDmaLoadConfig, validateDmaLoadConfig(invalid));
    invalid = valid;
    invalid.max_mapped_bytes -= 1;
    try std.testing.expectError(error.InvalidDmaLoadConfig, validateDmaLoadConfig(invalid));
}

/// Owned, reusable host-DMA workspace. A workspace may be borrowed by only one
/// load at a time; all registered arenas remain mapped until `deinit`.
pub const DmaPlatformSettings = struct {
    config: DmaLoadConfig,

    allocator: std.mem.Allocator,
    platform: *const Platform,
    workspace: DmaBenchmarkSourcePools,
    calibrated: bool,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        config: DmaLoadConfig,
        calibrated: bool,
    ) !DmaPlatformSettings {
        try validateDmaLoadConfig(config);
        try validateDmaPlatform(platform);
        if (config.device_numa_nodes.len != platform.devices.len)
            return error.DmaDeviceMismatch;
        const owned_config = try dupeDmaLoadConfig(allocator, config);
        errdefer freeDmaLoadConfig(allocator, owned_config);
        var workspace = try DmaBenchmarkSourcePools.init(
            allocator,
            io,
            platform,
            owned_config.device_numa_nodes,
            owned_config.max_mapped_bytes,
        );
        errdefer workspace.deinit();
        return .{
            .config = owned_config,
            .allocator = allocator,
            .platform = platform,
            .workspace = workspace,
            .calibrated = calibrated,
        };
    }

    fn adopt(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        config: DmaLoadConfig,
        workspace: DmaBenchmarkSourcePools,
    ) !DmaPlatformSettings {
        try validateDmaLoadConfig(config);
        try validateDmaPlatform(platform);
        if (config.device_numa_nodes.len != platform.devices.len)
            return error.DmaDeviceMismatch;
        const owned_config = try dupeDmaLoadConfig(allocator, config);
        return .{
            .config = owned_config,
            .allocator = allocator,
            .platform = platform,
            .workspace = workspace,
            .calibrated = true,
        };
    }

    fn validateLoad(self: *DmaPlatformSettings, platform: *const Platform) !void {
        try validateDmaLoadConfig(self.config);
        if (platform != self.platform) return error.DmaPlatformMismatch;
        if (self.config.device_numa_nodes.len != platform.devices.len)
            return error.DmaDeviceMismatch;
        if (self.workspace.allocatedBytes() > self.config.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;
    }

    fn retainedMappedBytes(self: *const DmaPlatformSettings) usize {
        return self.workspace.allocatedBytes();
    }

    fn numaPoolCount(self: *const DmaPlatformSettings) usize {
        return self.workspace.pools.len;
    }

    fn deinit(self: *DmaPlatformSettings) void {
        const io = self.workspace.io;
        const mapped_bytes = self.workspace.allocatedBytes();
        const started = std.Io.Timestamp.now(io, .awake);
        self.workspace.deinit();
        const elapsed_ns: u64 = @intCast(@max(
            started.untilNow(io, .awake).nanoseconds,
            0,
        ));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        freeDmaLoadConfig(self.allocator, self.config);
        self.* = undefined;
    }
};

const dma_platform_idle = 0;
const dma_platform_inspecting = 1;
const dma_platform_calibrating = 2;
const dma_platform_loading = 3;
const dma_platform_destroying = 4;

pub fn isDirectTransferPlatform(platform: *const Platform) bool {
    return platform.target == .cuda or platform.target == .rocm or
        platform.target == .oneapi;
}

fn validateDmaPlatform(platform: *const Platform) !void {
    if (platform.devices.len == 0 or platform.devices.len > 64)
        return error.DmaDeviceMismatch;
    const device_kind = platform.devices[0].kind();
    for (platform.devices[1..]) |device| {
        if (!std.mem.eql(u8, device_kind, device.kind()))
            return error.HeterogeneousDmaUnsupported;
    }
}

fn dmaSettingsFromOpaque(ptr: *anyopaque) *DmaPlatformSettings {
    return @ptrCast(@alignCast(ptr));
}

fn destroyDmaPlatformSettings(settings: *DmaPlatformSettings) void {
    const allocator = settings.allocator;
    settings.deinit();
    allocator.destroy(settings);
}

fn beginPlatformDmaOperation(platform: *const Platform, operation: u8) !void {
    const mutable: *Platform = @constCast(platform);
    if (mutable._dma.operation.cmpxchgStrong(
        dma_platform_idle,
        operation,
        .acq_rel,
        .acquire,
    ) != null) return error.DmaWorkspaceBusy;
}

fn endPlatformDmaOperation(platform: *const Platform, operation: u8) void {
    const mutable: *Platform = @constCast(platform);
    const previous = mutable._dma.operation.swap(dma_platform_idle, .release);
    std.debug.assert(previous == operation);
}

pub fn initPlatformDma(
    platform: *Platform,
    allocator: std.mem.Allocator,
    io: std.Io,
    defaults: platform_mod.TransferConfig,
) !void {
    if (!isDirectTransferPlatform(platform)) return;
    try validateDmaPlatform(platform);

    const numa_nodes = try resolveDmaNumaNodes(
        allocator,
        platform,
        defaults.device_numa_nodes,
    );
    defer allocator.free(numa_nodes);

    var settings = try DmaPlatformSettings.init(
        allocator,
        io,
        platform,
        .{
            .device_numa_nodes = numa_nodes,
            .block_size = defaults.block_size,
            .max_in_flight_per_device = defaults.max_in_flight_per_device,
            .max_mapped_bytes = defaults.max_mapped_bytes,
        },
        false,
    );
    errdefer settings.deinit();
    const owned = try allocator.create(DmaPlatformSettings);
    owned.* = settings;
    std.debug.assert(platform._dma.settings.swap(owned, .release) == null);
}

pub fn platformTransferSettings(platform: *Platform) !?platform_mod.TransferSettings {
    if (!isDirectTransferPlatform(platform)) return null;
    try beginPlatformDmaOperation(platform, dma_platform_inspecting);
    defer endPlatformDmaOperation(platform, dma_platform_inspecting);
    const raw = platform._dma.settings.load(.acquire) orelse return null;
    const settings = dmaSettingsFromOpaque(raw);
    return .{
        .calibrated = settings.calibrated,
        .block_size = settings.config.block_size,
        .max_in_flight_per_device = settings.config.max_in_flight_per_device,
        .max_mapped_bytes = settings.config.max_mapped_bytes,
        .retained_mapped_bytes = settings.retainedMappedBytes(),
        .numa_pool_count = settings.numaPoolCount(),
    };
}

pub fn deinitPlatformDma(platform: *Platform) void {
    if (!isDirectTransferPlatform(platform)) return;
    if (platform._dma.operation.cmpxchgStrong(
        dma_platform_idle,
        dma_platform_destroying,
        .acq_rel,
        .acquire,
    ) != null) @panic("Platform.deinit called while DMA state is borrowed");
    const raw = platform._dma.settings.swap(null, .acq_rel);
    if (raw) |ptr| destroyDmaPlatformSettings(dmaSettingsFromOpaque(ptr));
}

pub fn acquirePlatformDmaSettings(
    platform: *const Platform,
) !*DmaPlatformSettings {
    try beginPlatformDmaOperation(platform, dma_platform_loading);
    errdefer endPlatformDmaOperation(platform, dma_platform_loading);
    const mutable: *Platform = @constCast(platform);
    const raw = mutable._dma.settings.load(.acquire) orelse
        return error.DmaResourcesRequired;
    const settings = dmaSettingsFromOpaque(raw);
    try settings.validateLoad(platform);
    return settings;
}

pub fn releasePlatformDmaSettings(platform: *const Platform) void {
    endPlatformDmaOperation(platform, dma_platform_loading);
}

const DmaBenchmarkReport = struct {
    allocator: std.mem.Allocator,
    resources: DmaPlatformSettings,
    recommendation: DeviceDmaRecommendation,
    samples: []DmaBenchmarkSample,
    elapsed_ns: u64,
    setup_ns: u64 = 0,
    sampling_ns: u64 = 0,
    device_allocator_warmup_ns: u64 = 0,
    source_registration_ns: u64 = 0,
    benchmark_setup_ns: u64 = 0,
    benchmark_overhead_ns: u64 = 0,
    source_cleanup_ns: u64 = 0,
    calibration_ns: u64 = 0,
    windows: usize = 0,

    fn deinit(self: *DmaBenchmarkReport) void {
        self.resources.deinit();
        self.deinitReport();
    }

    fn deinitReport(self: *DmaBenchmarkReport) void {
        self.allocator.free(self.samples);
        self.* = undefined;
    }
};

pub const BenchTransferOptions = struct {
    block_sizes: []const usize = &default_dma_benchmark_block_sizes,
    /// Fixed per-device width used by the block screen and the loader.
    block_parallelism: usize = 8,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until the representative device completes the transfer target.
    duration_ns: u64 = 2 * std.time.ns_per_ms,
    minimum_transfers_per_device: u64 = 32,
    /// Borderline local decisions receive longer alternating paired windows.
    confirmation_duration_ns: u64 = 25 * std.time.ns_per_ms,
    confirmation_minimum_transfers_per_device: u64 = 256,
    confirmation_margin: f64 = 0.02,
    /// Prefer a smaller transaction once it supplies enough headroom over the
    /// source pipeline instead of maximizing isolated copy-engine throughput.
    block_selection_tolerance: f64 = 0.08,
    max_mapped_bytes: usize = 2 * 1024 * 1024 * 1024,
    /// Optional device-index to NUMA-node override. When absent, complete PJRT
    /// `numa_node` attributes select local pools; incomplete or unsupported
    /// topology falls back to one shared DmaMapped pool.
    device_numa_nodes: []const usize = &.{},
};

const DmaBenchmarkOpts = BenchTransferOptions;

const DmaBenchmarkNumaAllocator = struct {
    const max_nodes = 1024;
    const mpol_bind = 2;

    parent: std.mem.Allocator,
    node: ?usize,

    fn allocator(self: *DmaBenchmarkNumaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *DmaBenchmarkNumaAllocator = @ptrCast(@alignCast(ctx));
        const allocation = self.parent.rawAlloc(len, alignment, ret_addr) orelse return null;
        const node = self.node orelse return allocation;
        if (comptime builtin.os.tag != .linux) {
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }

        const word_bits = @bitSizeOf(usize);
        var node_mask: [max_nodes / word_bits]usize = @splat(0);
        node_mask[node / word_bits] = @as(usize, 1) << @intCast(node % word_bits);
        const rc = std.os.linux.syscall6(
            .mbind,
            @intFromPtr(allocation),
            len,
            mpol_bind,
            @intFromPtr(&node_mask),
            // Linux get_nodes() decrements maxnode before copying the mask;
            // raw callers include the same extra sentinel bit as libnuma.
            node + 2,
            0,
        );
        if (std.os.linux.errno(rc) != .SUCCESS) {
            log.err("unable to bind DMA benchmark allocation ({Bi:.2}) to NUMA node {d}: {s}", .{
                len,
                node,
                @tagName(std.os.linux.errno(rc)),
            });
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }
        return allocation;
    }

    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *DmaBenchmarkNumaAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, alignment, ret_addr);
    }
};

const PjrtPinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,

    fn init(memory: *const Memory, size: usize) !PjrtPinnedHostAllocation {
        const api = memory.platform.pjrt_api;
        const buffer = try memory.platform.pjrt_client.createUninitializedBuffer(api, .{
            .dims = &.{@intCast(size)},
            .element_type = .u8,
            .layout = .{
                .tiled = .{
                    .minor_to_major = &.{0},
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = .{ .memory = memory.pjrt_memory },
        });
        errdefer buffer.deinit(api);
        if (!buffer.isOnCpu(api)) return error.PinnedHostMemoryNotHostVisible;

        // The writable pointer is borrowed from PJRT. Keep both the external
        // reference and its owning buffer alive for the arena's whole lifetime.
        try buffer.increaseExternalReferenceCount(api);
        errdefer buffer.decreaseExternalReferenceCount(api) catch {};
        const ptr: [*]u8 = @ptrCast(try buffer.opaqueDeviceMemoryDataPointer(api));
        return .{
            .buffer = buffer,
            .api = api,
            .data = ptr[0..size],
        };
    }

    fn deinit(self: PjrtPinnedHostAllocation) void {
        self.buffer.decreaseExternalReferenceCount(self.api) catch unreachable;
        self.buffer.deinit(self.api);
    }
};

const DmaBenchmarkSourceAllocation = union(enum) {
    dma_map: []u8,
    pjrt_host: PjrtPinnedHostAllocation,

    fn data(self: *const DmaBenchmarkSourceAllocation) []u8 {
        return switch (self.*) {
            .dma_map => |bytes| bytes,
            .pjrt_host => |allocation| allocation.data,
        };
    }

    fn deinit(
        self: DmaBenchmarkSourceAllocation,
        dma_map_allocator: std.mem.Allocator,
    ) void {
        switch (self) {
            .dma_map => |bytes| dma_map_allocator.free(bytes),
            .pjrt_host => |allocation| allocation.deinit(),
        }
    }
};

const DmaBenchmarkSourcePool = struct {
    numa_allocator: DmaBenchmarkNumaAllocator,
    dma_map_allocator: mem.DmaMapAllocator,
    pjrt_host_memory: ?*const Memory,
    allocations: std.ArrayListUnmanaged(DmaBenchmarkSourceAllocation) = .empty,
    source: []u8 = &.{},
};

pub const DmaBenchmarkSourcePools = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pools: []DmaBenchmarkSourcePool,
    device_pool_indices: []usize,
    registration_ns: u64 = 0,
    max_mapped_bytes: usize,
    allocated_bytes: std.atomic.Value(usize) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        device_numa_nodes: []const ?usize,
        max_mapped_bytes: usize,
    ) !DmaBenchmarkSourcePools {
        var unique_nodes: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer unique_nodes.deinit(allocator);
        for (device_numa_nodes) |maybe_node| {
            if (maybe_node) |node| try unique_nodes.put(allocator, node, {});
        }
        const pool_count = @max(@as(usize, 1), unique_nodes.count());
        const pools = try allocator.alloc(DmaBenchmarkSourcePool, pool_count);
        errdefer allocator.free(pools);
        const device_pool_indices = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(device_pool_indices);
        if (unique_nodes.count() == 0) {
            const pool = &pools[0];
            pool.numa_allocator = .{ .parent = allocator, .node = null };
            pool.dma_map_allocator = .init(pool.numa_allocator.allocator(), platform);
            pool.pjrt_host_memory = if (platform.target == .rocm)
                platform.devices[0].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable
            else
                null;
            pool.allocations = .empty;
            pool.source = &.{};
            @memset(device_pool_indices, 0);
            return .{
                .allocator = allocator,
                .io = io,
                .pools = pools,
                .device_pool_indices = device_pool_indices,
                .max_mapped_bytes = max_mapped_bytes,
            };
        }

        var nodes: std.ArrayListUnmanaged(usize) = .empty;
        defer nodes.deinit(allocator);
        for (device_numa_nodes, 0..) |maybe_node, device_index| {
            const node = maybe_node orelse {
                device_pool_indices[device_index] = 0;
                continue;
            };
            var pool_index: ?usize = null;
            for (nodes.items, 0..) |existing, index| {
                if (existing == node) {
                    pool_index = index;
                    break;
                }
            }
            if (pool_index == null) {
                pool_index = nodes.items.len;
                try nodes.append(allocator, node);
                const pool = &pools[pool_index.?];
                pool.numa_allocator = .{ .parent = allocator, .node = node };
                pool.dma_map_allocator = .init(pool.numa_allocator.allocator(), platform);
                pool.pjrt_host_memory = if (platform.target == .rocm)
                    platform.devices[device_index].memory(.host_pinned) orelse
                        return error.PinnedHostMemoryUnavailable
                else
                    null;
                pool.allocations = .empty;
                pool.source = &.{};
            }
            device_pool_indices[device_index] = pool_index.?;
        }
        std.debug.assert(nodes.items.len == pools.len);
        return .{
            .allocator = allocator,
            .io = io,
            .pools = pools,
            .device_pool_indices = device_pool_indices,
            .max_mapped_bytes = max_mapped_bytes,
        };
    }

    fn deinit(self: *DmaBenchmarkSourcePools) void {
        for (self.pools) |*pool| {
            for (pool.allocations.items) |allocation| {
                allocation.deinit(pool.dma_map_allocator.allocator());
            }
            pool.allocations.deinit(self.allocator);
        }
        self.allocator.free(self.device_pool_indices);
        self.allocator.free(self.pools);
        self.* = undefined;
    }

    fn cleanupSourceForDevice(
        self: *const DmaBenchmarkSourcePools,
        device_index: usize,
        minimum_len: usize,
    ) []const u8 {
        const pool = &self.pools[self.device_pool_indices[device_index]];
        var index = pool.allocations.items.len;
        while (index != 0) {
            index -= 1;
            const arena = pool.allocations.items[index].data();
            if (arena.len >= minimum_len) return arena;
        }
        unreachable;
    }

    fn growPool(self: *DmaBenchmarkSourcePools, pool_index: usize, required_bytes: usize) !void {
        const pool = &self.pools[pool_index];
        if (required_bytes <= pool.source.len) return;
        if (try std.math.add(usize, self.allocatedBytes(), required_bytes) > self.max_mapped_bytes)
            return error.DmaBenchmarkPinnedBudgetExceeded;
        const started = std.Io.Timestamp.now(self.io, .awake);
        defer self.registration_ns +|= @intCast(@max(started.untilNow(self.io, .awake).nanoseconds, 0));
        try self.allocatePool(pool_index, required_bytes);
    }

    fn allocatePool(self: *DmaBenchmarkSourcePools, pool_index: usize, required_bytes: usize) !void {
        const pool = &self.pools[pool_index];
        const started = std.Io.Timestamp.now(self.io, .awake);
        var allocation: DmaBenchmarkSourceAllocation = if (pool.pjrt_host_memory) |host_memory|
            .{ .pjrt_host = try .init(host_memory, required_bytes) }
        else blk: {
            const dma_map_allocator = pool.dma_map_allocator.allocator();
            break :blk .{ .dma_map = try dma_map_allocator.alignedAlloc(
                u8,
                .fromByteUnits(std.heap.page_size_min),
                required_bytes,
            ) };
        };
        errdefer allocation.deinit(pool.dma_map_allocator.allocator());
        const mapped_at = std.Io.Timestamp.now(self.io, .awake);
        try pool.allocations.append(self.allocator, allocation);
        const replacement = allocation.data();
        _ = self.allocated_bytes.fetchAdd(replacement.len, .release);
        pool.source = replacement;
        const finished_at = std.Io.Timestamp.now(self.io, .awake);
        const map_ns: u64 = if (pool.pjrt_host_memory != null)
            0
        else
            @intCast(@max(started.durationTo(mapped_at).nanoseconds, 0));
        const elapsed_ns: u64 = @intCast(@max(started.durationTo(finished_at).nanoseconds, 0));
        const allocation_kind = if (pool.pjrt_host_memory != null) "pjrt_host" else "dma_map";
        if (pool.numa_allocator.node) |node| {
            log.info("DMA mapped arena numa_node={d} allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                node,
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        } else {
            log.info("DMA mapped arena numa_node=single allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        }
    }

    fn allocatedBytes(self: *const DmaBenchmarkSourcePools) usize {
        return self.allocated_bytes.load(.acquire);
    }

    /// Ensures every NUMA pool can feed its calibrated devices and hold one
    /// complete fixed-size source request. Independent nodes register their
    /// missing slabs concurrently; existing retained arenas are reused first.
    fn ensureLoadBlockReserves(
        self: *DmaBenchmarkSourcePools,
        block_size: usize,
        calibrated_reserves: []const usize,
    ) !void {
        if (block_size == 0 or calibrated_reserves.len != self.pools.len)
            return error.InvalidDmaLoadConfig;
        const request_blocks = try maximumCoalescedJobBlocks(
            max_load_read_request_size,
            block_size,
        );
        return self.ensureBlockReserves(block_size, request_blocks, calibrated_reserves);
    }

    fn ensureBlockReserves(
        self: *DmaBenchmarkSourcePools,
        block_size: usize,
        request_blocks: usize,
        calibrated_reserves: []const usize,
    ) !void {
        const missing_bytes = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(missing_bytes);
        var missing_total: usize = 0;
        for (self.pools, calibrated_reserves, missing_bytes) |pool, reserve, *missing| {
            var usable_blocks: usize = 0;
            for (pool.allocations.items) |arena| {
                usable_blocks = std.math.add(
                    usize,
                    usable_blocks,
                    arena.data().len / block_size,
                ) catch return error.DmaMappedBudgetExceeded;
            }
            const required_blocks = @max(reserve, request_blocks);
            const missing_blocks = required_blocks -| usable_blocks;
            missing.* = std.math.mul(usize, missing_blocks, block_size) catch
                return error.DmaMappedBudgetExceeded;
            missing_total = std.math.add(usize, missing_total, missing.*) catch
                return error.DmaMappedBudgetExceeded;
        }
        if (missing_total == 0) return;
        const mapped_after_growth = std.math.add(
            usize,
            self.allocatedBytes(),
            missing_total,
        ) catch return error.DmaMappedBudgetExceeded;
        if (mapped_after_growth > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;

        const Worker = struct {
            pools: *DmaBenchmarkSourcePools,
            pool_index: usize,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                worker.pools.allocatePool(worker.pool_index, worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(
                        0,
                        @intFromError(err),
                        .release,
                        .monotonic,
                    );
                };
            }
        };
        var first_error: std.atomic.Value(u16) = .init(0);
        var group: std.Io.Group = .init;
        var group_error: ?anyerror = null;
        const registration_started = std.Io.Timestamp.now(self.io, .awake);
        for (missing_bytes, 0..) |bytes, pool_index| {
            if (bytes == 0) continue;
            group.concurrent(self.io, Worker.run, .{Worker{
                .pools = self,
                .pool_index = pool_index,
                .bytes = bytes,
                .first_error = &first_error,
            }}) catch |err| {
                group_error = err;
                break;
            };
        }
        group.await(self.io) catch |err| if (group_error == null) {
            group_error = err;
        };
        self.registration_ns +|= @intCast(@max(
            registration_started.untilNow(self.io, .awake).nanoseconds,
            0,
        ));
        if (group_error) |err| return err;
        const error_code = first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
    }

    pub fn blockPoolArenaProvider(self: *DmaBenchmarkSourcePools) mem.DmaBlockPool.ArenaProvider {
        return .{
            .context = self,
            .node_count = self.pools.len,
            .arenaCountFn = struct {
                fn call(context: *anyopaque, node_index: usize) usize {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.pools[node_index].allocations.items.len;
                }
            }.call,
            .arenaFn = struct {
                fn call(context: *anyopaque, node_index: usize, arena_index: usize) []u8 {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.pools[node_index].allocations.items[arena_index].data();
                }
            }.call,
            .allocateFn = struct {
                fn call(context: *anyopaque, node_index: usize, len: usize) ![]u8 {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    if (try std.math.add(usize, pools.allocatedBytes(), len) > pools.max_mapped_bytes)
                        return error.DmaMappedBudgetExceeded;
                    const started = std.Io.Timestamp.now(pools.io, .awake);
                    try pools.allocatePool(node_index, len);
                    pools.registration_ns +|= @intCast(@max(
                        started.untilNow(pools.io, .awake).nanoseconds,
                        0,
                    ));
                    return pools.pools[node_index].allocations.items[
                        pools.pools[node_index].allocations.items.len - 1
                    ].data();
                }
            }.call,
            .mappedBytesFn = struct {
                fn call(context: *anyopaque) usize {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.allocatedBytes();
                }
            }.call,
        };
    }
};

const DmaBenchmarkRunMetrics = struct {
    bytes: u64,
    transfers: u64,
    total_latency_ns: u64,
    elapsed_ns: u64,

    fn bytesPerSecond(self: DmaBenchmarkRunMetrics) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed_ns));
    }

    fn averageLatencyNs(self: DmaBenchmarkRunMetrics) f64 {
        if (self.transfers == 0) return 0;
        return @as(f64, @floatFromInt(self.total_latency_ns)) /
            @as(f64, @floatFromInt(self.transfers));
    }
};

const DmaBenchmarkAtomicMetrics = struct {
    bytes: std.atomic.Value(u64) = .init(0),
    transfers: std.atomic.Value(u64) = .init(0),
    total_latency_ns: std.atomic.Value(u64) = .init(0),
};

const DmaBenchmarkManager = struct {
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffer: *pjrt.Buffer,
};

const ReusableDmaBenchmarkCohort = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    device_index: usize,
    block_size: usize,
    managers: std.ArrayListUnmanaged(DmaBenchmarkManager) = .empty,
    warmed_managers: usize = 0,
    first_error: std.atomic.Value(u16) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        device_index: usize,
        block_size: usize,
    ) ReusableDmaBenchmarkCohort {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .device_index = device_index,
            .block_size = block_size,
        };
    }

    fn recordError(self: *ReusableDmaBenchmarkCohort, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn transfer(
        self: *ReusableDmaBenchmarkCohort,
        source: []const u8,
        slot: usize,
        metrics: ?*DmaBenchmarkAtomicMetrics,
    ) void {
        const len = self.block_size;
        const source_offset = slot * self.block_size;
        const started = std.Io.Timestamp.now(self.io, .awake);
        const event = self.managers.items[slot].manager.transferData(
            self.platform.pjrt_api,
            0,
            source[source_offset..][0..len],
            0,
            false,
        ) catch |err| {
            self.recordError(err);
            return;
        };
        event.await(self.platform.pjrt_api, self.io) catch |err| {
            event.deinit(self.platform.pjrt_api);
            self.recordError(err);
            return;
        };
        event.deinit(self.platform.pjrt_api);
        if (metrics) |output| {
            const elapsed_ns: u64 = @intCast(@max(started.untilNow(self.io, .awake).nanoseconds, 0));
            _ = output.bytes.fetchAdd(@intCast(len), .monotonic);
            _ = output.transfers.fetchAdd(1, .monotonic);
            _ = output.total_latency_ns.fetchAdd(elapsed_ns, .monotonic);
        }
    }

    fn ensureReady(self: *ReusableDmaBenchmarkCohort, source: []const u8, parallelism: usize) !void {
        const required_bytes = std.math.mul(usize, self.block_size, parallelism) catch return error.OutOfMemory;
        if (required_bytes > source.len) return error.DmaBenchmarkPinnedBudgetExceeded;
        var dims = [_]i64{@intCast(self.block_size)};
        const shape_spec: pjrt.ShapeSpec = .init(&dims, .u8);
        const memory = self.platform.devices[self.device_index].memory(.default).?;
        while (self.managers.items.len < parallelism) {
            const manager = try self.platform.pjrt_client.createBuffersForAsyncHostToDevice(self.platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(self.platform.pjrt_api);
            const buffer = try manager.retrieveBuffer(self.platform.pjrt_api, 0);
            try self.managers.append(self.allocator, .{ .manager = manager, .buffer = buffer });
        }
        while (self.warmed_managers < parallelism) : (self.warmed_managers += 1) {
            const slot = self.warmed_managers;
            self.transfer(source, slot, null);
            self.transfer(source, slot, null);
            const error_code = self.first_error.load(.acquire);
            if (error_code != 0) return @errorFromInt(error_code);
        }
    }

    fn deinit(self: *ReusableDmaBenchmarkCohort, source: []const u8) void {
        for (self.managers.items) |manager| {
            const event = manager.manager.transferData(
                self.platform.pjrt_api,
                0,
                source[0..self.block_size],
                0,
                true,
            ) catch null;
            if (event) |done| {
                done.await(self.platform.pjrt_api, self.io) catch {};
                done.deinit(self.platform.pjrt_api);
            }
            manager.manager.deinit(self.platform.pjrt_api);
            manager.buffer.deinit(self.platform.pjrt_api);
        }
        self.managers.deinit(self.allocator);
        self.* = undefined;
    }
};

fn dmaBenchmarkWindowComplete(
    elapsed_ns: u64,
    minimum_duration_ns: u64,
    completed_transfers: u64,
    minimum_transfers: u64,
) bool {
    return elapsed_ns >= minimum_duration_ns and completed_transfers >= minimum_transfers;
}

fn runReusableDmaBenchmarkWindow(
    io: std.Io,
    cohort: *ReusableDmaBenchmarkCohort,
    source: []const u8,
    parallelism: usize,
    duration_ns: u64,
    minimum_transfers: u64,
    setup_ns: *u64,
) !DmaBenchmarkRunMetrics {
    var metrics: DmaBenchmarkAtomicMetrics = .{};
    const setup_started = std.Io.Timestamp.now(io, .awake);
    try cohort.ensureReady(source, parallelism);
    setup_ns.* +|= @intCast(@max(setup_started.untilNow(io, .awake).nanoseconds, 0));

    const Worker = struct {
        cohort: *ReusableDmaBenchmarkCohort,
        source: []const u8,
        slot: usize,
        metrics: *DmaBenchmarkAtomicMetrics,
        ready: *std.atomic.Value(usize),
        start: *std.Io.Event,
        stop: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            _ = self.ready.fetchAdd(1, .release);
            self.start.waitUncancelable(self.cohort.io);
            while (!self.stop.load(.acquire)) {
                self.cohort.transfer(self.source, self.slot, self.metrics);
                if (self.cohort.first_error.load(.acquire) != 0) return;
            }
        }
    };

    var ready: std.atomic.Value(usize) = .init(0);
    var start: std.Io.Event = .unset;
    var stop: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    for (0..parallelism) |slot| {
        group.concurrent(io, Worker.run, .{Worker{
            .cohort = cohort,
            .source = source,
            .slot = slot,
            .metrics = &metrics,
            .ready = &ready,
            .start = &start,
            .stop = &stop,
        }}) catch |err| {
            stop.store(true, .release);
            start.set(io);
            group.await(io) catch {};
            return err;
        };
    }
    while (ready.load(.acquire) != parallelism) try io.sleep(.fromMilliseconds(1), .awake);
    const measured_at = std.Io.Timestamp.now(io, .awake);
    start.set(io);
    while (true) {
        const elapsed_ns: u64 = @intCast(@max(measured_at.untilNow(io, .awake).nanoseconds, 0));
        const error_code = cohort.first_error.load(.acquire);
        if (error_code != 0) {
            stop.store(true, .release);
            try group.await(io);
            return @errorFromInt(error_code);
        }
        if (dmaBenchmarkWindowComplete(
            elapsed_ns,
            duration_ns,
            metrics.transfers.load(.acquire),
            minimum_transfers,
        )) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
    stop.store(true, .release);
    try group.await(io);
    const elapsed_ns: u64 = @intCast(@max(measured_at.untilNow(io, .awake).nanoseconds, 1));
    const error_code = cohort.first_error.load(.acquire);
    if (error_code != 0) return @errorFromInt(error_code);
    return .{
        .bytes = metrics.bytes.load(.acquire),
        .transfers = metrics.transfers.load(.acquire),
        .total_latency_ns = metrics.total_latency_ns.load(.acquire),
        .elapsed_ns = elapsed_ns,
    };
}

const ReusableDmaBenchmarkSession = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    samples: *std.ArrayListUnmanaged(DmaBenchmarkSample),
    cohorts: std.ArrayListUnmanaged(*ReusableDmaBenchmarkCohort) = .empty,
    setup_ns: u64 = 0,
    sampling_ns: u64 = 0,
    windows: usize = 0,

    fn createCohort(
        self: *ReusableDmaBenchmarkSession,
        device_index: usize,
        block_size: usize,
    ) !*ReusableDmaBenchmarkCohort {
        const cohort = try self.allocator.create(ReusableDmaBenchmarkCohort);
        errdefer self.allocator.destroy(cohort);
        cohort.* = .init(
            self.allocator,
            self.io,
            self.platform,
            device_index,
            block_size,
        );
        try self.cohorts.append(self.allocator, cohort);
        return cohort;
    }

    fn measure(
        self: *ReusableDmaBenchmarkSession,
        phase: DmaBenchmarkPhase,
        cohort: *ReusableDmaBenchmarkCohort,
        source: []const u8,
        parallelism: usize,
        duration_ns: u64,
        minimum_transfers: u64,
        repeat: usize,
    ) !DmaBenchmarkRunMetrics {
        const metrics = try runReusableDmaBenchmarkWindow(
            self.io,
            cohort,
            source,
            parallelism,
            duration_ns,
            minimum_transfers,
            &self.setup_ns,
        );
        self.sampling_ns +|= metrics.elapsed_ns;
        self.windows += 1;
        try self.samples.append(self.allocator, .{
            .phase = phase,
            .device_index = cohort.device_index,
            .block_size = cohort.block_size,
            .parallelism = parallelism,
            .repeat = repeat,
            .bytes = metrics.bytes,
            .transfers = metrics.transfers,
            .elapsed_ns = metrics.elapsed_ns,
            .total_latency_ns = metrics.total_latency_ns,
        });
        return metrics;
    }

    fn deinit(self: *ReusableDmaBenchmarkSession, source_pools: *const DmaBenchmarkSourcePools) void {
        for (self.cohorts.items) |cohort| {
            cohort.deinit(source_pools.cleanupSourceForDevice(
                cohort.device_index,
                cohort.block_size,
            ));
            self.allocator.destroy(cohort);
        }
        self.cohorts.deinit(self.allocator);
        self.* = undefined;
    }
};

fn finishDmaBenchmarkReport(
    result: *DmaBenchmarkReport,
    io: std.Io,
    benchmark_started: std.Io.Timestamp,
    device_allocator_warmup_ns: u64,
    session: *ReusableDmaBenchmarkSession,
    source_pools: *DmaBenchmarkSourcePools,
) void {
    const sampling_ns = session.sampling_ns;
    const benchmark_setup_ns = session.setup_ns;
    const windows = session.windows;
    const source_registration_ns = source_pools.registration_ns;
    session.deinit(source_pools);
    const elapsed_ns: u64 = @intCast(@max(
        benchmark_started.untilNow(io, .awake).nanoseconds,
        0,
    ));
    const calibration_ns = elapsed_ns -| device_allocator_warmup_ns -|
        source_registration_ns;
    result.elapsed_ns = elapsed_ns;
    result.setup_ns = device_allocator_warmup_ns +| source_registration_ns +| benchmark_setup_ns;
    result.sampling_ns = sampling_ns;
    result.device_allocator_warmup_ns = device_allocator_warmup_ns;
    result.source_registration_ns = source_registration_ns;
    result.benchmark_setup_ns = benchmark_setup_ns;
    result.benchmark_overhead_ns = calibration_ns -| sampling_ns -| benchmark_setup_ns;
    // Registered source arenas are now part of result.resources. Their
    // teardown is intentionally outside benchmark timing.
    result.source_cleanup_ns = 0;
    result.calibration_ns = calibration_ns;
    result.windows = windows;
}
const DmaBenchmarkCandidate = struct {
    value: usize,
    cohort: *ReusableDmaBenchmarkCohort,
    metrics: [dma_benchmark_repeats]DmaBenchmarkRunMetrics = undefined,
    metrics_len: usize = 0,

    fn appendMetric(self: *DmaBenchmarkCandidate, metric: DmaBenchmarkRunMetrics) void {
        std.debug.assert(self.metrics_len < self.metrics.len);
        self.metrics[self.metrics_len] = metric;
        self.metrics_len += 1;
    }

    fn metricSlice(self: *const DmaBenchmarkCandidate) []const DmaBenchmarkRunMetrics {
        return self.metrics[0..self.metrics_len];
    }

    fn median(self: DmaBenchmarkCandidate) DmaBenchmarkRunMetrics {
        std.debug.assert(self.metrics_len > 0);
        var scratch = self.metrics;
        const populated = scratch[0..self.metrics_len];
        std.mem.sort(DmaBenchmarkRunMetrics, populated, {}, struct {
            fn lessThan(_: void, lhs: DmaBenchmarkRunMetrics, rhs: DmaBenchmarkRunMetrics) bool {
                return lhs.bytesPerSecond() < rhs.bytesPerSecond();
            }
        }.lessThan);
        return populated[populated.len / 2];
    }
};

const DmaBenchmarkDecision = struct {
    index: usize,
    metrics: DmaBenchmarkRunMetrics,
};

fn selectDmaBenchmarkCandidate(
    allocator: std.mem.Allocator,
    candidates: []const DmaBenchmarkCandidate,
    tolerance: f64,
) !DmaBenchmarkDecision {
    std.debug.assert(candidates.len > 0);
    const medians = try allocator.alloc(DmaBenchmarkRunMetrics, candidates.len);
    defer allocator.free(medians);
    var fastest_index: usize = 0;
    for (candidates, medians, 0..) |candidate, *median, index| {
        median.* = candidate.median();
        if (median.bytesPerSecond() > medians[fastest_index].bytesPerSecond())
            fastest_index = index;
    }

    const floor = medians[fastest_index].bytesPerSecond() * (1.0 - tolerance);
    var selected_index = fastest_index;
    for (candidates, medians, 0..) |candidate, median, index| {
        if (median.bytesPerSecond() >= floor and
            candidate.value < candidates[selected_index].value)
            selected_index = index;
    }
    return .{ .index = selected_index, .metrics = medians[selected_index] };
}

fn dmaBenchmarkCandidateNeedsConfirmation(
    candidates: []const DmaBenchmarkCandidate,
    candidate_index: usize,
    peak_index: usize,
    tolerance: f64,
    margin: f64,
) bool {
    if (candidate_index == peak_index) return false;
    const candidate = candidates[candidate_index];
    const peak = candidates[peak_index];
    std.debug.assert(candidate.metrics_len == peak.metrics_len);
    var qualified_once = false;
    var rejected_once = false;
    for (candidate.metricSlice(), 0..) |metric, repeat| {
        var peak_rate: f64 = 0;
        for (candidates) |round_candidate| {
            std.debug.assert(round_candidate.metrics_len == candidate.metrics_len);
            peak_rate = @max(peak_rate, round_candidate.metrics[repeat].bytesPerSecond());
        }
        const ratio = if (peak_rate == 0) 0 else metric.bytesPerSecond() / peak_rate;
        if (ratio >= 1.0 - tolerance)
            qualified_once = true
        else
            rejected_once = true;
    }
    if (qualified_once and rejected_once) return true;
    const candidate_median = candidate.median();
    const peak_median = peak.median();
    const peak_rate = peak_median.bytesPerSecond();
    const ratio = if (peak_rate == 0) 0 else candidate_median.bytesPerSecond() / peak_rate;
    return @abs(ratio - (1.0 - tolerance)) <= margin;
}

fn medianDmaMetricRatioIndex(
    candidates: []const DmaBenchmarkRunMetrics,
    baselines: []const DmaBenchmarkRunMetrics,
) usize {
    std.debug.assert(candidates.len == baselines.len and candidates.len > 0);
    std.debug.assert(candidates.len <= dma_benchmark_repeats);
    var order_storage: [dma_benchmark_repeats]usize = undefined;
    const order = order_storage[0..candidates.len];
    for (order, 0..) |*index, i| index.* = i;
    const Context = struct {
        candidates: []const DmaBenchmarkRunMetrics,
        baselines: []const DmaBenchmarkRunMetrics,
    };
    std.mem.sort(usize, order, Context{ .candidates = candidates, .baselines = baselines }, struct {
        fn lessThan(context: Context, lhs: usize, rhs: usize) bool {
            const lhs_baseline = context.baselines[lhs].bytesPerSecond();
            const rhs_baseline = context.baselines[rhs].bytesPerSecond();
            const lhs_ratio = if (lhs_baseline == 0) 0 else context.candidates[lhs].bytesPerSecond() / lhs_baseline;
            const rhs_ratio = if (rhs_baseline == 0) 0 else context.candidates[rhs].bytesPerSecond() / rhs_baseline;
            return lhs_ratio < rhs_ratio;
        }
    }.lessThan);
    return order[order.len / 2];
}

fn confirmAndSelectDmaBenchmarkCandidate(
    session: *ReusableDmaBenchmarkSession,
    opts: DmaBenchmarkOpts,
    phase: DmaBenchmarkPhase,
    candidates: []const DmaBenchmarkCandidate,
    source: []const u8,
    parallelism: usize,
    tolerance: f64,
) !DmaBenchmarkDecision {
    const medians = try session.allocator.alloc(DmaBenchmarkRunMetrics, candidates.len);
    defer session.allocator.free(medians);
    const ratios = try session.allocator.alloc(f64, candidates.len);
    defer session.allocator.free(ratios);
    const confirmed_metrics = try session.allocator.alloc(?DmaBenchmarkRunMetrics, candidates.len);
    defer session.allocator.free(confirmed_metrics);
    @memset(confirmed_metrics, null);

    var peak_index: usize = 0;
    for (candidates, medians, 0..) |candidate, *median, index| {
        median.* = candidate.median();
        if (median.bytesPerSecond() > medians[peak_index].bytesPerSecond()) peak_index = index;
    }
    const peak_rate = medians[peak_index].bytesPerSecond();
    for (medians, ratios) |median, *ratio| {
        ratio.* = if (peak_rate == 0) 0 else median.bytesPerSecond() / peak_rate;
    }

    for (candidates, 0..) |_, candidate_index| {
        if (!dmaBenchmarkCandidateNeedsConfirmation(
            candidates,
            candidate_index,
            peak_index,
            tolerance,
            opts.confirmation_margin,
        )) continue;
        var candidate_runs: [dma_benchmark_repeats]DmaBenchmarkRunMetrics = undefined;
        var baseline_runs: [dma_benchmark_repeats]DmaBenchmarkRunMetrics = undefined;
        for (0..dma_benchmark_repeats) |repeat| {
            const order = if (repeat % 2 == 0)
                [_]usize{ candidate_index, peak_index }
            else
                [_]usize{ peak_index, candidate_index };
            for (order) |measured_index| {
                const measured = candidates[measured_index];
                const metrics = try session.measure(
                    phase,
                    measured.cohort,
                    source[0 .. measured.cohort.block_size * parallelism],
                    parallelism,
                    opts.confirmation_duration_ns,
                    opts.confirmation_minimum_transfers_per_device,
                    repeat,
                );
                if (measured_index == candidate_index)
                    candidate_runs[repeat] = metrics
                else
                    baseline_runs[repeat] = metrics;
            }
        }
        const representative = medianDmaMetricRatioIndex(
            &candidate_runs,
            &baseline_runs,
        );
        const baseline_rate = baseline_runs[representative].bytesPerSecond();
        ratios[candidate_index] = if (baseline_rate == 0) 0 else candidate_runs[representative].bytesPerSecond() / baseline_rate;
        confirmed_metrics[candidate_index] = candidate_runs[representative];
    }

    var maximum_ratio: f64 = 1;
    for (ratios) |ratio| maximum_ratio = @max(maximum_ratio, ratio);
    const floor = maximum_ratio * (1.0 - tolerance);
    var selected_index = peak_index;
    for (candidates, ratios, 0..) |candidate, ratio, index| {
        if (ratio >= floor and candidate.value < candidates[selected_index].value)
            selected_index = index;
    }
    return .{
        .index = selected_index,
        .metrics = confirmed_metrics[selected_index] orelse medians[selected_index],
    };
}

fn dmaBenchmarkTupleFeasible(source_len: usize, block_size: usize, parallelism: usize) bool {
    const bytes = std.math.mul(usize, block_size, parallelism) catch return false;
    return bytes <= source_len;
}

fn measureDmaBenchmarkCandidates(
    session: *ReusableDmaBenchmarkSession,
    phase: DmaBenchmarkPhase,
    candidates: []DmaBenchmarkCandidate,
    source: []const u8,
    parallelism: usize,
    duration_ns: u64,
    minimum_transfers_per_device: u64,
    repeats: usize,
) !void {
    for (0..repeats) |repeat| {
        for (0..candidates.len) |offset| {
            const index = (offset + repeat) % candidates.len;
            const candidate = &candidates[index];
            const metrics = try session.measure(
                phase,
                candidate.cohort,
                source[0 .. candidate.cohort.block_size * parallelism],
                parallelism,
                duration_ns,
                minimum_transfers_per_device,
                repeat,
            );
            candidate.appendMetric(metrics);
        }
    }
}

fn tuneDmaBenchmarkDevice(
    session: *ReusableDmaBenchmarkSession,
    opts: DmaBenchmarkOpts,
    source_pools: *DmaBenchmarkSourcePools,
    device_index: usize,
) !DeviceDmaRecommendation {
    const started_windows = session.windows;
    var block_count: usize = 0;
    var block_source_bytes: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_count += 1;
        block_source_bytes = @max(block_source_bytes, block_size * opts.block_parallelism);
    }
    if (block_count == 0) return error.NoFeasibleDmaBenchmarkTuple;
    const pool_index = source_pools.device_pool_indices[device_index];
    try source_pools.growPool(pool_index, block_source_bytes);
    const source_pool = &source_pools.pools[pool_index];

    const block_candidates = try session.allocator.alloc(DmaBenchmarkCandidate, block_count);
    errdefer session.allocator.free(block_candidates);
    var block_index: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_candidates[block_index] = .{
            .value = block_size,
            .cohort = try session.createCohort(device_index, block_size),
        };
        block_index += 1;
    }
    defer session.allocator.free(block_candidates);
    try measureDmaBenchmarkCandidates(
        session,
        .block,
        block_candidates,
        source_pool.source,
        opts.block_parallelism,
        opts.duration_ns,
        opts.minimum_transfers_per_device,
        dma_benchmark_repeats,
    );
    const block_decision = try confirmAndSelectDmaBenchmarkCandidate(
        session,
        opts,
        .block_confirmation,
        block_candidates,
        source_pool.source,
        opts.block_parallelism,
        opts.block_selection_tolerance,
    );
    const selected_cohort = block_candidates[block_decision.index].cohort;

    return .{
        .device_index = device_index,
        .device_id = session.platform.devices[device_index].id(),
        .dma_block_size = selected_cohort.block_size,
        .dma_parallelism = opts.block_parallelism,
        .measured_bytes_per_second = block_decision.metrics.bytesPerSecond(),
        .average_latency_ns = block_decision.metrics.averageLatencyNs(),
        .windows = session.windows - started_windows,
    };
}

fn warmupDmaBenchmarkDeviceAllocators(
    io: std.Io,
    platform: *const Platform,
) !void {
    const Worker = struct {
        platform: *const Platform,
        device_index: usize,
        first_error: *std.atomic.Value(u16),

        fn run(self: @This()) void {
            const dims: []const i64 = &.{1};
            const memory = self.platform.devices[self.device_index].memory(.default).?;
            const buffer = self.platform.pjrt_client.createUninitializedBuffer(self.platform.pjrt_api, .{
                .dims = dims,
                .element_type = pjrtx.bufferTypeFromDtype(.u8),
                .layout = self.platform.defaultMemoryLayout(dims, .u8),
                .dst = .{ .memory = memory.pjrt_memory },
            }) catch |err| {
                _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                return;
            };
            buffer.deinit(self.platform.pjrt_api);
        }
    };
    var first_error: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    for (platform.devices, 0..) |_, device_index| try group.concurrent(io, Worker.run, .{Worker{
        .platform = platform,
        .device_index = device_index,
        .first_error = &first_error,
    }});
    try group.await(io);
    const error_code = first_error.load(.acquire);
    if (error_code != 0) return @errorFromInt(error_code);
}

fn resolveDmaNumaNodes(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    override: []const usize,
) ![]?usize {
    const result = try allocator.alloc(?usize, platform.devices.len);
    @memset(result, null);
    if (override.len != 0) {
        if (override.len != platform.devices.len) return error.InvalidDmaBenchmarkOptions;
        if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
        for (override) |node| {
            if (node >= DmaBenchmarkNumaAllocator.max_nodes)
                return error.InvalidDmaBenchmarkOptions;
        }
        for (override, result) |node, *stored| stored.* = node;
        return result;
    }

    for (platform.devices, 0..) |_, device_index| {
        const node = platform.devices[device_index].numaNode() orelse {
            @memset(result, null);
            return result;
        };
        if (node >= DmaBenchmarkNumaAllocator.max_nodes) {
            @memset(result, null);
            return result;
        }
        result[device_index] = node;
    }
    if (comptime builtin.os.tag != .linux) @memset(result, null);
    return result;
}

/// Benchmarks synthetic DmaMapped PJRT transfers on one representative device.
/// Every addressable device allocator is still warmed and retained workspace is
/// prepared for the complete platform.
fn benchmarkSyntheticDma(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    opts: DmaBenchmarkOpts,
) !DmaBenchmarkReport {
    const benchmark_started = std.Io.Timestamp.now(io, .awake);
    if (platform.target != .cuda and platform.target != .rocm and platform.target != .oneapi)
        return error.DmaBenchmarkUnsupported;
    if (platform.devices.len == 0 or opts.block_sizes.len == 0)
        return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.duration_ns == 0 or opts.confirmation_duration_ns == 0)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.block_parallelism == 0 or opts.block_parallelism > max_load_dma_parallelism)
        return error.InvalidDmaBenchmarkOptions;
    if (!(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1) or
        !(opts.confirmation_margin >= 0 and opts.confirmation_margin < 1))
        return error.InvalidDmaBenchmarkOptions;
    var maximum_feasible_block_size: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (block_size == 0) return error.InvalidDmaBenchmarkOptions;
        if (dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            maximum_feasible_block_size = @max(maximum_feasible_block_size, block_size);
    }
    if (maximum_feasible_block_size == 0) return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.device_numa_nodes.len != 0 and
        opts.device_numa_nodes.len != platform.devices.len)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.device_numa_nodes.len != 0) {
        if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
        for (opts.device_numa_nodes) |node| {
            if (node >= DmaBenchmarkNumaAllocator.max_nodes)
                return error.InvalidDmaBenchmarkOptions;
        }
    }

    const representative_kind = platform.devices[0].kind();
    for (platform.devices[1..]) |device| {
        if (!std.mem.eql(u8, representative_kind, device.kind()))
            return error.HeterogeneousDmaUnsupported;
    }
    const resolved_numa_nodes = try resolveDmaNumaNodes(
        allocator,
        platform,
        opts.device_numa_nodes,
    );
    defer allocator.free(resolved_numa_nodes);

    const device_warmup_started = std.Io.Timestamp.now(io, .awake);
    try warmupDmaBenchmarkDeviceAllocators(io, platform);
    const device_allocator_warmup_ns: u64 = @intCast(@max(
        device_warmup_started.untilNow(io, .awake).nanoseconds,
        0,
    ));
    var source_pools: DmaBenchmarkSourcePools = try .init(
        allocator,
        io,
        platform,
        resolved_numa_nodes,
        opts.max_mapped_bytes,
    );
    var source_pools_active = true;
    defer if (source_pools_active) source_pools.deinit();
    const calibration_source_bytes = std.math.mul(
        usize,
        maximum_feasible_block_size,
        opts.block_parallelism,
    ) catch return error.DmaBenchmarkPinnedBudgetExceeded;
    try source_pools.growPool(
        source_pools.device_pool_indices[0],
        calibration_source_bytes,
    );

    var samples: std.ArrayListUnmanaged(DmaBenchmarkSample) = .empty;
    errdefer samples.deinit(allocator);
    var session: ReusableDmaBenchmarkSession = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .samples = &samples,
    };
    var session_active = true;
    defer if (session_active) session.deinit(&source_pools);

    const representative = try tuneDmaBenchmarkDevice(
        &session,
        opts,
        &source_pools,
        0,
    );

    const uniform_block_size = representative.dma_block_size;
    const uniform_parallelism = representative.dma_parallelism;
    const calibrated_node_reserves = try allocator.alloc(usize, source_pools.pools.len);
    defer allocator.free(calibrated_node_reserves);
    @memset(calibrated_node_reserves, 0);
    for (platform.devices, 0..) |_, device_index| {
        const pool_index = source_pools.device_pool_indices[device_index];
        calibrated_node_reserves[pool_index] = try std.math.add(
            usize,
            calibrated_node_reserves[pool_index],
            uniform_parallelism,
        );
    }

    try source_pools.ensureLoadBlockReserves(
        uniform_block_size,
        calibrated_node_reserves,
    );
    const owned_samples = try samples.toOwnedSlice(allocator);
    errdefer allocator.free(owned_samples);
    const resources = try DmaPlatformSettings.adopt(
        allocator,
        platform,
        .{
            .device_numa_nodes = resolved_numa_nodes,
            .block_size = uniform_block_size,
            .max_in_flight_per_device = uniform_parallelism,
            .max_mapped_bytes = opts.max_mapped_bytes,
        },
        source_pools,
    );
    source_pools_active = false;
    var result: DmaBenchmarkReport = .{
        .allocator = allocator,
        .resources = resources,
        .recommendation = representative,
        .samples = owned_samples,
        .elapsed_ns = 0,
    };
    finishDmaBenchmarkReport(
        &result,
        io,
        benchmark_started,
        device_allocator_warmup_ns,
        &session,
        &source_pools,
    );
    session_active = false;
    source_pools_active = false;
    return result;
}

fn logDmaBenchmarkReport(platform: *const Platform, result: *const DmaBenchmarkReport) void {
    log.info("dma_bench version=12 synthetic=true numa_pools={d} retained_mapped_bytes={d} platform={s} devices={d} elapsed_ms={d:.3} calibration_ms={d:.3} allocator_warmup_ms={d:.3} source_registration_ms={d:.3} benchmark_setup_ms={d:.3} sampling_ms={d:.3} benchmark_overhead_ms={d:.3} windows={d}", .{
        result.resources.numaPoolCount(),
        result.resources.retainedMappedBytes(),
        @tagName(platform.target),
        platform.devices.len,
        @as(f64, @floatFromInt(result.elapsed_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.calibration_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.device_allocator_warmup_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.source_registration_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.benchmark_setup_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.sampling_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.benchmark_overhead_ns)) / std.time.ns_per_ms,
        result.windows,
    });
    for (platform.devices, 0..) |device, device_index| {
        log.info("dma_bench_numa device_index={d} device_id={d} numa_node={?d}", .{
            device_index,
            device.id(),
            result.resources.config.device_numa_nodes[device_index],
        });
    }
    const recommendation = result.recommendation;
    const representative_device = platform.devices[recommendation.device_index];
    log.info("dma_bench_device device_index={d} device_id={d} kind=\"{s}\" block_bytes={d} parallelism={d} measured_gib_s={d:.3} average_latency_ms={d:.3} windows={d}", .{
        recommendation.device_index,
        recommendation.device_id,
        representative_device.kind(),
        recommendation.dma_block_size,
        recommendation.dma_parallelism,
        recommendation.measured_bytes_per_second / (1024 * 1024 * 1024),
        recommendation.average_latency_ns / std.time.ns_per_ms,
        recommendation.windows,
    });
    for (result.samples) |sample| {
        const device = platform.devices[sample.device_index];
        log.info("dma_bench_sample phase={s} device_index={d} device_id={d} block_bytes={d} parallelism={d} repeat={d} bytes={d} transfers={d} elapsed_ns={d} gib_s={d:.3} average_latency_ms={d:.3}", .{
            @tagName(sample.phase),
            sample.device_index,
            device.id(),
            sample.block_size,
            sample.parallelism,
            sample.repeat,
            sample.bytes,
            sample.transfers,
            sample.elapsed_ns,
            sample.bytesPerSecond() / (1024 * 1024 * 1024),
            sample.averageLatencyNs() / std.time.ns_per_ms,
        });
    }
}

/// Measures one representative device, prepares all-device workspace, and
/// atomically replaces the platform's private settings. The previous/default
/// settings remain active on failure.
pub fn benchTransfer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *Platform,
    opts: BenchTransferOptions,
) !void {
    if (platform.target == .cpu) return;

    try beginPlatformDmaOperation(platform, dma_platform_calibrating);
    defer endPlatformDmaOperation(platform, dma_platform_calibrating);

    var result = try benchmarkSyntheticDma(allocator, io, platform, opts);
    errdefer result.deinit();
    logDmaBenchmarkReport(platform, &result);

    const replacement = try allocator.create(DmaPlatformSettings);
    replacement.* = result.resources;
    result.deinitReport();

    const old = platform._dma.settings.swap(replacement, .acq_rel);
    if (old) |ptr| destroyDmaPlatformSettings(dmaSettingsFromOpaque(ptr));
}

test "DMA benchmark completion target has no time cap" {
    try std.testing.expect(!dmaBenchmarkWindowComplete(9, 10, 128, 128));
    try std.testing.expect(!dmaBenchmarkWindowComplete(10, 10, 127, 128));
    try std.testing.expect(dmaBenchmarkWindowComplete(10, 10, 128, 128));
    try std.testing.expect(dmaBenchmarkWindowComplete(1_000, 10, 128, 128));
    try std.testing.expect(dmaBenchmarkWindowComplete(10, 10, 0, 0));
}

test "DMA benchmark selection uses medians and prefers the smallest near-peak value" {
    const allocator = std.testing.allocator;
    var candidates = [_]DmaBenchmarkCandidate{
        .{ .value = 2, .cohort = undefined },
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
    };
    const rates = [_][3]u64{
        .{ 60, 10, 62 },
        .{ 98, 99, 97 },
        .{ 100, 101, 99 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| {
            candidate.appendMetric(.{
                .bytes = rate,
                .transfers = 1,
                .total_latency_ns = 1,
                .elapsed_ns = std.time.ns_per_s,
            });
        }
    }

    const decision = try selectDmaBenchmarkCandidate(allocator, &candidates, 0.05);
    try std.testing.expectEqual(@as(usize, 1), decision.index);
    try std.testing.expectEqual(@as(f64, 98), decision.metrics.bytesPerSecond());
}

test "DMA benchmark selection does not assume a unimodal response" {
    const allocator = std.testing.allocator;
    var candidates = [_]DmaBenchmarkCandidate{
        .{ .value = 2, .cohort = undefined },
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
        .{ .value = 16, .cohort = undefined },
    };
    const rates = [_]u64{ 80, 100, 70, 99 };
    for (&candidates, rates) |*candidate, rate| {
        candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .total_latency_ns = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    const decision = try selectDmaBenchmarkCandidate(allocator, &candidates, 0.02);
    try std.testing.expectEqual(@as(usize, 1), decision.index);
}

test "DMA benchmark tuple feasibility rejects pinned budget overflow" {
    try std.testing.expect(dmaBenchmarkTupleFeasible(128, 16, 8));
    try std.testing.expect(!dmaBenchmarkTupleFeasible(127, 16, 8));
    try std.testing.expect(!dmaBenchmarkTupleFeasible(std.math.maxInt(usize), std.math.maxInt(usize), 2));
}

test "DMA benchmark confirms a candidate when round qualification disagrees" {
    var candidates = [_]DmaBenchmarkCandidate{
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
    };
    const rates = [_][3]u64{
        .{ 96, 80, 97 },
        .{ 100, 100, 100 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| candidate.appendMetric(.{
            .bytes = rate,
            .transfers = 1,
            .total_latency_ns = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    try std.testing.expect(dmaBenchmarkCandidateNeedsConfirmation(
        &candidates,
        0,
        1,
        0.05,
        0.02,
    ));
}
