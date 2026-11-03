//! Loader-owned host arenas, cut into blocks once. The direct backend owns
//! its workspace for its whole life: it grows the set to what its load holds
//! and carves it, and its `destroy` releases every arena. Callers never
//! supply a workspace or a budget.
//!
//! The memory depends on the target: CUDA and oneAPI take page-backed arenas
//! registered with PJRT (DmaMap), ROCm and Furiosa PJRT `pinned_host`
//! buffers, and CPU plain pages, unregistered. The arenas' kind and the
//! transfer path are separate decisions: CPU loads through the direct
//! backend too. ROCm balances whole arenas across its devices' host nodes.
//! `Workspace.mapped_bytes_ceiling` (16 GiB) is a safety guard, not a
//! target. A failed allocation or registration, or a growth past the
//! ceiling, is logged with its cause and returns `OutOfMemory`.

const std = @import("std");
const Alignment = std.mem.Alignment;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const Platform = @import("../platform.zig").Platform;
const log = @import("log.zig").mem;

/// One ROCm host-memory allocation path and its bytes allocated so far.
const HostNode = struct {
    /// The plugin's `numa_node` device attribute (`Device.numaNode`); null
    /// for the one path kept when no device reports it.
    node: ?u32,
    any_device_index: usize,
    total_allocated_bytes: usize = 0,
};

/// One retained arena. The bytes of a `pinned_host` arena belong to its
/// PJRT buffer; page-backed arenas have none.
const Arena = struct {
    bytes: []u8,
    pjrt_buffer: ?*pjrt.Buffer = null,
};

/// How arenas are mapped and unmapped; each strategy keeps only the state
/// it uses.
const ArenaKind = union(enum) {
    /// With eight MI300X visible, hipHostRegister took ~6.3 s for 1 GiB
    /// versus 0.6-0.7 s through hipHostMalloc. KFD/IOMMU registration maps
    /// pages to every GPU; huge-page advice helped only ~9%, and touching
    /// pages or changing HIP flags did not remove the cost. Standard PJRT
    /// pinned_host buffers provide allocation-owned memory without a custom
    /// allocator extension.
    pjrt_host: struct {
        platform: *const Platform,
        host_nodes: std.ArrayList(HostNode),
    },
    /// Page-backed arenas, registered with the platform's PJRT client when
    /// there is one (`dma_map` in the arena log) and left pageable for CPU
    /// transfers when not (`pageable`), with huge-page alignment and advice.
    /// Registration only avoids staging if the plugin recognizes the range as
    /// pinned, including subranges. The oneAPI plugin must implement
    /// IsHostMemoryPinned for SYCL imports: without it, one B70 staged
    /// transfers at 11.01 GiB/s instead of 26.90 (eight DMA caps).
    pages: struct {
        platform: ?*const Platform,
    },

    const transparent_huge_page_size = 2 * 1024 * 1024;

    fn init(allocator: Allocator, platform: *const Platform) Allocator.Error!ArenaKind {
        return switch (platform.target) {
            .cuda, .oneapi => .{ .pages = .{ .platform = platform } },
            .cpu => .{ .pages = .{ .platform = null } },
            // Furiosa backs `.host_pinned` with SDK pinned (dma-buf) memory
            // from `/dev/dma_heap/system`, which the SDK recognizes by
            // address and transfers without pinning pages on every request.
            .rocm, .furiosa => {
                // PJRT-pinned allocations choose placement through the device's
                // host memory space. Retain one allocation path per reported
                // node; the blocks stay unpartitioned.
                var host_nodes: std.ArrayList(HostNode) = try .initCapacity(allocator, @max(platform.devices.len, 1));
                devices: for (platform.devices, 0..) |device, device_index| {
                    std.debug.assert(device.memory(.host_pinned) != null);
                    const node = device.numaNode() orelse continue;
                    for (host_nodes.items) |existing| {
                        if (existing.node == node) continue :devices;
                    }
                    host_nodes.appendAssumeCapacity(.{ .node = node, .any_device_index = device_index });
                }
                if (host_nodes.items.len == 0) host_nodes.appendAssumeCapacity(.{ .node = null, .any_device_index = 0 });
                return .{ .pjrt_host = .{ .platform = platform, .host_nodes = host_nodes } };
            },
            .tpu, .neuron, .metal => @panic("DMA is not accessible on this platform and shouldn't have been called."),
        };
    }

    /// Call after every arena is unmapped.
    fn deinit(self: *ArenaKind, allocator: Allocator) void {
        switch (self.*) {
            .pjrt_host => |*host| host.host_nodes.deinit(allocator),
            .pages => {},
        }
    }

    /// Maps one arena of `bytes`.
    fn map(self: *ArenaKind, allocator: Allocator, io: std.Io, bytes: usize) Allocator.Error!Arena {
        const started: std.Io.Timestamp = .now(io, .awake);
        switch (self.*) {
            .pjrt_host => |*host| {
                // Balance bytes, not arena/device counts: arenas differ in size.
                // Replicated Llama-3.1-8B on eight MI300X lost the fast mode when
                // all arenas came from device zero (1.27-1.45 s). Device rotation
                // left a 61/39 byte split and took 0.98-1.33 s; byte-balanced
                // allocation recovered 0.84-1.05 s over six runs, with 1.52 GiB
                // mapped versus 2.03 GiB for per-node pools. Host contention
                // affected both arms.
                var host_node = &host.host_nodes.items[0];
                for (host.host_nodes.items[1..]) |*candidate| {
                    if (candidate.total_allocated_bytes < host_node.total_allocated_bytes) host_node = candidate;
                }
                const device_index = host_node.any_device_index;
                // PJRT is the sole owner of these hipHostMalloc-backed bytes.
                // Never DmaMap or DmaUnmap the borrowed pointer: unregistering
                // allocation-owned memory breaks the eventual hipHostFree. The
                // buffer and its external reference both live until `unmap`,
                // after all reads and DMA finish.
                // The ROCm plugin must implement IsHostMemoryPinned: without
                // it, one MI300X staged transfers at 6.5 GiB/s instead of 46.6.
                const api = host.platform.pjrt_api;
                const memory = host.platform.devices[device_index].memory(.host_pinned).?;
                const buffer = host.platform.pjrt_client.createUninitializedBuffer(api, .{
                    .dims = &.{@intCast(bytes)},
                    .element_type = .u8,
                    .layout = .{
                        .tiled = .{
                            .minor_to_major = &.{0},
                            .tile_dims = &.{},
                            .tile_dims_sizes = &.{},
                        },
                    },
                    .dst = .{ .memory = memory.pjrt_memory },
                }) catch |err| {
                    log.err("pinned host allocation failed on device {d} for {Bi:.2}: {s}", .{ device_index, bytes, @errorName(err) });
                    return error.OutOfMemory;
                };
                errdefer buffer.deinit(api);
                std.debug.assert(buffer.isOnCpu(api));
                buffer.increaseExternalReferenceCount(api) catch |err| {
                    log.err("retaining pinned host allocation failed on device {d} for {Bi:.2}: {s}", .{ device_index, bytes, @errorName(err) });
                    return error.OutOfMemory;
                };
                errdefer buffer.decreaseExternalReferenceCount(api) catch {};
                const ptr: [*]u8 = @ptrCast(buffer.opaqueDeviceMemoryDataPointer(api) catch |err| {
                    log.err("accessing pinned host allocation failed on device {d} for {Bi:.2}: {s}", .{ device_index, bytes, @errorName(err) });
                    return error.OutOfMemory;
                });
                // Counted once mapped: the workspace retains every mapped arena.
                host_node.total_allocated_bytes += bytes;
                log.debug("DMA arena kind=pjrt_host device={d} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                    device_index,
                    @intFromPtr(ptr),
                    bytes,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, .now(io, .awake)))) / std.time.ns_per_ms,
                });
                return .{ .bytes = ptr[0..bytes], .pjrt_buffer = buffer };
            },
            .pages => |*pages| {
                const alignment = pageAlignment(bytes);
                const ptr = allocator.rawAlloc(bytes, alignment, @returnAddress()) orelse {
                    log.err("host memory allocation failed for {Bi:.2}: OutOfMemory", .{bytes});
                    return error.OutOfMemory;
                };
                const data = ptr[0..bytes];
                adviseHugePages(data);
                if (pages.platform) |p| p.pjrt_client.dmaMap(p.pjrt_api, data) catch |err| {
                    log.err("DMA registration failed for {Bi:.2} of host memory: {s}", .{ bytes, @errorName(err) });
                    allocator.rawFree(data, alignment, @returnAddress());
                    return error.OutOfMemory;
                };
                const finished: std.Io.Timestamp = .now(io, .awake);
                log.debug("DMA arena kind={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                    if (pages.platform != null) "dma_map" else "pageable",
                    @intFromPtr(data.ptr),
                    bytes,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, finished))) / std.time.ns_per_ms,
                });
                return .{ .bytes = data };
            },
        }
    }

    fn unmap(self: *const ArenaKind, allocator: Allocator, arena: Arena) void {
        switch (self.*) {
            .pjrt_host => |host| {
                const api = host.platform.pjrt_api;
                const buffer = arena.pjrt_buffer.?;
                buffer.decreaseExternalReferenceCount(api) catch unreachable;
                buffer.deinit(api);
            },
            .pages => |pages| {
                if (pages.platform) |platform| {
                    // Unregistering can fail after a device error. The pages may
                    // still be registered, so they are leaked, not reused.
                    platform.pjrt_client.dmaUnmap(platform.pjrt_api, arena.bytes) catch |err| {
                        log.warn("DMA unregistration failed for {Bi:.2} of host memory, which stays allocated: {s}", .{ arena.bytes.len, @errorName(err) });
                        return;
                    };
                }
                allocator.rawFree(arena.bytes, pageAlignment(arena.bytes.len), @returnAddress());
            },
        }
    }

    /// Page alignment, raised to the huge-page size for arenas that span one.
    fn pageAlignment(len: usize) Alignment {
        const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
        if (comptime builtin.os.tag != .linux) return alignment;
        if (len < transparent_huge_page_size) return alignment;
        return alignment.max(.fromByteUnits(transparent_huge_page_size));
    }

    fn adviseHugePages(data: []u8) void {
        if (comptime builtin.os.tag != .linux) return;
        if (data.len < transparent_huge_page_size) return;

        const ptr: [*]align(std.heap.page_size_min) u8 = @alignCast(data.ptr);
        std.posix.madvise(ptr, data.len, std.posix.MADV.HUGEPAGE) catch |err| {
            log.warn("MADV_HUGEPAGE failed for DMA buffer at 0x{x} ({Bi:.2}): {s}", .{
                @intFromPtr(data.ptr),
                data.len,
                @errorName(err),
            });
        };
    }
};

/// Host arenas owned by one direct loader. Arenas are mapped one at a time
/// (`allocate`); the loader then grows the set to what its load holds and
/// cuts it into blocks, once (`carve`). Deinitialize after all transfers
/// finish and before the platform.
pub const Workspace = struct {
    /// Safety guard on the arenas' total host memory (pinned on the DMA
    /// targets), not an allocation target: the load's pre-growth decides
    /// what is actually mapped. Fixed, because no caller ever had a reason
    /// to choose another value.
    pub const mapped_bytes_ceiling: usize = 16 * 1024 * 1024 * 1024;

    allocator: Allocator,
    io: std.Io,
    kind: ArenaKind,
    /// Every arena mapped, in mapping order, until `deinit`.
    arenas: std.ArrayList(Arena) = .empty,
    /// The arenas cut into blocks, set by `carve`: nothing maps after it.
    blocks: ?[]Block = null,

    pub fn init(
        allocator: Allocator,
        io: std.Io,
        platform: *const Platform,
    ) Allocator.Error!Workspace {
        return .{
            .allocator = allocator,
            .io = io,
            .kind = try .init(allocator, platform),
        };
    }

    pub fn deinit(self: *Workspace) void {
        const io = self.io;
        const mapped_bytes = self.mappedBytes();
        const started: std.Io.Timestamp = .now(io, .awake);
        if (self.blocks) |blocks| self.allocator.free(blocks);
        for (self.arenas.items) |arena| self.kind.unmap(self.allocator, arena);
        self.arenas.deinit(self.allocator);
        self.kind.deinit(self.allocator);
        const elapsed_ns = elapsedNanoseconds(started, .now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }

    /// The bytes of every retained arena, tails included.
    pub fn mappedBytes(self: *const Workspace) usize {
        var mapped: usize = 0;
        for (self.arenas.items) |arena| mapped += arena.bytes.len;
        return mapped;
    }

    /// Retains one new arena within the mapped-byte ceiling, before `carve`.
    pub fn allocate(self: *Workspace, bytes: usize) Allocator.Error![]u8 {
        std.debug.assert(self.blocks == null);
        const mapped = self.mappedBytes();
        if (bytes > mapped_bytes_ceiling - mapped) {
            log.err("host memory allocation exceeds mapped ceiling: requested={Bi:.2}, mapped={Bi:.2}, ceiling={Bi:.2}: OutOfMemory", .{ bytes, mapped, mapped_bytes_ceiling });
            return error.OutOfMemory;
        }
        // Reserved first, so a mapped arena is always retained: the ROCm
        // balancer counts the bytes it maps.
        try self.arenas.ensureUnusedCapacity(self.allocator, 1);
        const arena = try self.kind.map(self.allocator, self.io, bytes);
        self.arenas.appendAssumeCapacity(arena);
        return arena.bytes;
    }

    /// The blocks of `block_size` the workspace can hold: those of its
    /// retained arenas, and those the mapped ceiling still allows.
    pub fn capacity(self: *const Workspace, block_size: usize) usize {
        return self.usableBlocks(block_size) + (mapped_bytes_ceiling - self.mappedBytes()) / block_size;
    }

    /// Maps the blocks missing below `first_blocks` as one arena, then those
    /// missing below `total_blocks` as another, and cuts every retained arena
    /// into blocks of `block_size`, once. The set is then complete: nothing
    /// maps inside a load (a slab mapped there cost 146 to 230 ms of
    /// hipHostMalloc on MI300X), and `allocate` asserts it. Arena tails
    /// smaller than one block stay mapped and unused. Blocks are not split
    /// per NUMA node: strict per-node pools duplicated source reserves and
    /// constrained reads to the smallest node; placement belongs to arena
    /// allocation. The workspace owns the list and the memory.
    pub fn carve(self: *Workspace, block_size: usize, first_blocks: usize, total_blocks: usize) Allocator.Error![]const Block {
        std.debug.assert(self.blocks == null and block_size != 0);
        for ([_]usize{ first_blocks, total_blocks }) |target_blocks| {
            const missing_blocks = target_blocks -| self.usableBlocks(block_size);
            if (missing_blocks != 0) _ = try self.allocate(missing_blocks * block_size);
        }
        const blocks = try self.allocator.alloc(Block, self.usableBlocks(block_size));
        var next: usize = 0;
        for (self.arenas.items) |arena| {
            for (0..arena.bytes.len / block_size) |j| {
                blocks[next] = arena.bytes[j * block_size ..][0..block_size];
                next += 1;
            }
        }
        std.debug.assert(next == blocks.len);
        self.blocks = blocks;
        return blocks;
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    fn usableBlocks(self: *const Workspace, block_size: usize) usize {
        var usable: usize = 0;
        for (self.arenas.items) |arena| usable += arena.bytes.len / block_size;
        return usable;
    }

    /// Creates ordinary allocator-backed arenas for tests without a PJRT platform.
    fn initForTesting(allocator: Allocator, io: std.Io) Workspace {
        return .{
            .allocator = allocator,
            .io = io,
            .kind = .{ .pages = .{ .platform = null } },
        };
    }
};

pub const Block = []u8;

fn elapsedNanoseconds(started: std.Io.Timestamp, finished: std.Io.Timestamp) u64 {
    return @intCast(@max(started.durationTo(finished).nanoseconds, 0));
}

test "Workspace capacity counts whole blocks under the mapped ceiling, and carve refuses one more" {
    const allocator = std.testing.allocator;
    var workspace: Workspace = .initForTesting(allocator, std.testing.io);
    defer workspace.deinit();
    const ceiling = Workspace.mapped_bytes_ceiling;

    try std.testing.expectEqual(ceiling / 64, workspace.capacity(64));
    _ = try workspace.allocate(150);
    _ = try workspace.allocate(127);
    // Whole blocks only: the arenas hold three, their tails none.
    try std.testing.expectEqual(3 + (ceiling - 277) / 64, workspace.capacity(64));
    // Capacity is tight, as direct `create` sizes its carve from it: one
    // block more crosses the ceiling (reaching it would map 16 GiB).
    try std.testing.expectError(error.OutOfMemory, workspace.allocate(ceiling - 277 + 1));
    try std.testing.expectError(error.OutOfMemory, workspace.carve(64, workspace.capacity(64) + 1, 0));
    try std.testing.expectEqual(@as(usize, 277), workspace.mappedBytes());
}

test "Workspace carve maps each step's missing blocks as one arena and leaves tails out" {
    const allocator = std.testing.allocator;
    var workspace: Workspace = .initForTesting(allocator, std.testing.io);
    defer workspace.deinit();
    const first = try workspace.allocate(150);
    const second = try workspace.allocate(127);

    // Three blocks are mapped: two more make the first five, three more the
    // eight.
    const blocks = try workspace.carve(64, 5, 8);
    try std.testing.expectEqual(@as(usize, 8), blocks.len);
    try std.testing.expectEqual(@as(usize, 4), workspace.arenas.items.len);
    try std.testing.expectEqual(@as(usize, 2 * 64), workspace.arenas.items[2].bytes.len);
    try std.testing.expectEqual(@as(usize, 3 * 64), workspace.arenas.items[3].bytes.len);
    for (blocks) |block| try std.testing.expectEqual(@as(usize, 64), block.len);
    // Tails stay mapped without contributing a block.
    try std.testing.expectEqual(first.ptr, blocks[0].ptr);
    try std.testing.expectEqual(first.ptr + 64, blocks[1].ptr);
    try std.testing.expectEqual(second.ptr, blocks[2].ptr);
}

test "Workspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: Allocator) !void {
            var workspace: Workspace = .initForTesting(allocator, std.testing.io);
            defer workspace.deinit();

            _ = try workspace.allocate(64);
            _ = try workspace.allocate(128);
            // The first step is already mapped; the second maps the last
            // block.
            const blocks = try workspace.carve(64, 3, 4);
            try std.testing.expectEqual(@as(usize, 4), blocks.len);
            try std.testing.expectEqual(@as(usize, 256), workspace.mappedBytes());
            try std.testing.expectEqual(@as(usize, 3), workspace.arenas.items.len);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}
