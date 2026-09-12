//! Loader-owned host arenas and per-load block leases.

const std = @import("std");
const Alignment = std.mem.Alignment;
const Allocator = std.mem.Allocator;
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const Device = @import("../platform.zig").Device;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const log = @import("log.zig").mem;

pub const AcquireError = error{ Closed, RequestExceedsCapacity };

/// One ROCm host-memory allocation path and its bytes allocated so far.
const HostNode = struct {
    any_device_index: usize,
    total_allocated_bytes: usize = 0,
};

/// Each allocation strategy keeps only the state it uses.
const Backend = union(enum) {
    pjrt_host: PjrtHost,
    /// Page-backed arenas, registered with the platform's PJRT client when
    /// the allocator has one (`dma_map` in the arena log) and left pageable
    /// for CPU transfers when it does not (`pageable`).
    pages: Pages,

    /// With eight MI300X visible, hipHostRegister took ~6.3 s for 1 GiB
    /// versus 0.6-0.7 s through hipHostMalloc. KFD/IOMMU registration maps
    /// pages to every GPU; huge-page advice helped only ~9%, and touching
    /// pages or changing HIP flags did not remove the cost. Standard PJRT
    /// pinned_host buffers provide allocation-owned memory without a custom
    /// allocator extension.
    const PjrtHost = struct {
        platform: *const Platform,
        host_nodes: []HostNode,
        allocations: std.ArrayListUnmanaged(PinnedHostAllocation) = .empty,
    };

    /// Registration only avoids staging if the plugin recognizes the range as
    /// pinned, including subranges. Older oneAPI plugins treated SYCL imports
    /// as unknown; checking both ends against the same imported base removed
    /// the userspace copy. On one B70, DMA caps two/eight then measured
    /// 26.86/26.90 GiB/s instead of 21.17/11.01. More DMA credits had amplified
    /// staging, not improved DMA.
    const Pages = struct {
        allocator: HugePageAllocator,
        allocations: std.ArrayListUnmanaged([]align(std.heap.page_size_min) u8) = .empty,
    };

    fn init(
        allocator: Allocator,
        io: std.Io,
        platform: *const Platform,
    ) Allocator.Error!Backend {
        // Interleave memory-bearing nodes, not just device-associated nodes.
        // On four GB300, per-device local H2D was ~176-184 GiB/s versus ~110
        // remote, yet strict locality could put page-cache copies and DMA on
        // the same busy memory controller. With one shared interleaved pool, warm
        // replicated DeepSeek-V4-Flash took 4.35 s against 4.92 s with node-local
        // pools (three runs each, 16 MiB blocks, depth eight). Across one,
        // two and four GB300, interleave was never worst; every single-node
        // choice was worst somewhere.
        // Unplaced is not reliably neutral either: CUDA registration applied
        // a preferred-node policy based on the calling thread. These results
        // favor a knowledge-free default, not a universal locality rule.
        // Interleaving over one node is that node; leave it to the kernel.
        const numa_mask = interleaveMask(memoryNodeMask(allocator, io));
        return switch (platform.target) {
            .cuda, .oneapi => .{ .pages = .{
                .allocator = .init(allocator, platform, numa_mask),
            } },
            .cpu => .{ .pages = .{
                .allocator = .init(allocator, null, numa_mask),
            } },
            .rocm => {
                // PJRT-pinned allocations choose placement through the device's
                // host memory space, not our mbind policy. Retain one allocation
                // path per reported node; the common block pool stays unpartitioned.
                var discovered_nodes: std.ArrayListUnmanaged(struct {
                    device_index: usize,
                    node: usize,
                }) = .empty;
                defer discovered_nodes.deinit(allocator);
                devices: for (platform.devices, 0..) |device, device_index| {
                    std.debug.assert(device.memory(.host_pinned) != null);
                    const node = device.numaNode() orelse continue;
                    for (discovered_nodes.items) |existing| {
                        if (existing.node == node) continue :devices;
                    }
                    try discovered_nodes.append(allocator, .{ .device_index = device_index, .node = node });
                }
                const host_nodes = try allocator.alloc(HostNode, @max(discovered_nodes.items.len, 1));
                errdefer allocator.free(host_nodes);
                if (discovered_nodes.items.len == 0) {
                    host_nodes[0] = .{ .any_device_index = 0 };
                } else {
                    for (host_nodes, discovered_nodes.items) |*host_node, discovered| {
                        host_node.* = .{ .any_device_index = discovered.device_index };
                    }
                }
                return .{ .pjrt_host = .{
                    .platform = platform,
                    .host_nodes = host_nodes,
                } };
            },
            .tpu, .neuron, .metal => @panic("DMA is not accessible on this platform and shouldn't have been called."),
        };
    }

    fn deinit(self: *Backend, allocator: Allocator) void {
        switch (self.*) {
            .pjrt_host => |*host| {
                for (host.allocations.items) |allocation| allocation.destroy();
                host.allocations.deinit(allocator);
                allocator.free(host.host_nodes);
            },
            .pages => |*pages| {
                for (pages.allocations.items) |allocation| pages.allocator.free(allocation);
                pages.allocations.deinit(allocator);
            },
        }
    }

    fn arenaCount(self: *const Backend) usize {
        return switch (self.*) {
            .pjrt_host => |host| host.allocations.items.len,
            .pages => |pages| pages.allocations.items.len,
        };
    }

    fn arenaAt(self: *const Backend, index: usize) []u8 {
        return switch (self.*) {
            .pjrt_host => |host| host.allocations.items[index].data,
            .pages => |pages| pages.allocations.items[index],
        };
    }

    fn allocate(
        self: *Backend,
        allocator: Allocator,
        io: std.Io,
        bytes: usize,
    ) Allocator.Error![]u8 {
        switch (self.*) {
            .pjrt_host => |*host| {
                const started: std.Io.Timestamp = .now(io, .awake);
                // Balance bytes, not arena/device counts: arenas differ in size.
                // Replicated Llama-3.1-8B on eight MI300X lost the fast mode when
                // all arenas came from device zero (1.27-1.45 s). Device rotation
                // left a 61/39 byte split and took 0.98-1.33 s; byte-balanced
                // allocation recovered 0.84-1.05 s over six runs, with 1.52 GiB
                // mapped versus 2.03 GiB for the former per-node pools. Host
                // contention affected both arms.
                var host_node_index: usize = 0;
                for (host.host_nodes[1..], 1..) |host_node, index| {
                    if (host_node.total_allocated_bytes < host.host_nodes[host_node_index].total_allocated_bytes)
                        host_node_index = index;
                }
                const any_device_index = host.host_nodes[host_node_index].any_device_index;
                const memory = host.platform.devices[any_device_index].memory(.host_pinned).?;
                const allocation: PinnedHostAllocation = try .create(memory, any_device_index, bytes);
                errdefer allocation.destroy();
                try host.allocations.append(allocator, allocation);
                // Only retained allocations count: failed allocation/publication
                // must not bias the next node choice with nonexistent bytes.
                host.host_nodes[host_node_index].total_allocated_bytes += allocation.data.len;
                log.info("DMA arena kind=pjrt_host device={d} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                    any_device_index,
                    @intFromPtr(allocation.data.ptr),
                    allocation.data.len,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, .now(io, .awake)))) / std.time.ns_per_ms,
                });
                return allocation.data;
            },
            .pages => |*pages| {
                const started: std.Io.Timestamp = .now(io, .awake);
                const allocation = try pages.allocator.alloc(bytes);
                errdefer pages.allocator.free(allocation);
                try pages.allocations.append(allocator, allocation);
                const finished: std.Io.Timestamp = .now(io, .awake);
                const numa_mask = pages.allocator.numa_mask;
                const placement = if (numa_mask == 0) "unplaced" else "interleave";
                const kind = if (pages.allocator.platform != null) "dma_map" else "pageable";
                log.info("DMA arena kind={s} placement={s} nodes=0x{x} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                    kind,
                    placement,
                    numa_mask,
                    @intFromPtr(allocation.ptr),
                    allocation.len,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, finished))) / std.time.ns_per_ms,
                });
                return allocation;
            },
        }
    }

    /// Private owner of page-backed arena allocation policy. It applies
    /// huge-page alignment/advice and optionally registers the pages with
    /// the selected backend's PJRT client.
    const HugePageAllocator = struct {
        const transparent_huge_page_size = 2 * 1024 * 1024;
        const mpol_interleave = 3;

        parent: Allocator,
        /// Null leaves the pages pageable for CPU transfers.
        platform: ?*const Platform,
        /// Zero leaves placement to the kernel. Automatic placement can
        /// fall back to zero if the kernel refuses it.
        numa_mask: u64,

        fn init(parent: Allocator, platform: ?*const Platform, numa_mask: u64) HugePageAllocator {
            return .{
                .parent = parent,
                .platform = platform,
                .numa_mask = numa_mask,
            };
        }

        fn alloc(self: *HugePageAllocator, len: usize) Allocator.Error![]align(std.heap.page_size_min) u8 {
            const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
            const effective_alignment = effectiveAlignment(alignment, len);
            const ptr = self.parent.rawAlloc(len, effective_alignment, @returnAddress()) orelse
                return error.OutOfMemory;
            const data: []align(std.heap.page_size_min) u8 = @alignCast(ptr[0..len]);
            self.place(data);
            adviseHugePages(data);
            if (self.platform) |p| p.pjrt_client.dmaMap(p.pjrt_api, data) catch |err| {
                log.err("DMA registration failed for {Bi:.2} of host memory: {s}", .{ len, @errorName(err) });
                self.parent.rawFree(data, effective_alignment, @returnAddress());
                return error.OutOfMemory;
            };
            return data;
        }

        fn place(self: *HugePageAllocator, data: []u8) void {
            const mask = self.numa_mask;
            if (mask == 0) return;
            if (comptime builtin.os.tag != .linux) return;

            const node_mask: [1]u64 = .{mask};
            const highest_node: usize = 63 - @clz(mask);
            const rc = std.os.linux.syscall6(
                .mbind,
                @intFromPtr(data.ptr),
                data.len,
                mpol_interleave,
                @intFromPtr(&node_mask),
                // Linux get_nodes() decrements maxnode before copying the mask;
                // raw callers include the same extra sentinel bit as libnuma.
                highest_node + 2,
                0,
            );
            if (std.os.linux.errno(rc) == .SUCCESS) return;
            log.warn("NUMA placement of DMA arenas over nodes 0x{x} refused ({s}); leaving them unplaced", .{
                mask,
                @tagName(std.os.linux.errno(rc)),
            });
            // Automatic placement is not retried: the arenas stay unplaced.
            self.numa_mask = 0;
        }

        fn free(self: *const HugePageAllocator, buf: []align(std.heap.page_size_min) u8) void {
            if (self.platform) |platform| {
                platform.pjrt_client.dmaUnmap(platform.pjrt_api, @ptrCast(buf)) catch unreachable;
            }
            const alignment: Alignment = comptime .fromByteUnits(std.heap.page_size_min);
            self.parent.rawFree(buf, effectiveAlignment(alignment, buf.len), @returnAddress());
        }

        fn effectiveAlignment(alignment: Alignment, len: usize) Alignment {
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
};

/// Host arenas owned by one direct loader. Calibration fills the initial
/// arena set during initialization; loading reuses and grows it. Deinitialize
/// after all transfers finish and before the platform.
pub const Workspace = struct {
    /// Safety guard on the arenas' total host memory (pinned on the DMA
    /// targets), not an allocation target: calibration and the load's
    /// pre-growth decide what is actually mapped. Fixed, because no caller
    /// ever had a reason to choose another value.
    pub const mapped_bytes_ceiling: usize = 16 * 1024 * 1024 * 1024;

    allocator: Allocator,
    io: std.Io,
    backend: Backend,
    /// `mapped_bytes_ceiling`, or a test's own ceiling.
    max_mapped_bytes: usize,
    mapped_bytes: usize = 0,

    pub fn init(
        allocator: Allocator,
        io: std.Io,
        platform: *const Platform,
    ) Allocator.Error!Workspace {
        if (platform.devices.len == 0 or platform.devices.len > 64) {
            log.err("host workspace requires 1..64 devices, got {d}: UnsupportedPlatform", .{platform.devices.len});
            return error.UnsupportedPlatform;
        }
        const device_kind = platform.devices[0].kind();
        for (platform.devices[1..]) |device| {
            if (!std.mem.eql(u8, device_kind, device.kind())) {
                log.err("host workspace requires homogeneous devices: {s} differs from {s}: UnsupportedPlatform", .{ device_kind, device.kind() });
                return error.UnsupportedPlatform;
            }
        }

        return .{
            .allocator = allocator,
            .io = io,
            .backend = try .init(allocator, io, platform),
            .max_mapped_bytes = mapped_bytes_ceiling,
        };
    }

    pub fn deinit(self: *Workspace) void {
        const io = self.io;
        const mapped_bytes = self.mapped_bytes;
        const started: std.Io.Timestamp = .now(io, .awake);
        self.backend.deinit(self.allocator);
        const elapsed_ns = elapsedNanoseconds(started, .now(io, .awake));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        self.* = undefined;
    }

    /// Borrows the newest retained arena that fits, when no arena growth is running.
    pub fn findArena(self: *const Workspace, minimum_len: usize) ?[]u8 {
        var index = self.backend.arenaCount();
        while (index != 0) {
            index -= 1;
            const arena = self.backend.arenaAt(index);
            if (arena.len >= minimum_len) return arena;
        }
        return null;
    }

    /// Retains one new arena within the mapped-byte ceiling.
    pub fn allocate(self: *Workspace, bytes: usize) Allocator.Error![]u8 {
        if (bytes > self.max_mapped_bytes - self.mapped_bytes) {
            log.err("host memory allocation exceeds mapped ceiling: requested={Bi:.2}, mapped={Bi:.2}, ceiling={Bi:.2}: OutOfMemory", .{ bytes, self.mapped_bytes, self.max_mapped_bytes });
            return error.OutOfMemory;
        }
        const allocation = try self.backend.allocate(self.allocator, self.io, bytes);
        self.mapped_bytes += allocation.len;
        return allocation;
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    pub fn usableBlocks(self: *const Workspace, block_size: usize) usize {
        var usable: usize = 0;
        for (0..self.backend.arenaCount()) |index| usable += self.backend.arenaAt(index).len / block_size;
        return usable;
    }

    /// Maps the blocks missing below `target_blocks` as one arena. Requires a
    /// nonzero block_size; the workspace has one coordinating owner.
    pub fn growToBlocks(self: *Workspace, block_size: usize, target_blocks: usize) Allocator.Error!void {
        const usable_blocks = self.usableBlocks(block_size);
        const missing_blocks = target_blocks -| usable_blocks;
        if (missing_blocks == 0) return;
        if (missing_blocks > (self.max_mapped_bytes - self.mapped_bytes) / block_size) {
            log.err("host memory growth exceeds mapped ceiling: block_size={Bi:.2}, target_blocks={d}, usable_blocks={d}, mapped={Bi:.2}, ceiling={Bi:.2}: OutOfMemory", .{ block_size, target_blocks, usable_blocks, self.mapped_bytes, self.max_mapped_bytes });
            return error.OutOfMemory;
        }

        _ = try self.allocate(missing_blocks * block_size);
    }

    /// Creates ordinary allocator-backed arenas for tests without a PJRT platform.
    pub fn initForTesting(
        allocator: Allocator,
        io: std.Io,
        max_mapped_bytes: usize,
    ) !Workspace {
        if (!builtin.is_test) @compileError("initForTesting is only available in tests");
        return .{
            .allocator = allocator,
            .io = io,
            .backend = .{ .pages = .{
                .allocator = .init(allocator, null, 0),
            } },
            .max_mapped_bytes = max_mapped_bytes,
        };
    }
};

/// Owns a workspace and leases fixed-size blocks carved from its arenas.
/// One free list shares retained capacity across every destination. Strict
/// per-node pools duplicated source reserves and constrained leases to the
/// smallest node; placement belongs to arena allocation, not block matching.
/// Replicas share a source block until every child transfer releases it.
/// Copying replicas to another CPU socket first was rejected on MI300X:
/// raw local/remote H2D both reached ~49-50 GiB/s per GPU, while a single
/// CPU thread copied across sockets at only 5-10 GiB/s.
pub const BlockPool = struct {
    pub const Block = []u8;

    pub const Lease = struct {
        pool: *BlockPool,
        io: std.Io,
        data: []u8,
        remaining: std.atomic.Value(usize),

        pub fn init(pool: *BlockPool, io: std.Io, block: Block, references: usize) Lease {
            std.debug.assert(references > 0);
            return .{
                .pool = pool,
                .io = io,
                .data = block,
                .remaining = .init(references),
            };
        }

        /// Completes one reference and returns whether this was the final one.
        pub fn complete(self: *Lease) bool {
            const previous = self.remaining.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous == 1) self.pool.release(self.io, self.data);
            return previous == 1;
        }
    };

    allocator: Allocator,
    workspace: Workspace,
    free_blocks: std.ArrayListUnmanaged(Block),
    block_size: usize,
    /// Blocks carved from the arenas, fixed for the pool's life.
    capacity: usize,
    in_use: usize = 0,
    high_water: usize = 0,
    closed: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    /// Builds the free list from every retained arena, once: the caller has
    /// already grown the workspace to the whole set the load may use, so the
    /// pool never maps anything again (a slab mapped inside a load cost 146
    /// to 230 ms of hipHostMalloc on MI300X). Arena tails smaller than one
    /// selected block remain mapped and unused.
    /// Consumes and invalidates workspace on success; on failure the caller
    /// retains ownership. Calibration borrows must have ended before this call.
    pub fn init(
        allocator: Allocator,
        workspace: Workspace,
        block_size: usize,
    ) Allocator.Error!BlockPool {
        std.debug.assert(block_size != 0);

        var capacity: usize = 0;
        var free_blocks: std.ArrayListUnmanaged(Block) = .empty;
        errdefer free_blocks.deinit(allocator);

        for (0..workspace.backend.arenaCount()) |i| {
            const arena = workspace.backend.arenaAt(i);
            const block_count = arena.len / block_size;
            // Leased blocks are absent from `free_blocks`, so the storage is
            // sized for the total capacity, not for what is free.
            try free_blocks.ensureTotalCapacity(allocator, capacity + block_count);
            for (0..block_count) |j| {
                free_blocks.appendAssumeCapacity(arena[j * block_size ..][0..block_size]);
            }
            capacity += block_count;
        }

        return .{
            .allocator = allocator,
            .workspace = workspace,
            .block_size = block_size,
            .free_blocks = free_blocks,
            .capacity = capacity,
        };
    }

    pub fn deinit(self: *BlockPool) void {
        std.debug.assert(self.in_use == 0 and self.free_blocks.items.len == self.capacity);
        self.free_blocks.deinit(self.allocator);
        self.workspace.deinit();
        self.* = undefined;
    }

    /// Leases `output.len` blocks atomically, waiting for releases when the
    /// free list is short. Allocates nothing, ever.
    pub fn acquireMany(self: *BlockPool, io: std.Io, output: []Block) AcquireError!void {
        if (output.len == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.closed) return error.Closed;
        if (output.len > self.capacity) return error.RequestExceedsCapacity;
        while (self.free_blocks.items.len < output.len) {
            self.condition.waitUncancelable(io, &self.mutex);
            if (self.closed) return error.Closed;
        }
        for (output) |*block| block.* = self.free_blocks.pop().?;
        self.in_use += output.len;
        self.high_water = @max(self.high_water, self.in_use);
    }

    pub fn releaseMany(self: *BlockPool, io: std.Io, blocks: []const Block) void {
        if (blocks.len == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(blocks.len <= self.in_use);
        for (blocks) |block| {
            std.debug.assert(block.len == self.block_size);
            self.free_blocks.appendAssumeCapacity(block);
        }
        self.in_use -= blocks.len;
        self.condition.broadcast(io);
    }

    pub fn release(self: *BlockPool, io: std.Io, block: Block) void {
        self.releaseMany(io, &.{block});
    }

    pub fn close(self: *BlockPool, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed = true;
        self.condition.broadcast(io);
    }
};

/// PJRT is the sole owner of these hipHostMalloc-backed bytes. Never DmaMap
/// or DmaUnmap the borrowed pointer: unregistering allocation-owned memory
/// breaks the eventual hipHostFree. The buffer and external reference must
/// both survive until all reads and DMA finish.
/// The ROCm plugin must also recognize pinned ranges. On one MI300X, a
/// recognized pinned-host path measured 46.6 GiB/s while a stale plugin
/// allocated just as quickly but staged transfers at 6.5 GiB/s. The later
/// apparent degradation was traced to missing IsHostMemoryPinned support,
/// not the hardware.
const PinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,
    device_index: usize,

    fn create(memory: *const Memory, device_index: usize, size: usize) Allocator.Error!PinnedHostAllocation {
        const api = memory.platform.pjrt_api;
        const buffer = memory.platform.pjrt_client.createUninitializedBuffer(api, .{
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
        }) catch |err| {
            log.err("pinned host allocation failed on device {d} for {Bi:.2}: {s}", .{ device_index, size, @errorName(err) });
            return error.OutOfMemory;
        };
        errdefer buffer.deinit(api);
        std.debug.assert(buffer.isOnCpu(api));

        // The writable pointer is borrowed from PJRT. Keep both the external
        // reference and its owning buffer alive for the arena's whole lifetime.
        buffer.increaseExternalReferenceCount(api) catch |err| {
            log.err("retaining pinned host allocation failed on device {d} for {Bi:.2}: {s}", .{ device_index, size, @errorName(err) });
            return error.OutOfMemory;
        };
        errdefer buffer.decreaseExternalReferenceCount(api) catch {};
        const ptr: [*]u8 = @ptrCast(buffer.opaqueDeviceMemoryDataPointer(api) catch |err| {
            log.err("accessing pinned host allocation failed on device {d} for {Bi:.2}: {s}", .{ device_index, size, @errorName(err) });
            return error.OutOfMemory;
        });
        return .{
            .buffer = buffer,
            .api = api,
            .data = ptr[0..size],
            .device_index = device_index,
        };
    }

    fn destroy(self: PinnedHostAllocation) void {
        self.buffer.decreaseExternalReferenceCount(self.api) catch unreachable;
        self.buffer.deinit(self.api);
    }
};

/// Bits of `/sys/devices/system/node/has_memory`, or zero when unreadable.
/// Nodes 64 and above cannot be represented and are dropped.
/// Device-coherent HBM nodes are not candidates for host pages: the measured
/// four-GB300 topology had 34 NUMA nodes but only two with host memory.
fn memoryNodeMask(allocator: Allocator, io: std.Io) u64 {
    if (comptime builtin.os.tag != .linux) return 0;
    const contents = std.Io.Dir.cwd().readFileAlloc(
        io,
        "/sys/devices/system/node/has_memory",
        allocator,
        .limited(4096),
    ) catch return 0;
    defer allocator.free(contents);
    return parseNodeList(contents);
}

/// Interleaving is useful only when the host exposes several memory nodes.
fn interleaveMask(memory_node_mask: u64) u64 {
    return if (@popCount(memory_node_mask) > 1) memory_node_mask else 0;
}

/// Parses a kernel node list ("0-1", "0,2-3") into a mask; zero on any error.
fn parseNodeList(text: []const u8) u64 {
    var mask: u64 = 0;
    var ranges = std.mem.tokenizeAny(u8, text, ", \n");
    while (ranges.next()) |range| {
        var ends = std.mem.splitScalar(u8, range, '-');
        const first = std.fmt.parseInt(usize, ends.first(), 10) catch return 0;
        const last = if (ends.next()) |end|
            std.fmt.parseInt(usize, end, 10) catch return 0
        else
            first;
        if (last < first or ends.next() != null) return 0;
        var node = first;
        while (node <= last) : (node += 1) {
            if (node < 64) mask |= @as(u64, 1) << @intCast(node);
        }
    }
    return mask;
}

fn elapsedNanoseconds(started: std.Io.Timestamp, finished: std.Io.Timestamp) u64 {
    return @intCast(@max(started.durationTo(finished).nanoseconds, 0));
}

test "automatic NUMA placement interleaves only multiple memory nodes" {
    try std.testing.expectEqual(@as(u64, 0), interleaveMask(0));
    try std.testing.expectEqual(@as(u64, 0), interleaveMask(0b1));
    try std.testing.expectEqual(@as(u64, 0), interleaveMask(0b1000));
    try std.testing.expectEqual(@as(u64, 0b1001), interleaveMask(0b1001));
}

test "parseNodeList accepts kernel node lists" {
    try std.testing.expectEqual(@as(u64, 0b11), parseNodeList("0-1\n"));
    try std.testing.expectEqual(@as(u64, 0b1101), parseNodeList("0,2-3"));
    try std.testing.expectEqual(@as(u64, 0b1), parseNodeList("0"));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList(""));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList("1-0"));
    try std.testing.expectEqual(@as(u64, 0), parseNodeList("x"));
    try std.testing.expectEqual(@as(u64, 0b1), parseNodeList("0,64"));
}

test "Workspace finds retained arenas behind newer smaller allocations" {
    var workspace = try Workspace.initForTesting(std.testing.allocator, std.testing.io, 256);
    defer workspace.deinit();

    try std.testing.expect(workspace.findArena(1) == null);
    const large = try workspace.allocate(128);
    const small = try workspace.allocate(64);
    try std.testing.expectEqual(@intFromPtr(large.ptr), @intFromPtr(workspace.findArena(100).?.ptr));
    try std.testing.expectEqual(@intFromPtr(small.ptr), @intFromPtr(workspace.findArena(32).?.ptr));
    try std.testing.expect(workspace.findArena(129) == null);
    try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes);
}

test "Workspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: Allocator) !void {
            var workspace = try Workspace.initForTesting(allocator, std.testing.io, 256);
            defer workspace.deinit();

            _ = try workspace.allocate(64);
            _ = try workspace.allocate(128);
            try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes);
            try std.testing.expectError(error.OutOfMemory, workspace.allocate(128));
            try std.testing.expectEqual(@as(usize, 192), workspace.mapped_bytes);
            try std.testing.expectEqual(@as(usize, 128), workspace.findArena(100).?.len);
            try std.testing.expectEqual(@as(usize, 3), workspace.usableBlocks(64));
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes);
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes);
            try std.testing.expectError(error.OutOfMemory, workspace.growToBlocks(64, 5));
            try std.testing.expectEqual(@as(usize, 256), workspace.mapped_bytes);
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "Workspace growth maps missing blocks as one arena" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io, 64 * 64);
    defer workspace.deinit();

    _ = try workspace.allocate(3 * 64);
    try workspace.growToBlocks(64, 13);
    try std.testing.expectEqual(@as(usize, 13), workspace.usableBlocks(64));
    try std.testing.expectEqual(@as(usize, 2), workspace.backend.arenaCount());
    try std.testing.expectEqual(@as(usize, 10 * 64), workspace.backend.arenaAt(1).len);
    try workspace.growToBlocks(64, 14);
    try std.testing.expectEqual(@as(usize, 3), workspace.backend.arenaCount());
}

test "BlockPool ownership transfer cleans up allocation failures" {
    try std.testing.checkAllAllocationFailures(std.testing.allocator, struct {
        fn run(allocator: Allocator) !void {
            var arena: []u8 = undefined;
            var pool = pool_init: {
                var workspace = try Workspace.initForTesting(allocator, std.testing.io, 256);
                errdefer workspace.deinit();
                arena = try workspace.allocate(64);
                break :pool_init BlockPool.init(allocator, workspace, 64) catch |err| {
                    try std.testing.expectEqual(@as(usize, 64), workspace.mapped_bytes);
                    try std.testing.expectEqual(arena.ptr, workspace.findArena(64).?.ptr);
                    return err;
                };
            };
            defer pool.deinit();
            try std.testing.expectEqual(arena.ptr, pool.workspace.findArena(64).?.ptr);
            try std.testing.expectEqual(@as(usize, 1), pool.capacity);
            var blocks: [1]BlockPool.Block = undefined;
            try pool.acquireMany(std.testing.io, &blocks);
            pool.releaseMany(std.testing.io, &blocks);
        }
    }.run, .{});
}

test "BlockPool acquires request blocks atomically" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(allocator, std.testing.io, 4 * 64);
        errdefer workspace.deinit();

        _ = try workspace.allocate(4 * 64);
        break :pool_init try BlockPool.init(allocator, workspace, 64);
    };
    defer pool.deinit();

    var first: [3]BlockPool.Block = undefined;
    try pool.acquireMany(io, &first);
    try std.testing.expectEqual(@as(usize, 3 * 64), pool.high_water * pool.block_size);
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.workspace.mapped_bytes);
    var oversized: [5]BlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &oversized));

    var started: std.Io.Event = .unset;
    var acquired: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *BlockPool, io_: std.Io, started_: *std.Io.Event, acquired_: *std.Io.Event) void {
            var blocks: [2]BlockPool.Block = undefined;
            started_.set(io_);
            pool_.acquireMany(io_, &blocks) catch unreachable;
            acquired_.set(io_);
            pool_.releaseMany(io_, &blocks);
        }
    }.run, .{ &pool, io, &started, &acquired });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!acquired.isSet());

    pool.releaseMany(io, &first);
    try group.await(io);
    try std.testing.expect(acquired.isSet());
}

test "BlockPool acquisition allocates nothing once its arenas are attached" {
    const io = std.testing.io;
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{});
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(std.testing.allocator, io, 4 * 64);
        errdefer workspace.deinit();

        _ = try workspace.allocate(4 * 64);
        break :pool_init try BlockPool.init(failing.allocator(), workspace, 64);
    };
    defer pool.deinit();

    var blocks: [3]BlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    pool.releaseMany(io, &blocks);

    failing.fail_index = failing.alloc_index;
    try pool.acquireMany(io, &blocks);
    try std.testing.expect(!failing.has_induced_failure);
    pool.releaseMany(io, &blocks);
}

test "BlockPool close wakes blocked bulk acquisitions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(allocator, std.testing.io, 2 * 64);
        errdefer workspace.deinit();

        _ = try workspace.allocate(2 * 64);
        break :pool_init try BlockPool.init(allocator, workspace, 64);
    };
    defer pool.deinit();

    var held: [2]BlockPool.Block = undefined;
    try pool.acquireMany(io, &held);
    var started: std.Io.Event = .unset;
    var result: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(pool_: *BlockPool, io_: std.Io, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            var block: [1]BlockPool.Block = undefined;
            started_.set(io_);
            pool_.acquireMany(io_, &block) catch |err| {
                result_.store(@intFromError(err), .release);
                return;
            };
            pool_.releaseMany(io_, &block);
        }
    }.run, .{ &pool, io, &started, &result });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    pool.close(io);
    try group.await(io);
    pool.releaseMany(io, &held);
    try std.testing.expectEqual(@intFromError(error.Closed), result.load(.acquire));
}

test "BlockPool lease returns a block after out-of-order callbacks" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(allocator, std.testing.io, 64);
        errdefer workspace.deinit();

        _ = try workspace.allocate(64);
        break :pool_init try BlockPool.init(allocator, workspace, 64);
    };
    defer pool.deinit();

    var blocks: [1]BlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    var lease: BlockPool.Lease = .init(&pool, io, blocks[0], 4);
    var group: std.Io.Group = .init;
    for ([_]i64{ 4, 1, 3, 2 }) |delay_ms| {
        try group.concurrent(io, struct {
            fn run(lease_: *BlockPool.Lease, io_: std.Io, delay_ms_: i64) void {
                io_.sleep(.fromMilliseconds(delay_ms_), .awake) catch unreachable;
                _ = lease_.complete();
            }
        }.run, .{ &lease, io, delay_ms });
    }
    try group.await(io);
    try std.testing.expect(lease.remaining.load(.acquire) == 0);

    var reacquired: [1]BlockPool.Block = undefined;
    try pool.acquireMany(io, &reacquired);
    try std.testing.expectEqual(@intFromPtr(blocks[0].ptr), @intFromPtr(reacquired[0].ptr));
    pool.releaseMany(io, &reacquired);
}

test "BlockPool reblocks retained arenas of unequal size" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(allocator, std.testing.io, 349);
        errdefer workspace.deinit();

        _ = try workspace.allocate(150);
        _ = try workspace.allocate(70);
        _ = try workspace.allocate(65);

        break :pool_init try BlockPool.init(allocator, workspace, 64);
    };
    defer pool.deinit();
    try std.testing.expectEqual(@as(usize, 285), pool.workspace.mapped_bytes);
    // Two blocks from the first arena, one from each of the others.
    try std.testing.expectEqual(@as(usize, 4), pool.capacity);

    var blocks: [4]BlockPool.Block = undefined;
    try pool.acquireMany(io, &blocks);
    try std.testing.expectEqual(@as(usize, 3), pool.workspace.backend.arenaCount());
    try std.testing.expectEqual(@as(usize, 285), pool.workspace.mapped_bytes);
    try std.testing.expectEqual(@as(usize, 4 * 64), pool.high_water * pool.block_size);
    // Nothing grows: a request beyond the capacity can never be served.
    var oversized: [5]BlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &oversized));
    pool.releaseMany(io, &blocks);
}

test "BlockPool counts retained requests over arena tails" {
    const allocator = std.testing.allocator;
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(allocator, std.testing.io, 574);
        errdefer workspace.deinit();

        _ = try workspace.allocate(127);
        _ = try workspace.allocate(127);

        break :pool_init try BlockPool.init(allocator, workspace, 64);
    };
    defer pool.deinit();

    // Arena tails stay mapped without contributing a block: two 127-byte
    // arenas retain one 64-byte block each.
    try std.testing.expectEqual(@as(usize, 2), pool.capacity);
}

test "BlockPool rejects requests that can never fit without leasing" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool = pool_init: {
        var workspace = try Workspace.initForTesting(allocator, std.testing.io, 2 * 64);
        errdefer workspace.deinit();

        _ = try workspace.allocate(2 * 64);

        break :pool_init try BlockPool.init(allocator, workspace, 64);
    };
    defer pool.deinit();

    var impossible: [3]BlockPool.Block = undefined;
    try std.testing.expectError(error.RequestExceedsCapacity, pool.acquireMany(io, &impossible));
    try std.testing.expectEqual(@as(usize, 0), pool.high_water * pool.block_size);
    var fits: [2]BlockPool.Block = undefined;
    try pool.acquireMany(io, &fits);
    pool.releaseMany(io, &fits);
}
