//! Loader-owned host arenas, cut into blocks once. The direct backend owns
//! its workspace for its whole life: calibration maps its ring there, the
//! backend grows the set to what its load holds and carves it, and
//! `Loader.deinit` releases every arena. Callers never supply a workspace, a
//! budget, a NUMA policy or a calibration result.
//!
//! The memory depends on the target: CUDA and oneAPI take page-backed arenas
//! registered with PJRT (DmaMap), ROCm and Furiosa PJRT `pinned_host`
//! buffers, and CPU plain pages, unregistered. The arenas' kind and the
//! transfer path are separate decisions: CPU loads through the direct
//! backend too. Page-backed arenas interleave over the host's memory-bearing
//! NUMA nodes when the task's memory policy allows more than one; ROCm
//! balances whole arenas across its devices' host nodes.
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
const Backend = union(enum) {
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
    /// pinned, including subranges. Older oneAPI plugins treated SYCL imports
    /// as unknown; checking both ends against the same imported base removed
    /// the userspace copy. On one B70, DMA caps two/eight then measured
    /// 26.86/26.90 GiB/s instead of 21.17/11.01. More DMA credits had amplified
    /// staging, not improved DMA.
    pages: struct {
        platform: ?*const Platform,
        /// Zero leaves placement to the kernel. Automatic placement can
        /// fall back to zero if the kernel refuses it.
        numa_mask: u64,
    },

    const transparent_huge_page_size = 2 * 1024 * 1024;
    const mpol_interleave = 3;

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
        const numa_mask = interleaveMask(memoryNodeMask(allocator, io) & taskMemoryPolicyMask());
        return switch (platform.target) {
            .cuda, .oneapi => .{ .pages = .{ .platform = platform, .numa_mask = numa_mask } },
            .cpu => .{ .pages = .{ .platform = null, .numa_mask = numa_mask } },
            // Furiosa backs `.host_pinned` with SDK pinned (dma-buf) memory
            // from `/dev/dma_heap/system`, which the SDK recognizes by
            // address and transfers without pinning pages on every request.
            .rocm, .furiosa => {
                // PJRT-pinned allocations choose placement through the device's
                // host memory space, not our mbind policy. Retain one allocation
                // path per reported node; the blocks stay unpartitioned.
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
    fn deinit(self: *Backend, allocator: Allocator) void {
        switch (self.*) {
            .pjrt_host => |*host| host.host_nodes.deinit(allocator),
            .pages => {},
        }
    }

    /// Maps one arena of `bytes`.
    fn map(self: *Backend, allocator: Allocator, io: std.Io, bytes: usize) Allocator.Error!Arena {
        const started: std.Io.Timestamp = .now(io, .awake);
        switch (self.*) {
            .pjrt_host => |*host| {
                // Balance bytes, not arena/device counts: arenas differ in size.
                // Replicated Llama-3.1-8B on eight MI300X lost the fast mode when
                // all arenas came from device zero (1.27-1.45 s). Device rotation
                // left a 61/39 byte split and took 0.98-1.33 s; byte-balanced
                // allocation recovered 0.84-1.05 s over six runs, with 1.52 GiB
                // mapped versus 2.03 GiB for the former per-node pools. Host
                // contention affected both arms.
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
                // The ROCm plugin must also recognize pinned ranges. On one
                // MI300X, a recognized pinned-host path measured 46.6 GiB/s
                // while a stale plugin allocated just as quickly but staged
                // transfers at 6.5 GiB/s. The later apparent degradation was
                // traced to missing IsHostMemoryPinned support, not the hardware.
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
                log.info("DMA arena kind=pjrt_host device={d} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
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
                place(&pages.numa_mask, data);
                adviseHugePages(data);
                if (pages.platform) |p| p.pjrt_client.dmaMap(p.pjrt_api, data) catch |err| {
                    log.err("DMA registration failed for {Bi:.2} of host memory: {s}", .{ bytes, @errorName(err) });
                    allocator.rawFree(data, alignment, @returnAddress());
                    return error.OutOfMemory;
                };
                const finished: std.Io.Timestamp = .now(io, .awake);
                log.info("DMA arena kind={s} placement={s} nodes=0x{x} address=0x{x} size={Bi:.2} allocation_ms={d:.3}", .{
                    if (pages.platform != null) "dma_map" else "pageable",
                    if (pages.numa_mask == 0) "unplaced" else "interleave",
                    pages.numa_mask,
                    @intFromPtr(data.ptr),
                    bytes,
                    @as(f64, @floatFromInt(elapsedNanoseconds(started, finished))) / std.time.ns_per_ms,
                });
                return .{ .bytes = data };
            },
        }
    }

    fn unmap(self: *const Backend, allocator: Allocator, arena: Arena) void {
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

    /// Interleaves `data` over `numa_mask`'s nodes; on refusal, warns and
    /// zeroes the mask, so later arenas stay unplaced too.
    fn place(numa_mask: *u64, data: []u8) void {
        const mask = numa_mask.*;
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
        numa_mask.* = 0;
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

/// Host arenas owned by one direct loader. Calibration maps its ring
/// (`allocate`); the loader then grows the set to what its load holds and
/// cuts it into blocks, once (`carve`). Deinitialize after all transfers
/// finish and before the platform.
pub const Workspace = struct {
    /// Safety guard on the arenas' total host memory (pinned on the DMA
    /// targets), not an allocation target: calibration and the load's
    /// pre-growth decide what is actually mapped. Fixed, because no caller
    /// ever had a reason to choose another value.
    pub const mapped_bytes_ceiling: usize = 16 * 1024 * 1024 * 1024;

    allocator: Allocator,
    io: std.Io,
    backend: Backend,
    /// Every arena mapped, in mapping order, until `deinit`.
    arenas: std.ArrayList(Arena) = .empty,
    /// Set by `carve`: nothing maps after it.
    carved: bool = false,

    pub fn init(
        allocator: Allocator,
        io: std.Io,
        platform: *const Platform,
    ) Allocator.Error!Workspace {
        return .{
            .allocator = allocator,
            .io = io,
            .backend = try .init(allocator, io, platform),
        };
    }

    pub fn deinit(self: *Workspace) void {
        const io = self.io;
        const mapped_bytes = self.mappedBytes();
        const started: std.Io.Timestamp = .now(io, .awake);
        for (self.arenas.items) |arena| self.backend.unmap(self.allocator, arena);
        self.arenas.deinit(self.allocator);
        self.backend.deinit(self.allocator);
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
        std.debug.assert(!self.carved);
        const mapped = self.mappedBytes();
        if (bytes > mapped_bytes_ceiling - mapped) {
            log.err("host memory allocation exceeds mapped ceiling: requested={Bi:.2}, mapped={Bi:.2}, ceiling={Bi:.2}: OutOfMemory", .{ bytes, mapped, mapped_bytes_ceiling });
            return error.OutOfMemory;
        }
        // Reserved first, so a mapped arena is always retained: the ROCm
        // balancer counts the bytes it maps.
        try self.arenas.ensureUnusedCapacity(self.allocator, 1);
        const arena = try self.backend.map(self.allocator, self.io, bytes);
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
    /// allocation. The caller owns the list, allocated with the workspace's
    /// allocator; the workspace owns the memory.
    pub fn carve(self: *Workspace, block_size: usize, first_blocks: usize, total_blocks: usize) Allocator.Error![]Block {
        std.debug.assert(!self.carved and block_size != 0);
        try self.growToBlocks(block_size, first_blocks);
        try self.growToBlocks(block_size, total_blocks);
        const blocks = try self.allocator.alloc(Block, self.usableBlocks(block_size));
        var next: usize = 0;
        for (self.arenas.items) |arena| {
            for (0..arena.bytes.len / block_size) |j| {
                blocks[next] = arena.bytes[j * block_size ..][0..block_size];
                next += 1;
            }
        }
        std.debug.assert(next == blocks.len);
        self.carved = true;
        return blocks;
    }

    /// Counts complete blocks across retained arenas; block_size must be nonzero.
    fn usableBlocks(self: *const Workspace, block_size: usize) usize {
        var usable: usize = 0;
        for (self.arenas.items) |arena| usable += arena.bytes.len / block_size;
        return usable;
    }

    /// Maps the blocks missing below `target_blocks` as one arena. Requires a
    /// nonzero block_size; the workspace has one coordinating owner.
    fn growToBlocks(self: *Workspace, block_size: usize, target_blocks: usize) Allocator.Error!void {
        const usable_blocks = self.usableBlocks(block_size);
        const missing_blocks = target_blocks -| usable_blocks;
        if (missing_blocks == 0) return;
        const mapped = self.mappedBytes();
        if (missing_blocks > (mapped_bytes_ceiling - mapped) / block_size) {
            log.err("host memory growth exceeds mapped ceiling: block_size={Bi:.2}, target_blocks={d}, usable_blocks={d}, mapped={Bi:.2}, ceiling={Bi:.2}: OutOfMemory", .{ block_size, target_blocks, usable_blocks, mapped, mapped_bytes_ceiling });
            return error.OutOfMemory;
        }

        _ = try self.allocate(missing_blocks * block_size);
    }

    /// Creates ordinary allocator-backed arenas for tests without a PJRT platform.
    pub fn initForTesting(allocator: Allocator, io: std.Io) !Workspace {
        if (!builtin.is_test) @compileError("initForTesting is only available in tests");
        return .{
            .allocator = allocator,
            .io = io,
            .backend = .{ .pages = .{ .platform = null, .numa_mask = 0 } },
        };
    }
};

pub const Block = []u8;

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

/// The nodes the task's memory policy allows (`numactl --membind`,
/// `--interleave`, `--preferred`); every node under the default policy.
/// An arena placed outside them would defeat the policy: `mbind` on a
/// range overrides `set_mempolicy`.
fn taskMemoryPolicyMask() u64 {
    if (comptime builtin.os.tag != .linux) return std.math.maxInt(u64);
    var mode: c_int = 0;
    var nodes: [1]u64 = .{0};
    const rc = std.os.linux.syscall5(.get_mempolicy, @intFromPtr(&mode), @intFromPtr(&nodes), 64, 0, 0);
    if (std.os.linux.errno(rc) != .SUCCESS or mode == 0 or nodes[0] == 0) return std.math.maxInt(u64);
    return nodes[0];
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

test "Workspace capacity counts retained arenas and the room under the ceiling, past which growth refuses" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io);
    defer workspace.deinit();
    const ceiling = Workspace.mapped_bytes_ceiling;

    try std.testing.expectEqual(ceiling / 64, workspace.capacity(64));
    _ = try workspace.allocate(128);
    _ = try workspace.allocate(64);
    try std.testing.expectEqual(@as(usize, 192), workspace.mappedBytes());
    // Whole blocks only: the arenas hold one block of 100 bytes and their
    // tails none, beside those of the room left under the ceiling.
    try std.testing.expectEqual(1 + (ceiling - 192) / 100, workspace.capacity(100));
    try std.testing.expectEqual(ceiling / 64, workspace.capacity(64));
    // Growth refuses one block past capacity; reaching it would map 16 GiB.
    try std.testing.expectError(error.OutOfMemory, workspace.growToBlocks(100, workspace.capacity(100) + 1));
    try std.testing.expectError(error.OutOfMemory, workspace.growToBlocks(64, workspace.capacity(64) + 1));
    try std.testing.expectEqual(@as(usize, 192), workspace.mappedBytes());
    // The first step is already mapped; the second maps the last block.
    const blocks = try workspace.carve(64, 3, 4);
    defer allocator.free(blocks);
    try std.testing.expectEqual(@as(usize, 4), blocks.len);
    try std.testing.expectEqual(@as(usize, 256), workspace.mappedBytes());
    try std.testing.expectEqual(@as(usize, 3), workspace.arenas.items.len);
    try std.testing.expect(workspace.carved);
}

test "Workspace carve maps the first blocks as their own arena, then the rest" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io);
    defer workspace.deinit();

    const blocks = try workspace.carve(64, 2, 4);
    defer allocator.free(blocks);
    try std.testing.expectEqual(@as(usize, 4), blocks.len);
    try std.testing.expectEqual(@as(usize, 2), workspace.arenas.items.len);
    try std.testing.expectEqual(@as(usize, 128), workspace.arenas.items[0].bytes.len);
    try std.testing.expectEqual(@as(usize, 128), workspace.arenas.items[1].bytes.len);
}

test "Workspace arena ownership cleans up allocation failures" {
    const AllocationTest = struct {
        fn run(allocator: Allocator) !void {
            var workspace = try Workspace.initForTesting(allocator, std.testing.io);
            defer workspace.deinit();

            _ = try workspace.allocate(64);
            _ = try workspace.allocate(128);
            try std.testing.expectEqual(@as(usize, 192), workspace.mappedBytes());
            try std.testing.expectError(error.OutOfMemory, workspace.allocate(Workspace.mapped_bytes_ceiling - 192 + 1));
            try std.testing.expectEqual(@as(usize, 192), workspace.mappedBytes());
            try std.testing.expectEqual(@as(usize, 2), workspace.arenas.items.len);
            try std.testing.expectEqual(@as(usize, 3), workspace.usableBlocks(64));
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mappedBytes());
            try workspace.growToBlocks(64, 4);
            try std.testing.expectEqual(@as(usize, 256), workspace.mappedBytes());
            try std.testing.expectError(error.OutOfMemory, workspace.growToBlocks(64, workspace.capacity(64) + 1));
            try std.testing.expectEqual(@as(usize, 256), workspace.mappedBytes());
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "Workspace growth maps missing blocks as one arena" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io);
    defer workspace.deinit();

    _ = try workspace.allocate(3 * 64);
    try workspace.growToBlocks(64, 13);
    try std.testing.expectEqual(@as(usize, 13), workspace.usableBlocks(64));
    try std.testing.expectEqual(@as(usize, 2), workspace.arenas.items.len);
    try std.testing.expectEqual(@as(usize, 10 * 64), workspace.arenas.items[1].bytes.len);
    try workspace.growToBlocks(64, 14);
    try std.testing.expectEqual(@as(usize, 3), workspace.arenas.items.len);
}

test "blocks are carved from retained arenas of unequal size, tails left out" {
    const allocator = std.testing.allocator;
    var workspace = try Workspace.initForTesting(allocator, std.testing.io);
    defer workspace.deinit();
    const first = try workspace.allocate(150);
    _ = try workspace.allocate(70);
    _ = try workspace.allocate(65);
    // Tails stay mapped without contributing a block: 127 bytes hold one.
    _ = try workspace.allocate(127);
    _ = try workspace.allocate(127);

    const blocks = try workspace.carve(64, 0, 0);
    defer allocator.free(blocks);
    // Two blocks from the first arena, one from each of the others.
    try std.testing.expectEqual(@as(usize, 6), blocks.len);
    try std.testing.expectEqual(@as(usize, 539), workspace.mappedBytes());
    for (blocks) |block| try std.testing.expectEqual(@as(usize, 64), block.len);
    try std.testing.expectEqual(first.ptr, blocks[0].ptr);
    try std.testing.expectEqual(first.ptr + 64, blocks[1].ptr);
}
