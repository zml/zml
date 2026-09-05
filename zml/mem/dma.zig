//! Host allocation adapters for device transfers.

const std = @import("std");
const Alignment = std.mem.Alignment;
const builtin = @import("builtin");

const pjrt = @import("pjrt");

const Device = @import("../platform.zig").Device;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;

const log = std.log.scoped(.@"zml/mem");

/// Placement of page-backed arenas (CUDA, oneAPI and CPU). Interleaving
/// balances the host copies from the page cache as well as device transfers.
/// ROCm's PJRT-owned arenas are instead balanced over the devices' host nodes.
pub const NumaPlacement = union(enum) {
    /// Interleave page by page over every host node that has memory
    /// (`/sys/devices/system/node/has_memory`). A single node, or no
    /// readable list, applies no policy.
    memory_nodes,
    /// One bit per node: a single bit binds, several interleave.
    nodes: u64,
    /// No policy: wherever the kernel and the driver put the pages.
    none,
};

/// Selects the platform's host allocation strategy for individual transfers.
pub const Allocator = union(enum) {
    passthrough: std.mem.Allocator,
    buffer: BufferAllocator,
    map: MapAllocator,

    pub fn init(parent: std.mem.Allocator, device: *const Device) Allocator {
        return switch (device.platform.target) {
            .cuda, .oneapi, .rocm => .{ .map = .init(parent, device.platform) },
            .tpu => .{ .buffer = .init(device.memory(.host_pinned).?) },
            .cpu, .neuron, .metal => .{ .passthrough = parent },
        };
    }

    pub fn allocator(self: *const Allocator) std.mem.Allocator {
        return switch (self.*) {
            .passthrough => |a| a,
            inline else => |*a| a.allocator(),
        };
    }
};

/// Host allocator for the loader's page-backed arenas. An allocation of at
/// least one huge page is aligned to it and, on Linux, advised into
/// transparent huge pages (a valid ordinary-page mapping when unavailable).
/// With a platform the pages are then registered with its PJRT client
/// through `dmaMap` (CUDA, oneAPI); without one they stay plain pages,
/// which is all the CPU plugin's transfers read from.
pub const MapAllocator = struct {
    const transparent_huge_page_size = 2 * 1024 * 1024;

    parent: std.mem.Allocator,
    /// Null registers nothing.
    platform: ?*const Platform,

    pub fn init(parent: std.mem.Allocator, platform: *const Platform) MapAllocator {
        return .{
            .parent = parent,
            .platform = platform,
        };
    }

    pub fn initPageable(parent: std.mem.Allocator) MapAllocator {
        return .{
            .parent = parent,
            .platform = null,
        };
    }

    pub fn allocator(self: *const MapAllocator) std.mem.Allocator {
        return .{
            .ptr = @constCast(self),
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *const MapAllocator = @ptrCast(@alignCast(ctx));
        const effective_alignment = effectiveAlignment(alignment, len);
        const allocation = self.parent.rawAlloc(len, effective_alignment, ret_addr);
        if (allocation) |loc| {
            const data = loc[0..len];
            adviseHugePages(data);
            if (self.platform) |platform| {
                platform.pjrt_client.dmaMap(platform.pjrt_api, @ptrCast(data)) catch {
                    self.parent.rawFree(data, effective_alignment, ret_addr);
                    return null;
                };
            }
        }
        return allocation;
    }

    fn resize(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) bool {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return false;
    }

    fn remap(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, new_len: usize, ret_addr: usize) ?[*]u8 {
        _ = ctx;
        _ = buf;
        _ = alignment;
        _ = new_len;
        _ = ret_addr;
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *const MapAllocator = @ptrCast(@alignCast(ctx));
        if (self.platform) |platform| {
            platform.pjrt_client.dmaUnmap(platform.pjrt_api, @ptrCast(buf[0..buf.len])) catch unreachable;
        }
        self.parent.rawFree(buf, effectiveAlignment(alignment, buf.len), ret_addr);
    }

    /// Nothing below one huge page can be backed by one.
    fn effectiveAlignment(alignment: Alignment, len: usize) Alignment {
        if (comptime builtin.os.tag != .linux) return alignment;
        if (len < transparent_huge_page_size) return alignment;
        return alignment.max(.fromByteUnits(transparent_huge_page_size));
    }

    fn adviseHugePages(data: []u8) void {
        if (comptime builtin.os.tag != .linux) {
            return;
        }
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

/// Allocates host memory owned by PJRT buffers, as required by TPU.
pub const BufferAllocator = struct {
    memory: *const Memory,

    pub fn init(memory: *const Memory) BufferAllocator {
        return .{
            .memory = memory,
        };
    }

    pub fn allocator(self: *const BufferAllocator) std.mem.Allocator {
        return .{
            .ptr = @constCast(self),
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    const Header = struct {
        buffer: *pjrt.Buffer,
    };

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, _: usize) ?[*]u8 {
        const self: *BufferAllocator = @ptrCast(@alignCast(ctx));
        const pjrt_api = self.memory.platform.pjrt_api;
        const pjrt_client = self.memory.platform.pjrt_client;

        const total_len = allocationSize(len, alignment) orelse return null;
        const dimension = std.math.cast(i64, total_len) orelse return null;

        const pjrt_buffer = pjrt_client.createUninitializedBuffer(pjrt_api, .{
            .dims = &.{dimension},
            .element_type = .u8,
            .layout = .{
                .tiled = .{
                    .minor_to_major = &.{0},
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = .{ .memory = self.memory.pjrt_memory },
        }) catch return null;

        const opaque_ptr: [*]u8 = @ptrCast(pjrt_buffer.opaqueDeviceMemoryDataPointer(pjrt_api) catch unreachable);
        const data = alignedData(opaque_ptr, alignment);
        allocationHeader(data).* = .{ .buffer = pjrt_buffer };
        return data;
    }

    fn free(ctx: *anyopaque, buf: []u8, _: Alignment, _: usize) void {
        const self: *BufferAllocator = @ptrCast(@alignCast(ctx));
        const pjrt_api = self.memory.platform.pjrt_api;
        allocationHeader(buf.ptr).buffer.deinit(pjrt_api);
    }

    fn resize(_: *anyopaque, _: []u8, _: Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn allocationSize(len: usize, alignment: Alignment) ?usize {
        const header_and_data = std.math.add(usize, @sizeOf(Header), len) catch return null;
        const padding = alignment.max(.fromByteUnits(@alignOf(Header))).toByteUnits() - 1;
        return std.math.add(usize, header_and_data, padding) catch null;
    }

    fn alignedData(base: [*]u8, alignment: Alignment) [*]u8 {
        // PJRT's base need not meet the requested alignment. Place the header
        // immediately before the aligned data so free can always recover it.
        const effective_alignment = alignment.max(.fromByteUnits(@alignOf(Header)));
        return @ptrFromInt(std.mem.alignForward(
            usize,
            @intFromPtr(base) + @sizeOf(Header),
            effective_alignment.toByteUnits(),
        ));
    }

    fn allocationHeader(data: [*]u8) *Header {
        return @ptrFromInt(@intFromPtr(data) - @sizeOf(Header));
    }
};

test "BufferAllocator fits aligned data and its header at every base offset" {
    const owner: *pjrt.Buffer = @ptrFromInt(8);
    for ([_]usize{ 1, 2, 8, 64, 4096 }) |alignment_bytes| {
        const alignment: Alignment = .fromByteUnits(alignment_bytes);
        const base_offsets = @max(alignment_bytes, @alignOf(BufferAllocator.Header));
        for ([_]usize{ 1, 7, 100, 4097 }) |len| {
            const size = BufferAllocator.allocationSize(len, alignment).?;
            const storage = try std.testing.allocator.alloc(u8, size + base_offsets - 1);
            defer std.testing.allocator.free(storage);

            for (0..base_offsets) |offset| {
                const allocation = storage[offset..][0..size];
                const data = BufferAllocator.alignedData(allocation.ptr, alignment);
                const header = BufferAllocator.allocationHeader(data);
                try std.testing.expectEqual(@as(usize, 0), @intFromPtr(data) % alignment_bytes);
                try std.testing.expectEqual(@as(usize, 0), @intFromPtr(header) % @alignOf(BufferAllocator.Header));
                try std.testing.expect(@intFromPtr(header) >= @intFromPtr(allocation.ptr));
                try std.testing.expect(@intFromPtr(data) + len <= @intFromPtr(allocation.ptr) + allocation.len);

                header.* = .{ .buffer = owner };
                @memset(data[0..len], 0xa5);
                try std.testing.expectEqual(owner, BufferAllocator.allocationHeader(data).buffer);
            }
        }
    }
    try std.testing.expectEqual(null, BufferAllocator.allocationSize(std.math.maxInt(usize), .@"1"));
    try std.testing.expectEqual(null, BufferAllocator.allocationSize(std.math.maxInt(usize) - @sizeOf(BufferAllocator.Header), .@"64"));
}
