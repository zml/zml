//! A `Buffer` in `.host_pinned` memory that the host reads and writes in place
//! through views, which a plain `Buffer` placed there doesn't offer. Views
//! address elements as PJRT stores them, with sub-byte elements packed in bytes.
//! The host may only access the buffer between executions. Host pointers are
//! read at creation, and by `await` when an execution replaced the handles.
//! Executions must preserve shape, sharding and layout.
//!
//! Like `Buffer`, it is a value. Pass `buffer` to an execution to read it as
//! an input, or `&buffer` to donate it as an output, then `await` the
//! HostAccessible whose `buffer` the execution replaced. Copies share the
//! mapping, and a copy whose handles the execution did not replace panics on
//! `view` rather than write to an allocation the execution may have released.
const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("../buffer.zig").Buffer;
const emptyShell = @import("../buffer.zig").emptyShell;
const Platform = @import("../platform.zig").Platform;
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const testing = @import("../testing.zig");

const HostAccessible = @This();

/// The device side, with one PJRT buffer per device, for executions to read
/// or donate.
buffer: Buffer,
/// What views read, heap allocated so views can borrow it while the
/// HostAccessible moves.
mapping: *Mapping,

/// The host side of the buffer. Views and blocks copy nothing from it: the
/// layout and shard placement are prepared once at creation and take several
/// KB, and copying them into every view, slice and block made memcpy most of
/// llmd's decode thread.
const Mapping = struct {
    /// The PJRT shape, which packs sub-byte elements in bytes, unlike
    /// `buffer._shape`. Views address its elements.
    packed_shape: Shape,
    shard_shape: Shape,
    layout_metadata: LayoutMetadata,
    allocations: HostPinnedAllocations,
    /// Every shard holds the whole array densely, so element `i` sits at
    /// `i * element_size` in each of them, as in replicated buffers on every
    /// platform but TPU. `get`, `set` and blocks then skip locating elements
    /// through the origins and the layout.
    contiguous: bool,

    /// Takes ownership of the allocations, on success only.
    fn create(
        allocator: std.mem.Allocator,
        shape_: Shape,
        shard_shape: Shape,
        layout: pjrt.MemoryLayout,
        allocations: HostPinnedAllocations,
    ) !*Mapping {
        const mapping = try allocator.create(Mapping);
        errdefer allocator.destroy(mapping);

        const layout_metadata: LayoutMetadata = .init(shape_, shard_shape, layout);

        mapping.* = .{
            .packed_shape = shape_,
            .shard_shape = shard_shape,
            .layout_metadata = layout_metadata,
            .allocations = allocations,
            .contiguous = layout_metadata.addressing == .row_major and
                allocations.regions.len == 1 and
                shard_shape.count() == shape_.count(),
        };

        return mapping;
    }
};

/// The host allocations behind the buffer's shards. Devices that share an
/// allocation map to one entry, and entries holding the same region of the
/// array are its replicas. Entries stay in region order, so a region's
/// replicas are a range of every field.
const HostPinnedAllocations = struct {
    entries: std.MultiArrayList(Allocation) = .empty,
    /// The regions the entries hold, in order of first appearance.
    regions: stdx.BoundedArray(Region, Platform.MAX_NUM_DEVICES) = .empty,

    const empty: HostPinnedAllocations = .{};

    const Allocation = struct {
        /// Global coordinates of the allocation's first element.
        origin: Coordinates,
        /// Read from `handle`. Blocks borrow their region's range.
        ptr: [*]u8,
        /// The shard owning the allocation, which must destroy it: an index
        /// into `buffer._shards`, whose handles execution may replace.
        shard_index: usize,
        /// The handle `ptr` was read from. Execution replaces the handles of
        /// donated buffers with its outputs, which may not reuse the allocation.
        handle: *pjrt.Buffer,
        /// Whether `handle` holds an external reference, which keeps the
        /// allocation alive for the device views of other devices.
        shared: bool = false,
    };

    /// `entries[start..][0..len]` hold the region's replicas, in the order they
    /// were added. Reads use the first.
    const Region = struct { start: u8, len: u8 };

    fn deinit(self: *HostPinnedAllocations, allocator: std.mem.Allocator) void {
        self.entries.deinit(allocator);
    }

    /// Add an allocation to the replicas of its region, within the
    /// reserved capacity.
    fn add(self: *HostPinnedAllocations, shard_shape: Shape, allocation: Allocation) void {
        const region_index = self.findRegion(allocation.origin, shard_shape) orelse {
            self.regions.appendAssumeCapacity(.{ .start = @intCast(self.entries.len), .len = 1 });
            self.entries.appendAssumeCapacity(allocation);
            return;
        };
        const region = &self.regions.slice()[region_index];
        // Keep each region's entries together by shifting the regions after it.
        self.entries.insertAssumeCapacity(region.start + region.len, allocation);
        region.len += 1;
        for (self.regions.slice()[region_index + 1 ..]) |*next| next.start += 1;
    }

    /// The index of the region holding the shards that start at `origin`.
    fn findRegion(self: *const HostPinnedAllocations, origin: Coordinates, shard_shape: Shape) ?usize {
        const origins = self.entries.items(.origin);
        for (self.regions.constSlice(), 0..) |region, i| {
            if (origin.eqlWithin(origins[region.start], shard_shape)) return i;
        }
        return null;
    }

    /// The host pointers of a region's replicas.
    fn replicas(self: *const HostPinnedAllocations, region: Region) []const [*]u8 {
        return self.entries.items(.ptr)[region.start..][0..region.len];
    }

    /// Read the host pointers of the allocations from their current handles.
    /// Each read takes and releases a hold on the PJRT buffer and allocates an
    /// external reference, about 0.9 us: llmd would make thousands per step if
    /// views read them, so they are read at creation and by `await` only.
    fn refreshPointers(self: *HostPinnedAllocations, api: *const pjrt.Api, handles: []const *pjrt.Buffer) !void {
        const shard_indices = self.entries.items(.shard_index);
        // Read every pointer before replacing any, so a failure keeps the previous ones.
        var ptrs: [Platform.MAX_NUM_DEVICES][*]u8 = undefined;
        for (shard_indices, ptrs[0..shard_indices.len]) |shard_index, *ptr| {
            ptr.* = @ptrCast(try handles[shard_index].opaqueDeviceMemoryDataPointer(api));
        }
        @memcpy(self.entries.items(.ptr), ptrs[0..shard_indices.len]);
        for (self.entries.items(.handle), shard_indices) |*handle, shard_index| handle.* = handles[shard_index];
    }

    /// Whether the host pointers were read from the current handles, which
    /// execution replaces for donated buffers.
    fn pointersAreCurrent(self: *const HostPinnedAllocations, handles: []const *pjrt.Buffer) bool {
        // Host memory in tests has no handles: its pointers never change.
        if (handles.len == 0) return true;
        for (self.entries.items(.handle), self.entries.items(.shard_index)) |handle, shard_index| {
            if (handle != handles[shard_index]) return false;
        }
        return true;
    }

    /// Destroy every handle, the device views before the allocations they borrow.
    fn deinitHandles(self: *const HostPinnedAllocations, api: *const pjrt.Api, handles: []const *pjrt.Buffer) void {
        const shard_indices = self.entries.items(.shard_index);
        for (handles, 0..) |handle, index| {
            if (std.mem.indexOfScalar(usize, shard_indices, index) == null) handle.deinit(api);
        }
        for (shard_indices, self.entries.items(.shared)) |shard_index, shared| {
            const handle = handles[shard_index];
            // `shared` is set by the one successful increase, and only undonatable
            // buffers share allocations, whose handles execution never replaces.
            if (shared) handle.decreaseExternalReferenceCount(api) catch |err| {
                std.debug.panic("shared HostAccessible allocation has no reference to release, was it donated? {}", .{err});
            };
            handle.deinit(api);
        }
    }
};

/// Host memory standing in for PJRT buffers, to test layouts and shardings
/// that no available platform produces: every platform but TPU uses dense
/// row-major layouts, and CI has few devices. `origins` holds the row-major
/// index in `shape_` of each allocation's first element, and `ptrs` one pointer
/// per origin. Copies the layout, the origins and the pointers, and borrows the
/// storage.
pub fn initForTests(
    allocator: std.mem.Allocator,
    shape_: Shape,
    shard_shape: Shape,
    layout: pjrt.MemoryLayout,
    origins: []const usize,
    ptrs: []const [*]u8,
) !HostAccessible {
    if (!builtin.is_test) @compileError("HostAccessible.initForTests is only available in tests");

    var allocations: HostPinnedAllocations = .empty;
    errdefer allocations.deinit(allocator);

    try allocations.entries.ensureTotalCapacity(allocator, origins.len);
    // Without PJRT buffers, there are no handles to read pointers from or destroy.
    for (origins, ptrs) |origin, ptr| {
        allocations.add(
            shard_shape,
            .{
                // An empty array has no element to locate the origin of.
                .origin = if (shape_.count() == 0) .zero else .unflatten(shape_, origin),
                .ptr = ptr,
                .shard_index = undefined,
                .handle = undefined,
            },
        );
    }

    return .{
        .buffer = .{ ._platform = undefined, ._shape = shape_, ._shards = .empty },
        .mapping = try .create(allocator, shape_, shard_shape, layout, allocations),
    };
}

/// With `.undonatable`, devices holding the same shard share one allocation on
/// CPU, CUDA and ROCm: the first allocates it, and the others read it through
/// device views. Elsewhere each device owns an allocation, and writes reach
/// every one of them. Execution never replaces the handles of an undonatable
/// buffer, so `await` never reads its host pointers again.
/// With `.donatable`, each device owns an allocation: writing a shared one
/// would expose the output of a device to the reads of another.
pub fn init(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    shape_: Shape,
    donation: Buffer.Donation,
) !HostAccessible {
    const can_share_allocations = switch (platform.target) {
        .cpu, .cuda, .rocm => donation == .undonatable,
        else => false,
    };
    const api = platform.pjrt_api;
    // Executions only read undonatable buffers: let them skip holds and usage
    // events. Converted before any view or external reference is taken.
    const undonatable_buffers = if (donation == .undonatable) api.undonatableBuffers() else null;

    var buffer, const metadata = emptyShell(platform, shape_);
    const devices = platform.physical_mesh.devices_in_canonical_order;

    var allocations: HostPinnedAllocations = .empty;
    errdefer allocations.deinit(allocator);

    try allocations.entries.ensureTotalCapacity(allocator, devices.len);
    errdefer allocations.deinitHandles(api, buffer._shards.constSlice());

    for (devices, 0..) |device, shard_index| {
        const memory = platform.devices[device.id].memory(.host_pinned).?.pjrt_memory;
        const origin: Coordinates = .ofShard(&metadata.placement, device);
        const shared_region = if (can_share_allocations) allocations.findRegion(origin, metadata.placement.shape) else null;

        if (shared_region) |region_index| {
            // Devices sharing allocations add no replicas: the region has a single entry.
            const entry = allocations.regions.get(region_index).start;
            const existing = allocations.entries.get(entry);
            if (!existing.shared) {
                try existing.handle.increaseExternalReferenceCount(api);
                allocations.entries.items(.shared)[entry] = true;
            }
            const view_handle = try platform.pjrt_client.createViewOfDeviceBuffer(api, .{
                .device_buffer_ptr = try existing.handle.opaqueDeviceMemoryDataPointer(api),
                .dims = metadata.placement.shape.dims(),
                .element_type = metadata.ty,
                .layout = try existing.handle.memoryLayout(api),
                // The backing host allocation is accessible across devices
                // (portable pinned memory on CUDA/ROCm). This memory space
                // associates the view with the current device rather than
                // the device that originally allocated the memory.
                .memory = memory,
            });
            buffer._shards.appendAssumeCapacity(view_handle);
            if (undonatable_buffers) |ext| try ext.makeUndonatable(api, view_handle);
        } else {
            const handle = try platform.pjrt_client.createUninitializedBuffer(api, .{
                .dims = metadata.placement.shape.dims(),
                .element_type = metadata.ty,
                .layout = metadata.layout,
                .dst = .{ .memory = memory },
            });
            buffer._shards.appendAssumeCapacity(handle);
            if (undonatable_buffers) |ext| try ext.makeUndonatable(api, handle);
            // The pointer is read once the creation completes, after `await` below.
            allocations.add(metadata.placement.shape, .{
                .origin = origin,
                .ptr = undefined,
                .shard_index = shard_index,
                .handle = handle,
            });
        }
    }

    try buffer.await(io);
    try allocations.refreshPointers(api, buffer._shards.constSlice());

    return .{
        .buffer = buffer,
        .mapping = try .create(allocator, metadata.shape, metadata.placement.shape, metadata.layout, allocations),
    };
}

/// Execution must be done with the buffer. Copies must not be used after.
pub fn deinit(self: *const HostAccessible, allocator: std.mem.Allocator) void {
    // Host memory in tests has neither handles nor a platform.
    if (self.buffer._shards.len != 0) self.mapping.allocations.deinitHandles(self.buffer._platform.pjrt_api, self.buffer._shards.constSlice());
    self.mapping.allocations.deinit(allocator);
    allocator.destroy(self.mapping);
}

/// Wait for execution to finish, and read the host pointers of handles that
/// execution replaced. Take new views to access the output.
pub fn await(self: *HostAccessible, io: std.Io) !void {
    try self.buffer.await(io);
    const handles = self.buffer._shards.constSlice();
    const allocations = &self.mapping.allocations;
    if (!allocations.pointersAreCurrent(handles)) try allocations.refreshPointers(self.buffer._platform.pjrt_api, handles);
}

/// Borrows the mapping. Executions using the buffer must have completed, and
/// `await` must follow any that replaced the handles: `view` panics before
/// then rather than write to an allocation the execution may have released.
/// The view and its blocks are valid until the next execution starts or the
/// HostAccessible is deinitialized.
pub fn view(self: *const HostAccessible, comptime T: type) View(T) {
    const mapping = self.mapping;
    std.debug.assert(@sizeOf(T) == mapping.shard_shape.dtype().sizeOf());
    stdx.debug.assert(mapping.allocations.pointersAreCurrent(self.buffer._shards.constSlice()), "await the HostAccessible after execution replaces its handles", .{});
    return .{ .mapping = mapping, .len = mapping.packed_shape.count() };
}

/// Prints every element, read through a view: the same rules as `view` apply.
pub fn format(self: *const HostAccessible, writer: *std.Io.Writer) std.Io.Writer.Error!void {
    try writer.print("HostAccessible({f}) {{", .{self.mapping.packed_shape});
    switch (self.mapping.shard_shape.dtype()) {
        inline else => |dt| {
            if (comptime dt.bitSizeOf() < 8) unreachable;
            const values = self.view(dt.toZigType());
            var blocks = values.blocks();
            while (blocks.next()) |block| {
                for (block.firstReplica()) |value| try writer.print(" {any}", .{value});
            }
        },
    }
    try writer.writeAll(" }");
}

/// A range of the array's elements in row-major order, borrowed from a
/// HostAccessible. Reads use the first replica, and writes reach every replica.
pub fn View(comptime T: type) type {
    return struct {
        mapping: *const Mapping,
        start: usize = 0,
        len: usize,

        const Self = @This();

        /// A view without elements, which never accesses the buffer.
        pub const empty: Self = .{
            .mapping = undefined,
            .len = 0,
        };

        /// Select `len_` elements starting at `start`, or the rest of the view for null.
        pub fn slice(self: Self, start: usize, len_: ?usize) Self {
            std.debug.assert(start <= self.len);
            const len = len_ orelse self.len - start;
            std.debug.assert(len <= self.len - start);
            var result = self;
            result.start += start;
            result.len = len;
            return result;
        }

        // Single elements are addressed directly, without slicing and iterating.
        pub fn get(self: Self, index: usize) T {
            const element = self.locateElement(index);
            return itemAt(self.mapping.allocations.replicas(element.region)[0], element.byte_offset).*;
        }

        pub fn set(self: Self, index: usize, value: T) void {
            const element = self.locateElement(index);
            for (self.mapping.allocations.replicas(element.region)) |ptr| itemAt(ptr, element.byte_offset).* = value;
        }

        fn locateElement(self: *const Self, index: usize) struct { region: HostPinnedAllocations.Region, byte_offset: usize } {
            std.debug.assert(index < self.len);
            if (self.mapping.contiguous) {
                return .{ .region = self.mapping.allocations.regions.get(0), .byte_offset = (self.start + index) * @sizeOf(T) };
            }
            const location = self.locate(.unflatten(self.mapping.packed_shape, self.start + index));
            return .{ .region = location.region, .byte_offset = self.mapping.layout_metadata.byteOffset(location.offset, location.local) };
        }

        pub fn fill(self: Self, value: T) void {
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                for (0..block.replicas.len) |i| @memset(block.replica(i), value);
            }
        }

        /// Fill with `first + i * step`, where `i` is relative to this view.
        /// `T` must be an integer type holding every `i` and value; safe builds check it.
        pub fn fillIota(self: Self, first: T, step: T) void {
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                const items = block.firstReplica();
                // TODO: SIMD
                for (items, block.offset..) |*value, i| value.* = first + @as(T, @intCast(i)) * step;
                for (1..block.replicas.len) |i| @memcpy(block.replica(i), items);
            }
        }

        pub fn copyFrom(self: Self, values: []const T) void {
            std.debug.assert(values.len == self.len);
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                for (0..block.replicas.len) |i| @memcpy(block.replica(i), values[block.offset..][0..block.len]);
            }
        }

        pub fn copyTo(self: Self, values: []T) void {
            std.debug.assert(values.len == self.len);
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                @memcpy(values[block.offset..][0..block.len], block.firstReplica());
            }
        }

        /// The index of the first element equal to `value`, searching each
        /// block with `std.mem.indexOfScalar`.
        pub fn indexOfScalar(self: Self, value: T) ?usize {
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                if (std.mem.indexOfScalar(T, block.firstReplica(), value)) |index| return block.offset + index;
            }
            return null;
        }

        /// The index of the first element equal to any of `values`, searching
        /// each block with `std.mem.indexOfAny`.
        pub fn indexOfAny(self: Self, values: []const T) ?usize {
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                if (std.mem.indexOfAny(T, block.firstReplica(), values)) |index| return block.offset + index;
            }
            return null;
        }

        /// Source and destination must not overlap.
        pub fn copyFromView(self: Self, src: Self) void {
            std.debug.assert(src.len == self.len);
            var iterator = self.blocks();
            while (iterator.next()) |block| {
                const items = block.firstReplica();
                src.slice(block.offset, block.len).copyTo(items);
                for (1..block.replicas.len) |i| @memcpy(block.replica(i), items);
            }
        }

        /// The elements as one read-only slice, for reads that index or zip
        /// several views. Contiguous buffers, such as replicated ones on every
        /// platform but TPU, return their first replica in place; others
        /// copy into `scratch`, which must hold `len` elements on every
        /// platform. The slice is valid until the next execution, and
        /// writes must go through the view.
        pub fn constSlice(self: Self, scratch: []T) []const T {
            std.debug.assert(scratch.len >= self.len);
            if (self.inPlace()) |block| return block.firstReplica();
            self.copyTo(scratch[0..self.len]);
            return scratch[0..self.len];
        }

        /// A slice to write the elements into, then `commit`, for writes
        /// that compute elements in place. Contiguous buffers hand out their
        /// first replica; others hand out `scratch`, which must hold `len`
        /// elements on every platform.
        pub fn edit(self: Self, scratch: []T) Edit {
            std.debug.assert(scratch.len >= self.len);
            return .{
                .view = self,
                .items = if (self.inPlace()) |block| block.firstReplica() else scratch[0..self.len],
            };
        }

        pub const Edit = struct {
            view: Self,
            /// Holds the previous values or garbage: write every element.
            items: []T,

            /// Make the writes reach every replica: copy the first replica to
            /// the others, or `items` into the view.
            pub fn commit(self: Edit) void {
                const block = self.view.inPlace() orelse return self.view.copyFrom(self.items);
                for (1..block.replicas.len) |i| @memcpy(block.replica(i), self.items);
            }
        };

        /// The view as a single block, when the buffer is contiguous.
        fn inPlace(self: Self) ?Block {
            if (self.len == 0 or !self.mapping.contiguous) return null;
            var iterator = self.blocks();
            return iterator.next().?;
        }

        /// Visit runs of consecutive logical elements in global row-major
        /// order, without padding, grouping replicas of each logical range.
        /// Shards must cover the view with disjoint regions or exact replicas.
        pub fn blocks(self: Self) BlockIterator {
            return .init(self);
        }

        /// Visit the same blocks in reverse order. Elements within each
        /// replica slice remain in ascending logical order.
        pub fn reverseBlocks(self: Self) ReverseBlockIterator {
            return .init(self);
        }

        /// A run of elements that are consecutive in the view's row-major order and
        /// contiguous in memory, so each replica holds them as a plain slice.
        /// Writes must reach every replica; reads use the first. Borrows the buffer's
        /// host pointers, like the view it comes from.
        pub const Block = struct {
            /// Logical position of the first element, relative to this view.
            offset: usize,
            len: usize,
            /// Byte offset of the first element in the allocation of every replica.
            byte_offset: usize,
            /// Host pointer of each replica's allocation.
            replicas: []const [*]u8,

            /// The block's elements in its `i`-th replica.
            pub fn replica(self: Block, i: usize) []T {
                const ptr: [*]T = @ptrCast(@alignCast(self.replicas[i] + self.byte_offset));
                return ptr[0..self.len];
            }

            /// The elements in the first replica, which reads use.
            pub fn firstReplica(self: Block) []T {
                return self.replica(0);
            }
        };

        pub const BlockIterator = OrderedBlockIterator(false);
        pub const ReverseBlockIterator = OrderedBlockIterator(true);

        fn OrderedBlockIterator(comptime reverse: bool) type {
            return struct {
                view: Self,
                /// Next logical offset, or one past it for reverse traversal.
                offset: usize,
                /// Coordinates of the next block's first (or reverse last) element.
                global: Coordinates,

                const Iterator = @This();

                fn init(v: Self) Iterator {
                    return .{
                        .view = v,
                        .offset = if (reverse) v.len else 0,
                        .global = if (v.len == 0 or v.mapping.contiguous) .zero else .unflatten(
                            v.mapping.packed_shape,
                            v.start + (if (reverse) v.len - 1 else 0),
                        ),
                    };
                }

                pub fn next(self: *Iterator) ?Block {
                    const v = &self.view;
                    if (if (reverse) self.offset == 0 else self.offset == v.len) return null;
                    if (v.mapping.contiguous) {
                        // The whole view is one run in every replica.
                        self.offset = if (reverse) 0 else v.len;
                        return v.blockAt(v.mapping.allocations.regions.get(0), 0, v.len, v.start * @sizeOf(T));
                    }
                    const location = v.locate(self.global);
                    const bounds = v.blockBounds(location.offset);
                    const len = if (reverse)
                        @min(location.offset + 1 - bounds.start, self.offset)
                    else
                        @min(bounds.end - location.offset, v.len - self.offset);
                    // A block is physically contiguous, so its last element's
                    // address also gives its start without decoding coordinates again.
                    const byte_offset = v.mapping.layout_metadata.byteOffset(location.offset, location.local) -
                        (if (reverse) (len - 1) * v.mapping.layout_metadata.element_size else 0);
                    const offset = if (reverse) self.offset - len else self.offset;
                    self.offset = if (reverse) offset else offset + len;
                    if (if (reverse) self.offset != 0 else self.offset != v.len) {
                        self.global.move(reverse, v.mapping.packed_shape, len);
                    }
                    return v.blockAt(location.region, offset, len, byte_offset);
                }
            };
        }

        const Bounds = struct { start: usize, end: usize };

        /// The block containing a shard-local element: elements contiguous in both
        /// physical memory and global row-major order. Bounds are shard-local
        /// row-major indices, not physical memory offsets.
        ///
        /// Example: an unsharded [2,5] array with row-major 2x2 tiles.
        ///
        /// 1. Logical indices, with tile boundaries. `_` denotes padding:
        /// ```text
        ///       tile 0    tile 1    tile 2
        ///       +-----+   +-----+   +-----+
        ///       | 0 1 |   | 2 3 |   | 4 _ |
        ///       | 5 6 |   | 7 8 |   | 9 _ |
        ///       +-----+   +-----+   +-----+
        /// ```
        /// 2. Tiles are stored consecutively, each in row-major order:
        /// ```text
        ///       [0 1 5 6] [2 3 7 8] [4 _ 9 _]
        /// ```
        /// Thus 0 and 1 are adjacent in memory, but 1 and 2 are not.
        /// `block_size = 2` is the maximum contiguous logical run.
        ///
        /// 3. Align runs within each logical row, clipping before padding:
        /// ```text
        ///       last_dim_size = 5
        ///       alignment_size = max(5, 2) = 5
        ///
        ///       alignment_start    blocks of logical elements      row end
        ///              0           [0 1] [2 3] [4]                    5
        ///              5           [5 6] [7 8] [9]                   10
        ///       actual block sizes:   2     2    1
        ///
        ///       local   alignment_start  start   end   returned bounds
        ///         4            0           4      5       [4,5)
        ///         5            5           5      7       [5,7)
        ///         6            5           5      7       [5,7)
        ///         9            5           9     10       [9,10)
        /// ```
        /// For local=6, start = 6 - (6-5)%2 = 5, end = min(5+2, 5+5) = 7.
        /// For local=4, end = min(4+2, 0+5) = 5: the final run is shortened.
        ///
        /// Without tiles, the same dense row-major array has block_size=10
        /// and alignment_size=10: one block [0,10) spanning both rows.
        fn blockBounds(self: *const Self, local: usize) Bounds {
            const block_size = self.mapping.layout_metadata.block_size;
            // Runs shorter than a row align within that row. Larger runs
            // span a whole number of rows, so alignment uses the run itself.
            const alignment_size = self.mapping.layout_metadata.alignment_size;
            const alignment_start = local - local % alignment_size;
            // Round the position relative to alignment_start down to a run boundary.
            const start = local - (local - alignment_start) % block_size;
            // A row's final run may be shorter; stop at the alignment interval's end.
            return .{ .start = start, .end = @min(start + block_size, alignment_start + alignment_size) };
        }

        fn blockAt(self: *const Self, region: HostPinnedAllocations.Region, offset: usize, len: usize, byte_offset: usize) Block {
            return .{
                .offset = offset,
                .len = len,
                .byte_offset = byte_offset,
                .replicas = self.mapping.allocations.replicas(region),
            };
        }

        fn itemAt(ptr: [*]u8, byte_offset: usize) *T {
            return @ptrCast(@alignCast(ptr + byte_offset));
        }

        const Location = struct {
            region: HostPinnedAllocations.Region,
            /// Coordinates of the element within its shard.
            local: Coordinates,
            /// Row-major index of the element within its shard.
            offset: usize,
        };

        /// Locate the region holding an element, retaining its shard-local
        /// coordinates for physical addressing.
        fn locate(self: *const Self, global: Coordinates) Location {
            const origins = self.mapping.allocations.entries.items(.origin);
            for (self.mapping.allocations.regions.constSlice()) |region| {
                const origin = origins[region.start];
                var local: Coordinates = .zero;
                var offset: usize = 0;
                for (self.mapping.shard_shape.dims(), 0..) |dim_, axis| {
                    const dim: usize = @intCast(dim_);
                    if (global.axes[axis] < origin.axes[axis] or global.axes[axis] - origin.axes[axis] >= dim) break;
                    local.axes[axis] = global.axes[axis] - origin.axes[axis];
                    offset = offset * dim + local.axes[axis];
                } else {
                    return .{ .region = region, .local = local, .offset = offset };
                }
            }
            @panic("The union of all regions should cover the global shape.");
        }
    };
}

/// A position in an array, one coordinate per axis. Axes past the rank stay zero.
const Coordinates = struct {
    axes: [Shape.MAX_RANK]usize,

    const zero: Coordinates = .{ .axes = @splat(0) };

    /// The global coordinates of the first element of a device's shard.
    fn ofShard(placement: *const Sharding.Placement, device: Sharding.Device) Coordinates {
        var result: Coordinates = .zero;
        for (placement.slices(device.coords).constSlice(), 0..) |slice, axis| result.axes[axis] = @intCast(slice.start);
        return result;
    }

    /// The coordinates of the element at `offset` of `shape`, in row-major order.
    fn unflatten(shape_: Shape, offset: usize) Coordinates {
        var result: Coordinates = .zero;
        var remaining = offset;
        var axis = shape_.rank();
        while (axis > 0) {
            axis -= 1;
            const dim: usize = @intCast(shape_.dim(axis));
            result.axes[axis] = remaining % dim;
            remaining /= dim;
        }
        return result;
    }

    /// Whether the coordinates match on the axes of `shape`.
    fn eqlWithin(self: Coordinates, other: Coordinates, shape_: Shape) bool {
        return std.mem.eql(usize, self.axes[0..shape_.rank()], other.axes[0..shape_.rank()]);
    }

    /// Advance in row-major order, decoding only axes that carry or borrow.
    /// The resulting coordinates must still be within the shape.
    fn move(self: *Coordinates, comptime reverse: bool, shape_: Shape, count: usize) void {
        const axes = &self.axes;
        var carry = count;
        var axis = shape_.rank();
        while (carry != 0 and axis > 0) {
            axis -= 1;
            const dim: usize = @intCast(shape_.dim(axis));
            if (reverse) {
                if (carry <= axes[axis]) {
                    axes[axis] -= carry;
                    return;
                }
                carry -= axes[axis] + 1;
                axes[axis] = dim - 1 - carry % dim;
            } else {
                const available = dim - axes[axis];
                if (carry < available) {
                    axes[axis] += carry;
                    return;
                }
                carry -= available;
                axes[axis] = carry % dim;
            }
            carry = carry / dim + 1;
        }
        std.debug.assert(carry == 0);
    }
};

/// Physical addressing and logical block boundaries prepared once per HostAccessible.
/// All metadata is stored by value; the source layout is needed only during init.
const LayoutMetadata = struct {
    rank: u4,
    element_size: usize,
    block_size: usize,
    alignment_size: usize,
    addressing: Addressing,

    const Addressing = union(enum) {
        row_major,
        strided: ByteStrides,
        tiled: Tiled,

        const ByteStrides = [Shape.MAX_RANK]usize;

        const Tiled = struct {
            /// Unit and padding dimensions always hold index 0 and are left out.
            digits: stdx.BoundedArray(Digit, Shape.MAX_RANK + pjrt.DefaultMemoryLayout.MAX_TILE_DIMS) = .empty,

            /// One physical dimension of a tiled layout, as a digit of one coordinate.
            const Digit = struct {
                byte_stride: usize,
                /// Divides the coordinate, unless `div_shift` replaces it for a power of two.
                divisor: usize,
                /// A mask when `is_mask` is set; otherwise a general modulus.
                modulus: usize,
                axis: u3,
                div_shift: ?std.math.Log2Int(usize),
                is_mask: bool,

                fn init(axis: u3, dim: TiledDims.Dim, byte_stride: usize) Digit {
                    // Unbounded digits keep every bit of the quotient.
                    const modulus = if (dim.modulus == 0) std.math.maxInt(usize) else dim.modulus;
                    const is_mask = dim.modulus == 0 or std.math.isPowerOfTwo(modulus);
                    return .{
                        .byte_stride = byte_stride,
                        .divisor = dim.divisor,
                        .modulus = if (is_mask and dim.modulus != 0) modulus - 1 else modulus,
                        .axis = axis,
                        .div_shift = if (std.math.isPowerOfTwo(dim.divisor)) @intCast(@ctz(dim.divisor)) else null,
                        .is_mask = is_mask,
                    };
                }

                fn value(self: Digit, coordinate: usize) usize {
                    const quotient = if (self.div_shift) |shift| coordinate >> shift else coordinate / self.divisor;
                    return if (self.is_mask) quotient & self.modulus else quotient % self.modulus;
                }
            };
        };
    };

    fn init(shape: Shape, shard_shape: Shape, layout: pjrt.MemoryLayout) LayoutMetadata {
        const last_dim_size: usize = if (shard_shape.rank() == 0) 1 else @intCast(shard_shape.dim(-1));
        const element_size: usize = shard_shape.dtype().sizeOf();

        // Longest stretch where stepping to the next local index also steps to the next element in memory.
        // Suppose a [2,5] shard tiled by (2,2): tiles are stored one after another, each row-major.
        // With local indices the memory holds:
        // `0 1 5 6 | 2 3 7 8 | 4 _ 9 _`
        // 0 and 1 are adjacent but 1 and 2 aren't, so the run is 2. A row-major layout is a single run.
        const addressing: Addressing, const physical_run: usize = addr: switch (layout) {
            .strides => |strides| {
                std.debug.assert(strides.byte_strides.len == shard_shape.rank());

                const row_major = shard_shape.computeByteStrides();
                if (std.mem.eql(i64, strides.byte_strides, row_major.constSlice())) break :addr .{ .row_major, shard_shape.count() };

                var byte_strides: Addressing.ByteStrides = @splat(0);
                for (strides.byte_strides, 0..) |stride, axis| {
                    stdx.debug.assert(stride >= 0, "negative destination byte strides are unsupported, got {}", .{stride});
                    byte_strides[axis] = @intCast(stride);
                }

                const physical_run =
                    // Rows padded to an alignment keep their elements adjacent: a run is a row.
                    if (byte_strides[shard_shape.rank() - 1] == element_size) last_dim_size
                    // Permuted axes, e.g. a column-major [3,4] f32 with strides [4,12], put the
                    // innermost axis elsewhere in memory, so no two elements of a row are adjacent.
                    else 1;
                break :addr .{ .{ .strided = byte_strides }, physical_run };
            },
            .tiled => |tiled| {
                stdx.debug.assert(tiled.minor_to_major.len == shard_shape.rank(), "layout rank {} doesn't match shape rank {}", .{ tiled.minor_to_major.len, shard_shape.rank() });

                if (tiled.tile_dims.len == 0 and isMinorToMajorRowMajor(tiled.minor_to_major, shard_shape.rank())) break :addr .{ .row_major, shard_shape.count() };

                // The expanded dimensions are laid out row-major.
                const expanded: TiledDims = .init(shard_shape, tiled);
                var tiled_addressing: Addressing.Tiled = .{};
                var byte_stride = element_size;
                // From the most minor physical dimension, the run grows while each
                // dimension is the next digit of the innermost axis.
                const last_axis = shard_shape.rank() -| 1;
                var physical_run: usize = 1;
                var run_grows = true;
                var i = expanded.dims.len;
                while (i > 0) {
                    i -= 1;
                    const dim = expanded.dims.get(i);
                    if (dim.size > 1) {
                        if (dim.axis) |axis| tiled_addressing.digits.appendAssumeCapacity(.init(axis, dim, byte_stride));
                        if (run_grows) {
                            const next_digit = if (dim.axis) |axis| axis == last_axis and dim.divisor == physical_run else false;
                            if (!next_digit) {
                                run_grows = false;
                            } else if (dim.modulus == 0) {
                                // An unbounded digit holds the rest of the axis.
                                physical_run = last_dim_size;
                                run_grows = false;
                            } else {
                                physical_run *= dim.modulus;
                                // Padding after the digit's last value ends the run.
                                run_grows = dim.size == dim.modulus;
                            }
                        }
                    }
                    byte_stride *= dim.size;
                }
                physical_run = @min(physical_run, last_dim_size);

                break :addr .{ .{ .tiled = tiled_addressing }, physical_run };
            },
        };

        // Longest stretch where stepping to the next local index also steps to the next global index.
        // Suppose a global [2,6] split into two [2,3] shards, each holding half the columns.
        // With global indices the shards hold:
        // - shard 1: `0 1 2 | 6 7 8`
        // - shard 2: `3 4 5 | 9 10 11`
        // 2 and 6 are adjacent in shard 1 but not globally, so the run is 3.
        const shard_run = run: {
            var run: usize = 1;
            var axis = shard_shape.rank();
            while (axis > 0) {
                axis -= 1;
                run *= @intCast(shard_shape.dim(axis));
                if (shape.dim(axis) != shard_shape.dim(axis)) break;
            }
            break :run run;
        };
        const block_size = if (shard_shape.count() == 0) 0 else @min(physical_run, shard_run);

        return .{
            .rank = shard_shape.rank(),
            .element_size = element_size,
            .block_size = block_size,
            // See `View.blockBounds`.
            .alignment_size = @max(last_dim_size, block_size),
            .addressing = addressing,
        };
    }

    /// The byte offset in its shard of the element at shard-local row-major index
    /// `local` and coordinates `coordinates`: row-major layouts use the index,
    /// the others the coordinates.
    fn byteOffset(self: *const LayoutMetadata, local: usize, coordinates: Coordinates) usize {
        switch (self.addressing) {
            .row_major => return local * self.element_size,
            .strided => |*byte_strides| {
                var offset: usize = 0;
                for (coordinates.axes[0..self.rank], byte_strides[0..self.rank]) |coord, byte_stride| offset += coord * byte_stride;
                return offset;
            },
            .tiled => |*tiled| {
                var offset: usize = 0;
                for (tiled.digits.constSlice()) |digit| offset += digit.value(coordinates.axes[digit.axis]) * digit.byte_stride;
                return offset;
            },
        }
    }
};

/// Whether a PJRT minor-to-major order lays the axes out row-major, like ZML shapes.
fn isMinorToMajorRowMajor(minor_to_major: []const i64, rank: usize) bool {
    if (minor_to_major.len != rank) return false;
    for (minor_to_major, 0..) |axis, i| {
        if (axis != @as(i64, @intCast(rank - i - 1))) return false;
    }
    return true;
}

/// The physical dimensions of a tiled layout, major to minor. XLA lays a tiled array
/// out row-major over these dimensions; see https://openxla.org/xla/tiled_layout.
const TiledDims = struct {
    dims: stdx.BoundedArray(Dim, Shape.MAX_RANK + 2 * pjrt.DefaultMemoryLayout.MAX_TILE_DIMS) = .empty,

    /// Every dimension holds the digit `(coordinate[axis] / divisor) % modulus`.
    const Dim = struct {
        /// Null for padding, which only holds index 0.
        axis: ?u3,
        size: usize,
        divisor: usize = 1,
        /// Zero for a digit without an upper bound, which holds `coordinate / divisor`.
        modulus: usize = 0,

        /// The index of the tile along this dimension.
        fn tileCount(self: Dim, tile: usize) Dim {
            const size = std.math.divCeil(usize, self.size, tile) catch unreachable;
            const axis = self.axis orelse return .{ .axis = null, .size = size };
            if (self.modulus == 0) return .{ .axis = axis, .size = size, .divisor = self.divisor * tile };
            if (self.modulus % tile == 0) return .{ .axis = axis, .size = size, .divisor = self.divisor * tile, .modulus = self.modulus / tile };
            // One tile holds every value of the digit.
            return .{ .axis = null, .size = size };
        }

        /// The index within the tile along this dimension.
        fn tileIndex(self: Dim, tile: usize) Dim {
            const axis = self.axis orelse return .{ .axis = null, .size = tile };
            // XLA assumes nested tiles add no padding.
            stdx.debug.assert(
                self.modulus == 0 or self.modulus % tile == 0 or tile % self.modulus == 0,
                "nested tile {} must divide, or be a multiple of, the {} values it tiles",
                .{ tile, self.modulus },
            );
            if (self.modulus != 0 and self.modulus < tile) return .{ .axis = axis, .size = tile, .divisor = self.divisor, .modulus = self.modulus };
            return .{ .axis = axis, .size = tile, .divisor = self.divisor, .modulus = tile };
        }
    };

    /// Follows XLA's LayoutUtil::LinearIndexForNestedTiling. The physical dimensions
    /// start in minor_to_major order. Each tile then splits the most minor ones,
    /// padded with leading unit dimensions if the tile has more, into tile counts
    /// followed by indices within the tile. Tile sizes are listed major to minor, so
    /// a later tile splits the previous tile's indices and possibly its counts.
    ///
    /// Two consequences are easy to get wrong:
    /// - `tile_dims[j]` of a k-dimensional tile belongs to `minor_to_major[k-1-j]`,
    ///   not to `minor_to_major[j]` nor to logical axis j. TPU's default
    ///   `{2,1,0:T(8,128)}` for rank-3 32-bit arrays tiles rows by 8 and columns
    ///   by 128.
    /// - A later tile splits the previous tile's dimensions, not the grid of
    ///   tiles: `(8,128)(2,1)`, the bf16 packing, interleaves pairs of rows
    ///   within each tile. TPU tiles bf16 and 8-bit arrays this way.
    fn init(shape: Shape, layout: pjrt.MemoryLayout.Tiled) TiledDims {
        const rank = shape.rank();
        stdx.debug.assert(layout.minor_to_major.len == rank, "layout rank {} doesn't match shape rank {}", .{ layout.minor_to_major.len, rank });
        var result: TiledDims = .{};
        var i = rank;
        while (i > 0) {
            i -= 1;
            const axis: u3 = @intCast(layout.minor_to_major[i]);
            result.dims.appendAssumeCapacity(.{ .axis = axis, .size = @intCast(shape.dim(axis)) });
        }
        var cursor: usize = 0;
        for (layout.tile_dims_sizes) |tile_rank| {
            const tile = layout.tile_dims[cursor..][0..tile_rank];
            cursor += tile_rank;
            while (result.dims.len < tile_rank) result.dims.insert(0, .{ .axis = null, .size = 1 }) catch unreachable;
            const start = result.dims.len - tile_rank;
            for (tile, start..) |size_, d| {
                // XLA marks dimensions combined with the next one by a negative size.
                stdx.debug.assert(size_ > 0, "unsupported tile dimension {}", .{size_});
                const size: usize = @intCast(size_);
                result.dims.appendAssumeCapacity(result.dims.get(d).tileIndex(size));
                result.dims.set(d, result.dims.get(d).tileCount(size));
            }
        }
        stdx.debug.assert(cursor == layout.tile_dims.len, "layout tile metadata is inconsistent", .{});
        return result;
    }
};

//
// =============
// === TESTS ===
// =============
//

/// The byte offset of an element, computed from the layout directly to check `LayoutMetadata`.
fn referenceByteOffset(shape_: Shape, layout: pjrt.MemoryLayout, index: []const usize) usize {
    if (!builtin.is_test) @compileError("referenceByteOffset is only available in tests");

    return switch (layout) {
        .tiled => |tiled| tiledElementOffset(shape_, tiled, index) * shape_.dtype().sizeOf(),
        .strides => |strides| b: {
            var byte_offset: i64 = 0;
            for (index, strides.byte_strides) |coord, stride| {
                stdx.debug.assert(stride >= 0, "negative destination byte strides are unsupported, got {}", .{stride});
                byte_offset += @as(i64, @intCast(coord)) * stride;
            }
            stdx.debug.assert(byte_offset >= 0, "destination byte offset must be non-negative, got {}", .{byte_offset});
            break :b @as(usize, @intCast(byte_offset));
        },
    };
}

/// Computes the element offset for one logical index in XLA/PJRT tiled layout order,
/// as XLA's LayoutUtil::LinearIndexForNestedTiling does. A direct port of that
/// function, kept independent of `TiledDims`: a reference sharing the prepared
/// model of tiling once let tests miss offsets that disagreed with XLA.
fn tiledElementOffset(shape_: Shape, layout: pjrt.MemoryLayout.Tiled, index: []const usize) usize {
    if (!builtin.is_test) @compileError("tiledElementOffset is only available in tests");

    const rank = shape_.rank();
    stdx.debug.assert(layout.minor_to_major.len == rank, "layout rank {} doesn't match shape rank {}", .{ layout.minor_to_major.len, rank });

    // Physical dimensions and indices, major to minor.
    const Expanded = stdx.BoundedArray(usize, Shape.MAX_RANK + 2 * pjrt.DefaultMemoryLayout.MAX_TILE_DIMS);
    var dims: Expanded = .empty;
    var indices: Expanded = .empty;
    var i = rank;
    while (i > 0) {
        i -= 1;
        const axis: usize = @intCast(layout.minor_to_major[i]);
        dims.appendAssumeCapacity(@intCast(shape_.dim(axis)));
        indices.appendAssumeCapacity(index[axis]);
    }

    var cursor: usize = 0;
    for (layout.tile_dims_sizes) |tile_rank| {
        const tile = layout.tile_dims[cursor..][0..tile_rank];
        cursor += tile_rank;
        while (dims.len < tile_rank) {
            dims.insert(0, 1) catch unreachable;
            indices.insert(0, 0) catch unreachable;
        }
        // Tiled dimensions become tile counts, followed by the indices within the tile.
        const start = dims.len - tile_rank;
        for (tile, start..) |tile_dim_, d| {
            const tile_dim: usize = @intCast(tile_dim_);
            dims.appendAssumeCapacity(tile_dim);
            indices.appendAssumeCapacity(indices.get(d) % tile_dim);
            dims.set(d, std.math.divCeil(usize, dims.get(d), tile_dim) catch unreachable);
            indices.set(d, indices.get(d) / tile_dim);
        }
    }
    stdx.debug.assert(cursor == layout.tile_dims.len, "layout tile metadata is inconsistent", .{});

    var offset: usize = 0;
    for (dims.constSlice(), indices.constSlice()) |dim, coord| offset = offset * dim + coord;
    return offset;
}

test "LayoutMetadata byte offsets match physical layouts" {
    const unit_tiles: [pjrt.DefaultMemoryLayout.MAX_TILE_DIMS]i64 = @splat(1);
    const tile_ranks: [pjrt.DefaultMemoryLayout.MAX_NUM_TILES]usize = @splat(Shape.MAX_RANK);
    const Case = struct { shape: Shape, layout: pjrt.MemoryLayout, block_size: usize };
    const cases = [_]Case{
        .{ .shape = .init(.{}, .i32), .layout = .{ .strides = .{ .byte_strides = &.{} } }, .block_size = 1 },
        .{ .shape = .init(.{ 3, 5 }, .i32), .layout = .{ .strides = .{ .byte_strides = &.{ 20, 4 } } }, .block_size = 15 },
        .{ .shape = .init(.{ 3, 5 }, .f16), .layout = .{ .strides = .{ .byte_strides = &.{ 16, 2 } } }, .block_size = 5 },
        .{ .shape = .init(.{ 3, 5 }, .i32), .layout = .{ .strides = .{ .byte_strides = &.{ 4, 16 } } }, .block_size = 1 },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .block_size = 15,
        },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .block_size = 1,
        },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} } },
            .block_size = 2,
        },
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} } },
            .block_size = 1,
        },
        .{
            .shape = .init(.{ 3, 5 }, .f16),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2, 2, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .block_size = 1,
        },
        .{
            .shape = .init(.{ 4, 8 }, .f16),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 4, 2, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .block_size = 1,
        },
        // The nested tile is larger than the values it tiles.
        .{
            .shape = .init(.{ 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2, 4, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .block_size = 1,
        },
        // The nested tile also splits the tile counts.
        .{
            .shape = .init(.{ 4, 8 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 1, 4, 2, 1, 1 }, .tile_dims_sizes = &.{ 2, 3 } } },
            .block_size = 1,
        },
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 4, 2 }, .tile_dims_sizes = &.{2} } },
            .block_size = 2,
        },
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 3, 2 }, .tile_dims_sizes = &.{2} } },
            .block_size = 2,
        },
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 3, 2, 3, 1 }, .tile_dims_sizes = &.{ 2, 2 } } },
            .block_size = 1,
        },
        // The nested tile moves pairs of the innermost axis to the most minor dimension.
        .{
            .shape = .init(.{ 2, 3, 5 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 2, 1 }, .tile_dims = &.{ 2, 4, 2, 2, 1 }, .tile_dims_sizes = &.{ 3, 2 } } },
            .block_size = 2,
        },
        // The tile has more dimensions than the shape.
        .{
            .shape = .init(.{5}, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{0}, .tile_dims = &.{ 2, 4 }, .tile_dims_sizes = &.{2} } },
            .block_size = 4,
        },
        .{
            .shape = .init(.{ 1, 1, 1, 1, 1, 1, 2, 3 }, .i32),
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 7, 6, 5, 4, 3, 2, 1, 0 }, .tile_dims = &unit_tiles, .tile_dims_sizes = &tile_ranks } },
            .block_size = 3,
        },
    };
    for (cases) |case| {
        const metadata: LayoutMetadata = .init(case.shape, case.shape, case.layout);
        try std.testing.expectEqual(case.block_size, metadata.block_size);
        for (0..case.shape.count()) |local| {
            const index = Coordinates.unflatten(case.shape, local);
            const expected = referenceByteOffset(case.shape, case.layout, index.axes[0..case.shape.rank()]);
            const actual = metadata.byteOffset(local, index);
            try std.testing.expectEqual(expected, actual);
            // Elements after the first of a block follow the previous one in memory.
            if ((local % metadata.alignment_size) % metadata.block_size != 0) {
                try std.testing.expectEqual(metadata.byteOffset(local - 1, Coordinates.unflatten(case.shape, local - 1)) + metadata.element_size, actual);
            }
        }
    }
}

test "tiled layout offsets follow XLA" {
    const Tiled = pjrt.MemoryLayout.Tiled;
    // Figure 1 of https://openxla.org/xla/tiled_layout: F32[3,5]{1,0:T(2,2)}.
    const tiles: Tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(17, tiledElementOffset(.init(.{ 3, 5 }, .f32), tiles, &.{ 2, 3 }));
    const shape_ = Shape.init(.{ 3, 4 }, .i32);
    try std.testing.expectEqual(0, tiledElementOffset(shape_, tiles, &.{ 0, 0 }));
    try std.testing.expectEqual(1, tiledElementOffset(shape_, tiles, &.{ 0, 1 }));
    try std.testing.expectEqual(2, tiledElementOffset(shape_, tiles, &.{ 1, 0 }));
    try std.testing.expectEqual(3, tiledElementOffset(shape_, tiles, &.{ 1, 1 }));
    try std.testing.expectEqual(4, tiledElementOffset(shape_, tiles, &.{ 0, 2 }));
    try std.testing.expectEqual(8, tiledElementOffset(shape_, tiles, &.{ 2, 0 }));

    // Figure 2: 4x8 tiled by (2,4)(2,1). The nested tile pairs rows within each
    // tile, as the (8,128)(2,1) tiling of bf16 on TPU does.
    const nested: Tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 2, 4, 2, 1 }, .tile_dims_sizes = &.{ 2, 2 } };
    const figure2 = [4][8]usize{
        .{ 0, 2, 4, 6, 8, 10, 12, 14 },
        .{ 1, 3, 5, 7, 9, 11, 13, 15 },
        .{ 16, 18, 20, 22, 24, 26, 28, 30 },
        .{ 17, 19, 21, 23, 25, 27, 29, 31 },
    };
    for (figure2, 0..) |offsets, row| {
        for (offsets, 0..) |offset, col| try std.testing.expectEqual(offset, tiledElementOffset(.init(.{ 4, 8 }, .bf16), nested, &.{ row, col }));
    }

    // A tile lists sizes for the most minor physical dimensions, major to minor.
    // Like {2,1,0:T(8,128)} on TPU, this tiles rows by 2 and columns by 4:
    // offset = a*32 + (b/2)*16 + (c/4)*8 + (b%2)*4 + c%4.
    const rank3: Shape = .init(.{ 2, 3, 5 }, .f32);
    const partial: Tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 2, 4 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(4, tiledElementOffset(rank3, partial, &.{ 0, 1, 0 }));
    try std.testing.expectEqual(8, tiledElementOffset(rank3, partial, &.{ 0, 0, 4 }));
    try std.testing.expectEqual(12, tiledElementOffset(rank3, partial, &.{ 0, 1, 4 }));
    try std.testing.expectEqual(16, tiledElementOffset(rank3, partial, &.{ 0, 2, 0 }));
    try std.testing.expectEqual(51, tiledElementOffset(rank3, partial, &.{ 1, 2, 3 }));

    // Tiles follow the physical order: offset = (c/2)*8 + (r/2)*4 + (c%2)*2 + r%2.
    const transposed: Tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{ 2, 2 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(1, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 1, 0 }));
    try std.testing.expectEqual(2, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 0, 1 }));
    try std.testing.expectEqual(4, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 2, 0 }));
    try std.testing.expectEqual(8, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 0, 2 }));
    try std.testing.expectEqual(14, tiledElementOffset(.init(.{ 3, 5 }, .f32), transposed, &.{ 2, 3 }));

    // A tile with more dimensions than the shape pads it with leading unit
    // dimensions: offset = (c/4)*8 + c%4.
    const wide: Tiled = .{ .minor_to_major = &.{0}, .tile_dims = &.{ 2, 4 }, .tile_dims_sizes = &.{2} };
    try std.testing.expectEqual(3, tiledElementOffset(.init(.{5}, .f32), wide, &.{3}));
    try std.testing.expectEqual(8, tiledElementOffset(.init(.{5}, .f32), wide, &.{4}));

    // A later tile can also split the previous tile counts:
    // offset = r*8 + (c%4)*2 + (c/4)%2.
    const cross_tile: Tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{ 1, 4, 2, 1, 1 }, .tile_dims_sizes = &.{ 2, 3 } };
    try std.testing.expectEqual(1, tiledElementOffset(.init(.{ 4, 8 }, .f32), cross_tile, &.{ 0, 4 }));
    try std.testing.expectEqual(2, tiledElementOffset(.init(.{ 4, 8 }, .f32), cross_tile, &.{ 0, 1 }));
    try std.testing.expectEqual(11, tiledElementOffset(.init(.{ 4, 8 }, .f32), cross_tile, &.{ 1, 5 }));
    try std.testing.expectEqual(31, tiledElementOffset(.init(.{ 4, 8 }, .f32), cross_tile, &.{ 3, 7 }));
}

test "HostAccessible.HostPinnedAllocations keeps each region's replicas together as allocations are added" {
    var allocations: HostAccessible.HostPinnedAllocations = .empty;
    defer allocations.deinit(std.testing.allocator);
    try allocations.entries.ensureTotalCapacity(std.testing.allocator, 5);
    var storage: [5]u8 = undefined;
    const base: [*]u8 = &storage;
    // A vector of 12 split into shards of 4, holding regions A, B, A, C, B:
    // replicas arrive interleaved with other regions.
    const shard_shape: Shape = .init(.{4}, .u8);
    for ([_]usize{ 0, 4, 0, 8, 4 }, 0..) |row, i| {
        allocations.add(shard_shape, .{ .origin = .{ .axes = .{row} ++ .{0} ** (Shape.MAX_RANK - 1) }, .ptr = base + i, .shard_index = undefined, .handle = undefined });
    }

    try std.testing.expectEqualSlices([*]u8, &.{ base, base + 2, base + 1, base + 4, base + 3 }, allocations.entries.items(.ptr));
    for (allocations.regions.constSlice(), [_][2]u8{ .{ 0, 2 }, .{ 2, 2 }, .{ 4, 1 } }) |region, expected| {
        try std.testing.expectEqual(expected, [2]u8{ region.start, region.len });
    }
}

test "HostAccessible.format prints the elements in row-major order" {
    const shape_ = Shape.init(.{ 2, 2 }, .i32);
    const column_major: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } };
    var storage = [_]i32{ 0, 2, 1, 3 };
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, column_major, &.{0}, &.{@ptrCast(&storage)});
    defer pinned.deinit(std.testing.allocator);
    var buffer: [128]u8 = undefined;
    const printed = try std.fmt.bufPrint(&buffer, "{f}", .{pinned});
    try std.testing.expectStringEndsWith(printed, ") { 0 1 2 3 }");
}

test "HostAccessible.View borrows the buffer's pointers through slicing" {
    const shape_ = Shape.init(.{4}, .i32);
    const layout: pjrt.MemoryLayout = .{ .strides = .{ .byte_strides = &.{4} } };
    var storage = [_]i32{ 0, 1, 2, 3 };
    var other = [_]i32{ 10, 11, 12, 13 };
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, layout, &.{0}, &.{@ptrCast(&storage)});
    defer pinned.deinit(std.testing.allocator);
    const nested = pinned.view(i32).slice(1, null).slice(0, 2);

    // Slices read the pointers in place, which lets `await` refresh them
    // without copying them into every view.
    try std.testing.expectEqual(@as(*const HostAccessible.Mapping, pinned.mapping), nested.mapping);
    pinned.mapping.allocations.entries.items(.ptr)[0] = @ptrCast(&other);
    nested.fill(99);
    try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3 }, &storage);
    try std.testing.expectEqualSlices(i32, &.{ 10, 99, 99, 13 }, &other);
}

test "HostAccessible.View blocks skip tile padding" {
    const shape_ = Shape.init(.{ 3, 5 }, .i32);
    const layout: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 1, 0 },
        .tile_dims = &.{ 2, 2 },
        .tile_dims_sizes = &.{2},
    } };
    var storage: [24]i32 = @splat(-1);
    const ptrs = [_][*]u8{@ptrCast(&storage)};
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, layout, &.{0}, &ptrs);
    defer pinned.deinit(std.testing.allocator);
    const values = pinned.view(i32);
    try std.testing.expectEqual(2, pinned.mapping.layout_metadata.block_size);
    try std.testing.expectEqual(HostAccessible.View(i32).Bounds{ .start = 0, .end = 2 }, values.blockBounds(1));
    try std.testing.expectEqual(HostAccessible.View(i32).Bounds{ .start = 4, .end = 5 }, values.blockBounds(4));
    var iterator = values.blocks();
    while (iterator.next()) |block| {
        for (0..block.replicas.len) |i| {
            const items = block.replica(i);
            for (items, block.offset..) |*value, index| value.* = @intCast(index + 1);
        }
    }
    try std.testing.expectEqualSlices(i32, &.{
        1,  2,  6,  7,  3,  4,  8,  9,  5,  -1, 10, -1,
        11, 12, -1, -1, 13, 14, -1, -1, 15, -1, -1, -1,
    }, &storage);
    values.slice(4, 3).fill(90);
    try std.testing.expectEqualSlices(i32, &.{
        1,  2,  90, 90, 3,  4,  8,  9,  90, -1, 10, -1,
        11, 12, -1, -1, 13, 14, -1, -1, 15, -1, -1, -1,
    }, &storage);
}

test "HostAccessible.View nested tiles" {
    const shape_ = Shape.init(.{ 3, 5 }, .i32);
    const layout: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 1, 0 },
        .tile_dims = &.{ 2, 2, 2, 1 },
        .tile_dims_sizes = &.{ 2, 2 },
    } };
    var storage: [24]i32 = @splat(-1);
    const ptrs = [_][*]u8{@ptrCast(&storage)};
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, layout, &.{0}, &ptrs);
    defer pinned.deinit(std.testing.allocator);
    const values = pinned.view(i32);
    // Pairs of rows are interleaved within each tile, so every block is one element.
    try std.testing.expectEqual(1, pinned.mapping.layout_metadata.block_size);
    values.copyFrom(&.{ 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15 });
    try std.testing.expectEqualSlices(i32, &.{
        1,  6,  2,  7,  3,  8,  4,  9,  5,  10, -1, -1,
        11, -1, 12, -1, 13, -1, 14, -1, 15, -1, -1, -1,
    }, &storage);
}

test "HostAccessible.View dense, transposed, strided and partial-rank tiled layouts" {
    const shape_ = Shape.init(.{ 2, 3 }, .i32);
    const Case = struct { layout: pjrt.MemoryLayout, block_size: usize, expected: []const i32 };
    const cases = [_]Case{
        .{
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .block_size = 6,
            .expected = &.{ 1, 2, 3, 4, 5, 6, -1, -1 },
        },
        .{
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
            .block_size = 1,
            .expected = &.{ 1, 4, 2, 5, 3, 6, -1, -1 },
        },
        .{
            .layout = .{ .strides = .{ .byte_strides = &.{ 12, 4 } } },
            .block_size = 6,
            .expected = &.{ 1, 2, 3, 4, 5, 6, -1, -1 },
        },
        .{
            .layout = .{ .strides = .{ .byte_strides = &.{ 20, 4 } } },
            .block_size = 3,
            .expected = &.{ 1, 2, 3, -1, -1, 4, 5, 6 },
        },
        .{
            .layout = .{ .strides = .{ .byte_strides = &.{ 4, 12 } } },
            .block_size = 1,
            .expected = &.{ 1, 4, -1, 2, 5, -1, 3, 6 },
        },
        .{
            .layout = .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{2}, .tile_dims_sizes = &.{1} } },
            .block_size = 3,
            .expected = &.{ 1, 2, 3, -1, 4, 5, 6, -1 },
        },
    };
    for (cases) |case| {
        var storage: [8]i32 = @splat(-1);
        const ptrs = [_][*]u8{@ptrCast(&storage)};
        const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, case.layout, &.{0}, &ptrs);
        defer pinned.deinit(std.testing.allocator);
        const values = pinned.view(i32);
        try std.testing.expectEqual(case.block_size, pinned.mapping.layout_metadata.block_size);
        values.copyFrom(&.{ 1, 2, 3, 4, 5, 6 });
        try std.testing.expectEqualSlices(i32, case.expected, &storage);
    }
}

test "HostAccessible.View row-major layouts return one block across rows and replicas" {
    const shape_: Shape = .init(.{ 3, 5 }, .i32);
    const layouts = [_]pjrt.MemoryLayout{
        .{ .strides = .{ .byte_strides = &.{ 20, 4 } } },
        .{ .tiled = .{ .minor_to_major = &.{ 1, 0 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } },
    };
    for (layouts) |layout| {
        var storage: [2][15]i32 = @splat(@splat(0));
        const ptrs = [_][*]u8{ @ptrCast(&storage[0]), @ptrCast(&storage[1]) };
        const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, layout, &.{ 0, 0 }, &ptrs);
        defer pinned.deinit(std.testing.allocator);
        const values = pinned.view(i32);
        for ([_]HostAccessible.View(i32){ values, values.slice(1, null).slice(1, 10) }) |selected| {
            var forward = selected.blocks();
            var reverse = selected.reverseBlocks();
            for ([_]HostAccessible.View(i32).Block{ forward.next().?, reverse.next().? }) |block| {
                try std.testing.expectEqual(0, block.offset);
                try std.testing.expectEqual(selected.len, block.len);
                try std.testing.expectEqual(2, block.replicas.len);
                for (0..block.replicas.len, &storage) |i, *replica| {
                    const items = block.replica(i);
                    try std.testing.expectEqual(replica[selected.start..].ptr, items.ptr);
                    try std.testing.expectEqual(selected.len, items.len);
                }
            }
            try std.testing.expectEqual(null, forward.next());
            try std.testing.expectEqual(null, reverse.next());
        }
    }
}

test "HostAccessible.View dense shards stop at gaps in global row-major order" {
    const Case = struct {
        shape: Shape,
        shard_shape: Shape,
        block_size: usize,
        origins: [2]usize,
        expected: [2][]const i32,
    };
    const cases = [_]Case{
        .{
            .shape = .init(.{ 2, 6 }, .i32),
            .shard_shape = .init(.{ 2, 3 }, .i32),
            .block_size = 3,
            .origins = .{ 0, 3 },
            .expected = .{ &.{ 0, 1, 2, 6, 7, 8 }, &.{ 3, 4, 5, 9, 10, 11 } },
        },
        .{
            .shape = .init(.{ 2, 6, 2 }, .i32),
            .shard_shape = .init(.{ 2, 3, 2 }, .i32),
            .block_size = 6,
            .origins = .{ 0, 6 },
            .expected = .{
                &.{ 0, 1, 2, 3, 4, 5, 12, 13, 14, 15, 16, 17 },
                &.{ 6, 7, 8, 9, 10, 11, 18, 19, 20, 21, 22, 23 },
            },
        },
    };
    for (cases) |case| {
        const strides = case.shard_shape.computeByteStrides();
        var storage: [2][12]i32 = @splat(@splat(-1));
        var ptrs: [2][*]u8 = undefined;
        for (&ptrs, &storage) |*ptr, *data| ptr.* = @ptrCast(data);
        const pinned = try HostAccessible.initForTests(std.testing.allocator, case.shape, case.shard_shape, .{ .strides = .{ .byte_strides = strides.constSlice() } }, &case.origins, &ptrs);
        defer pinned.deinit(std.testing.allocator);
        const elements = pinned.view(i32);
        try std.testing.expectEqual(case.block_size, pinned.mapping.layout_metadata.block_size);
        try std.testing.expectEqual(case.block_size, elements.blockBounds(0).end);
        var tail = elements.slice(case.block_size - 1, null).blocks();
        try std.testing.expectEqual(1, tail.next().?.len);

        elements.fillIota(0, 1);
        for (storage, case.expected) |data, expected| {
            try std.testing.expectEqualSlices(i32, expected, data[0..expected.len]);
        }
        var actual: [24]i32 = undefined;
        elements.copyTo(actual[0..case.shape.count()]);
        for (actual[0..case.shape.count()], 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(i)), value);

        // A partial range crosses both a global gap and a physical shard boundary.
        const start = case.block_size - 1;
        const values = [_]i32{ 90, 91, 92, 93 };
        elements.slice(start, values.len).copyFrom(&values);
        for (storage, case.expected) |data, expected| {
            for (data[0..expected.len], expected) |value, global| {
                const offset: usize = @intCast(global);
                try std.testing.expectEqual(if (offset >= start and offset < start + values.len) values[offset - start] else global, value);
            }
        }
        var copied: [4]i32 = undefined;
        elements.slice(start, copied.len).copyTo(&copied);
        try std.testing.expectEqualSlices(i32, &values, &copied);
    }
}

/// A 4x6 array split into 2x3 shards. Shards 4 to 7 are separate copies of shards 0 to 3.
const TestShards = struct {
    storage: [8][8]i32 = @splat(@splat(-1)),
    ptrs: [8][*]u8 = undefined,
    origins: [8]usize = @splat(0),
    pinned: ?HostAccessible = null,

    const shape_ = Shape.init(.{ 4, 6 }, .i32);
    const shard_shape = Shape.init(.{ 2, 3 }, .i32);
    const tiled: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 1, 0 },
        .tile_dims = &.{ 2, 2 },
        .tile_dims_sizes = &.{2},
    } };

    fn deinit(self: *TestShards) void {
        if (self.pinned) |pinned| pinned.deinit(std.testing.allocator);
    }

    /// The view borrows `pinned`, which borrows the storage: the storage must
    /// stay in place, and the next call replaces `pinned`.
    fn view(self: *TestShards, layout: pjrt.MemoryLayout) !HostAccessible.View(i32) {
        for (&self.ptrs, &self.origins, &self.storage, 0..) |*ptr, *origin, *data, i| {
            ptr.* = @ptrCast(data);
            const row = ((i % 4) / 2) * 2;
            const column = (i % 2) * 3;
            origin.* = row * 6 + column;
        }
        self.deinit();
        self.pinned = null;
        const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shard_shape, layout, &self.origins, &self.ptrs);
        self.pinned = pinned;
        return pinned.view(i32);
    }

    /// With the tiled layout, every copy of each element holds `expected[global]`
    /// and the tile padding is untouched.
    fn expectValues(self: *const TestShards, expected: [24]i32) !void {
        // Explicit offsets for the two rows of a 2x3 shard tiled by 2x2.
        const physical_offsets = [_]usize{ 0, 1, 4, 2, 3, 6 };
        for (self.storage, self.origins) |data, origin| {
            for (physical_offsets, 0..) |physical, local| {
                const global = origin + (local / 3) * 6 + local % 3;
                try std.testing.expectEqual(expected[global], data[physical]);
            }
            try std.testing.expectEqual(-1, data[5]);
            try std.testing.expectEqual(-1, data[7]);
        }
    }
};

test "HostAccessible.View blocks visit partial global ranges in every shard" {
    var fixture: TestShards = .{};
    defer fixture.deinit();
    const values = try fixture.view(TestShards.tiled);
    var visits: [24]u8 = @splat(0);
    // Nested slicing must keep block offsets relative to the selected range.
    var iterator = values.slice(1, null).slice(1, 18).blocks();
    while (iterator.next()) |block| {
        try std.testing.expect(block.offset + block.len <= 18);
        try std.testing.expectEqual(2, block.replicas.len);
        for (0..block.replicas.len) |i| {
            const items = block.replica(i);
            for (items, block.offset + 2..) |*value, global| {
                visits[global] += 1;
                value.* = @intCast(global);
            }
        }
    }
    var expected: [24]i32 = @splat(-1);
    for (visits, &expected, 0..) |count, *value, global| {
        const selected = global >= 2 and global < 20;
        try std.testing.expectEqual(@as(u8, if (selected) 2 else 0), count);
        if (selected) value.* = @intCast(global);
    }
    try fixture.expectValues(expected);
    // Starting after the first row partition must skip those shards entirely.
    values.set(17, 99);
    try std.testing.expectEqual(99, fixture.storage[3][4]);
    try std.testing.expectEqual(99, fixture.storage[7][4]);

    const Case = struct { step: i32, expected: [18]i32 };
    const cases = [_]Case{
        .{ .step = 3, .expected = .{ 7, 10, 13, 16, 19, 22, 25, 28, 31, 34, 37, 40, 43, 46, 49, 52, 55, 58 } },
        .{ .step = -2, .expected = .{ 7, 5, 3, 1, -1, -3, -5, -7, -9, -11, -13, -15, -17, -19, -21, -23, -25, -27 } },
        .{ .step = 0, .expected = @splat(7) },
    };
    for (cases) |case| {
        values.slice(1, null).slice(1, 18).fillIota(7, case.step);
        expected[2..20].* = case.expected;
        try fixture.expectValues(expected);
    }
}

test "HostAccessible.View blocks merge shuffled shards and group replicas" {
    var fixture: TestShards = .{};
    defer fixture.deinit();
    _ = try fixture.view(TestShards.tiled);
    const order = [_]usize{ 3, 4, 1, 6, 2, 0, 7, 5 };
    var ptrs: [8][*]u8 = undefined;
    var origins: [8]usize = undefined;
    for (order, &ptrs, &origins) |index, *ptr, *origin| {
        ptr.* = fixture.ptrs[index];
        origin.* = fixture.origins[index];
    }
    const pinned = try HostAccessible.initForTests(std.testing.allocator, TestShards.shape_, TestShards.shard_shape, TestShards.tiled, &origins, &ptrs);
    defer pinned.deinit(std.testing.allocator);
    const values = pinned.view(i32);
    const selected = values.slice(1, null).slice(1, 18);
    const offsets = [_]usize{ 0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16 };
    const lengths = [_]usize{ 1, 2, 1, 2, 1, 2, 1, 2, 1, 2, 1, 2 };
    const replicas = [_][2]usize{ .{ 4, 0 }, .{ 1, 5 }, .{ 6, 2 }, .{ 3, 7 } };
    const physical_offsets = [_]usize{ 0, 1, 4, 2, 3, 6 };
    var iterator = selected.blocks();
    for (offsets, lengths) |offset, len| {
        const block = iterator.next().?;
        try std.testing.expectEqual(offset, block.offset);
        try std.testing.expectEqual(len, block.len);
        try std.testing.expectEqual(2, block.replicas.len);
        const block_start = offset + 2;
        const region = block_start / 12 * 2 + block_start % 6 / 3;
        const physical = physical_offsets[block_start / 6 % 2 * 3 + block_start % 3];
        for (0..block.replicas.len, replicas[region]) |i, storage_index| {
            const items = block.replica(i);
            try std.testing.expectEqual(fixture.storage[storage_index][physical..].ptr, items.ptr);
        }
        for (0..block.replicas.len) |i| {
            const items = block.replica(i);
            for (items, offset + 2..) |*value, global| value.* = @intCast(global);
        }
    }
    try std.testing.expectEqual(null, iterator.next());
    var expected: [24]i32 = @splat(-1);
    for (expected[2..20], 2..) |*value, global| value.* = @intCast(global);
    try fixture.expectValues(expected);
    try expectReverseBlocksMirrorForward(selected);
}

test "HostAccessible.View blocks group the maximum number of replicas" {
    var storage: [Platform.MAX_NUM_DEVICES][4]i32 = @splat(@splat(-1));
    var ptrs: [Platform.MAX_NUM_DEVICES][*]u8 = undefined;
    const origins: [Platform.MAX_NUM_DEVICES]usize = @splat(0);
    for (&ptrs, &storage) |*ptr, *items| ptr.* = @ptrCast(items);
    const shape_: Shape = .init(.{4}, .i32);
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, .{ .strides = .{ .byte_strides = &.{4} } }, &origins, &ptrs);
    defer pinned.deinit(std.testing.allocator);
    const values = pinned.view(i32);
    var iterator = values.blocks();
    const block = iterator.next().?;
    try std.testing.expectEqual(0, block.offset);
    try std.testing.expectEqual(4, block.len);
    try std.testing.expectEqual(Platform.MAX_NUM_DEVICES, block.replicas.len);
    try std.testing.expectEqual(null, iterator.next());
    // The descriptors remain valid after advancing the iterator.
    for (0..block.replicas.len, &storage) |i, *replica| {
        const items = block.replica(i);
        try std.testing.expectEqual(replica[0..].ptr, items.ptr);
        @memset(items, 42);
        try std.testing.expectEqualSlices(i32, &.{ 42, 42, 42, 42 }, replica);
    }
    try expectReverseBlocksMirrorForward(values);
}

test "HostAccessible.View visits the maximum number of distinct shuffled regions" {
    var storage: [Platform.MAX_NUM_DEVICES]i32 = @splat(-1);
    var ptrs: [Platform.MAX_NUM_DEVICES][*]u8 = undefined;
    var origins: [Platform.MAX_NUM_DEVICES]usize = undefined;
    for (&ptrs, &origins, 0..) |*ptr, *origin, shard| {
        const index = storage.len - shard - 1;
        ptr.* = @ptrCast(&storage[index]);
        origin.* = index;
    }
    const shape_: Shape = .init(.{Platform.MAX_NUM_DEVICES}, .i32);
    const shard_shape: Shape = .init(.{1}, .i32);
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shard_shape, .{ .strides = .{ .byte_strides = &.{4} } }, &origins, &ptrs);
    defer pinned.deinit(std.testing.allocator);
    const values = pinned.view(i32);
    values.fillIota(0, 1);
    var forward = values.blocks();
    var reverse = values.reverseBlocks();
    for (storage, 0..) |value, index| {
        try std.testing.expectEqual(@as(i32, @intCast(index)), value);
        const block = forward.next().?;
        try std.testing.expectEqual(index, block.offset);
        try std.testing.expectEqual(1, block.len);
        try std.testing.expectEqual(1, block.replicas.len);
        try std.testing.expectEqual(storage[index..].ptr, block.firstReplica().ptr);
        const reverse_block = reverse.next().?;
        const reverse_index = storage.len - index - 1;
        try std.testing.expectEqual(reverse_index, reverse_block.offset);
        try std.testing.expectEqual(storage[reverse_index..].ptr, reverse_block.firstReplica().ptr);
    }
    try std.testing.expectEqual(null, forward.next());
    try std.testing.expectEqual(null, reverse.next());
}

test "HostAccessible.View multidimensional slices preserve coordinates across carries" {
    const shape_: Shape = .init(.{ 2, 3, 5 }, .i32);
    const layouts = [_]pjrt.MemoryLayout{
        .{ .strides = .{ .byte_strides = &.{ 60, 20, 4 } } },
        .{ .tiled = .{ .minor_to_major = &.{ 2, 1, 0 }, .tile_dims = &.{ 3, 2 }, .tile_dims_sizes = &.{2} } },
    };
    for (layouts) |layout| {
        var storage: [48]i32 = @splat(-1);
        const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, layout, &.{0}, &.{@ptrCast(&storage)});
        defer pinned.deinit(std.testing.allocator);
        const values = pinned.view(i32);
        values.fillIota(0, 1);
        var expected: [48]i32 = @splat(-1);
        for (0..shape_.count()) |local| {
            const index = Coordinates.unflatten(shape_, local);
            const physical = referenceByteOffset(shape_, layout, index.axes[0..shape_.rank()]) / @sizeOf(i32);
            expected[physical] = @intCast(local);
            try std.testing.expectEqual(@as(i32, @intCast(local)), values.get(local));
        }
        try std.testing.expectEqualSlices(i32, &expected, &storage);
        for (0..shape_.count() + 1) |start| {
            for (0..shape_.count() - start + 1) |len| {
                try expectReverseBlocksMirrorForward(values.slice(start, len));
            }
        }
    }
}

test "HostAccessible.View reverse blocks mirror forward blocks" {
    var fixture: TestShards = .{};
    defer fixture.deinit();
    const layouts = [_]pjrt.MemoryLayout{
        TestShards.tiled,
        .{ .strides = .{ .byte_strides = &.{ 12, 4 } } },
        .{ .strides = .{ .byte_strides = &.{ 4, 8 } } },
    };
    for (layouts) |layout| {
        const values = try fixture.view(layout);
        for (0..values.len + 1) |start| {
            for (0..values.len - start + 1) |len| {
                try expectReverseBlocksMirrorForward(values.slice(start, len));
            }
        }
    }
    // Dense blocks can span multiple rows, and scalar blocks have no row axis.
    const dense_pinned = try HostAccessible.initForTests(std.testing.allocator, TestShards.shard_shape, TestShards.shard_shape, layouts[1], fixture.origins[0..1], fixture.ptrs[0..1]);
    defer dense_pinned.deinit(std.testing.allocator);
    const dense = dense_pinned.view(i32);
    try expectReverseBlocksMirrorForward(dense.slice(1, 4));
    const scalar_shape: Shape = .init(.{}, .i32);
    const scalar_pinned = try HostAccessible.initForTests(std.testing.allocator, scalar_shape, scalar_shape, .{ .strides = .{ .byte_strides = &.{} } }, fixture.origins[0..1], fixture.ptrs[0..1]);
    defer scalar_pinned.deinit(std.testing.allocator);
    const scalar = scalar_pinned.view(i32);
    try expectReverseBlocksMirrorForward(scalar);
}

fn expectReverseBlocksMirrorForward(values: HostAccessible.View(i32)) !void {
    var expected: [48]HostAccessible.View(i32).Block = undefined;
    var len: usize = 0;
    var offset: usize = 0;
    var forward = values.blocks();
    while (forward.next()) |block| : (len += 1) {
        try std.testing.expectEqual(offset, block.offset);
        expected[len] = block;
        offset += block.len;
    }
    try std.testing.expectEqual(values.len, offset);
    var reverse = values.reverseBlocks();
    while (len > 0) {
        len -= 1;
        const block = reverse.next().?;
        try std.testing.expectEqual(expected[len].offset, block.offset);
        try std.testing.expectEqual(expected[len].len, block.len);
        try std.testing.expectEqual(expected[len].byte_offset, block.byte_offset);
        try std.testing.expectEqualSlices([*]u8, expected[len].replicas, block.replicas);
    }
    try std.testing.expectEqual(null, reverse.next());
}

test "HostAccessible.View scalar, empty ranges and rank-one padding" {
    var scalar: i32 = 0;
    const scalar_ptrs = [_][*]u8{@ptrCast(&scalar)};
    const shape_ = Shape.scalar(.i32);
    const scalar_layout: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{}, .tile_dims = &.{}, .tile_dims_sizes = &.{} } };
    const scalar_pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, scalar_layout, &.{0}, &scalar_ptrs);
    defer scalar_pinned.deinit(std.testing.allocator);
    const scalar_view = scalar_pinned.view(i32);
    scalar_view.copyFrom(&.{42});
    try std.testing.expectEqual(42, scalar);
    var past_end = scalar_view.slice(1, null).blocks();
    try std.testing.expectEqual(null, past_end.next());

    var storage: [8]bool = @splat(false);
    const vector_ptrs = [_][*]u8{@ptrCast(&storage)};
    const vector_shape = Shape.init(.{5}, .bool);
    const vector_layout: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{0}, .tile_dims = &.{4}, .tile_dims_sizes = &.{1} } };
    const vector_pinned = try HostAccessible.initForTests(std.testing.allocator, vector_shape, vector_shape, vector_layout, &.{0}, &vector_ptrs);
    defer vector_pinned.deinit(std.testing.allocator);
    const vector = vector_pinned.view(bool);
    vector.fill(true);
    try std.testing.expectEqualSlices(bool, &.{ true, true, true, true, true, false, false, false }, &storage);

    const empty_shape = Shape.init(.{0}, .bool);
    const empty_pinned = try HostAccessible.initForTests(std.testing.allocator, empty_shape, empty_shape, vector_layout, &.{0}, &vector_ptrs);
    defer empty_pinned.deinit(std.testing.allocator);
    const empty_view = empty_pinned.view(bool);
    var empty_blocks = empty_view.blocks();
    try std.testing.expectEqual(null, empty_blocks.next());

    // The empty view has no storage or metadata to read.
    const none: HostAccessible.View(u32) = .empty;
    none.fill(0);
    none.copyTo(&.{});
    var none_blocks = none.reverseBlocks();
    try std.testing.expectEqual(null, none_blocks.next());
}

test "HostAccessible devices holding a shard read one allocation" {
    const platform = testing.env();
    const shape_ = Shape.init(.{ .b = @as(i64, @intCast(platform.devices.len)), .d = 5 }, .i32);
    try expectDevicesReadShards(platform, shape_);
    try expectDevicesReadShards(platform, shape_.withPartitioning(platform.meshes.get("model").?, .{ .b = .model }));
}

test "HostAccessible devices holding a shard read one allocation in data-parallel groups" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Two data-parallel groups of two tensor-parallel devices.
    const platform = Platform.auto(allocator, io, .{
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .cpu = .{ .device_count = 4 },
    }) catch return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    const mesh: Sharding.Mesh = try .init(
        "data_parallel",
        &platform.physical_mesh,
        .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .parseBindings(.{ .batch = .link_x, .model = .link_y }),
    );
    try expectDevicesReadShards(platform, Shape.init(.{ .b = 4, .d = 5 }, .i32).withPartitioning(&mesh, .{ .b = .batch }));
}

/// Rows of `.d` elements written once through a view reach every device
/// holding them, and devices holding the same shard share its allocation
/// where the platform allows it.
fn expectDevicesReadShards(platform: *const Platform, shape_: Shape) !void {
    if (!builtin.is_test) @compileError("expectDevicesReadShards is only available in tests");

    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const api = platform.pjrt_api;
    const pinned = try HostAccessible.init(allocator, io, platform, shape_, .undonatable);
    defer pinned.deinit(allocator);
    const placement: Sharding.Placement = try .init(shape_);
    const expected_allocations = switch (platform.target) {
        .cpu, .cuda, .rocm => shape_.count() / placement.shape.count(),
        else => platform.devices.len,
    };
    try std.testing.expectEqual(expected_allocations, pinned.mapping.allocations.entries.len);

    pinned.view(i32).fillIota(0, 1);
    const row_size: usize = @intCast(shape_.dim(.d));
    const values = try allocator.alloc(i32, placement.shape.count());
    defer allocator.free(values);
    for (pinned.buffer._shards.constSlice(), platform.physical_mesh.devices_in_canonical_order) |shard, device| {
        @memset(values, -1);
        if (try shard.toHostBuffer(api, std.mem.sliceAsBytes(values))) |event| {
            defer event.deinit(api);
            try event.await(api, io);
        }
        const first_row = Coordinates.ofShard(&placement, device).axes[0];
        for (values, 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(first_row * row_size + i)), value);
    }
}

/// Like `zml/io.zig`'s, a 2x2 torus over the first four devices.
fn buildMesh2x2(
    allocator: std.mem.Allocator,
    target: @import("../platform.zig").Target,
    devices: []const @import("../platform.zig").Device,
) !Sharding.PhysicalMesh {
    if (!builtin.is_test) @compileError("buildMesh2x2 is only available in tests");

    if (devices.len < 4) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[0]),
            .device(devices[1]),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[2]),
            .device(devices[3]),
        }),
    });
    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

test "HostAccessible executions read shared inputs on every device" {
    const zml = @import("../zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const api = platform.pjrt_api;
    const shape_ = Shape.init(.{ 3, 5 }, .i32);
    const Test = struct {
        fn increment(input: zml.Tensor) zml.Tensor {
            return input.onMemory(.host_pinned).toMemory(.device).addConstant(1);
        }
    };
    var exe = try platform.compileFn(allocator, io, Test.increment, .{zml.Tensor.fromShape(shape_)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    const pinned = try HostAccessible.init(allocator, io, platform, shape_, .undonatable);
    defer pinned.deinit(allocator);
    for (0..2) |iteration| {
        pinned.view(i32).fillIota(@intCast(iteration * 100), 1);
        var output: Buffer = undefined;
        runner.run(io, .{pinned.buffer}, .{&output}, .{});
        defer output.deinit();
        try output.await(io);
        for (output._shards.constSlice()) |shard| {
            var values: [15]i32 = @splat(-1);
            if (try shard.toHostBuffer(api, std.mem.asBytes(&values))) |event| {
                defer event.deinit(api);
                try event.await(api, io);
            }
            for (values, 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(iteration * 100 + i + 1)), value);
        }
    }
}

test "HostAccessible.View.constSlice borrows contiguous buffers and copies the others" {
    const shape_ = Shape.init(.{ 2, 3 }, .i32);
    var scratch: [6]i32 = @splat(-1);

    var dense: [2][6]i32 = .{ .{ 0, 1, 2, 3, 4, 5 }, @splat(-1) };
    const dense_ptrs = [_][*]u8{ @ptrCast(&dense[0]), @ptrCast(&dense[1]) };
    const dense_pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, .{ .strides = .{ .byte_strides = &.{ 12, 4 } } }, &.{ 0, 0 }, &dense_ptrs);
    defer dense_pinned.deinit(std.testing.allocator);
    const borrowed = dense_pinned.view(i32).slice(1, 4).constSlice(&scratch);
    try std.testing.expectEqual(dense[0][1..].ptr, borrowed.ptr);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, borrowed);

    var column_major = [_]i32{ 0, 3, 1, 4, 2, 5 };
    const transposed: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } };
    const transposed_pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, transposed, &.{0}, &.{@ptrCast(&column_major)});
    defer transposed_pinned.deinit(std.testing.allocator);
    const copied = transposed_pinned.view(i32).slice(1, 4).constSlice(&scratch);
    try std.testing.expectEqual(@as([*]const i32, &scratch), copied.ptr);
    try std.testing.expectEqualSlices(i32, &.{ 1, 2, 3, 4 }, copied);

    const none: HostAccessible.View(i32) = .empty;
    try std.testing.expectEqual(0, none.constSlice(&.{}).len);
}

test "HostAccessible.View.edit writes contiguous buffers in place and commits every replica" {
    const shape_ = Shape.init(.{ 2, 3 }, .i32);
    var scratch: [6]i32 = @splat(-1);

    // In place: the first replica takes the writes, and `commit` copies it to the others.
    var dense: [2][6]i32 = @splat(@splat(0));
    const dense_ptrs = [_][*]u8{ @ptrCast(&dense[0]), @ptrCast(&dense[1]) };
    const dense_pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, .{ .strides = .{ .byte_strides = &.{ 12, 4 } } }, &.{ 0, 0 }, &dense_ptrs);
    defer dense_pinned.deinit(std.testing.allocator);
    const dense_edit = dense_pinned.view(i32).slice(1, 4).edit(&scratch);
    try std.testing.expectEqual(dense[0][1..].ptr, dense_edit.items.ptr);
    @memcpy(dense_edit.items, &[_]i32{ 1, 2, 3, 4 });
    dense_edit.commit();
    for (dense) |replica| try std.testing.expectEqualSlices(i32, &.{ 0, 1, 2, 3, 4, 0 }, &replica);

    // Through the scratch: the buffer only changes on `commit`.
    var column_major: [6]i32 = @splat(0);
    const transposed: pjrt.MemoryLayout = .{ .tiled = .{ .minor_to_major = &.{ 0, 1 }, .tile_dims = &.{}, .tile_dims_sizes = &.{} } };
    const transposed_pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, transposed, &.{0}, &.{@ptrCast(&column_major)});
    defer transposed_pinned.deinit(std.testing.allocator);
    const copied_edit = transposed_pinned.view(i32).slice(1, 4).edit(&scratch);
    try std.testing.expectEqual(@as([*]i32, &scratch), copied_edit.items.ptr);
    @memcpy(copied_edit.items, &[_]i32{ 1, 2, 3, 4 });
    try std.testing.expectEqualSlices(i32, &.{ 0, 0, 0, 0, 0, 0 }, &column_major);
    copied_edit.commit();
    try std.testing.expectEqualSlices(i32, &.{ 0, 3, 1, 4, 2, 0 }, &column_major);

    const none: HostAccessible.View(i32) = .empty;
    none.edit(&.{}).commit();
}

test "HostAccessible.View.indexOfScalar and indexOfAny search blocks in row-major order" {
    var fixture: TestShards = .{};
    defer fixture.deinit();
    const values = try fixture.view(TestShards.tiled);
    values.fillIota(0, 1);
    values.set(20, 7);
    // Tiles split the view into blocks, and shards into regions.
    try std.testing.expectEqual(7, values.indexOfScalar(7));
    try std.testing.expectEqual(12, values.slice(8, null).indexOfScalar(7));
    try std.testing.expectEqual(null, values.slice(8, 12).indexOfScalar(7));
    try std.testing.expectEqual(null, values.indexOfScalar(-1));
    try std.testing.expectEqual(null, HostAccessible.View(i32).empty.indexOfScalar(0));

    try std.testing.expectEqual(5, values.indexOfAny(&.{ 7, 5 }));
    try std.testing.expectEqual(12, values.slice(8, null).indexOfAny(&.{ -1, 7 }));
    try std.testing.expectEqual(null, values.slice(8, 12).indexOfAny(&.{ 7, -1 }));
    try std.testing.expectEqual(null, values.indexOfAny(&.{}));
}

test "HostAccessible.View indexes replicas of a dense array directly" {
    const shape_ = Shape.init(.{ 2, 3 }, .i32);
    var storage: [2][6]i32 = @splat(@splat(-1));
    const ptrs = [_][*]u8{ @ptrCast(&storage[0]), @ptrCast(&storage[1]) };
    const pinned = try HostAccessible.initForTests(std.testing.allocator, shape_, shape_, .{ .strides = .{ .byte_strides = &.{ 12, 4 } } }, &.{ 0, 0 }, &ptrs);
    defer pinned.deinit(std.testing.allocator);
    try std.testing.expect(pinned.mapping.contiguous);
    const values = pinned.view(i32);
    for (0..6) |i| values.set(i, @intCast(10 + i));
    values.slice(4, null).set(1, 99);
    for (storage) |replica| try std.testing.expectEqualSlices(i32, &.{ 10, 11, 12, 13, 14, 99 }, &replica);
    try std.testing.expectEqual(99, values.get(5));
    try std.testing.expectEqual(13, values.slice(2, 3).get(1));

    inline for (.{ false, true }) |reverse| {
        const nested = values.slice(1, 4);
        var blocks = if (reverse) nested.reverseBlocks() else nested.blocks();
        const block = blocks.next().?;
        try std.testing.expectEqual(0, block.offset);
        try std.testing.expectEqual(2, block.replicas.len);
        for (0..block.replicas.len, &storage) |i, *replica| {
            const items = block.replica(i);
            try std.testing.expectEqualSlices(i32, &.{ 11, 12, 13, 14 }, items);
            try std.testing.expectEqual(replica[1..].ptr, items.ptr);
        }
        try std.testing.expectEqual(null, blocks.next());
    }
}

test "HostAccessible.View set and get address single elements in every replica" {
    var fixture: TestShards = .{};
    defer fixture.deinit();
    const values = try fixture.view(TestShards.tiled);
    var expected: [24]i32 = undefined;
    var i: usize = expected.len;
    while (i > 0) {
        i -= 1;
        expected[i] = @intCast(100 + i);
        values.set(i, expected[i]);
    }
    try fixture.expectValues(expected);
    for (expected, 0..) |value, index| try std.testing.expectEqual(value, values.get(index));

    // Indices of a nested view are relative to its start.
    const nested = values.slice(7, 10);
    nested.set(3, -5);
    expected[10] = -5;
    try fixture.expectValues(expected);
    try std.testing.expectEqual(-5, nested.get(3));
}

test "HostAccessible.View copies across layouts, shards and replicas" {
    var fixture: TestShards = .{};
    defer fixture.deinit();
    const values = try fixture.view(TestShards.tiled);
    values.fill(0);
    values.slice(2, 20).fillIota(10, 1);
    values.set(17, 99);
    const expected = [_]i32{ 0, 0, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 99, 26, 27, 28, 29, 0, 0 };
    var actual: [24]i32 = undefined;
    values.copyTo(&actual);
    try std.testing.expectEqualSlices(i32, &expected, &actual);
    for (expected, 0..) |value, i| try std.testing.expectEqual(value, values.get(i));
    try fixture.expectValues(expected);

    // Blocks expose every replica, respect physical boundaries, and stop at
    // the end of a nested view even when the physical run continues.
    const nested = values.slice(1, null).slice(2, 4);
    var nested_blocks = nested.blocks();
    const first = nested_blocks.next().?;
    try std.testing.expectEqual(0, first.offset);
    try std.testing.expectEqual(2, first.replicas.len);
    for (0..first.replicas.len, [_]usize{ 1, 5 }) |i, storage_index| {
        const items = first.replica(i);
        try std.testing.expectEqualSlices(i32, expected[3..5], items);
        try std.testing.expectEqual(fixture.storage[storage_index][0..].ptr, items.ptr);
    }
    const middle = nested_blocks.next().?;
    try std.testing.expectEqual(2, middle.offset);
    for (0..middle.replicas.len) |i| try std.testing.expectEqualSlices(i32, expected[5..6], middle.replica(i));
    const last = nested_blocks.next().?;
    try std.testing.expectEqual(3, last.offset);
    for (0..last.replicas.len, [_]usize{ 0, 4 }) |i, storage_index| {
        const items = last.replica(i);
        try std.testing.expectEqualSlices(i32, expected[6..7], items);
        try std.testing.expectEqual(fixture.storage[storage_index][2..].ptr, items.ptr);
    }
    try std.testing.expectEqual(null, nested_blocks.next());

    // Copy between differently shaped/tiled ranges, including a nonzero source
    // and destination start, and verify every destination replica physically.
    const dst_shape = Shape.init(.{ 3, 7 }, .i32);
    const dst_layout: pjrt.MemoryLayout = .{ .tiled = .{
        .minor_to_major = &.{ 0, 1 },
        .tile_dims = &.{},
        .tile_dims_sizes = &.{},
    } };
    var dst_storage: [2][21]i32 = @splat(@splat(-1));
    const dst_ptrs = [_][*]u8{ @ptrCast(&dst_storage[0]), @ptrCast(&dst_storage[1]) };
    const dst_pinned = try HostAccessible.initForTests(std.testing.allocator, dst_shape, dst_shape, dst_layout, &.{ 0, 0 }, &dst_ptrs);
    defer dst_pinned.deinit(std.testing.allocator);
    const dst = dst_pinned.view(i32);
    dst.slice(1, 19).copyFromView(values.slice(1, 22).slice(1, 19));
    for (dst_storage) |data| {
        for (0..21) |i| {
            const physical = (i % 7) * 3 + i / 7;
            try std.testing.expectEqual(if (i > 0 and i < 20) expected[i + 1] else -1, data[physical]);
        }
    }
    var empty: [0]i32 = .{};
    values.slice(values.len, 0).copyTo(&empty);
    values.slice(0, 0).copyFromView(dst.slice(0, 0));
    values.slice(values.len, null).fill(-1);
    values.slice(values.len, null).fillIota(-1, 1);
    values.slice(values.len, null).copyFrom(&empty);
    values.copyTo(&actual);
    try std.testing.expectEqualSlices(i32, &expected, &actual);
}

test "HostAccessible await after executable donation permits the next host update" {
    const zml = @import("../zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const shape_ = Shape.init(.{ 3, 5 }, .i32);
    const Test = struct {
        fn increment(input: zml.Tensor) zml.Tensor {
            const host_input = input.onMemory(.host_pinned);
            return host_input.toMemory(.device).addConstant(1).toMemory(.host_pinned).reuseBuffer(host_input);
        }
    };
    var exe = try platform.compileFn(allocator, io, Test.increment, .{zml.Tensor.fromShape(shape_)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    var pinned = try HostAccessible.init(allocator, io, platform, shape_, .donatable);
    defer pinned.deinit(allocator);
    try std.testing.expectEqual(pinned.buffer._shards.len, pinned.mapping.allocations.entries.len);
    const first_ptr = pinned.mapping.allocations.entries.items(.ptr)[0];
    pinned.view(i32).fillIota(0, 1);

    // Reuse the same runner and output destination as the model execution paths.
    // Only await waits for the asynchronous executable before host access.
    for (0..3) |iteration| {
        runner.run(io, .{pinned.buffer}, .{&pinned.buffer}, .{});
        try pinned.await(io);

        // Reacquire from the output handles, even when donation reuses memory.
        const values = pinned.view(i32);
        var actual: [15]i32 = undefined;
        values.copyTo(&actual);
        try std.testing.expectEqual(first_ptr, pinned.mapping.allocations.entries.items(.ptr)[0]);
        for (actual, 0..) |value, i| {
            const expected: i32 = if (iteration == 0 or i < 3 or i >= 12)
                @intCast(i + iteration + 1)
            else
                @intCast(iteration * 100 + i - 3 + 1);
            try std.testing.expectEqual(expected, value);
        }

        // Also check the executable's logical output through PJRT, independently
        // of the host-visible layout used above.
        var device_values: [15]i32 = undefined;
        try pinned.buffer.toSlice(io, .init(shape_, std.mem.asBytes(&device_values)));
        try std.testing.expectEqualSlices(i32, &actual, &device_values);

        values.slice(3, 9).fillIota(@intCast((iteration + 1) * 100), 1);
    }
}

test "HostAccessible await follows an output that doesn't reuse the pinned allocation" {
    const zml = @import("../zml.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    const shape_ = Shape.init(.{ 3, 5 }, .i32);
    const Test = struct {
        fn increment(input: zml.Tensor) zml.Tensor {
            // Without reuseBuffer, execution writes a new host allocation.
            return input.onMemory(.host_pinned).toMemory(.device).addConstant(1).toMemory(.host_pinned);
        }
    };
    var exe = try platform.compileFn(allocator, io, Test.increment, .{zml.Tensor.fromShape(shape_)}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);

    var pinned = try HostAccessible.init(allocator, io, platform, shape_, .donatable);
    defer pinned.deinit(allocator);
    pinned.view(i32).fillIota(0, 1);

    for (0..2) |iteration| {
        // The input isn't donated, so release it once execution replaced it.
        var input = pinned.buffer;
        defer input.deinit();
        const previous = pinned.mapping.allocations.entries.items(.ptr)[0];
        runner.run(io, .{input}, .{&pinned.buffer}, .{});
        try pinned.await(io);
        const values = pinned.view(i32);
        try std.testing.expect(pinned.mapping.allocations.entries.items(.ptr)[0] != previous);

        var actual: [15]i32 = undefined;
        values.copyTo(&actual);
        for (actual, 0..) |value, i| try std.testing.expectEqual(@as(i32, @intCast(iteration * 100 + i + 1)), value);
        var device_values: [15]i32 = undefined;
        try pinned.buffer.toSlice(io, .init(shape_, std.mem.asBytes(&device_values)));
        try std.testing.expectEqualSlices(i32, &actual, &device_values);

        // Host writes now land in the output, which the next execution reads.
        values.fillIota(@intCast((iteration + 1) * 100), 1);
    }
}
