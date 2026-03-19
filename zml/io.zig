const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const CreateOptions = @import("platform.zig").CreateOptions;
const mem = @import("mem.zig");
const Memory = @import("platform.zig").Memory;
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;
const sharding_ = @import("sharding.zig");
const Partitioner = sharding_.Partitioner;
const Partitioning = sharding_.Partitioning;
const Placement = sharding_.Placement;
const Sharding = sharding_.Sharding;
const Slice = @import("slice.zig").Slice;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/io");

pub const TensorStore = struct {
    registry: *safetensors.TensorRegistry,
    id_map: std.AutoHashMapUnmanaged(usize, *safetensors.Tensor),
    allocator: std.mem.Allocator,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry) TensorStore {
        return .{
            .registry = registry,
            .id_map = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.id_map.deinit(self.allocator);
    }

    fn bindIdToKey(self: *TensorStore, key: []const u8, id: usize) !void {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key).?;

        const gop = try self.id_map.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Key {s} already has an associated tensor (id: {})", .{ key, gop.key_ptr.* });
        }
        errdefer self.id_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = tensor_desc_ptr;
    }

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.id_map.get(id) orelse return null;
        return tensor_desc_ptr;
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;

        return safetensors.TensorReader.init(io, tensor_desc.*, buffer, .{});
    }

    pub fn view(self: *TensorStore) View {
        return .{ .store = self };
    }

    pub const View = struct {
        store: *TensorStore,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn root(self: *const View) View {
            return .{
                .store = self.store,
            };
        }

        pub fn parent(self: *const View) View {
            const slice = self.prefix() orelse unreachable;
            const index = std.mem.lastIndexOfScalar(u8, slice[0 .. slice.len - 1], '.') orelse return self.root();
            var buffer: [256]u8 = undefined;
            @memcpy(buffer[0 .. index + 1], slice[0 .. index + 1]);
            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = index + 1,
            };
        }

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        pub fn withLayer(self: *const View, index: usize) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn hasKey(self: *const View, subkey: []const u8) bool {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return for (self.store.registry.tensors.keys()) |k| {
                if (std.mem.startsWith(u8, k, key)) break true;
            } else false;
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            if (@TypeOf(tagz) != @TypeOf(null)) {
                switch (@typeInfo(@TypeOf(tagz))) {
                    .optional => if (tagz) |t| {
                        ptr.shape = ptr.shape.withTags(t);
                    },
                    else => ptr.shape = ptr.shape.withTags(tagz),
                }
            }

            if (@TypeOf(partitioning) != @TypeOf(null)) {
                switch (@typeInfo(@TypeOf(partitioning))) {
                    .optional => if (partitioning) |p| {
                        ptr.shape = ptr.shape.withPartitioning(p);
                    },
                    else => ptr.shape = ptr.shape.withPartitioning(partitioning),
                }
            }

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) Tensor {
            return self.maybeCreateTensor(subkey, tagz, partitioning).?;
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            };
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
            var key_buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return self.store.getReader(key, io, buffer);
        }

        pub fn count(self: View) usize {
            var count_: usize = 0;
            const prefix_ = self.prefix() orelse "";
            var it = self.store.registry.tensors.iterator();
            while (it.next()) |item| {
                const key = item.key_ptr.*;
                if (std.mem.startsWith(u8, key, prefix_)) {
                    count_ += 1;
                }
            }
            return count_;
        }
    };
};

pub const ProgressWriter = struct {
    inner: *std.Io.Writer,
    progress: *std.Progress.Node,
    interface: std.Io.Writer,
    total: usize = 0,
    scale: usize,

    pub const InitOpts = struct {
        scale: usize = 1,
    };

    pub fn init(inner_: *std.Io.Writer, progress_: *std.Progress.Node, opts: InitOpts) ProgressWriter {
        return .{
            .inner = inner_,
            .progress = progress_,
            .scale = opts.scale,
            .interface = .{
                .buffer = inner_.buffer,
                .end = inner_.end,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                    .sendFile = sendFile,
                },
            },
        };
    }

    pub fn pre(self: *ProgressWriter) usize {
        self.inner.buffer = self.interface.buffer;
        self.inner.end = self.interface.end;
        return self.inner.end;
    }

    pub fn post(self: *ProgressWriter, len_pre: usize, total: usize) void {
        self.interface.buffer = self.inner.buffer;
        self.interface.end = self.inner.end;
        const drained_pre = len_pre -| self.interface.end;
        self.total += drained_pre + total;
        self.progress.setCompletedItems(self.total / self.scale);
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        errdefer self.post(len_pre, 0);
        const total = try self.inner.vtable.drain(self.inner, data, splat);
        self.post(len_pre, total);
        return total;
    }

    pub fn sendFile(w: *std.Io.Writer, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        errdefer self.post(len_pre, 0);
        const total = try self.inner.vtable.sendFile(self.inner, file_reader, limit);
        self.post(len_pre, total);
        return total;
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        defer self.post(len_pre, 0);
        try self.inner.vtable.flush(self.inner);
    }

    pub fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        defer self.post(len_pre, 0);
        try self.inner.vtable.rebase(self.inner, preserve, capacity);
    }
};

pub const MemoryWriter = union(enum) {
    direct: DirectMemoryWriter,
    buffered: BufferedMemoryWriter,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !MemoryWriter {
        return switch (platform.target) {
            .cuda => .{ .direct = try DirectMemoryWriter.init(allocator, io, platform, pools, dma_allocators, shape, sharding, buffer) },
            .rocm, .tpu, .neuron, .cpu => .{ .buffered = try BufferedMemoryWriter.init(allocator, io, platform, shape, sharding, buffer) },
        };
    }

    pub fn interface(self: *MemoryWriter) *std.Io.Writer {
        return switch (self.*) {
            .direct => &self.direct.interface,
            .buffered => &self.buffered.interface,
        };
    }

    pub fn deinit(self: *MemoryWriter, allocator: std.mem.Allocator) void {
        switch (self.*) {
            .direct => self.direct.deinit(),
            .buffered => self.buffered.deinit(allocator),
        }
    }

    pub fn setProgress(self: *MemoryWriter, progress: ?*std.Progress.Node) void {
        switch (self.*) {
            .direct => self.direct.setProgress(progress),
            .buffered => {},
        }
    }
};

pub const BufferedMemoryWriter = struct {
    io: std.Io,
    platform: *const Platform,
    shape: Shape,
    sharding: Sharding,
    buffer: *Buffer,
    interface: std.Io.Writer,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding, buffer: *Buffer) !BufferedMemoryWriter {
        return .{
            .io = io,
            .platform = platform,
            .shape = shape,
            .sharding = sharding,
            .buffer = buffer,
            .interface = .{
                .buffer = try allocator.alloc(u8, shape.byteSize()),
                .vtable = &.{
                    .drain = std.Io.Writer.fixedDrain,
                    .flush = flush,
                    .rebase = std.Io.Writer.failingRebase,
                },
            },
        };
    }

    pub fn deinit(self: *BufferedMemoryWriter, allocator: std.mem.Allocator) void {
        if (self.interface.buffer.len > 0) {
            allocator.free(self.interface.buffer);
        }
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *BufferedMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        self.buffer.* = Buffer.from(
            self.io,
            self.platform,
            self.shape,
            self.sharding,
            @ptrCast(self.interface.buffer),
            .{ .wait = true },
        ) catch return std.Io.Writer.Error.WriteFailed;
    }
};

const DirectShardWriter = struct {
    const EventContext = struct {
        self: *DirectShardWriter,
        err: ?*pjrt.Error = null,
        pjrt_event: *pjrt.Event,
        event: std.Io.Event = .unset,
        buffer: []u8,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    memory: *const Memory,
    pool: *mem.DynamicBufferPool,
    total: usize,
    pjrt_buffer: *pjrt.Buffer,

    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    offset: usize = 0,

    interface: std.Io.Writer,
    flip_flop: u1 = 0,
    events_contexts: [2]?EventContext = @splat(null),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, memory: *const Memory, pool: *mem.DynamicBufferPool, shape: Shape) !DirectShardWriter {
        const shape_spec: pjrt.ShapeSpec = .init(shape.dims(), pjrtx.bufferTypeFromDtype(shape.dtype()));
        const transfer_manager = try memory.platform.pjrt_client.createBuffersForAsyncHostToDevice(
            memory.platform.pjrt_api,
            .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            },
        );

        const pjrt_buffer = transfer_manager.retrieveBuffer(memory.platform.pjrt_api, 0) catch unreachable;

        return .{
            .allocator = allocator,
            .io = io,
            .memory = memory,
            .pool = pool,
            .total = shape.byteSize(),
            .pjrt_buffer = pjrt_buffer,
            .transfer_manager = transfer_manager,
            .interface = .{
                .buffer = try pool.get(allocator, io),
                .end = 0,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    pub fn deinit(self: *DirectShardWriter) void {
        self.transfer_manager.deinit(self.memory.platform.pjrt_api);
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        var consumed: usize = 0;

        const pushChunk = struct {
            fn call(self_: *DirectShardWriter, chunk: []const u8, consumed_: *usize) std.Io.Writer.Error!void {
                var remaining = chunk;
                while (remaining.len > 0) {
                    if (self_.offset + self_.interface.end >= self_.total) {
                        return std.Io.Writer.Error.WriteFailed;
                    }

                    if (self_.interface.end == self_.interface.buffer.len) {
                        try self_.submitBuffered();
                    }

                    const writable_now = self_.interface.buffer.len - self_.interface.end;
                    const writable_to_total = self_.total - (self_.offset + self_.interface.end);
                    const to_copy = @min(remaining.len, @min(writable_now, writable_to_total));

                    @memcpy(
                        self_.interface.buffer[self_.interface.end .. self_.interface.end + to_copy],
                        remaining[0..to_copy],
                    );
                    self_.interface.end += to_copy;
                    consumed_.* += to_copy;
                    remaining = remaining[to_copy..];

                    if (self_.interface.end == self_.interface.buffer.len or self_.offset + self_.interface.end == self_.total) {
                        try self_.submitBuffered();
                    }
                }
            }
        }.call;

        for (data) |chunk| {
            try pushChunk(self, chunk, &consumed);
        }
        if (data.len > 0 and splat > 1) {
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                try pushChunk(self, last, &consumed);
            }
        }

        return consumed;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        try self.flushBuffered();
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        if (self.interface.buffer.len - self.interface.end >= capacity) return;
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;

        if (self.interface.end > 0) {
            try self.submitBuffered();
        }

        if (self.interface.buffer.len - self.interface.end < capacity) {
            return std.Io.Writer.Error.WriteFailed;
        }
    }

    fn flushBuffered(self: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.offset >= self.total) return;
        if (self.interface.end == 0) return;
        try self.submitBuffered();
    }

    fn submitBuffered(self: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.offset >= self.total) return;

        const pjrt_api = self.memory.platform.pjrt_api;

        const current_buffer = self.interface.buffer;
        const buffered = self.interface.buffered();
        const sliced_buffer = self.interface.buffered()[0..@min(buffered.len, self.total - self.offset)];
        const written = sliced_buffer.len;
        const is_last = (self.offset + sliced_buffer.len) >= self.total;

        const transfer_event = self.transfer_manager.transferData(
            pjrt_api,
            0,
            sliced_buffer,
            @intCast(self.offset),
            is_last,
        ) catch |err| {
            log.err("error when transferring data to device: {any}", .{err});
            return std.Io.Writer.Error.WriteFailed;
        };
        const ctx = &self.events_contexts[@intCast(self.flip_flop)];
        ctx.* = .{
            .self = self,
            .buffer = current_buffer,
            .pjrt_event = transfer_event,
        };
        transfer_event.onReady(pjrt_api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                ctx_.self.pool.put(ctx_.self.io, ctx_.buffer);
                ctx_.err = err;
                ctx_.event.set(ctx_.self.io);
            }
        }.call, &(ctx.*.?)) catch |err| {
            log.err("error when setting up transfer completion callback: {any}", .{err});
            return std.Io.Writer.Error.WriteFailed;
        };

        if (self.events_contexts[@intCast(self.flip_flop ^ 1)]) |*ctx_previous| {
            defer self.events_contexts[@intCast(self.flip_flop ^ 1)] = null;
            ctx_previous.event.waitUncancelable(self.io);
            defer ctx_previous.pjrt_event.deinit(pjrt_api);
            if (ctx_previous.err) |e| {
                defer e.deinit(pjrt_api);
                log.err("error while awaiting: {s}: {s}", .{
                    @tagName(e.getCode(pjrt_api)),
                    e.getMessage(pjrt_api),
                });
                return std.Io.Writer.Error.WriteFailed;
            }
        }

        if (is_last) {
            defer ctx.* = null;
            defer self.interface = .failing;
            const ctx_ = &ctx.*.?;
            ctx_.event.waitUncancelable(self.io);
            defer ctx_.pjrt_event.deinit(pjrt_api);
            if (ctx_.err) |e| {
                defer e.deinit(pjrt_api);
                log.err("error while awaiting: {s}: {s}", .{
                    @tagName(e.getCode(pjrt_api)),
                    e.getMessage(pjrt_api),
                });
                return std.Io.Writer.Error.WriteFailed;
            }
        } else {
            self.interface.end = 0;
            self.interface.buffer = self.pool.get(self.allocator, self.io) catch |err| {
                log.err("unable to get a new buffer from the pool: {any}", .{err});
                return std.Io.Writer.Error.WriteFailed;
            };
        }
        self.flip_flop ^= 1;
        self.offset += written;
    }
};

pub const DirectMemoryWriter = struct {
    const StreamPlanner = struct {
        const AxisRange = struct {
            start: i64,
            size: i64,
        };

        allocator: std.mem.Allocator,
        shape: Shape,
        placement: Placement,
        raw_segments: std.ArrayListUnmanaged(RawSegment) = .{},
        segments: std.ArrayListUnmanaged(StreamSegment) = .{},
        mirrors: std.ArrayListUnmanaged(usize) = .{},

        fn init(allocator: std.mem.Allocator, shape: Shape, placement: Placement) StreamPlanner {
            return .{
                .allocator = allocator,
                .shape = shape,
                .placement = placement,
            };
        }

        fn deinit(self: *StreamPlanner) void {
            self.raw_segments.deinit(self.allocator);
            self.segments.deinit(self.allocator);
            self.mirrors.deinit(self.allocator);
        }

        fn build(self: *StreamPlanner) !StreamPlan {
            try self.collectRawSegments();
            self.sortRawSegments();
            try self.compactSegments();

            const owned_segments = try self.segments.toOwnedSlice(self.allocator);
            errdefer self.allocator.free(owned_segments);
            const owned_mirrors = try self.mirrors.toOwnedSlice(self.allocator);

            return .{
                .segments = owned_segments,
                .mirrors = owned_mirrors,
            };
        }

        fn collectRawSegments(self: *StreamPlanner) !void {
            for (self.placement.shards.constSlice(), 0..) |shard, writer_index| {
                try self.appendShardSegments(shard, writer_index);
            }
        }

        fn sortRawSegments(self: *StreamPlanner) void {
            std.mem.sort(RawSegment, self.raw_segments.items, {}, struct {
                fn lessThan(_: void, lhs: RawSegment, rhs: RawSegment) bool {
                    if (lhs.start != rhs.start) return lhs.start < rhs.start;
                    if (lhs.len != rhs.len) return lhs.len < rhs.len;
                    return lhs.writer_index < rhs.writer_index;
                }
            }.lessThan);
        }

        fn compactSegments(self: *StreamPlanner) !void {
            var i: usize = 0;
            var cursor: usize = 0;
            while (i < self.raw_segments.items.len) {
                const first = self.raw_segments.items[i];
                if (first.start != cursor) return error.NonContiguousShardPlacement;

                const mirror_start = self.mirrors.items.len;
                var j = i + 1;
                while (j < self.raw_segments.items.len and
                    self.raw_segments.items[j].start == first.start and
                    self.raw_segments.items[j].len == first.len) : (j += 1)
                {
                    try self.mirrors.append(self.allocator, self.raw_segments.items[j].writer_index);
                }

                try self.segments.append(self.allocator, .{
                    .len = first.len,
                    .primary_writer = first.writer_index,
                    .mirror_start = mirror_start,
                    .mirror_len = j - i - 1,
                });
                cursor += first.len;
                i = j;
            }

            if (cursor != self.shape.byteSize()) return error.NonContiguousShardPlacement;
        }

        fn appendRawSegment(self: *StreamPlanner, writer_index: usize, start: usize, len: usize) !void {
            if (len == 0) return;

            if (self.raw_segments.items.len > 0) {
                const last = &self.raw_segments.items[self.raw_segments.items.len - 1];
                if (last.writer_index == writer_index and last.start + last.len == start) {
                    last.len += len;
                    return;
                }
            }

            try self.raw_segments.append(self.allocator, .{
                .writer_index = writer_index,
                .start = start,
                .len = len,
            });
        }

        fn axisRange(self: *const StreamPlanner, shard: Placement.Shard, axis: usize) AxisRange {
            for (shard.slices.constSlice()) |slice| {
                if (slice.axis == axis) {
                    return .{
                        .start = slice.start,
                        .size = slice.size,
                    };
                }
            }

            return .{
                .start = 0,
                .size = self.shape.dim(axis),
            };
        }

        fn appendShardSegmentsAtAxis(
            self: *StreamPlanner,
            shard: Placement.Shard,
            writer_index: usize,
            rank: usize,
            axis: usize,
            base_start: i64,
            strides: []const i64,
        ) !void {
            const range = self.axisRange(shard, axis);
            if (range.size == 0) return;

            if (axis + 1 == rank) {
                const segment_start: usize = @intCast(base_start + range.start * strides[axis]);
                const segment_len: usize = @intCast(range.size * strides[axis]);
                try self.appendRawSegment(writer_index, segment_start, segment_len);
                return;
            }

            var i: i64 = 0;
            while (i < range.size) : (i += 1) {
                const child_start = base_start + (range.start + i) * strides[axis];
                try self.appendShardSegmentsAtAxis(shard, writer_index, rank, axis + 1, child_start, strides);
            }
        }

        fn appendShardSegments(self: *StreamPlanner, shard: Placement.Shard, writer_index: usize) !void {
            const rank = self.shape.rank();
            if (rank == 0) {
                try self.appendRawSegment(writer_index, 0, self.shape.byteSize());
                return;
            }

            const strides = self.shape.computeByteStrides();
            try self.appendShardSegmentsAtAxis(shard, writer_index, rank, 0, 0, strides.constSlice());
        }
    };

    const StreamSegment = struct {
        len: usize,
        primary_writer: usize,
        mirror_start: usize,
        mirror_len: usize,
    };

    const RawSegment = struct {
        writer_index: usize,
        start: usize,
        len: usize,
    };

    const StreamPlan = struct {
        segments: []StreamSegment,
        mirrors: []usize,
    };

    const ShardProgress = struct {
        node: std.Progress.Node,
        label: [32]u8 = undefined,
        scale: usize = 1024,
    };

    allocator: std.mem.Allocator,
    shard_writers: []DirectShardWriter,
    stream_segments: []StreamSegment,
    segment_mirrors: []usize,
    segment_index: usize = 0,
    segment_offset: usize = 0,
    active_writer_index: ?usize = null,
    shard_progress: ?[]ShardProgress = null,
    interface: std.Io.Writer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !DirectMemoryWriter {
        const placement = try Placement.init(sharding, shape);
        var shard_writers = try allocator.alloc(DirectShardWriter, placement.shards.len);
        errdefer allocator.free(shard_writers);

        var initialized: usize = 0;
        errdefer for (shard_writers[0..initialized]) |*writer| {
            writer.deinit();
        };

        var pjrt_buffers: Buffer.Shards = .{};
        for (placement.shards.constSlice(), 0..) |shard, i| {
            defer initialized += 1;

            const device_index = shard.platformDeviceIndex(platform) orelse return error.UnknownShardDevice;
            const pool = &pools[device_index];
            const shard_dma_allocator = dma_allocators[device_index].allocator();

            shard_writers[i] = try DirectShardWriter.init(shard_dma_allocator, io, shard.memory(platform, .default), pool, shard.shape);

            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }

        buffer.* = Buffer.fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());

        var planner = StreamPlanner.init(allocator, shape, placement);
        defer planner.deinit();

        const stream_plan = try planner.build();
        errdefer {
            allocator.free(stream_plan.segments);
            allocator.free(stream_plan.mirrors);
        }

        const first_segment = stream_plan.segments[0];
        const initial_writer_index = first_segment.primary_writer;
        const initial_writer = &shard_writers[initial_writer_index];
        const shard_remaining = initial_writer.total - (initial_writer.offset + initial_writer.interface.end);
        const initial_visible_len: usize = @min(
            initial_writer.interface.buffer.len,
            initial_writer.interface.end + @min(first_segment.len, shard_remaining),
        );

        return .{
            .allocator = allocator,
            .shard_writers = shard_writers,
            .stream_segments = stream_plan.segments,
            .segment_mirrors = stream_plan.mirrors,
            .active_writer_index = initial_writer_index,
            .interface = .{
                .buffer = initial_writer.interface.buffer[0..initial_visible_len],
                .end = initial_writer.interface.end,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                },
            },
        };
    }

    pub fn deinit(self: *DirectMemoryWriter) void {
        self.setProgress(null);
        for (self.shard_writers) |*writer| {
            writer.deinit();
        }
        self.allocator.free(self.shard_writers);
        self.allocator.free(self.stream_segments);
        self.allocator.free(self.segment_mirrors);
    }

    pub fn setProgress(self: *DirectMemoryWriter, progress: ?*std.Progress.Node) void {
        if (self.shard_progress) |states| {
            for (states) |*s| {
                s.node.end();
            }
            self.allocator.free(states);
            self.shard_progress = null;
        }

        const parent = progress orelse return;
        const states = self.allocator.alloc(ShardProgress, self.shard_writers.len) catch return;
        for (states, self.shard_writers, 0..) |*state, writer, i| {
            const scale: usize = 1024;
            const total_items = std.math.divCeil(usize, writer.total, scale) catch unreachable;
            const label = std.fmt.bufPrint(&state.label, "shard[{d}]", .{i}) catch unreachable;
            state.scale = scale;
            state.node = parent.start(label, total_items);
            state.node.setCompletedItems(0);
        }
        self.shard_progress = states;
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        var consumed: usize = 0;

        for (data) |chunk| {
            consumed += try self.writeChunk(chunk);
        }
        if (data.len > 0 and splat > 1) {
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                consumed += try self.writeChunk(last);
            }
        }

        return consumed;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        try self.syncProxyIntoActive();
        self.refreshExposedWindow();

        if (self.currentSegment() != null) return std.Io.Writer.Error.WriteFailed;

        for (self.shard_writers, 0..) |*shard_writer, i| {
            if (shard_writer.interface.end > 0) {
                try shard_writer.flushBuffered();
            }
            if (shard_writer.offset < shard_writer.total) return std.Io.Writer.Error.WriteFailed;
            self.updateShardProgress(i);
        }

        self.interface = .failing;
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;
        try self.ensureWritable(capacity);
    }

    fn ensureWritable(self: *DirectMemoryWriter, capacity: usize) std.Io.Writer.Error!void {
        if (capacity == 0) return;
        while (self.interface.buffer.len - self.interface.end < capacity) {
            try self.syncProxyIntoActive();
            self.refreshExposedWindow();

            const segment = self.currentSegment() orelse {
                self.interface = .failing;
                return std.Io.Writer.Error.WriteFailed;
            };

            if (self.interface.buffer.len - self.interface.end >= capacity) return;
            if (capacity > self.interface.buffer.len and self.interface.buffer.len != 0) {
                return std.Io.Writer.Error.WriteFailed;
            }

            const shard_writer = &self.shard_writers[segment.primary_writer];
            if (shard_writer.interface.end == 0) return std.Io.Writer.Error.WriteFailed;
            try shard_writer.flushBuffered();

            self.refreshExposedWindow();
            if (self.interface.buffer.len - self.interface.end >= capacity) return;

            if (self.currentSegmentRemaining() == 0) continue;
            return std.Io.Writer.Error.WriteFailed;
        }
    }

    fn currentSegment(self: *const DirectMemoryWriter) ?StreamSegment {
        if (self.segment_index >= self.stream_segments.len) return null;
        return self.stream_segments[self.segment_index];
    }

    fn currentSegmentRemaining(self: *const DirectMemoryWriter) usize {
        const segment = self.currentSegment() orelse return 0;
        return segment.len -| self.segment_offset;
    }

    fn writeChunk(self: *DirectMemoryWriter, chunk: []const u8) std.Io.Writer.Error!usize {
        var consumed: usize = 0;
        while (consumed < chunk.len) {
            try self.ensureWritable(1);

            const segment_remaining = self.currentSegmentRemaining();
            const writable = self.interface.buffer.len - self.interface.end;
            const to_write = @min(chunk.len - consumed, @min(segment_remaining, writable));
            if (to_write == 0) return std.Io.Writer.Error.WriteFailed;

            @memcpy(
                self.interface.buffer[self.interface.end .. self.interface.end + to_write],
                chunk[consumed..][0..to_write],
            );
            self.interface.end += to_write;
            try self.syncProxyIntoActive();
            consumed += to_write;
        }

        return consumed;
    }

    fn refreshExposedWindow(self: *DirectMemoryWriter) void {
        self.skipCompletedSegments();
        if (self.currentSegment()) |segment| {
            self.exposeShardWindow(segment.primary_writer, self.currentSegmentRemaining());
        } else {
            self.hideExposedWindow();
        }
    }

    fn skipCompletedSegments(self: *DirectMemoryWriter) void {
        while (self.currentSegment()) |_| {
            if (self.currentSegmentRemaining() != 0) return;
            self.segment_index += 1;
            self.segment_offset = 0;
        }
    }

    fn hideExposedWindow(self: *DirectMemoryWriter) void {
        self.active_writer_index = null;
        stdx.debug.assert(self.shard_writers.len > 0, "DirectMemoryWriter requires at least one shard writer", .{});
        self.interface.buffer = self.shard_writers[0].interface.buffer[0..0];
        self.interface.end = 0;
    }

    fn syncProxyIntoActive(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        const writer_index = self.active_writer_index orelse return;
        const segment = self.currentSegment() orelse return;
        if (segment.primary_writer != writer_index) return;

        const shard_writer = &self.shard_writers[writer_index];
        if (self.interface.end <= shard_writer.interface.end) {
            self.interface.end = shard_writer.interface.end;
            return;
        }

        const old_end = shard_writer.interface.end;
        const new_end = self.interface.end;
        if (new_end > shard_writer.interface.buffer.len) return std.Io.Writer.Error.WriteFailed;
        const chunk = shard_writer.interface.buffer[old_end..new_end];
        if (chunk.len > self.currentSegmentRemaining()) return std.Io.Writer.Error.WriteFailed;

        shard_writer.interface.end = new_end;
        self.segment_offset += chunk.len;
        try self.mirrorChunk(segment, chunk);
        self.updateShardProgress(writer_index);
    }

    fn exposeShardWindow(self: *DirectMemoryWriter, writer_index: usize, segment_remaining: usize) void {
        const shard_writer = &self.shard_writers[writer_index];
        self.active_writer_index = writer_index;
        const shard_remaining = shard_writer.total - (shard_writer.offset + shard_writer.interface.end);
        const visible_remaining = @min(shard_remaining, segment_remaining);
        const visible_len = @min(shard_writer.interface.buffer.len, shard_writer.interface.end + visible_remaining);
        self.interface.buffer = shard_writer.interface.buffer[0..visible_len];
        self.interface.end = shard_writer.interface.end;
    }

    fn mirrorChunk(self: *DirectMemoryWriter, segment: StreamSegment, chunk: []const u8) std.Io.Writer.Error!void {
        if (chunk.len == 0) return;
        if (segment.mirror_len == 0) return;

        for (self.segment_mirrors[segment.mirror_start .. segment.mirror_start + segment.mirror_len]) |writer_index| {
            const shard_writer = &self.shard_writers[writer_index];
            const wrote = try shard_writer.interface.vtable.drain(&shard_writer.interface, &.{chunk}, 1);
            if (wrote != chunk.len) return std.Io.Writer.Error.WriteFailed;
            self.updateShardProgress(writer_index);
        }
    }

    fn updateShardProgress(self: *DirectMemoryWriter, writer_index: usize) void {
        const states = self.shard_progress orelse return;
        const state = &states[writer_index];
        const writer = &self.shard_writers[writer_index];
        const completed_bytes = writer.offset + writer.interface.end;
        state.node.setCompletedItems(completed_bytes / state.scale);
    }
};

pub const LoadOpts = struct {
    pub const auto: LoadOpts = .{
        .parallelism = 1,
        .shardings = &.{},
        .dma_chunks = 2,
        .dma_chunk_size = 4096,
    };

    parallelism: usize,
    shardings: []const Sharding,
    progress: ?*std.Progress.Node = null,
    dma_chunks: usize,
    dma_chunk_size: usize,
    total_bytes: ?*usize = null,
};

pub fn load(
    comptime ModelType: type,
    model: *const ModelType,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: LoadOpts,
) !Bufferized(ModelType) {
    var bufferized = try mem.bufferize(allocator, ModelType, model);

    const pool_count = platform.devices.len;
    const dma_allocators = try allocator.alloc(mem.DmaAllocator, pool_count);
    defer allocator.free(dma_allocators);
    for (platform.devices, 0..) |*device, i| {
        dma_allocators[i] = .init(allocator, device);
    }

    const buffer_pools = try allocator.alloc(mem.DynamicBufferPool, pool_count);
    defer allocator.free(buffer_pools);
    for (buffer_pools) |*pool_| {
        pool_.* = .init(opts.dma_chunks, opts.dma_chunk_size);
    }
    defer for (buffer_pools, 0..) |*pool_, i| {
        pool_.deinit(dma_allocators[i].allocator());
    };

    const replicated_sharding = try sharding_.replicatedSharding(platform);

    const Ctx = struct {
        allocator: std.mem.Allocator,
        dma_allocators: []const mem.DmaAllocator,
        pinned_buffer_pools: []mem.DynamicBufferPool,
        io: std.Io,
        platform: *const Platform,
        buffers: []*Buffer,
        shardings: []const Sharding,
        replicated_sharding: Sharding,
        store: *const TensorStore,
        group: stdx.Io.LimitedGroup,
        total: std.atomic.Value(usize) = .init(0),
        progress: ?*std.Progress.Node,
    };
    var walk_ctx: Ctx = .{
        .platform = platform,
        .buffers = try allocator.alloc(*Buffer, meta.count(Tensor, model)),
        .store = store,
        .allocator = allocator,
        .dma_allocators = dma_allocators,
        .pinned_buffer_pools = buffer_pools,
        .io = io,
        .shardings = opts.shardings,
        .replicated_sharding = replicated_sharding,
        .progress = opts.progress,
        .group = .init(opts.parallelism),
    };
    defer allocator.free(walk_ctx.buffers);

    defer if (opts.total_bytes) |total_bytes_ptr| {
        total_bytes_ptr.* = walk_ctx.total.load(.monotonic);
    };

    meta.forEachVisit(&bufferized, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, ctx: *Ctx) void {
            ctx.buffers[i] = buffer;
        }
    }.call, .{&walk_ctx});

    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, ctx: *Ctx) void {
            ctx.group.concurrent(ctx.io, struct {
                fn call(i_: usize, tensor_: *const Tensor, ctx_: *Ctx) !void {
                    var reader = ctx_.store.getReaderById(tensor_.id, ctx_.io, &.{}) catch unreachable;
                    defer reader.deinit();

                    const shape = reader.tensor.shape;
                    const select_sharding = sharding_.pickSharding(ctx_.shardings, shape, .explicit_axis_binding);
                    const sharding = if (select_sharding) |s| s else blk: {
                        log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding:\n {f}", .{ reader.tensor.name, shape, ctx_.replicated_sharding });
                        break :blk ctx_.replicated_sharding;
                    };

                    var writer = MemoryWriter.init(
                        ctx_.allocator,
                        ctx_.io,
                        ctx_.platform,
                        ctx_.pinned_buffer_pools,
                        ctx_.dma_allocators,
                        shape,
                        sharding,
                        ctx_.buffers[i_],
                    ) catch unreachable;
                    defer writer.deinit(ctx_.allocator);

                    const scale = 1024;

                    if (ctx_.progress) |progress| {
                        var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
                        defer node.end();
                        writer.setProgress(&node);
                        defer writer.setProgress(null);
                        var progress_writer: ProgressWriter = .init(writer.interface(), &node, .{ .scale = scale });
                        const total = reader.interface.streamRemaining(&progress_writer.interface) catch unreachable;
                        progress_writer.interface.flush() catch unreachable;
                        _ = ctx_.total.fetchAdd(total, .monotonic);
                    } else {
                        const total = reader.interface.streamRemaining(writer.interface()) catch unreachable;
                        writer.interface().flush() catch unreachable;
                        _ = ctx_.total.fetchAdd(total, .monotonic);
                    }
                }
            }.call, .{ i, tensor, ctx }) catch unreachable;
        }
    }.call, .{&walk_ctx});
    walk_ctx.group.await(io) catch unreachable;

    return bufferized;
}

fn buildMesh2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !sharding_.PhysicalMesh {
    if (devices.len < 4) return error.NotEnoughDevices;
    const topology: sharding_.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[0]),
            .device(devices[1]),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[2]),
            .device(devices[3]),
        }),
    });

    return sharding_.PhysicalMesh.fromTree(allocator, target, topology);
}

fn buildMesh2x2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !sharding_.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevices;
    const topology: sharding_.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[0]),
                .device(devices[1]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[2]),
                .device(devices[3]),
            }),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[4]),
                .device(devices[5]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[6]),
                .device(devices[7]),
            }),
        }),
    });

    return sharding_.PhysicalMesh.fromTree(allocator, target, topology);
}

const DirectMemoryWriterDeviceTest = struct {
    const WriteMode = enum {
        stream_remaining,
        writable_slice_greedy,
    };

    pub const Scenario = struct {
        name: []const u8,
        create_options: CreateOptions,
        shape: Shape,
        logical_mesh: sharding_.LogicalMesh,
        strategy: sharding_.Strategy,
        write_mode: WriteMode = .stream_remaining,
        writable_slice_min_len: usize = 128,
        pool_chunks: usize = 4,
        pool_chunk_size: usize = 1 << 20,
    };

    allocator: std.mem.Allocator,
    io: std.Io,

    fn run(self: DirectMemoryWriterDeviceTest, scenario: Scenario) !void {
        _ = scenario.name;

        var platform = Platform.auto(self.allocator, self.io, scenario.create_options) catch return error.SkipZigTest;
        defer platform.deinit(self.allocator, self.io);

        const sharding: Sharding = try .initFromStrategy(platform, scenario.logical_mesh, scenario.strategy);
        try self.runDirectMemoryWriter(
            platform,
            scenario.shape,
            sharding,
            scenario.write_mode,
            scenario.writable_slice_min_len,
            scenario.pool_chunks,
            scenario.pool_chunk_size,
        );
    }

    fn runDirectMemoryWriter(
        self: DirectMemoryWriterDeviceTest,
        platform: *const Platform,
        shape: Shape,
        sharding: Sharding,
        write_mode: WriteMode,
        writable_slice_min_len: usize,
        pool_chunks: usize,
        pool_chunk_size: usize,
    ) !void {
        const slice = try Slice.alloc(self.allocator, shape);
        defer slice.free(self.allocator);

        for (slice.items(f32), 0..) |*e, i| {
            e.* = @as(f32, @floatFromInt(i));
        }

        const pool_count = platform.devices.len;
        const dma_allocators = try self.allocator.alloc(mem.DmaAllocator, pool_count);
        defer self.allocator.free(dma_allocators);
        for (platform.devices, 0..) |*device, i| {
            dma_allocators[i] = .init(self.allocator, device);
        }

        const pools = try self.allocator.alloc(mem.DynamicBufferPool, pool_count);
        defer self.allocator.free(pools);
        for (pools) |*pool| {
            pool.* = .init(pool_chunks, pool_chunk_size);
        }
        defer for (pools, 0..) |*pool, i| {
            pool.deinit(dma_allocators[i].allocator());
        };

        var written_buffer: Buffer = undefined;
        var writer: DirectMemoryWriter = try .init(
            self.allocator,
            self.io,
            platform,
            pools,
            dma_allocators,
            shape,
            sharding,
            &written_buffer,
        );
        defer writer.deinit();
        defer written_buffer.deinit();

        switch (write_mode) {
            .stream_remaining => {
                var reader: std.Io.Reader = .fixed(slice.constData());
                const streamed = try reader.streamRemaining(&writer.interface);
                try std.testing.expectEqual(slice.constData().len, streamed);
            },
            .writable_slice_greedy => {
                var offset: usize = 0;
                while (offset < slice.constData().len) {
                    const min_len = @max(@as(usize, 1), writable_slice_min_len);
                    const dest = try writer.interface.writableSliceGreedy(min_len);
                    const to_write = @min(dest.len, slice.constData().len - offset);
                    if (to_write == 0) return std.Io.Writer.Error.WriteFailed;
                    @memcpy(dest[0..to_write], slice.constData()[offset..][0..to_write]);
                    writer.interface.advance(to_write);
                    offset += to_write;
                }
            },
        }

        try writer.interface.flush();
        try written_buffer.await(self.io);

        var written_slice = try written_buffer.toSliceAlloc(self.allocator, self.io);
        defer written_slice.free(self.allocator);
        try std.testing.expectEqualSlices(u8, slice.constData(), written_slice.constData());
    }
};

test "DirectMemoryWriter: replicated with auto topology" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "replicated_auto",
        .create_options = .{
            .physical_mesh = .auto,
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 128 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .replicated }),
        .logical_mesh = try .init("replicated_cpu", .{ .x = .high_bandwidth }),
        .strategy = blk: {
            var strategy: sharding_.Strategy = .init;
            try strategy.addBinding(.x, .link_x);

            break :blk strategy;
        },
    });
}

test "DirectMemoryWriter: 1D model split with 2x2 physical mesh" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "model_auto",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .model }),
        .logical_mesh = try .init("model_cpu", .{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: sharding_.Strategy = .init;
            try strategy.addBinding(.model, .link_x);

            break :blk strategy;
        },
    });
}

test "DirectMemoryWriter: 2D batch/model split with 2x2 physical mesh" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "batch_model_2d_torus",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .batch = 8, .model = 1024 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model }),
        .logical_mesh = try .init("batch_model_cpu", .{
            .batch = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .strategy = blk: {
            var strategy: sharding_.Strategy = .init;
            try strategy.addBinding(.batch, .link_x);
            try strategy.addBinding(.model, .link_y);

            break :blk strategy;
        },
    });
}

test "DirectMemoryWriter: folded model sharding with 2x2 physical mesh" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "model_folded_2d_torus",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .model = 4096 }, .f32).withPartitioning(.{ .model = .model }),
        .logical_mesh = try .init("model_folded_cpu", .{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: sharding_.Strategy = .init;
            try strategy.addBinding(.model, .link_x);
            try strategy.addFold(.link_x, &.{ .link_x, .link_y });

            break :blk strategy;
        },
    });
}

test "DirectMemoryWriter: writableSliceGreedy with mirrored shards" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "model_auto_writable_slice",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2 },
            .cpu = .{ .device_count = 4 },
        },
        .shape = Shape.init(.{ .rows = 8, .cols = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .model }),
        .logical_mesh = try .init("model_cpu_greedy", .{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: sharding_.Strategy = .init;
            try strategy.addBinding(.model, .link_x);

            break :blk strategy;
        },
        .write_mode = .writable_slice_greedy,
        .writable_slice_min_len = 64,
        .pool_chunk_size = 1024,
    });
}

test "DirectMemoryWriter: 3D topology folded model + replicated batch" {
    const case: DirectMemoryWriterDeviceTest = .{
        .allocator = std.testing.allocator,
        .io = std.testing.io,
    };

    try case.run(.{
        .name = "topology_3d_folded_model",
        .create_options = .{
            .physical_mesh = .{ .custom = buildMesh2x2x2 },
            .cpu = .{ .device_count = 8 },
        },
        .shape = Shape.init(.{ .batch = 16, .model = 4096 }, .f32)
            .withPartitioning(.{ .batch = .replicated, .model = .model }),
        .logical_mesh = try .init("topology_3d_folded_model_mesh", .{
            .batch = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .strategy = blk: {
            var strategy: sharding_.Strategy = .init;
            try strategy.addBinding(.model, .link_x);
            try strategy.addFold(.link_x, &.{ .link_x, .link_z });

            break :blk strategy;
        },
    });
}
