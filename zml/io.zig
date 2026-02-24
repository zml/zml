const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
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

        pub fn TensorLoader(comptime is_optional: bool) type {
            return struct {
                const Self = @This();

                store: *TensorStore,
                key_buffer: [256]u8,
                key_len: usize,
                ptr: ?*safetensors.Tensor,

                fn init(view_: View, subkey: []const u8) Self {
                    var key_buffer: [256]u8 = undefined;
                    const key = std.fmt.bufPrint(&key_buffer, "{s}{s}", .{ view_.prefix() orelse "", subkey }) catch unreachable;
                    return .{
                        .store = view_.store,
                        .key_buffer = key_buffer,
                        .key_len = key.len,
                        .ptr = view_.store.getPtrFromKey(key),
                    };
                }

                pub fn optional(self: Self) TensorLoader(true) {
                    return .{
                        .store = self.store,
                        .key_buffer = self.key_buffer,
                        .key_len = self.key_len,
                        .ptr = self.ptr,
                    };
                }

                pub fn withTags(self: Self, tags_: anytype) Self {
                    if (self.ptr) |ptr| {
                        if (@TypeOf(tags_) == @TypeOf(null)) return;
                        switch (@typeInfo(@TypeOf(tags_))) {
                            .optional => if (tags_) |t| {
                                ptr.shape = ptr.shape.withTags(t);
                            },
                            else => ptr.shape = ptr.shape.withTags(tags_),
                        }
                    }
                    return self;
                }

                pub fn withPartitioning(self: Self, partitioning_: anytype) Self {
                    if (self.ptr) |ptr| {
                        if (@TypeOf(partitioning_) == @TypeOf(null)) return;
                        switch (@typeInfo(@TypeOf(partitioning_))) {
                            .optional => if (partitioning_) |p| {
                                ptr.shape = ptr.shape.withPartitioning(p);
                            },
                            else => ptr.shape = ptr.shape.withPartitioning(partitioning_),
                        }
                    }
                    return self;
                }

                pub fn toTensor(self: Self) if (is_optional) ?Tensor else Tensor {
                    const ptr = self.ptr orelse {
                        if (comptime is_optional) return null;
                        unreachable;
                    };

                    const key = self.key_buffer[0..self.key_len];
                    const result: Tensor = .fromShape(ptr.shape);
                    self.store.bindIdToKey(key, result.id) catch unreachable;

                    if (comptime is_optional) {
                        return result;
                    }
                    return result;
                }
            };
        }

        pub fn load(self: View, subkey: []const u8) TensorLoader(false) {
            return TensorLoader(false).init(self, subkey);
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
        pool: *mem.DynamicBufferPool,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !MemoryWriter {
        return switch (platform.target) {
            .cuda => .{ .direct = try DirectMemoryWriter.init(allocator, io, platform, pool, shape, sharding, buffer) },
            .rocm, .tpu, .neuron, .cpu => .{ .buffered = try BufferedMemoryWriter.init(allocator, io, platform, shape, sharding, buffer) },
        };
    }

    pub fn initWithDevicePools(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !MemoryWriter {
        return switch (platform.target) {
            .cuda => .{ .direct = try DirectMemoryWriter.initWithDevicePools(allocator, io, platform, pools, shape, sharding, buffer) },
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

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        memory: *const Memory,
        pool: *mem.DynamicBufferPool,
        shape: Shape,
    ) !DirectShardWriter {
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
                    .sendFile = sendFile,
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

    fn sendFile(w: *std.Io.Writer, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        return w.sendFileReadingAll(file_reader, limit);
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
    const ShardSpan = struct {
        writer_index: usize,
        start: usize,
        len: usize,
    };

    const Layout = union(enum) {
        partitioned: []usize,
        replicated,
    };

    allocator: std.mem.Allocator,
    shard_writers: []DirectShardWriter,
    stream_order: ?[]usize = null,
    stream_index: usize = 0,
    interface: std.Io.Writer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pool: *mem.DynamicBufferPool,
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
            shard_writers[i] = try DirectShardWriter.init(
                allocator,
                io,
                shard.memory(platform, .default),
                pool,
                shard.shape,
            );
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }
        return initWriter(allocator, shape, sharding, placement, shard_writers, platform, pjrt_buffers.constSlice(), buffer);
    }

    pub fn initWithDevicePools(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
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
            shard_writers[i] = try DirectShardWriter.init(
                allocator,
                io,
                shard.memory(platform, .default),
                poolForDevice(platform, pools, shard.device_id),
                shard.shape,
            );
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }
        return initWriter(allocator, shape, sharding, placement, shard_writers, platform, pjrt_buffers.constSlice(), buffer);
    }

    fn initWriter(
        allocator: std.mem.Allocator,
        shape: Shape,
        sharding: Sharding,
        placement: Placement,
        shard_writers: []DirectShardWriter,
        platform: *const Platform,
        pjrt_buffers: []const *pjrt.Buffer,
        buffer: *Buffer,
    ) !DirectMemoryWriter {
        buffer.* = Buffer.fromPjrtBuffers(platform, shape, sharding, pjrt_buffers);
        stdx.debug.assert(shard_writers.len > 0, "DirectMemoryWriter requires at least one shard writer", .{});

        const layout = try chooseLayout(allocator, shape, placement);

        return switch (layout) {
            .replicated => blk: {
                const first_shard_remaining = shard_writers[0].total;
                const first_visible_len = @min(shard_writers[0].interface.buffer.len, first_shard_remaining);
                break :blk .{
                    .allocator = allocator,
                    .shard_writers = shard_writers,
                    .stream_order = null,
                    .stream_index = 0,
                    .interface = .{
                        .buffer = shard_writers[0].interface.buffer[0..first_visible_len],
                        .end = shard_writers[0].interface.end,
                        .vtable = &.{
                            .drain = replicatedDrain,
                            .flush = replicatedFlush,
                            .rebase = replicatedRebase,
                            .sendFile = sendFile,
                        },
                    },
                };
            },
            .partitioned => |order| blk: {
                const first = order[0];
                const first_shard_remaining = shard_writers[first].total;
                const first_visible_len = @min(shard_writers[first].interface.buffer.len, first_shard_remaining);
                break :blk .{
                    .allocator = allocator,
                    .shard_writers = shard_writers,
                    .stream_order = order,
                    .stream_index = 0,
                    .interface = .{
                        .buffer = shard_writers[first].interface.buffer[0..first_visible_len],
                        .end = shard_writers[first].interface.end,
                        .vtable = &.{
                            .drain = partitionedDrain,
                            .flush = partitionedFlush,
                            .rebase = partitionedRebase,
                            .sendFile = sendFile,
                        },
                    },
                };
            },
        };
    }

    fn chooseLayout(allocator: std.mem.Allocator, shape: Shape, placement: Placement) !Layout {
        var spans = try allocator.alloc(ShardSpan, placement.shards.len);
        defer allocator.free(spans);

        for (placement.shards.constSlice(), 0..) |shard, i| {
            spans[i] = shardSpan(shape, shard, i) orelse return error.NonContiguousShardPlacement;
        }

        var all_replicated = true;
        for (spans) |span| {
            if (span.start != 0 or span.len != shape.byteSize()) {
                all_replicated = false;
                break;
            }
        }

        if (all_replicated) {
            return .replicated;
        }

        std.mem.sort(ShardSpan, spans, {}, struct {
            fn lessThan(_: void, lhs: ShardSpan, rhs: ShardSpan) bool {
                return lhs.start < rhs.start;
            }
        }.lessThan);

        var covered: usize = 0;
        for (spans) |span| {
            if (span.start != covered) return error.NonContiguousShardPlacement;
            covered += span.len;
        }
        if (covered != shape.byteSize()) return error.NonContiguousShardPlacement;

        var order = try allocator.alloc(usize, spans.len);
        errdefer allocator.free(order);
        for (spans, 0..) |span, i| {
            order[i] = span.writer_index;
        }
        return .{ .partitioned = order };
    }

    fn shardSpan(shape: Shape, shard: Placement.Shard, writer_index: usize) ?ShardSpan {
        const global_strides = shape.computeByteStrides();
        var shard_shape = shape;
        var start: i64 = 0;

        for (shard.slices.constSlice()) |s| {
            shard_shape = shard_shape.set(s.axis, s.size);
            start += s.start * global_strides.get(s.axis);
        }

        if (start < 0) return null;

        const expected = shard_shape.computeByteStrides();
        for (0..shape.rank()) |ax| {
            if (global_strides.get(ax) != expected.get(ax)) return null;
        }

        if (shard_shape.byteSize() != shard.shape.byteSize()) return null;

        return .{
            .writer_index = writer_index,
            .start = @intCast(start),
            .len = shard_shape.byteSize(),
        };
    }

    fn poolForDevice(platform: *const Platform, pools: []mem.DynamicBufferPool, device_id: usize) *mem.DynamicBufferPool {
        stdx.debug.assert(pools.len == platform.devices.len, "Expected one DMA pool per device, got pools={} devices={}", .{ pools.len, platform.devices.len });
        for (platform.devices, 0..) |d, i| {
            if (d.id() == device_id) return &pools[i];
        }
        unreachable;
    }

    pub fn deinit(self: *DirectMemoryWriter) void {
        for (self.shard_writers) |*writer| {
            writer.deinit();
        }
        self.allocator.free(self.shard_writers);
        if (self.stream_order) |order| {
            self.allocator.free(order);
        }
    }

    pub fn setProgress(_: *DirectMemoryWriter, _: ?*std.Progress.Node) void {}

    fn sendFile(w: *std.Io.Writer, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        if (self.stream_order != null) {
            return w.sendFileReadingAll(file_reader, limit);
        }
        return self.replicatedSendFile(file_reader, limit);
    }

    fn replicatedSendFile(self: *DirectMemoryWriter, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        var remaining = @intFromEnum(limit);
        var total: usize = 0;

        while (remaining > 0) {
            try self.replicatedEnsureWritable(1);
            const writable = self.interface.buffer[self.interface.end..];
            if (writable.len == 0) return if (total == 0) error.EndOfStream else total;

            const to_read = @min(writable.len, remaining);
            const read_n = try file_reader.interface.readSliceShort(writable[0..to_read]);
            if (read_n == 0) return if (total == 0) error.EndOfStream else total;

            self.interface.end += read_n;
            self.shard_writers[0].interface.end = self.interface.end;

            const chunk = writable[0..read_n];
            for (self.shard_writers[1..]) |*shard_writer| {
                const wrote = try shard_writer.interface.vtable.drain(&shard_writer.interface, &.{chunk}, 1);
                if (wrote != chunk.len) return error.WriteFailed;
            }

            self.replicatedSyncFromFirst();
            total += read_n;
            remaining -= read_n;
        }

        return total;
    }

    fn partitionedDrain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        var consumed: usize = 0;

        const writeChunk = struct {
            fn call(self_: *DirectMemoryWriter, chunk: []const u8, consumed_: *usize) std.Io.Writer.Error!void {
                var remaining = chunk;
                while (remaining.len > 0) {
                    try self_.partitionedEnsureWritable(1);
                    const order = self_.stream_order.?;
                    if (self_.stream_index >= order.len) return std.Io.Writer.Error.WriteFailed;

                    const writer_index = order[self_.stream_index];
                    const shard_writer = &self_.shard_writers[writer_index];
                    try self_.partitionedSyncIntoShard(shard_writer);

                    const shard_remaining = shard_writer.total - (shard_writer.offset + shard_writer.interface.end);
                    if (shard_remaining == 0) {
                        self_.partitionedSyncFromShard(shard_writer);
                        try self_.partitionedAdvance();
                        continue;
                    }

                    const to_write = @min(remaining.len, shard_remaining);
                    const wrote = try shard_writer.interface.vtable.drain(
                        &shard_writer.interface,
                        &.{remaining[0..to_write]},
                        1,
                    );
                    if (wrote != to_write) return std.Io.Writer.Error.WriteFailed;

                    self_.partitionedSyncFromShard(shard_writer);
                    consumed_.* += wrote;
                    remaining = remaining[wrote..];

                    if (shard_writer.offset + shard_writer.interface.end == shard_writer.total) {
                        try self_.partitionedAdvance();
                    }
                }
            }
        }.call;

        for (data) |chunk| {
            try writeChunk(self, chunk, &consumed);
        }
        if (data.len > 0 and splat > 1) {
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                try writeChunk(self, last, &consumed);
            }
        }

        return consumed;
    }

    fn partitionedFlush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        const order = self.stream_order.?;

        while (self.stream_index < order.len) {
            const shard_writer = &self.shard_writers[order[self.stream_index]];
            try self.partitionedSyncIntoShard(shard_writer);

            if (shard_writer.offset + shard_writer.interface.end < shard_writer.total and shard_writer.interface.end == 0) {
                return std.Io.Writer.Error.WriteFailed;
            }

            if (shard_writer.interface.end > 0) {
                try shard_writer.flushBuffered();
            }

            self.partitionedSyncFromShard(shard_writer);

            if (shard_writer.offset < shard_writer.total) return std.Io.Writer.Error.WriteFailed;
            self.stream_index += 1;
            if (self.stream_index < order.len) {
                const next = &self.shard_writers[order[self.stream_index]];
                self.partitionedSyncFromShard(next);
            }
        }

        self.interface = .failing;
    }

    fn partitionedRebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;
        try self.partitionedEnsureWritable(capacity);
    }

    fn partitionedEnsureWritable(self: *DirectMemoryWriter, capacity: usize) std.Io.Writer.Error!void {
        if (capacity == 0) return;
        while (self.interface.buffer.len - self.interface.end < capacity) {
            if (capacity > self.interface.buffer.len and self.interface.buffer.len != 0) {
                return std.Io.Writer.Error.WriteFailed;
            }
            try self.partitionedAdvance();
            if (self.interface.buffer.len == 0) return std.Io.Writer.Error.WriteFailed;
            if (capacity > self.interface.buffer.len) return std.Io.Writer.Error.WriteFailed;
        }
    }

    fn partitionedAdvance(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        const order = self.stream_order.?;

        while (self.stream_index < order.len) {
            const shard_writer = &self.shard_writers[order[self.stream_index]];
            try self.partitionedSyncIntoShard(shard_writer);

            const written = shard_writer.offset + shard_writer.interface.end;
            if (written < shard_writer.total and shard_writer.interface.end < shard_writer.interface.buffer.len) {
                self.partitionedSyncFromShard(shard_writer);
                return;
            }

            if (shard_writer.interface.end > 0) {
                try shard_writer.flushBuffered();
            }
            self.partitionedSyncFromShard(shard_writer);

            if (shard_writer.offset < shard_writer.total) {
                if (self.interface.buffer.len > self.interface.end) return;
                continue;
            }

            self.stream_index += 1;
            if (self.stream_index < order.len) {
                const next = &self.shard_writers[order[self.stream_index]];
                self.partitionedSyncFromShard(next);
            }
        }

        self.interface = .failing;
    }

    fn partitionedSyncIntoShard(self: *DirectMemoryWriter, shard_writer: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.interface.buffer.ptr != shard_writer.interface.buffer.ptr) return std.Io.Writer.Error.WriteFailed;
        shard_writer.interface.end = self.interface.end;
    }

    fn partitionedSyncFromShard(self: *DirectMemoryWriter, shard_writer: *const DirectShardWriter) void {
        const shard_remaining = shard_writer.total - (shard_writer.offset + shard_writer.interface.end);
        const visible_len = @min(shard_writer.interface.buffer.len, shard_writer.interface.end + shard_remaining);
        self.interface.buffer = shard_writer.interface.buffer[0..visible_len];
        self.interface.end = shard_writer.interface.end;
    }

    fn replicatedDrain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        var consumed: usize = 0;

        const writeChunk = struct {
            fn call(self_: *DirectMemoryWriter, chunk: []const u8, consumed_: *usize) std.Io.Writer.Error!void {
                var remaining = chunk;
                while (remaining.len > 0) {
                    try self_.replicatedEnsureWritable(1);
                    const tensor_remaining = self_.replicatedRemaining();
                    if (tensor_remaining == 0) return std.Io.Writer.Error.WriteFailed;
                    const to_write = @min(remaining.len, tensor_remaining);
                    try self_.replicatedBroadcast(remaining[0..to_write]);
                    consumed_.* += to_write;
                    remaining = remaining[to_write..];
                }
            }
        }.call;

        for (data) |chunk| {
            try writeChunk(self, chunk, &consumed);
        }
        if (data.len > 0 and splat > 1) {
            const last = data[data.len - 1];
            for (0..splat - 1) |_| {
                try writeChunk(self, last, &consumed);
            }
        }

        return consumed;
    }

    fn replicatedFlush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        for (self.shard_writers) |*shard_writer| {
            try shard_writer.interface.vtable.flush(&shard_writer.interface);
        }
        self.replicatedSyncFromFirst();
        self.interface = .failing;
    }

    fn replicatedRebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;
        try self.replicatedEnsureWritable(capacity);
    }

    fn replicatedEnsureWritable(self: *DirectMemoryWriter, capacity: usize) std.Io.Writer.Error!void {
        if (capacity == 0) return;
        while (self.interface.buffer.len - self.interface.end < capacity) {
            if (capacity > self.interface.buffer.len and self.interface.buffer.len != 0) {
                return std.Io.Writer.Error.WriteFailed;
            }
            try self.replicatedAdvance();
            if (self.interface.buffer.len == 0) return std.Io.Writer.Error.WriteFailed;
            if (capacity > self.interface.buffer.len) return std.Io.Writer.Error.WriteFailed;
        }
    }

    fn replicatedAdvance(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        if (self.replicatedRemaining() == 0) {
            self.interface = .failing;
            return;
        }

        const first = &self.shard_writers[0];
        if (first.interface.end > 0) {
            for (self.shard_writers) |*shard_writer| {
                try shard_writer.flushBuffered();
            }
        }
        self.replicatedSyncFromFirst();
    }

    fn replicatedBroadcast(self: *DirectMemoryWriter, chunk: []const u8) std.Io.Writer.Error!void {
        for (self.shard_writers) |*shard_writer| {
            const wrote = try shard_writer.interface.vtable.drain(&shard_writer.interface, &.{chunk}, 1);
            if (wrote != chunk.len) return std.Io.Writer.Error.WriteFailed;
        }
        self.replicatedSyncFromFirst();
    }

    fn replicatedRemaining(self: *const DirectMemoryWriter) usize {
        const first = &self.shard_writers[0];
        const first_written = first.offset + first.interface.end;
        for (self.shard_writers[1..]) |*shard_writer| {
            stdx.debug.assert(
                shard_writer.offset + shard_writer.interface.end == first_written,
                "replicated writers out of sync: expected written={}, got {}",
                .{ first_written, shard_writer.offset + shard_writer.interface.end },
            );
        }
        return first.total - first_written;
    }

    fn replicatedSyncFromFirst(self: *DirectMemoryWriter) void {
        const first = &self.shard_writers[0];
        const remaining = first.total - (first.offset + first.interface.end);
        const visible_len = @min(first.interface.buffer.len, first.interface.end + remaining);
        self.interface.buffer = first.interface.buffer[0..visible_len];
        self.interface.end = first.interface.end;
    }
};

pub const LoadOpts = struct {
    parallelism: usize,
    store: *const TensorStore,
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
    opts: LoadOpts,
) !Bufferized(ModelType) {
    var bufferized = try mem.bufferize(allocator, ModelType, model);

    const first_device = platform.devices[0]; // Temporary until sharding is re-exposed
    const dma_alloc: mem.DmaAllocator = .init(allocator, &first_device);
    const pool_count = platform.devices.len;
    // One DirectShardWriter reserves one chunk from its device pool at init time.
    // Keep per-device capacity at least equal to loader parallelism to avoid
    // blocking all workers during writer initialization.
    const chunks_per_pool = @max(@as(usize, 1), @max(opts.dma_chunks, opts.parallelism));
    const buffer_pools = try allocator.alloc(mem.DynamicBufferPool, pool_count);
    defer allocator.free(buffer_pools);
    for (buffer_pools) |*pool_| {
        pool_.* = .init(chunks_per_pool, opts.dma_chunk_size);
    }
    defer for (buffer_pools) |*pool_| {
        pool_.deinit(dma_alloc.allocator());
    };

    const replicated_sharding = try sharding_.replicatedSharding(opts.shardings[0].physical);

    const Ctx = struct {
        allocator: std.mem.Allocator,
        dma_allocator: std.mem.Allocator,
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
        .store = opts.store,
        .allocator = allocator,
        .dma_allocator = dma_alloc.allocator(),
        .pinned_buffer_pools = buffer_pools,
        .io = io,
        .shardings = opts.shardings,
        .replicated_sharding = replicated_sharding,
        .progress = opts.progress,
        .group = .init(opts.parallelism),
    };

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
                    const select_sharding = selectSharding(ctx_.shardings, shape);
                    const sharding = if (select_sharding) |s| s else blk: {
                        log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding:\n {f}", .{ reader.tensor.name, shape, ctx_.replicated_sharding });
                        break :blk ctx_.replicated_sharding;
                    };

                    var writer = MemoryWriter.initWithDevicePools(
                        ctx_.dma_allocator,
                        ctx_.io,
                        ctx_.platform,
                        ctx_.pinned_buffer_pools,
                        shape,
                        sharding,
                        ctx_.buffers[i_],
                    ) catch unreachable;
                    defer writer.deinit(ctx_.dma_allocator);

                    const scale = 4 * 1024 * 1024;

                    if (ctx_.progress) |progress| {
                        var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
                        defer node.end();
                        writer.setProgress(&node);
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

// todo: move
pub fn selectSharding(shardings: []const Sharding, shape: Shape) ?Sharding {
    for (shardings) |s| {
        var match = false;
        for (0..shape.rank()) |i| {
            const spec = shape.partition(i);
            if (spec == .axis) {
                if (s.binding(spec.axis)) |_| {
                    match = true;
                } else {
                    match = false;
                    break;
                }
            }
        }
        if (match) return s;
    }
    return null;
}
