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
        dma_allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: DirectMemoryWriter.Pools,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !MemoryWriter {
        return switch (platform.target) {
            .cuda => .{ .direct = try DirectMemoryWriter.init(allocator, io, platform, pools, shape, sharding, buffer) },
            .rocm, .tpu, .neuron, .cpu => .{ .buffered = try BufferedMemoryWriter.init(dma_allocator, io, platform, shape, sharding, buffer) },
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
    const StreamSegment = struct {
        start: usize,
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

    const SegmentBuildError = error{
        NegativeStart,
        InvalidSliceSize,
        InvalidRank,
        InvalidWriterIndex,
    } || std.mem.Allocator.Error;

    pub const Pools = union(enum) {
        single: struct {
            pool: *mem.DynamicBufferPool,
            allocator: std.mem.Allocator,
        },
        per_device: struct {
            pools: []mem.DynamicBufferPool,
            allocators: []const std.mem.Allocator,
        },
    };

    allocator: std.mem.Allocator,
    shard_writers: []DirectShardWriter,
    stream_segments: []StreamSegment,
    segment_mirrors: []usize,
    segment_index: usize = 0,
    segment_offset: usize = 0,
    active_writer_index: ?usize = null,
    interface: std.Io.Writer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: Pools,
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
            const pool, const shard_dma_allocator = switch (pools) {
                .single => |single| .{ single.pool, single.allocator },
                .per_device => |per_device| .{
                    poolForDevice(platform, per_device.pools, shard.device_id),
                    allocatorForDevice(platform, per_device.allocators, shard.device_id),
                },
            };
            shard_writers[i] = try DirectShardWriter.init(
                shard_dma_allocator,
                io,
                shard.memory(platform, .default),
                pool,
                shard.shape,
            );
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }

        buffer.* = Buffer.fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());
        stdx.debug.assert(shard_writers.len > 0, "DirectMemoryWriter requires at least one shard writer", .{});

        const layout = try buildLayout(allocator, shape, placement);

        const initial_writer_index = if (layout.segments.len > 0)
            layout.segments[0].primary_writer
        else
            0;
        const initial_writer = &shard_writers[initial_writer_index];
        const initial_visible_len: usize = if (layout.segments.len == 0)
            0
        else blk: {
            const segment = layout.segments[0];
            const shard_remaining = initial_writer.total - (initial_writer.offset + initial_writer.interface.end);
            break :blk @min(
                initial_writer.interface.buffer.len,
                initial_writer.interface.end + @min(segment.len, shard_remaining),
            );
        };

        return .{
            .allocator = allocator,
            .shard_writers = shard_writers,
            .stream_segments = layout.segments,
            .segment_mirrors = layout.mirrors,
            .active_writer_index = if (layout.segments.len > 0) initial_writer_index else null,
            .interface = .{
                .buffer = initial_writer.interface.buffer[0..initial_visible_len],
                .end = initial_writer.interface.end,
                .vtable = &.{
                    .drain = partitionedDrain,
                    .flush = partitionedFlush,
                    .rebase = partitionedRebase,
                },
            },
        };
    }

    fn buildLayout(
        allocator: std.mem.Allocator,
        shape: Shape,
        placement: Placement,
    ) !struct { segments: []StreamSegment, mirrors: []usize } {
        var raw_segments_builder: std.ArrayListUnmanaged(RawSegment) = .{};
        defer raw_segments_builder.deinit(allocator);
        for (placement.shards.constSlice(), 0..) |shard, i| {
            appendShardSegments(allocator, &raw_segments_builder, shape, shard, i, placement.shards.len) catch |err| {
                log.warn(
                    "DirectMemoryWriter.chooseLayout: segment build failure shard_index={d} reason={s} global_shape={f} shard_shape={f} shard={f}",
                    .{ i, @errorName(err), shape, shard.shape, shard },
                );
                return error.NonContiguousShardPlacement;
            };
        }
        const raw_segments = try raw_segments_builder.toOwnedSlice(allocator);
        defer allocator.free(raw_segments);

        var filtered_len: usize = 0;
        for (raw_segments, 0..) |segment, segment_index| {
            if (segment.len == 0) {
                // Defensive: zero-length segments are never meaningful for the
                // stream plan and can appear if upstream placement metadata is noisy.
                if (segment.writer_index >= placement.shards.len) {
                    log.warn(
                        "DirectMemoryWriter.chooseLayout: dropping empty segment with invalid writer index raw_segment_index={d} writer_index={d} shards_len={d} start={d} global_shape={f}",
                        .{ segment_index, segment.writer_index, placement.shards.len, segment.start, shape },
                    );
                }
                continue;
            }
            if (segment.writer_index >= placement.shards.len) {
                log.warn(
                    "DirectMemoryWriter.chooseLayout: invalid writer index at raw_segment_index={d} writer_index={d} shards_len={d} start={d} len={d} global_shape={f}",
                    .{ segment_index, segment.writer_index, placement.shards.len, segment.start, segment.len, shape },
                );
                return error.NonContiguousShardPlacement;
            }
            if (segment.start + segment.len > shape.byteSize()) {
                log.warn(
                    "DirectMemoryWriter.chooseLayout: raw segment out of range at raw_segment_index={d} start={d} len={d} global_size={d} writer_index={d} global_shape={f}",
                    .{ segment_index, segment.start, segment.len, shape.byteSize(), segment.writer_index, shape },
                );
                return error.NonContiguousShardPlacement;
            }
            raw_segments[filtered_len] = segment;
            filtered_len += 1;
        }
        const planned_segments = raw_segments[0..filtered_len];

        std.mem.sort(RawSegment, planned_segments, {}, struct {
            fn lessThan(_: void, lhs: RawSegment, rhs: RawSegment) bool {
                if (lhs.start != rhs.start) return lhs.start < rhs.start;
                if (lhs.len != rhs.len) return lhs.len < rhs.len;
                return lhs.writer_index < rhs.writer_index;
            }
        }.lessThan);

        // Validate that each shard writer receives exactly its shard payload.
        var writer_totals = try allocator.alloc(usize, placement.shards.len);
        defer allocator.free(writer_totals);
        @memset(writer_totals, 0);
        for (planned_segments) |segment| {
            writer_totals[segment.writer_index] += segment.len;
        }
        for (placement.shards.constSlice(), 0..) |shard, i| {
            if (writer_totals[i] != shard.shape.byteSize()) {
                log.warn(
                    "DirectMemoryWriter.chooseLayout: writer byte mismatch writer_index={d} scheduled={d} expected={d} shard={f}",
                    .{ i, writer_totals[i], shard.shape.byteSize(), shard },
                );
                return error.NonContiguousShardPlacement;
            }
        }

        var stream_segments_builder: std.ArrayListUnmanaged(StreamSegment) = .{};
        defer stream_segments_builder.deinit(allocator);
        var mirrors_builder: std.ArrayListUnmanaged(usize) = .{};
        defer mirrors_builder.deinit(allocator);

        var i: usize = 0;
        var cursor: usize = 0;
        while (i < planned_segments.len) {
            const seg_start = planned_segments[i].start;
            const seg_len = planned_segments[i].len;

            if (seg_start != cursor) {
                log.warn(
                    "DirectMemoryWriter.chooseLayout: non-contiguous coverage at sorted_segment_index={d} expected_start={d} got_start={d} len={d} writer_index={d} global_size={d}",
                    .{ i, cursor, seg_start, seg_len, planned_segments[i].writer_index, shape.byteSize() },
                );
                return error.NonContiguousShardPlacement;
            }

            const mirror_start = mirrors_builder.items.len;
            var j = i + 1;
            while (j < planned_segments.len and planned_segments[j].start == seg_start and planned_segments[j].len == seg_len) : (j += 1) {
                try mirrors_builder.append(allocator, planned_segments[j].writer_index);
            }

            try stream_segments_builder.append(allocator, .{
                .start = seg_start,
                .len = seg_len,
                .primary_writer = planned_segments[i].writer_index,
                .mirror_start = mirror_start,
                .mirror_len = j - i - 1,
            });
            cursor += seg_len;
            i = j;
        }

        if (cursor != shape.byteSize()) {
            log.warn(
                "DirectMemoryWriter.chooseLayout: coverage mismatch covered={d} global_size={d}",
                .{ cursor, shape.byteSize() },
            );
            return error.NonContiguousShardPlacement;
        }

        const segments = try stream_segments_builder.toOwnedSlice(allocator);
        errdefer allocator.free(segments);
        const mirrors = try mirrors_builder.toOwnedSlice(allocator);

        return .{
            .segments = segments,
            .mirrors = mirrors,
        };
    }

    fn appendShardSegments(
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged(RawSegment),
        shape: Shape,
        shard: Placement.Shard,
        writer_index: usize,
        writer_count: usize,
    ) SegmentBuildError!void {
        const strides = shape.computeByteStrides();
        const rank: usize = shape.rank();
        if (rank > Shape.MAX_RANK) return error.InvalidRank;
        if (writer_index >= writer_count) return error.InvalidWriterIndex;

        // Canonicalize per-axis [start, size] regardless of shard.slices order.
        var starts: [Shape.MAX_RANK]i64 = [_]i64{0} ** Shape.MAX_RANK;
        var sizes: [Shape.MAX_RANK]i64 = [_]i64{0} ** Shape.MAX_RANK;
        for (0..rank) |ax| {
            sizes[ax] = shape.dim(ax);
        }
        for (shard.slices.constSlice()) |s| {
            if (s.size < 0 or s.start < 0) return error.InvalidSliceSize;
            const ax: usize = @intCast(s.axis);
            starts[ax] = s.start;
            sizes[ax] = s.size;
        }

        for (0..rank) |ax| {
            const dim = shape.dim(ax);
            if (sizes[ax] > dim) return error.InvalidSliceSize;
            if (starts[ax] + sizes[ax] > dim) return error.InvalidSliceSize;
            // A full dimension cannot have a non-zero start.
            if (sizes[ax] == dim and starts[ax] != 0) return error.InvalidSliceSize;
        }

        if (rank == 0) {
            try appendMergedRawSegment(allocator, out, writer_index, writer_count, 0, shape.byteSize());
            return;
        }

        try appendAxisSegments(
            allocator,
            out,
            writer_index,
            writer_count,
            0,
            0,
            rank,
            &starts,
            &sizes,
            strides.constSlice(),
        );
    }

    fn appendAxisSegments(
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged(RawSegment),
        writer_index: usize,
        writer_count: usize,
        axis: usize,
        base_start: i64,
        rank: usize,
        starts: *const [Shape.MAX_RANK]i64,
        sizes: *const [Shape.MAX_RANK]i64,
        strides: []const i64,
    ) SegmentBuildError!void {
        if (writer_index >= writer_count) return error.InvalidWriterIndex;
        const axis_start = starts[axis];
        const axis_size = sizes[axis];
        if (axis_size == 0) return;

        if (axis + 1 == rank) {
            const segment_start = base_start + axis_start * strides[axis];
            if (segment_start < 0) return error.NegativeStart;
            const segment_len_i64 = axis_size * strides[axis];
            if (segment_len_i64 < 0) return error.InvalidSliceSize;
            try appendMergedRawSegment(
                allocator,
                out,
                writer_index,
                writer_count,
                @intCast(segment_start),
                @intCast(segment_len_i64),
            );
            return;
        }

        var i: i64 = 0;
        while (i < axis_size) : (i += 1) {
            const child_start = base_start + (axis_start + i) * strides[axis];
            if (child_start < 0) return error.NegativeStart;
            try appendAxisSegments(
                allocator,
                out,
                writer_index,
                writer_count,
                axis + 1,
                child_start,
                rank,
                starts,
                sizes,
                strides,
            );
        }
    }

    fn appendMergedRawSegment(
        allocator: std.mem.Allocator,
        out: *std.ArrayListUnmanaged(RawSegment),
        writer_index: usize,
        writer_count: usize,
        start: usize,
        len: usize,
    ) SegmentBuildError!void {
        if (writer_index >= writer_count) return error.InvalidWriterIndex;
        if (len == 0) return;
        if (out.items.len > 0) {
            const last = &out.items[out.items.len - 1];
            if (last.writer_index == writer_index and last.start + last.len == start) {
                last.len += len;
                return;
            }
        }
        try out.append(allocator, .{
            .writer_index = writer_index,
            .start = start,
            .len = len,
        });
    }

    fn poolForDevice(platform: *const Platform, pools: []mem.DynamicBufferPool, device_id: usize) *mem.DynamicBufferPool {
        stdx.debug.assert(pools.len == platform.devices.len, "Expected one DMA pool per device, got pools={} devices={}", .{ pools.len, platform.devices.len });
        for (platform.devices, 0..) |d, i| {
            if (d.id() == device_id) return &pools[i];
        }
        unreachable;
    }

    fn allocatorForDevice(platform: *const Platform, allocators: []const std.mem.Allocator, device_id: usize) std.mem.Allocator {
        stdx.debug.assert(allocators.len == platform.devices.len, "Expected one DMA allocator per device, got allocators={} devices={}", .{ allocators.len, platform.devices.len });
        for (platform.devices, 0..) |d, i| {
            if (d.id() == device_id) return allocators[i];
        }
        unreachable;
    }

    pub fn deinit(self: *DirectMemoryWriter) void {
        for (self.shard_writers) |*writer| {
            writer.deinit();
        }
        self.allocator.free(self.shard_writers);
        self.allocator.free(self.stream_segments);
        self.allocator.free(self.segment_mirrors);
    }

    pub fn setProgress(_: *DirectMemoryWriter, _: ?*std.Progress.Node) void {}

    fn partitionedDrain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        var consumed: usize = 0;

        const writeChunk = struct {
            fn call(self_: *DirectMemoryWriter, chunk: []const u8, consumed_: *usize) std.Io.Writer.Error!void {
                var remaining = chunk;
                while (remaining.len > 0) {
                    try self_.partitionedEnsureWritable(1);
                    const segment_remaining = self_.currentSegmentRemaining();
                    if (segment_remaining == 0) return std.Io.Writer.Error.WriteFailed;
                    const writable = self_.interface.buffer.len - self_.interface.end;
                    if (writable == 0) return std.Io.Writer.Error.WriteFailed;

                    const to_write = @min(remaining.len, @min(segment_remaining, writable));
                    @memcpy(
                        self_.interface.buffer[self_.interface.end .. self_.interface.end + to_write],
                        remaining[0..to_write],
                    );
                    self_.interface.end += to_write;
                    try self_.partitionedSyncIntoActive();

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

    fn partitionedFlush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        try self.partitionedSyncIntoActive();
        self.partitionedAdvanceCompleted();

        if (self.currentSegment() != null) return std.Io.Writer.Error.WriteFailed;

        for (self.shard_writers) |*shard_writer| {
            if (shard_writer.interface.end > 0) {
                try shard_writer.flushBuffered();
            }
            if (shard_writer.offset < shard_writer.total) return std.Io.Writer.Error.WriteFailed;
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
            try self.partitionedSyncIntoActive();
            self.partitionedAdvanceCompleted();

            const segment = self.currentSegment() orelse {
                self.interface = .failing;
                return std.Io.Writer.Error.WriteFailed;
            };

            self.partitionedSyncFromShard(segment.primary_writer, self.currentSegmentRemaining());
            if (self.interface.buffer.len - self.interface.end >= capacity) return;
            if (capacity > self.interface.buffer.len and self.interface.buffer.len != 0) {
                return std.Io.Writer.Error.WriteFailed;
            }

            const shard_writer = &self.shard_writers[segment.primary_writer];
            if (shard_writer.interface.end > 0) {
                try shard_writer.flushBuffered();
                self.partitionedSyncFromShard(segment.primary_writer, self.currentSegmentRemaining());
                if (self.interface.buffer.len - self.interface.end >= capacity) return;
            }

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

    fn segmentMirrors(self: *const DirectMemoryWriter, segment: StreamSegment) []const usize {
        return self.segment_mirrors[segment.mirror_start .. segment.mirror_start + segment.mirror_len];
    }

    fn partitionedAdvanceCompleted(self: *DirectMemoryWriter) void {
        while (self.currentSegment()) |segment| {
            if (self.currentSegmentRemaining() != 0) {
                self.partitionedSyncFromShard(segment.primary_writer, self.currentSegmentRemaining());
                return;
            }
            self.segment_index += 1;
            self.segment_offset = 0;
        }
        if (self.shard_writers.len > 0) {
            self.active_writer_index = null;
            self.interface.buffer = self.shard_writers[0].interface.buffer[0..0];
            self.interface.end = 0;
        } else {
            self.interface = .failing;
        }
    }

    fn partitionedSyncIntoActive(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        const writer_index = self.active_writer_index orelse return;
        const segment = self.currentSegment() orelse return;
        if (segment.primary_writer != writer_index) return;

        const shard_writer = &self.shard_writers[writer_index];
        if (self.interface.end > shard_writer.interface.end) {
            // writableSlice APIs write directly into the exposed shard buffer;
            // mirror those bytes and advance stream progress when end increases.
            const old_end = shard_writer.interface.end;
            const new_end = self.interface.end;
            if (new_end > shard_writer.interface.buffer.len) return std.Io.Writer.Error.WriteFailed;
            const chunk = shard_writer.interface.buffer[old_end..new_end];
            shard_writer.interface.end = new_end;
            self.segment_offset += chunk.len;
            try self.mirrorChunk(segment, chunk);
            return;
        }

        if (self.interface.end < shard_writer.interface.end) {
            self.interface.end = shard_writer.interface.end;
        }
    }

    fn partitionedSyncFromShard(self: *DirectMemoryWriter, writer_index: usize, segment_remaining: usize) void {
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

        for (self.segmentMirrors(segment)) |writer_index| {
            const shard_writer = &self.shard_writers[writer_index];
            const wrote = try shard_writer.interface.vtable.drain(&shard_writer.interface, &.{chunk}, 1);
            if (wrote != chunk.len) return std.Io.Writer.Error.WriteFailed;
        }
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

    // `load` can run per-tensor work concurrently. Callers may pass allocators
    // that are not thread-safe (for example arena allocators in examples/llama),
    // so serialize allocator access for writer/layout temporary allocations.
    var thread_safe_allocator: std.heap.ThreadSafeAllocator = .{
        .child_allocator = allocator,
        .io = io,
    };
    const concurrent_allocator = thread_safe_allocator.allocator();

    const pool_count = platform.devices.len;
    const dma_allocators = try allocator.alloc(mem.DmaAllocator, pool_count);
    defer allocator.free(dma_allocators);
    const dma_allocator_views = try allocator.alloc(std.mem.Allocator, pool_count);
    defer allocator.free(dma_allocator_views);
    for (platform.devices, 0..) |*device, i| {
        dma_allocators[i] = .init(concurrent_allocator, device);
        dma_allocator_views[i] = dma_allocators[i].allocator();
    }

    // One DirectShardWriter reserves one chunk from its device pool at init time.
    // Keep per-device capacity at least equal to loader parallelism to avoid
    // blocking all workers during writer initialization.
    const chunks_per_pool = @max(@as(usize, 1), @max(opts.dma_chunks, opts.parallelism));
    const buffer_pools = try allocator.alloc(mem.DynamicBufferPool, pool_count);
    defer allocator.free(buffer_pools);
    for (buffer_pools) |*pool_| {
        pool_.* = .init(chunks_per_pool, opts.dma_chunk_size);
    }
    defer for (buffer_pools, 0..) |*pool_, i| {
        pool_.deinit(dma_allocator_views[i]);
    };

    const replicated_sharding = try sharding_.replicatedSharding(platform);

    const Ctx = struct {
        allocator: std.mem.Allocator,
        buffered_allocator: std.mem.Allocator,
        dma_allocators: []const std.mem.Allocator,
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
        .allocator = concurrent_allocator,
        .buffered_allocator = concurrent_allocator,
        .dma_allocators = dma_allocator_views,
        .pinned_buffer_pools = buffer_pools,
        .io = io,
        .shardings = opts.shardings,
        .replicated_sharding = replicated_sharding,
        .progress = opts.progress,
        .group = .init(opts.parallelism),
    };
    // defer allocator.free(walk_ctx.buffers);

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

                    var writer = MemoryWriter.init(
                        ctx_.allocator,
                        ctx_.buffered_allocator,
                        ctx_.io,
                        ctx_.platform,
                        .{
                            .per_device = .{
                                .pools = ctx_.pinned_buffer_pools,
                                .allocators = ctx_.dma_allocators,
                            },
                        },
                        shape,
                        sharding,
                        ctx_.buffers[i_],
                    ) catch unreachable;
                    defer writer.deinit(ctx_.buffered_allocator);

                    const scale = 1024;

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
