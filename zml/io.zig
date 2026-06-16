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
const tracer = @import("profiling/tracer.zig");
const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Placement = Sharding.Placement;
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

            if (@TypeOf(partitioning) == @TypeOf(null)) {
                @compileError("TensorStore.View.createTensor partitioning cannot be null; pass .replicated or an explicit partitioning");
            }

            switch (@typeInfo(@TypeOf(partitioning))) {
                .optional => @compileError("TensorStore.View.createTensor partitioning cannot be optional; pass .replicated or an explicit partitioning"),
                .enum_literal => switch (partitioning) {
                    .replicated => ptr.shape = ptr.shape.withReplicatedPartitioning(),
                    else => @compileError("Only .replicated is supported as a standalone partitioning enum literal"),
                },
                else => ptr.shape = ptr.shape.withPartitioning(partitioning),
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
        dma_chunk_size: usize,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !MemoryWriter {
        return switch (platform.target) {
            .cuda, .oneapi => .{ .direct = try DirectMemoryWriter.init(allocator, io, platform, pools, dma_allocators, dma_chunk_size, shape, sharding, buffer) },
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

        const buf = try pool.get(allocator, io);

        return .{
            .allocator = allocator,
            .io = io,
            .memory = memory,
            .pool = pool,
            .total = shape.byteSize(),
            .pjrt_buffer = pjrt_buffer,
            .transfer_manager = transfer_manager,
            .interface = .{
                .buffer = buf[0..@min(buf.len, shape.byteSize())],
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

    fn drain(w: *std.Io.Writer, data: []const []const u8, _: usize) std.Io.Writer.Error!usize {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));

        const chunk = data[0];
        if (chunk.len > self.interface.buffer.len) return std.Io.Writer.Error.WriteFailed;
        if (chunk.len > self.total - (self.offset + self.interface.end)) return std.Io.Writer.Error.WriteFailed;

        const needs_fresh_buffer = chunk.len > self.interface.buffer.len - self.interface.end;
        if (needs_fresh_buffer) {
            try self.interface.flush();
        }

        @memcpy(self.interface.buffer[self.interface.end..][0..chunk.len], chunk);
        self.interface.end += chunk.len;

        const buffer_full = self.interface.end == self.interface.buffer.len;
        if (buffer_full) {
            try self.interface.flush();
        }

        return chunk.len;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        try self.flushBuffered();
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectShardWriter = @alignCast(@fieldParentPtr("interface", w));
        if (self.interface.buffer.len - self.interface.end >= capacity) return;
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;
        try self.interface.flush();
    }

    fn flushBuffered(self: *DirectShardWriter) std.Io.Writer.Error!void {
        if (self.offset >= self.total) return;

        const pjrt_api = self.memory.platform.pjrt_api;

        const current_buffer = self.interface.buffer;
        const buffered = self.interface.buffered();

        const slice = buffered[0..@min(buffered.len, self.total - self.offset)];
        if (slice.len == 0) return;
        const is_last = (self.offset + slice.len) >= self.total;

        const transfer_event = self.transfer_manager.transferData(pjrt_api, 0, slice, @intCast(self.offset), is_last) catch |err| {
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
            const buf = self.pool.get(self.allocator, self.io) catch |err| {
                log.err("unable to get a new buffer from the pool: {any}", .{err});
                return std.Io.Writer.Error.WriteFailed;
            };
            self.interface.buffer = buf[0..@min(buf.len, self.total - (self.offset + slice.len))];
        }
        self.flip_flop ^= 1;
        self.offset += slice.len;
    }
};

const ShardProgress = struct {
    const scale: usize = 1024;

    node: std.Progress.Node,
    label: [32]u8 = undefined,
    completed: usize = 0,

    fn set(self: *ShardProgress, completed: usize) void {
        self.completed = completed;
        self.node.setCompletedItems(std.math.divCeil(usize, self.completed, scale) catch unreachable);
    }
};

// Dispatch planning bridges two different orders:
//
// 1. Placement traversal emits byte ranges per shard writer, in device order.
//    That order is convenient for asking "what bytes belong to this shard?",
//    but it does not match how a tensor file is read.
//
// 2. Readers stream tensor bytes in global row-major order. DirectMemoryWriter
//    must therefore consume dispatch spans in increasing global byte offset.
//
// We first collect placement spans, sort them by global byte range, then fold
// identical ranges into one primary dispatch span plus mirror writers. Identical
// ranges represent replicated shard data: the reader provides those bytes once,
// the primary writer receives them zero-copy, and mirrors receive a copy.
const DispatchSpans = struct {
    const DispatchSpan = struct {
        start: usize,
        end: usize,
        primary_writer: usize,
        mirror_writer_start: usize,
        mirror_writer_len: usize,
    };

    const PlacementSpan = struct {
        writer_index: usize,
        start: usize,
        len: usize,
    };

    const PlacementSpans = std.MultiArrayList(PlacementSpan);

    spans: []DispatchSpan,
    mirror_writers: []usize,

    fn init(allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) !DispatchSpans {
        var placement_spans: PlacementSpans = .empty;
        defer placement_spans.deinit(allocator);
        var placement_span_order: std.ArrayList(usize) = .empty;
        defer placement_span_order.deinit(allocator);

        const placement = try sharding.placement(shape);
        const byte_strides = shape.computeByteStrides();

        for (sharding.devicesInCanonicalOrder(), 0..) |device, writer_index| {
            try appendShardPlacementSpans(
                allocator,
                &placement_spans,
                &placement_span_order,
                shape,
                placement.slices(device.coords).constSlice(),
                byte_strides.constSlice(),
                writer_index,
            );
        }
        sortPlacementSpans(placement_spans, placement_span_order.items);

        var spans: std.ArrayList(DispatchSpan) = .empty;
        errdefer spans.deinit(allocator);
        var mirror_writers: std.ArrayList(usize) = .empty;
        errdefer mirror_writers.deinit(allocator);

        try compactPlacementSpans(allocator, placement_spans, placement_span_order.items, shape.byteSize(), &spans, &mirror_writers);

        const spans_ = try spans.toOwnedSlice(allocator);
        errdefer allocator.free(spans_);

        const mirror_writers_ = try mirror_writers.toOwnedSlice(allocator);
        errdefer allocator.free(mirror_writers_);

        return .{
            .spans = spans_,
            .mirror_writers = mirror_writers_,
        };
    }

    fn deinit(self: DispatchSpans, allocator: std.mem.Allocator) void {
        allocator.free(self.spans);
        allocator.free(self.mirror_writers);
    }

    fn sortPlacementSpans(placement_spans: PlacementSpans, placement_span_order: []usize) void {
        const placement = placement_spans.slice();

        const SortContext = struct {
            starts: []const usize,
            lens: []const usize,
            writer_indices: []const usize,

            fn lessThan(ctx: @This(), lhs_index: usize, rhs_index: usize) bool {
                if (ctx.starts[lhs_index] != ctx.starts[rhs_index]) return ctx.starts[lhs_index] < ctx.starts[rhs_index];
                if (ctx.lens[lhs_index] != ctx.lens[rhs_index]) return ctx.lens[lhs_index] < ctx.lens[rhs_index];
                return ctx.writer_indices[lhs_index] < ctx.writer_indices[rhs_index];
            }
        };

        std.mem.sort(usize, placement_span_order, SortContext{
            .starts = placement.items(.start),
            .lens = placement.items(.len),
            .writer_indices = placement.items(.writer_index),
        }, SortContext.lessThan);
    }

    fn compactPlacementSpans(
        allocator: std.mem.Allocator,
        placement_spans: PlacementSpans,
        placement_span_order: []const usize,
        total_bytes: usize,
        spans: *std.ArrayList(DispatchSpan),
        mirror_writers: *std.ArrayList(usize),
    ) !void {
        const placement = placement_spans.slice();
        const starts = placement.items(.start);
        const lens = placement.items(.len);
        const writer_indices = placement.items(.writer_index);

        var i: usize = 0;
        var cursor: usize = 0;
        while (i < placement_span_order.len) {
            const first_index = placement_span_order[i];
            const first_start = starts[first_index];
            const first_len = lens[first_index];
            if (first_start != cursor) return error.NonContiguousShardPlacement;

            const mirror_writer_start = mirror_writers.items.len;
            var j = i + 1;
            while (j < placement_span_order.len) : (j += 1) {
                const mirror_index = placement_span_order[j];
                if (starts[mirror_index] != first_start or lens[mirror_index] != first_len) break;
                try mirror_writers.append(allocator, writer_indices[mirror_index]);
            }

            try spans.append(allocator, .{
                .start = first_start,
                .end = first_start + first_len,
                .primary_writer = writer_indices[first_index],
                .mirror_writer_start = mirror_writer_start,
                .mirror_writer_len = j - i - 1,
            });
            cursor += first_len;
            i = j;
        }

        if (cursor != total_bytes) return error.NonContiguousShardPlacement;
    }

    fn appendPlacementSpan(
        allocator: std.mem.Allocator,
        placement_spans: *PlacementSpans,
        placement_span_order: *std.ArrayList(usize),
        writer_index: usize,
        start: usize,
        len: usize,
    ) !void {
        try placement_spans.append(allocator, .{ .writer_index = writer_index, .start = start, .len = len });
        try placement_span_order.append(allocator, placement_spans.len - 1);
    }

    fn appendShardPlacementSpans(
        allocator: std.mem.Allocator,
        placement_spans: *PlacementSpans,
        placement_span_order: *std.ArrayList(usize),
        shape: Shape,
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        writer_index: usize,
    ) !void {
        if (shape.rank() == 0) {
            try appendPlacementSpan(allocator, placement_spans, placement_span_order, writer_index, 0, shape.byteSize());
            return;
        }

        try appendShardAxisPlacementSpans(allocator, placement_spans, placement_span_order, slices, byte_strides, writer_index, 0, contiguousSliceAxis(shape, slices), 0);
    }

    fn appendShardAxisPlacementSpans(
        allocator: std.mem.Allocator,
        placement_spans: *PlacementSpans,
        placement_span_order: *std.ArrayList(usize),
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        writer_index: usize,
        axis: usize,
        contiguous_axis: usize,
        base_start: i64,
    ) !void {
        const slice = slices[axis];
        if (slice.size == 0) return;

        if (axis == contiguous_axis) {
            const span_start: usize = @intCast(base_start + slice.start * byte_strides[axis]);
            const span_len: usize = @intCast(slice.size * byte_strides[axis]);
            try appendPlacementSpan(allocator, placement_spans, placement_span_order, writer_index, span_start, span_len);
            return;
        }

        var i: i64 = 0;
        while (i < slice.size) : (i += 1) {
            const child_start = base_start + (slice.start + i) * byte_strides[axis];
            try appendShardAxisPlacementSpans(allocator, placement_spans, placement_span_order, slices, byte_strides, writer_index, axis + 1, contiguous_axis, child_start);
        }
    }

    fn contiguousSliceAxis(shape: Shape, slices: []const Placement.Slice1d) usize {
        var axis = shape.rank() - 1;
        while (axis > 0) {
            const slice = slices[axis];
            if (slice.start != 0 or slice.size != shape.dim(axis)) break;
            axis -= 1;
        }
        return axis;
    }
};

// Direct load writer state machine.
//
//     std.Io.Reader
//          |
//          v
//     DirectMemoryWriter.interface.buffer
//          | aliases active primary DirectShardWriter DMA buffer
//          v
//     commitWindow()
//          |-- primary: advance DirectShardWriter.interface.end
//          |-- mirrors: copy committed bytes with writeAll()
//          |-- boundaries: flush full shard buffers / chunk fence
//          v
//     DirectShardWriter.flushBuffered()
//          |
//          v
//     PJRT AsyncHostToDeviceTransferManager
//
// `byte_cursor` is the global tensor position. Shard writer `interface.end` is
// local to the current shard DMA buffer.
pub const DirectMemoryWriter = struct {
    allocator: std.mem.Allocator,
    shard_writers: []DirectShardWriter,
    dispatch_spans: DispatchSpans,
    span_index: usize = 0,
    byte_cursor: usize = 0,
    chunk_size: usize,
    shard_progress: ?[]ShardProgress = null,
    interface: std.Io.Writer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pools: []mem.DynamicBufferPool,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        shape: Shape,
        sharding: Sharding,
        buffer: *Buffer,
    ) !DirectMemoryWriter {
        const ordered_devices = sharding.devicesInCanonicalOrder();
        var shard_writers = try allocator.alloc(DirectShardWriter, ordered_devices.len);
        errdefer allocator.free(shard_writers);

        var initialized: usize = 0;
        errdefer for (shard_writers[0..initialized]) |*writer| {
            writer.deinit();
        };

        var pjrt_buffers: Buffer.Shards = .empty;
        const placement = try sharding.placement(shape);
        for (ordered_devices, 0..) |device, i| {
            defer initialized += 1;

            const pool = &pools[device.id];
            const shard_dma_allocator = dma_allocators[device.id].allocator();
            const pjrt_mem = platform.devices[device.id].memory(.default).?;

            shard_writers[i] = try .init(shard_dma_allocator, io, pjrt_mem, pool, placement.shape);

            pjrt_buffers.appendAssumeCapacity(shard_writers[i].pjrt_buffer);
        }

        buffer.* = .fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());

        const dispatch_spans: DispatchSpans = try .init(allocator, shape, sharding);
        errdefer dispatch_spans.deinit(allocator);

        const first_span = dispatch_spans.spans[0];
        const first_writer = &shard_writers[first_span.primary_writer];
        const first_window = @min(first_span.end - first_span.start, @min(dma_chunk_size, first_writer.interface.buffer.len));

        return .{
            .allocator = allocator,
            .shard_writers = shard_writers,
            .dispatch_spans = dispatch_spans,
            .chunk_size = dma_chunk_size,
            .interface = .{
                .buffer = first_writer.interface.buffer[0..first_window],
                .end = 0,
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
        self.dispatch_spans.deinit(self.allocator);
    }

    pub fn setProgress(self: *DirectMemoryWriter, progress: ?*std.Progress.Node) void {
        const parent = progress orelse {
            const states = self.shard_progress orelse return;
            for (states) |*s| {
                s.node.end();
            }
            self.allocator.free(states);
            self.shard_progress = null;
            return;
        };

        std.debug.assert(self.shard_progress == null);
        const states = self.allocator.alloc(ShardProgress, self.shard_writers.len) catch return;
        for (states, self.shard_writers, 0..) |*state, writer, i| {
            state.completed = 0;
            const label = std.fmt.bufPrint(&state.label, "shard[{d}]", .{i}) catch unreachable;
            const total_items = std.math.divCeil(usize, writer.total, ShardProgress.scale) catch unreachable;
            state.node = parent.start(label, total_items);
            state.node.setCompletedItems(0);
        }
        self.shard_progress = states;
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, _: usize) std.Io.Writer.Error!usize {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        const chunk = data[0];
        try self.commitWindow();

        const writable = self.interface.buffer.len - self.interface.end;
        if (writable == 0) return std.Io.Writer.Error.WriteFailed;

        const n = @min(writable, chunk.len);
        @memcpy(self.interface.buffer[self.interface.end..][0..n], chunk[0..n]);
        self.interface.end += n;
        try self.commitWindow();

        return n;
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        try self.commitWindow();
        if (self.span_index < self.dispatch_spans.spans.len) return std.Io.Writer.Error.WriteFailed;

        for (self.shard_writers, 0..) |*shard_writer, i| {
            try shard_writer.interface.flush();
            if (shard_writer.offset != shard_writer.total) return std.Io.Writer.Error.WriteFailed;
            if (self.shard_progress) |states| {
                states[i].set(shard_writer.total);
            }
        }

        self.interface = .failing;
    }

    fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        if (self.interface.buffer.len - self.interface.end >= capacity) return;
        if (preserve != 0) return std.Io.Writer.Error.WriteFailed;

        try self.commitWindow();
        if (self.interface.buffer.len - self.interface.end < capacity) return std.Io.Writer.Error.WriteFailed;
    }

    fn commitWindow(self: *DirectMemoryWriter) std.Io.Writer.Error!void {
        if (self.span_index >= self.dispatch_spans.spans.len) {
            self.interface.buffer = &.{};
            self.interface.end = 0;
            return;
        }

        const span = self.dispatch_spans.spans[self.span_index];
        const shard_writer = &self.shard_writers[span.primary_writer];

        if (self.interface.end < shard_writer.interface.end) return std.Io.Writer.Error.WriteFailed;

        if (self.interface.end > shard_writer.interface.end) {
            const chunk = self.interface.buffer[shard_writer.interface.end..self.interface.end];

            shard_writer.interface.end = self.interface.end;
            self.byte_cursor += chunk.len;

            if (span.mirror_writer_len != 0) {
                const mirror_writer_end = span.mirror_writer_start + span.mirror_writer_len;
                for (self.dispatch_spans.mirror_writers[span.mirror_writer_start..mirror_writer_end]) |mirror_writer_index| {
                    const mirror_writer = &self.shard_writers[mirror_writer_index];
                    try mirror_writer.interface.writeAll(chunk);
                    if (self.shard_progress) |states| {
                        states[mirror_writer_index].set(states[mirror_writer_index].completed + chunk.len);
                    }
                }
            }

            if (self.shard_progress) |states| {
                states[span.primary_writer].set(states[span.primary_writer].completed + chunk.len);
            }

            const span_complete = self.byte_cursor == span.end;
            const chunk_complete = @mod(self.byte_cursor, self.chunk_size) == 0;
            const shard_full = shard_writer.interface.end == shard_writer.interface.buffer.len;

            if (shard_full) {
                try shard_writer.interface.flush();
            }

            if (span_complete) {
                self.span_index += 1;
            }

            if (chunk_complete) {
                for (self.shard_writers) |*writer| {
                    try writer.interface.flush();
                }
            }
        }

        if (self.span_index >= self.dispatch_spans.spans.len) {
            self.interface.buffer = &.{};
            self.interface.end = 0;
            return;
        }

        const next_span = self.dispatch_spans.spans[self.span_index];
        const next_writer = &self.shard_writers[next_span.primary_writer];

        const span_remaining = next_span.end - self.byte_cursor;
        const buffer_remaining = next_writer.interface.buffer.len - next_writer.interface.end;
        const chunk_remaining = self.chunk_size - @mod(self.byte_cursor, self.chunk_size);
        const visible_remaining = @min(span_remaining, @min(chunk_remaining, buffer_remaining));
        const visible_len = next_writer.interface.end + visible_remaining;

        self.interface.buffer = next_writer.interface.buffer[0..visible_len];
        self.interface.end = next_writer.interface.end;
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
    shardings: []const Sharding = &.{},
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
    var span = tracer.span("zml.io.load", .{
        .tensor_count = meta.count(Tensor, model),
    });
    defer span.end();

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

    const Ctx = struct {
        allocator: std.mem.Allocator,
        dma_allocators: []const mem.DmaAllocator,
        dma_chunk_size: usize,
        pinned_buffer_pools: []mem.DynamicBufferPool,
        io: std.Io,
        platform: *const Platform,
        buffers: []*Buffer,
        shardings: []const Sharding,
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
        .dma_chunk_size = opts.dma_chunk_size,
        .pinned_buffer_pools = buffer_pools,
        .io = io,
        .shardings = opts.shardings,
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
                    const sharding = Sharding.pickSharding(ctx_.shardings, shape, .explicit_axis_binding) orelse blk: {
                        log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ reader.tensor.name, shape });
                        break :blk ctx_.platform.replicated_sharding;
                    };

                    var writer = MemoryWriter.init(
                        ctx_.allocator,
                        ctx_.io,
                        ctx_.platform,
                        ctx_.pinned_buffer_pools,
                        ctx_.dma_allocators,
                        ctx_.dma_chunk_size,
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
) !Sharding.PhysicalMesh {
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

fn buildMesh2x2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
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

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
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
        logical_mesh: Sharding.LogicalMesh,
        strategy: Sharding.Strategy,
        write_mode: WriteMode = .stream_remaining,
        writable_slice_min_len: usize = 128,
        pool_chunks: usize = 4,
        pool_chunk_size: usize = 1 << 20,
    };

    allocator: std.mem.Allocator,
    io: std.Io,

    fn run(self: DirectMemoryWriterDeviceTest, scenario: Scenario) !void {
        var platform = Platform.auto(self.allocator, self.io, scenario.create_options) catch return error.SkipZigTest;
        defer platform.deinit(self.allocator, self.io);

        const sharding: Sharding.Data = try .init(scenario.name, &platform.physical_mesh, scenario.logical_mesh, scenario.strategy);
        try self.runDirectMemoryWriter(
            platform,
            scenario.shape,
            .{ .data = &sharding },
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
            pool_chunk_size,
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
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
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
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
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
        .logical_mesh = .mesh(.{
            .batch = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y }),
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
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_y });
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
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
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
        .logical_mesh = .mesh(.{
            .batch = .low_bandwidth,
            .model = .high_bandwidth,
        }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_z });
            break :blk strategy;
        },
    });
}
