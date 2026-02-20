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
        self.total += (len_pre - self.interface.end) + total;
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

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, _: *mem.DynamicBufferPool, shape: Shape, buffer: *Buffer) !MemoryWriter {
        return switch (platform.target) {
            // todo: reenable DirectMemoryWriter with proper initialization
            // .cuda => .{ .direct = try .init(allocator, io, platform, pool, shape, buffer) },
            .cuda, .rocm, .tpu, .neuron, .cpu => .{ .buffered = try .init(allocator, io, platform, shape, buffer) },
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

pub const DirectMemoryWriter = struct {
    const EventContext = struct {
        self: *DirectMemoryWriter,
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
    buffer: *Buffer,

    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    offset: usize = 0,

    interface: std.Io.Writer,
    flip_flop: u1 = 0,
    events_contexts: [2]?EventContext = @splat(null),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, memory: *const Memory, pool: *mem.DynamicBufferPool, shape: Shape, buffer: *Buffer) !DirectMemoryWriter {
        const shape_spec: pjrt.ShapeSpec = .init(shape.dims(), pjrtx.bufferTypeFromDtype(shape.dtype()));
        const transfer_manager = try memory.platform.pjrt_client.createBuffersForAsyncHostToDevice(
            memory.platform.pjrt_api,
            .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            },
        );

        const pjrt_buffer = transfer_manager.retrieveBuffer(memory.platform.pjrt_api, 0) catch unreachable;
        buffer.* = Buffer.fromPjrtBuffers(memory.platform, shape, &.{pjrt_buffer});

        return .{
            .allocator = allocator,
            .io = io,
            .memory = memory,
            .pool = pool,
            .total = shape.byteSize(),
            .buffer = buffer,
            .transfer_manager = transfer_manager,
            .interface = .{
                .buffer = try pool.get(allocator, io),
                .end = 0,
                .vtable = &.{
                    .drain = drain,
                },
            },
        };
    }

    pub fn deinit(self: *DirectMemoryWriter) void {
        self.transfer_manager.deinit(self.memory.platform.pjrt_api);
    }

    fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        _ = data; // autofix
        _ = splat; // autofix
        const self: *DirectMemoryWriter = @alignCast(@fieldParentPtr("interface", w));
        if (self.offset >= self.total) {
            return std.Io.Writer.Error.WriteFailed;
        }

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

        return 0;
    }
};

// pub const LoadOpts = struct {
//     parallelism: usize,
//     store: *const TensorStore,
//     progress: ?*std.Progress.Node = null,
//     dma_chunks: usize,
//     dma_chunk_size: usize,
//     total_bytes: ?*usize = null,
// };

// pub fn load(
//     comptime ModelType: type,
//     model: *const ModelType,
//     allocator: std.mem.Allocator,
//     io: std.Io,
//     platform: *const Platform,
//     opts: LoadOpts,
// ) !Bufferized(ModelType) {
//     var bufferized = try mem.bufferize(allocator, ModelType, model);

//     const first_device = platform.devices[0]; // Temporary until sharding is re-exposed
//     const dma_alloc: mem.DmaAllocator = .init(allocator, &first_device);
//     var buffer_pool: mem.DynamicBufferPool = .init(opts.dma_chunks, opts.dma_chunk_size);
//     defer buffer_pool.deinit(dma_alloc.allocator());

//     const Ctx = struct {
//         allocator: std.mem.Allocator,
//         dma_allocator: std.mem.Allocator,
//         pinned_buffer_pool: *mem.DynamicBufferPool,
//         io: std.Io,
//         buffers: []*Buffer,
//         store: *const TensorStore,
//         memory: *const Memory,
//         group: stdx.Io.LimitedGroup,
//         total: std.atomic.Value(usize) = .init(0),
//         progress: ?*std.Progress.Node,
//     };
//     var walk_ctx: Ctx = .{
//         .buffers = try allocator.alloc(*Buffer, meta.count(Tensor, model)),
//         .store = opts.store,
//         .allocator = allocator,
//         .dma_allocator = dma_alloc.allocator(),
//         .pinned_buffer_pool = &buffer_pool,
//         .io = io,
//         .memory = first_device.memory(.default),
//         .progress = opts.progress,
//         .group = .init(opts.parallelism),
//     };
//     defer allocator.free(walk_ctx.buffers);

//     defer if (opts.total_bytes) |total_bytes_ptr| {
//         total_bytes_ptr.* = walk_ctx.total.load(.monotonic);
//     };

//     meta.forEachVisit(&bufferized, *Buffer, struct {
//         fn call(i: usize, buffer: *Buffer, ctx: *Ctx) void {
//             ctx.buffers[i] = buffer;
//         }
//     }.call, .{&walk_ctx});

//     meta.forEachVisit(model, *const Tensor, struct {
//         fn call(i: usize, tensor: *const Tensor, ctx: *Ctx) void {
//             ctx.group.concurrent(ctx.io, struct {
//                 fn call(i_: usize, tensor_: *const Tensor, ctx_: *Ctx) !void {
//                     var reader = ctx_.store.getReaderById(tensor_.id, ctx_.io, &.{}) catch unreachable;
//                     defer reader.deinit();

//                     var memory_writer = MemoryWriter.init(
//                         ctx_.dma_allocator,
//                         ctx_.io,
//                         ctx_.memory,
//                         ctx_.pinned_buffer_pool,
//                         reader.tensor.shape,
//                         ctx_.buffers[i_],
//                     ) catch unreachable;
//                     defer memory_writer.deinit(ctx_.dma_allocator);

//                     const scale = 1024;

//                     if (ctx_.progress) |progress| {
//                         var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
//                         defer node.end();
//                         var progress_writer: ProgressWriter = .init(memory_writer.interface(), &node, .{ .scale = scale });
//                         const total = reader.interface.streamRemaining(&progress_writer.interface) catch unreachable;
//                         progress_writer.interface.flush() catch unreachable;
//                         _ = ctx_.total.fetchAdd(total, .monotonic);
//                     } else {
//                         const total = reader.interface.streamRemaining(memory_writer.interface()) catch unreachable;
//                         memory_writer.interface().flush() catch unreachable;
//                         _ = ctx_.total.fetchAdd(total, .monotonic);
//                     }
//                 }
//             }.call, .{ i, tensor, ctx }) catch unreachable;
//         }
//     }.call, .{&walk_ctx});
//     walk_ctx.group.await(io) catch unreachable;

//     return bufferized;
// }

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
    var buffer_pool: mem.DynamicBufferPool = .init(opts.dma_chunks, opts.dma_chunk_size);
    defer buffer_pool.deinit(dma_alloc.allocator());

    const replicated_sharding = try sharding_.replicatedSharding(opts.shardings[0].physical);

    const Ctx = struct {
        allocator: std.mem.Allocator,
        dma_allocator: std.mem.Allocator,
        pinned_buffer_pool: *mem.DynamicBufferPool,
        io: std.Io,
        platform: *const Platform,
        buffers: []*Buffer,
        shardings: []const Sharding,
        replicated_sharding: Sharding,
        store: *const TensorStore,
        memory: *const Memory,
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
        .pinned_buffer_pool = &buffer_pool,
        .io = io,
        .shardings = opts.shardings,
        .replicated_sharding = replicated_sharding,
        .memory = first_device.memory(.default),
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

                    var writer = BufferedMemoryWriter.init(
                        ctx_.dma_allocator,
                        ctx_.io,
                        ctx_.platform,
                        shape,
                        sharding,
                        ctx_.buffers[i_],
                    ) catch unreachable;
                    defer writer.deinit(ctx_.dma_allocator);

                    const scale = 1024;

                    if (ctx_.progress) |progress| {
                        var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
                        defer node.end();
                        var progress_writer: ProgressWriter = .init(&writer.interface, &node, .{ .scale = scale });
                        const total = reader.interface.streamRemaining(&progress_writer.interface) catch unreachable;
                        progress_writer.interface.flush() catch unreachable;
                        _ = ctx_.total.fetchAdd(total, .monotonic);
                    } else {
                        const total = reader.interface.streamRemaining(&writer.interface) catch unreachable;
                        writer.interface.flush() catch unreachable;
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
