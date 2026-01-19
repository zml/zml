const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;
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

        return safetensors.TensorReader.init(io, tensor_desc.*, buffer);
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

        pub fn maybeCreateTensor(self: View, subkey: []const u8) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8) Tensor {
            return self.maybeCreateTensor(subkey).?;
        }

        pub fn maybeCreateTensorWithTags(self: View, subkey: []const u8, tagz: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            ptr.shape = ptr.shape.withTags(tagz);

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensorWithTags(self: View, subkey: []const u8, tagz: anytype) Tensor {
            return self.maybeCreateTensorWithTags(subkey, tagz).?;
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

pub fn TensorBufferTransfer(comptime UserData: type) type {
    return struct {
        io: std.Io,
        platform: Platform,
        tensor: safetensors.Tensor,
        buffer: *Buffer,
        cb_ctx: UserData,
    };
}

pub fn CallbackTensorBufferTransfer(comptime UserData: type) type {
    return fn (t: TensorBufferTransfer(UserData)) std.Io.Cancelable!void;
}

pub fn BufferizeContext(comptime UserData: type) type {
    return struct {
        allocator: std.mem.Allocator,
        arena: *std.heap.ArenaAllocator,
        io: std.Io,
        platform: Platform,
        cb_ctx: UserData,
    };
}

pub fn bufferize(
    comptime UserData: type,
    bufferize_ctx: BufferizeContext(UserData),
    model: anytype,
    bufferized: *Bufferized(@TypeOf(model.*)),
    store: TensorStore.View,
) ![]TensorBufferTransfer(UserData) {
    comptime {
        switch (@typeInfo(@TypeOf(model))) {
            .pointer => {},
            else => @compileError("zml.io.bufferize expects a pointer to the model (pass &model or &self.field)"),
        }
    }

    var tensor_count: usize = 0;
    meta.visit(struct {
        fn cb(ctx: *usize, _: *const Tensor) void {
            ctx.* += 1;
        }
    }.cb, &tensor_count, model);

    initBufferizedFrom(model.*, bufferized);

    const arena_allocator = bufferize_ctx.arena.allocator();

    const transfers = try arena_allocator.alloc(TensorBufferTransfer(UserData), tensor_count);
    const buffers = try bufferize_ctx.allocator.alloc(*Buffer, tensor_count);
    defer bufferize_ctx.allocator.free(buffers);

    const VisitBuffersCtx = struct {
        index: usize = 0,
        buffers: []*Buffer,
    };

    var visit_buffers_ctx: VisitBuffersCtx = .{ .buffers = buffers };
    meta.visit(struct {
        fn cb(ctx: *VisitBuffersCtx, buffer: *Buffer) void {
            ctx.buffers[ctx.index] = buffer;
            ctx.index += 1;
        }
    }.cb, &visit_buffers_ctx, bufferized);

    const FillBuffersCtx = struct {
        bufferize_ctx: BufferizeContext(UserData),
        store: TensorStore.View,
        buffers: []*Buffer,
        cb_ctx: UserData,
        transfers: []TensorBufferTransfer(UserData),
        idx: usize = 0,
    };
    var fill_buffers_ctx: FillBuffersCtx = .{
        .bufferize_ctx = bufferize_ctx,
        .store = store,
        .buffers = buffers,
        .cb_ctx = bufferize_ctx.cb_ctx,
        .transfers = transfers,
    };

    meta.visit(struct {
        fn cb(ctx: *FillBuffersCtx, tensor: *const Tensor) void {
            const ptr = ctx.store.store.getPtrFromId(tensor.id) orelse return;
            const buffer = ctx.buffers[ctx.idx];

            ctx.transfers[ctx.idx] = .{
                .io = ctx.bufferize_ctx.io,
                .platform = ctx.bufferize_ctx.platform,
                .tensor = ptr.*,
                .buffer = buffer,
                .cb_ctx = ctx.cb_ctx,
            };
            ctx.idx += 1;
        }
    }.cb, &fill_buffers_ctx, model);

    return transfers;
}

pub const Writer = union(enum) { device: DeviceWriter, buffered_device: BufferedDeviceWriter };

pub fn loadBuffersFromId(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: Platform,
    model: anytype,
    store: TensorStore.View,
    read_pool_config: ConcurrentBufferPool.Config,
    write_pool_config: ConcurrentBufferPool.Config,
) !Bufferized(@TypeOf(model)) {
    var bufferized: Bufferized(@TypeOf(model)) = undefined;
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    var read_pool, var write_pool = try ConcurrentBufferPool.initRW(
        allocator,
        platform,
        read_pool_config,
        write_pool_config,
    );
    defer {
        read_pool.deinit();
        write_pool.deinit();
    }

    const LoadCtx = struct { allocator: std.mem.Allocator, read_pool: *ConcurrentBufferPool, write_pool: *ConcurrentBufferPool };

    const bufferize_ctx: BufferizeContext(LoadCtx) = .{
        .allocator = allocator,
        .arena = &arena,
        .io = io,
        .platform = platform,
        .cb_ctx = .{ .allocator = allocator, .read_pool = &read_pool, .write_pool = &write_pool },
    };

    const transfers = try bufferize(LoadCtx, bufferize_ctx, &model, &bufferized, store);

    var group = try stdx.Io.AllocatingLimitedConcurrentGroup.init(allocator, read_pool.config.concurrency);
    defer {
        group.cancel(io);
        group.deinit();
    }

    for (transfers) |t| {
        try group.concurrent(io, struct {
            fn run(ctx: TensorBufferTransfer(LoadCtx)) !void {
                const allocator_ = ctx.cb_ctx.allocator;
                const read_pool_ = ctx.cb_ctx.read_pool;
                const write_pool_ = ctx.cb_ctx.write_pool;

                const read_buffer = read_pool_.acquire(ctx.io) catch unreachable;
                defer read_pool_.release(ctx.io, read_buffer) catch {};

                const write_buffer = write_pool_.acquire(ctx.io) catch unreachable;
                defer write_pool_.release(ctx.io, write_buffer) catch {};

                var reader = safetensors.TensorReader.init(ctx.io, ctx.tensor, read_buffer) catch unreachable;
                defer reader.deinit();

                var writer: Writer = if (ctx.platform.target == .neuron) blk: {
                    break :blk .{ .buffered_device = BufferedDeviceWriter.init(allocator_, ctx.io, null, ctx.platform, ctx.buffer.shape(), ctx.buffer) catch unreachable };
                } else blk: {
                    break :blk .{ .device = DeviceWriter.init(ctx.io, null, ctx.platform, ctx.buffer, .device, write_buffer) catch unreachable };
                };

                switch (writer) {
                    .device => |*d| {
                        defer {
                            d.interface.flush() catch unreachable;
                            d.deinit();
                        }
                        _ = reader.interface.streamRemaining(&d.interface) catch unreachable;
                    },
                    .buffered_device => |*d| {
                        defer {
                            d.interface.flush() catch unreachable;
                            d.deinit();
                        }
                        _ = reader.interface.streamRemaining(&d.interface) catch unreachable;
                    },
                }
            }
        }.run, .{t});
    }

    try group.await(io);

    return bufferized;
}

pub const DeviceWriter = struct {
    io: std.Io,
    progress: ?*std.Progress.Node,
    platform: Platform,
    shape: Shape,
    buffer: *Buffer,
    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    offset: usize = 0,
    interface: std.Io.Writer,

    pub fn init(io: std.Io, progress: ?*std.Progress.Node, platform: Platform, shape: Shape, buffer: *Buffer, memory: Buffer.Memory, buf: []u8) !DeviceWriter {
        const memories = platform.getDevices()[0].addressableMemories(platform.pjrt_api);

        const mem = blk: {
            for (memories) |mem_| {
                if (mem_.kind(platform.pjrt_api) == memory) {
                    break :blk mem_;
                }
            }
            return error.MemoryNotFound;
        };

        const shape_spec: pjrt.ShapeSpec = .init(shape.dims(), pjrtx.bufferTypeFromDtype(shape.dtype()));
        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
            .shape_specs = &.{shape_spec},
            .memory = mem,
        });

        const pjrt_buffer = transfer_manager.retrieveBuffer(platform.pjrt_api, 0) catch unreachable;
        buffer.* = Buffer.fromPjrtBuffers(platform, shape, &.{pjrt_buffer});

        return .{
            .io = io,
            .progress = progress,
            .platform = platform,
            .shape = shape,
            .buffer = buffer,
            .transfer_manager = transfer_manager,
            .interface = .{
                .buffer = buf,
                .end = 0,
                .vtable = &.{
                    .drain = drain,
                },
            },
        };
    }

    pub fn deinit(self: *const DeviceWriter) void {
        self.transfer_manager.deinit(self.platform.pjrt_api);
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        _ = data;
        _ = splat;
        const self: *DeviceWriter = @alignCast(@fieldParentPtr("interface", w));

        stdx.debug.assert(self.offset + w.end <= self.shape.byteSize(), "Can't write more data than required", .{});

        const is_last_transfer = self.offset + w.end >= self.shape.byteSize();
        const event = self.transfer_manager.transferData(self.platform.pjrt_api, 0, w.buffer[0..w.end], @intCast(self.offset), is_last_transfer) catch return error.WriteFailed;
        event.await(self.platform.pjrt_api, self.io) catch return error.WriteFailed;

        self.offset += w.end;
        if (self.progress) |p| p.setCompletedItems(self.offset);
        w.end = 0;

        return 0;
    }
};

pub const BufferedDeviceWriter = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    progress: ?*std.Progress.Node,
    transfer_buffer: []u8,
    platform: Platform,
    shape: Shape,
    buffer: *Buffer,
    interface: std.Io.Writer,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, progress: ?*std.Progress.Node, platform: Platform, shape: Shape, buffer: *Buffer) !BufferedDeviceWriter {
        const transfer_buffer = try allocator.alloc(u8, shape.byteSize());
        return .{
            .allocator = allocator,
            .io = io,
            .progress = progress,
            .transfer_buffer = transfer_buffer,
            .shape = shape,
            .platform = platform,
            .buffer = buffer,
            .interface = .{
                .buffer = transfer_buffer,
                .vtable = &.{
                    .drain = std.Io.Writer.fixedDrain,
                    .flush = flush,
                    .rebase = std.Io.Writer.failingRebase,
                },
            },
        };
    }

    pub fn deinit(self: *BufferedDeviceWriter) void {
        self.allocator.free(self.transfer_buffer);
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *BufferedDeviceWriter = @alignCast(@fieldParentPtr("interface", w));
        self.buffer.* = Buffer.fromBytes(self.io, self.platform, self.shape, self.transfer_buffer) catch return std.Io.Writer.Error.WriteFailed;
        if (self.progress) |p| p.setCompletedItems(self.shape.byteSize());
        _ = self.buffer.await(self.io) catch return std.Io.Writer.Error.WriteFailed;
    }
};

fn initBufferizedFrom(model: anytype, bufferized_: *Bufferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(Bufferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (Bufferized(Model) == Buffer) return;
            inline for (struct_type_info.fields) |field| {
                initBufferizedFrom(@field(model, field.name), &@field(bufferized_, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(Bufferized(Model), @tagName(tag), undefined);
                    initBufferizedFrom(v, @field(bufferized_, @tagName(tag)));
                },
            }
        },
        .optional => |optional_type_info| {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = @as(optional_type_info.child, undefined);
                initBufferizedFrom(model.?, &bufferized_.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

pub const ConcurrentBufferPool = struct {
    pub const Error = error{ OutOfMemory, Notfound, NotAcquired };

    const alignment: std.mem.Alignment = .fromByteUnits(4 * 1024);

    allocator: std.mem.Allocator,
    mutex: std.Io.Mutex = .init,

    platform: Platform,
    locked: []bool,
    buffers: []u8,
    config: Config,

    pub const Config = struct {
        size: usize,
        concurrency: usize,
        dma: bool = true,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        platform: Platform,
        config: Config,
    ) !ConcurrentBufferPool {
        const buffers = try allocator.alignedAlloc(u8, alignment, config.concurrency * config.size);
        errdefer allocator.free(buffers);

        if (config.dma and platform.target != .cpu) {
            try platform.pjrt_client.dmaMap(platform.pjrt_api, buffers);
        }

        const locked = try allocator.alloc(bool, config.concurrency);
        @memset(locked, false);

        return .{
            .allocator = allocator,
            .mutex = .init,
            .platform = platform,
            .locked = locked,
            .buffers = buffers,
            .config = config,
        };
    }

    pub fn initRW(allocator: std.mem.Allocator, platform: Platform, read_config: Config, write_config: Config) !struct { ConcurrentBufferPool, ConcurrentBufferPool } {
        return .{ try .init(allocator, platform, read_config), try .init(allocator, platform, write_config) };
    }

    pub fn deinit(self: *ConcurrentBufferPool) void {
        if (self.config.dma and self.platform.target != .cpu) {
            self.platform.pjrt_client.dmaUnmap(self.platform.pjrt_api, self.buffers) catch {};
        }

        self.allocator.free(self.buffers);
        self.allocator.free(self.locked);
    }

    pub fn acquire(self: *ConcurrentBufferPool, io: std.Io) Error![]u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        for (self.locked, 0..) |locked, i| {
            if (!locked) {
                self.locked[i] = true;
                return self.buffers[i * self.config.size .. (i + 1) * self.config.size];
            }
        }

        return Error.OutOfMemory;
    }

    pub fn release(self: *ConcurrentBufferPool, io: std.Io, buffer: []u8) Error!void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);

        for (self.locked, 0..) |locked, i| {
            const slice = self.buffers[i * self.config.size .. (i + 1) * self.config.size];
            if (slice.ptr == buffer.ptr) {
                if (locked) {
                    self.locked[i] = false;
                    return;
                } else {
                    return Error.NotAcquired;
                }
            }
        }

        return Error.Notfound;
    }
};
