const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const meta = @import("meta.zig");
const pjrt = @import("pjrtx.zig");
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
    };
};

fn collectShapes(allocator: std.mem.Allocator, v: anytype) ![]Shape {
    const LocalContext = struct {
        list: *std.array_list.Managed(Shape),
    };
    var list = std.array_list.Managed(Shape).init(allocator);
    errdefer list.deinit();

    var context: LocalContext = .{ .list = &list };
    try meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.list.append(tensor.shape());
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

fn collectTensorDesc(allocator: std.mem.Allocator, store: TensorStore.View, v: anytype) ![]safetensors.Tensor {
    const LocalContext = struct {
        list: *std.array_list.Managed(safetensors.Tensor),
        store: TensorStore.View,
    };
    var list = std.array_list.Managed(safetensors.Tensor).init(allocator);
    var context: LocalContext = .{ .list = &list, .store = store };
    meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            const tensor_desc = ctx_.store.store.getPtrFromId(tensor.id).?.*;
            ctx_.list.append(tensor_desc) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

pub fn loadBuffersFromId(allocator: std.mem.Allocator, io: std.Io, model: anytype, store: TensorStore.View, platform: Platform) !Bufferized(@TypeOf(model)) {
    const Model = @TypeOf(model);
    var result: Bufferized(Model) = undefined;
    initBufferizedFrom(model, &result);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const shapes = try collectShapes(arena.allocator(), &model);

    var transfer: Transfer = try .init(arena.allocator(), shapes, platform);
    defer transfer.deinit(platform);

    const tensor_descs = try collectTensorDesc(arena.allocator(), store, &model);

    // TODO(Corentin): Find a way to inject that
    const buffer_reader = try allocator.alloc(u8, 16 * 1024 * 1024);
    defer allocator.free(buffer_reader);
    const buffer_writer = try allocator.alloc(u8, 16 * 1024 * 1024);
    defer allocator.free(buffer_writer);

    const LocalContext = struct {
        tensor_descs: []safetensors.Tensor,
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
        buffer_reader: []u8,
        buffer_writer: []u8,
        store: TensorStore.View,
        allocator: std.mem.Allocator,
        io: std.Io,
    };
    var context: LocalContext = .{
        .tensor_descs = tensor_descs,
        .shapes = shapes,
        .platform = platform,
        .transfer = &transfer,
        .buffer_reader = buffer_reader,
        .buffer_writer = buffer_writer,
        .store = store,
        .allocator = allocator,
        .io = io,
    };
    try meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *Buffer) !void {
            const tensor_desc = context_.tensor_descs[context_.index];

            var reader = try safetensors.TensorReader.init(context_.io, tensor_desc, context_.buffer_reader);
            defer reader.deinit();

            var writer = try context_.transfer.getWriter(context_.io, context_.index, context_.buffer_writer);

            _ = try reader.interface.streamRemaining(&writer.interface);
            try writer.interface.flush();

            buffer.* = try context_.transfer.getBuffer(context_.index);
            context_.index += 1;
        }
    }.cb, &context, &result);

    return result;
}

pub fn initBufferizedFrom(model: anytype, bufferized_: *Bufferized(@TypeOf(model))) void {
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
        .optional => {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = undefined;
                initBufferizedFrom(model.?, &bufferized_.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

pub const SimpleWriter = struct {
    offset: usize = 0,
    io: std.Io,
    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    shape: Shape,
    buffer_index: usize,
    platform: Platform,
    interface: std.Io.Writer,

    pub fn init(buffer: []u8, io: std.Io, transfer_manager: *pjrt.AsyncHostToDeviceTransferManager, shape: Shape, buffer_index: usize, platform: Platform) SimpleWriter {
        return .{
            .io = io,
            .transfer_manager = transfer_manager,
            .shape = shape,
            .buffer_index = buffer_index,
            .platform = platform,
            .interface = .{
                .buffer = buffer,
                .end = 0,
                .vtable = &.{
                    .drain = drain,
                },
            },
        };
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        _ = data;
        _ = splat;
        const writer: *SimpleWriter = @alignCast(@fieldParentPtr("interface", w));
        stdx.debug.assert(writer.offset + w.end <= writer.shape.byteSize(), "Can't write more data than required", .{});
        const is_last_transfer = writer.offset + w.end >= writer.shape.byteSize();
        log.debug("Writing {} bytes", .{w.end});
        const event = writer.transfer_manager.transferData(writer.platform.pjrt_api, writer.buffer_index, w.buffer[0..w.end], @intCast(writer.offset), is_last_transfer) catch return error.WriteFailed;
        event.await(writer.platform.pjrt_api, writer.io) catch return error.WriteFailed;
        const written = w.end;
        writer.offset += written;
        w.end = 0;
        return 0;
    }
};

pub const Transfer = struct {
    shapes: []Shape,
    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    arena: std.heap.ArenaAllocator,
    platform: Platform,

    pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, platform: Platform) !Transfer {
        const shape_specs = try allocator.alloc(pjrt.ShapeSpec, shapes.len);
        defer allocator.free(shape_specs);

        var temp_arena = std.heap.ArenaAllocator.init(allocator);
        defer temp_arena.deinit();

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        for (shape_specs, shapes) |*spec, shape| {
            const dims = try temp_arena.allocator().dupe(i64, shape.dims());
            spec.* = pjrt.ShapeSpec.init(dims, pjrt.bufferTypeFromDtype(shape.dtype()));
        }

        const memory = platform.pjrt_client.memoryByKind(platform.pjrt_api, .device).?;

        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{ .shape_specs = shape_specs, .memory = memory });
        errdefer transfer_manager.deinit(platform.pjrt_api);

        return .{
            .shapes = try arena.allocator().dupe(Shape, shapes),
            .transfer_manager = transfer_manager,
            .arena = arena,
            .platform = platform,
        };
    }

    pub fn deinit(self: Transfer, platform: Platform) void {
        self.arena.deinit();
        self.transfer_manager.deinit(platform.pjrt_api);
    }

    pub fn getBuffer(self: *const Transfer, index: usize) !Buffer {
        const pjrt_buffer = self.transfer_manager.retrieveBuffer(self.platform.pjrt_api, index) catch return error.NotFound;
        return .fromPjrtBuffers(self.platform, self.shapes[index], &.{pjrt_buffer});
    }

    pub fn getWriter(self: *const Transfer, io: std.Io, index: usize, buffer: []u8) !SimpleWriter {
        const writer: SimpleWriter = .init(buffer, io, self.transfer_manager, self.shapes[index], index, self.platform);
        return writer;
    }
};
