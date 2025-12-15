const std = @import("std");
const stdx = @import("stdx");

const zml = @import("zml");

const log = std.log.scoped(.mnist);

pub const std_options: std.Options = .{
    .log_level = .info,
};

/// Model definition
const Mnist = struct {
    fc1: Layer,
    fc2: Layer,

    const Layer = struct {
        weight: zml.Tensor,
        bias: zml.Tensor,

        pub fn init(store: TensorStore.View) Layer {
            return .{
                .weight = store.createTensorWithTags("weight", .{ .d_out, .d }),
                .bias = store.createTensorWithTags("bias", .{.d_out}),
            };
        }

        pub fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
            return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
        }
    };

    pub fn init(store: TensorStore.View) Mnist {
        return .{
            .fc1 = .init(store.withPrefix("fc1")),
            .fc2 = .init(store.withPrefix("fc2")),
        };
    }

    pub fn loadBuffers(self: Mnist, allocator: std.mem.Allocator, io: std.Io, vfs: *zml.io.VFS, store: TensorStore.View, platform: zml.Platform) !zml.Bufferized(Mnist) {
        return loadBuffersFromId(allocator, io, vfs, self, store, platform);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mnist)) void {
        self.fc1.weight.deinit();
        self.fc1.bias.deinit();
        self.fc2.weight.deinit();
        self.fc2.bias.deinit();
    }

    /// just two linear layers + relu activation
    pub fn forward(self: Mnist, input: zml.Tensor) zml.Tensor {
        // std.log.info("Compiling for target: {s}", .{@tagName(input.getContext().target())});
        var x = input.flatten().convert(.f32).withTags(.{.d});
        const layers: []const Layer = &.{ self.fc1, self.fc2 };
        for (layers) |layer| {
            x = layer.forward(x);
        }
        return x.argMax(0).indices.convert(.u8);
    }
};

pub fn main() !void {
    const allocator = std.heap.c_allocator;
    //var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    //defer _ = gpa.deinit();

    //const allocator = gpa.allocator();

    var threaded: std.Io.Threaded = .init(allocator);
    defer threaded.deinit();

    var vfs_file: zml.io.VFS.File = .init(threaded.io());

    var vfs: zml.io.VFS = .init(allocator, threaded.io());
    defer vfs.deinit();

    try vfs.register("file", vfs_file.io());

    const io = vfs.io();

    zml.init();
    defer zml.deinit();

    // Auto-select platform
    const platform: zml.Platform = try .auto(threaded.io(), .{});

    // Parse program args
    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);
    const model_path = process_args[1];
    const t10kfilename = process_args[2];

    // Read model shapes.
    var registry = try zml.safetensors.parseFromPath(allocator, io, &vfs, model_path);
    defer registry.deinit();

    // Init model
    var store: TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const mnist_model = Mnist.init(store.view());

    // Compile model
    log.info("Compiling model to MLIR....", .{});
    var timer = try stdx.time.Timer.start();
    const input: zml.Tensor = .init(.{ 28, 28 }, .u8);
    var exe = try platform.compileModel(allocator, io, Mnist.forward, mnist_model, .{input});
    defer exe.deinit();

    log.info("✅ Compiled model in {f}", .{timer.read()});

    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    // Load buffers
    var mnist_buffers = try mnist_model.loadBuffers(allocator, io, &vfs, store.view(), platform);
    defer Mnist.unloadBuffers(&mnist_buffers);
    log.info("✅ Weights transferred in {f}", .{timer.read()});

    log.info("Starting inference...", .{});

    // Load a random digit image from the dataset.
    const dataset = try vfs.openAbsoluteFile(io, t10kfilename, .{ .mode = .read_only });
    defer dataset.close(io);

    const now = std.Io.Clock.now(.awake, io) catch unreachable;
    var rng = std.Random.Xoshiro256.init(@intCast(now.toMilliseconds()));

    // inference - can be looped
    {
        const idx = rng.random().intRangeAtMost(u64, 0, 10000 - 1);
        var sample: [28 * 28]u8 align(16) = undefined;
        var reader = dataset.reader(io, &.{});
        try reader.seekTo(16 + (idx * 28 * 28));
        _ = try reader.interface.readSliceShort(&sample);

        var input_buffer: zml.Buffer = try .fromSlice(io, platform, zml.Slice.init(input.shape(), &sample));
        defer input_buffer.deinit();

        args.set(.{ mnist_buffers, input_buffer });

        printDigit(sample);
        exe.call(args, &results, io);

        var result: zml.Buffer = results.get(zml.Buffer);
        defer result.deinit();

        log.info(
            \\✅ RECOGNIZED DIGIT:
            \\                       +-------------+
            \\{s}
            \\                       +-------------+
            \\
        , .{digits[try result.getValue(u8, io)]});
    }
}

fn printDigit(digit: [28 * 28]u8) void {
    var buffer: [28][30][2]u8 = undefined;
    for (0..28) |y| {
        buffer[y][0] = .{ '|', ' ' };
        buffer[y][29] = .{ '|', '\n' };
        for (1..29) |x| {
            const idx = (y * 28) + (x - 1);
            const val = digit[idx];
            buffer[y][x] = blk: {
                if (val > 240) break :blk .{ '*', '*' };
                if (val > 225) break :blk .{ 'o', 'o' };
                if (val > 210) break :blk .{ '.', '.' };
                break :blk .{ ' ', ' ' };
            };
        }
    }

    log.info(
        \\
        \\     R E C O G N I Z I N G   I N P U T   I M A G E :
        \\+---------------------------------------------------------+
        \\{s}+---------------------------------------------------------+
        \\
    , .{std.mem.asBytes(&buffer)});
}

const digits = [_][]const u8{
    \\                       |     ###     |
    \\                       |    #   #    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #   #    |
    \\                       |     ###     |
    ,
    \\                       |      #      |
    \\                       |     ##      |
    \\                       |    # #      |
    \\                       |      #      |
    \\                       |      #      |
    \\                       |      #      |
    \\                       |    #####    |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |         #   |
    \\                       |    #####    |
    \\                       |   #         |
    \\                       |   #         |
    \\                       |   #######   |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |         #   |
    \\                       |    #####    |
    \\                       |         #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |   #         |
    \\                       |   #    #    |
    \\                       |   #    #    |
    \\                       |   #    #    |
    \\                       |   #######   |
    \\                       |        #    |
    \\                       |        #    |
    ,
    \\                       |   #######   |
    \\                       |   #         |
    \\                       |   #         |
    \\                       |   ######    |
    \\                       |         #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #         |
    \\                       |   ######    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |   #######   |
    \\                       |   #    #    |
    \\                       |       #     |
    \\                       |      #      |
    \\                       |     #       |
    \\                       |     #       |
    \\                       |     #       |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
    \\                       |    #####    |
    \\                       |   #     #   |
    \\                       |   #     #   |
    \\                       |    ######   |
    \\                       |         #   |
    \\                       |   #     #   |
    \\                       |    #####    |
    ,
};

pub const TensorStore = struct {
    registry: *zml.safetensors.TensorRegistry,
    id_map: std.AutoHashMapUnmanaged(usize, *zml.safetensors.Tensor),
    allocator: std.mem.Allocator,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *zml.safetensors.TensorRegistry) TensorStore {
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

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*zml.safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*zml.safetensors.Tensor {
        const tensor_desc_ptr = self.id_map.get(id) orelse return null;
        return tensor_desc_ptr;
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, vfs: *zml.io.VFS, buffer: []u8) !zml.safetensors.TensorReader {
        return self.registry.reader(io, vfs, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, vfs: *zml.io.VFS, buffer: []u8) !zml.safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;

        return zml.safetensors.TensorReader.init(io, vfs, tensor_desc.*, buffer);
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

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8) ?zml.Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;

            const tensor: zml.Tensor = .init(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8) zml.Tensor {
            return self.maybeCreateTensor(subkey).?;
        }

        pub fn maybeCreateTensorWithTags(self: View, subkey: []const u8, tagz: anytype) ?zml.Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            ptr.shape = ptr.shape.withTags(tagz);

            const tensor: zml.Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensorWithTags(self: View, subkey: []const u8, tagz: anytype) zml.Tensor {
            return self.maybeCreateTensorWithTags(subkey, tagz).?;
        }

        pub fn getShape(self: View, subkey: []const u8) ?zml.Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?zml.Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            };
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getReader(self: View, subkey: []const u8, io: std.Io, vfs: *zml.io.VFS, buffer: []u8) !zml.safetensors.TensorReader {
            var key_buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return self.store.getReader(key, io, vfs, buffer);
        }
    };
};

fn collectShapes(allocator: std.mem.Allocator, v: anytype) ![]zml.Shape {
    const LocalContext = struct {
        list: *std.array_list.Managed(zml.Shape),
    };
    var list = std.array_list.Managed(zml.Shape).init(allocator);
    errdefer list.deinit();

    var context: LocalContext = .{ .list = &list };
    try zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const zml.Tensor) !void {
            try ctx_.list.append(tensor.shape());
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

fn collectTensorDesc(allocator: std.mem.Allocator, store: TensorStore.View, v: anytype) ![]zml.safetensors.Tensor {
    const LocalContext = struct {
        list: *std.array_list.Managed(zml.safetensors.Tensor),
        store: TensorStore.View,
    };
    var list = std.array_list.Managed(zml.safetensors.Tensor).init(allocator);
    var context: LocalContext = .{ .list = &list, .store = store };
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const zml.Tensor) void {
            const tensor_desc = ctx_.store.store.getPtrFromId(tensor.id).?.*;
            ctx_.list.append(tensor_desc) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

pub fn loadBuffersFromId(allocator: std.mem.Allocator, io: std.Io, vfs: *zml.io.VFS, model: anytype, store: TensorStore.View, platform: zml.Platform) !zml.Bufferized(@TypeOf(model)) {
    const Model = @TypeOf(model);
    var result: zml.Bufferized(Model) = undefined;
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
        tensor_descs: []zml.safetensors.Tensor,
        shapes: []const zml.Shape,
        platform: zml.Platform,
        transfer: *Transfer,
        index: usize = 0,
        buffer_reader: []u8,
        buffer_writer: []u8,
        store: TensorStore.View,
        allocator: std.mem.Allocator,
        io: std.Io,
        vfs: *zml.io.VFS,
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
        .vfs = vfs,
    };
    try zml.meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *zml.Buffer) !void {
            const tensor_desc = context_.tensor_descs[context_.index];

            var reader = try zml.safetensors.TensorReader.init(context_.io, context_.vfs, tensor_desc, context_.buffer_reader);
            defer reader.deinit();

            var writer = try context_.transfer.getWriter(context_.index, context_.buffer_writer);

            _ = try reader.interface.streamRemaining(&writer.interface);
            try writer.interface.flush();

            buffer.* = try context_.transfer.getBuffer(context_.index);
            context_.index += 1;
        }
    }.cb, &context, &result);

    return result;
}

pub fn initBufferizedFrom(model: anytype, bufferized_: *zml.Bufferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(zml.Bufferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (zml.Bufferized(Model) == zml.Buffer) return;
            inline for (struct_type_info.fields) |field| {
                initBufferizedFrom(@field(model, field.name), &@field(bufferized_, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(zml.Bufferized(Model), @tagName(tag), undefined);
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

const SimpleWriter = struct {
    offset: usize = 0,
    transfer_manager: *zml.pjrt.AsyncHostToDeviceTransferManager,
    shape: zml.Shape,
    buffer_index: usize,
    platform: zml.Platform,
    interface: std.Io.Writer,

    pub fn init(buffer: []u8, transfer_manager: *zml.pjrt.AsyncHostToDeviceTransferManager, shape: zml.Shape, buffer_index: usize, platform: zml.Platform) SimpleWriter {
        return .{
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
        _ = data; // autofix
        _ = splat; // autofix
        const writer: *SimpleWriter = @alignCast(@fieldParentPtr("interface", w));
        stdx.debug.assert(writer.offset + w.end <= writer.shape.byteSize(), "Can't write more data than required", .{});
        const is_last_transfer = writer.offset + w.end >= writer.shape.byteSize();
        log.debug("Writing {} bytes", .{w.end});
        _ = writer.transfer_manager.transferData(writer.platform.pjrt_api, writer.buffer_index, w.buffer[0..w.end], @intCast(writer.offset), is_last_transfer) catch return error.WriteFailed;
        const written = w.end;
        writer.offset += written;
        w.end = 0;
        return 0;
    }
};

const Transfer = struct {
    shapes: []zml.Shape,
    transfer_manager: *zml.pjrt.AsyncHostToDeviceTransferManager,
    arena: std.heap.ArenaAllocator,
    platform: zml.Platform,

    pub fn init(allocator: std.mem.Allocator, shapes: []const zml.Shape, platform: zml.Platform) !Transfer {
        const shape_specs = try allocator.alloc(zml.pjrt.ShapeSpec, shapes.len);
        defer allocator.free(shape_specs);

        var temp_arena = std.heap.ArenaAllocator.init(allocator);
        defer temp_arena.deinit();

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        for (shape_specs, shapes) |*spec, shape| {
            const dims = try temp_arena.allocator().dupe(i64, shape.dims());
            spec.* = zml.pjrt.ShapeSpec.init(dims, zml.pjrt.bufferTypeFromDtype(shape.dtype()));
        }

        const memory = platform.pjrt_client.memoryByKind(platform.pjrt_api, .device).?;

        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{ .shape_specs = shape_specs, .memory = memory });
        errdefer transfer_manager.deinit(platform.pjrt_api);

        return .{
            .shapes = try arena.allocator().dupe(zml.Shape, shapes),
            .transfer_manager = transfer_manager,
            .arena = arena,
            .platform = platform,
        };
    }

    pub fn deinit(self: Transfer, platform: zml.Platform) void {
        self.arena.deinit();
        self.transfer_manager.deinit(platform.pjrt_api);
    }

    pub fn getBuffer(self: *const Transfer, index: usize) !zml.Buffer {
        const pjrt_buffer = self.transfer_manager.retrieveBuffer(self.platform.pjrt_api, index) catch return error.NotFound;
        return .fromPjrtBuffers(self.platform, self.shapes[index], &.{pjrt_buffer});
    }

    pub fn getWriter(self: *const Transfer, index: usize, buffer: []u8) !SimpleWriter {
        const writer: SimpleWriter = .init(buffer, self.transfer_manager, self.shapes[index], index, self.platform);
        return writer;
    }
};
