const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const log = std.log.scoped(.vfs);

pub const std_options: std.Options = .{
    .log_level = .debug,
};

fn concatenate(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, keys: []const []const u8, axis: i64) !zml.Shape {
    var shape_list: std.ArrayList(zml.Shape) = try .initCapacity(allocator, keys.len);
    defer shape_list.deinit(allocator);

    for (keys) |key| {
        const shape = store.getShape(key) orelse return error.NotFound;
        shape_list.appendAssumeCapacity(shape);
    }

    const shapes = shape_list.items;

    var concatenated_dim: i64 = 0;
    for (shapes) |shape| {
        concatenated_dim += shape.dim(axis);
    }

    const result_shape = shapes[0].setDim(axis, concatenated_dim);
    return result_shape;
}

pub const ExecutableCache = struct {
    map: std.StringHashMapUnmanaged(zml.Exe) = .empty,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator) ExecutableCache {
        return .{ .allocator = allocator };
    }

    pub fn deinit(self: *ExecutableCache) void {
        var it = self.map.iterator();
        while (it.next()) |entry| {
            entry.value_ptr.deinit();
        }
        self.map.deinit(self.allocator);
    }
};

pub fn getTensors(allocator: std.mem.Allocator, sources: []const *zml.safetensors.Tensor) ![]zml.Tensor {
    var tensor_list: std.ArrayList(zml.Tensor) = try .initCapacity(allocator, sources.len);
    defer tensor_list.deinit(allocator);

    for (sources) |source| {
        const tensor: zml.Tensor = .fromShape(source.shape);
        tensor_list.appendAssumeCapacity(tensor);
    }

    return try tensor_list.toOwnedSlice(allocator);
}

pub const Layer = struct {
    qkv_proj: zml.nn.Linear,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) Layer {
        const weight_keys = &.{ "q_proj.weight", "v_proj.weight", "v_proj.weight" };

        var weight_shape = concatenate(allocator, store, weight_keys, 0) catch unreachable;
        weight_shape = weight_shape.withTags(.{ .dout, .d }).withPartitioning(.{ .dout = .model });

        const qkv_weight = store.maybeCreateBinding(weight_keys, weight_shape).?;

        const bias_keys = &.{ "q_proj.bias", "v_proj.bias", "v_proj.bias" };
        const qkv_bias = blk: {
            var bias_shape = concatenate(allocator, store, bias_keys, 0) catch break :blk null;
            bias_shape = bias_shape.withTags(.{.dout}).withPartitioning(.{ .dout = .model });
            break :blk store.maybeCreateBinding(bias_keys, bias_shape);
        };

        return .{ .qkv_proj = .init(qkv_weight, qkv_bias, .d) };
    }

    pub fn preload(
        self: *const Layer,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.Sharding,
        exe_cache: *ExecutableCache,
    ) !void {
        const weight_func = struct {
            fn func(inputs: []const zml.Tensor) zml.Tensor {
                return zml.Tensor.concatenate(inputs, 0).withTags(.{ .d, .dout }).withPartitioning(.{ .dout = .model });
            }
        }.func;

        const bias_func = struct {
            fn func(inputs: []const zml.Tensor) zml.Tensor {
                return zml.Tensor.concatenate(inputs, 0).withTags(.{.dout}).withPartitioning(.{ .dout = .model });
            }
        }.func;

        const weight_gop = try exe_cache.map.getOrPut(exe_cache.allocator, @tagName(.qkv_proj_weight));
        errdefer exe_cache.map.removeByPtr(weight_gop.key_ptr);

        if (!weight_gop.found_existing) {
            const sources = store.getSourcesById(self.qkv_proj.weight.id).?;
            const tensors = try getTensors(allocator, sources);
            defer allocator.free(tensors);
            var exe = try platform.compileFn(allocator, io, weight_func, .{tensors}, .{ .shardings = shardings });
            errdefer exe.deinit();

            weight_gop.value_ptr.* = exe;
        }

        if (self.qkv_proj.bias != null) {
            const bias_gop = try exe_cache.map.getOrPut(exe_cache.allocator, @tagName(.qkv_proj_bias));
            errdefer exe_cache.map.removeByPtr(bias_gop.key_ptr);

            if (!bias_gop.found_existing) {
                const sources = store.getSourcesById(self.qkv_proj.bias.?.id).?;
                const tensors = try getTensors(allocator, sources);
                defer allocator.free(tensors);
                var exe = try platform.compileFn(allocator, io, bias_func, .{tensors}, .{ .shardings = shardings });
                errdefer exe.deinit();

                bias_gop.value_ptr.* = exe;
            }
        }
    }

    pub fn load(self: *const Layer, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *const zml.io.TensorStore, shardings: []const zml.Sharding, exe_cache: *const ExecutableCache) !zml.Bufferized(Layer) {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        var buffers = try zml.mem.bufferize(allocator, Layer, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        defer loader.deinit();

        try loader.loadExecute(arena.allocator(), io, self.qkv_proj.weight, &buffers.qkv_proj.weight, store, shardings, &exe_cache.map.get(@tagName(.qkv_proj_weight)).?);
        if (self.qkv_proj.bias) |t| {
            try loader.loadExecute(arena.allocator(), io, t, &buffers.qkv_proj.bias.?, store, shardings, &exe_cache.map.get(@tagName(.qkv_proj_bias)).?);
        }
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Layer)) void {
        buffers.qkv_proj.weight.deinit();
        if (buffers.qkv_proj.bias) |*b| b.deinit();
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(init.arena.allocator());
    const path = args[1];

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();

    var it = registry.tensors.iterator();
    while (it.next()) |entry| {
        log.info("{s}: {f}", .{ entry.key_ptr.*, entry.value_ptr.shape });
    }

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const layer: Layer = .init(allocator, store.view());

    const model_sharding = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth }));

    var exe_cache: ExecutableCache = .init(allocator);
    defer exe_cache.deinit();

    const shardings = &.{model_sharding};

    try layer.preload(allocator, io, platform, &store, shardings, &exe_cache);

    var buffers = try layer.load(allocator, io, platform, &store, shardings, &exe_cache);
    defer Layer.unloadBuffers(&buffers);
}
