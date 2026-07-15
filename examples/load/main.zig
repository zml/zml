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

pub fn generateExpertsKeys(allocator: std.mem.Allocator, comptime fmt: []const u8, num_experts: usize) ![]const []const u8 {
    var keys_list: std.ArrayList([]const u8) = try .initCapacity(allocator, num_experts);
    defer keys_list.deinit(allocator);
    defer for (keys_list.items) |key| allocator.free(key);

    for (0..num_experts) |i| {
        const key = try std.fmt.allocPrint(allocator, fmt, .{i});
        keys_list.appendAssumeCapacity(key);
    }

    return try keys_list.toOwnedSlice(allocator);
}

pub const Experts = struct {
    down_proj: zml.nn.Linear,
    gate_up_proj: zml.nn.Linear,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Experts {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        const num_experts = 256;
        const down_proj_weight_keys = try generateExpertsKeys(arena.allocator(), "{}.down_proj.weight", num_experts);

        var down_proj_weight_shape = store.getShape(down_proj_weight_keys[0]).?;
        down_proj_weight_shape = down_proj_weight_shape.insert(0, .{ .experts = num_experts }).withTags(.{ .experts, .d, .dout }).withPartitioning(.{ .experts = .model });

        const down_proj_weight = store.maybeCreateBinding(down_proj_weight_keys, down_proj_weight_shape).?;

        const gate_proj_weight_keys = try generateExpertsKeys(arena.allocator(), "{}.gate_proj.weight", num_experts);
        const up_proj_weight_keys = try generateExpertsKeys(arena.allocator(), "{}.up_proj.weight", num_experts);

        var gate_proj_weight_shape = store.getShape(gate_proj_weight_keys[0]).?;
        var up_proj_weight_shape = store.getShape(up_proj_weight_keys[0]).?;
        const concatenated_dim = gate_proj_weight_shape.dim(0) + up_proj_weight_shape.dim(0);
        const gate_up_proj_weight_shape = gate_proj_weight_shape.setDim(0, concatenated_dim).insert(0, .{ .experts = num_experts }).withTags(.{ .experts, .dout, .d }).withPartitioning(.{ .experts = .model });

        const gate_up_proj_weight_keys = try arena.allocator().alloc([]const u8, gate_proj_weight_keys.len + up_proj_weight_keys.len);
        @memcpy(gate_up_proj_weight_keys[0..gate_proj_weight_keys.len], gate_proj_weight_keys);
        @memcpy(gate_up_proj_weight_keys[gate_proj_weight_keys.len..][0..up_proj_weight_keys.len], up_proj_weight_keys);

        const gate_up_proj_weight = store.maybeCreateBinding(gate_up_proj_weight_keys, gate_up_proj_weight_shape).?;

        return .{
            .down_proj = .init(down_proj_weight, null, .d),
            .gate_up_proj = .init(gate_up_proj_weight, null, .d),
        };
    }

    pub fn preload(
        self: *const Experts,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.Sharding,
        exe_cache: *ExecutableCache,
    ) !void {
        const down_weight_func = struct {
            fn cb(inputs: []const zml.Tensor) zml.Tensor {
                return zml.Tensor.stack(inputs, 0, .experts).withTags(.{ .experts, .d, .dout }).withPartitioning(.{ .experts = .model });
            }
        }.cb;

        const down_weight_gop = try exe_cache.map.getOrPut(exe_cache.allocator, @tagName(.down_proj_weight));
        errdefer exe_cache.map.removeByPtr(down_weight_gop.key_ptr);

        if (!down_weight_gop.found_existing) {
            const sources = store.getSourcesById(self.down_proj.weight.id).?;
            const tensors = try getTensors(allocator, sources);
            defer allocator.free(tensors);
            var exe = try platform.compileFn(allocator, io, down_weight_func, .{tensors}, .{ .shardings = shardings });
            errdefer exe.deinit();

            down_weight_gop.value_ptr.* = exe;
        }

        const gate_up_weight_func = struct {
            fn cb(inputs: []const zml.Tensor) zml.Tensor {
                const gate_proj_weights = inputs[0 .. inputs.len / 2];
                const up_proj_weights = inputs[inputs.len / 2 ..];

                const stacked_gate_proj = zml.Tensor.stack(gate_proj_weights, 0, .experts);
                const stacked_up_proj = zml.Tensor.stack(up_proj_weights, 0, .experts);
                return zml.Tensor.concatenate(&.{ stacked_gate_proj, stacked_up_proj }, 1).withTags(.{ .experts, .dout, .d }).withPartitioning(.{ .experts = .model });
            }
        }.cb;

        const gate_up_weight_gop = try exe_cache.map.getOrPut(exe_cache.allocator, @tagName(.gate_up_proj_weight));
        errdefer exe_cache.map.removeByPtr(gate_up_weight_gop.key_ptr);

        if (!gate_up_weight_gop.found_existing) {
            const sources = store.getSourcesById(self.gate_up_proj.weight.id).?;
            const tensors = try getTensors(allocator, sources);
            defer allocator.free(tensors);
            var exe = try platform.compileFn(allocator, io, gate_up_weight_func, .{tensors}, .{ .shardings = shardings });
            errdefer exe.deinit();

            gate_up_weight_gop.value_ptr.* = exe;
        }
    }

    pub fn load(self: *const Experts, allocator: std.mem.Allocator, io: std.Io, loader: *zml.io.Loader, buffers: *zml.Bufferized(Experts), store: *const zml.io.TensorStore, shardings: []const zml.Sharding, exe_cache: *const ExecutableCache) !void {
        var arena: std.heap.ArenaAllocator = .init(allocator);
        defer arena.deinit();

        try loader.loadExecute(arena.allocator(), io, self.down_proj.weight, &buffers.down_proj.weight, store, shardings, &exe_cache.map.get(@tagName(.down_proj_weight)).?);
        try loader.loadExecute(arena.allocator(), io, self.gate_up_proj.weight, &buffers.gate_up_proj.weight, store, shardings, &exe_cache.map.get(@tagName(.gate_up_proj_weight)).?);
        try loader.await(io);
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Experts)) void {
        buffers.down_proj.weight.deinit();
        if (buffers.down_proj.bias) |*b| b.deinit();
    }
};

pub const Mlp = struct {
    experts: Experts,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Mlp {
        const experts = try Experts.init(allocator, store.withPrefix("experts"));
        return .{ .experts = experts };
    }

    pub fn preload(
        self: *const Mlp,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.Sharding,
        exe_cache: *ExecutableCache,
    ) !void {
        try self.experts.preload(allocator, io, platform, store, shardings, exe_cache);
    }

    pub fn load(self: *const Mlp, allocator: std.mem.Allocator, io: std.Io, loader: *zml.io.Loader, buffers: *zml.Bufferized(Mlp), store: *const zml.io.TensorStore, shardings: []const zml.Sharding, exe_cache: *const ExecutableCache) !void {
        try self.experts.load(allocator, io, loader, &buffers.experts, store, shardings, exe_cache);
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Mlp)) void {
        Experts.unloadBuffers(&buffers.experts);
    }
};

pub const Layer = struct {
    mlp: Mlp,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Layer {
        const mlp = try Mlp.init(allocator, store.withPrefix("mlp"));
        return .{ .mlp = mlp };
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
        try self.mlp.preload(allocator, io, platform, store, shardings, exe_cache);
    }

    pub fn load(self: *const Layer, allocator: std.mem.Allocator, io: std.Io, loader: *zml.io.Loader, buffers: *zml.Bufferized(Layer), store: *const zml.io.TensorStore, shardings: []const zml.Sharding, exe_cache: *const ExecutableCache) !void {
        try self.mlp.load(allocator, io, loader, &buffers.mlp, store, shardings, exe_cache);
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Layer)) void {
        Mlp.unloadBuffers(&buffers.mlp);
    }
};

pub const Model = struct {
    layers: []Layer,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View) !Model {
        const num_layers = 39;
        var layers_list: std.ArrayList(Layer) = try .initCapacity(allocator, num_layers);
        defer layers_list.deinit(allocator);

        for (0..num_layers) |i| {
            const layer = try Layer.init(allocator, store.withPrefix("layers").withLayer(i + 1));
            layers_list.appendAssumeCapacity(layer);
        }

        return .{ .layers = try layers_list.toOwnedSlice(allocator) };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn preload(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.Sharding,
        exe_cache: *ExecutableCache,
    ) !void {
        try self.layers[0].preload(allocator, io, platform, store, shardings, exe_cache);
    }

    pub fn load(self: *const Model, allocator: std.mem.Allocator, io: std.Io, loader: *zml.io.Loader, buffers: *zml.Bufferized(Model), store: *const zml.io.TensorStore, shardings: []const zml.Sharding, exe_cache: *const ExecutableCache) !void {
        for (self.layers, 0..) |layer, i| {
            try layer.load(allocator, io, loader, &buffers.layers[i], store, shardings, exe_cache);
        }
    }

    pub fn unloadBuffers(buffers: *zml.Bufferized(Model)) void {
        for (buffers.layers) |*layer| {
            Layer.unloadBuffers(layer);
        }
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

    const model: Model = try .init(allocator, store.view().withPrefix("model"));
    defer model.deinit(allocator);

    const model_sharding = try platform.registerSharding("model", .mesh(.{ .model = .high_bandwidth }));

    var exe_cache: ExecutableCache = .init(allocator);
    defer exe_cache.deinit();

    const shardings = &.{model_sharding};

    try model.preload(allocator, io, platform, &store, shardings, &exe_cache);

    var buffers = try zml.mem.bufferize(allocator, Model, &model);
    defer Model.unloadBuffers(&buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .{ .parallelism = 16, .dma_chunks = 32, .dma_chunk_size = 128 * zml.MiB });
    defer loader.deinit();

    try model.load(allocator, io, &loader, &buffers, &store, shardings, &exe_cache);
}
