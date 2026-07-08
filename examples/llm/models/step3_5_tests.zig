const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate Step 3.5 Flash layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>         Path to the model repository
        \\   --activations=<path>   Path to activation safetensors
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var model_registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer model_registry.deinit();
    var model_store: zml.io.TensorStore = .fromRegistry(allocator, &model_registry);
    defer model_store.deinit();

    try run(allocator, io, platform, repo, args.activations, &model_store);
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    repo: std.Io.Dir,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
) !void {
    var activation_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer activation_registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &activation_registry);
    defer activation_store.deinit();

    var parsed_config = try common.parseConfig(model.Config, allocator, io, repo);
    defer parsed_config.deinit();

    const shardings: common.Shardings = try .init(platform);
    var ctx: TestContext = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .model_store = model_store,
        .activation_store = &activation_store,
        .config = parsed_config.value,
        .shardings = shardings,
        .all_shardings = shardings.all(),
    };

    try ctx.run();
}

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    config: model.Config,
    shardings: common.Shardings,
    all_shardings: [2]zml.Sharding,

    fn run(self: *TestContext) !void {
        std.log.info("Step 3.5 black-box layer tests:", .{});

        const num_layers: usize = @intCast(self.config.numMainLayers());
        for (0..num_layers) |layer_idx| {
            try self.testDecoderLayer(layer_idx);
        }
    }

    fn testDecoderLayer(self: *TestContext, layer_idx: usize) !void {
        const layer_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}", .{layer_idx});
        defer self.allocator.free(layer_name);
        const layer = try model.TransformerLayer.init(self.model_store.view().withPrefix(layer_name), layer_idx, self.config);
        try self.testKvLayer(layer_name, layer, .{ .absolute_tolerance = 2e-2, .minimum_close_fraction = 0.99 });

        const attn_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.self_attn", .{layer_idx});
        defer self.allocator.free(attn_name);
        const attn = try model.Attn.init(self.model_store.view().withPrefix(attn_name), layer_idx, self.config);
        try self.testKvLayer(attn_name, attn, .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 });

        const input_norm_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.input_layernorm", .{layer_idx});
        defer self.allocator.free(input_norm_name);
        const input_norm = model.RmsNorm.init(self.model_store.view().withPrefix(input_norm_name), 1e-5);
        try self.testRmsNorm(input_norm_name, input_norm, .{ .absolute_tolerance = 1e-2 });

        const post_norm_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.post_attention_layernorm", .{layer_idx});
        defer self.allocator.free(post_norm_name);
        const post_norm = model.RmsNorm.init(self.model_store.view().withPrefix(post_norm_name), 1e-5);
        try self.testRmsNorm(post_norm_name, post_norm, .{ .absolute_tolerance = 1e-2 });

        const shared_limit: f32 = if (layer_idx < self.config.swiglu_limits_shared.len)
            self.config.swiglu_limits_shared[layer_idx]
        else
            0.0;

        if (self.isMoeLayer(layer_idx)) {
            const moe_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.moe", .{layer_idx});
            defer self.allocator.free(moe_name);
            const moe = try model.Moe.init(self.model_store.view().withPrefix(moe_name), layer_idx, self.config);
            try self.testSimpleLayer(moe_name, moe, .{ .absolute_tolerance = 1e-2, .minimum_close_fraction = 0.99 });

            const share_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.share_expert", .{layer_idx});
            defer self.allocator.free(share_name);
            const share = model.Mlp.init(self.model_store.view().withPrefix(share_name), shared_limit);
            try self.testSimpleLayer(share_name, share, .{ .absolute_tolerance = 1e-2 });
        } else {
            const mlp_name = try std.fmt.allocPrint(self.allocator, "model.layers.{d}.mlp", .{layer_idx});
            defer self.allocator.free(mlp_name);
            const mlp = model.Mlp.init(self.model_store.view().withPrefix(mlp_name), shared_limit);
            try self.testSimpleLayer(mlp_name, mlp, .{ .absolute_tolerance = 1e-2 });
        }
    }

    fn testSimpleLayer(self: *TestContext, name: []const u8, layer: anytype, opts: zml.testing.CompareOpts) !void {
        if (!self.hasFixture(name)) {
            std.log.warn("skipping {s}: no in.0/out.0 activations recorded", .{name});
            return;
        }
        std.log.info("Testing layer: {s}", .{name});

        var layer_buffers = try zml.io.load(@TypeOf(layer), &layer, self.allocator, self.io, self.platform, self.model_store, .auto);
        defer deinitBuffers(&layer_buffers);

        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, try self.key(name, "in.0"), .replicated);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape());

        var expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, try self.key(name, "out.0"), .replicated);
        defer expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &self.all_shardings });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(args, &results);

        var actual = results.get(zml.Buffer);
        defer releaseBuffer(in_buffer, &actual);

        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn testRmsNorm(self: *TestContext, name: []const u8, rms: model.RmsNorm, opts: zml.testing.CompareOpts) !void {
        if (!self.hasFixture(name)) {
            std.log.warn("skipping {s}: no in.0/out.0 activations recorded", .{name});
            return;
        }
        std.log.info("Testing layer: {s}", .{name});

        var rms_buffers = try zml.io.load(model.RmsNorm, &rms, self.allocator, self.io, self.platform, self.model_store, .auto);
        defer deinitBuffers(&rms_buffers);

        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, try self.key(name, "in.0"), .replicated);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape());

        var expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, try self.key(name, "out.0"), .replicated);
        defer expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, rmsNormForward, .{ rms, in_tensor }, .{ .shardings = &self.all_shardings });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ rms_buffers, in_buffer });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(args, &results);

        var actual = results.get(zml.Buffer);
        defer releaseBuffer(in_buffer, &actual);

        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn testKvLayer(self: *TestContext, name: []const u8, layer: anytype, opts: zml.testing.CompareOpts) !void {
        if (!self.hasFixture(name)) {
            std.log.warn("skipping {s}: no in.0/out.0 activations recorded", .{name});
            return;
        }
        std.log.info("Testing layer: {s}", .{name});

        var layer_buffers = try zml.io.load(@TypeOf(layer), &layer, self.allocator, self.io, self.platform, self.model_store, .auto);
        defer deinitBuffers(&layer_buffers);

        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, try self.key(name, "in.0"), .replicated);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(.{ .b, .s, .d });

        var expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activation_store, try self.key(name, "out.0"), .replicated);
        defer expected.deinit();

        const token_index: zml.Tensor = .init(.{ .s = 1 }, .u32);
        const kv_cache = self.kvCacheForInput(in_tensor);
        const backend = zml.attention.attention.Backend.auto(self.platform);
        const metadata_num_heads: i64 = @intCast(if (self.config.attention_other_setting) |other|
            @max(self.config.num_attention_heads, other.num_attention_heads)
        else
            self.config.num_attention_heads);
        const attention_metadata: zml.attention.attention.Metadata = .init(.fromBackend(backend, in_tensor.dim(.s), metadata_num_heads));
        const attention_parameters: zml.attention.attention.Parameters = .init(.fromBackend(backend));

        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            @TypeOf(layer).forward,
            .{ layer, in_tensor, token_index, kv_cache, attention_metadata, attention_parameters },
            .{ .shardings = &self.all_shardings },
        );
        defer exe.deinit();

        var token_index_buffer = try zml.Buffer.fromSlice(
            self.io,
            self.platform,
            .init(zml.Shape.init(.{ .s = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0})),
            .replicated,
        );
        defer token_index_buffer.deinit();

        var kv_cache_buffers = try kv_cache.initBuffer(self.allocator, self.io, self.platform, self.shardings.model);
        defer model.KvCache.deinitBuffer(&kv_cache_buffers);

        var attention_metadata_buffers = try attention_metadata.initBuffer(self.io, self.platform, self.shardings.model);
        defer zml.attention.attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer, token_index_buffer, kv_cache_buffers, attention_metadata_buffers });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(args, &results);

        var actual, var updated_kv_cache = results.get(struct { zml.Buffer, zml.Bufferized(model.KvCache) });
        defer releaseKvCacheBuffers(kv_cache_buffers, &updated_kv_cache);
        defer releaseBuffer(in_buffer, &actual);

        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn kvCacheForInput(self: *TestContext, input: zml.Tensor) model.KvCache {
        const model_partitions = self.shardings.model.numPartitionsForLogicalAxis(.model);
        const raw_shape = zml.Shape.init(.{
            .layer = 1,
            .b = input.dim(.b),
            .k = input.dim(.s),
            .h = @as(i64, @intCast(self.config.num_attention_groups)),
            .hd = @as(i64, @intCast(self.config.head_dim)),
        }, input.dtype());
        const shape = model.partitionKvCacheShape(raw_shape, @intCast(self.config.num_attention_groups), model_partitions);
        return .init(shape);
    }

    fn hasFixture(self: *TestContext, name: []const u8) bool {
        const in_key = self.key(name, "in.0") catch return false;
        defer self.allocator.free(in_key);
        const out_key = self.key(name, "out.0") catch return false;
        defer self.allocator.free(out_key);
        const view = self.activation_store.view();
        return view.hasKey(in_key) and view.hasKey(out_key);
    }

    fn key(self: *TestContext, name: []const u8, suffix: []const u8) ![]u8 {
        return std.fmt.allocPrint(self.allocator, "{s}.{s}", .{ name, suffix });
    }

    fn isMoeLayer(self: *TestContext, layer_idx: usize) bool {
        for (self.config.moe_layers_enum.layers) |moe_layer_idx| {
            if (moe_layer_idx == layer_idx) return true;
        }
        return false;
    }
};

fn rmsNormForward(rms: model.RmsNorm, input: zml.Tensor) zml.Tensor {
    return rms.forward(input, .d);
}

fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.Sharding,
) !zml.Buffer {
    defer allocator.free(key);
    const shape = store.view().getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

fn releaseKvCacheBuffers(expected: zml.Bufferized(model.KvCache), actual: *zml.Bufferized(model.KvCache)) void {
    releaseBuffer(expected.k, &actual.k);
    releaseBuffer(expected.v, &actual.v);
    releaseBuffer(expected.layer_index, &actual.layer_index);
}

fn releaseBuffer(expected: zml.Buffer, actual: *zml.Buffer) void {
    if (!sameBufferHandle(expected, actual.*)) actual.deinit();
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
