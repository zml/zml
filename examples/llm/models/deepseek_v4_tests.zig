const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const common = @import("common.zig");
const deepseek = @import("deepseek_v4.zig");
const model = deepseek.model;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const log = std.log.scoped(.deepseek_v4);

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use deepseek_v4_tests --model=<path> --activations=<path> [options]
        \\
        \\ Validate the Deepseek v4 implementation against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository
        \\   --activations=<path>      Path to activation safetensors
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
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var repo_model = try deepseek.LoadedModel.init(allocator, io, repo, store.view(), .{});
    defer repo_model.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    const shardings: common.Shardings = try .init(platform);

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    progress.end();
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    const backend = attention.Backend.auto(platform);
    const params = deepseek.CompilationParameters.init(repo_model.parsed_config.value.max_position_embeddings, repo_model.parsed_config.value, repo_model.inner, shardings, backend);

    // try testSparseAttn(allocator, io, platform, platform.replicated_sharding);
    try run(allocator, io, platform, args.activations, repo_model.parsed_config.value, repo_model.inner, &model_buffers, params.attention_metadata, params.attention_parameters);
}

fn testSparseAttn(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.Sharding) !void {
    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(allocator, io, "/Users/dmehala/models/sparse_attn.safetensors");
    defer activations_registry.deinit();
    log.info("Found {} activations", .{ activations_registry.tensors.count() });

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activation_store.deinit();

// ├── KV [shape={1,1024,128,bf16} size=0.26MB]
// ├── Q [shape={1,512,8,128,bf16} size=1.05MB]
// ├── attn_sink [shape={8,bf16} size=16B]
// ├── o [shape={1,512,8,128,bf16} size=1.05MB]
// └── topk_idxs [shape={1,512,256,i32} size=0.52MB]
    var kv_buffer = try loadBufferFromStore(allocator, io, platform, &activation_store, "KV", sharding);
    defer kv_buffer.deinit();
    const kv_tensor = zml.Tensor.fromShape(kv_buffer.shape()).withTags(.{ .batch, .k, .hd});

    var q_buffer = try loadBufferFromStore(allocator, io, platform, &activation_store, "Q", sharding);
    defer q_buffer.deinit();
    const q_tensor = zml.Tensor.fromShape(q_buffer.shape()).withTags(.{ .batch, .q, .h, .hd});

    var sink_buffer = try loadBufferFromStore(allocator, io, platform, &activation_store, "attn_sink", sharding);
    defer sink_buffer.deinit();
    const sink_tensor = zml.Tensor.fromShape(sink_buffer.shape());//.withTags(.{});

    var topk_buffer = try loadBufferFromStore(allocator, io, platform, &activation_store, "topk_idxs", sharding);
    defer topk_buffer.deinit();
    const topk_tensor = zml.Tensor.fromShape(topk_buffer.shape()).withTags(.{ .batch, .seq, .topk });

    var o_buffer = try loadBufferFromStore(allocator, io, platform, &activation_store, "o", sharding);
    defer o_buffer.deinit();

    const exe = try platform.compileFn(allocator, io, deepseek.model.sparse_attn, .{ q_tensor, kv_tensor, sink_tensor, topk_tensor, null }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ q_buffer, kv_buffer, sink_buffer, topk_buffer, null });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    var out_result = res.get(zml.Buffer);
    defer out_result.deinit();

    try zml.testing.expectClose(io, out_result, o_buffer, .{});
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    config: deepseek.Config,
    mdl: deepseek.Model,
    model_buffers: *deepseek.Buffers,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
) !void {
    _ = config; // autofix
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    var ctx = TestContext{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .activations_store = &activation_store,
        .attention_metadata = attention_metadata,
        .attention_parameters = attention_parameters,
        .sharding = platform.replicated_sharding,
    };

    // TODO: write why
    const dequant_opts: zml.testing.CompareOpts = .{.absolute_tolerance = 5e-2, .relative_tolerance = 2e-2 };

    try ctx.testLayer("embed", .{ .batch, .seq }, mdl.embeds, model_buffers.embeds, .{});
    // try ctx.testLayer("head", .{ .batch, .seq, .hc, .d }, mdl.lm_head, model_buffers.lm_head, dequant_opts);

    const n = 3;//16; 
    // const n = config.num_hidden_layers;

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const arena_allocator = arena.allocator();

    for (0..n) |i| {
        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn_norm", .{i}),
            .{ .batch, .seq, .d },
            mdl.layers[i].attn_norm,
            model_buffers.layers[i].attn_norm,
            .{}
        );

        // Attention
        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.wq_a", .{i}),
        .{ .batch, .seq, .d },
        mdl.layers[i].attn.wq_a,
        model_buffers.layers[i].attn.wq_a,
        dequant_opts,
        );

        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.q_norm", .{i}),
            .{ .batch, .seq, .q },
            mdl.layers[i].attn.q_norm,
            model_buffers.layers[i].attn.q_norm,
            .{},
        );

        // Disable because the activation only captured a 1/4 of the activations due
        // to Parallel operations.
        // try ctx.testLayer(
        //     try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.wq_b", .{i}),
        //     .{ .batch, .seq, .q },
        //     mdl.layers[0].attn.wq_b,
        //     model_buffers.layers[0].attn.wq_b,
        //     dequant_opts,
        // );

        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.wkv", .{i}),
            .{ .batch, .seq, .d },
            mdl.layers[i].attn.wkv,
            model_buffers.layers[i].attn.wkv,
            dequant_opts,
        );

        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.kv_norm", .{i}),
            .{ .batch, .seq, .hd },
            mdl.layers[i].attn.kv_norm,
            model_buffers.layers[i].attn.kv_norm,
            .{},
        );

        // try ctx.testLayer(
        //     try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.wo_a", .{i}),
        //     .{ .batch, .seq, .d },
        //     mdl.layers[i].attn.wo_a,
        //     model_buffers.layers[i].attn.wo_a,
        //     dequant_opts
        // );

        // try ctx.testLayer(
        //     try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.wo_b", .{i}),
        //     .{ .batch, .seq, .d },
        //     mdl.layers[0].attn.wo_b,
        //     model_buffers.layers[0].attn.wo_b,
        //     dequant_opts
        // );

        if (mdl.layers[i].attn.compressor) |compressor| {
            try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.compressor.wkv", .{i}),
            .{ .batch, .seq, .d },
            compressor.wkv,
            model_buffers.layers[i].attn.compressor.?.wkv,
            .{}
            );

            try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.compressor.wgate", .{i}),
            .{ .batch, .seq, .d },
            compressor.wgate,
            model_buffers.layers[i].attn.compressor.?.wgate,
            .{}
            );

            // try ctx.testLayer(
            // try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.compressor.norm", .{i}),
            // .{ .batch, .seq, .hd },
            // compressor.norm,
            // model_buffers.layers[i].attn.compressor.?.norm,
            // .{}
            // );

            try ctx.testCompressorLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.compressor", .{i}),
            .{ .batch, .seq, .d },
            compressor,
            model_buffers.layers[i].attn.compressor.?,
            .{},
            );
        }

        if (mdl.layers[i].attn.indexer) |indexer| {
            // try ctx.testCompressorLayer(
            // try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.indexer.compressor", .{i}),
            // .{ .batch, .seq, .d },
            // indexer.compressor,
            // model_buffers.layers[i].attn.compressor.?,
            // .{}
            // );

            // try ctx.testLayer(
            // try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.indexer.weights_proj", .{i}),
            // .{ .batch, .seq, .d },
            // indexer.proj,
            // model_buffers.layers[i].attn.indexer.?.proj,
            // .{}
            // );

            // try ctx.testLayer(
            // try std.fmt.allocPrint(arena_allocator, "layers.{}.attn.indexer.wq_b", .{i}),
            // .{ .batch, .seq, .d },
            // indexer.wq_b,
            // model_buffers.layers[i].attn.indexer.?.wq_b,
            // .{}
            // );

            try ctx.testIndexerLayer(
            "layers.2.attn.indexer",
            .{ .batch, .seq, .d },
            indexer,
            model_buffers.layers[i].attn.indexer.?,
            .{},
            );
        }

        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn_norm", .{i}),
            .{ .batch, .seq, .d },
            mdl.layers[i].ffn_norm,
            model_buffers.layers[i].ffn_norm,
            .{}
        );

        // MoE
        // try ctx.testGateLayer(
        //     try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn_gate", .{i}),
        //     .{ .seq, .d },
        //     mdl.layers[i].ffn.gate,
        //     model_buffers.layers[i].ffn.gate,
        //     .{}
        // );
        
        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn.shared_experts.w1", .{i}),
            .{ .seq, .d },
            mdl.layers[i].ffn.shared_experts.w1,
            model_buffers.layers[i].ffn.shared_experts.w1,
            dequant_opts,
        );

        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn.shared_experts.w2", .{i}),
            .{ .seq, .dint },
            mdl.layers[i].ffn.shared_experts.w2,
            model_buffers.layers[i].ffn.shared_experts.w2,
            dequant_opts,
        );

        try ctx.testLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn.shared_experts.w3", .{i}),
            .{ .seq, .d },
            mdl.layers[i].ffn.shared_experts.w3,
            model_buffers.layers[i].ffn.shared_experts.w3,
            dequant_opts,
        );

        try ctx.testExpertLayer(
            try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn.shared_experts", .{i}),
            .{ .seq, .d },
            mdl.layers[i].ffn.shared_experts,
            model_buffers.layers[i].ffn.shared_experts,
            dequant_opts,
        );

        // TEST: MoE (complete)
        // try ctx.testMoELayer(
        //     try std.fmt.allocPrint(arena_allocator, "layers.{}.ffn", .{i}),
        //     .{ .batch, .seq, .d },
        //     mdl.layers[0].ffn,
        //     model_buffers.layers[0].ffn,
        //     .{}
        // );
    }
}

pub const Layer = struct {
    name: []const u8,
    tagz: zml.Shape.Tag,
};

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_store: *zml.io.TensorStore,
    attention_metadata: attention.Metadata,
    attention_parameters: attention.Parameters,
    sharding: zml.Sharding,

    fn testLayerPrint(self: *TestContext, comptime name_fmt: []const u8, name_args: anytype, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        const name = try std.fmt.allocPrint(self.allocator, name_fmt, name_args);
        defer self.allocator.free(name);
        try self.testLayer(name, tagz, layer, layer_buffers, opts);
    }

    fn testLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testLayerLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const in_key_1 = try std.fmt.allocPrint(self.allocator, "{s}.in.1", .{name});
        defer self.allocator.free(in_key_1);
        var in_buffer_1 = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key_1, self.sharding);
        defer in_buffer_1.deinit();
        const in_tensor_1 = zml.Tensor.fromShape(in_buffer_1.shape());

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, in_tensor_1 }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer, in_buffer_1 });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testCompressorLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            @TypeOf(layer).forward,
            .{ 
                layer,
                in_tensor,
                0,
            },
            .{ .shardings = &.{self.sharding} }
        );
        defer exe.deinit();

        var attention_metadata_buffers = try self.attention_metadata.initBuffer(self.io, self.platform, self.sharding);
        defer attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{
            layer_buffers,
            in_buffer,
        });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testAttentionLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const token_idx_offset = zml.Tensor.scalar(@as(u32, 0), .u32);
        const layer_idx = zml.Tensor.scalar(@as(u32, 0), .u32);
        const cache = model.KVCache.init(.init());

        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            @TypeOf(layer).forward,
            .{ 
                layer,
                in_tensor,
                token_idx_offset,
                layer_idx,
                cache,
                self.*.attention_metadata,
                self.*.attention_parameters,
            },
            .{ .shardings = &.{self.sharding} }
        );
        defer exe.deinit();

        var attention_metadata_buffers = try self.attention_metadata.initBuffer(self.io, self.platform, self.sharding);
        defer attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{
            layer_buffers,
            in_buffer,
            // token_idx_buffer,
            // cache_buffer,
            attention_metadata_buffers,
        });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testLayerV2(self: *TestContext, input_layers: []Layer, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        _ = layer; // autofix
        _ = layer_buffers; // autofix
        _ = opts; // autofix
        std.log.info("Testing layer: {s}", .{input_layers[0].name});

        const buffers = try self.allocator.alloc(zml.Buffer, input_layers.len);
        // const tensors = try self.allocator.alloc(zml.Tensor, input_layers.len);
        for (buffers, 0..) |*buffer, i| {
            buffer.* = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, input_layers[i], self.sharding);
        }

        // const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        // defer self.allocator.free(in_key);
        // var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        // defer in_buffer.deinit();
        // const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);
        //
        // const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        // defer self.allocator.free(out_key);
        // var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        // defer out_buffer_expected.deinit();
        //
        // const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor }, .{ .shardings = &.{self.sharding} });
        // defer exe.deinit();
        //
        // var args = try exe.args(self.allocator);
        // defer args.deinit(self.allocator);
        // args.set(.{ layer_buffers, in_buffer });
        //
        // var res = try exe.results(self.allocator);
        // defer res.deinit(self.allocator);
        //
        // exe.call(args, &res);
        //
        // var out_result = res.get(zml.Buffer);
        // defer out_result.deinit();
        //
        // try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{input_layers[0].name});
    }

    fn testIndexerLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const in_1_key = try std.fmt.allocPrint(self.allocator, "{s}.in.1", .{name});
        defer self.allocator.free(in_1_key);
        var in_1_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_1_key, self.sharding);
        defer in_1_buffer.deinit();
        const in_1_tensor = zml.Tensor.fromShape(in_1_buffer.shape()).withTags(tagz);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, in_1_tensor, 0 }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer, in_1_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testGateLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const in_1_key = try std.fmt.allocPrint(self.allocator, "{s}.in.1", .{name});
        defer self.allocator.free(in_1_key);
        var in_1_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_1_key, self.sharding);
        defer in_1_buffer.deinit();
        const in_1_tensor = zml.Tensor.fromShape(in_1_buffer.shape()).withTags(.{.seq});

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const out_1_key = try std.fmt.allocPrint(self.allocator, "{s}.out.1", .{name});
        defer self.allocator.free(out_1_key);
        var out_1_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_1_key, self.sharding);
        defer out_1_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, in_1_tensor }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer, in_1_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result, var out_1_result = res.get(struct { zml.Buffer, zml.Buffer });
        defer out_result.deinit();
        defer out_1_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        try zml.testing.expectClose(self.io, out_1_result, out_1_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testMoELayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const in_1_key = try std.fmt.allocPrint(self.allocator, "{s}.in.1", .{name});
        defer self.allocator.free(in_1_key);
        var in_1_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_1_key, self.sharding);
        defer in_1_buffer.deinit();
        const in_1_tensor = zml.Tensor.fromShape(in_1_buffer.shape()).withTags(.{.batch,.seq});

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, in_1_tensor }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer, in_1_buffer });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testExpertLayer(self: *TestContext, name: []const u8, tagz: anytype, layer: anytype, layer_buffers: anytype, opts: zml.testing.CompareOpts) !void {
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in.0", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(tagz);

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out.0", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, null }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, in_buffer, null });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);

        exe.call(args, &res);

        var out_result = res.get(zml.Buffer);
        defer out_result.deinit();

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }

    fn testAttnLayer(self: *TestContext, ix: usize, cache_ix: usize, layer: model.Attention, layer_buffers: zml.Bufferized(model.Attention), opts: zml.testing.CompareOpts) !void {
        const name = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn", .{ix});
        defer self.allocator.free(name);
        std.log.info("Testing layer: {s}", .{name});

        const in_key = try std.fmt.allocPrint(self.allocator, "{s}.in", .{name});
        defer self.allocator.free(in_key);
        var in_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, in_key, self.sharding);
        defer in_buffer.deinit();
        const in_tensor = zml.Tensor.fromShape(in_buffer.shape()).withTags(.{ .batch, .seq, .d });

        const key_cache_key = try std.fmt.allocPrint(self.allocator, "{s}.cache.key", .{name});
        defer self.allocator.free(key_cache_key);
        var key_cache_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, key_cache_key, self.sharding);
        defer key_cache_buffer.deinit();
        const key_cache_tensor = zml.Tensor.fromShape(key_cache_buffer.shape()).withTags(.{ .layer, .batch, .h, .k, .hd });

        const value_cache_key = try std.fmt.allocPrint(self.allocator, "{s}.cache.value", .{name});
        defer self.allocator.free(value_cache_key);
        var value_cache_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, value_cache_key, self.sharding);
        defer value_cache_buffer.deinit();
        const value_cache_tensor = zml.Tensor.fromShape(value_cache_buffer.shape()).withTags(.{ .layer, .batch, .h, .k, .hd });

        const cache_pos_key = try std.fmt.allocPrint(self.allocator, "{s}.cache_position", .{name});
        defer self.allocator.free(cache_pos_key);
        var cache_pos_buffer = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, cache_pos_key, self.sharding);
        defer cache_pos_buffer.deinit();
        const cache_pos_tensor = zml.Tensor.fromShape(cache_pos_buffer.shape()).withTags(.{.batch});

        const out_key = try std.fmt.allocPrint(self.allocator, "{s}.out", .{name});
        defer self.allocator.free(out_key);
        var out_buffer_expected = try loadBufferFromStore(self.allocator, self.io, self.platform, self.activations_store, out_key, self.sharding);
        defer out_buffer_expected.deinit();

        const cache_index_tensor: zml.Tensor = .init(.{}, .u32);

        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, in_tensor, cache_pos_tensor, model.KvCache{ .k = key_cache_tensor, .v = value_cache_tensor }, cache_index_tensor, self.attention_metadata, self.attention_parameters }, .{ .shardings = &.{self.sharding} });
        defer exe.deinit();

        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        const kv_cache: zml.Bufferized(model.KvCache) = .{ .k = key_cache_buffer, .v = value_cache_buffer };

        var attention_metadata_buffers = try self.attention_metadata.initBuffer(self.io, self.platform, self.sharding);
        defer attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        var cache_index_buf: zml.Buffer = try .scalar(self.io, self.platform, cache_ix, .u32);
        defer cache_index_buf.deinit();

        args.set(.{ layer_buffers, in_buffer, cache_pos_buffer, kv_cache, cache_index_buf, attention_metadata_buffers });

        var res = try exe.results(self.allocator);
        defer res.deinit(self.allocator);
        exe.call(args, &res);

        var out_result, var updated_kv = res.get(struct { zml.Buffer, zml.Bufferized(model.KvCache) });
        defer out_result.deinit();
        defer model.KvCache.unloadBuffers(&updated_kv);

        try zml.testing.expectClose(self.io, out_result, out_buffer_expected, opts);
        std.log.info("Layer {s} passed!", .{name});
    }
};

fn loadBufferFromStore(allocator: std.mem.Allocator, io: anytype, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8, sharding: zml.Sharding) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}
