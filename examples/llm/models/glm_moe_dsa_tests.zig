const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const glm = @import("glm_moe_dsa.zig");
const model = @import("glm_moe_dsa/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,

    pub const help =
        \\Use glm_moe_dsa_tests --model=<path> --activations=<path>
        \\
        \\ Validate a four-layer GLM-5.2 prefix against PyTorch activation fixtures.
        \\ The test uses index_topk=8 and all devices exposed to the ZML platform.
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

    var repo_model = try glm.LoadedModel.init(allocator, io, repo, store.view(), .{
        .layer_limit = 4,
        .index_topk_override = 8,
    });
    defer repo_model.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    const shardings: common.Shardings = try .init(platform);
    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);
    progress.end();

    try run(
        allocator,
        io,
        platform,
        args.activations,
        repo_model.inner,
        &model_buffers,
        repo_model.parsed_config.value,
        shardings,
    );
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    mdl: model.Model,
    model_buffers: *model.Buffers,
    config: model.Config,
    shardings: common.Shardings,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();
    var activations: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activations.deinit();

    const test_context: TestContext = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .activations = activations.view(),
        .shardings = shardings,
    };

    try test_context.testFullAttentionPrefillDecode(
        mdl.layers[0].self_attn,
        model_buffers.layers[0].self_attn,
        config,
        0,
    );
    try test_context.testSharedAttentionPrefill(
        mdl.layers[3].self_attn,
        model_buffers.layers[3].self_attn,
        config,
        3,
    );

    // Leaf tests localize layout, dtype, and weight-loading mistakes before the full graph.
    try test_context.testLayer("prefill.embed_tokens", .{ .b, .s }, mdl.embed_tokens, model_buffers.embed_tokens, .{});
    try test_context.testLayer("prefill.norm", .{ .b, .s, .d }, mdl.norm, model_buffers.norm, .{ .absolute_tolerance = 2e-2 });
    try test_context.testLayer("prefill.lm_head", .{ .b, .s, .d }, mdl.lm_head, model_buffers.lm_head, .{ .absolute_tolerance = 5e-2 });

    const dense_layer = mdl.layers[0];
    const dense_buffers = model_buffers.layers[0];
    try test_context.testLayer("prefill.layers.0.input_layernorm", .{ .b, .s, .d }, dense_layer.input_layernorm, dense_buffers.input_layernorm, .{ .absolute_tolerance = 2e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.q_a_proj", .{ .b, .s, .d }, dense_layer.self_attn.q_a_proj, dense_buffers.self_attn.q_a_proj, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.q_a_layernorm", .{ .b, .s, .q_lora }, dense_layer.self_attn.q_a_layernorm, dense_buffers.self_attn.q_a_layernorm, .{ .absolute_tolerance = 2e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.q_b_proj", .{ .b, .s, .d }, dense_layer.self_attn.q_b_proj, dense_buffers.self_attn.q_b_proj, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.kv_a_proj_with_mqa", .{ .b, .s, .d }, dense_layer.self_attn.kv_a_proj_with_mqa, dense_buffers.self_attn.kv_a_proj_with_mqa, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.kv_a_layernorm", .{ .b, .s, .kv_lora }, dense_layer.self_attn.kv_a_layernorm, dense_buffers.self_attn.kv_a_layernorm, .{ .absolute_tolerance = 2e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.kv_b_proj", .{ .b, .h, .s, .d }, dense_layer.self_attn.kv_b_proj, dense_buffers.self_attn.kv_b_proj, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.o_proj", .{ .b, .s, .d }, dense_layer.self_attn.o_proj, dense_buffers.self_attn.o_proj, .{ .absolute_tolerance = 5e-2 });

    const indexer = dense_layer.self_attn.indexer.?;
    const indexer_buffers = dense_buffers.self_attn.indexer.?;
    try test_context.testLayer("prefill.layers.0.self_attn.indexer.wq_b", .{ .b, .s, .d }, indexer.wq_b, indexer_buffers.wq_b, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.indexer.wk", .{ .b, .s, .d }, indexer.wk, indexer_buffers.wk, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.indexer.k_norm", .{ .b, .s, .d }, indexer.k_norm, indexer_buffers.k_norm, .{ .absolute_tolerance = 2e-2 });
    try test_context.testLayer("prefill.layers.0.self_attn.indexer.weights_proj", .{ .b, .s, .d }, indexer.weights_proj, indexer_buffers.weights_proj, .{ .absolute_tolerance = 3e-2 });

    const dense_mlp = switch (dense_layer.feed_forward) {
        .dense => |value| value,
        .sparse => unreachable,
    };
    const dense_mlp_buffers = switch (dense_buffers.feed_forward) {
        .dense => |value| value,
        .sparse => unreachable,
    };
    try test_context.testLayer("prefill.layers.0.mlp.gate_proj", .{ .b, .s, .d }, dense_mlp.gate_proj, dense_mlp_buffers.gate_proj, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.mlp.up_proj", .{ .b, .s, .d }, dense_mlp.up_proj, dense_mlp_buffers.up_proj, .{ .absolute_tolerance = 3e-2 });
    try test_context.testLayer("prefill.layers.0.mlp.down_proj", .{ .b, .s, .d }, dense_mlp.down_proj, dense_mlp_buffers.down_proj, .{ .absolute_tolerance = 5e-2 });
    try test_context.testLayer("prefill.layers.0.mlp", .{ .b, .s, .d }, dense_mlp, dense_mlp_buffers, .{ .absolute_tolerance = 7e-2 });

    const sparse_mlp = switch (mdl.layers[3].feed_forward) {
        .sparse => |value| value,
        .dense => unreachable,
    };
    const sparse_mlp_buffers = switch (model_buffers.layers[3].feed_forward) {
        .sparse => |value| value,
        .dense => unreachable,
    };
    try test_context.testLayer("prefill.layers.3.mlp.shared_experts", .{ .b, .s, .d }, sparse_mlp.shared_experts, sparse_mlp_buffers.shared_experts, .{ .absolute_tolerance = 7e-2 });

    const moe_backend = zml.moe.Backend.auto(platform, mdl.embed_tokens.weight.dtype()) catch |err| {
        if (err == error.UnimplementedMoEBackend) {
            std.log.warn("The {s} backend has no fused MoE implementation; ROCm is required for the MoE and end-to-end checks", .{@tagName(platform.target)});
            return;
        }
        return err;
    };
    const moe_parameters: zml.moe.Parameters = .init(.fromBackend(moe_backend, config.num_experts_per_tok, .silu));
    const moe_metadata: zml.moe.Metadata = .init(.fromBackend(moe_backend));
    try test_context.testMoe(
        "prefill.layers.3.mlp",
        sparse_mlp,
        sparse_mlp_buffers,
        moe_metadata,
        moe_parameters,
        .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 },
    );

    try test_context.testModel(mdl, model_buffers, config, moe_metadata, moe_parameters);
}

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations: zml.io.TensorStore.View,
    shardings: common.Shardings,

    fn testLayer(
        self: TestContext,
        name: []const u8,
        tagz: anytype,
        layer: anytype,
        layer_buffers: zml.Bufferized(@TypeOf(layer)),
        opts: zml.testing.CompareOpts,
    ) !void {
        std.log.info("Testing {s}", .{name});
        const input_key = try std.fmt.allocPrint(self.allocator, "{s}.in", .{name});
        defer self.allocator.free(input_key);
        const output_key = try std.fmt.allocPrint(self.allocator, "{s}.out", .{name});
        defer self.allocator.free(output_key);

        var input_buffer = try loadBuffer(self.allocator, self.io, self.platform, self.activations, input_key, .replicated);
        defer input_buffer.deinit();
        var expected = try loadBuffer(self.allocator, self.io, self.platform, self.activations, output_key, .replicated);
        defer expected.deinit();
        const input = zml.Tensor.fromShape(input_buffer.shape()).withTags(tagz);

        const all_shardings = self.shardings.all();
        const exe = try self.platform.compileFn(self.allocator, self.io, @TypeOf(layer).forward, .{ layer, input }, .{ .shardings = &all_shardings });
        defer exe.deinit();
        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ layer_buffers, input_buffer });
        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(args, &results);

        var actual = results.get(zml.Buffer);
        defer actual.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn testMoe(
        self: TestContext,
        name: []const u8,
        moe: model.Moe,
        moe_buffers: zml.Bufferized(model.Moe),
        metadata: zml.moe.Metadata,
        parameters: zml.moe.Parameters,
        opts: zml.testing.CompareOpts,
    ) !void {
        std.log.info("Testing {s}", .{name});
        const input_key = try std.fmt.allocPrint(self.allocator, "{s}.in", .{name});
        defer self.allocator.free(input_key);
        const output_key = try std.fmt.allocPrint(self.allocator, "{s}.out", .{name});
        defer self.allocator.free(output_key);
        var input_buffer = try loadBuffer(self.allocator, self.io, self.platform, self.activations, input_key, .replicated);
        defer input_buffer.deinit();
        var expected = try loadBuffer(self.allocator, self.io, self.platform, self.activations, output_key, .replicated);
        defer expected.deinit();

        const input = zml.Tensor.fromShape(input_buffer.shape()).withTags(.{ .b, .s, .d });
        const all_shardings = self.shardings.all();
        const exe = try self.platform.compileFn(self.allocator, self.io, model.Moe.forward, .{ moe, input, metadata, parameters }, .{ .shardings = &all_shardings });
        defer exe.deinit();
        var metadata_buffers = try metadata.initBuffer(self.io, self.platform);
        defer zml.moe.Metadata.deinitBuffer(&metadata_buffers);
        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ moe_buffers, input_buffer, metadata_buffers });
        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(args, &results);

        var actual = results.get(zml.Buffer);
        defer actual.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
    }

    fn testFullAttentionPrefillDecode(
        self: TestContext,
        attention: model.Attention,
        attention_buffers: zml.Bufferized(model.Attention),
        config: model.Config,
        layer_index: u32,
    ) !void {
        std.log.info("Testing layer {} full DSA attention prefill and decode", .{layer_index});
        var prefill_input = try loadBuffer(self.allocator, self.io, self.platform, self.activations, "prefill.layers.0.self_attn.in", .replicated);
        defer prefill_input.deinit();
        const prompt_length = prefill_input.shape().dim(1);
        const cache = model.Cache.init(4, 1, prompt_length + 1, config, attention.q_a_proj.weight.dtype());
        var zero_cache = try initZeroCache(self.allocator, self.io, self.platform, cache, self.shardings.model);
        defer model.Cache.deinitBuffers(&zero_cache);
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();
        var layer_index_buffer = try zml.Buffer.scalar(self.io, self.platform, layer_index, .u32);
        defer layer_index_buffer.deinit();

        const hidden = zml.Tensor.fromShape(prefill_input.shape()).withTags(.{ .b, .s, .d });
        const token_index: zml.Tensor = .init(.{}, .u32);
        const layer_index_tensor: zml.Tensor = .init(.{}, .u32);
        const all_shardings = self.shardings.all();
        const prefill_exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            forwardFullAttention,
            .{ attention, hidden, token_index, cache, layer_index_tensor },
            .{ .shardings = &all_shardings },
        );
        defer prefill_exe.deinit();
        var prefill_args = try prefill_exe.args(self.allocator);
        defer prefill_args.deinit(self.allocator);
        prefill_args.set(.{ attention_buffers, prefill_input, token_index_buffer, zero_cache, layer_index_buffer });
        var prefill_results = try prefill_exe.results(self.allocator);
        defer prefill_results.deinit(self.allocator);
        prefill_exe.call(prefill_args, &prefill_results);
        var prefill_output, var prefill_cache, var prefill_topk = prefill_results.get(struct { zml.Buffer, zml.Bufferized(model.Cache), zml.Buffer });
        defer prefill_output.deinit();
        defer model.Cache.deinitBuffers(&prefill_cache);
        defer prefill_topk.deinit();

        try self.expectBuffer("prefill.layers.0.self_attn.out.0", prefill_output, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("prefill.layers.0.self_attn.cache.k", prefill_cache.k, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
        try self.expectBuffer("prefill.layers.0.self_attn.cache.v", prefill_cache.v, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
        try self.expectBuffer("prefill.layers.0.self_attn.cache.indexer_k", prefill_cache.indexer_k, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });

        var decode_input = try loadBuffer(self.allocator, self.io, self.platform, self.activations, "decode.layers.0.self_attn.in", .replicated);
        defer decode_input.deinit();
        const decode_hidden = zml.Tensor.fromShape(decode_input.shape()).withTags(.{ .b, .s, .d });
        const decode_exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            forwardFullAttention,
            .{ attention, decode_hidden, token_index, cache, layer_index_tensor },
            .{ .shardings = &all_shardings },
        );
        defer decode_exe.deinit();
        var decode_token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(prompt_length)), .u32);
        defer decode_token_index_buffer.deinit();
        var decode_args = try decode_exe.args(self.allocator);
        defer decode_args.deinit(self.allocator);
        decode_args.set(.{ attention_buffers, decode_input, decode_token_index_buffer, prefill_cache, layer_index_buffer });
        var decode_results = try decode_exe.results(self.allocator);
        defer decode_results.deinit(self.allocator);
        decode_exe.call(decode_args, &decode_results);
        var decode_output, var decode_cache, var decode_topk = decode_results.get(struct { zml.Buffer, zml.Bufferized(model.Cache), zml.Buffer });
        defer decode_output.deinit();
        defer model.Cache.deinitBuffers(&decode_cache);
        defer decode_topk.deinit();

        try self.expectBuffer("decode.layers.0.self_attn.out.0", decode_output, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("decode.layers.0.self_attn.out.2", decode_topk, .exact_match);
        try self.expectBuffer("decode.layers.0.self_attn.cache.k", decode_cache.k, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
        try self.expectBuffer("decode.layers.0.self_attn.cache.v", decode_cache.v, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
        try self.expectBuffer("decode.layers.0.self_attn.cache.indexer_k", decode_cache.indexer_k, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
    }

    fn testSharedAttentionPrefill(
        self: TestContext,
        attention: model.Attention,
        attention_buffers: zml.Bufferized(model.Attention),
        config: model.Config,
        layer_index: u32,
    ) !void {
        std.log.info("Testing layer {} shared DSA attention prefill", .{layer_index});
        var input = try loadBuffer(self.allocator, self.io, self.platform, self.activations, "prefill.layers.3.self_attn.in", .replicated);
        defer input.deinit();
        var previous_topk = try loadBuffer(self.allocator, self.io, self.platform, self.activations, "prefill.layers.2.self_attn.out.2", .replicated);
        defer previous_topk.deinit();
        const prompt_length = input.shape().dim(1);
        const cache = model.Cache.init(4, 1, prompt_length + 1, config, attention.q_a_proj.weight.dtype());
        var zero_cache = try initZeroCache(self.allocator, self.io, self.platform, cache, self.shardings.model);
        defer model.Cache.deinitBuffers(&zero_cache);
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();
        var layer_index_buffer = try zml.Buffer.scalar(self.io, self.platform, layer_index, .u32);
        defer layer_index_buffer.deinit();

        const hidden = zml.Tensor.fromShape(input.shape()).withTags(.{ .b, .s, .d });
        const topk = zml.Tensor.fromShape(previous_topk.shape()).withTags(.{ .b, .s, .topk });
        const token_index: zml.Tensor = .init(.{}, .u32);
        const layer_index_tensor: zml.Tensor = .init(.{}, .u32);
        const all_shardings = self.shardings.all();
        const exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            forwardSharedAttention,
            .{ attention, hidden, token_index, cache, layer_index_tensor, topk },
            .{ .shardings = &all_shardings },
        );
        defer exe.deinit();
        var args = try exe.args(self.allocator);
        defer args.deinit(self.allocator);
        args.set(.{ attention_buffers, input, token_index_buffer, zero_cache, layer_index_buffer, previous_topk });
        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(args, &results);
        var output, var updated_cache, var output_topk = results.get(struct { zml.Buffer, zml.Bufferized(model.Cache), zml.Buffer });
        defer output.deinit();
        defer model.Cache.deinitBuffers(&updated_cache);
        defer output_topk.deinit();

        try self.expectBuffer("prefill.layers.3.self_attn.out.0", output, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("prefill.layers.3.self_attn.cache.k", updated_cache.k, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
        try self.expectBuffer("prefill.layers.3.self_attn.cache.v", updated_cache.v, .{ .absolute_tolerance = 5e-2, .relative_tolerance = 3e-2 });
        try self.expectBuffer("prefill.layers.3.self_attn.cache.indexer_k", updated_cache.indexer_k, .exact_match);
    }

    fn testModel(
        self: TestContext,
        mdl: model.Model,
        model_buffers: *model.Buffers,
        config: model.Config,
        metadata: zml.moe.Metadata,
        parameters: zml.moe.Parameters,
    ) !void {
        std.log.info("Testing four-layer prefill and cached decode", .{});
        var input_ids = try loadBuffer(self.allocator, self.io, self.platform, self.activations, "input_ids", .replicated);
        defer input_ids.deinit();
        var next_token = try loadBuffer(self.allocator, self.io, self.platform, self.activations, "next_token", .replicated);
        defer next_token.deinit();

        const prompt_length = input_ids.shape().dim(1);
        const max_sequence_length = prompt_length + 1;
        const cache = model.Cache.init(mdl.layers.len, 1, max_sequence_length, config, mdl.embed_tokens.weight.dtype());
        var cache_buffers = try initZeroCache(self.allocator, self.io, self.platform, cache, self.shardings.model);
        defer model.Cache.deinitBuffers(&cache_buffers);
        var metadata_buffers = try metadata.initBuffer(self.io, self.platform);
        defer zml.moe.Metadata.deinitBuffer(&metadata_buffers);

        const token_index: zml.Tensor = .init(.{}, .u32);
        const prefill_tokens = zml.Tensor.fromShape(input_ids.shape()).withTags(.{ .b, .s });
        const all_shardings = self.shardings.all();
        const prefill_exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            model.Model.forward,
            .{ mdl, prefill_tokens, token_index, cache, metadata, parameters },
            .{ .shardings = &all_shardings },
        );
        defer prefill_exe.deinit();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();
        var prefill_args = try prefill_exe.args(self.allocator);
        defer prefill_args.deinit(self.allocator);
        prefill_args.set(.{ model_buffers.*, input_ids, token_index_buffer, cache_buffers, metadata_buffers });
        var prefill_results = try prefill_exe.results(self.allocator);
        defer prefill_results.deinit(self.allocator);
        prefill_exe.call(prefill_args, &prefill_results);
        var prefill_logits, var prefill_cache = prefill_results.get(struct { zml.Buffer, zml.Bufferized(model.Cache) });
        defer prefill_logits.deinit();
        defer model.Cache.deinitBuffers(&prefill_cache);

        try self.expectBuffer("prefill.logits", prefill_logits, .{
            .absolute_tolerance = 2e-1,
            .relative_tolerance = 7e-2,
            .minimum_close_fraction = 0.99,
        });
        try self.expectBuffer("prefill.cache.k", prefill_cache.k, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("prefill.cache.v", prefill_cache.v, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("prefill.cache.indexer_k", prefill_cache.indexer_k, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });

        const decode_tokens = zml.Tensor.fromShape(next_token.shape()).withTags(.{ .b, .s });
        const decode_exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            model.Model.forward,
            .{ mdl, decode_tokens, token_index, cache, metadata, parameters },
            .{ .shardings = &all_shardings },
        );
        defer decode_exe.deinit();
        var decode_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(prompt_length)), .u32);
        defer decode_index_buffer.deinit();
        var decode_args = try decode_exe.args(self.allocator);
        defer decode_args.deinit(self.allocator);
        decode_args.set(.{ model_buffers.*, next_token, decode_index_buffer, prefill_cache, metadata_buffers });
        var decode_results = try decode_exe.results(self.allocator);
        defer decode_results.deinit(self.allocator);
        decode_exe.call(decode_args, &decode_results);
        var decode_logits, var decode_cache = decode_results.get(struct { zml.Buffer, zml.Bufferized(model.Cache) });
        defer decode_logits.deinit();
        defer model.Cache.deinitBuffers(&decode_cache);

        try self.expectBuffer("decode.logits", decode_logits, .{
            .absolute_tolerance = 2e-1,
            .relative_tolerance = 7e-2,
            .minimum_close_fraction = 0.99,
        });
        try self.expectBuffer("decode.cache.k", decode_cache.k, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("decode.cache.v", decode_cache.v, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
        try self.expectBuffer("decode.cache.indexer_k", decode_cache.indexer_k, .{ .absolute_tolerance = 1e-1, .relative_tolerance = 5e-2 });
    }

    fn expectBuffer(self: TestContext, key: []const u8, actual: zml.Buffer, opts: zml.testing.CompareOpts) !void {
        var expected = try loadBuffer(self.allocator, self.io, self.platform, self.activations, key, .replicated);
        defer expected.deinit();
        try zml.testing.expectClose(self.io, actual, expected, opts);
    }
};

fn forwardFullAttention(
    attention: model.Attention,
    hidden: zml.Tensor,
    token_index: zml.Tensor,
    cache: model.Cache,
    layer_index: zml.Tensor,
) struct { zml.Tensor, model.Cache, zml.Tensor } {
    return attention.forward(hidden, token_index, cache, layer_index, null);
}

fn forwardSharedAttention(
    attention: model.Attention,
    hidden: zml.Tensor,
    token_index: zml.Tensor,
    cache: model.Cache,
    layer_index: zml.Tensor,
    previous_topk: zml.Tensor,
) struct { zml.Tensor, model.Cache, zml.Tensor } {
    return attention.forward(hidden, token_index, cache, layer_index, previous_topk);
}

fn initZeroCache(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    cache: model.Cache,
    sharding: zml.Sharding,
) !zml.Bufferized(model.Cache) {
    return .{
        .k = try initZeroBuffer(allocator, io, platform, cache.k.shape(), sharding),
        .v = try initZeroBuffer(allocator, io, platform, cache.v.shape(), sharding),
        .indexer_k = try initZeroBuffer(allocator, io, platform, cache.indexer_k.shape(), sharding),
    };
}

fn initZeroBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    sharding: zml.Sharding,
) !zml.Buffer {
    const bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(bytes);
    @memset(bytes, 0);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, bytes);
}

fn loadBuffer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: zml.io.TensorStore.View,
    key: []const u8,
    sharding: zml.Sharding,
) !zml.Buffer {
    const shape = store.getShape(key) orelse {
        std.log.err("Missing activation {s}", .{key});
        return error.NotFound;
    };
    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);
    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}
