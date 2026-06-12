const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const step3p5flash = @import("step3_5flash.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,
    max_new_tokens: u32 = 60,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate the Step 3.5 Flash MoE layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>             Path to the model repository
        \\   --activations=<path>       Path to activation safetensors
        \\   --max-new-tokens=<number>  Tokens to generate after the captured prompt (default: 60)
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

    // Registry stores the memory of tensors
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    // Tensor store is a ZML representation of tensors - i.e., parent.child.leaf
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const shardings: common.Shardings = try .init(platform);

    var repo_model = try step3p5flash.LoadedModel.init(allocator, io, repo, store.view(), shardings);
    defer repo_model.deinit(allocator);

    // Loading bar (single global Progress). End it as soon as weights are
    // loaded so generation output isn't interleaved with progress redraws.
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    var progress_ended = false;
    defer if (!progress_ended) progress.end();

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    progress.end();
    progress_ended = true;

    try run(
        allocator,
        io,
        platform,
        args.activations,
        &store,
        shardings,
        &repo_model.inner,
        &model_buffers,
        repo,
        args.max_new_tokens,
    );
}

pub fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    model_store: *zml.io.TensorStore,
    shardings: common.Shardings,
    full_model: *const model.Model,
    full_buffers: *zml.Bufferized(model.Model),
    repo: std.Io.Dir,
    max_new_tokens: u32,
) !void {
    const text_model = &full_model.text_model;
    _ = text_model; // autofix
    const text_buffers = &full_buffers.text_model;
    _ = text_buffers; // autofix
    _ = model_store; // autofix
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    // // Per-layer transformer test: each layer fed the HF reference inputs.
    // // Only iterate the main transformer layers; trailing MTP layers are not
    // // captured in the HF fixture (the dumper sets num_hidden_layers = 45).
    // const num_main_layers = model.default_config.numMainLayers();
    // std.log.info("Transformer layer (isolated):", .{});
    // const shardings_array = shardings.all();
    // for (0..num_main_layers) |layer_idx| {
    //     runTransformerLayer(allocator, io, platform, model_store, &activation_store, &shardings_array, layer_idx) catch |err| {
    //         std.log.warn("skipping model.layers.{d}: {s}", .{ layer_idx, @errorName(err) });
    //     };
    // }

    // Chained-layer test: feed layer 0 input, run N layers, compare against
    // HF activation at layer N-1. We expect bf16 noise to grow like sqrt(N);
    // a sharp jump at a specific N localizes a real bug. Each step compiles
    // a separate graph that holds GPU scratch — keep this list small so the
    // generation graphs below have room.
    // const num_main_layers = model.default_config.numMainLayers();
    // const shardings_array = shardings.all();
    // std.log.info("Transformer layer (chained):", .{});
    // const chain_steps = [_]usize{ 1, 45 };
    // for (chain_steps) |n| {
    //     if (n > num_main_layers) continue;
    //     runLayerChain(allocator, io, platform, &activation_store, &shardings_array, text_model, text_buffers, n) catch |err| {
    //         std.log.warn("skipping chain[0..{d}]: {s}", .{ n, @errorName(err) });
    //     };
    // }

    // // Per-stage attention probe: drills into Attn.forwardTemp so we can see
    // // which intermediate (rope.q_in, rope.q_embed, cos, sin, attn, gated, out)
    // // diverges on a sliding-attn layer.
    // std.log.info("Attention stages (per layer):", .{});
    // const attn_probe_layers = [_]usize{ 0, 1, 2, 3, 4 };
    // for (attn_probe_layers) |li| {
    //     if (li >= num_main_layers) continue;
    //     runAttnStages(allocator, io, platform, model_store, &activation_store, &shardings_array, li) catch |err| {
    //         std.log.warn("skipping attn-stages layer {d}: {s}", .{ li, @errorName(err) });
    //     };
    // }

    // std.log.info("Full model:", .{});
    // try runFullTextModel(allocator, io, platform, &activation_store, shardings, text_model, text_buffers);

    // std.log.info("Argmax agreement (full model + lm_head):", .{});
    // try runArgmax(allocator, io, platform, &activation_store, shardings, full_model, full_buffers);

    std.log.info("Generation ({d} new tokens):", .{max_new_tokens});
    try runGenerate(allocator, io, platform, &activation_store, repo, shardings, full_model, full_buffers, max_new_tokens);
}

// Wraps a TransformerLayer and exposes each intermediate activation in the
// same order HF dumps them, so we can localize per-layer drift to a stage
// rather than only inspecting the layer output.
const DenseLayerStages = struct {
    layer: model.TransformerLayer,

    pub fn forward(
        self: @This(),
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
    ) struct {
        zml.Tensor, // input_layernorm.out.0
        zml.Tensor, // self_attn.out.0  (o_proj output, pre-residual)
        zml.Tensor, // post_attention_layernorm.out.0
        zml.Tensor, // mlp.out.0
        zml.Tensor, // layer .out.0 (final residual)
        model.KvCache,
    } {
        const attn_input = self.layer.input_layernorm.forward(x0, .d);
        const attn_delta, const new_kv = self.layer.attn.forward(attn_input, token_index, kv_cache);
        const x1 = x0.add(attn_delta);
        const ffn_input = self.layer.post_attention_layernorm.forward(x1, .d);
        const ffn_delta = switch (self.layer.ffn) {
            .mlp => |mlp| mlp.forward(ffn_input),
            .moe => unreachable,
        };
        const final = x1.add(ffn_delta);
        return .{ attn_input, attn_delta, ffn_input, ffn_delta, final, new_kv };
    }
};

const MoeLayerStages = struct {
    layer: model.TransformerLayer,

    pub fn forward(
        self: @This(),
        x0: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
    ) struct {
        zml.Tensor, // input_layernorm.out.0
        zml.Tensor, // self_attn.out.0
        zml.Tensor, // post_attention_layernorm.out.0
        zml.Tensor, // moe.out.0 (routed experts only)
        zml.Tensor, // share_expert.out.0 (shared MLP)
        zml.Tensor, // layer .out.0 (final residual)
        model.KvCache,
    } {
        const attn_input = self.layer.input_layernorm.forward(x0, .d);
        const attn_delta, const new_kv = self.layer.attn.forward(attn_input, token_index, kv_cache);
        const x1 = x0.add(attn_delta);
        const ffn_input = self.layer.post_attention_layernorm.forward(x1, .d);
        const moe_out, const share_out = switch (self.layer.ffn) {
            .mlp => unreachable,
            .moe => |m| .{ m.experts.forward(ffn_input), m.shared.forward(ffn_input) },
        };
        const ffn_delta = moe_out.add(share_out);
        const final = x1.add(ffn_delta);
        return .{ attn_input, attn_delta, ffn_input, moe_out, share_out, final, new_kv };
    }
};

fn runTransformerLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    layer_idx: usize,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer_idx});
    defer allocator.free(name);

    const model_partitions = shardings[0].numPartitionsForLogicalAxis(.model);
    const layer = try model.TransformerLayer.init(model_store.view().withPrefix(name), layer_idx, model_partitions);
    var layer_weights = try zml.io.load(model.TransformerLayer, &layer, allocator, io, platform, model_store, .{
        .parallelism = 1,
        .shardings = shardings,
        .dma_chunks = 2,
        .dma_chunk_size = 4096,
    });
    defer deinitBuffers(&layer_weights);

    const layer_view = activation_store.view().withPrefix(name);
    if (!layer_view.hasKey("in.0")) {
        std.log.warn("skipping {s}: no in.0 (hidden_states)", .{name});
        return;
    }
    if (!layer_view.hasKey("out.0")) {
        std.log.warn("skipping {s}: no out.0 (final hidden_states)", .{name});
        return;
    }
    const self_attn_name = try std.fmt.allocPrint(allocator, "{s}.self_attn", .{name});
    defer allocator.free(self_attn_name);
    const self_attn_view = activation_store.view().withPrefix(self_attn_name);
    if (!self_attn_view.hasKey("in.3")) {
        std.log.warn("skipping {s}: no self_attn.in.3 (cache_position)", .{name});
        return;
    }

    const hidden_states = layer_view.createTensor("in.0", .{ .b, .s, .d }, .replicated);
    const cache_position = self_attn_view.createTensor("in.3", null, .replicated);

    const batch_dim = hidden_states.dim(.b);
    const seq_dim = hidden_states.dim(.s);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = batch_dim,
        .k = seq_dim,
        .h = layer.attn.num_kv_heads,
        .hd = layer.attn.head_dim,
    }, hidden_states.dtype());
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, layer.attn.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape).atLayer(layer_idx);

    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const trace_args: TraceArgs = .{ hidden_states, cache_position, kv_traced };

    const is_moe = layer.ffn == .moe;

    const ActArgs = struct { zml.Tensor, zml.Tensor };
    var act_args: ActArgs = .{ hidden_states, cache_position };
    var act_buffers = try zml.io.load(ActArgs, &act_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&act_buffers);

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);

    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings[0], k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings[0], v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    // Per-stage tolerances. Keep close_fraction strict for individual stages —
    // each one is fed clean HF inputs, so genuine algorithmic drift will show
    // up as a low close_fraction even when mean error is tiny. The full layer
    // tolerance stays at 0.99 to match the previous report.
    const stage_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-2,
        .minimum_close_fraction = 0.99,
    };
    const layer_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-2,
        .minimum_close_fraction = 0.99,
    };

    const input_norm_name = try std.fmt.allocPrint(allocator, "{s}.input_layernorm", .{name});
    defer allocator.free(input_norm_name);
    const post_norm_name = try std.fmt.allocPrint(allocator, "{s}.post_attention_layernorm", .{name});
    defer allocator.free(post_norm_name);
    const mlp_name = try std.fmt.allocPrint(allocator, "{s}.mlp", .{name});
    defer allocator.free(mlp_name);
    const moe_name = try std.fmt.allocPrint(allocator, "{s}.moe", .{name});
    defer allocator.free(moe_name);
    const share_name = try std.fmt.allocPrint(allocator, "{s}.share_expert", .{name});
    defer allocator.free(share_name);

    if (is_moe) {
        const wrapper: MoeLayerStages = .{ .layer = layer };
        const exe = try platform.compile(allocator, io, wrapper, .forward, trace_args, .{ .shardings = shardings });
        defer exe.deinit();

        var exe_args = try exe.args(allocator);
        defer exe_args.deinit(allocator);
        var exe_results = try exe.results(allocator);
        defer exe_results.deinit(allocator);

        const wrapper_buffers: zml.Bufferized(MoeLayerStages) = .{ .layer = layer_weights };
        exe_args.set(.{ wrapper_buffers, .{ act_buffers[0], act_buffers[1], kv_buffers } });
        exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

        var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
        defer allocator.free(results);
        exe_results.fill(.{results});

        try compareSingleOutput(allocator, io, layer_view.withPrefix("input_layernorm"), "out.0", &results[0], input_norm_name, stage_opts);
        try compareSingleOutput(allocator, io, self_attn_view, "out.0", &results[1], self_attn_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view.withPrefix("post_attention_layernorm"), "out.0", &results[2], post_norm_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view.withPrefix("moe"), "out.0", &results[3], moe_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view.withPrefix("share_expert"), "out.0", &results[4], share_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view, "out.0", &results[5], name, layer_opts);
    } else {
        const wrapper: DenseLayerStages = .{ .layer = layer };
        const exe = try platform.compile(allocator, io, wrapper, .forward, trace_args, .{ .shardings = shardings });
        defer exe.deinit();

        var exe_args = try exe.args(allocator);
        defer exe_args.deinit(allocator);
        var exe_results = try exe.results(allocator);
        defer exe_results.deinit(allocator);

        const wrapper_buffers: zml.Bufferized(DenseLayerStages) = .{ .layer = layer_weights };
        exe_args.set(.{ wrapper_buffers, .{ act_buffers[0], act_buffers[1], kv_buffers } });
        exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

        var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
        defer allocator.free(results);
        exe_results.fill(.{results});

        try compareSingleOutput(allocator, io, layer_view.withPrefix("input_layernorm"), "out.0", &results[0], input_norm_name, stage_opts);
        try compareSingleOutput(allocator, io, self_attn_view, "out.0", &results[1], self_attn_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view.withPrefix("post_attention_layernorm"), "out.0", &results[2], post_norm_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view.withPrefix("mlp"), "out.0", &results[3], mlp_name, stage_opts);
        try compareSingleOutput(allocator, io, layer_view, "out.0", &results[4], name, layer_opts);
    }
}

const LayerChain = struct {
    layers: []const model.TransformerLayer,

    pub fn forward(
        self: LayerChain,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
    ) struct { zml.Tensor, model.KvCache } {
        var hidden = x;
        var kv = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden, kv = layer.forward(hidden, token_index, kv.atLayer(i));
        }
        return .{ hidden, kv };
    }
};

fn runLayerChain(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: *zml.io.TensorStore,
    shardings_array: []const zml.Sharding,
    text_model: *const model.TextModel,
    text_buffers: *zml.Bufferized(model.TextModel),
    num_layers: usize,
) !void {
    if (num_layers == 0 or num_layers > text_model.layers.len) return;

    const first_view = activation_store.view().withPrefix("model.layers.0");
    const first_attn_view = activation_store.view().withPrefix("model.layers.0.self_attn");
    if (!first_view.hasKey("in.0") or !first_attn_view.hasKey("in.3")) {
        std.log.warn("skipping chain[0..{d}]: missing layer-0 inputs", .{num_layers});
        return;
    }

    const ref_name = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{num_layers - 1});
    defer allocator.free(ref_name);
    const ref_view = activation_store.view().withPrefix(ref_name);
    if (!ref_view.hasKey("out.0")) {
        std.log.warn("skipping chain[0..{d}]: no {s}.out.0", .{ num_layers, ref_name });
        return;
    }

    const chain: LayerChain = .{ .layers = text_model.layers[0..num_layers] };

    const hidden_states = first_view.createTensor("in.0", .{ .b, .s, .d }, .replicated);
    const cache_position = first_attn_view.createTensor("in.3", null, .replicated);

    const attn0 = text_model.layers[0].attn;
    const model_partitions = shardings_array[0].numPartitionsForLogicalAxis(.model);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = hidden_states.dim(.b),
        .k = hidden_states.dim(.s),
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, hidden_states.dtype());
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn0.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape);

    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const trace_args: TraceArgs = .{ hidden_states, cache_position, kv_traced };
    const exe = try platform.compile(allocator, io, chain, .forward, trace_args, .{ .shardings = shardings_array });
    defer exe.deinit();

    const ActArgs = struct { zml.Tensor, zml.Tensor };
    var act_args: ActArgs = .{ hidden_states, cache_position };
    var act_buffers = try zml.io.load(ActArgs, &act_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&act_buffers);

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings_array[0], k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings_array[0], v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    const chain_buffers: zml.Bufferized(LayerChain) = .{ .layers = text_buffers.layers[0..num_layers] };
    exe_args.set(.{ chain_buffers, .{ act_buffers[0], act_buffers[1], kv_buffers } });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});

    const tag = try std.fmt.allocPrint(allocator, "chain[0..{d}]", .{num_layers});
    defer allocator.free(tag);
    // bf16 residual stream noise compounds like sqrt(N) * ULP. 1 bf16 ULP ~= 0.78%
    // relative, so ~5% relative is the physical floor after 40+ layers. Keep an
    // absolute floor to catch near-zero entries.
    try compareSingleOutput(allocator, io, ref_view, "out.0", &results[0], tag, .{
        .absolute_tolerance = 5e-2,
        .relative_tolerance = 5e-2,
        .minimum_close_fraction = 0.95,
    });
}

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

fn runFullTextModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activation_store: *zml.io.TensorStore,
    shardings: common.Shardings,
    text_model: *const model.TextModel,
    text_buffers: *zml.Bufferized(model.TextModel),
) !void {
    const model_view = activation_store.view().withPrefix("model");
    if (!model_view.hasKey("in.0") or !model_view.hasKey("out.0")) {
        std.log.warn("skipping full TextModel: missing in.0 / out.0", .{});
        return;
    }

    // 1. Build trace-time tensors for compile.
    const tokens_t = model_view.createTensor("in.0", .{ .b, .s }, .replicated);
    // token_index follows HF `cache_position` convention: rank-1 i32 of length S.
    const seq_len = tokens_t.dim(.s);
    const token_index_t = zml.Tensor.fromShape(zml.Shape.init(.{ .s = seq_len }, .i32));

    // 2. KV cache shape: (layer, b, k, h, hd).
    const attn0 = text_model.layers[0].attn;
    const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = tokens_t.dim(.b),
        .k = tokens_t.dim(.s),
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, .bf16);
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn0.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape);

    // 3. Compile forward.
    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const exe = try platform.compile(
        allocator,
        io,
        text_model.*,
        .forward,
        TraceArgs{ tokens_t, token_index_t, kv_traced },
        .{ .shardings = &shardings.all() },
    );
    defer exe.deinit();

    // 4. Build runtime buffers: tokens from store, cache_position = [0..S-1], zeroed KV.
    var token_args: struct { zml.Tensor } = .{tokens_t};
    var token_buffers = try zml.io.load(@TypeOf(token_args), &token_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&token_buffers);

    const token_index_shape = zml.Shape.init(.{ .s = seq_len }, .i32);
    const token_index_data = try allocator.alloc(i32, @intCast(seq_len));
    defer allocator.free(token_index_data);
    for (token_index_data, 0..) |*v, i| v.* = @intCast(i);
    var token_index_buf: zml.Buffer = try .fromBytes(io, platform, token_index_shape, .replicated, std.mem.sliceAsBytes(token_index_data));
    defer token_index_buf.deinit();

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    // 5. Run.
    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ text_buffers.*, .{ token_buffers[0], token_index_buf, kv_buffers } });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    // 6. Pull results[0] (hidden), compare with looser tolerance (error compounds across 48 layers).
    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});
    try compareSingleOutput(allocator, io, model_view, "out.0", &results[0], "model", .{
        .absolute_tolerance = 5e-2,
        .relative_tolerance = 5e-2,
        .minimum_close_fraction = 0.95,
    });
}

// End-to-end argmax agreement check: runs the full text model + lm_head and
// compares argmax(logits) token-for-token against the argmax of HF's captured
// `lm_head.out.0`. This is the gold-standard signal for "does the model
// generate the right next token?" — bf16 ULP noise on logits almost never
// flips an argmax when the top-1 has any reasonable margin.
const ArgmaxWrapper = struct {
    full: model.Model,

    pub fn forward(
        self: ArgmaxWrapper,
        tokens_raw: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
    ) struct { zml.Tensor, model.KvCache } {
        const tokens = tokens_raw.withPartialTags(.{.s});
        const hidden, const new_kv = self.full.text_model.forward(tokens, token_index, kv_cache);
        const logits = self.full.lm_head.forward(hidden.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const am = logits.argMax(.voc);
        return .{ am.indices, new_kv };
    }
};

fn runArgmax(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activation_store: *zml.io.TensorStore,
    shardings: common.Shardings,
    full_model: *const model.Model,
    full_buffers: *zml.Bufferized(model.Model),
) !void {
    const model_view = activation_store.view().withPrefix("model");
    const lm_view = activation_store.view().withPrefix("lm_head");
    if (!model_view.hasKey("in.0") or !lm_view.hasKey("out.0")) {
        std.log.warn("skipping argmax: missing model.in.0 / lm_head.out.0", .{});
        return;
    }

    // Trace-time tensors.
    const tokens_t = model_view.createTensor("in.0", .{ .b, .s }, .replicated);
    const seq_len = tokens_t.dim(.s);
    const token_index_t = zml.Tensor.fromShape(zml.Shape.init(.{ .s = seq_len }, .i32));

    // KV cache shape (same construction as runFullTextModel).
    const attn0 = full_model.text_model.layers[0].attn;
    const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = tokens_t.dim(.b),
        .k = tokens_t.dim(.s),
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, .bf16);
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn0.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape);

    const wrapper: ArgmaxWrapper = .{ .full = full_model.* };
    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const exe = try platform.compile(
        allocator,
        io,
        wrapper,
        .forward,
        TraceArgs{ tokens_t, token_index_t, kv_traced },
        .{ .shardings = &shardings.all() },
    );
    defer exe.deinit();

    // Runtime buffers.
    var token_args: struct { zml.Tensor } = .{tokens_t};
    var token_buffers = try zml.io.load(@TypeOf(token_args), &token_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&token_buffers);

    const token_index_shape = zml.Shape.init(.{ .s = seq_len }, .i32);
    const token_index_data = try allocator.alloc(i32, @intCast(seq_len));
    defer allocator.free(token_index_data);
    for (token_index_data, 0..) |*v, i| v.* = @intCast(i);
    var token_index_buf: zml.Buffer = try .fromBytes(io, platform, token_index_shape, .replicated, std.mem.sliceAsBytes(token_index_data));
    defer token_index_buf.deinit();

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    const wrapper_buffers: zml.Bufferized(ArgmaxWrapper) = .{ .full = full_buffers.* };
    exe_args.set(.{ wrapper_buffers, .{ token_buffers[0], token_index_buf, kv_buffers } });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});

    // Pull our argmax (i32, shape [B, S]).
    const our_slice = try results[0].toSliceAlloc(allocator, io);
    defer our_slice.free(allocator);
    const our_i32: []const i32 = @alignCast(std.mem.bytesAsSlice(i32, our_slice.data()));

    // Load HF logits and compute argmax on CPU.
    const ref_shape = lm_view.getShape("out.0") orelse return;
    const num_tokens: usize = @intCast(our_i32.len);
    const dims = ref_shape.dims();
    if (dims.len == 0) return;
    const voc: usize = @intCast(dims[dims.len - 1]);
    var ref_buffer: [4096]u8 = undefined;
    const ref_slice: zml.Slice = try .alloc(allocator, ref_shape);
    defer ref_slice.free(allocator);
    var reader = try lm_view.getReader("out.0", io, &ref_buffer);
    defer reader.deinit();
    try reader.interface.readSliceAll(ref_slice.data());

    const hf_argmax = try allocator.alloc(i32, num_tokens);
    defer allocator.free(hf_argmax);

    const elem_size = ref_shape.dtype().sizeOf();
    if (ref_shape.dtype() == .bf16) {
        const bf16_view: []const zml.floats.BFloat16 = @alignCast(std.mem.bytesAsSlice(zml.floats.BFloat16, ref_slice.data()));
        for (0..num_tokens) |t| {
            const base = t * voc;
            var best_idx: usize = 0;
            var best_val: f32 = bf16_view[base].toF32();
            for (1..voc) |v| {
                const val = bf16_view[base + v].toF32();
                if (val > best_val) {
                    best_val = val;
                    best_idx = v;
                }
            }
            hf_argmax[t] = @intCast(best_idx);
        }
    } else if (ref_shape.dtype() == .f32) {
        const f32_view: []const f32 = @alignCast(std.mem.bytesAsSlice(f32, ref_slice.data()));
        for (0..num_tokens) |t| {
            const base = t * voc;
            var best_idx: usize = 0;
            var best_val: f32 = f32_view[base];
            for (1..voc) |v| {
                const val = f32_view[base + v];
                if (val > best_val) {
                    best_val = val;
                    best_idx = v;
                }
            }
            hf_argmax[t] = @intCast(best_idx);
        }
    } else {
        std.log.warn("argmax: unsupported lm_head dtype {s} (elem={d})", .{ @tagName(ref_shape.dtype()), elem_size });
        return;
    }

    var matches: usize = 0;
    for (our_i32, hf_argmax) |o, h| {
        if (o == h) matches += 1;
    }
    const accuracy = @as(f32, @floatFromInt(matches)) / @as(f32, @floatFromInt(num_tokens));
    if (matches == num_tokens) {
        std.log.info("ok lm_head argmax: {d}/{d} tokens match (100%)", .{ matches, num_tokens });
    } else {
        std.log.warn("lm_head argmax: {d}/{d} tokens match ({d:.2}%)", .{ matches, num_tokens, accuracy * 100 });
        // Show first few disagreements for debugging.
        var shown: usize = 0;
        for (our_i32, hf_argmax, 0..) |o, h, t| {
            if (o != h and shown < 8) {
                std.log.warn("  token[{d}]: ours={d}, hf={d}", .{ t, o, h });
                shown += 1;
            }
        }
    }
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !zml.tokenizer.Tokenizer {
    const file = try dir.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    return try .fromBytes(allocator, bytes);
}

// Autoregressive generation using two compiled graphs:
//   - prefill: tokens [1, S_prompt], token_index [S_prompt], KV [k=cache_max]
//   - decode:  tokens [1, 1],        token_index [1],        KV [k=cache_max]
// Greedy (argmax) sampling. Reads the prompt from `model.in.0` so we generate
// continuations of the same captured prompt that the activation fixture used.
fn runGenerate(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activation_store: *zml.io.TensorStore,
    repo: std.Io.Dir,
    shardings: common.Shardings,
    full_model: *const model.Model,
    full_buffers: *zml.Bufferized(model.Model),
    max_new_tokens: u32,
) !void {
    // 1. Load tokenizer + a streaming decoder.
    var tokenizer = loadTokenizer(allocator, io, repo) catch |err| {
        std.log.warn("skipping generate: tokenizer load failed: {s}", .{@errorName(err)});
        return;
    };
    defer tokenizer.deinit();
    var detok = try tokenizer.decoder();
    defer detok.deinit();
    var detok_buf: [4096]u8 = undefined;

    // 2. Discover prompt shape from activations.
    const model_view = activation_store.view().withPrefix("model");
    const prompt_shape = model_view.getShape("in.0") orelse {
        std.log.warn("skipping generate: no model.in.0", .{});
        return;
    };
    if (prompt_shape.rank() != 2) {
        std.log.warn("skipping generate: model.in.0 rank={d}, want 2", .{prompt_shape.rank()});
        return;
    }
    const prompt_len: i64 = prompt_shape.dim(1);
    const cache_max: i64 = prompt_len + @as(i64, @intCast(max_new_tokens));

    // 3. Build KV cache shape, shared between prefill and decode.
    const attn0 = full_model.text_model.layers[0].attn;
    const model_partitions = shardings.model.numPartitionsForLogicalAxis(.model);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = 1,
        .k = cache_max,
        .h = attn0.num_kv_heads,
        .hd = attn0.head_dim,
    }, .bf16);
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn0.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape);

    // 4. Compile prefill at [B=1, S=prompt_len].
    const prefill_tokens_t = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .s = prompt_len }, .i32));
    const prefill_token_index_t = zml.Tensor.fromShape(zml.Shape.init(.{ .s = prompt_len }, .i32));
    const wrapper: ArgmaxWrapper = .{ .full = full_model.* };
    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const prefill_exe = try platform.compile(
        allocator,
        io,
        wrapper,
        .forward,
        TraceArgs{ prefill_tokens_t, prefill_token_index_t, kv_traced },
        .{ .shardings = &shardings.all() },
    );
    defer prefill_exe.deinit();

    // 5. Compile decode at [B=1, S=1].
    const decode_tokens_t = zml.Tensor.fromShape(zml.Shape.init(.{ .b = 1, .s = 1 }, .i32));
    const decode_token_index_t = zml.Tensor.fromShape(zml.Shape.init(.{ .s = 1 }, .i32));
    const decode_exe = try platform.compile(
        allocator,
        io,
        wrapper,
        .forward,
        TraceArgs{ decode_tokens_t, decode_token_index_t, kv_traced },
        .{ .shardings = &shardings.all() },
    );
    defer decode_exe.deinit();

    // 6. Persistent KV cache buffers (zeroed).
    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);
    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings.model, v_init);
    defer v_buf.deinit();
    var kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    // 7. Load prompt tokens from activations into a CPU i32 buffer.
    const prompt_tokens = try allocator.alloc(i32, @intCast(prompt_len));
    defer allocator.free(prompt_tokens);
    {
        const ref_slice: zml.Slice = try .alloc(allocator, prompt_shape);
        defer ref_slice.free(allocator);
        var rbuf: [4096]u8 = undefined;
        var reader = try model_view.getReader("in.0", io, &rbuf);
        defer reader.deinit();
        try reader.interface.readSliceAll(ref_slice.data());
        switch (prompt_shape.dtype()) {
            .i64 => {
                const s: []align(1) const i64 = std.mem.bytesAsSlice(i64, ref_slice.data());
                for (s, prompt_tokens) |v, *out| out.* = @intCast(v);
            },
            .u64 => {
                const s: []align(1) const u64 = std.mem.bytesAsSlice(u64, ref_slice.data());
                for (s, prompt_tokens) |v, *out| out.* = @intCast(v);
            },
            .i32 => {
                const s: []align(1) const i32 = std.mem.bytesAsSlice(i32, ref_slice.data());
                @memcpy(prompt_tokens, s);
            },
            .u32 => {
                const s: []align(1) const u32 = std.mem.bytesAsSlice(u32, ref_slice.data());
                for (s, prompt_tokens) |v, *out| out.* = @intCast(v);
            },
            else => {
                std.log.warn("generate: unsupported prompt dtype {s}", .{@tagName(prompt_shape.dtype())});
                return;
            },
        }
    }

    // 8. Stdout writer.
    var stdout = std.Io.File.stdout().writerStreaming(io, &.{});
    try stdout.interface.writeAll("\n=== prompt ===\n");
    for (prompt_tokens) |tid| {
        const t = try detok.feedOne(@intCast(tid), &detok_buf);
        try stdout.interface.writeAll(t);
    }
    try stdout.interface.writeAll("\n=== completion ===\n");
    try stdout.interface.flush();

    // 9. Build prefill input buffers.
    const prefill_token_index_data = try allocator.alloc(i32, @intCast(prompt_len));
    defer allocator.free(prefill_token_index_data);
    for (prefill_token_index_data, 0..) |*v, i| v.* = @intCast(i);

    var prefill_tokens_buf: zml.Buffer = try .fromBytes(io, platform, prefill_tokens_t.shape(), .replicated, std.mem.sliceAsBytes(prompt_tokens));
    defer prefill_tokens_buf.deinit();
    var prefill_token_index_buf: zml.Buffer = try .fromBytes(io, platform, prefill_token_index_t.shape(), .replicated, std.mem.sliceAsBytes(prefill_token_index_data));
    defer prefill_token_index_buf.deinit();

    // 10. Run prefill: fills argmax + KV in place. ZML donates input buffers
    // to outputs, so the returned KV handles ARE our kv_buffers handles — never
    // deinit them mid-loop or we'd double-free at scope exit.
    var prefill_argmax: zml.Buffer = try .uninitialized(
        io,
        platform,
        zml.Shape.init(.{ .b = 1, .s = prompt_len }, .i32).withReplicatedPartitioning(),
        .replicated,
        .{},
    );
    defer prefill_argmax.deinit();
    {
        var exe_args = try prefill_exe.args(allocator);
        defer exe_args.deinit(allocator);
        var exe_results = try prefill_exe.results(allocator);
        defer exe_results.deinit(allocator);
        const wbuf: zml.Bufferized(ArgmaxWrapper) = .{ .full = full_buffers.* };
        exe_args.set(.{ wbuf, .{ prefill_tokens_buf, prefill_token_index_buf, kv_buffers } });
        prefill_exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });
        exe_results.fill(.{ &prefill_argmax, &kv_buffers });
    }

    // 11. Read the last-position argmax to seed decode.
    var next_token: i32 = blk: {
        const am_slice = try prefill_argmax.toSliceAlloc(allocator, io);
        defer am_slice.free(allocator);
        const am_i32: []align(1) const i32 = std.mem.bytesAsSlice(i32, am_slice.data());
        break :blk am_i32[@intCast(prompt_len - 1)];
    };

    // 12. Decode loop. Reuse a single argmax buffer + persistent KV across steps.
    var decode_argmax: zml.Buffer = try .uninitialized(
        io,
        platform,
        zml.Shape.init(.{ .b = 1, .s = 1 }, .i32).withReplicatedPartitioning(),
        .replicated,
        .{},
    );
    defer decode_argmax.deinit();

    var cur_pos: i64 = prompt_len;
    var step: u32 = 0;
    while (step < max_new_tokens) : (step += 1) {
        // Emit the token we just predicted.
        const t = try detok.feedOne(@intCast(next_token), &detok_buf);
        std.log.info("token: {}", .{next_token});
        try stdout.interface.writeAll(t);
        try stdout.interface.flush();

        if (step + 1 == max_new_tokens) break;

        // Build decode inputs for next step.
        var tok_arr: [1]i32 = .{next_token};
        var idx_arr: [1]i32 = .{@intCast(cur_pos)};
        var dec_tokens_buf: zml.Buffer = try .fromBytes(io, platform, decode_tokens_t.shape(), .replicated, std.mem.sliceAsBytes(&tok_arr));
        defer dec_tokens_buf.deinit();
        var dec_token_index_buf: zml.Buffer = try .fromBytes(io, platform, decode_token_index_t.shape(), .replicated, std.mem.sliceAsBytes(&idx_arr));
        defer dec_token_index_buf.deinit();

        var exe_args = try decode_exe.args(allocator);
        defer exe_args.deinit(allocator);
        var exe_results = try decode_exe.results(allocator);
        defer exe_results.deinit(allocator);
        const wbuf: zml.Bufferized(ArgmaxWrapper) = .{ .full = full_buffers.* };
        exe_args.set(.{ wbuf, .{ dec_tokens_buf, dec_token_index_buf, kv_buffers } });
        decode_exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });
        exe_results.fill(.{ &decode_argmax, &kv_buffers });

        // Read argmax (shape [1, 1]).
        const am_slice = try decode_argmax.toSliceAlloc(allocator, io);
        defer am_slice.free(allocator);
        const am_i32: []align(1) const i32 = std.mem.bytesAsSlice(i32, am_slice.data());
        next_token = am_i32[0];
        cur_pos += 1;
    }
    try stdout.interface.writeAll("\n");
    const tail = try detok.finalize(&detok_buf);
    try stdout.interface.writeAll(tail);
    try stdout.interface.flush();
}

// Wraps Attn.forwardTemp to expose per-stage intermediates (q/k pre/post rope,
// cos/sin, attention output, gated output, final o_proj) so we can pinpoint
// which sub-step of self-attention is responsible for the divergence on
// sliding-attention layers.
const AttnStages = struct {
    attn: model.Attn,

    pub fn forward(
        self: @This(),
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
    ) struct {
        zml.Tensor, // q_pre_rope_hf  -> self_attn.rope.q_in
        zml.Tensor, // k_pre_rope_hf  -> self_attn.rope.k_in
        zml.Tensor, // cos            -> self_attn.rope.cos
        zml.Tensor, // sin            -> self_attn.rope.sin
        zml.Tensor, // q_rope_hf      -> self_attn.rope.q_embed
        zml.Tensor, // k_rope_hf      -> self_attn.rope.k_embed
        zml.Tensor, // attn           -> self_attn.attn
        zml.Tensor, // gated          -> self_attn.gated
        zml.Tensor, // out            -> self_attn.out.0
        model.KvCache,
    } {
        const stages, const new_kv = self.attn.forwardTemp(x, token_index, kv_cache);
        return .{
            stages.q_pre_rope_hf,
            stages.k_pre_rope_hf,
            stages.cos,
            stages.sin,
            stages.q_rope_hf,
            stages.k_rope_hf,
            stages.attn,
            stages.gated,
            stages.out,
            new_kv,
        };
    }
};

fn runAttnStages(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_store: *zml.io.TensorStore,
    activation_store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    layer_idx: usize,
) !void {
    const name = try std.fmt.allocPrint(allocator, "model.layers.{d}", .{layer_idx});
    defer allocator.free(name);
    const self_attn_name = try std.fmt.allocPrint(allocator, "{s}.self_attn", .{name});
    defer allocator.free(self_attn_name);
    const rope_name = try std.fmt.allocPrint(allocator, "{s}.rope", .{self_attn_name});
    defer allocator.free(rope_name);

    const model_partitions = shardings[0].numPartitionsForLogicalAxis(.model);
    const attn = try model.Attn.init(model_store.view().withPrefix(self_attn_name), layer_idx, model_partitions);
    var attn_buffers = try zml.io.load(model.Attn, &attn, allocator, io, platform, model_store, .{
        .parallelism = 1,
        .shardings = shardings,
        .dma_chunks = 2,
        .dma_chunk_size = 4096,
    });
    defer deinitBuffers(&attn_buffers);

    const attn_view = activation_store.view().withPrefix(self_attn_name);
    if (!attn_view.hasKey("in.0") or !attn_view.hasKey("in.3")) {
        std.log.warn("skipping {s}: missing in.0/in.3", .{self_attn_name});
        return;
    }

    const hidden = attn_view.createTensor("in.0", .{ .b, .s, .d }, .replicated);
    const cache_position = attn_view.createTensor("in.3", null, .replicated);

    const batch_dim = hidden.dim(.b);
    const seq_dim = hidden.dim(.s);
    const kv_shape_unsharded = zml.Shape.init(.{
        .layer = @as(i64, @intCast(model.default_config.numMainLayers())),
        .b = batch_dim,
        .k = seq_dim,
        .h = attn.num_kv_heads,
        .hd = attn.head_dim,
    }, hidden.dtype());
    const kv_shape = model.partitionKvCacheShape(kv_shape_unsharded, attn.num_kv_heads, model_partitions);
    const kv_traced = model.KvCache.init(kv_shape).atLayer(layer_idx);

    const TraceArgs = struct { zml.Tensor, zml.Tensor, model.KvCache };
    const trace_args: TraceArgs = .{ hidden, cache_position, kv_traced };

    const ActArgs = struct { zml.Tensor, zml.Tensor };
    var act_args: ActArgs = .{ hidden, cache_position };
    var act_buffers = try zml.io.load(ActArgs, &act_args, allocator, io, platform, activation_store, .auto);
    defer deinitBuffers(&act_buffers);

    const kv_bytes = kv_shape.byteSize();
    const k_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(k_init);
    @memset(k_init, 0);
    const v_init = try allocator.alloc(u8, kv_bytes);
    defer allocator.free(v_init);
    @memset(v_init, 0);

    var k_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings[0], k_init);
    defer k_buf.deinit();
    var v_buf: zml.Buffer = try .fromBytes(io, platform, kv_shape, shardings[0], v_init);
    defer v_buf.deinit();
    const kv_buffers: model.KvCache.Buffer = .{ .k = k_buf, .v = v_buf };

    const wrapper: AttnStages = .{ .attn = attn };
    const exe = try platform.compile(allocator, io, wrapper, .forward, trace_args, .{ .shardings = shardings });
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    const wrapper_buffers: zml.Bufferized(AttnStages) = .{ .attn = attn_buffers };
    exe_args.set(.{ wrapper_buffers, .{ act_buffers[0], act_buffers[1], kv_buffers } });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});

    const opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-2,
        .minimum_close_fraction = 0.99,
    };

    const rope_view = activation_store.view().withPrefix(rope_name);

    try compareSingleOutput(allocator, io, rope_view, "q_in", &results[0], rope_name, opts);
    try compareSingleOutput(allocator, io, rope_view, "k_in", &results[1], rope_name, opts);
    try compareSingleOutput(allocator, io, rope_view, "cos", &results[2], rope_name, opts);
    try compareSingleOutput(allocator, io, rope_view, "sin", &results[3], rope_name, opts);
    try compareSingleOutput(allocator, io, rope_view, "q_embed", &results[4], rope_name, opts);
    try compareSingleOutput(allocator, io, rope_view, "k_embed", &results[5], rope_name, opts);
    try compareSingleOutput(allocator, io, attn_view, "attn", &results[6], self_attn_name, opts);
    try compareSingleOutput(allocator, io, attn_view, "gated", &results[7], self_attn_name, opts);
    try compareSingleOutput(allocator, io, attn_view, "out.0", &results[8], self_attn_name, opts);
}

fn deinitBuffers(bufs: anytype) void {
    zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, bufs);
}

// RmsNorm.forward takes a comptime axis, so wrap it to fix axis=.d for compilation.
const RmsNormForD = struct {
    inner: model.RmsNorm,

    pub fn forward(self: RmsNormForD, x: zml.Tensor) zml.Tensor {
        return self.inner.forward(x, .d);
    }
};

fn runRmsNormSynthetic(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.Sharding,
) !void {
    const eps: f32 = 1e-5;
    const d: i64 = 4;

    const weight_shape = zml.Shape.init(.{ .d = d }, .f32);
    const x_shape = zml.Shape.init(.{ .b = 1, .s = 1, .d = d }, .f32);

    const wrapper_t: RmsNormForD = .{ .inner = .{
        .weight = zml.Tensor.fromShape(weight_shape),
        .eps = eps,
    } };
    const x_t = zml.Tensor.fromShape(x_shape);

    const exe = try platform.compileFn(allocator, io, RmsNormForD.forward, .{ wrapper_t, x_t }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    const weight_data = [_]f32{ 0.0, 1.0, 0.0, -1.0 };
    const x_data = [_]f32{ 1.0, 2.0, 3.0, 4.0 };
    const expected = [_]f32{ 0.3651, 1.4606, 1.0954, 0.0 };

    var weight_buf: zml.Buffer = try .fromBytes(io, platform, weight_shape, .replicated, std.mem.sliceAsBytes(&weight_data));
    defer weight_buf.deinit();
    var x_buf: zml.Buffer = try .fromBytes(io, platform, x_shape, .replicated, std.mem.sliceAsBytes(&x_data));
    defer x_buf.deinit();

    const wrapper_buffers: zml.Bufferized(RmsNormForD) = .{ .inner = .{ .weight = weight_buf } };

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe_args.set(.{ wrapper_buffers, x_buf });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var out_buf = exe_results.get(zml.Buffer);
    defer out_buf.deinit();

    var expected_slice: zml.Slice = try .alloc(allocator, x_shape);
    defer expected_slice.free(allocator);
    @memcpy(expected_slice.data()[0 .. expected.len * @sizeOf(f32)], std.mem.sliceAsBytes(&expected));

    var got_slice = try out_buf.toSliceAlloc(allocator, io);
    defer got_slice.free(allocator);

    try zml.testing.expectClose(io, expected_slice, got_slice, .{ .absolute_tolerance = 1e-3 });
    std.log.info("ok RmsNorm synthetic matches", .{});
}

// When layers return multiple outputs, zml.testing.testLayer no longer suffices, hence the following function.
fn compareSingleOutput(
    allocator: std.mem.Allocator,
    io: std.Io,
    view: zml.io.TensorStore.View,
    subkey: []const u8,
    got: *zml.Buffer,
    name: []const u8,
    opts: zml.testing.CompareOpts,
) !void {
    const ref_shape = view.getShape(subkey) orelse {
        std.log.warn("{s}.{s}: no reference shape", .{ name, subkey });
        return;
    };

    const got_shape = got.shape();
    var ref_count: i64 = 1;
    for (ref_shape.dims()) |d| ref_count *= d;
    var got_count: i64 = 1;
    for (got_shape.dims()) |d| got_count *= d;
    if (ref_count != got_count) {
        std.log.warn("{s}.{s}: shape mismatch ref={f} ours={f}, skipping", .{ name, subkey, ref_shape, got_shape });
        return;
    }

    var reader_buffer: [4096]u8 = undefined;
    const expected_slice: zml.Slice = try .alloc(allocator, ref_shape);
    defer expected_slice.free(allocator);
    var reader = try view.getReader(subkey, io, &reader_buffer);
    defer reader.deinit();
    try reader.interface.readSliceAll(expected_slice.data());

    const got_slice = try got.toSliceAlloc(allocator, io);
    defer got_slice.free(allocator);

    zml.testing.expectClose(io, expected_slice, got_slice, opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            std.log.warn("X {s}.{s} doesn't match", .{ name, subkey });
            return;
        },
        else => return err,
    };
    std.log.info("ok {s}.{s} matches", .{ name, subkey });
}
