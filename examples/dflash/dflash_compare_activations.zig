const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const dflash = @import("dflash_model.zig");

comptime {
    @setEvalBranchQuota(10_000);
}

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,
    absolute_tolerance: f32 = 2e-2,
    relative_tolerance: f32 = 2e-2,
    minimum_close_fraction: f32 = 0.999,

    pub const help =
        \\Use dflash_compare_activations --model=<path> --activations=<path> [options]
        \\
        \\Validate the Zig DFlash implementation against activation safetensors
        \\created by test/extract_reference.py.
        \\
        \\Options:
        \\  --model=<path>                       Path to the DFlash model repository
        \\  --activations=<path>                 Path to activation safetensors
        \\  --absolute-tolerance=<float>         Default: 2e-2
        \\  --relative-tolerance=<float>         Default: 2e-2
        \\  --minimum-close-fraction=<float>     Default: 0.999
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    if (init.environ_map.get("BUILD_WORKING_DIRECTORY")) |build_working_directory| {
        var working_dir = try std.Io.Dir.openDirAbsolute(init.io, build_working_directory, .{});
        defer working_dir.close(init.io);
        try std.process.setCurrentDir(init.io, working_dir);
    }

    const args = zml.stdx.flags.parse(init.minimal.args, Args);
    const opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = args.absolute_tolerance,
        .relative_tolerance = args.relative_tolerance,
        .minimum_close_fraction = args.minimum_close_fraction,
    };

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    const parsed_config = try common.parseConfig(dflash.Config, allocator, io, repo);
    defer parsed_config.deinit();

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model = try dflash.Model.init(allocator, store.view(), parsed_config.value);
    defer model.deinit(allocator);

    const shardings: common.Shardings = try .init(platform);
    const all_shardings = shardings.all();

    std.log.info("Loading DFlash weights...", .{});
    var model_buffers = try zml.io.load(dflash.Model, &model, allocator, io, platform, &store, .{
        .parallelism = 16,
        .shardings = &all_shardings,
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
    });
    defer dflash.Model.unloadBuffers(&model_buffers, allocator);

    var activation_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, args.activations);
    defer activation_registry.deinit();
    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &activation_registry);
    defer activation_store.deinit();

    var ctx: TestContext = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .activation_store = activation_store.view(),
        .model = model,
        .model_buffers = &model_buffers,
        .sharding = platform.replicated_sharding,
        .model_sharding = shardings.model,
        .opts = opts,
    };

    var failures: usize = 0;
    ctx.testFullModel() catch |err| {
        failures += 1;
        std.log.err("Full DFlash forward failed: {}", .{err});
    };
    for (model.layers, 0..) |layer, i| {
        ctx.testLayer(i, layer, model_buffers.layers[i]) catch |err| {
            failures += 1;
            std.log.err("DFlash layer {d} failed: {}", .{ i, err });
        };
    }
    if (failures != 0) return error.TestUnexpectedResult;
}

fn runFullModelF32(
    model: dflash.Model,
    target_hidden: zml.Tensor,
    noise_embedding: zml.Tensor,
    position_ids: zml.Tensor,
) zml.Tensor {
    return model.forwardF32(target_hidden, noise_embedding, position_ids);
}

fn runLayerF32(
    layer: dflash.DecoderLayer,
    hidden_: zml.Tensor,
    target_hidden_: zml.Tensor,
    position_ids: zml.Tensor,
) zml.Tensor {
    const hidden = hidden_.convert(layer.input_layernorm.weight.dtype());
    const target_hidden = target_hidden_.convert(layer.self_attn.k_proj.weight.dtype());
    return layer.forward(hidden, target_hidden, position_ids).convert(.f32);
}

const LayerDebugTensors = struct {
    input_layernorm_out: zml.Tensor,
    q_proj_out: zml.Tensor,
    k_proj_out: zml.Tensor,
    v_proj_out: zml.Tensor,
    q_norm_out: zml.Tensor,
    k_norm_out: zml.Tensor,
    q_rope_out: zml.Tensor,
    k_rope_out: zml.Tensor,
    v_out: zml.Tensor,
    sdpa_q_grouped: zml.Tensor,
    sdpa_k_scaled: zml.Tensor,
    sdpa_v_grouped: zml.Tensor,
    sdpa_logits_grouped: zml.Tensor,
    sdpa_weights_grouped: zml.Tensor,
    sdpa_grouped_out: zml.Tensor,
    sdpa_grouped_merged_out: zml.Tensor,
    sdpa_weights_grouped_f32: zml.Tensor,
    sdpa_grouped_out_f32: zml.Tensor,
    sdpa_grouped_merged_out_f32: zml.Tensor,
    sdpa_grouped_out_replay_bf16: zml.Tensor,
    sdpa_grouped_out_replay_f32: zml.Tensor,
    sdpa_zml_pre_transpose: zml.Tensor,
    sdpa_zml_transposed: zml.Tensor,
    sdpa_zml_merged_out: zml.Tensor,
    sdpa_out: zml.Tensor,
    sdpa_merged_out: zml.Tensor,
    o_proj_out: zml.Tensor,
    post_attn_residual_out: zml.Tensor,
    post_attention_layernorm_out: zml.Tensor,
    mlp_out: zml.Tensor,
    out: zml.Tensor,
};

const LayerDebugBuffers = zml.Bufferized(LayerDebugTensors);

const SdpaFixtureReplayTensors = struct {
    grouped_out_bf16: zml.Tensor,
    grouped_out_f32: zml.Tensor,
};

const SdpaFixtureReplayBuffers = zml.Bufferized(SdpaFixtureReplayTensors);

const ProjectionFixtureReplayTensors = struct {
    q_proj_out: zml.Tensor,
    k_proj_out: zml.Tensor,
    v_proj_out: zml.Tensor,
};

const ProjectionFixtureReplayBuffers = zml.Bufferized(ProjectionFixtureReplayTensors);

fn deinitLayerDebugBuffers(buffers: *LayerDebugBuffers) void {
    buffers.input_layernorm_out.deinit();
    buffers.q_proj_out.deinit();
    buffers.k_proj_out.deinit();
    buffers.v_proj_out.deinit();
    buffers.q_norm_out.deinit();
    buffers.k_norm_out.deinit();
    buffers.q_rope_out.deinit();
    buffers.k_rope_out.deinit();
    buffers.v_out.deinit();
    buffers.sdpa_q_grouped.deinit();
    buffers.sdpa_k_scaled.deinit();
    buffers.sdpa_v_grouped.deinit();
    buffers.sdpa_logits_grouped.deinit();
    buffers.sdpa_weights_grouped.deinit();
    buffers.sdpa_grouped_out.deinit();
    buffers.sdpa_grouped_merged_out.deinit();
    buffers.sdpa_weights_grouped_f32.deinit();
    buffers.sdpa_grouped_out_f32.deinit();
    buffers.sdpa_grouped_merged_out_f32.deinit();
    buffers.sdpa_grouped_out_replay_bf16.deinit();
    buffers.sdpa_grouped_out_replay_f32.deinit();
    buffers.sdpa_zml_pre_transpose.deinit();
    buffers.sdpa_zml_transposed.deinit();
    buffers.sdpa_zml_merged_out.deinit();
    buffers.sdpa_out.deinit();
    buffers.sdpa_merged_out.deinit();
    buffers.o_proj_out.deinit();
    buffers.post_attn_residual_out.deinit();
    buffers.post_attention_layernorm_out.deinit();
    buffers.mlp_out.deinit();
    buffers.out.deinit();
}

fn deinitSdpaFixtureReplayBuffers(buffers: *SdpaFixtureReplayBuffers) void {
    buffers.grouped_out_bf16.deinit();
    buffers.grouped_out_f32.deinit();
}

fn deinitProjectionFixtureReplayBuffers(buffers: *ProjectionFixtureReplayBuffers) void {
    buffers.q_proj_out.deinit();
    buffers.k_proj_out.deinit();
    buffers.v_proj_out.deinit();
}

fn runProjectionFixtureReplay(
    layer: dflash.DecoderLayer,
    input_norm_: zml.Tensor,
    target_hidden_: zml.Tensor,
) ProjectionFixtureReplayTensors {
    const input_norm = input_norm_
        .withPartialTags(.{ .s, .d })
        .convert(layer.self_attn.q_proj.weight.dtype())
        .withPartitioning(.{ .d = .replicated });
    const target_hidden = target_hidden_
        .withPartialTags(.{ .s, .d })
        .convert(layer.self_attn.k_proj.weight.dtype())
        .withPartitioning(.{ .d = .replicated });
    const attn = layer.self_attn;

    const q_proj_out = attn.q_proj.forward(input_norm)
        .splitAxis(-1, .{ .h = attn.num_heads, .hd = .auto });
    const k_proj_out = zml.Tensor.concatenate(&.{
        attn.k_proj.forward(target_hidden),
        attn.k_proj.forward(input_norm),
    }, .s).splitAxis(-1, .{ .h = attn.num_kv_heads, .hd = .auto });
    const v_proj_out = zml.Tensor.concatenate(&.{
        attn.v_proj.forward(target_hidden),
        attn.v_proj.forward(input_norm),
    }, .s).splitAxis(-1, .{ .h = attn.num_kv_heads, .hd = .auto });

    return .{
        .q_proj_out = q_proj_out.convert(.f32),
        .k_proj_out = k_proj_out.convert(.f32),
        .v_proj_out = v_proj_out.convert(.f32),
    };
}

fn runSdpaFixtureReplay(
    weights_: zml.Tensor,
    weights_f32_: zml.Tensor,
    v_: zml.Tensor,
) SdpaFixtureReplayTensors {
    const weights = weights_.withPartialTags(.{ .s, .h, .hq, .k });
    const weights_f32 = weights_f32_.withPartialTags(.{ .s, .h, .hq, .k });
    const v = v_.withPartialTags(.{ .k, .h, .hd });
    return .{
        .grouped_out_bf16 = weights.convert(.bf16).dot(v.convert(.bf16), .k).transpose(.{ .s, .h, .hq, .hd }).convert(.f32),
        .grouped_out_f32 = weights_f32.convert(.f32).dot(v.convert(.f32), .k).transpose(.{ .s, .h, .hq, .hd }),
    };
}

fn runLayerDebugF32(
    layer: dflash.DecoderLayer,
    hidden_: zml.Tensor,
    target_hidden_: zml.Tensor,
    position_ids_: zml.Tensor,
) LayerDebugTensors {
    const hidden_states = hidden_
        .withPartialTags(.{ .s, .d })
        .convert(layer.input_layernorm.weight.dtype())
        .withPartitioning(.{ .d = .replicated });
    const target_hidden = target_hidden_
        .withPartialTags(.{ .s, .d })
        .convert(layer.self_attn.k_proj.weight.dtype())
        .withPartitioning(.{ .d = .replicated });
    const position_ids = position_ids_.withPartialTags(.{.s});

    const residual = hidden_states.withPartitioning(.{ .d = .replicated });
    const input_layernorm_out = layer.input_layernorm.forward(residual);

    const attn = layer.self_attn;
    const q_proj_out = attn.q_proj.forward(input_layernorm_out)
        .splitAxis(-1, .{ .h = attn.num_heads, .hd = .auto });
    const k_proj_out = zml.Tensor.concatenate(&.{
        attn.k_proj.forward(target_hidden),
        attn.k_proj.forward(input_layernorm_out),
    }, .s).splitAxis(-1, .{ .h = attn.num_kv_heads, .hd = .auto });
    const v_proj_out = zml.Tensor.concatenate(&.{
        attn.v_proj.forward(target_hidden),
        attn.v_proj.forward(input_layernorm_out),
    }, .s).splitAxis(-1, .{ .h = attn.num_kv_heads, .hd = .auto });

    const q_norm_out = attn.q_norm.forward(q_proj_out.rename(.{ .hd = .d })).rename(.{ .d = .hd });
    const k_norm_out = attn.k_norm.forward(k_proj_out.rename(.{ .hd = .d })).rename(.{ .d = .hd });

    const q_pos = if (position_ids.dim(.s) == input_layernorm_out.dim(.s))
        position_ids
    else
        position_ids.slice1d(.s, .{
            .start = position_ids.dim(.s) - input_layernorm_out.dim(.s),
            .end = position_ids.dim(.s),
        });

    const q_rope_out = zml.nn.rope(q_norm_out, q_pos, attn.rope_opts).rename(.{ .s = .q });
    const k_rope_out = zml.nn.rope(k_norm_out, position_ids, attn.rope_opts).rename(.{ .s = .k });
    const v_out = v_proj_out.rename(.{ .s = .k });

    const sdpa_q_grouped = q_rope_out.splitAxis(.h, .{ .h = k_rope_out.dim(.h), .hq = .auto });
    const sqrt_head_dim: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(sdpa_q_grouped.dim(.hd))));
    const sdpa_k_scaled = k_rope_out.mul(zml.Tensor.scalar(sqrt_head_dim, k_rope_out.dtype()));
    const sdpa_v_grouped = v_out;
    const sdpa_logits_grouped = sdpa_q_grouped.dot(sdpa_k_scaled, .hd);
    const sdpa_weights_grouped = sdpa_logits_grouped.convert(.f32).softmax(.k).convert(sdpa_q_grouped.dtype());
    const sdpa_grouped_out = sdpa_weights_grouped.dot(sdpa_v_grouped, .k);
    const sdpa_grouped_merged_out = sdpa_grouped_out
        .transpose(sdpa_q_grouped.shape())
        .merge(.{ .h = .{ .h, .hq } })
        .rename(.{ .q = .s });
    const sdpa_weights_grouped_f32 = sdpa_logits_grouped.convert(.f32).softmax(.k);
    const sdpa_grouped_out_f32 = sdpa_weights_grouped_f32.dot(sdpa_v_grouped.convert(.f32), .k);
    const sdpa_grouped_merged_out_f32 = sdpa_grouped_out_f32
        .transpose(sdpa_q_grouped.shape())
        .merge(.{ .h = .{ .h, .hq } })
        .rename(.{ .q = .s });
    const sdpa_grouped_out_replay_bf16 = sdpa_weights_grouped
        .convert(.bf16)
        .dot(sdpa_v_grouped.convert(.bf16), .k);
    const sdpa_grouped_out_replay_f32 = sdpa_weights_grouped_f32
        .dot(sdpa_v_grouped.convert(.f32), .k);
    const sdpa_zml_pre_transpose = sdpa_grouped_out;
    const sdpa_zml_transposed = sdpa_zml_pre_transpose.transpose(sdpa_q_grouped.shape());
    const sdpa_zml_merged_out = sdpa_zml_transposed
        .merge(.{ .h = .{ .h, .hq } })
        .rename(.{ .q = .s });

    const sdpa_out = zml.nn.sdpa(q_rope_out, k_rope_out, v_out, .{})
        .withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
    const sdpa_merged_out = sdpa_out
        .merge(.{ .d = .{ .h, .hd } })
        .rename(.{ .q = .s });
    const o_proj_out = attn.o_proj.forward(sdpa_merged_out)
        .rename(.{ .dout = .d })
        .withPartitioning(.{ .d = .replicated });
    const post_attn_residual_out = residual.add(o_proj_out).withPartitioning(.{ .d = .replicated });
    const post_attention_layernorm_out = layer.post_attention_layernorm.forward(post_attn_residual_out);
    const mlp_out = layer.mlp.forward(post_attention_layernorm_out)
        .rename(.{ .dout = .d })
        .withPartitioning(.{ .d = .replicated });
    const out = post_attn_residual_out.add(mlp_out).withPartitioning(.{ .d = .replicated });

    return .{
        .input_layernorm_out = input_layernorm_out.convert(.f32),
        .q_proj_out = q_proj_out.convert(.f32),
        .k_proj_out = k_proj_out.convert(.f32),
        .v_proj_out = v_proj_out.convert(.f32),
        .q_norm_out = q_norm_out.convert(.f32),
        .k_norm_out = k_norm_out.convert(.f32),
        .q_rope_out = q_rope_out.rename(.{ .q = .s }).convert(.f32),
        .k_rope_out = k_rope_out.rename(.{ .k = .s }).convert(.f32),
        .v_out = v_out.rename(.{ .k = .s }).convert(.f32),
        .sdpa_q_grouped = sdpa_q_grouped.rename(.{ .q = .s }).convert(.f32),
        .sdpa_k_scaled = sdpa_k_scaled.rename(.{ .k = .s }).convert(.f32),
        .sdpa_v_grouped = sdpa_v_grouped.rename(.{ .k = .s }).convert(.f32),
        .sdpa_logits_grouped = sdpa_logits_grouped.transpose(.{ .q, .h, .hq, .k }).rename(.{ .q = .s }).convert(.f32),
        .sdpa_weights_grouped = sdpa_weights_grouped.transpose(.{ .q, .h, .hq, .k }).rename(.{ .q = .s }).convert(.f32),
        .sdpa_grouped_out = sdpa_grouped_out.transpose(.{ .q, .h, .hq, .hd }).rename(.{ .q = .s }).convert(.f32),
        .sdpa_grouped_merged_out = sdpa_grouped_merged_out.convert(.f32),
        .sdpa_weights_grouped_f32 = sdpa_weights_grouped_f32.transpose(.{ .q, .h, .hq, .k }).rename(.{ .q = .s }),
        .sdpa_grouped_out_f32 = sdpa_grouped_out_f32.transpose(.{ .q, .h, .hq, .hd }).rename(.{ .q = .s }),
        .sdpa_grouped_merged_out_f32 = sdpa_grouped_merged_out_f32,
        .sdpa_grouped_out_replay_bf16 = sdpa_grouped_out_replay_bf16.transpose(.{ .q, .h, .hq, .hd }).rename(.{ .q = .s }).convert(.f32),
        .sdpa_grouped_out_replay_f32 = sdpa_grouped_out_replay_f32.transpose(.{ .q, .h, .hq, .hd }).rename(.{ .q = .s }),
        .sdpa_zml_pre_transpose = sdpa_zml_pre_transpose.transpose(.{ .q, .h, .hq, .hd }).rename(.{ .q = .s }).convert(.f32),
        .sdpa_zml_transposed = sdpa_zml_transposed.rename(.{ .q = .s }).convert(.f32),
        .sdpa_zml_merged_out = sdpa_zml_merged_out.convert(.f32),
        .sdpa_out = sdpa_out.rename(.{ .q = .s }).convert(.f32),
        .sdpa_merged_out = sdpa_merged_out.convert(.f32),
        .o_proj_out = o_proj_out.convert(.f32),
        .post_attn_residual_out = post_attn_residual_out.convert(.f32),
        .post_attention_layernorm_out = post_attention_layernorm_out.convert(.f32),
        .mlp_out = mlp_out.convert(.f32),
        .out = out.convert(.f32),
    };
}

const TestContext = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activation_store: zml.io.TensorStore.View,
    model: dflash.Model,
    model_buffers: *dflash.Buffers,
    sharding: zml.Sharding,
    model_sharding: zml.Sharding,
    opts: zml.testing.CompareOpts,

    fn testFullModel(self: *TestContext) !void {
        comptime {
            @setEvalBranchQuota(10_000);
        }
        std.log.info("Testing full DFlash forward", .{});
        var target_hidden_buffer = try self.load("target_hidden", self.sharding);
        defer target_hidden_buffer.deinit();
        var noise_embedding_buffer = try self.load("noise_embedding", self.sharding);
        defer noise_embedding_buffer.deinit();
        var position_ids_buffer = try self.load("position_ids", self.sharding);
        defer position_ids_buffer.deinit();
        var expected = try self.load("final_out", self.sharding);
        defer expected.deinit();

        const target_hidden = zml.Tensor.fromShape(target_hidden_buffer.shape()).withTags(.{ .s, .d });
        const noise_embedding = zml.Tensor.fromShape(noise_embedding_buffer.shape()).withTags(.{ .s, .d });
        const position_ids = zml.Tensor.fromShape(position_ids_buffer.shape()).withTags(.{.s});

        var exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            runFullModelF32,
            .{ self.model, target_hidden, noise_embedding, position_ids },
            .{ .shardings = &.{self.model_sharding} },
        );
        defer exe.deinit();

        var exe_args = try exe.args(self.allocator);
        defer exe_args.deinit(self.allocator);
        exe_args.set(.{ self.model_buffers.*, target_hidden_buffer, noise_embedding_buffer, position_ids_buffer });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(exe_args, &results);

        var actual = results.get(zml.Buffer);
        defer actual.deinit();
        try self.expectClose("full.final_out", actual, expected);
    }

    fn testLayer(
        self: *TestContext,
        layer_index: usize,
        layer: dflash.DecoderLayer,
        layer_buffers: zml.Bufferized(dflash.DecoderLayer),
    ) !void {
        comptime {
            @setEvalBranchQuota(10_000);
        }
        std.log.info("Testing DFlash layer {d}", .{layer_index});
        const in_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.in", .{layer_index});
        defer self.allocator.free(in_key);

        var hidden_buffer = try self.load(in_key, self.sharding);
        defer hidden_buffer.deinit();
        var target_hidden_buffer = try self.load("target_hidden_projected", self.sharding);
        defer target_hidden_buffer.deinit();
        var position_ids_buffer = try self.load("position_ids", self.sharding);
        defer position_ids_buffer.deinit();

        const hidden = zml.Tensor.fromShape(hidden_buffer.shape()).withTags(.{ .s, .d });
        const target_hidden = zml.Tensor.fromShape(target_hidden_buffer.shape()).withTags(.{ .s, .d });
        const position_ids = zml.Tensor.fromShape(position_ids_buffer.shape()).withTags(.{.s});

        var exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            runLayerDebugF32,
            .{ layer, hidden, target_hidden, position_ids },
            .{ .shardings = &.{self.model_sharding} },
        );
        defer exe.deinit();

        var exe_args = try exe.args(self.allocator);
        defer exe_args.deinit(self.allocator);
        exe_args.set(.{ layer_buffers, hidden_buffer, target_hidden_buffer, position_ids_buffer });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(exe_args, &results);

        var actual = results.get(LayerDebugBuffers);
        defer deinitLayerDebugBuffers(&actual);

        var failures: usize = 0;
        try self.compareLayerDebug(layer_index, &actual, &failures);
        try self.testLayerProjectionFixtureReplay(layer_index, layer, layer_buffers, &failures);
        try self.testLayerSdpaFixtureReplay(layer_index, &failures);
        if (failures != 0) return error.TestUnexpectedResult;
    }

    fn testLayerProjectionFixtureReplay(
        self: *TestContext,
        layer_index: usize,
        layer: dflash.DecoderLayer,
        layer_buffers: zml.Bufferized(dflash.DecoderLayer),
        failures: *usize,
    ) !void {
        comptime {
            @setEvalBranchQuota(10_000);
        }
        const input_norm_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.input_layernorm.out", .{layer_index});
        defer self.allocator.free(input_norm_key);

        var input_norm_buffer = try self.load(input_norm_key, self.sharding);
        defer input_norm_buffer.deinit();
        var target_hidden_buffer = try self.load("target_hidden_projected", self.sharding);
        defer target_hidden_buffer.deinit();

        const input_norm = zml.Tensor.fromShape(input_norm_buffer.shape()).withTags(.{ .s, .d });
        const target_hidden = zml.Tensor.fromShape(target_hidden_buffer.shape()).withTags(.{ .s, .d });

        var exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            runProjectionFixtureReplay,
            .{ layer, input_norm, target_hidden },
            .{ .shardings = &.{self.model_sharding} },
        );
        defer exe.deinit();

        var exe_args = try exe.args(self.allocator);
        defer exe_args.deinit(self.allocator);
        exe_args.set(.{ layer_buffers, input_norm_buffer, target_hidden_buffer });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(exe_args, &results);

        var replay = results.get(ProjectionFixtureReplayBuffers);
        defer deinitProjectionFixtureReplayBuffers(&replay);

        const q_name = try std.fmt.allocPrint(self.allocator, "layers.{d}.fixture_replay.q_proj", .{layer_index});
        defer self.allocator.free(q_name);
        const k_name = try std.fmt.allocPrint(self.allocator, "layers.{d}.fixture_replay.k_proj", .{layer_index});
        defer self.allocator.free(k_name);
        const v_name = try std.fmt.allocPrint(self.allocator, "layers.{d}.fixture_replay.v_proj", .{layer_index});
        defer self.allocator.free(v_name);

        try self.compareLayerValueWithName(layer_index, "self_attn.q_proj.out", q_name, replay.q_proj_out, failures);
        try self.compareLayerValueWithName(layer_index, "self_attn.k_proj.out", k_name, replay.k_proj_out, failures);
        try self.compareLayerValueWithName(layer_index, "self_attn.v_proj.out", v_name, replay.v_proj_out, failures);
    }

    fn testLayerSdpaFixtureReplay(self: *TestContext, layer_index: usize, failures: *usize) !void {
        comptime {
            @setEvalBranchQuota(10_000);
        }
        const weights_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn.sdpa.weights_grouped", .{layer_index});
        defer self.allocator.free(weights_key);
        const weights_f32_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn.sdpa.weights_grouped_f32", .{layer_index});
        defer self.allocator.free(weights_f32_key);
        const v_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn.sdpa.v_grouped", .{layer_index});
        defer self.allocator.free(v_key);

        var weights_buffer = try self.load(weights_key, self.sharding);
        defer weights_buffer.deinit();
        var weights_f32_buffer = try self.load(weights_f32_key, self.sharding);
        defer weights_f32_buffer.deinit();
        var v_buffer = try self.load(v_key, self.sharding);
        defer v_buffer.deinit();

        const weights = zml.Tensor.fromShape(weights_buffer.shape()).withTags(.{ .s, .h, .hq, .k });
        const weights_f32 = zml.Tensor.fromShape(weights_f32_buffer.shape()).withTags(.{ .s, .h, .hq, .k });
        const v = zml.Tensor.fromShape(v_buffer.shape()).withTags(.{ .k, .h, .hd });

        var exe = try self.platform.compileFn(
            self.allocator,
            self.io,
            runSdpaFixtureReplay,
            .{ weights, weights_f32, v },
            .{},
        );
        defer exe.deinit();

        var exe_args = try exe.args(self.allocator);
        defer exe_args.deinit(self.allocator);
        exe_args.set(.{ weights_buffer, weights_f32_buffer, v_buffer });

        var results = try exe.results(self.allocator);
        defer results.deinit(self.allocator);
        exe.call(exe_args, &results);

        var replay = results.get(SdpaFixtureReplayBuffers);
        defer deinitSdpaFixtureReplayBuffers(&replay);

        const bf16_expected_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn.sdpa.grouped_out_replay_bf16", .{layer_index});
        defer self.allocator.free(bf16_expected_key);
        const f32_expected_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn.sdpa.grouped_out_replay_f32", .{layer_index});
        defer self.allocator.free(f32_expected_key);
        const f32_tf32_expected_key = try std.fmt.allocPrint(self.allocator, "layers.{d}.self_attn.sdpa.grouped_out_replay_f32_tf32", .{layer_index});
        defer self.allocator.free(f32_tf32_expected_key);

        var bf16_expected = try self.load(bf16_expected_key, self.sharding);
        defer bf16_expected.deinit();
        var f32_expected = try self.load(f32_expected_key, self.sharding);
        defer f32_expected.deinit();
        var f32_tf32_expected = try self.load(f32_tf32_expected_key, self.sharding);
        defer f32_tf32_expected.deinit();

        const bf16_name = try std.fmt.allocPrint(self.allocator, "layers.{d}.fixture_replay.bf16_weights_bf16_v", .{layer_index});
        defer self.allocator.free(bf16_name);
        const f32_name = try std.fmt.allocPrint(self.allocator, "layers.{d}.fixture_replay.f32_weights_f32_v", .{layer_index});
        defer self.allocator.free(f32_name);
        const f32_tf32_name = try std.fmt.allocPrint(self.allocator, "layers.{d}.fixture_replay.f32_weights_f32_v_vs_python_tf32", .{layer_index});
        defer self.allocator.free(f32_tf32_name);

        self.expectClose(bf16_name, replay.grouped_out_bf16, bf16_expected) catch |err| {
            failures.* += 1;
            std.log.err("{s} failed: {}", .{ bf16_name, err });
        };
        self.expectClose(f32_name, replay.grouped_out_f32, f32_expected) catch |err| {
            failures.* += 1;
            std.log.err("{s} failed: {}", .{ f32_name, err });
        };
        self.expectClose(f32_tf32_name, replay.grouped_out_f32, f32_tf32_expected) catch |err| {
            failures.* += 1;
            std.log.err("{s} failed: {}", .{ f32_tf32_name, err });
        };
    }

    fn compareLayerDebug(self: *TestContext, layer_index: usize, actual: *LayerDebugBuffers, failures: *usize) !void {
        try self.compareLayerValue(layer_index, "input_layernorm.out", actual.input_layernorm_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.q_proj.out", actual.q_proj_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.k_proj.out", actual.k_proj_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.v_proj.out", actual.v_proj_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.q_norm.out", actual.q_norm_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.k_norm.out", actual.k_norm_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.q_rope.out", actual.q_rope_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.k_rope.out", actual.k_rope_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.v.out", actual.v_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.q_grouped", actual.sdpa_q_grouped, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.k_scaled", actual.sdpa_k_scaled, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.v_grouped", actual.sdpa_v_grouped, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.logits_grouped", actual.sdpa_logits_grouped, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.weights_grouped", actual.sdpa_weights_grouped, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.grouped_out", actual.sdpa_grouped_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.grouped_merged_out", actual.sdpa_grouped_merged_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.weights_grouped_f32", actual.sdpa_weights_grouped_f32, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.grouped_out_f32", actual.sdpa_grouped_out_f32, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.grouped_merged_out_f32", actual.sdpa_grouped_merged_out_f32, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.grouped_out_replay_bf16", actual.sdpa_grouped_out_replay_bf16, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.grouped_out_replay_f32", actual.sdpa_grouped_out_replay_f32, failures);
        try self.compareBuffers("sdpa bf16 grouped_out vs f32 grouped_out", actual.sdpa_grouped_out, actual.sdpa_grouped_out_f32, failures);
        try self.compareBuffers("sdpa explicit grouped_out vs zml pre_transpose", actual.sdpa_grouped_out, actual.sdpa_zml_pre_transpose, failures);
        try self.compareBuffers("sdpa explicit merged_out vs zml merged_out", actual.sdpa_grouped_merged_out, actual.sdpa_zml_merged_out, failures);
        try self.compareBuffers("sdpa explicit merged_out vs zml.nn.sdpa out", actual.sdpa_grouped_merged_out, actual.sdpa_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa.out", actual.sdpa_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.sdpa_merged.out", actual.sdpa_merged_out, failures);
        try self.compareLayerValue(layer_index, "self_attn.o_proj.out", actual.o_proj_out, failures);
        try self.compareLayerValue(layer_index, "post_attn_residual.out", actual.post_attn_residual_out, failures);
        try self.compareLayerValue(layer_index, "post_attention_layernorm.out", actual.post_attention_layernorm_out, failures);
        try self.compareLayerValue(layer_index, "mlp.out", actual.mlp_out, failures);
        try self.compareLayerValue(layer_index, "out", actual.out, failures);
    }

    fn compareLayerValue(self: *TestContext, layer_index: usize, suffix: []const u8, actual: zml.Buffer, failures: *usize) !void {
        try self.compareLayerValueWithName(layer_index, suffix, null, actual, failures);
    }

    fn compareLayerValueWithName(
        self: *TestContext,
        layer_index: usize,
        suffix: []const u8,
        name_override: ?[]const u8,
        actual: zml.Buffer,
        failures: *usize,
    ) !void {
        const key = try std.fmt.allocPrint(self.allocator, "layers.{d}.{s}", .{ layer_index, suffix });
        defer self.allocator.free(key);
        var expected = try self.load(key, self.sharding);
        defer expected.deinit();
        const name = name_override orelse key;
        self.expectClose(name, actual, expected) catch |err| {
            failures.* += 1;
            std.log.err("{s} failed: {}", .{ name, err });
        };
    }

    fn compareBuffers(self: *TestContext, name: []const u8, actual: zml.Buffer, expected: zml.Buffer, failures: *usize) !void {
        self.expectClose(name, actual, expected) catch |err| {
            failures.* += 1;
            std.log.err("{s} failed: {}", .{ name, err });
        };
    }

    fn expectClose(self: *TestContext, name: []const u8, actual: zml.Buffer, expected: zml.Buffer) !void {
        std.log.info(
            "Comparing {s}: actual dtype={} shape={f}; expected dtype={} shape={f}",
            .{
                name,
                actual.shape().dtype(),
                actual.shape(),
                expected.shape().dtype(),
                expected.shape(),
            },
        );
        try zml.testing.expectClose(self.io, actual, expected, self.opts);
    }

    fn load(self: *TestContext, key: []const u8, sharding: zml.Sharding) !zml.Buffer {
        const shape = self.activation_store.getShape(key) orelse {
            std.log.err("Missing activation tensor: {s}", .{key});
            return error.NotFound;
        };

        const host_bytes = try self.allocator.alloc(u8, shape.byteSize());
        defer self.allocator.free(host_bytes);

        var io_buffer: [8 * 1024]u8 = undefined;
        var reader = try self.activation_store.getReader(key, self.io, &io_buffer);
        defer reader.deinit();
        _ = try reader.interface.readSliceAll(host_bytes);

        return zml.Buffer.fromBytes(self.io, self.platform, shape, sharding, host_bytes);
    }
};
