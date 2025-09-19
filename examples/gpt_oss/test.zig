const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const flags = stdx.flags;
const zml = @import("zml");

const GptOss = @import("./GptOss.zig");

const log = std.log.scoped(.gpt_oss);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ test-implementation --model-path=gpt_oss/ --activations=activation.pt
        ;
        model_path: []const u8,
        activations: []const u8,
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // Select platform
    const platform = context.autoPlatform(.{}).withCompilationOptions(.{
        .xla_dump_to = "/tmp/zml/GptOss.test",
        .sharding_enabled = false,
    });

    // Resolve model files
    var args = std.process.args();
    const cli = flags.parse(&args, CliArgs);

    const model_tokenizer_path = try std.fs.path.join(allocator, &.{ cli.model_path, "tokenizer.json" });
    defer allocator.free(model_tokenizer_path);

    // Memory arena dedicated to model shapes and weights
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    // Parse model config
    const config: GptOss.Config = cfg: {
        const model_config_path = try std.fs.path.join(allocator, &.{ cli.model_path, "config.json" });
        defer allocator.free(model_config_path);
        var config_json_file = try asynk.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(&config_json_buffer);
        var reader = std.json.Reader.init(model_arena, &config_reader.interface);
        defer reader.deinit();
        const cfg = try std.json.parseFromTokenSourceLeaky(GptOss.Config, model_arena, &reader, .{ .ignore_unknown_fields = true });

        std.log.info("Parsed GptOss config: {}", .{cfg});
        break :cfg cfg;
    };

    // Read model shapes.
    var store: zml.aio.BufferStore = st: {
        const simple_path = try std.fs.path.join(allocator, &.{ cli.model_path, "model.safetensors" });
        defer allocator.free(simple_path);

        if (asynk.File.access(simple_path, .{})) {
            log.info("\tLoading GptOss weights from {s}...", .{simple_path});
            break :st try zml.aio.detectFormatAndOpen(allocator, simple_path);
        } else |_| {}

        const sharded_path = try std.fs.path.join(allocator, &.{ cli.model_path, "model.safetensors.index.json" });
        defer allocator.free(sharded_path);
        log.info("\tLoading GptOss weights from {s}...", .{sharded_path});
        break :st try zml.aio.detectFormatAndOpen(allocator, sharded_path);
    };
    defer store.deinit();

    // Create the model and configure it.
    const options: GptOss.Options = .{
        .max_seq_len = 256,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };
    const gpt_oss = try GptOss.init(model_arena, store, config, options);

    // Load the weights.
    var gpt_oss_weights = try gpt_oss.loadBuffers(model_arena, store, platform);
    defer zml.aio.unloadBuffers(&gpt_oss_weights);

    // Load the activations.
    var activation_store = try zml.aio.detectFormatAndOpen(allocator, cli.activations);
    defer activation_store.deinit();
    for (activation_store.buffers.keys(), activation_store.buffers.values()) |k, *v| {
        if (std.mem.indexOf(u8, k, ".layers.") != null and (std.mem.endsWith(u8, k, ".in.0") or std.mem.endsWith(u8, k, ".out.0"))) {
            if (v.dim(0) == 1 and v.rank() == 3) {
                v.* = v.squeeze(0).withTags(.{ .s, .d });
            }
        }

        // log.info("Loaded activations: {s} -> {f}", .{ k, v });
        // if (std.mem.indexOf(u8, k, ".22.self_attn.in.") != null) {
        //     log.info("Self attn input {s}: {d:32.3}", .{ k, v.* });
        // }
        // if (std.mem.indexOf(u8, k, ".22.self_attn.out.") != null) {
        //     log.info("Self attn output {s}: {d:32.3}", .{ k, v.* });
        // }

        // if (std.mem.endsWith(u8, k, "layers.23.mlp.experts.in.1")) log.info("Routing {s}: {d:32.2}", .{k, v.*});
        // if (std.mem.endsWith(u8, k, "layers.23.mlp.experts.in.2")) log.info("Routing weights {s}: {d:32.2}", .{k, v.*});
        // if (std.mem.endsWith(u8, k, ".0.mlp.experts.in.2")) log.info("Routing weights {s}: {d:32.2}", .{ k, v.* });
    }

    // Test implementation
    try testImplementation(platform, gpt_oss, gpt_oss_weights, activation_store);
}

fn testImplementation(
    platform: zml.Platform,
    gpt_oss: GptOss,
    gpt_oss_weights: zml.Bufferized(GptOss),
    store: zml.aio.BufferStore,
) !void {
    try zml.testing.testLayer(platform, store, "model.model.embed_tokens", gpt_oss.model.embed_tokens, gpt_oss_weights.model.embed_tokens, 1e-3);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.v_proj", gpt_oss.model.layers[0].self_attn.v_proj, gpt_oss_weights.model.layers[0].self_attn.v_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.q_proj", gpt_oss.model.layers[0].self_attn.q_proj, gpt_oss_weights.model.layers[0].self_attn.q_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.k_proj", gpt_oss.model.layers[0].self_attn.k_proj, gpt_oss_weights.model.layers[0].self_attn.k_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.o_proj", gpt_oss.model.layers[0].self_attn.o_proj, gpt_oss_weights.model.layers[0].self_attn.o_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.input_layernorm", gpt_oss.model.layers[0].input_layernorm, gpt_oss_weights.model.layers[0].input_layernorm, 1e-3);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.post_attention_layernorm", gpt_oss.model.layers[0].post_attention_layernorm, gpt_oss_weights.model.layers[0].post_attention_layernorm, 1e-3);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.mlp.router", gpt_oss.model.layers[0].mlp.router, gpt_oss_weights.model.layers[0].mlp.router, 1e-2);

    const allocator = std.heap.smp_allocator;
    {
        const input: zml.HostBuffer = store.buffers.get("model.model.layers.22.self_attn.in.0").?;
        const kv_shape = zml.Shape.init(.{ .layer = 1, .k = input.dim(.s), .h = 8, .hd = 64 }, .bf16);
        const self_attn_mod = try zml.compileModel(
            allocator,
            GptOss.SelfAttn.forward,
            gpt_oss.model.layers[22].self_attn,
            .{
                store.buffers.get("model.model.layers.22.self_attn.in.0").?.shape(),
                zml.Shape.scalar(.u32),
                GptOss.KvCache.initShape(kv_shape),
            },
            platform,
        );
        defer self_attn_mod.deinit();

        const self_attn_exe = self_attn_mod.prepare(gpt_oss_weights.model.layers[22].self_attn);
        const out, _ = self_attn_exe.call(.{
            try input.toDevice(platform),
            try .scalar(platform, 0, .u32),
            try GptOss.KvCache.initBuffer(kv_shape, platform),
        });

        const expected = store.buffers.get("model.model.layers.22.self_attn.out.0").?;
        // Very imprecise !
        try zml.testing.expectClose(expected, out, 0.2);
        log.info("All good for self_attn", .{});
    }

    {
        const w = gpt_oss.model.layers[22].mlp.experts.gate_up_proj;
        const dequant_mod = try zml.compileFn(allocator, dequantize, .{.{
            .blocks = w.blocks.shape(),
            .scale = w.scale.shape(),
            .bias = w.bias.?.shape(),
            .blocks_dtype = w.blocks_dtype,
        }}, platform);
        defer dequant_mod.deinit();

        const dequant_weight = dequant_mod.call(.{gpt_oss_weights.model.layers[22].mlp.experts.gate_up_proj});
        const expected = store.buffers.get("model.model.layers.22.mlp.experts.gate_up_proj").?;

        try zml.testing.expectClose(expected, dequant_weight, 1e-4);
        log.info("All good for dequantize gate_up_proj", .{});
    }

    {
        const w = gpt_oss.model.layers[22].mlp.experts.down_proj;
        const dequant_mod = try zml.compileFn(allocator, dequantize, .{.{
            .blocks = w.blocks.shape(),
            .scale = w.scale.shape(),
            .bias = w.bias.?.shape(),
            .blocks_dtype = w.blocks_dtype,
        }}, platform);
        defer dequant_mod.deinit();

        const dequant_weight = dequant_mod.call(.{gpt_oss_weights.model.layers[22].mlp.experts.down_proj});
        const expected = store.buffers.get("model.model.layers.22.mlp.experts.down_proj").?;

        try zml.testing.expectClose(expected, dequant_weight, 1e-4);
        log.info("All good for dequantize down_proj", .{});
    }

    {
        const moe: MoeAllToAll = .{ .experts = gpt_oss.model.layers[0].mlp.experts };

        try zml.testing.testLayer(platform, store, "model.model.layers.0.mlp.experts", moe, .{ .experts = gpt_oss_weights.model.layers[0].mlp.experts }, 5e-2);
    }

    try zml.testing.testLayer(platform, store, "model.model.layers.0.mlp", gpt_oss.model.layers[0].mlp, gpt_oss_weights.model.layers[0].mlp, 1e-2);
}

fn dequantize(self: GptOss.BlockScaledLinear) zml.Tensor {
    // Bitcast to our actual type. This allows to load weights in a packed layout.
    const blocks_0 = self.blocks.bitCast(self.blocks_dtype);
    const blocks = blocks_0.merge(.{ .d_block = .{ .d_block, .bitcast } });

    const scale = self.scale.bitCast(.f8e8m0);

    const dequantized_weight: zml.Tensor = .mul(
        blocks.convert(.bf16),
        scale.convert(.bf16).appendAxes(.{.d_block}),
    );
    return dequantized_weight.merge(.{ .d = .{ .d, .d_block } }).contiguous(.{ .d, .out });
}

const MoeAllToAll = struct {
    experts: GptOss.Mlp,

    /// Fork of GptOss.Moe where we also feed the gating.
    pub fn forward(self: MoeAllToAll, input: zml.Tensor, routing: zml.Tensor, gating: zml.Tensor) zml.Tensor {
        _ = routing;

        return GptOss.mixtureOfExpertsAllToAll(GptOss.Mlp, self.experts, input, gating.withTags(.{ .s, .expert }));
    }
};
