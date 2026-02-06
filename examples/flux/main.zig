const std = @import("std");
const zml = @import("zml");
const tools = @import("tools.zig");
const utils = @import("utils.zig");
const flux_model_transformer2d = @import("flux2_transformer2d_model.zig");
const flow_match_euler_discrete_scheduler = @import("scheduling_flow_match_euler_discrete.zig");
const autoencoder_kl = @import("autoencoder_kl_flux2.zig");

// from tokenization_qwen2_fast import Qwen2TokenizerFast
const Qwen2TokenizerFast = @import("tokenization_qwen2_fast.zig").Qwen2TokenizerFast;

// from modeling_qwen3 import Qwen3ForCausalLM
const modeling_qwen3 = @import("modeling_qwen3.zig");

const stdx = zml.stdx;
const log = std.log.scoped(.flux2_main);

pub const std_options: std.Options = .{
    .log_level = .info,
};

const enable_stage_debug = false;
const stage_to_debug = flux_model_transformer2d.Flux2Transformer2DModel.DebugStage.block0_attn_concat_img;

// bazel run //examples/flux -- --model=/Users/kevin/FLUX.2-klein-4B

pub fn main() !void {
    @setEvalBranchQuota(10_000);

    // log.info("Flux was compiled with {}", .{@import("builtin").mode});

    var debug_allocator: ?std.heap.DebugAllocator(.{}) = null;
    const allocator = if (@import("builtin").mode == .Debug) blk: {
        debug_allocator = .init;
        break :blk debug_allocator.?.allocator();
    } else std.heap.c_allocator;
    defer if (debug_allocator) |*da| std.debug.assert(da.deinit() == .ok);

    // Setup ZML environment
    var threaded = std.Io.Threaded.init(allocator, .{});
    defer threaded.deinit();
    var vfs = try zml.io.VFS.init(allocator, threaded.io());
    defer vfs.deinit();
    const io = vfs.io();

    var platform_auto = try zml.Platform.init(allocator, io, .cpu, .{});
    defer platform_auto.deinit(allocator);

    const CliArgs = struct {
        model: []const u8 = "/Users/kevin/FLUX.2-klein-4B",
    };
    const args = stdx.flags.parseProcessArgs(CliArgs);
    var progress = std.Progress.start(io, .{ .root_name = args.model });

    log.info("\n>>> Tokenizing Prompt...", .{});

    // Tokenizer Setup
    var tokenizer = try Qwen2TokenizerFast.from_pretrained(allocator, io, args.model, .{ .subfolder = "tokenizer" });
    defer tokenizer.deinit();

    const prompt = "A flying surperman style cat";
    const max_length = 20;

    // Apply Chat Template
    const messages = [_]Qwen2TokenizerFast.ChatMessage{
        .{ .role = "user", .content = prompt },
    };
    const text_templated = try tokenizer.apply_chat_template(&messages, .{
        .tokenize = false,
        .add_generation_prompt = true,
        .enable_thinking = false,
    });
    defer allocator.free(text_templated);

    // Let's use `log.info` as it seems to be the pattern in `main.zig`.
    log.info("text_templated: from {s} to {s}", .{ prompt, text_templated });

    // Tokenize
    var tokens = try tokenizer.tokenize(io, platform_auto, text_templated, .{
        .padding = "max_length",
        .max_length = max_length,
        .truncation = true,
        .return_tensors = "pt",
    });
    defer tokens.input_ids.deinit();
    defer tokens.attention_mask.deinit();

    try tools.printFlatten(allocator, io, tokens.input_ids, 20, "input_ids", .{ .include_shape = false });
    try tools.printFlatten(allocator, io, tokens.attention_mask, 20, "attention_mask", .{ .include_shape = false });

    log.info("End Tokenizing Prompt", .{});

    // Qwen3ForCausalLM
    log.info("\n>>> Encoding Prompt...", .{});

    var qwen_node = progress.start("Loading Qwen3", 0);
    var qwen_future = try io.concurrent(modeling_qwen3.Qwen3ForCausalLM.loadFromFile, .{ allocator, io, platform_auto, args.model, &qwen_node });
    var qwen_ctx = try qwen_future.await(io);
    qwen_node.end();
    defer qwen_ctx.deinit(allocator);

    // Prepare RoPE
    const seq_len: usize = @intCast(tokens.input_ids.shape().dim(1));
    const head_dim: usize = @intCast(qwen_ctx.model.model.layers[0].self_attn.head_dim);
    const rope_cpu = try computeRoPE_CPU(allocator, seq_len, head_dim, qwen_ctx.config.rope_theta);
    defer allocator.free(rope_cpu.cos);
    defer allocator.free(rope_cpu.sin);

    const rope_shape = zml.Shape.init(.{ 1, 1, @as(i64, @intCast(seq_len)), @as(i64, @intCast(head_dim)) }, .f32);
    var rope_cos_dev = try zml.Buffer.fromBytes(io, platform_auto, rope_shape, std.mem.sliceAsBytes(rope_cpu.cos));
    defer rope_cos_dev.deinit();
    var rope_sin_dev = try zml.Buffer.fromBytes(io, platform_auto, rope_shape, std.mem.sliceAsBytes(rope_cpu.sin));
    defer rope_sin_dev.deinit();

    // Forward pass
    const QwenEncodingStep = struct {
        pub fn forward(
            self: @This(),
            model: modeling_qwen3.Qwen3ForCausalLM,
            input_ids: zml.Tensor,
            attention_mask: zml.Tensor,
            rope_cos: zml.Tensor,
            rope_sin: zml.Tensor,
        ) zml.Tensor {
            _ = self;
            const rope = modeling_qwen3.RoPE{ .cos = rope_cos, .sin = rope_sin };
            const out = model.forward(input_ids.convert(.i32), rope, attention_mask.convert(.f32), true);
            const h9 = out.hidden_states.?.get(9);
            const h18 = out.hidden_states.?.get(18);
            const h27 = out.hidden_states.?.get(27);
            return zml.Tensor.concatenate(&.{ h9, h18, h27 }, -1);
        }
    };

    const input_shape = tokens.input_ids.shape();
    var qwen_exe = try platform_auto.compile(allocator, io, QwenEncodingStep{}, .forward, .{
        qwen_ctx.model,
        zml.Tensor.fromShape(input_shape),
        zml.Tensor.fromShape(input_shape),
        zml.Tensor.fromShape(rope_shape),
        zml.Tensor.fromShape(rope_shape),
    });
    defer qwen_exe.deinit();

    var qwen_args = try qwen_exe.args(allocator);
    defer qwen_args.deinit(allocator);
    qwen_args.set(.{ qwen_ctx.weights, tokens.input_ids, tokens.attention_mask, rope_cos_dev, rope_sin_dev });

    var qwen_res = try qwen_exe.results(allocator);
    defer qwen_res.deinit(allocator);
    qwen_exe.call(qwen_args, &qwen_res);

    const prompt_embeds = qwen_res.get(zml.Buffer);
    const prompt_seq_len: usize = @intCast(prompt_embeds.shape().dim(1));
    var prompt_text_ids: zml.Buffer = try utils.prepare_text_ids(allocator, io, platform_auto, prompt_seq_len);
    defer prompt_text_ids.deinit();

    try tools.printFlatten(allocator, io, prompt_text_ids, 20, "    text_ids (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, prompt_embeds, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

    // 1. Loading Embeds & Text IDs (Pre-computed for now)
    const embeds_path = "/Users/kevin/zml/flux_klein_notebook_embeds.npy";
    const text_ids_path = "/Users/kevin/zml/flux_klein_notebook_text_ids.npy";
    var prompt_embeds_from_python: zml.Buffer = try utils.loadInput(allocator, io, platform_auto, embeds_path, .{ 1, 20, 7680 });
    defer prompt_embeds_from_python.deinit();
    var text_ids_from_python: zml.Buffer = try utils.loadInput(allocator, io, platform_auto, text_ids_path, .{ 1, 20, 4 });
    defer text_ids_from_python.deinit();

    try tools.printFlatten(allocator, io, text_ids_from_python, 20, "    text_ids (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, prompt_embeds_from_python, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

    // Assert computed text_ids matches reference
    try tools.printFlatten(allocator, io, prompt_text_ids, 200, "    prompt_text_ids (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, text_ids_from_python, 200, "    text_ids_from_python (first 20).", .{ .include_shape = true });
    // try tools.assertBuffersEqual(allocator, io, prompt_text_ids, text_ids_from_python, 1e-6);
    // std.process.exit(0);
    try tools.assertBuffersEqual(allocator, io, prompt_embeds, prompt_embeds_from_python, 1e-1);

    {
        // text_ids_selector is the pb
        const text_ids_selector = prompt_text_ids;
        const prompt_embeds_selector = prompt_embeds;

        // const text_ids_selector = text_ids_from_python;
        // const prompt_embeds_selector = prompt_embeds_from_python;

        // 2. Load Transformer & Scheduler
        var transformer2d_model_ctx = try flux_model_transformer2d.loadFromFile(allocator, io, platform_auto, args.model);
        defer transformer2d_model_ctx.deinit(allocator);

        var scheduler = try flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler.loadFromFile(allocator, io, args.model);
        defer scheduler.deinit();

        log.info("Models initialized successfully", .{});

        // 3. Prepare Latents
        log.info(">>Preparing Latents...", .{});
        const img_dim = 128;

        var latent_buf, var latent_ids_buf = try utils.get_latents(allocator, io, platform_auto, transformer2d_model_ctx.config, img_dim);
        defer latent_buf.deinit();
        defer latent_ids_buf.deinit();

        try tools.printFlatten(allocator, io, latent_buf, 20, "    Latents (first 20).", .{ .include_shape = true });
        try tools.printFlatten(allocator, io, latent_ids_buf, 20, "    Latent_ids (first 20).", .{ .include_shape = true });

        // 4. Schedule (Sampling Loop)
        log.info("\n>>> Preparing Timesteps...", .{});
        if (enable_stage_debug) {
            try utils.debugForwardStage(
                transformer2d_model_ctx.model,
                transformer2d_model_ctx.weights,
                latent_buf,
                latent_ids_buf,
                prompt_embeds_selector,
                text_ids_selector,
                allocator,
                io,
                platform_auto,
                stage_to_debug,
            );
            return;
        }

        var latents_out = try utils.schedule(
            transformer2d_model_ctx.model,
            transformer2d_model_ctx.weights,
            scheduler,
            latent_buf,
            latent_ids_buf,
            prompt_embeds_selector,
            text_ids_selector,
            1, // num_inference_steps
            allocator,
            io,
            platform_auto,
        );
        defer latents_out.deinit();

        try tools.printFlatten(allocator, io, latents_out, 20, "    Latents Out (first 20).", .{ .include_shape = true });

        log.info("\n>>> Decoding Latents...", .{});

        var vae_ctx = try autoencoder_kl.AutoencoderKLFlux2.loadFromFile(allocator, io, platform_auto, args.model);
        defer vae_ctx.deinit(allocator);

        const image_decoded_buf = try utils.variational_auto_encode(allocator, io, platform_auto, vae_ctx, latents_out);

        try tools.printFlatten(allocator, io, image_decoded_buf, 20, "    Image Decoded (first 20).", .{ .include_shape = true });

        try tools.saveBufferToNpy(allocator, io, platform_auto, image_decoded_buf, "/Users/kevin/zml/flux_klein_zml_result.npy");

        log.info("\n>>> Pipeline Complete.", .{});
    }

    std.process.exit(0);
}

fn computeRoPE_CPU(allocator: std.mem.Allocator, seq_len: usize, head_dim: usize, theta: f32) !struct { cos: []f32, sin: []f32 } {
    const dim = head_dim;
    const half = dim / 2;

    const inv_freq = try allocator.alloc(f32, half);
    defer allocator.free(inv_freq);

    for (0..half) |i| {
        const x = -@log(theta) * @as(f32, @floatFromInt(i)) / @as(f32, @floatFromInt(half));
        inv_freq[i] = @exp(x);
    }

    const total_elems = seq_len * dim;
    const cos_data = try allocator.alloc(f32, total_elems);
    // errdefer allocator.free(cos_data);
    const sin_data = try allocator.alloc(f32, total_elems);
    // errdefer { allocator.free(cos_data); allocator.free(sin_data); }

    for (0..seq_len) |s| { // position
        for (0..half) |i| { // freq index
            const freq = @as(f32, @floatFromInt(s)) * inv_freq[i];
            const cos_val = @cos(freq);
            const sin_val = @sin(freq);

            const offset = s * dim;
            cos_data[offset + i] = cos_val;
            cos_data[offset + i + half] = cos_val;

            sin_data[offset + i] = sin_val;
            sin_data[offset + i + half] = sin_val;
        }
    }

    return .{ .cos = cos_data, .sin = sin_data };
}
