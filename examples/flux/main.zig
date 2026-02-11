const std = @import("std");
const zml = @import("zml");

const tools = @import("tools.zig");
const utils = @import("utils.zig");
const flux_model_transformer2d = @import("flux2_transformer2d_model.zig");
const flow_match_euler_discrete_scheduler = @import("scheduling_flow_match_euler_discrete.zig");
const autoencoder_kl = @import("autoencoder_kl_flux2.zig");
const Qwen2TokenizerFast = @import("tokenization_qwen2_fast.zig").Qwen2TokenizerFast;
const modeling_qwen3 = @import("modeling_qwen3.zig");
const config = @import("config");

const stdx = zml.stdx;
const log = std.log.scoped(.flux2_main);

pub const std_options: std.Options = .{
    .log_level = .info,
};

// bazel run //examples/flux -- --model=/Users/kevin/FLUX.2-klein-4B

// bazel run --//examples/flux:enable_stage_debug=true //examples/flux -- --model=/Users/kevin/FLUX.2-klein-4B

// bazel run //examples/flux -- --model=//flux2:FLUX.2-klein-4B

// bazel run //examples/flux -- --model=flux2:FLUX.2-klein-4B --prompt="a photo of a cat"

// bazel run //examples/flux -- --model=hf://black-forest-labs/FLUX.2-klein-4B --prompt="a photo of a cat"

// bazel run //examples/flux -- \
// --model=/Users/kevin/FLUX.2-klein-4B \
// --prompt="a photo of a cat sleeping on the sofa" \
// --output-image-path=/Users/kevin/zml/flux_klein_zml_result.png \
// --output-image-size=256 \
// --num-inference-steps=4 \
// --random-seed=0 \
// --generator-type=torch \
// --async-limit=1

// bazel run //examples/flux -- \
// --model=/Users/kevin/FLUX.2-klein-4B \
// --prompt="a photo of a cat sleeping on the sofa" \
// --output-image-path=/Users/kevin/zml/flux_klein_zml_result.png \
// --output-image-size=256 \
// --num-inference-steps=4 \
// --random-seed=0 \
// --generator-type=accelerator_box_muller \
// --async-limit=1

// bazel run //examples/flux -- \
// --model=/Users/kevin/FLUX.2-klein-4B \
// --prompt="a photo of a cat sleeping on the sofa" \
// --output-image-path=/Users/kevin/zml/flux_klein_zml_result.png \
// --output-image-size=128 \
// --num-inference-steps=1 \
// --random-seed=1 \
// --generator-type=torch \
// --async-limit=1

const CliArgs = struct {
    model: []const u8 = "hf://black-forest-labs/FLUX.2-klein-4B",
    prompt: []const u8 = "A photo of a cat",
    seqlen: usize = 256,
    output_image_path: []const u8 = "output.png",
    output_image_size: usize = 128,
    random_seed: u64 = 0,
    num_inference_steps: usize = 4,
    async_limit: ?usize = null,
    generator_type: utils.GeneratorType = .torch,
    // 3840x2160
    // 1920x1080
    // 1024x1024
    // 1280x720
    // 512x512
    const Resolution = enum { HLD, LD, SD, HD, FHD, QHD, UHD };

    pub const help =
        \\ Usage: flux \
        \\ --model <model> \
        \\ --prompt <prompt> \
        \\ --seqlen <seqlen> \
        \\ --output-image-path <path> \
        \\ --output-image-size <size> \
        \\ --random-seed <seed> \
        \\ --async-limit <limit>
        \\
        \\ Run Flux with a given model and prompt.
        \\
        \\ Options:
        \\   --model <model>    Model to use
        \\   --prompt <prompt>  Prompt to use
        \\   --seqlen <seqlen>  Sequence length
        \\   --output-image-path <path>  Output image path
        \\   --output-image-size <size>  Output image size
        \\   --random-seed <seed>  Random seed
        \\   --async-limit <limit>  Async limit
        \\
        \\ Example: flux --model hf://black-forest-labs/FLUX.2-klein-4B --prompt "A photo of a cat" --seqlen 256
        \\ Example: flux --model hf://black-forest-labs/FLUX.2-klein-4B --prompt "A photo of a cat" --seqlen 256 --output-image-path output.png --output-image-size 256 --random-seed 0 --async-limit 1
        \\ Example: flux --model hf://black-forest-labs/FLUX.2-klein-4B --prompt "A photo of a cat" --seqlen 256 --output-image-path output.png --output-image-size 256 --random-seed 0 --async-limit 1 --model /Users/kevin/FLUX.2-klein-4B
        \\ Example: flux --model hf://black-forest-labs/FLUX.2-klein-4B --prompt "A photo of a cat" --seqlen 256 --output-image-path output.png --output-image-size 256 --random-seed 0 --async-limit 1 --model /Users/kevin/FLUX.2-klein-4B --prompt "A photo of a cat"
        \\ Example: flux --model hf://black-forest-labs/FLUX.2-klein-4B --prompt "A photo of a cat" --seqlen 256 --output-image-path output.png --output-image-size 256 --random-seed 0 --async-limit 1 --model /Users/kevin/FLUX.2-klein-4B --prompt "A photo of a cat" --seqlen 256
    ;
};

pub fn main() !void {
    @setEvalBranchQuota(10_000);

    log.info("Flux was compiled with {}", .{@import("builtin").mode});

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

    var platform_auto: *zml.Platform = try zml.Platform.init(allocator, io, .cpu, .{});
    defer platform_auto.deinit(allocator);

    const args = stdx.flags.parseProcessArgs(CliArgs);
    log.info("args: {s} : {s} : {d}", .{ args.model, args.prompt, args.seqlen });

    var http_client: std.http.Client = .{
        .allocator = allocator,
        .io = threaded.io(),
    };
    try http_client.initDefaultProxies(allocator);
    defer http_client.deinit();

    var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});
    defer vfs_file.deinit();
    try vfs.register("file", vfs_file.io());

    var vfs_https: zml.io.VFS.HTTP = try .init(allocator, threaded.io(), &http_client, .https);
    defer vfs_https.deinit();
    try vfs.register("https", vfs_https.io());

    var hf_vfs: zml.io.VFS.HF = try .auto(allocator, threaded.io(), &http_client);
    defer hf_vfs.deinit();
    try vfs.register("hf", hf_vfs.io());

    log.info("Resolving model repo", .{});
    const repo: std.Io.Dir = try zml.safetensors.resolveModelRepo(io, args.model);

    if (comptime config.enable_stage_debug) {
        std.log.info("Stage Debug Enabled", .{});
        std.process.exit(0);
    } else {
        std.log.info("Stage Debug Disabled", .{});
    }

    var progress = std.Progress.start(io, .{ .root_name = args.model });

    // ==================== Tokenizing Prompt ====================

    log.info("\n>>> Tokenizing Prompt...", .{});

    var qwen2_future = try io.concurrent(Qwen2TokenizerFast.pipelineTokenizer, .{ allocator, io, repo, platform_auto, null, .{ .prompt = args.prompt, .max_length = args.seqlen } });

    defer _ = qwen2_future.cancel(io) catch unreachable;

    var tokens = try qwen2_future.await(io);
    defer tokens.deinit();

    try tools.printFlatten(allocator, io, tokens.input_ids, 20, "input_ids", .{ .include_shape = false });
    try tools.printFlatten(allocator, io, tokens.attention_mask, 20, "attention_mask", .{ .include_shape = false });

    log.info("End Tokenizing Prompt", .{});

    // ==================== Encoding Prompt ====================

    // Qwen3ForCausalLM
    log.info("\n>>> Encoding Prompt...", .{});
    var qwen_node = progress.start("Loading Qwen3", 0);
    var qwen3_model_ctx = try modeling_qwen3.Qwen3ForCausalLM.loadFromFile(allocator, io, platform_auto, repo, &qwen_node);
    qwen_node.end();
    defer qwen3_model_ctx.deinit(allocator);

    // Prepare RoPE
    const seq_len: usize = @intCast(tokens.input_ids.shape().dim(1));
    const head_dim: usize = @intCast(qwen3_model_ctx.model.model.layers[0].self_attn.head_dim);
    const rope_cpu = try utils.computeRoPE_CPU(allocator, seq_len, head_dim, qwen3_model_ctx.config.rope_theta);
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
        qwen3_model_ctx.model,
        zml.Tensor.fromShape(input_shape),
        zml.Tensor.fromShape(input_shape),
        zml.Tensor.fromShape(rope_shape),
        zml.Tensor.fromShape(rope_shape),
    });
    defer qwen_exe.deinit();

    var qwen_args = try qwen_exe.args(allocator);
    defer qwen_args.deinit(allocator);
    qwen_args.set(.{ qwen3_model_ctx.weights, tokens.input_ids, tokens.attention_mask, rope_cos_dev, rope_sin_dev });

    var qwen_res = try qwen_exe.results(allocator);
    defer qwen_res.deinit(allocator);
    qwen_exe.call(qwen_args, &qwen_res);

    const prompt_embeds = qwen_res.get(zml.Buffer);
    const prompt_seq_len: usize = @intCast(prompt_embeds.shape().dim(1));
    var prompt_text_ids: zml.Buffer = try utils.prepare_text_ids(allocator, io, platform_auto, prompt_seq_len);
    defer prompt_text_ids.deinit();

    try tools.printFlatten(allocator, io, prompt_text_ids, 20, "    text_ids (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, prompt_embeds, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

    {
        // 1. Loading Embeds & Text IDs (Pre-computed for now)
        // const embeds_path = "/Users/kevin/zml/flux_klein_notebook_embeds.npy";
        // const text_ids_path = "/Users/kevin/zml/flux_klein_notebook_text_ids.npy";
        // var prompt_embeds_from_python: zml.Buffer = try utils.loadInput(allocator, io, platform_auto, embeds_path, .{ 1, 20, 7680 });
        // defer prompt_embeds_from_python.deinit();
        // var text_ids_from_python: zml.Buffer = try utils.loadInput(allocator, io, platform_auto, text_ids_path, .{ 1, 20, 4 });
        // defer text_ids_from_python.deinit();

        // try tools.printFlatten(allocator, io, text_ids_from_python, 20, "    text_ids (first 20).", .{ .include_shape = true });
        // try tools.printFlatten(allocator, io, prompt_embeds_from_python, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });
        // try tools.assertBuffersEqual(allocator, io, prompt_text_ids, text_ids_from_python, 1e-6);
        // try tools.assertBuffersEqual(allocator, io, prompt_embeds, prompt_embeds_from_python, 1e-1);
    }
    // std.process.exit(0);

    // // text_ids_selector is the pb
    // const text_ids_selector = prompt_text_ids;
    // const prompt_embeds_selector = prompt_embeds;

    // const text_ids_selector = text_ids_from_python;
    // const prompt_embeds_selector = prompt_embeds_from_python;

    // 2. Load Transformer & Scheduler
    var flux2_transformer2d_node = progress.start("Loading Flux2Transformer2DModel", 0);
    var transformer2d_model_ctx = try flux_model_transformer2d.ModelContext.loadFromFile(allocator, io, platform_auto, repo, &flux2_transformer2d_node, .{});
    defer transformer2d_model_ctx.deinit(allocator);
    flux2_transformer2d_node.end();

    var scheduler_node = progress.start("Loading FlowMatchEulerDiscreteScheduler", 0);
    var scheduler = try flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler.loadFromFile(allocator, io, repo, &scheduler_node, .{});
    defer scheduler.deinit();
    scheduler_node.end();

    log.info("Models initialized successfully", .{});

    // 3. Prepare Latents
    log.info(">>Preparing Latents...", .{});

    var latent_buf, var latent_ids_buf = try utils.get_latents(allocator, io, platform_auto, transformer2d_model_ctx.config, args.output_image_size, args.generator_type, args.random_seed);
    defer latent_buf.deinit();
    defer latent_ids_buf.deinit();

    try tools.printFlatten(allocator, io, latent_buf, 20, "    Latents (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, latent_ids_buf, 20, "    Latent_ids (first 20).", .{ .include_shape = true });

    // // 4. Schedule (Sampling Loop)
    log.info("\n>>> Preparing Timesteps...", .{});

    var latents_out = try utils.schedule(
        transformer2d_model_ctx.model,
        transformer2d_model_ctx.weights,
        scheduler,
        latent_buf,
        latent_ids_buf,
        prompt_embeds,
        prompt_text_ids,
        args.num_inference_steps,
        allocator,
        io,
        platform_auto,
    );
    defer latents_out.deinit();

    try tools.printFlatten(allocator, io, latents_out, 20, "    Latents Out (first 20).", .{ .include_shape = true });

    log.info("\n>>> Decoding Latents...", .{});

    var vae_ctx = try autoencoder_kl.AutoencoderKLFlux2.loadFromFile(allocator, io, platform_auto, repo, null, .{});
    defer vae_ctx.deinit(allocator);

    const image_decoded_buf: zml.Buffer = try utils.variational_auto_encode(allocator, io, platform_auto, vae_ctx, latents_out);
    try tools.printFlatten(allocator, io, image_decoded_buf, 20, "    Image Decoded (first 20).", .{ .include_shape = true });

    try utils.saveFluxImageToPng(allocator, io, image_decoded_buf, args.output_image_path);

    log.info("\n>>> Pipeline Complete.", .{});

    std.process.exit(0);
}
