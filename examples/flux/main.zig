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

// --prompt="a photo of a cat sleeping on a licorne in magic forest" \

// bazel run //examples/flux -- \
// --model=/Users/kevin/FLUX.2-klein-4B \
// --prompt="a photo of a cat sleeping on the sofa" \
// --output-image-path=/Users/kevin/zml/flux_klein_zml_result.png \
// --kitty-output
// --resolution=HLD \
// --num-inference-steps=1 \
// --random-seed=1 \
// --generator-type=torch \
// --data-type=f32 \
// --seqlen=32
// --async-limit=1

const Resolution = enum { HLD, LD, SD, HD, FHD, QHD, UHD };

const CliArgs = struct {
    // model: []const u8 = "hf://black-forest-labs/FLUX.2-klein-4B",
    model: []const u8 = "/Users/kevin/FLUX.2-klein-4B",
    prompt: []const u8 = "A photo of a cat",
    seqlen: usize = 512,
    // output_image_path: []const u8 = "output.png",
    // output_image_path: []const u8 = "/Users/kevin/zml/flux_klein_zml_result.png",
    output_image_path: ?[]const u8 = null,
    kitty_output: bool = false,
    random_seed: u64 = 0,
    num_inference_steps: usize = 1,
    async_limit: ?usize = null,
    generator_type: utils.GeneratorType = .accelerator_box_muller,
    data_type: zml.DataType = .f32,
    // 8K UHD 7680x4320
    // 4K QHD 3840x2160
    // FHD 1920x1080
    // HD 1280x720
    // SD 512x512
    // LD 256x256
    // HLD 128x128
    resolution: Resolution = .HLD,

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
    const args = stdx.flags.parseProcessArgs(CliArgs);
    log.info("args: {s} : {s} : {d}", .{ args.model, args.prompt, args.seqlen });

    // var platform_auto: *zml.Platform = try zml.Platform.init(allocator, io, .cpu, .{});
    var platform_auto: *zml.Platform = try zml.Platform.auto(allocator, io, .{});
    defer platform_auto.deinit(allocator);

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
        log.info("Stage Debug Enabled", .{});
        std.process.exit(0);
    } else {
        log.info("Stage Debug Disabled", .{});
    }

    const parallelism_level: usize = if (args.async_limit) |limit| limit else try std.Thread.getCpuCount();

    log.info("Parallelism Level: {}", .{parallelism_level});

    var progress = std.Progress.start(io, .{ .root_name = args.model });

    // ==================== Tokenizing Prompt ====================

    var qwen2_node = progress.start("Tokenizing Prompt", 0);
    var qwen2_future = try io.concurrent(Qwen2TokenizerFast.pipelineTokenizer, .{ allocator, io, repo, platform_auto, &qwen2_node, .{ .prompt = args.prompt, .max_length = args.seqlen } });

    defer _ = qwen2_future.cancel(io) catch unreachable;
    qwen2_node.end();

    var tokens = try qwen2_future.await(io);
    defer tokens.deinit();

    // try tools.printFlatten(allocator, io, tokens.input_ids, 20, "input_ids", .{ .include_shape = false });
    // try tools.printFlatten(allocator, io, tokens.attention_mask, 20, "attention_mask", .{ .include_shape = false });

    log.info("End Tokenizing Prompt", .{});

    // ==================== Encoding Prompt ====================

    var qwen_node = progress.start("Loading Qwen3", 0);
    var qwen3_model_ctx = try modeling_qwen3.Qwen3ForCausalLM.loadFromFile(allocator, io, platform_auto, repo, parallelism_level, &qwen_node);
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

    // try tools.printFlatten(allocator, io, prompt_text_ids, 20, "    text_ids (first 20).", .{ .include_shape = true });
    // try tools.printFlatten(allocator, io, prompt_embeds, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

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
    var transformer2d_model_ctx = try flux_model_transformer2d.ModelContext.loadFromFile(allocator, io, platform_auto, repo, parallelism_level, &flux2_transformer2d_node, .{});
    defer transformer2d_model_ctx.deinit(allocator);
    flux2_transformer2d_node.end();

    var scheduler_node = progress.start("Loading FlowMatchEulerDiscreteScheduler", 0);
    var scheduler = try flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler.loadFromFile(allocator, io, repo, &scheduler_node, .{});
    defer scheduler.deinit();
    scheduler_node.end();

    log.info("Models initialized successfully", .{});

    // 3. Prepare Latents
    log.info(">> Preparing Latents...", .{});

    const output_image_dim: utils.ResolutionInfo = switch (args.resolution) {
        .HLD => .{ .width = 128, .height = 128 },
        .LD => .{ .width = 256, .height = 256 },
        .SD => .{ .width = 512, .height = 512 },
        .HD => .{ .width = 1280, .height = 720 },
        .FHD => .{ .width = 1920, .height = 1080 },
        .QHD => .{ .width = 2560, .height = 1440 },
        .UHD => .{ .width = 3840, .height = 2160 },
    };

    var latent_buf, var latent_ids_buf = try utils.get_latents(allocator, io, platform_auto, transformer2d_model_ctx.config, output_image_dim, args.generator_type, args.random_seed);
    defer latent_buf.deinit();
    defer latent_ids_buf.deinit();

    // try tools.printFlatten(allocator, io, latent_buf, 20, "    Latents (first 20).", .{ .include_shape = true });
    // try tools.printFlatten(allocator, io, latent_ids_buf, 20, "    Latent_ids (first 20).", .{ .include_shape = true });

    log.info(">>> Preparing Timesteps...", .{});

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

    // try tools.printFlatten(allocator, io, latents_out, 20, "    Latents Out (first 20).", .{ .include_shape = true });

    log.info(">>> Decoding Latents...", .{});

    var vae_ctx = try autoencoder_kl.AutoencoderKLFlux2.loadFromFile(allocator, io, platform_auto, repo, null, .{});
    defer vae_ctx.deinit(allocator);

    const image_decoded_buf: zml.Buffer = try utils.variational_auto_encode(allocator, io, platform_auto, vae_ctx, latents_out);
    // try tools.printFlatten(allocator, io, image_decoded_buf, 20, "    Image Decoded (first 20).", .{ .include_shape = true });

    // Print directly in terminal without writing to disk

    const rgb_image_buffer = try utils.decodeImageToRgb(allocator, io, image_decoded_buf);
    defer rgb_image_buffer.free(allocator);

    if (args.kitty_output) {
        log.info(">>> Printing Image to Terminal...", .{});
        try utils.printFluxImageToTerminalKittyFromBuffer(allocator, &rgb_image_buffer);
    } else {
        log.warn("kitty_output flag not set, skipping terminal image output.", .{});
    }
    // Optional: save to disk
    if (args.output_image_path) |output_image_path| {
        log.info(">>> Saving Image to Disk at {s}...", .{output_image_path});
        try utils.saveFluxImageToPng(allocator, &rgb_image_buffer, output_image_path);
    } else {
        log.warn("No output_image_path provided, skipping saving image to disk.", .{});
    }

    log.info(">>> Pipeline Complete.", .{});

    std.process.exit(0);
}
