const std = @import("std");
const zml = @import("zml");

const Flux2Transformer2D = @import("flux2_transformer2d_model.zig").Flux2Transformer2D;
const FlowMatchEulerDiscreteScheduler = @import("scheduling_flow_match_euler_discrete.zig").FlowMatchEulerDiscreteScheduler;
const AutoencoderKLFlux2 = @import("autoencoder_kl_flux2.zig").AutoencoderKLFlux2;
const Qwen2TokenizerFast = @import("tokenization_qwen2_fast.zig").Qwen2TokenizerFast;
const Qwen3ForCausalLM = @import("modeling_qwen3.zig").Qwen3ForCausalLM;
const config = @import("config");
const tools = @import("tools.zig");
const utils = @import("utils.zig");
const interactive = @import("interactive.zig").interactive;

const stdx = zml.stdx;
const log = std.log.scoped(.flux2_main);

pub const std_options: std.Options = .{
    .log_level = .info,
};

// hf download black-forest-labs/FLUX.2-klein-4B --local-dir FLUX.2-klein-4B

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

// bazel run //examples/flux --//platforms:cuda=true -- \
// --model=/var/models/black-forest-labs/FLUX.2-klein-4B/ \
// --prompt="A photo of a cat" \
// --output-image-path=/home/kevin/flux_klein_zml_result.png \
// --resolution=FHD \
// --num-inference-steps=4 \
// --random-seed=0 \
// --generator-type=torch

// export CUDA_VISIBLE_DEVICES=1

const Resolution = enum { DBGD, HLD, LD, SD, HD, FHD, QHD, UHD };

const CliArgs = struct {
    model: []const u8 = "hf://black-forest-labs/FLUX.2-klein-4B",
    // model: []const u8 = "/Users/kevin/FLUX.2-klein-4B",
    // model: []const u8 = "/home/kevin/.cache/huggingface/hub/models--black-forest-labs--FLUX.2-klein-4B/snapshots/5e67da950fce4a097bc150c22958a05716994cea",
    // model: []const u8 = "/home/kevin/FLUX.2-klein-4B",

    prompt: []const u8 = "A photo of a cat",
    seqlen: usize = 512, // 512
    output_image_path: ?[]const u8 = null,
    kitty_output: bool = false,
    random_seed: u64 = 0,
    num_inference_steps: usize = 4,
    async_limit: ?usize = null,
    generator_type: utils.GeneratorType = .torch,
    interactive: bool = false,
    // 8K UHD 7680x4320
    // 4K QHD 3840x2160
    // FHD 1920x1080
    // HD 1280x720
    // SD 512x512
    // LD 256x256
    // HLD 128x128
    resolution: Resolution = .HD,

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

pub fn main(init: std.process.Init) !void {
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
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    if (args.output_image_path == null and !args.kitty_output) {
        log.err("No output method specified, the generated image will not be saved or displayed. Use --output-image-path=\"<path>\" or --kitty-output to save or display the image.", .{});
        std.process.exit(1);
    }

    if (args.interactive) {
        try interactive(allocator);
        return;
    }

    // var zml_compute_platform: *zml.Platform = try zml.Platform.init(allocator, io, .cpu, .{});
    var zml_compute_platform: *zml.Platform = try zml.Platform.auto(allocator, io, .{});
    defer zml_compute_platform.deinit(allocator);

    // var http_client: std.http.Client = .{
    //     .allocator = allocator,
    //     .io = threaded.io(),
    // };
    // try http_client.initDefaultProxies(arena);
    // defer http_client.deinit();

    // var vfs_file: zml.io.VFS.File = .init(allocator, threaded.io(), .{});
    // defer vfs_file.deinit();
    // try vfs.register("file", vfs_file.io());

    // var vfs_https: zml.io.VFS.HTTP = try .init(allocator, threaded.io(), &http_client, .https);
    // defer vfs_https.deinit();
    // try vfs.register("https", vfs_https.io());

    // var hf_vfs: zml.io.VFS.HF = try .auto(allocator, threaded.io(), &http_client);
    // defer hf_vfs.deinit();
    // try vfs.register("hf", hf_vfs.io());

    log.info("Resolving model repo", .{});
    const repo: std.Io.Dir = try zml.safetensors.resolveModelRepo(io, args.model);

    if (comptime config.enable_stage_debug) {
        log.info("Stage Debug Enabled", .{});
        std.process.exit(0);
    } else {
        log.info("Stage Debug Disabled", .{});
    }
    const timer_full_pipline = std.Io.Clock.awake.now(io);

    const parallelism_level: usize = if (args.async_limit) |limit| limit else try std.Thread.getCpuCount();

    log.info("Parallelism Level: {}", .{parallelism_level});

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    // ============ Model loader =================

    var qwen2_node = progress.start("Tokenizing Prompt", 0);
    var qwen2_future = io.concurrent(Qwen2TokenizerFast.pipelineRun, .{ allocator, io, repo, zml_compute_platform, &qwen2_node, args.prompt, args.seqlen }) catch |err| {
        log.err("Error running Qwen2TokenizerFast pipeline: {}", .{err});
        return err;
    };

    var qwen3_model_ctx_node = progress.start("Loading Qwen3", 0);
    var qwen3_model_ctx_future = io.concurrent(Qwen3ForCausalLM.loadFromFile, .{ allocator, io, zml_compute_platform, repo, parallelism_level, args.seqlen, &qwen3_model_ctx_node }) catch |err| {
        log.err("Error loading Qwen3 model: {}", .{err});
        return err;
    };

    var flux2_transformer2d_node = progress.start("Loading Flux2Transformer2D", 0);
    var transformer2d_model_ctx_future = io.concurrent(Flux2Transformer2D.loadFromFile, .{ allocator, io, zml_compute_platform, repo, parallelism_level, &flux2_transformer2d_node, .{} }) catch |err| {
        log.err("Error loading Flux2Transformer2D model: {}", .{err});
        return err;
    };

    var auto_encoder_node = progress.start("Loading AutoencoderKLFlux2", 0);
    var vae_ctx_future = io.concurrent(AutoencoderKLFlux2.loadFromFile, .{ allocator, io, zml_compute_platform, repo, &auto_encoder_node, .{} }) catch |err| {
        log.err("Error loading AutoencoderKLFlux2 model: {}", .{err});
        return err;
    };

    var scheduler_node = progress.start("Loading FlowMatchEulerDiscreteScheduler", 0);
    var scheduler_future = io.concurrent(FlowMatchEulerDiscreteScheduler.loadFromFile, .{ allocator, io, repo, &scheduler_node, .{} }) catch |err| {
        log.err("Error loading FlowMatchEulerDiscreteScheduler: {}", .{err});
        return err;
    };

    // ==================== Tokenizing Prompt ====================

    log.info("Qwen2 tokenizing prompt", .{});

    var tokens: Qwen2TokenizerFast.TokenizeOutput = try qwen2_future.await(io);
    qwen2_node.end();
    defer tokens.deinit();

    // try tools.printFlatten(allocator, io, tokens.input_ids, 20, "input_ids", .{ .include_shape = false });
    // try tools.printFlatten(allocator, io, tokens.attention_mask, 20, "attention_mask", .{ .include_shape = false });

    // ==================== Encoding Prompt ====================

    var qwen3_model_ctx: Qwen3ForCausalLM.ModelContext = try qwen3_model_ctx_future.await(io);
    qwen3_model_ctx_node.end();
    defer qwen3_model_ctx.deinit(allocator);

    var qwen3_node = progress.start("Running Qwen3", 0);
    const timer_qwen3_start = std.Io.Clock.awake.now(io);
    var embeding_output: Qwen3ForCausalLM.EmbedingOutput = try Qwen3ForCausalLM.pipelineRun(allocator, io, zml_compute_platform, qwen3_model_ctx, tokens);
    const qwen3_ns: i64 = @intCast(timer_qwen3_start.untilNow(io, .awake).toNanoseconds());
    const qwen3_time_ms = @as(f64, @floatFromInt(qwen3_ns)) / 1_000_000.0;
    log.info("Qwen3 completed in {d:.2} ms.", .{qwen3_time_ms});
    qwen3_node.end();
    defer embeding_output.deinit();

    // try tools.printFlatten(allocator, io, embeding_output.text_ids, 20, "    text_ids (first 20).", .{ .include_shape = true });
    // try tools.printFlatten(allocator, io, embeding_output.text_embedding, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

    // ==================== End Encoding Prompt ====================

    // std.process.exit(0);

    // {
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
    // }
    // std.process.exit(0);

    // // text_ids_selector is the pb
    // const text_ids_selector = prompt_text_ids;
    // const prompt_embeds_selector = prompt_embeds;

    // const text_ids_selector = text_ids_from_python;
    // const prompt_embeds_selector = prompt_embeds_from_python;

    const output_image_dim: utils.ResolutionInfo = switch (args.resolution) {
        .DBGD => .{ .width = 16, .height = 16 },
        .HLD => .{ .width = 128, .height = 128 },
        .LD => .{ .width = 256, .height = 256 },
        .SD => .{ .width = 512, .height = 512 },
        .HD => .{ .width = 1280, .height = 720 },
        .FHD => .{ .width = 1920, .height = 1080 },
        .QHD => .{ .width = 2560, .height = 1440 },
        .UHD => .{ .width = 3840, .height = 2160 },
    };

    var transformer2d_model_ctx: Flux2Transformer2D = try transformer2d_model_ctx_future.await(io);
    flux2_transformer2d_node.end();
    defer transformer2d_model_ctx.deinit(allocator);

    log.info(">> Preparing Latents...", .{});

    const timer_prepare_latents_start = std.Io.Clock.awake.now(io);
    var latent_buf, var latent_ids_buf = try utils.get_latents(allocator, io, zml_compute_platform, transformer2d_model_ctx.config, output_image_dim, args.generator_type, args.random_seed);
    const prepare_latents_ns: i64 = @intCast(timer_prepare_latents_start.untilNow(io, .awake).toNanoseconds());
    const prepare_latents_time_ms = @as(f64, @floatFromInt(prepare_latents_ns)) / 1_000_000.0;
    log.info("Latents prepared in {d:.2} ms.", .{prepare_latents_time_ms});

    defer latent_buf.deinit();
    defer latent_ids_buf.deinit();
    // try tools.printFlatten(allocator, io, latent_buf, 20, "    Latents (first 20).", .{ .include_shape = true });
    // try tools.printFlatten(allocator, io, latent_ids_buf, 20, "    Latent_ids (first 20).", .{ .include_shape = true });

    var scheduler: *FlowMatchEulerDiscreteScheduler = try scheduler_future.await(io);
    defer scheduler.deinit();
    scheduler_node.end();

    const timer_scheduler_start = std.Io.Clock.awake.now(io);
    var latents_out = try utils.schedule(
        transformer2d_model_ctx.model,
        transformer2d_model_ctx.weights,
        scheduler,
        latent_buf,
        latent_ids_buf,
        embeding_output.text_embedding,
        embeding_output.text_ids,
        args.num_inference_steps,
        allocator,
        io,
        zml_compute_platform,
    );
    const scheduler_ns: i64 = @intCast(timer_scheduler_start.untilNow(io, .awake).toNanoseconds());
    const scheduler_load_time_ms = @as(f64, @floatFromInt(scheduler_ns)) / 1_000_000.0;
    log.info("Scheduler completed in {d:.2} ms.", .{scheduler_load_time_ms});
    defer latents_out.deinit();

    // try tools.printFlatten(allocator, io, latents_out, 20, "    Latents Out (first 20).", .{ .include_shape = true });=

    log.info(">>> Decoding Latents...", .{});

    var vae_ctx: AutoencoderKLFlux2 = try vae_ctx_future.await(io);
    auto_encoder_node.end();
    defer vae_ctx.deinit(allocator);

    const timer_vae_start = std.Io.Clock.awake.now(io);
    const image_decoded_buf: zml.Buffer = try utils.variational_auto_encode(allocator, io, zml_compute_platform, vae_ctx, latents_out);
    const vae_ns: i64 = @intCast(timer_vae_start.untilNow(io, .awake).toNanoseconds());
    const vae_load_time_ms = @as(f64, @floatFromInt(vae_ns)) / 1_000_000.0;
    log.info("VAE completed in {d:.2} ms.", .{vae_load_time_ms});

    // try tools.printFlatten(allocator, io, image_decoded_buf, 20, "    Image Decoded (first 20).", .{ .include_shape = true });

    // Print directly in terminal without writing to disk

    const timer_rgb_start = std.Io.Clock.awake.now(io);
    const rgb_image_buffer = try utils.decodeImageToRgb(allocator, io, image_decoded_buf);
    const rgb_conversion_ns: i64 = @intCast(timer_rgb_start.untilNow(io, .awake).toNanoseconds());
    const rgb_conversion_time_ms = @as(f64, @floatFromInt(rgb_conversion_ns)) / 1_000_000.0;
    log.info("RGB conversion completed in {d:.2} ms.", .{rgb_conversion_time_ms});

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
    log.info("Full pipeline completed in {d:.2} ms", .{timer_full_pipline.untilNow(io, .awake).toMilliseconds()});

    // std.process.exit(0);
}
