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
const FluxPipeline = @import("pipeline.zig").FluxPipeline;

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
    prompt: []const u8 = "A photo of a cat",
    seqlen: usize = 512,
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
    defer if (debug_allocator) |*da| {
        const result = da.deinit();
        if (result != .ok) {
            log.warn("DebugAllocator detected memory leaks", .{});
        }
    };

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

    var zml_compute_platform: *zml.Platform = try zml.Platform.auto(allocator, io, .{});
    defer zml_compute_platform.deinit(allocator);

    log.info("Resolving model repo", .{});
    const repo: std.Io.Dir = try zml.safetensors.resolveModelRepo(io, args.model);

    if (comptime config.enable_stage_debug) {
        log.info("Stage Debug Enabled", .{});
        std.process.exit(0);
    } else {
        log.info("Stage Debug Disabled", .{});
    }
    const timer_full_pipline = std.Io.Clock.awake.now(io);
    defer log.info("Full pipeline completed in {d:.2} ms", .{timer_full_pipline.untilNow(io, .awake).toMilliseconds()});

    const parallelism_level: usize = if (args.async_limit) |limit| limit else try std.Thread.getCpuCount();

    log.info("Parallelism Level: {}", .{parallelism_level});

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    // ============ Model loader =================

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

    var qwen3_model_ctx_node = progress.start("Loading Qwen3", 0);
    var qwen3_model_ctx_future = io.concurrent(Qwen3ForCausalLM.loadFromFile, .{ allocator, io, zml_compute_platform, repo, parallelism_level, args.seqlen, &qwen3_model_ctx_node }) catch |err| {
        log.err("Error loading Qwen3 model: {}", .{err});
        return err;
    };

    var flux2_transformer2d_node = progress.start("Loading Flux2Transformer2D", 0);
    var transformer2d_model_ctx_future = io.concurrent(Flux2Transformer2D.loadFromFile, .{ allocator, io, zml_compute_platform, repo, parallelism_level, output_image_dim.height, output_image_dim.width, args.seqlen, &flux2_transformer2d_node, .{} }) catch |err| {
        log.err("Error loading Flux2Transformer2D model: {}", .{err});
        return err;
    };

    var scheduler_node = progress.start("Loading FlowMatchEulerDiscreteScheduler", 0);
    var scheduler_future = io.concurrent(FlowMatchEulerDiscreteScheduler.loadFromFile, .{ allocator, io, repo, &scheduler_node, .{} }) catch |err| {
        log.err("Error loading FlowMatchEulerDiscreteScheduler: {}", .{err});
        return err;
    };

    var auto_encoder_node = progress.start("Loading AutoencoderKLFlux2", 0);
    var vae_ctx_future = io.concurrent(AutoencoderKLFlux2.loadFromFile, .{ allocator, io, zml_compute_platform, repo, output_image_dim.height, output_image_dim.width, &auto_encoder_node, .{} }) catch |err| {
        log.err("Error loading AutoencoderKLFlux2 model: {}", .{err});
        return err;
    };

    // Tokenizer
    log.info("Loading Tokenizer...", .{});
    var tokenizer = try Qwen2TokenizerFast.fromPretrained(allocator, io, repo, .{ .subfolder = "tokenizer" });
    errdefer tokenizer.deinit();

    var qwen3_model_ctx: Qwen3ForCausalLM.ModelContext = try qwen3_model_ctx_future.await(io);
    qwen3_model_ctx_node.end();
    errdefer qwen3_model_ctx.deinit(allocator);

    var transformer2d_model_ctx: Flux2Transformer2D = try transformer2d_model_ctx_future.await(io);
    flux2_transformer2d_node.end();
    errdefer transformer2d_model_ctx.deinit(allocator);

    var scheduler: *FlowMatchEulerDiscreteScheduler = try scheduler_future.await(io);
    scheduler_node.end();
    errdefer scheduler.deinit();

    var vae_ctx: AutoencoderKLFlux2 = try vae_ctx_future.await(io);
    auto_encoder_node.end();
    errdefer vae_ctx.deinit(allocator);

    progress.end();

    var pipeline = FluxPipeline{
        .allocator = allocator,
        .io = io,
        .platform = zml_compute_platform,
        .tokenizer = tokenizer,
        .qwen3 = qwen3_model_ctx,
        .transformer = transformer2d_model_ctx,
        .scheduler = scheduler,
        .vae = vae_ctx,
        .config = .{
            .prompt = args.prompt,
            .output_image_path = args.output_image_path,
            .kitty_output = args.kitty_output,
            .num_inference_steps = args.num_inference_steps,
            .random_seed = args.random_seed,
            .generator_type = args.generator_type,
            .max_sequence_length = args.seqlen,
            .output_image_dim = output_image_dim,
        },
    };
    defer pipeline.deinit();

    if (args.interactive) {
        try interactive(allocator, &pipeline);
    } else {
        try pipeline.generate(null);
    }
}
