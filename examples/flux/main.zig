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
const Qwen3ForCausalLM = @import("modeling_qwen3.zig").Qwen3ForCausalLM;

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

    // print("\n>>> Tokenizing Prompt...")
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
    try tools.printFlatten(allocator, io, tokens.attention_mask, 20, "attention_mask", .{ .include_shape = false });

    log.info("End Tokenizing Prompt", .{});

    // Qwen3ForCausalLM
    std.process.exit(0);
}


// Do Not Remove
    // // 1. Loading Embeds & Text IDs (Pre-computed for now)
    // const embeds_path = "/Users/kevin/zml/flux_klein_notebook_embeds.npy";
    // const text_ids_path = "/Users/kevin/zml/flux_klein_notebook_text_ids.npy";

    // var prompt_embeds_buf = try utils.loadInput(allocator, io, platform_auto, embeds_path, .{ 1, 20, 7680 });
    // defer prompt_embeds_buf.deinit();
    // var text_ids_buf = try utils.loadInput(allocator, io, platform_auto, text_ids_path, .{ 1, 20, 4 });
    // defer text_ids_buf.deinit();

    // try tools.printFlatten(allocator, io, text_ids_buf, 20, "    text_ids (first 20).", .{ .include_shape = true });
    // try tools.printFlatten(allocator, io, prompt_embeds_buf, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

    // // 2. Load Transformer & Scheduler
    // var transformer2d_model_ctx = try flux_model_transformer2d.loadFromFile(allocator, io, platform_auto, args.model);
    // defer transformer2d_model_ctx.deinit(allocator);

    // var scheduler = try flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler.loadFromFile(allocator, io, args.model);
    // defer scheduler.deinit();

    // log.info("Models initialized successfully", .{});

    // // 3. Prepare Latents
    // log.info(">>Preparing Latents...", .{});
    // const img_dim = 128;

    // var latent_buf, var latent_ids_buf = try utils.get_latents(allocator, io, platform_auto, transformer2d_model_ctx.config, img_dim);
    // defer latent_buf.deinit();
    // defer latent_ids_buf.deinit();

    // try tools.printFlatten(allocator, io, latent_buf, 20, "    Latents (first 20).", .{ .include_shape = true });
    // try tools.printFlatten(allocator, io, latent_ids_buf, 20, "    Latent_ids (first 20).", .{ .include_shape = true });

    // // 4. Schedule (Sampling Loop)
    // log.info("\n>>> Preparing Timesteps...", .{});
    // if (enable_stage_debug) {
    //     try utils.debugForwardStage(
    //         transformer2d_model_ctx.model,
    //         transformer2d_model_ctx.weights,
    //         latent_buf,
    //         latent_ids_buf,
    //         prompt_embeds_buf,
    //         text_ids_buf,
    //         allocator,
    //         io,
    //         platform_auto,
    //         stage_to_debug,
    //     );
    //     return;
    // }

    // var latents_out = try utils.schedule(
    //     transformer2d_model_ctx.model,
    //     transformer2d_model_ctx.weights,
    //     scheduler,
    //     latent_buf,
    //     latent_ids_buf,
    //     prompt_embeds_buf,
    //     text_ids_buf,
    //     1, // num_inference_steps
    //     allocator,
    //     io,
    //     platform_auto,
    // );
    // defer latents_out.deinit();

    // try tools.printFlatten(allocator, io, latents_out, 20, "    Latents Out (first 20).", .{ .include_shape = true });

    // log.info("\n>>> Decoding Latents...", .{});

    // var vae_ctx = try autoencoder_kl.AutoencoderKLFlux2.loadFromFile(allocator, io, platform_auto, args.model);
    // defer vae_ctx.deinit(allocator);

    // const image_decoded_buf = try utils.variational_auto_encode(allocator, io, platform_auto, vae_ctx, latents_out);

    // try tools.printFlatten(allocator, io, image_decoded_buf, 20, "    Image Decoded (first 20).", .{ .include_shape = true });

    // try tools.saveBufferToNpy(allocator, io, platform_auto, image_decoded_buf, "/Users/kevin/zml/flux_klein_zml_result.npy");

    // log.info("\n>>> Pipeline Complete.", .{});
