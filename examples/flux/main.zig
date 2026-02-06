const std = @import("std");
const zml = @import("zml");
const tools = @import("tools.zig");
const utils = @import("utils.zig");
const flux_model_transformer2d = @import("flux2_transformer2d_model.zig");
const flow_match_euler_discrete_scheduler = @import("scheduling_flow_match_euler_discrete.zig");
const autoencoder_kl = @import("autoencoder_kl_flux2.zig");

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

    var platform_auto = try zml.Platform.init(allocator, io, .cpu, .{});
    defer platform_auto.deinit();

    const CliArgs = struct {
        model: []const u8 = "/Users/kevin/FLUX.2-klein-4B",
    };
    const args = stdx.flags.parseProcessArgs(CliArgs);

    // 1. Loading Embeds & Text IDs (Pre-computed for now)
    const embeds_path = "/Users/kevin/zml/flux_klein_notebook_embeds.npy";
    const text_ids_path = "/Users/kevin/zml/flux_klein_notebook_text_ids.npy";

    var prompt_embeds_buf = try loadInput(allocator, io, platform_auto, embeds_path, .{ 1, 20, 7680 });
    defer prompt_embeds_buf.deinit();
    var text_ids_buf = try loadInput(allocator, io, platform_auto, text_ids_path, .{ 1, 20, 4 });
    defer text_ids_buf.deinit();

    try tools.printFlatten(allocator, io, text_ids_buf, 20, "    text_ids (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, prompt_embeds_buf, 20, "    token_encoded_embeds (first 20).", .{ .include_shape = true });

    // 2. Load Transformer & Scheduler
    var transformer2d_model_ctx = try flux_model_transformer2d.loadFromFile(allocator, io, platform_auto, args.model);
    defer transformer2d_model_ctx.deinit(allocator);

    var scheduler = try flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler.loadFromFile(allocator, io, args.model);
    defer scheduler.deinit();

    log.info("Models initialized successfully", .{});

    // 3. Prepare Latents
    log.info(">>Preparing Latents...", .{});
    const img_dim = 128;

    const latent_buf, const latent_ids_buf = try utils.get_latents(allocator, io, platform_auto, transformer2d_model_ctx.config, img_dim);
    defer latent_buf.deinit();
    defer latent_ids_buf.deinit();

    try tools.printFlatten(allocator, io, latent_buf, 20, "    Latents (first 20).", .{ .include_shape = true });
    try tools.printFlatten(allocator, io, latent_ids_buf, 20, "    Latent_ids (first 20).", .{ .include_shape = true });

    // 4. Schedule (Sampling Loop)
    log.info("\n>>> Preparing Timesteps...", .{});
    if (enable_stage_debug) {
        try debugForwardStage(
            transformer2d_model_ctx.model,
            transformer2d_model_ctx.weights,
            latent_buf,
            latent_ids_buf,
            prompt_embeds_buf,
            text_ids_buf,
            allocator,
            io,
            platform_auto,
            stage_to_debug,
        );
        return;
    }

    const latents_out = try schedule(
        transformer2d_model_ctx.model,
        transformer2d_model_ctx.weights,
        scheduler,
        latent_buf,
        latent_ids_buf,
        prompt_embeds_buf,
        text_ids_buf,
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

    const image_decoded_buf = try variational_auto_encode(allocator, io, platform_auto, vae_ctx, latents_out);

    try tools.printFlatten(allocator, io, image_decoded_buf, 20, "    Image Decoded (first 20).", .{ .include_shape = true });

    try tools.saveBufferToNpy(allocator, io, platform_auto, image_decoded_buf, "/Users/kevin/zml/flux_klein_zml_result.npy");

    log.info("\n>>> Pipeline Complete.", .{});
    std.process.exit(0);
}

fn variational_auto_encode(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, vae_ctx: autoencoder_kl.AutoencoderKLFlux2.ModelContext, latents: zml.Buffer) !zml.Buffer {
    const VAEDecodeStep = struct {
        pub fn forward(self: @This(), model: autoencoder_kl.AutoencoderKLFlux2, latents_tensor: zml.Tensor) zml.Tensor {
            _ = self;
            return autoencoder_kl.VariationalAutoEncoder.forward(autoencoder_kl.VariationalAutoEncoder{}, model, latents_tensor);
        }
    };

    const latents_shape = latents.shape();
    const sym_latents = zml.Tensor.fromShape(latents_shape);

    log.info("Compiling VAE Decode...", .{});
    var vae_exe = try platform.compile(allocator, io, VAEDecodeStep{}, .forward, .{ vae_ctx.model, sym_latents });
    defer vae_exe.deinit();

    var vae_args = try vae_exe.args(allocator);
    defer vae_args.deinit(allocator);
    vae_args.set(.{ vae_ctx.weights, latents });

    var vae_res = try vae_exe.results(allocator);
    defer vae_res.deinit(allocator);

    vae_exe.call(vae_args, &vae_res);

    return vae_res.get(zml.Buffer);
}

fn loadInput(allocator: std.mem.Allocator, io: std.Io, platform: zml.Platform, path: []const u8, fallback_shape: anytype) !zml.Buffer {
    const npy = tools.NpyData.load(allocator, path) catch {
        log.warn("Input not found at {s}, using zeros", .{path});
        const shape = zml.Shape.init(fallback_shape, .f32);
        const data = try allocator.alloc(f32, shape.count());
        defer allocator.free(data);
        @memset(data, 0);
        return try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(data));
    };
    var npy_v = npy;
    defer npy_v.deinit();
    log.info("Loaded input from {s}", .{path});
    return try npy_v.toBuffer(io, platform);
}

pub fn schedule(
    transformer: flux_model_transformer2d.Flux2Transformer2DModel,
    weights: zml.Bufferized(flux_model_transformer2d.Flux2Transformer2DModel),
    scheduler: *flow_match_euler_discrete_scheduler.FlowMatchEulerDiscreteScheduler,
    latents: zml.Buffer,
    latent_ids: zml.Buffer,
    prompt_embeds: zml.Buffer,
    text_ids: zml.Buffer,
    num_inference_steps: usize,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: zml.Platform,
) !zml.Buffer {
    @setEvalBranchQuota(10_000);
    const image_seq_len: f64 = @floatFromInt(latents.shape().dim(1));

    const empirical_mu = utils.compute_empirical_mu(image_seq_len, @floatFromInt(num_inference_steps));

    var sigmas = try std.ArrayList(f32).initCapacity(allocator, num_inference_steps);
    defer sigmas.deinit(allocator);

    const start_sigma: f32 = 1.0;
    const end_sigma: f32 = 1.0 / @as(f32, @floatFromInt(num_inference_steps));
    const step_sigma = if (num_inference_steps > 1) (end_sigma - start_sigma) / @as(f32, @floatFromInt(num_inference_steps - 1)) else 0;

    for (0..num_inference_steps) |i| {
        sigmas.appendAssumeCapacity(start_sigma + step_sigma * @as(f32, @floatFromInt(i)));
    }

    try scheduler.set_timesteps(num_inference_steps, sigmas.items, empirical_mu, null);

    // Precompute frequencies and Rotary Embeddings (CPU)
    const freq_vals_arr = flux_model_transformer2d.computeTimestepFrequencies(transformer.config.timestep_guidance_channels);

    // --- RoPE Precompute ---
    const txt_shape = text_ids.shape();
    const img_shape = latent_ids.shape();
    const txt_len = txt_shape.dim(1);
    const img_len = img_shape.dim(1);
    const total_len = txt_len + img_len;

    var txt_ids_slice = try zml.Slice.alloc(allocator, txt_shape);
    defer txt_ids_slice.free(allocator);
    try text_ids.toSlice(io, txt_ids_slice);

    var img_ids_slice = try zml.Slice.alloc(allocator, img_shape);
    defer img_ids_slice.free(allocator);
    try latent_ids.toSlice(io, img_ids_slice);

    const comb_shape = zml.Shape.init(.{ .b = 1, .s = total_len, .coord = 4 }, .f32);
    // Combine IDs into one CPU buffer
    const txt_bytes = txt_ids_slice.items(u8);
    const img_bytes = img_ids_slice.items(u8);

    var comb_slice = try zml.Slice.alloc(allocator, comb_shape);
    defer comb_slice.free(allocator);
    const comb_bytes_slice = comb_slice.items(u8);

    const comb_f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(comb_bytes_slice)));

    // Txt IDs: i64 -> f32
    const txt_i64 = std.mem.bytesAsSlice(i64, @as([]align(8) u8, @alignCast(txt_bytes)));
    for (txt_i64, 0..) |v, i| {
        comb_f32[i] = @floatFromInt(v);
    }

    // Img IDs: i32 -> f32
    const img_i32 = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(img_bytes)));
    const offset = txt_i64.len;
    for (img_i32, 0..) |v, i| {
        comb_f32[offset + i] = @floatFromInt(v);
    }

    const ids_data_slice = comb_f32;
    std.log.info("RoPE Theta: {d}", .{transformer.config.rope_theta});

    const rope_slices = try flux_model_transformer2d.computeRotaryEmbedding(allocator, ids_data_slice, comb_shape, transformer.config.axes_dims_rope, transformer.config.rope_theta);
    defer rope_slices[0].free(allocator);
    defer rope_slices[1].free(allocator);

    var rotary_cos_dev = try zml.Buffer.fromSlice(io, platform, rope_slices[0]);
    var rotary_sin_dev = try zml.Buffer.fromSlice(io, platform, rope_slices[1]);
    defer rotary_cos_dev.deinit();
    defer rotary_sin_dev.deinit();

    // -----------------------

    log.info("   Sigmas: {any}", .{sigmas.items});
    log.info("   Empirical Mu: {d:.4}", .{empirical_mu});

    const FluxStep = struct {
        pub fn forward(self: @This(), model: flux_model_transformer2d.Flux2Transformer2DModel, hidden_states: zml.Tensor, encoder_hidden_states: zml.Tensor, timesteps_proj: zml.Tensor, guidance_proj: ?zml.Tensor, rotary_cos: zml.Tensor, rotary_sin: zml.Tensor) zml.Tensor {
            _ = self;
            return model.forward(hidden_states, encoder_hidden_states, timesteps_proj, guidance_proj, rotary_cos, rotary_sin);
        }
    };

    const latents_shape = latents.shape();
    const pixel_latents_shape = zml.Shape.init(.{ .b = 1, .s = latents_shape.dim(1), .d = latents_shape.dim(2) }, .f32);
    const t_proj_shape = zml.Shape.init(.{ .b = 1, .d = 256 }, .f32);
    const rotary_shape = rotary_cos_dev.shape();

    const sym_latents = zml.Tensor.fromShape(pixel_latents_shape);
    const sym_t_proj = zml.Tensor.fromShape(t_proj_shape);
    const sym_g_proj = zml.Tensor.fromShape(t_proj_shape);
    const sym_prompt = zml.Tensor.fromShape(prompt_embeds.shape());
    const sym_rotary_cos = zml.Tensor.fromShape(rotary_shape);
    const sym_rotary_sin = zml.Tensor.fromShape(rotary_shape);

    log.info("Compiling Step...", .{});
    var step_exe = try platform.compile(allocator, io, FluxStep{}, .forward, .{ transformer, sym_latents, sym_prompt, sym_t_proj, sym_g_proj, sym_rotary_cos, sym_rotary_sin });
    defer step_exe.deinit();

    var current_latents = latents;
    var is_first = true;
    scheduler.set_begin_index(0);

    for (scheduler.timesteps, 0..) |t_step, idx| {
        log.info("   Step {d}/{d} (t={d:.2})", .{ idx + 1, num_inference_steps, t_step });

        // Python output suggests t=1000 is used inside embedding (chaotic output).
        // transformer_flux2.py multiplies by 1000 inside forward.
        // So passing t_step (1000) is correct.
        const t_val: f32 = t_step;
        // Heap allocate to avoid stack-pointer issues with Buffer
        const t_proj_slice = try allocator.alloc(f32, 256);
        defer allocator.free(t_proj_slice);

        // Project t_val using freq_vals_arr
        for (0..128) |j| {
            const arg = t_val * freq_vals_arr[j];
            t_proj_slice[j] = @cos(arg);
            t_proj_slice[j + 128] = @sin(arg);
        }
        const t_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.sliceAsBytes(t_proj_slice));
        // defer t_proj_buf.deinit();

        // Guidance Projection
        const guidance_scale: f32 = 3.5;
        const g_val: f32 = guidance_scale * 1000.0;
        const g_proj_slice = try allocator.alloc(f32, 256);
        defer allocator.free(g_proj_slice);

        for (0..128) |j| {
            const arg = g_val * freq_vals_arr[j];
            g_proj_slice[j] = @cos(arg);
            g_proj_slice[j + 128] = @sin(arg);
        }
        const g_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.sliceAsBytes(g_proj_slice));
        // defer g_proj_buf.deinit();

        // Debug: Print Timestep Projection
        if (idx == 0) {
            std.log.info("Timestep Frequencies (first 10): {any}", .{freq_vals_arr[0..10]});
            try tools.printBuffer(allocator, io, t_proj_buf, 20, "T_Proj (Step 1)");
        }

        defer t_proj_buf.deinit();

        var args_step = try step_exe.args(allocator);
        defer args_step.deinit(allocator);

        // Debug: Print RoPE for verification
        try tools.printBuffer(allocator, io, rotary_cos_dev, 20, "RoPE Cos");
        try tools.printBuffer(allocator, io, rotary_sin_dev, 20, "RoPE Sin");

        args_step.set(.{ weights, current_latents, prompt_embeds, t_proj_buf, g_proj_buf, rotary_cos_dev, rotary_sin_dev });

        var res_step = try step_exe.results(allocator);
        defer res_step.deinit(allocator);

        step_exe.call(args_step, &res_step);

        const noise_pred_buf = res_step.get(zml.Buffer);

        const shape_flat = current_latents.shape().count();
        const noise_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{shape_flat}, .f32));
        defer noise_slice.free(allocator);
        try noise_pred_buf.toSlice(io, noise_slice);

        const latents_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{shape_flat}, .f32));
        defer latents_slice.free(allocator);
        try current_latents.toSlice(io, latents_slice);

        const n_data = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(noise_slice.items(u8))));
        const l_data = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(latents_slice.items(u8))));

        try tools.printFlatten(allocator, io, noise_pred_buf, 20, "   noise_pred (transformer output) (first 20)", .{});
        try tools.printFlatten(allocator, io, current_latents, 20, "   Latents before step.", .{ .include_shape = true });

        const out_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{shape_flat}, .f32));
        defer out_slice.free(allocator);
        const out_data = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(out_slice.items(u8))));

        try scheduler.step(n_data, t_step, l_data, out_data);

        const next_latents_buf = try zml.Buffer.fromBytes(io, platform, current_latents.shape(), std.mem.sliceAsBytes(out_data));

        if (!is_first) current_latents.deinit();
        current_latents = next_latents_buf;
        is_first = false;

        try tools.printFlatten(allocator, io, current_latents, 20, "   Latents after step (first 20).", .{ .include_shape = true });
    }

    // unpack_latents_with_ids - scatter tokens into correct spatial positions
    const unpacked = try unpackLatentsWithIds(allocator, io, platform, current_latents, latent_ids);

    // Print here to catch output before potential deinit crash
    try tools.printFlatten(allocator, io, unpacked, 20, "    Latents Out (first 20).", .{ .include_shape = true });

    if (!is_first) current_latents.deinit();

    return unpacked;
}

fn debugForwardStage(
    transformer: flux_model_transformer2d.Flux2Transformer2DModel,
    weights: zml.Bufferized(flux_model_transformer2d.Flux2Transformer2DModel),
    latents: zml.Buffer,
    latent_ids: zml.Buffer,
    prompt_embeds: zml.Buffer,
    text_ids: zml.Buffer,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: zml.Platform,
    stage: flux_model_transformer2d.Flux2Transformer2DModel.DebugStage,
) !void {
    const freq_vals_arr = flux_model_transformer2d.computeTimestepFrequencies(transformer.config.timestep_guidance_channels);

    const txt_shape = text_ids.shape();
    const img_shape = latent_ids.shape();
    const txt_len = txt_shape.dim(1);
    const img_len = img_shape.dim(1);
    const total_len = txt_len + img_len;

    var txt_ids_slice = try zml.Slice.alloc(allocator, txt_shape);
    defer txt_ids_slice.free(allocator);
    try text_ids.toSlice(io, txt_ids_slice);

    var img_ids_slice = try zml.Slice.alloc(allocator, img_shape);
    defer img_ids_slice.free(allocator);
    try latent_ids.toSlice(io, img_ids_slice);

    const comb_shape = zml.Shape.init(.{ .b = 1, .s = total_len, .coord = 4 }, .f32);
    var comb_slice = try zml.Slice.alloc(allocator, comb_shape);
    defer comb_slice.free(allocator);
    const comb_bytes = comb_slice.items(u8);
    const comb_f32 = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(comb_bytes)));

    const txt_i64 = std.mem.bytesAsSlice(i64, @as([]align(8) u8, @alignCast(txt_ids_slice.items(u8))));
    for (txt_i64, 0..) |v, i| {
        comb_f32[i] = @floatFromInt(v);
    }

    const img_i32 = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(img_ids_slice.items(u8))));
    const offset = txt_i64.len;
    for (img_i32, 0..) |v, i| {
        comb_f32[offset + i] = @floatFromInt(v);
    }

    const rope_slices = try flux_model_transformer2d.computeRotaryEmbedding(allocator, comb_f32, comb_shape, transformer.config.axes_dims_rope, transformer.config.rope_theta);
    defer rope_slices[0].free(allocator);
    defer rope_slices[1].free(allocator);

    var rotary_cos_dev = try zml.Buffer.fromSlice(io, platform, rope_slices[0]);
    defer rotary_cos_dev.deinit();
    var rotary_sin_dev = try zml.Buffer.fromSlice(io, platform, rope_slices[1]);
    defer rotary_sin_dev.deinit();

    const FluxDebugStep = struct {
        stage: flux_model_transformer2d.Flux2Transformer2DModel.DebugStage,

        pub fn forward(
            self: @This(),
            model: flux_model_transformer2d.Flux2Transformer2DModel,
            hidden_states: zml.Tensor,
            encoder_hidden_states: zml.Tensor,
            timesteps_proj: zml.Tensor,
            guidance_proj: ?zml.Tensor,
            rotary_cos: zml.Tensor,
            rotary_sin: zml.Tensor,
        ) zml.Tensor {
            return model.forwardStage(hidden_states, encoder_hidden_states, timesteps_proj, guidance_proj, rotary_cos, rotary_sin, self.stage);
        }
    };

    const latents_shape = latents.shape();
    const pixel_latents_shape = zml.Shape.init(.{ .b = 1, .s = latents_shape.dim(1), .d = latents_shape.dim(2) }, .f32);
    const t_proj_shape = zml.Shape.init(.{ .b = 1, .d = 256 }, .f32);
    const rotary_shape = rotary_cos_dev.shape();

    const sym_latents = zml.Tensor.fromShape(pixel_latents_shape);
    const sym_prompt = zml.Tensor.fromShape(prompt_embeds.shape());
    const sym_t_proj = zml.Tensor.fromShape(t_proj_shape);
    const sym_g_proj = zml.Tensor.fromShape(t_proj_shape);
    const sym_rotary_cos = zml.Tensor.fromShape(rotary_shape);
    const sym_rotary_sin = zml.Tensor.fromShape(rotary_shape);

    var step_exe = try platform.compile(allocator, io, FluxDebugStep{ .stage = stage }, .forward, .{ transformer, sym_latents, sym_prompt, sym_t_proj, sym_g_proj, sym_rotary_cos, sym_rotary_sin });
    defer step_exe.deinit();

    // Create Projection Buffers
    var t_proj_arr: [256]f32 = undefined;
    const t_val: f32 = 1000.0;
    for (0..128) |j| {
        const arg = t_val * freq_vals_arr[j];
        t_proj_arr[j] = @cos(arg);
        t_proj_arr[j + 128] = @sin(arg);
    }
    var t_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.asBytes(&t_proj_arr));
    defer t_proj_buf.deinit();

    // Guidance Projection
    const guidance_scale: f32 = 3.5;
    const g_val: f32 = guidance_scale * 1000.0;
    var g_proj_arr: [256]f32 = undefined;
    for (0..128) |j| {
        const arg = g_val * freq_vals_arr[j];
        g_proj_arr[j] = @cos(arg);
        g_proj_arr[j + 128] = @sin(arg);
    }
    var g_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.asBytes(&g_proj_arr));
    defer g_proj_buf.deinit();

    var args_step = try step_exe.args(allocator);
    defer args_step.deinit(allocator);
    args_step.set(.{ weights, latents, prompt_embeds, t_proj_buf, g_proj_buf, rotary_cos_dev, rotary_sin_dev });

    var res_step = try step_exe.results(allocator);
    defer res_step.deinit(allocator);
    step_exe.call(args_step, &res_step);

    const output_buf = res_step.get(zml.Buffer);

    // Print logic inside helper
    var label_buf: [256]u8 = undefined;
    const label = try std.fmt.bufPrint(&label_buf, "      [Block0] {s}", .{@tagName(stage)});
    try tools.printFlatten(allocator, io, output_buf, 20, label, .{});
}

/// Implements unpack_latents_with_ids from Python
/// Scatters latent tokens into correct spatial positions using position IDs
fn unpackLatentsWithIds(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: zml.Platform,
    latents: zml.Buffer,
    latent_ids: zml.Buffer,
) !zml.Buffer {
    const lat_shape = latents.shape(); // [B, seq_len, C]
    const ids_shape = latent_ids.shape(); // [B, seq_len, 4]

    const batch_size = lat_shape.dim(0);
    const seq_len = lat_shape.dim(1);
    const channels = lat_shape.dim(2);

    // Read latents and IDs to CPU
    var lat_slice = try zml.Slice.alloc(allocator, lat_shape);
    defer lat_slice.free(allocator);
    try latents.toSlice(io, lat_slice);

    var ids_slice = try zml.Slice.alloc(allocator, ids_shape);
    defer ids_slice.free(allocator);
    try latent_ids.toSlice(io, ids_slice);

    const lat_data = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(lat_slice.items(u8))));
    const ids_data = std.mem.bytesAsSlice(i32, @as([]align(4) u8, @alignCast(ids_slice.items(u8))));

    // Find max h and w from IDs (column 1 = h_ids, column 2 = w_ids)
    var max_h: i32 = 0;
    var max_w: i32 = 0;
    for (0..@intCast(seq_len)) |s| {
        const h_id = ids_data[s * 4 + 1];
        const w_id = ids_data[s * 4 + 2];
        if (h_id > max_h) max_h = h_id;
        if (w_id > max_w) max_w = w_id;
    }
    const h: usize = @intCast(max_h + 1);
    const w: usize = @intCast(max_w + 1);

    // Allocate output: [B, C, H, W]
    const out_shape = zml.Shape.init(.{ batch_size, channels, h, w }, .f32);
    const out_count = out_shape.count();
    const out_bytes = try allocator.alloc(f32, out_count);
    defer allocator.free(out_bytes);

    // Initialize to zero
    @memset(out_bytes, 0);

    // Scatter tokens into place
    for (0..@intCast(batch_size)) |b| {
        for (0..@intCast(seq_len)) |s| {
            const h_id: usize = @intCast(ids_data[b * @as(usize, @intCast(seq_len)) * 4 + s * 4 + 1]);
            const w_id: usize = @intCast(ids_data[b * @as(usize, @intCast(seq_len)) * 4 + s * 4 + 2]);

            for (0..@intCast(channels)) |c| {
                // Input: [B, seq_len, C]
                const in_idx = b * @as(usize, @intCast(seq_len)) * @as(usize, @intCast(channels)) +
                    s * @as(usize, @intCast(channels)) + c;
                // Output: [B, C, H, W]
                const out_idx = b * @as(usize, @intCast(channels)) * h * w +
                    c * h * w +
                    h_id * w +
                    w_id;
                out_bytes[out_idx] = lat_data[in_idx];
            }
        }
    }

    return try zml.Buffer.fromBytes(io, platform, out_shape, std.mem.sliceAsBytes(out_bytes));
}
