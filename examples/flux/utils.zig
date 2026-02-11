const std = @import("std");
const zml = @import("zml");
const tools = @import("tools.zig");
const flux_model = @import("flux2_transformer2d_model.zig");
const autoencoder_kl = @import("autoencoder_kl_flux2.zig");
const flux_model_transformer2d = @import("flux2_transformer2d_model.zig");
const flow_match_euler_discrete_scheduler = @import("scheduling_flow_match_euler_discrete.zig");

const c_interface = @import("c");

const stdx = zml.stdx;
const log = std.log.scoped(.utils);

pub const LatentPacker = struct {
    pub fn forward(input: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        var dummy: flux_model.Flux2Transformer2DModel = undefined;
        // Methods ignore self, so dummy is safe.
        const latents = dummy.pack_latents(input);
        const ids = dummy.prepare_latent_ids(input);
        return .{ latents, ids };
    }
};

pub fn pack_latents(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    latents_in: zml.Buffer,
) !struct { zml.Buffer, zml.Buffer } {
    const latents_tensor = zml.Tensor.fromShape(latents_in.shape());
    var exe = try zml.module.compile(allocator, io, LatentPacker.forward, .{latents_tensor}, platform);
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{latents_in});

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    const out_buffers = res.get(struct { zml.Buffer, zml.Buffer });
    return .{ out_buffers[0], out_buffers[1] };
}

pub fn compute_empirical_mu(image_seq_len: f64, num_steps: f64) f32 {
    const a1 = 8.73809524e-05;
    const b1 = 1.89833333;
    const a2 = 0.00016927;
    const b2 = 0.45666666;

    if (image_seq_len > 4300.0) {
        return @floatCast(a2 * image_seq_len + b2);
    }

    const m_200 = a2 * image_seq_len + b2;
    const m_10 = a1 * image_seq_len + b1;
    const a = (m_200 - m_10) / 190.0;
    const b = m_200 - 200.0 * a;
    return @floatCast(a * num_steps + b);
}

pub fn get_latents(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    config: flux_model.Config,
    img_dim: usize,
) !struct { zml.Buffer, zml.Buffer } {
    const in_channels = config.in_channels;
    const vae_scale_factor: usize = 8;

    const batch_size = 1;
    const num_channels_latents = @as(usize, @intCast(in_channels)) / 4;
    const adjusted_height = 2 * (img_dim / (vae_scale_factor * 2));
    const adjusted_width = 2 * (img_dim / (vae_scale_factor * 2));

    const shape_latents = [_]usize{ batch_size, num_channels_latents * 4, adjusted_height / 2, adjusted_width / 2 };

    // Generate Raw Latents on Host
    var rng = BoxMullerGenerator.init(0);
    const latents_raw_data = try rng.randn(allocator, &shape_latents);
    defer allocator.free(latents_raw_data);

    // Print raw first 20 for user verification
    std.log.info("Latents Raw (first 20): {any}", .{latents_raw_data[0..20]});

    // Upload to Device
    const latents_raw_shape = zml.Shape.init(.{ .b = @as(i64, @intCast(shape_latents[0])), .c = @as(i64, @intCast(shape_latents[1])), .h = @as(i64, @intCast(shape_latents[2])), .w = @as(i64, @intCast(shape_latents[3])) }, .f32);

    var latents_raw_buffer = try zml.Buffer.fromBytes(io, platform, latents_raw_shape, std.mem.sliceAsBytes(latents_raw_data));
    defer latents_raw_buffer.deinit();

    // Compile and Execute Packing
    const latents_tensor = zml.Tensor.fromShape(latents_raw_shape);
    var exe = try zml.module.compile(allocator, io, LatentPacker.forward, .{latents_tensor}, platform);
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{latents_raw_buffer});

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    const out_buffers = res.get(struct { zml.Buffer, zml.Buffer });
    return out_buffers;
}

pub const BoxMullerGenerator = struct {
    seed: i64,
    counter: i64,

    pub fn init(seed: i64) BoxMullerGenerator {
        return .{
            .seed = seed,
            .counter = 0,
        };
    }

    fn hash(self: BoxMullerGenerator, index: i64) f32 {
        var x = index +% self.seed;
        x = (x ^ (x >> 16)) *% 0x45d9f3b;
        x = (x ^ (x >> 16)) *% 0x45d9f3b;
        x = x ^ (x >> 16);

        // Normalize to [0, 1]
        // Python: (x % 1000000).float() / 1000000.0
        // Use @mod to emulate Python's modulo behavior (positive result).
        const m = @mod(x, 1000000);
        return @as(f32, @floatFromInt(m)) / 1000000.0;
    }

    pub fn randn(self: *BoxMullerGenerator, allocator: std.mem.Allocator, shape: []const usize) ![]f32 {
        var count: usize = 1;
        for (shape) |s| count *= s;

        const result = try allocator.alloc(f32, count);
        errdefer allocator.free(result);

        const constant_u2_offset = 0x9e3779b9;

        for (result, 0..) |*item, i| {
            // Python: indices = torch.arange(self.counter, self.counter + num_elements)
            const idx = self.counter + @as(i64, @intCast(i));

            // val1 = self._hash(indices)
            const val1 = self.hash(idx);
            // val2 = self._hash(indices + 0x9e3779b9)
            const val2 = self.hash(idx +% constant_u2_offset);

            // Box-Muller Transform
            // mag = sqrt(-2 ln(val1))
            const ln_val1 = @log(val1 + 1e-10);
            const mag = @sqrt(-2.0 * ln_val1);

            // dist = mag * cos(2pi * val2)
            const two_pi = 2.0 * 3.1415926535;
            const cos_val = @cos(two_pi * val2);

            item.* = mag * cos_val;
        }
        self.counter += @as(i64, @intCast(count));
        return result;
    }
};

pub fn prepare_text_ids(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, seq_len: usize) !zml.Buffer {
    const batch_size = 1;
    const count = batch_size * seq_len * 4;
    const data = try allocator.alloc(i64, count);
    defer allocator.free(data);

    // For Python default t_coord=None -> t=0
    for (0..seq_len) |s| {
        data[s * 4 + 0] = 0; // t
        data[s * 4 + 1] = 0; // h
        data[s * 4 + 2] = 0; // w
        data[s * 4 + 3] = @as(i64, @intCast(s)); // l
    }

    const shape = zml.Shape.init(.{ @as(i64, @intCast(batch_size)), @as(i64, @intCast(seq_len)), 4 }, .i64);

    return try zml.Buffer.fromBytes(io, platform, shape, std.mem.sliceAsBytes(data));
}

test "BoxMullerGenerator matches Python reference" {
    const allocator = std.testing.allocator;
    // Python: generator = BoxMullerGenerator(seed=42, device="cpu")
    var gen = BoxMullerGenerator.init(42);

    // Python: samples = generator.randn((2, 3, 4))
    const shape = [_]usize{ 2, 3, 4 };
    const samples = try gen.randn(allocator, &shape);
    defer allocator.free(samples);

    // Reference output from Python script
    const expected = [_]f32{
        -0.0603, 0.9283,  0.1725,  1.3793,
        0.5367,  -0.1172, 0.3583,  1.0377,
        2.6232,  0.8390,  -1.3245, -0.3546,
        -0.0135, -1.3908, -2.3215, 1.5630,
        -0.4535, -1.4687, -0.3142, 0.5771,
        1.2876,  -1.0406, -0.5745, -1.1020,
    };

    var max_err: f32 = 0;
    for (samples, 0..) |val, i| {
        const diff = @abs(val - expected[i]);
        if (diff > max_err) max_err = diff;

        // Relaxed tolerance due to minor float implementation diffs (e.g. log, cos precision)
        try std.testing.expectApproxEqAbs(expected[i], val, 1e-3);
    }
    // std.debug.print("Max difference: {d}\n", .{max_err});
}

pub fn variational_auto_encode(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, vae_ctx: autoencoder_kl.AutoencoderKLFlux2.ModelContext, latents: zml.Buffer) !zml.Buffer {
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

pub fn loadInput(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, path: []const u8, fallback_shape: anytype) !zml.Buffer {
    const npy = tools.NumpyData.load(allocator, path) catch {
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
    platform: *const zml.Platform,
) !zml.Buffer {
    @setEvalBranchQuota(10_000);
    const image_seq_len: f64 = @floatFromInt(latents.shape().dim(1));

    const empirical_mu = compute_empirical_mu(image_seq_len, @floatFromInt(num_inference_steps));

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
        var t_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.sliceAsBytes(t_proj_slice));
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
        var g_proj_buf = try zml.Buffer.fromBytes(io, platform, t_proj_shape, std.mem.sliceAsBytes(g_proj_slice));
        defer g_proj_buf.deinit();

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

/// Implements unpack_latents_with_ids from Python
/// Scatters latent tokens into correct spatial positions using position IDs
fn unpackLatentsWithIds(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
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

pub fn computeRoPE_CPU(allocator: std.mem.Allocator, seq_len: usize, head_dim: usize, theta: f32) !struct { cos: []f32, sin: []f32 } {
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
    const sin_data = try allocator.alloc(f32, total_elems);
    errdefer {
        allocator.free(cos_data);
        allocator.free(sin_data);
    }

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

pub fn saveFluxImageToPng(allocator: std.mem.Allocator, io: std.Io, image_decoded_buf: zml.Buffer, filename: []const u8) !void {
    const shape = image_decoded_buf.shape();
    const dim_c = shape.dim(1);
    _ = dim_c;
    const dim_h = shape.dim(2);
    const dim_w = shape.dim(3);

    const w: c_int = @intCast(dim_w);
    const h: c_int = @intCast(dim_h);
    const comp: c_int = 3;
    const stride_in_bytes: c_int = w * comp;

    // Fetch data from device
    const slice = try zml.Slice.alloc(allocator, shape);
    defer slice.free(allocator);
    try image_decoded_buf.toSlice(io, slice);

    const png_data = try allocator.alloc(u8, @intCast(dim_h * dim_w * comp));
    defer allocator.free(png_data);

    // NCHW -> NHWC + Denormalize
    var idx: usize = 0;
    for (0..@intCast(dim_h)) |y| {
        for (0..@intCast(dim_w)) |x| {
            for (0..@intCast(comp)) |c| {
                // Input index: 0, c, y, x
                const in_idx = c * (@as(usize, @intCast(dim_h)) * @as(usize, @intCast(dim_w))) +
                    y * @as(usize, @intCast(dim_w)) +
                    x;

                const val_f64 = tools.getElementAsF64(slice, shape.dtype(), in_idx);
                var val: f64 = (val_f64 / 2.0) + 0.5;
                if (val < 0.0) val = 0.0;
                if (val > 1.0) val = 1.0;

                png_data[idx] = @intFromFloat(val * 255.0);
                idx += 1;
            }
        }
    }

    const filename_z = try allocator.dupeZ(u8, filename);
    defer allocator.free(filename_z);

    if (c_interface.stbi_write_png(filename_z.ptr, w, h, comp, png_data.ptr, stride_in_bytes) == 0) {
        log.err("Failed to write PNG to {s}", .{filename});
    } else {
        log.info("Saved PNG to {s}", .{filename});
    }
}
