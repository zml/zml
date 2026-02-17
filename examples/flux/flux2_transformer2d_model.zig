const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;

const tools = @import("tools.zig");
const utils = @import("utils.zig");

const log = std.log.scoped(.flux2_model);

pub const Config = struct {
    patch_size: i64 = 1,
    in_channels: i64 = 128,
    out_channels: ?i64 = null,
    num_layers: i64 = 5,
    num_single_layers: i64 = 20,
    attention_head_dim: i64 = 128,
    num_attention_heads: i64 = 24,
    joint_attention_dim: i64 = 4096,
    timestep_guidance_channels: i64 = 256,
    mlp_ratio: f32 = 4.0,
    axes_dims_rope: [4]i64 = .{ 16, 56, 56, 64 },
    rope_theta: f32 = 10000.0,
    eps: f32 = 1e-6,
    guidance_embeds: bool = true,
    hidden_size: i64 = 3072,
};

// --- Helpers ---

const Linear = struct {
    inner: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Linear {
        const weight = store.createTensor("weight");
        const w = weight.withTags(.{ .out, .d });
        const bias = store.maybeCreateTensor("bias");
        return .{
            .inner = zml.nn.Linear.init(w, bias, .d),
        };
    }

    pub fn deinit(self: *@This()) void {
        _ = self; // autofix
    }

    pub fn forward(self: Linear, x: Tensor) Tensor {
        return self.inner.forward(x.withPartialTags(.{.d})).rename(.{ .out = .d });
    }
};

const LayerNorm = struct {
    inner: ?zml.nn.LayerNorm,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, elementwise_affine: bool) LayerNorm {
        if (elementwise_affine) {
            const weight = store.createTensor("weight");
            const bias = store.maybeCreateTensor("bias");
            return .{
                .inner = .{ .weight = weight, .bias = bias, .eps = eps },
                .eps = eps,
            };
        } else {
            return .{
                .inner = null,
                .eps = eps,
            };
        }
    }

    pub fn deinit(self: *@This()) void {
        _ = self; // autofix
    }

    pub fn forward(self: LayerNorm, x: Tensor) Tensor {
        if (self.inner) |ln| {
            return ln.forward(x);
        }
        return zml.nn.normalizeVariance(x, self.eps);
    }
};

/// RMSNorm: Root Mean Square Layer Normalization
/// Unlike LayerNorm, RMSNorm does NOT subtract the mean, and has no bias.
/// Formula: x / rms(x) * gamma, where rms(x) = sqrt(mean(x^2) + eps)
const RMSNorm = struct {
    weight: ?Tensor = null,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32, elementwise_affine: bool) RMSNorm {
        return .{
            .weight = if (elementwise_affine) store.createTensor("weight") else null,
            .eps = eps,
        };
    }

    pub fn deinit(self: *@This()) void {
        _ = self; // autofix
    }

    pub fn forward(self: RMSNorm, x: Tensor) Tensor {
        var out = zml.nn.rmsNorm(x, -1, self.eps);
        if (self.weight) |w| {
            const weight = w.convert(x.dtype()).broadcastLeft(out.shape());
            out = out.mul(weight);
        }
        return out;
    }
};

// --- Flux2 Modules ---

fn get_1d_rotary_pos_embed(dim: i64, pos: Tensor, theta: f32) struct { Tensor, Tensor } {
    const half = @divExact(dim, 2);

    // Symbolic frequency calculation using iota
    const shape = zml.Shape.init(.{half}, pos.dtype());
    const iota = Tensor.iota(shape, 0);

    // Using standard RoPE formula: theta ^ (-i / half)
    const log_theta = std.math.log(f32, theta, std.math.e);
    const exponent_factor = -log_theta / @as(f32, @floatFromInt(half));
    const exponents = iota.mul(Tensor.scalar(exponent_factor, pos.dtype()));
    const freq_half = exponents.exp(); // [half]

    // Interleave: [f0, f1] -> [f0, f0, f1, f1]
    // [half] -> [half, 1] -> broad [half, 2] -> reshape [dim]
    const freq = freq_half.reshape(.{ half, 1 }).broad(zml.Shape.init(.{ half, 2 }, pos.dtype())).reshape(.{dim});

    // pos: [S] -> [S, 1]
    // freq: [D] -> [1, D]
    // out: [S, D]
    const out = pos.reshape(.{ -1, 1 }).mul(freq.reshape(.{ 1, -1 }));
    return .{ out.cos(), out.sin() };
}

fn applyRoPE(x: Tensor, cos: Tensor, sin: Tensor) Tensor {
    const shape = x.shape();
    const last_dim = x.rank() - 1;
    const d = shape.dim(last_dim);
    const half = @divExact(d, 2);

    var new_shape_dims: [8]i64 = undefined;
    const rank = x.rank();
    for (0..rank - 1) |i| new_shape_dims[i] = shape.dim(i);
    new_shape_dims[rank - 1] = half;
    new_shape_dims[rank] = 2;

    const B = shape.dim(0);
    const L = shape.dim(1);
    const H = shape.dim(2);
    // D is dim(3).

    const x_pairs_reshaped = x.reshape(.{ B, L, H, half, 2 });

    // Split into even/odd
    const chunks = x_pairs_reshaped.split(4, &.{ 1, 1 }); // split last dim (rank 4)
    const x_even = chunks[0].reshape(.{ B, L, H, half }); // [B, L, H, half]
    const x_odd = chunks[1].reshape(.{ B, L, H, half });

    // x_twisted: [-odd, even]
    const x_twisted_even = x_odd.mul(Tensor.scalar(-1.0, .bf16));
    const x_twisted_odd = x_even;

    // Concatenate back to [B, L, H, half, 2]
    const x_twisted_pairs = Tensor.concatenate(&.{ x_twisted_even.reshape(.{ B, L, H, half, 1 }), x_twisted_odd.reshape(.{ B, L, H, half, 1 }) }, 4);
    const x_twisted = x_twisted_pairs.reshape(shape);

    // Apply
    // cos, sin are [L, D]. Broad to [B, L, H, D].
    const cos_broad = cos.broad(shape);
    const sin_broad = sin.broad(shape);

    return x.mul(cos_broad).add(x_twisted.mul(sin_broad));
}

fn scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor) Tensor {
    const shape = q.shape();
    const B = shape.dim(0);
    const L = shape.dim(1);
    const H = shape.dim(2);
    const D = shape.dim(3);
    const scale = 1.0 / @sqrt(@as(f32, @floatFromInt(D)));
    const q_scaled = q.mul(Tensor.scalar(scale, .bf16)).withPartialTags(.{ .b, .l, .h, .d });
    const k_tagged = k.withPartialTags(.{ .b, .m, .h, .d });
    // dot produces [B, H, L, M] (batching dims [B, H] first, then lhs non-batch [L], then rhs non-batch [M])
    const logits = q_scaled.dot(k_tagged, .d);
    const probs = logits.softmax(3);
    const v_tagged = v.withPartialTags(.{ .b, .m, .h, .d });
    // dot produces [B, H, L, D]
    const attn_out = probs.dot(v_tagged, .m);
    // Transpose from [B, H, L, D] to [B, L, H, D] to match Python
    const attn_transposed = attn_out.transpose(.{ 0, 2, 1, 3 });
    return attn_transposed.reshape(.{ B, L, H * D }).withPartialTags(.{ .b, .l, .d });
}

pub fn computeRotaryEmbedding(allocator: std.mem.Allocator, ids_data: []const f32, shape: zml.Shape, axes_dim: [4]i64, theta: f32) ![2]zml.Slice {
    const B = shape.dim(0);
    const S = shape.dim(1); // ids: [B, S, 4]

    // Total dim = sum(axes_dim)
    var total_dim: i64 = 0;
    for (axes_dim) |d| total_dim += d;

    // Output shape: [B, S, total_dim]
    const out_shape = zml.Shape.init(.{ .b = B, .s = S, .d = total_dim }, .f32);

    // Alloc temporary arrays
    const cos_slice = try zml.Slice.alloc(allocator, out_shape);
    const sin_slice = try zml.Slice.alloc(allocator, out_shape);

    const cos_data = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(cos_slice.items(u8))));
    const sin_data = std.mem.bytesAsSlice(f32, @as([]align(4) u8, @alignCast(sin_slice.items(u8))));

    var offset: usize = 0;
    // Iterate over axes
    for (axes_dim, 0..) |dim, axis_idx| {
        const half = @divExact(dim, 2);
        // Precompute freqs for this axis
        var freq_data: [128]f32 = undefined;
        // theta ^ (-i / half)
        // logic from get_1d_rotary_pos_embed
        for (0..@intCast(half)) |h| {
            const e = @as(f32, @floatFromInt(h)) / @as(f32, @floatFromInt(half));
            freq_data[h] = std.math.pow(f32, theta, -e);
        }

        // Interleave freqs: [f0, f0, f1, f1...]
        var freqs: [256]f32 = undefined;
        for (0..@intCast(half)) |h| {
            freqs[2 * h] = freq_data[h];
            freqs[2 * h + 1] = freq_data[h];
        }

        // Loop over B, S
        for (0..@intCast(B)) |b| {
            for (0..@intCast(S)) |s| {
                // Get id value for this axis
                // ids is [B, S, 4]. Flat index: b*(S*4) + s*4 + axis_idx
                const idx = b * (@as(usize, @intCast(S)) * 4) + s * 4 + axis_idx;
                const pos = ids_data[idx];

                // Compute cos/sin for this axis segment
                for (0..@intCast(dim)) |d| {
                    const arg = pos * freqs[d];

                    // Output index: b*(S*total_dim) + s*total_dim + offset + d
                    const out_idx = b * (@as(usize, @intCast(S)) * @as(usize, @intCast(total_dim))) + s * @as(usize, @intCast(total_dim)) + offset + d;

                    cos_data[out_idx] = @cos(arg);
                    sin_data[out_idx] = @sin(arg);
                }
            }
        }
        offset += @intCast(dim);
    }

    // Convert f32 to bf16
    const out_shape_bf16 = zml.Shape.init(.{ .b = B, .s = S, .d = total_dim }, .bf16);
    const cos_slice_bf16 = try zml.Slice.alloc(allocator, out_shape_bf16);
    const sin_slice_bf16 = try zml.Slice.alloc(allocator, out_shape_bf16);

    const cos_data_bf16 = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(cos_slice_bf16.items(u8))));
    const sin_data_bf16 = std.mem.bytesAsSlice(u16, @as([]align(2) u8, @alignCast(sin_slice_bf16.items(u8))));

    for (cos_data, 0..) |val, i| {
        const bits: u32 = @bitCast(val);
        cos_data_bf16[i] = @truncate(bits >> 16);
    }
    for (sin_data, 0..) |val, i| {
        const bits: u32 = @bitCast(val);
        sin_data_bf16[i] = @truncate(bits >> 16);
    }

    // Free the f32 slices
    cos_slice.free(allocator);
    sin_slice.free(allocator);

    return .{ cos_slice_bf16, sin_slice_bf16 };
}

pub fn computeTimestepFrequencies(dim: i64) [128]f32 {
    const half = @divExact(dim, 2);
    // Hardcoded max size to 128 (dim=256) for now, or return on stack buffer.
    var freq_data: [128]f32 = undefined;
    const log_v: f32 = -9.210340371976184; // -@log(10000.0)
    // Python diffusers Timesteps: exponent = -math.log(10000) / half_dim
    const factor: f32 = log_v / @as(f32, @floatFromInt(half));
    for (0..@intCast(half)) |i| {
        freq_data[i] = @exp(@as(f32, @floatFromInt(i)) * factor);
    }
    return freq_data;
}

fn getTimesteps(timesteps: Tensor, freq_vals: Tensor) Tensor {
    // timesteps: [B] -> [B, 1]
    // freq_vals: [half] -> [1, half] (broadcasts to [B, half] via mul)
    const out = timesteps.reshape(.{ -1, 1 }).mul(freq_vals.reshape(.{ 1, -1 }));
    return Tensor.concatenate(&.{ out.cos(), out.sin() }, -1);
}

pub const Flux2PosEmbed = struct {
    theta: f32,
    axes_dim: [4]i64,

    pub fn call(self: @This(), ids: Tensor) struct { Tensor, Tensor } {
        const S = ids.shape().dim(1);

        const ids_splits = ids.split(2, &.{ 1, 1, 1, 1 });

        var cos_list: [4]Tensor = undefined;
        var sin_list: [4]Tensor = undefined;

        for (self.axes_dim, 0..) |dim, i| {
            const pos = ids_splits[i].reshape(.{S});
            const res = get_1d_rotary_pos_embed(dim, pos, self.theta);
            cos_list[i] = res[0];
            sin_list[i] = res[1];
        }

        const cos = Tensor.concatenate(&cos_list, -1);
        const sin = Tensor.concatenate(&sin_list, -1);

        return .{ cos, sin };
    }
};

const TimestepEmbedding = struct {
    linear_1: Linear,
    linear_2: Linear,

    pub fn init(store: zml.io.TensorStore.View) @This() {
        return .{
            .linear_1 = Linear.init(store.withPrefix("linear_1")),
            .linear_2 = Linear.init(store.withPrefix("linear_2")),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.linear_1.deinit();
        self.linear_2.deinit();
    }

    pub fn forward(self: @This(), sample: Tensor) Tensor {
        var x = self.linear_1.forward(sample);
        x = x.silu();
        x = self.linear_2.forward(x);
        return x;
    }
};

const Flux2TimestepGuidanceEmbeddings = struct {
    channels: i64,
    timestep_embedder: TimestepEmbedding,
    guidance_embedder: ?TimestepEmbedding,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Flux2TimestepGuidanceEmbeddings {
        return .{
            .channels = config.timestep_guidance_channels,
            .timestep_embedder = TimestepEmbedding.init(store.withPrefix("timestep_embedder")),
            .guidance_embedder = if (config.guidance_embeds)
                TimestepEmbedding.init(store.withPrefix("guidance_embedder"))
            else
                null,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.timestep_embedder.deinit();
        if (self.guidance_embedder) |*ge| ge.deinit();
    }

    pub fn forward(self: @This(), timesteps_proj: Tensor, guidance_proj: ?Tensor) Tensor {
        var timesteps_emb = self.timestep_embedder.forward(timesteps_proj);

        if (guidance_proj) |g| {
            if (self.guidance_embedder) |ge| {
                const guidance_emb = ge.forward(g);
                timesteps_emb = timesteps_emb.add(guidance_emb);
            }
        }
        return timesteps_emb;
    }
};

const Flux2Modulation = struct {
    linear: Linear,
    mod_param_sets: i64,

    pub fn init(store: zml.io.TensorStore.View, mod_param_sets: i64) Flux2Modulation {
        return .{
            .linear = Linear.init(store.withPrefix("linear")),
            .mod_param_sets = mod_param_sets,
        };
    }

    pub fn deinit(self: *@This()) void {
        self.linear.deinit();
    }

    pub fn forward(self: @This(), temb: Tensor) stdx.BoundedArray(Tensor, 32) {
        var x = temb.silu();
        x = self.linear.forward(x);

        const last_dim = x.rank() - 1;
        const total_params = 3 * self.mod_param_sets;
        const dim = @divExact(x.shape().dim(last_dim), total_params);

        var split_sizes_buf: [32]i64 = undefined;
        const split_sizes = split_sizes_buf[0..@intCast(total_params)];
        for (split_sizes) |*s| s.* = dim;

        const chunks = x.split(last_dim, split_sizes);
        var result = stdx.BoundedArray(Tensor, 32).init(0) catch unreachable;
        for (chunks) |c| {
            // Reshape to [B, 1, D] for broadcasting over sequence dimension (matching Python)
            const b = c.shape().dim(0);
            const d = c.shape().dim(1);
            result.append(c.reshape(.{ b, 1, d }).withPartialTags(.{ .b, .s, .d })) catch unreachable;
        }
        return result;
    }
};

const Flux2Attention = struct {
    num_heads: i64,
    head_dim: i64,

    to_q: Linear,
    to_k: Linear,
    to_v: Linear,

    add_q_proj: Linear,
    add_k_proj: Linear,
    add_v_proj: Linear,

    to_out: Linear,
    to_add_out: Linear,

    norm_q: RMSNorm,
    norm_k: RMSNorm,
    norm_added_q: RMSNorm,
    norm_added_k: RMSNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Flux2Attention {
        return .{
            .num_heads = config.num_attention_heads,
            .head_dim = config.attention_head_dim,

            .to_q = Linear.init(store.withPrefix("to_q")),
            .to_k = Linear.init(store.withPrefix("to_k")),
            .to_v = Linear.init(store.withPrefix("to_v")),

            .add_q_proj = Linear.init(store.withPrefix("add_q_proj")),
            .add_k_proj = Linear.init(store.withPrefix("add_k_proj")),
            .add_v_proj = Linear.init(store.withPrefix("add_v_proj")),

            .to_out = Linear.init(store.withPrefix("to_out.0")),
            .to_add_out = Linear.init(store.withPrefix("to_add_out")),

            // Use RMSNorm to match Python's torch.nn.RMSNorm
            .norm_q = RMSNorm.init(store.withPrefix("norm_q"), config.eps, true),
            .norm_k = RMSNorm.init(store.withPrefix("norm_k"), config.eps, true),
            .norm_added_q = RMSNorm.init(store.withPrefix("norm_added_q"), config.eps, true),
            .norm_added_k = RMSNorm.init(store.withPrefix("norm_added_k"), config.eps, true),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.to_q.deinit();
        self.to_k.deinit();
        self.to_v.deinit();
        self.add_q_proj.deinit();
        self.add_k_proj.deinit();
        self.add_v_proj.deinit();
        self.to_out.deinit();
        self.to_add_out.deinit();
        self.norm_q.deinit();
        self.norm_k.deinit();
        self.norm_added_q.deinit();
        self.norm_added_k.deinit();
    }

    pub fn forward(
        self: Flux2Attention,
        img: Tensor,
        txt: Tensor,
        rotary_emb: struct { Tensor, Tensor },
    ) struct { Tensor, Tensor } {
        const B = img.shape().dim(0);
        const L_img = img.shape().dim(1);
        const L_txt = txt.shape().dim(1);
        const H = self.num_heads;
        const D = self.head_dim;

        // Project Q, K, V for image stream
        const q_img_proj = self.to_q.forward(img);
        const k_img_proj = self.to_k.forward(img);
        const v_img_proj = self.to_v.forward(img);

        // Project Q, K, V for text stream
        const q_txt_proj = self.add_q_proj.forward(txt);
        const k_txt_proj = self.add_k_proj.forward(txt);
        const v_txt_proj = self.add_v_proj.forward(txt);

        // Reshape to [B, L, H, D] BEFORE normalizing (matching Python)
        const q_img_reshaped = q_img_proj.reshape(.{ B, L_img, H, D });
        const k_img_reshaped = k_img_proj.reshape(.{ B, L_img, H, D });
        const v_img_reshaped = v_img_proj.reshape(.{ B, L_img, H, D });

        const q_txt_reshaped = q_txt_proj.reshape(.{ B, L_txt, H, D });
        const k_txt_reshaped = k_txt_proj.reshape(.{ B, L_txt, H, D });
        const v_txt_reshaped = v_txt_proj.reshape(.{ B, L_txt, H, D });

        // Apply RMSNorm (on reshaped tensor, matching Python)
        const q_img_normed = self.norm_q.forward(q_img_reshaped);
        const k_img_normed = self.norm_k.forward(k_img_reshaped);

        const q_txt_normed = self.norm_added_q.forward(q_txt_reshaped);
        const k_txt_normed = self.norm_added_k.forward(k_txt_reshaped);

        // Concatenate text then image (Python: torch.cat([encoder_query, query], dim=1))
        var q = Tensor.concatenate(&.{ q_txt_normed, q_img_normed }, 1);
        var k = Tensor.concatenate(&.{ k_txt_normed, k_img_normed }, 1);
        const v = Tensor.concatenate(&.{ v_txt_reshaped, v_img_reshaped }, 1);

        // Apply RoPE
        const cos = rotary_emb[0].withPartialTags(.{ .l, .d });
        const sin = rotary_emb[1].withPartialTags(.{ .l, .d });
        q = applyRoPE(q.withPartialTags(.{ .b, .l, .h, .d }), cos, sin);
        k = applyRoPE(k.withPartialTags(.{ .b, .l, .h, .d }), cos, sin);

        // SDPA
        var out = scaled_dot_product_attention(q, k, v);

        // Flatten from [B, L, H, D] to [B, L, H*D] (matching Python line 205)
        const out_flat = out.reshape(.{ B, L_txt + L_img, H * D });

        // Split back and project
        const txt_slice = out_flat.slice(&.{ .{}, .{ .end = L_txt }, .{} });
        const out_txt = self.to_add_out.forward(txt_slice);
        const out_img = self.to_out.forward(out_flat.slice(&.{ .{}, .{ .start = L_txt }, .{} }));
        return .{ out_img, out_txt };
    }
};

const Flux2TransformerBlock = struct {
    norm1: LayerNorm,
    norm1_context: LayerNorm,
    norm2: LayerNorm,
    norm2_context: LayerNorm,
    ff_linear_in: Linear,
    ff_linear_out: Linear,
    ff_context_linear_in: Linear,
    ff_context_linear_out: Linear,
    attn: Flux2Attention,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Flux2TransformerBlock {
        return .{
            .norm1 = LayerNorm.init(store.withPrefix("norm1"), config.eps, false),
            .norm1_context = LayerNorm.init(store.withPrefix("norm1_context"), config.eps, false),
            .norm2 = LayerNorm.init(store.withPrefix("norm2"), config.eps, false),
            .norm2_context = LayerNorm.init(store.withPrefix("norm2_context"), config.eps, false),
            .ff_linear_in = Linear.init(store.withPrefix("ff.linear_in")),
            .ff_linear_out = Linear.init(store.withPrefix("ff.linear_out")),
            .ff_context_linear_in = Linear.init(store.withPrefix("ff_context.linear_in")),
            .ff_context_linear_out = Linear.init(store.withPrefix("ff_context.linear_out")),
            .attn = Flux2Attention.init(store.withPrefix("attn"), config),
        };
    }

    pub fn deinit(self: *Flux2TransformerBlock) void {
        self.norm1.deinit();
        self.norm1_context.deinit();
        self.norm2.deinit();
        self.norm2_context.deinit();
        self.ff_linear_in.deinit();
        self.ff_linear_out.deinit();
        self.ff_context_linear_in.deinit();
        self.ff_context_linear_out.deinit();
        self.attn.deinit();
    }

    fn geglu(input: Tensor) Tensor {
        const last_dim = input.rank() - 1;
        const d = input.shape().dim(last_dim);
        const half = @divExact(d, 2);
        const chunks = input.split(last_dim, &.{ half, half });
        return chunks[0].silu().mul(chunks[1]);
    }

    pub fn forward(
        self: Flux2TransformerBlock,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        temb_mod_params_img: []const Tensor,
        temb_mod_params_txt: []const Tensor,
        rotary_emb: struct { Tensor, Tensor },
    ) struct { Tensor, Tensor } {
        // Modulation parameters: (shift, scale, gate)
        const shift_msa = temb_mod_params_img[0];
        const scale_msa = temb_mod_params_img[1];
        const gate_msa = temb_mod_params_img[2];

        const shift_mlp = temb_mod_params_img[3];
        const scale_mlp = temb_mod_params_img[4];
        const gate_mlp = temb_mod_params_img[5];

        const c_shift_msa = temb_mod_params_txt[0];
        const c_scale_msa = temb_mod_params_txt[1];
        const c_gate_msa = temb_mod_params_txt[2];

        const c_shift_mlp = temb_mod_params_txt[3];
        const c_scale_mlp = temb_mod_params_txt[4];
        const c_gate_mlp = temb_mod_params_txt[5];

        // Img stream: norm1 + modulation
        var norm_hidden_states = self.norm1.forward(hidden_states).withPartialTags(.{ .b, .s, .d });
        norm_hidden_states = norm_hidden_states.mul(scale_msa.add(Tensor.scalar(1.0, .bf16)).broad(norm_hidden_states.shape()))
            .add(shift_msa.broad(norm_hidden_states.shape()));

        // Txt stream: norm1_context + modulation
        var norm_encoder_hidden_states = self.norm1_context.forward(encoder_hidden_states).withPartialTags(.{ .b, .s, .d });
        norm_encoder_hidden_states = norm_encoder_hidden_states.mul(c_scale_msa.add(Tensor.scalar(1.0, .bf16)).broad(norm_encoder_hidden_states.shape()))
            .add(c_shift_msa.broad(norm_encoder_hidden_states.shape()));

        // Attention
        const attn_outputs = self.attn.forward(norm_hidden_states, norm_encoder_hidden_states, rotary_emb);
        const attn_out = gate_msa.broad(attn_outputs[0].shape()).mul(attn_outputs[0]);
        const context_attn_out = c_gate_msa.broad(attn_outputs[1].shape()).mul(attn_outputs[1]);

        // Res + MLP (Img)
        var h = hidden_states.add(attn_out);
        var norm_h = self.norm2.forward(h).withPartialTags(.{ .b, .s, .d });
        norm_h = norm_h.mul(scale_mlp.add(Tensor.scalar(1.0, .bf16)).broad(norm_h.shape()))
            .add(shift_mlp.broad(norm_h.shape()));

        var ff_out = self.ff_linear_in.forward(norm_h);
        ff_out = geglu(ff_out);
        ff_out = self.ff_linear_out.forward(ff_out);
        const out_hidden_states = h.add(gate_mlp.broad(ff_out.shape()).mul(ff_out));

        // Res + MLP (Txt)
        var c = encoder_hidden_states.add(context_attn_out);
        var norm_c = self.norm2_context.forward(c).withPartialTags(.{ .b, .s, .d });
        norm_c = norm_c.mul(c_scale_mlp.add(Tensor.scalar(1.0, .bf16)).broad(norm_c.shape()))
            .add(c_shift_mlp.broad(norm_c.shape()));

        var ff_c_out = self.ff_context_linear_in.forward(norm_c);
        ff_c_out = geglu(ff_c_out);
        ff_c_out = self.ff_context_linear_out.forward(ff_c_out);
        const out_encoder_hidden_states = c.add(c_gate_mlp.broad(ff_c_out.shape()).mul(ff_c_out));

        return .{ out_encoder_hidden_states, out_hidden_states };
    }
};

const Flux2SingleTransformerBlock = struct {
    norm: LayerNorm,
    linear1: Linear,
    linear2: Linear,

    // RMSNorm for Q and K (matching Python's Flux2ParallelSelfAttention)
    norm_q: RMSNorm,
    norm_k: RMSNorm,

    hidden_dim: i64,
    mlp_dim: i64,
    num_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Flux2SingleTransformerBlock {
        // Calculate dimensions
        // hidden_size = config.hidden_size (3072)
        // mlp_ratio = config.mlp_ratio (3)
        // mlp_dim = hidden_size * mlp_ratio (9216)
        // BUT we need to be careful with int types.
        const h = config.hidden_size;
        const m = @as(i64, @intFromFloat(@as(f32, @floatFromInt(h)) * config.mlp_ratio));

        return .{
            .norm = LayerNorm.init(store.withPrefix("norm"), config.eps, false),
            .linear1 = Linear.init(store.withPrefix("attn.to_qkv_mlp_proj")),
            .linear2 = Linear.init(store.withPrefix("attn.to_out")),
            // RMSNorm for Q and K (Python: attn.norm_q, attn.norm_k)
            .norm_q = RMSNorm.init(store.withPrefix("attn.norm_q"), config.eps, true),
            .norm_k = RMSNorm.init(store.withPrefix("attn.norm_k"), config.eps, true),
            .hidden_dim = h,
            .mlp_dim = @intCast(m),
            .num_heads = config.num_attention_heads,
            .head_dim = config.attention_head_dim,
        };
    }

    pub fn deinit(self: *@This()) void {
        _ = self; // autofix
    }

    fn geglu(input: Tensor) Tensor {
        const last_dim = input.rank() - 1;
        const d = input.shape().dim(last_dim);
        const half = @divExact(d, 2);
        const chunks = input.split(last_dim, &.{ half, half });
        return chunks[0].silu().mul(chunks[1]);
    }

    pub fn forward(
        self: Flux2SingleTransformerBlock,
        hidden_states: Tensor,
        temb_mod_params: []const Tensor,
        rotary_emb: struct { Tensor, Tensor },
    ) Tensor {
        // Modulation: (shift, scale, gate)
        const B = hidden_states.shape().dim(0);
        const D = self.hidden_dim;

        const mod_shift = temb_mod_params[0].reshape(.{ B, 1, D });
        const mod_scale = temb_mod_params[1].reshape(.{ B, 1, D });
        const mod_gate = temb_mod_params[2].reshape(.{ B, 1, D });

        var h = self.norm.forward(hidden_states);
        h = h.mul(mod_scale.add(Tensor.scalar(1.0, .bf16)).broad(h.shape())).add(mod_shift.broad(h.shape()));

        var qkv_mlp = self.linear1.forward(h);

        // Split QKV (3*hidden) and MLP (2*mlp_dim)
        const qkv_dim = self.hidden_dim * 3;
        const mlp_geglu_dim = self.mlp_dim * 2;

        const last_dim = qkv_mlp.rank() - 1;
        const chunks = qkv_mlp.split(last_dim, &.{ qkv_dim, mlp_geglu_dim });
        const qkv = chunks[0].withPartialTags(.{ .b, .s, .d });
        const mlp_in = chunks[1].withPartialTags(.{ .b, .s, .d });

        const qkv_chunks = qkv.split(last_dim, &.{ self.hidden_dim, self.hidden_dim, self.hidden_dim });

        // Reshape to [B, L, H, D] BEFORE applying RMSNorm (matching Python)
        const L = hidden_states.shape().dim(1);
        var q = qkv_chunks[0].reshape(.{ .b = B, .s = L, .h = self.num_heads, .d = self.head_dim });
        var k = qkv_chunks[1].reshape(.{ .b = B, .s = L, .h = self.num_heads, .d = self.head_dim });
        const v = qkv_chunks[2].reshape(.{ .b = B, .s = L, .h = self.num_heads, .d = self.head_dim });

        // Apply RMSNorm to Q and K (matching Python lines 287-288)
        q = self.norm_q.forward(q);
        k = self.norm_k.forward(k);

        // Apply RoPE
        const q_rope = applyRoPE(q, rotary_emb[0], rotary_emb[1]);
        const k_rope = applyRoPE(k, rotary_emb[0], rotary_emb[1]);

        const attn_out_raw = scaled_dot_product_attention(q_rope, k_rope, v);
        const attn_out = attn_out_raw.reshape(qkv_chunks[0].shape());

        // MLP
        const mlp_out = geglu(mlp_in);

        // Concat
        const joint_out = Tensor.concatenate(&.{ attn_out, mlp_out }, last_dim);
        const x = self.linear2.forward(joint_out);

        return hidden_states.add(mod_gate.broad(x.shape()).mul(x));
    }
};

const AdaLayerNormContinuous = struct {
    norm: LayerNorm,
    linear: Linear,

    pub fn init(store: zml.io.TensorStore.View, config: Config) @This() {
        return .{
            .norm = LayerNorm.init(store.withPrefix("norm"), config.eps, false),
            .linear = Linear.init(store.withPrefix("linear")),
        };
    }

    pub fn deinit(self: *@This()) void {
        self.norm.deinit();
        self.linear.deinit();
    }

    pub fn forward(self: @This(), x: Tensor, temb: Tensor) Tensor {
        const mod = self.linear.forward(temb.silu());
        const last_dim = mod.rank() - 1;
        const dim = @divExact(mod.shape().dim(last_dim), 2);
        const chunks = mod.split(last_dim, &.{ dim, dim });
        const scale = chunks[0];
        const shift = chunks[1];

        var out = self.norm.forward(x).withPartialTags(.{ .b, .s, .d });
        const scale_b = scale.add(Tensor.scalar(1.0, .bf16));
        out = out.mul(scale_b.broad(out.shape())).add(shift.broad(out.shape()));
        return out;
    }
};

pub const Flux2Transformer2DModel = struct {
    config: Config,

    // Layers
    pos_embed: Flux2PosEmbed,
    time_guidance_embed: Flux2TimestepGuidanceEmbeddings,

    double_stream_modulation_img: Flux2Modulation,
    double_stream_modulation_txt: Flux2Modulation,
    single_stream_modulation: Flux2Modulation,

    x_embedder: Linear,
    context_embedder: Linear,

    transformer_blocks: []Flux2TransformerBlock,
    single_transformer_blocks: []Flux2SingleTransformerBlock,

    norm_out: AdaLayerNormContinuous,
    proj_out: Linear,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !@This() {
        const blocks = try allocator.alloc(Flux2TransformerBlock, @intCast(config.num_layers));
        for (blocks, 0..) |*b, i| {
            var buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "transformer_blocks.{d}", .{i});
            b.* = Flux2TransformerBlock.init(store.withPrefix(name), config);
        }

        const single_blocks = try allocator.alloc(Flux2SingleTransformerBlock, @intCast(config.num_single_layers));
        for (single_blocks, 0..) |*b, i| {
            var buf: [64]u8 = undefined;
            const name = try std.fmt.bufPrint(&buf, "single_transformer_blocks.{d}", .{i});
            b.* = Flux2SingleTransformerBlock.init(store.withPrefix(name), config);
        }

        const model = @This(){
            .config = config,
            .pos_embed = .{ .theta = config.rope_theta, .axes_dim = config.axes_dims_rope }, // Fix axes_dim naming match
            .time_guidance_embed = Flux2TimestepGuidanceEmbeddings.init(store.withPrefix("time_guidance_embed"), config),

            .double_stream_modulation_img = Flux2Modulation.init(store.withPrefix("double_stream_modulation_img"), 2),
            .double_stream_modulation_txt = Flux2Modulation.init(store.withPrefix("double_stream_modulation_txt"), 2),
            .single_stream_modulation = Flux2Modulation.init(store.withPrefix("single_stream_modulation"), 1),

            .x_embedder = Linear.init(store.withPrefix("x_embedder")),
            .context_embedder = Linear.init(store.withPrefix("context_embedder")),

            .transformer_blocks = blocks,
            .single_transformer_blocks = single_blocks,

            .norm_out = AdaLayerNormContinuous.init(store.withPrefix("norm_out"), config),
            .proj_out = Linear.init(store.withPrefix("proj_out")),
        };
        return model;
    }

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        self.time_guidance_embed.deinit();
        self.double_stream_modulation_img.deinit();
        self.double_stream_modulation_txt.deinit();
        self.single_stream_modulation.deinit();
        self.x_embedder.deinit();
        self.context_embedder.deinit();
        for (self.transformer_blocks) |*b| b.deinit();
        allocator.free(self.transformer_blocks);
        for (self.single_transformer_blocks) |*b| b.deinit();
        allocator.free(self.single_transformer_blocks);
        self.norm_out.deinit();
        self.proj_out.deinit();
    }

    pub fn prepare_text_ids(self: Flux2Transformer2DModel, prompt_embeds: Tensor) Tensor {
        _ = self;
        const shape = prompt_embeds.shape();
        const B = shape.dim(0);
        const L = shape.dim(1);
        const L_int = @as(i64, @intCast(L));
        const one = @as(i64, 1);

        const shape_l = zml.Shape.init(.{ .l = L }, .i32);
        const l_idx = Tensor.iota(shape_l, .l).reshape(.{ L_int, one });
        const z_idx = Tensor.zeroes(shape_l).reshape(.{ L_int, one });

        const grid = Tensor.concatenate(&.{ z_idx, z_idx, z_idx, l_idx }, 1);
        const grid_expanded = grid.reshape(.{ 1, L_int, 4 }).broad(zml.Shape.init(.{ .b = B, .seq = L, .coord = 4 }, .i32));
        return grid_expanded;
    }

    pub fn pack_latents(self: Flux2Transformer2DModel, latents: Tensor) Tensor {
        _ = self;
        const s = latents.shape();
        const B = s.dim(0);
        const C = s.dim(1);
        const H = s.dim(2);
        const W = s.dim(3);
        const flat = latents.reshape(.{ B, C, H * W });
        const out = flat.transpose(.{ 0, 2, 1 });
        return out;
    }

    pub fn prepare_latent_ids(self: Flux2Transformer2DModel, latents: Tensor) Tensor {
        _ = self;
        const shape = latents.shape();
        const B = shape.dim(0);
        const H = shape.dim(2);
        const W = shape.dim(3);
        const shape_hw = zml.Shape.init(.{ .h = H, .w = W }, .i32);
        const h_idx = Tensor.iota(shape_hw, .h);
        const w_idx = Tensor.iota(shape_hw, .w);
        const z_idx = Tensor.zeroes(shape_hw);
        const HW = H * W;
        const h_flat = h_idx.reshape(.{@as(i64, @intCast(HW))});
        const w_flat = w_idx.reshape(.{@as(i64, @intCast(HW))});
        const z_flat = z_idx.reshape(.{@as(i64, @intCast(HW))});

        const t_col = z_flat.reshape(.{ HW, 1 });
        const h_col = h_flat.reshape(.{ HW, 1 });
        const w_col = w_flat.reshape(.{ HW, 1 });
        const l_col = z_flat.reshape(.{ HW, 1 });
        const grid = Tensor.concatenate(&.{ t_col, h_col, w_col, l_col }, 1);
        const grid_expanded = grid.reshape(.{ 1, HW, 4 }).broad(zml.Shape.init(.{ .b = B, .seq = HW, .coord = 4 }, .i32));
        return grid_expanded;
    }

    pub fn forward(
        self: Flux2Transformer2DModel,
        hidden_states: Tensor,
        encoder_hidden_states: Tensor,
        timesteps_proj: Tensor,
        guidance_proj: ?Tensor,
        rotary_cos: Tensor,
        rotary_sin: Tensor,
    ) Tensor {
        // 1. Embeddings
        var img = self.x_embedder.forward(hidden_states);
        var txt = self.context_embedder.forward(encoder_hidden_states);

        const temb = self.time_guidance_embed.forward(timesteps_proj, guidance_proj);

        // 2. Modulation (keep backing storage alive for slices)
        var double_stream_mod_img_buf = self.double_stream_modulation_img.forward(temb);
        var double_stream_mod_txt_buf = self.double_stream_modulation_txt.forward(temb);
        var single_stream_mod_buf = self.single_stream_modulation.forward(temb);

        const double_stream_mod_img = double_stream_mod_img_buf.slice();
        const double_stream_mod_txt = double_stream_mod_txt_buf.slice();
        const single_stream_mod = single_stream_mod_buf.slice();

        // 3. Positional Embeddings
        const concat_rotary_emb = .{ rotary_cos, rotary_sin };

        // 4. Double Stream Blocks
        for (self.transformer_blocks) |block| {
            const out = block.forward(img, txt, double_stream_mod_img, double_stream_mod_txt, concat_rotary_emb);
            txt = out[0];
            img = out[1];
        }

        // 5. Concatenate (txt then img)
        const joint = Tensor.concatenate(&.{ txt, img }, 1);

        // 6. Single Stream Blocks
        var single_stream = joint;
        for (self.single_transformer_blocks) |block| {
            single_stream = block.forward(single_stream, single_stream_mod, concat_rotary_emb);
        }

        // 7. Output Layer
        const num_txt_tokens = txt.shape().dim(1);
        const img_out = single_stream.slice(&.{ .{}, .{ .start = num_txt_tokens }, .{} });

        const normed = self.norm_out.forward(img_out, temb);
        const projected = self.proj_out.forward(normed);

        return projected;
    }
};

pub const Flux2Transformer2D = struct {
    model: Flux2Transformer2DModel,
    store: zml.io.TensorStore,
    registry: zml.safetensors.TensorRegistry,
    config: Config,
    weights: zml.Bufferized(Flux2Transformer2DModel),
    step_exe: zml.Exe,
    euler_exe: zml.Exe,

    pub fn deinit(self: *@This(), allocator: std.mem.Allocator) void {
        utils.unloadWeights(allocator, &self.weights);
        self.step_exe.deinit();
        self.euler_exe.deinit();
        self.model.deinit(allocator);
        self.store.deinit();
        self.registry.deinit();
    }

    pub fn loadFromFile(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, repo_dir: std.Io.Dir, parallelism_level: usize, image_height: usize, image_width: usize, seqlen: usize, progress: ?*std.Progress.Node, options: struct { subfolder: []const u8 = "transformer", json_name: []const u8 = "config.json", safetensors_name: []const u8 = "diffusion_pytorch_model.safetensors" }) !@This() {
        @setEvalBranchQuota(10_000);
        const timer_start = std.Io.Clock.awake.now(io);
        defer log.info("Loaded Flux2Transformer2D Model in {} ms", .{std.time.milliTimestamp(std.Io.Clock.awake.now(io) - timer_start)});

        var config_json = try tools.parseConfig(Config, allocator, io, repo_dir, .{ .subfolder = options.subfolder, .json_name = options.json_name });
        defer config_json.deinit();
        const config = config_json.value;

        const transformer_dir = try repo_dir.openDir(io, options.subfolder, .{});
        defer transformer_dir.close(io);

        var tensor_registry = try zml.safetensors.TensorRegistry.fromFile(allocator, io, transformer_dir, options.safetensors_name);
        errdefer tensor_registry.deinit();

        var tensor_store = zml.io.TensorStore.fromRegistry(allocator, &tensor_registry);
        errdefer tensor_store.deinit();

        var model = try Flux2Transformer2DModel.init(allocator, tensor_store.view(), config);
        errdefer model.deinit(allocator);

        var weights = try zml.io.load(
            Flux2Transformer2DModel,
            &model,
            allocator,
            io,
            platform,
            .{ .parallelism = parallelism_level, .store = &tensor_store, .dma_chunks = 4, .dma_chunk_size = 64 * 1024 * 1024, .progress = progress },
        );
        errdefer utils.unloadWeights(allocator, &weights);

        const adjusted_height = image_height / 16;
        const adjusted_width = image_width / 16;
        const S = adjusted_height * adjusted_width;
        const dim_latents = @as(i64, @intCast(config.in_channels));
        const pixel_latents_shape = zml.Shape.init(.{ .b = 1, .s = @as(i64, @intCast(S)), .d = dim_latents }, .bf16);

        const t_proj_shape = zml.Shape.init(.{ .b = 1, .d = 256 }, .bf16);

        const prompt_embeds_shape = zml.Shape.init(.{ .b = 1, .s = @as(i64, @intCast(seqlen)), .d = @as(i64, @intCast(config.joint_attention_dim)) }, .bf16);

        var total_rope_dim: i64 = 0;
        for (config.axes_dims_rope) |d| total_rope_dim += d;
        const rotary_shape = zml.Shape.init(.{ .b = 1, .s = @as(i64, @intCast(S + seqlen)), .d = total_rope_dim }, .bf16);
        const sym_latents = zml.Tensor.fromShape(pixel_latents_shape);
        const sym_t_proj = zml.Tensor.fromShape(t_proj_shape);
        const sym_g_proj = zml.Tensor.fromShape(t_proj_shape);
        const sym_prompt = zml.Tensor.fromShape(prompt_embeds_shape);
        const sym_rotary_cos = zml.Tensor.fromShape(rotary_shape);
        const sym_rotary_sin = zml.Tensor.fromShape(rotary_shape);

        const FluxStep = struct {
            pub fn forward(self: @This(), m: Flux2Transformer2DModel, hidden_states: zml.Tensor, encoder_hidden_states: zml.Tensor, timesteps_proj: zml.Tensor, guidance_proj: ?zml.Tensor, rotary_cos: zml.Tensor, rotary_sin: zml.Tensor) zml.Tensor {
                _ = self;
                return m.forward(hidden_states, encoder_hidden_states, timesteps_proj, guidance_proj, rotary_cos, rotary_sin);
            }
        };

        log.info("Compiling Flux Step...", .{});
        var step_exe = try platform.compile(allocator, io, FluxStep{}, .forward, .{ model, sym_latents, sym_prompt, sym_t_proj, sym_g_proj, sym_rotary_cos, sym_rotary_sin });
        errdefer step_exe.deinit();

        const EulerStep = struct {
            pub fn forward(self: @This(), sample: zml.Tensor, model_output: zml.Tensor, dt: zml.Tensor) zml.Tensor {
                _ = self;
                const s_f32 = sample.convert(.f32);
                const m_f32 = model_output.convert(.f32);
                const res = s_f32.add(m_f32.mul(dt));
                return res.convert(.bf16);
            }
        };
        const sym_dt = zml.Tensor.fromShape(zml.Shape.init(.{}, .f32));
        const sym_sample = zml.Tensor.fromShape(pixel_latents_shape);
        const sym_model_out = zml.Tensor.fromShape(pixel_latents_shape);

        log.info("Compiling Euler Step...", .{});
        var euler_exe = try platform.compile(allocator, io, EulerStep{}, .forward, .{ sym_sample, sym_model_out, sym_dt });
        errdefer euler_exe.deinit();

        return .{
            .model = model,
            .store = tensor_store,
            .registry = tensor_registry,
            .config = config,
            .weights = weights,
            .step_exe = step_exe,
            .euler_exe = euler_exe,
        };
    }
};
