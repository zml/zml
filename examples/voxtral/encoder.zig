const std = @import("std");
const builtin = @import("builtin");
const log = std.log;

const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const cfg = @import("config.zig");
const Config = cfg.Config;

pub const Encoder = struct {
    conv0: CausalConv1d,
    conv1: CausalConv1d,
    layers: []TransformerLayer,
    norm: RmsNorm,
    config: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) Encoder {
        const conv_store = store.withPrefix("conv_layers");
        const transformer_store = store.withPrefix("transformer");

        const enc = config.encoder();
        const layers = allocator.alloc(TransformerLayer, enc.n_layers) catch unreachable;
        for (layers, 0..) |*layer, i| {
            layer.* = TransformerLayer.init(transformer_store.withPrefix("layers").withLayer(i), config);
        }

        return .{
            .conv0 = CausalConv1d.init(conv_store.withLayer(0).withPrefix("conv"), 1),
            .conv1 = CausalConv1d.init(conv_store.withLayer(1).withPrefix("conv"), 2),
            .layers = layers,
            .norm = RmsNorm.init(transformer_store.withPrefix("norm"), enc.norm_eps),
            .config = config,
        };
    }

    pub fn deinit(self: Encoder, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Encoder), allocator: std.mem.Allocator) void {
        CausalConv1d.unloadBuffers(&self.conv0);
        CausalConv1d.unloadBuffers(&self.conv1);
	
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
	
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    /// mel: [channels=128, time=frames]
    /// Returns: [s, d=1280]
    pub fn forward(self: Encoder, mel: Tensor) Tensor {
        // Conv stem: [channels, time] -> [batch=1, channels, time]
        var h = mel.insertAxes(.channels, .{.batch});

        h = self.conv0.forward(h).gelu();
        h = self.conv1.forward(h).gelu();

        // [batch=1, channels=1280, time'] -> [s, d=1280]
        h = h.squeeze(.batch);
        h = h.transpose(.{ .time, .channels }).rename(.{ .time = .s, .channels = .d });

        // Left-truncate to multiple of downsample_factor
        const seq_len = h.dim(.s);
        const trunc: i64 = @intCast(@as(usize, @intCast(seq_len)) % self.config.downsample_factor());
	
        if (trunc > 0) {
            h = h.slice1d(.s, .{ .start = trunc });
        }

        // Transformer layers
        for (self.layers) |layer| {
            h = layer.forward(h, self.config);
        }

        // Final norm
        return self.norm.forward(h);
    }
};

/// Expects input layout [batch, channels, time] and kernel [cout, cin, k].
pub const CausalConv1d = struct {
    weight: Tensor,
    bias: Tensor,
    stride: i64,

    pub fn init(store: zml.io.TensorStore.View, stride: i64) CausalConv1d {
        return .{
            .weight = store.createTensorWithTags("weight", .{ .cout, .cin, .k }),
            .bias = store.createTensorWithTags("bias", .{.channels}),
            .stride = stride,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(CausalConv1d)) void {
        self.weight.deinit();
        self.bias.deinit();
    }

    pub fn forward(self: CausalConv1d, input: Tensor) Tensor {
        const kernel_size: i64 = @intCast(self.weight.dim(.k));
        const stride = self.stride;
        const input_len: i64 = @intCast(input.dim(.time));

        // Causal padding: padding_total goes on the left.
        // Right padding is computed to align the output length with stride.
        const padding_left = kernel_size - stride;
        const numerator = input_len - kernel_size + padding_left;
        const n_frames = @divTrunc(numerator + stride - 1, stride) + 1;
        const target_length = (n_frames - 1) * stride + kernel_size - padding_left;
        const padding_right = target_length - input_len;

        const dtype = input.dtype();
        var h = input.conv1d(self.weight.convert(dtype), .{
            .window_strides = stride,
            .padding = &.{ padding_left, padding_right },
        });

        // Bias: [channels] broadcasts to [batch, channels, time]
        h = h.add(self.bias.convert(dtype).broad(h.shape()));
        return h;
    }
};

pub const TransformerLayer = struct {
    attention_norm: RmsNorm,
    attention: SelfAttention,
    ffn_norm: RmsNorm,
    feed_forward: SwiGluFfn,

    pub fn init(store: zml.io.TensorStore.View, config: Config) TransformerLayer {
        const enc = config.encoder();
        return .{
            .attention_norm = RmsNorm.init(store.withPrefix("attention_norm"), enc.norm_eps),
            .attention = SelfAttention.init(store.withPrefix("attention")),
            .ffn_norm = RmsNorm.init(store.withPrefix("ffn_norm"), enc.norm_eps),
            .feed_forward = SwiGluFfn.init(store.withPrefix("feed_forward")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.attention_norm);
        SelfAttention.unloadBuffers(&self.attention);
        RmsNorm.unloadBuffers(&self.ffn_norm);
        SwiGluFfn.unloadBuffers(&self.feed_forward);
    }

    /// h: [s, d] -> [s, d]
    pub fn forward(self: TransformerLayer, h: Tensor, config: Config) Tensor {
        // Pre-attention norm + attention + residual
        const attn_out = self.attention.forward(self.attention_norm.forward(h), config);
        var out = h.add(attn_out);

        // Pre-FFN norm + FFN + residual
        const ffn_out = self.feed_forward.forward(self.ffn_norm.forward(out));
        out = out.add(ffn_out);

        return out;
    }
};

pub const SelfAttention = struct {
    wq: Linear,
    wk: Linear,
    wv: Linear,
    wo: Linear,

    pub fn init(store: zml.io.TensorStore.View) SelfAttention {
        return .{
            .wq = Linear.init(store.withPrefix("wq")),
            .wk = Linear.init(store.withPrefix("wk")),
            .wv = Linear.init(store.withPrefix("wv")),
            .wo = Linear.init(store.withPrefix("wo")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttention)) void {
        Linear.unloadBuffers(&self.wq);
        Linear.unloadBuffers(&self.wk);
        Linear.unloadBuffers(&self.wv);
        Linear.unloadBuffers(&self.wo);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SelfAttention, x: Tensor, config: Config) Tensor {
        const enc = config.encoder();
        const dtype = x.dtype();

        // Q, K, V projections
        // Encoder: Q and V have biases, K does not
        var q = self.wq.forward(x); // [s, n_heads * head_dim]
        var k = self.wk.forward(x); // [s, n_kv_heads * head_dim]
        var v = self.wv.forward(x); // [s, n_kv_heads * head_dim]

        // Split into heads: [s, d] -> [s, h, hd]
        q = q.splitAxis(.d, .{ .h = enc.n_heads, .hd = enc.head_dim });
        k = k.splitAxis(.d, .{ .h = enc.n_kv_heads, .hd = enc.head_dim });
        v = v.splitAxis(.d, .{ .h = enc.n_kv_heads, .hd = enc.head_dim });

        // RoPE (interleaved, matching is_neox_style=False in Python)
        const rope_opts: zml.nn.RopeOpts = .{
            .layout = .interleaved,
            .freq_base = enc.rope_theta,
        };
	
        q = zml.nn.rope(q, null, rope_opts);
        k = zml.nn.rope(k, null, rope_opts);

        // Rename seq dim for sdpa: .s -> .q/.k
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });

        // Causal attention with sliding window
        const seq_len = q.dim(.q);
        const attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, dtype, enc.sliding_window);
        const attn_out = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask });

        // Merge heads: [s, h, hd] -> [s, d]
        const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

        // Output projection
        return self.wo.forward(merged);
    }
};

pub const SwiGluFfn = struct {
    w1: Linear,
    w2: Linear,
    w3: Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGluFfn {
        return .{
            .w1 = Linear.init(store.withPrefix("w1")),
            .w2 = Linear.init(store.withPrefix("w2")),
            .w3 = Linear.init(store.withPrefix("w3")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SwiGluFfn)) void {
        Linear.unloadBuffers(&self.w1);
        Linear.unloadBuffers(&self.w2);
        Linear.unloadBuffers(&self.w3);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SwiGluFfn, x: Tensor) Tensor {
        // SwiGLU: silu(gate) * up, then down
        const gate = self.w1.forward(x).silu();
        const up = self.w3.forward(x);
        return self.w2.forward(gate.mul(up));
    }
};

pub const Linear = struct {
    weight: Tensor,
    bias: ?Tensor,

    pub fn init(store: zml.io.TensorStore.View) Linear {
        return .{
            .weight = store.createTensorWithTags("weight", .{ .out, .d }),
            .bias = store.maybeCreateTensorWithTags("bias", .{.d}),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Linear)) void {
        self.weight.deinit();
        if (self.bias) |*b| b.deinit();
    }

    /// x: [..., d] -> [..., out] (renamed back to d)
    pub fn forward(self: Linear, x: Tensor) Tensor {
        const dtype = x.dtype();
        var y = x.dot(self.weight.convert(dtype), .d).rename(.{ .out = .d });
	
        if (self.bias) |bias| {
            y = y.add(bias.convert(dtype).broad(y.shape()));
        }
	
        return y;
    }
};

pub const RmsNorm = struct {
    weight: Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .weight = store.createTensorWithTags("weight", .{.d}),
            .eps = eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: Tensor) Tensor {
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.convert(x.dtype()).broad(x.shape()));
    }
};
