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
    norm: Tensor,
    norm_eps: f32,
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
            .norm = transformer_store.withPrefix("norm").createTensorWithTags("weight", .{.d}),
            .norm_eps = enc.norm_eps,
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
        self.norm.deinit();
    }

    /// mel: [channels=128, time=frames]
    /// Returns: [s, d=1280]
    pub fn forward(self: Encoder, mel: Tensor) Tensor {
        // Standardize precision to match model weights (usually bf16)
        const dtype = self.conv0.weight.dtype();
        var h = mel.convert(dtype).insertAxes(.channels, .{.batch});

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

        return rmsNorm(h, self.norm, self.norm_eps);
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
        std.debug.assert(dtype == self.weight.dtype());

        var h = input.conv1d(self.weight, .{
            .window_strides = stride,
            .padding = &.{ padding_left, padding_right },
        });

        // Bias: [channels] broadcasts to [batch, channels, time]
        h = h.add(self.bias.broad(h.shape()));
        return h;
    }
};

pub const TransformerLayer = struct {
    attention_norm: Tensor,
    attention: SelfAttention,
    ffn_norm: Tensor,
    feed_forward: SwiGluFfn,
    norm_eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) TransformerLayer {
        const enc = config.encoder();
        return .{
            .attention_norm = store.withPrefix("attention_norm").createTensorWithTags("weight", .{.d}),
            .attention = SelfAttention.init(store.withPrefix("attention")),
            .ffn_norm = store.withPrefix("ffn_norm").createTensorWithTags("weight", .{.d}),
            .feed_forward = SwiGluFfn.init(store.withPrefix("feed_forward")),
            .norm_eps = enc.norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        self.attention_norm.deinit();
        SelfAttention.unloadBuffers(&self.attention);
        self.ffn_norm.deinit();
        SwiGluFfn.unloadBuffers(&self.feed_forward);
    }

    /// h: [s, d] -> [s, d]
    pub fn forward(self: TransformerLayer, h: Tensor, config: Config) Tensor {
        // Pre-attention norm + attention + residual
        const attn_out = self.attention.forward(rmsNorm(h, self.attention_norm, self.norm_eps), config);
        var out = h.add(attn_out);

        // Pre-FFN norm + FFN + residual
        const ffn_out = self.feed_forward.forward(rmsNorm(out, self.ffn_norm, self.norm_eps));
        out = out.add(ffn_out);

        return out;
    }
};

pub const SelfAttention = struct {
    wq: zml.nn.Linear,
    wk: zml.nn.Linear,
    wv: zml.nn.Linear,
    wo: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SelfAttention {
        return .{
            .wq = initLinear(store.withPrefix("wq")),
            .wk = initLinear(store.withPrefix("wk")),
            .wv = initLinear(store.withPrefix("wv")),
            .wo = initLinear(store.withPrefix("wo")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttention)) void {
        deinitLinear(&self.wq);
        deinitLinear(&self.wk);
        deinitLinear(&self.wv);
        deinitLinear(&self.wo);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SelfAttention, x: Tensor, config: Config) Tensor {
        const enc = config.encoder();
        const dtype = x.dtype();

        // Q, K, V projections (encoder: Q and V have biases, K does not)
        var q = self.wq.forward(x);
        var k = self.wk.forward(x);
        var v = self.wv.forward(x);

        // Split into heads: [s, dout] -> [s, h, hd]
        q = q.splitAxis(.dout, .{ .h = enc.n_heads, .hd = enc.head_dim });
        k = k.splitAxis(.dout, .{ .h = enc.n_kv_heads, .hd = enc.head_dim });
        v = v.splitAxis(.dout, .{ .h = enc.n_kv_heads, .hd = enc.head_dim });

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

        // Merge heads and output projection: [q, h, hd] -> [s, d]
        const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return self.wo.forward(merged).rename(.{ .dout = .d });
    }
};

pub const SwiGluFfn = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGluFfn {
        return .{
            .w1 = initLinear(store.withPrefix("w1")),
            .w2 = initLinear(store.withPrefix("w2")),
            .w3 = initLinear(store.withPrefix("w3")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SwiGluFfn)) void {
        deinitLinear(&self.w1);
        deinitLinear(&self.w2);
        deinitLinear(&self.w3);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SwiGluFfn, x: Tensor) Tensor {
        const gate = self.w1.forward(x).silu();
        const up = self.w3.forward(x);
        return self.w2.forward(gate.mul(up).rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

// -- Helpers

fn initLinear(store: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        store.createTensorWithTags("weight", .{ .dout, .d }),
        store.maybeCreateTensorWithTags("bias", .{.dout}),
        .d,
    );
}

fn deinitLinear(l: *zml.Bufferized(zml.nn.Linear)) void {
    l.weight.deinit();
    if (l.bias) |*b| b.deinit();
}

pub fn rmsNorm(x: Tensor, weight: Tensor, eps: f32) Tensor {
    return zml.nn.rmsNorm(x, .d, eps).mul(weight.broad(x.shape()));
}
