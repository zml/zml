const std = @import("std");
const zml = @import("zml");
const stdx = zml.stdx;

const Tensor = zml.Tensor;

pub const RmsNorm = struct {
    /// `{.d}` learned scale.
    scale: Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{
            .scale = store.createTensor("scale", .{.d}, .replicated, .{ .d = .replicated }),
            .eps = eps,
        };
    }

    /// Normalizes over `.d`, in f32, then casts back to the input dtype before scaling.
    pub fn forward(self: RmsNorm, x: Tensor) Tensor {
        const out = zml.nn.rmsNorm(x, .d, self.eps);
        return out.mul(self.scale.convert(out.dtype()).broad(out.shape()));
    }
};

/// Rotary position embedding (sin/cos concatenation), with partial rotation.
///
/// Only the first `rotary_dim` dimensions of each head rotate. The inverse-frequency
/// table is zero-padded past `rotary_dim / 2`, so the remaining dimensions get `cos=1`,
/// `sin=0`, which is the identity rotation.
pub const SinCosRope = struct {
    /// Head dimension.
    dim: u32,
    base: f32,
    rotary_dim: u32,
    /// Output dtype of the `cos`/`sin` tables.
    dtype: zml.DataType,

    pub const CosSin = struct {
        cos: Tensor,
        sin: Tensor,
    };

    /// `{axis = dim / 2}` f32 inverse frequencies, zero past `rotary_dim / 2`.
    fn invFreq(self: SinCosRope, axis: anytype) Tensor {
        const idx: Tensor = .arange(.{ .end = self.rotary_dim, .step = 2 }, .f32);
        const exponent = idx.divByConst(@as(f32, @floatFromInt(self.rotary_dim)));
        const base: Tensor = .scalar(self.base, .f32);
        const rotating = Tensor.scalar(1.0, .f32).broad(exponent.shape())
            .div(base.broad(exponent.shape()).pow(exponent));
        const n_pad = self.dim / 2 - self.rotary_dim / 2;
        const out = if (n_pad == 0)
            rotating
        else
            Tensor.concatenate(&.{ rotating, .zeroes(.init(.{n_pad}, .f32)) }, 0);
        return out.withTags(.{axis});
    }

    /// `(cos, sin)` tables in `dtype`, each `positions.shape ++ {axis = dim}`, for integer `positions` of any shape.
    pub fn forward(self: SinCosRope, positions: Tensor, axis: anytype) CosSin {
        const freqs = positions.convert(.f32).outer(self.invFreq(axis));
        const emb: Tensor = .concatenate(&.{ freqs, freqs }, axis);
        return .{
            .cos = emb.cos().convert(self.dtype),
            .sin = emb.sin().convert(self.dtype),
        };
    }
};

pub const ResidualBlock = struct {
    hidden_layer: zml.nn.Linear,
    output_layer: zml.nn.Linear,
    residual_layer: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) ResidualBlock {
        return .{
            .hidden_layer = loadLinear(store.withPrefix("hidden_layer"), .{ .h, .in }),
            .output_layer = loadLinear(store.withPrefix("output_layer"), .{ .out, .h }),
            .residual_layer = loadLinear(store.withPrefix("residual_layer"), .{ .out, .in }),
        };
    }

    /// `x` has an `.in` axis; the output has it replaced by `.out`.
    pub fn forward(self: ResidualBlock, x: Tensor) Tensor {
        const h = self.hidden_layer.forward(x, x.dtype()).silu();
        const out = self.output_layer.forward(h, h.dtype());
        return out.add(self.residual_layer.forward(x, x.dtype()));
    }
};

/// SwiGLU MLP with one fused input projection; computes `gate * silu(up)`.
pub const FusedSwiGlu = struct {
    /// `{.out = 2 * hidden_dim, .in = d}`, fused gate and up projections.
    w12: zml.nn.Linear,
    /// `{.out = d, .in = hidden_dim}` output projection.
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) FusedSwiGlu {
        return .{
            .w12 = loadLinear(store.withPrefix("w12"), .{ .out, .in }),
            .w3 = loadLinear(store.withPrefix("w3"), .{ .out, .in }),
        };
    }

    /// `x` has a `.d` axis; the output has the same shape.
    pub fn forward(self: FusedSwiGlu, x: Tensor) Tensor {
        const gate, const up = self.w12.forward(x.renameTag(.d, .in), x.dtype()).chunkExact(.out, 2);
        const h = gate.mul(up.silu()).renameTag(.out, .in);
        return self.w3.forward(h, x.dtype()).renameTag(.out, .d);
    }
};

/// Multi-head attention over the time axis (`is_time=True`) or the variate axis.
///
/// Time attention applies RoPE; variate attention has no positional encoding. With the
/// `sdpa_sigmoid` gate, the attention output is multiplied by `sigmoid(x @ W_g)` before
/// the output projection.
pub const Attention = struct {
    pub const AttentionGate = enum { none, sdpa_sigmoid };

    const Options = struct {
        embed_dim: u32,
        num_heads: u32,
        is_time: bool,
        gate: AttentionGate,

        pub fn head_dim(self: Options) u32 {
            return self.embed_dim / self.num_heads;
        }
    };

    options: Options,
    wQKV: zml.nn.Linear,
    wO: zml.nn.Linear,
    _gate: ?zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View, options: Options) Attention {
        const gate = if (store.hasKey("_gate"))
            loadLinear(store.withPrefix("_gate"), .{ .hidden, .in })
        else
            null;
        return .{
            .options = options,
            .wQKV = loadLinear(store.withPrefix("wQKV"), .{ .hidden, .in }),
            .wO = loadLinear(store.withPrefix("wO"), .{ .out, .hidden }),
            ._gate = gate,
        };
    }

    pub fn forward(self: Attention, x: Tensor, mask: Tensor, sin_cos: SinCosRope.CosSin) Tensor {
        var qkv = self.forwardQKV(x);
        if (self.options.is_time) {
            qkv = applyRotary(qkv, sin_cos, .in);
        }
        // TODO: apply config.split_context_horizon_attn to reduce size of attention matrix

        var h = sdpa(qkv, mask);
        if (self._gate) |gate| {
            const g = gate.forward(x, x.dtype());
            h = .mul(h, g.sigmoid());
        }
        return self.wO.forward(h, h.dtype());
    }

    const QKV = struct {
        q: Tensor,
        k: Tensor,
        v: Tensor,
    };

    /// Project and fused the input into Q, K, V heads.
    fn forwardQKV(self: Attention, x: Tensor) QKV {
        const qkv = self.wQKV.forward(x, x.dtype()).splitAxis(.h, .{
            .qkv = 3,
            .head_dim = self.options.head_dim(),
            .head = self.options.num_heads,
        }).chunkExact(.qkv, 3);
        return .{
            .q = qkv[0].squeeze(.qkv),
            .k = qkv[1].squeeze(.qkv),
            .v = qkv[2].squeeze(.qkv),
        };
    }

    fn rotateHalf(x: Tensor, axis: anytype) Tensor {
        const midpoint = x.dim(axis) / 2;
        const x1 = x.slice(axis, .{ .end = midpoint });
        const x2 = x.slice(axis, .{ .start = midpoint });
        return .concatenate(&.{ x2.negate(), x1 }, axis);
    }

    /// Rotate `q`, `k` along the head-dim `axis`; `cos`/`sin` broadcast to `q`'s shape (e.g. over `.head`).
    fn applyRotary(qkv: QKV, sin_cos: SinCosRope.CosSin, axis: anytype) QKV {
        const shape = qkv.q.shape();
        const cos = sin_cos.cos.broad(shape);
        const sin = sin_cos.sin.broad(shape);
        const q: Tensor = .add(.mul(qkv.q, cos), .mul(rotateHalf(qkv.q, axis), sin));
        const k: Tensor = .add(.mul(qkv.k, cos), .mul(rotateHalf(qkv.k, axis), sin));
        return .{ .q = q, .k = k, .v = qkv.v };
    }

    /// Scaled dot-product attention; `q`, `k`, `v` carry `.head` and `.head_dim` axes.
    ///
    /// `mask` is boolean, true = attend. Returns `.head` and `.head_dim` merged (head-major) into `.d`.
    fn sdpa(qkv: QKV, mask: Tensor) Tensor {
        const dtype = qkv.q.dtype();
        const attn_mask = mask.select(Tensor.scalar(0, dtype), Tensor.scalar(-std.math.inf(f32), dtype));
        const out = zml.nn.sdpa(
            qkv.q.renameTag(.head, .h).renameTag(.head_dim, .hd),
            qkv.k.renameTag(.head, .h).renameTag(.head_dim, .hd),
            qkv.v.renameTag(.head, .h).renameTag(.head_dim, .hd),
            .{ .attn_mask = attn_mask },
        );
        return out.mergeTranspose(.{ .h, .hd }, .out);
    }
};

/// Load a replicated linear layer with `weight` tagged `{out, in}` and `bias` tagged `{out}`.
fn loadLinear(store: zml.io.TensorStore.View, tags: anytype) zml.nn.Linear {
    const out, const in = tags;
    return .init(
        store.createTensor("weight", .{ out, in }, .replicated, .{}),
        store.maybeCreateTensor("bias", .{out}, .replicated, .{}),
        in,
    );
}
