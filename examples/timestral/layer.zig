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
        const hl = store.withPrefix("hidden_layer");
        const hidden_layer: zml.nn.Linear = .init(
            hl.createTensor(
                "weight",
                .{ .h, .d },
                .replicated,
                .{ .h = .replicated, .d = .replicated },
            ),
            hl.maybeCreateTensor("bias", .{.h}, .replicated, .{ .h = .replicated }),
            .d,
        );
        const ol = store.withPrefix("output_layer");
        const output_layer: zml.nn.Linear = .init(
            ol.createTensor(
                "weight",
                .{ .out, .h },
                .replicated,
                .{ .out = .replicated, .h = .replicated },
            ),
            ol.maybeCreateTensor("bias", .{.out}, .replicated, .{ .out = .replicated }),
            .h,
        );
        const rl = store.withPrefix("residual_layer");
        const residual_layer: zml.nn.Linear = .init(
            rl.createTensor(
                "weight",
                .{ .out, .d },
                .replicated,
                .{ .out = .replicated, .d = .replicated },
            ),
            rl.maybeCreateTensor("bias", .{.out}, .replicated, .{ .out = .replicated }),
            .d,
        );

        return .{
            .hidden_layer = hidden_layer,
            .output_layer = output_layer,
            .residual_layer = residual_layer,
        };
    }

    pub fn forward(self: ResidualBlock, x: Tensor) Tensor {
        const h = self.hidden_layer.forward(x, x.dtype()).silu();
        const out = self.output_layer.forward(h, h.dtype());
        return out.add(self.residual_layer.forward(x, x.dtype()));
    }
};

/// SwiGLU MLP with one fused input projection; computes `gate * silu(up)`.
pub const FusedSwiGlu = struct {
    /// `{.hidden = 2 * hidden_dim, .d}`, fused gate and up projections.
    w12: zml.nn.Linear,
    /// `{.d, .hidden}` output projection.
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) FusedSwiGlu {
        const s12 = store.withPrefix("w12");
        const w12: zml.nn.Linear = .init(
            s12.createTensor(
                "weight",
                .{ .hidden, .d },
                .replicated,
                .{ .hidden = .replicated, .d = .replicated },
            ),
            s12.maybeCreateTensor("bias", .{.hidden}, .replicated, .{ .hidden = .replicated }),
            .d,
        );
        const s3 = store.withPrefix("w3");
        const w3: zml.nn.Linear = .init(
            s3.createTensor(
                "weight",
                .{ .d, .hidden },
                .replicated,
                .{ .d = .replicated, .hidden = .replicated },
            ),
            s3.maybeCreateTensor("bias", .{.d}, .replicated, .{ .d = .replicated }),
            .hidden,
        );
        return .{ .w12 = w12, .w3 = w3 };
    }

    /// `x` has a `.d` axis; the output has the same shape.
    pub fn forward(self: FusedSwiGlu, x: Tensor) Tensor {
        const gate, const up = self.w12.forward(x, x.dtype()).chunkExact(.hidden, 2);
        return self.w3.forward(gate.mul(up.silu()), x.dtype());
    }
};
