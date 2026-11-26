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

/// Load a replicated linear layer with `weight` tagged `{out, in}` and `bias` tagged `{out}`.
fn loadLinear(store: zml.io.TensorStore.View, tags: anytype) zml.nn.Linear {
    const out, const in = tags;
    return .init(
        store.createTensor("weight", .{ out, in }, .replicated, .{}),
        store.maybeCreateTensor("bias", .{out}, .replicated, .{}),
        in,
    );
}
