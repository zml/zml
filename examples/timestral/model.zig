const std = @import("std");

const zml = @import("zml");

const Tensor = zml.Tensor;

pub const Config = struct {
    pub const LayerKind = enum { time, variate, time_variate, variate_time };
    pub const AttentionGate = enum { none, sdpa_sigmoid };

    patch_size: u32,
    d_model: u32,
    num_heads: u32,
    mlp_hidden_dim: u32,
    block_schedule: []const LayerKind,
    quantile_knots: []const f32,
    block_size: u32,
    context_length: u32,
    rope_theta: f32,
    rope_partial_factor: f32,
    norm_eps: f32,
    attn_bias: bool,
    mlp_bias: bool,
    time_attn_gate: AttentionGate,
    variate_attn_gate: AttentionGate,
    minimum_scale: f32,
    use_asinh: bool,
    stabilize_with_global: bool,
    covariate_enabled: bool,
    split_context_horizon_attn: bool,
    attend_missing_patches: bool,
    quantile_width_factor: f32,
};

const ResidualBlock = struct {
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

pub const Model = ResidualBlock;
