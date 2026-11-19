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

pub const Roles = enum(u8) {
    target = 0,
    past_covariate = 1,
    know_covariate = 2,
};

// TODO we will need to merge all inputs into one (with roles) to handle fixed shape.
// TODO missing left pad_to_patch when not a multiple of 32
pub const PreProcessor = struct {
    /// Series whose dimensions are predicted.
    pub const TargetsSeries = struct {
        values: Tensor,
        mask: Tensor,

        /// Compute mask from infinite and NaN data and replace missing values.
        ///
        /// Input tensor with dimensions {.b, .v, .t}
        pub fn make(targets: Tensor) TargetsSeries {
            const mask = targets.isFinite(); // exclude NaN and inf
            return .{
                .values = mask.select(targets, .scalar(0, targets.dtype())),
                .mask = mask,
            };
        }
    };

    const Rows = struct {
        values: Tensor,
        mask: Tensor,
        roles: Tensor,

        pub fn make(targets: TargetsSeries) Rows {
            const n_batch = targets.values.dim(.b);
            const n_targets = targets.values.dim(.v);
            const target_roles = Tensor.scalar(@backingInt(Roles.target), .u8)
                .broad(.init(.{ .v = n_targets }, .u8));
            return .{
                .values = ffill_mask(targets.values, targets.mask, .t).renameTag(.v, .d),
                .mask = targets.mask.renameTag(.v, .d),
                .roles = target_roles.renameTag(.v, .d).repeat1d(.v, @intCast(n_batch)),
            };
        }

        /// Forward-fill unobserved positions.
        /// Leading unobserved positions are forward filled with the unobserved value.
        fn ffill_mask(inputs: Tensor, mask: Tensor, axis: anytype) Tensor {
            // A tensor like inputs with an arange across axis.
            // We replaced the masked indices with preceding non masked index.
            var idx = Tensor.iota(inputs.shape(), axis);
            idx = mask.select(idx, .scalar(0, idx.dtype()));
            idx = idx.cumulativeMax(axis);

            // Gather indices to forward fill masked values.
            const ax = inputs.axis(axis);
            return zml.ops.gather(
                inputs,
                &.{ax},
                &.{idx.renameTag(ax, ._gather_axis)}, // gathered axis must not share its tag with inputs
                .{ .indices_are_sorted = true },
            ).renameTag(._gather_axis, inputs.shape().tag(ax));
        }
    };

    pub fn forward(targets: Tensor) Rows {
        return .make(.make(targets));
    }
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

pub const Model = struct {
    residuals: ResidualBlock,

    pub fn forward(self: Model, targets: Tensor) PreProcessor.Rows {
        _ = self;
        return PreProcessor.forward(targets);
    }
};
