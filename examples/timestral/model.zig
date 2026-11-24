const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const scaling = @import("./scaling.zig");
const layer = @import("./layer.zig");
const masking = @import("./masking.zig");
const debug = @import("./debug.zig");

const assertTags = debug.assertTags;
const assertedTags = debug.assertedTags;
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
        /// Input tensor with dimensions {.batch, .variate, .time}
        pub fn make(targets: Tensor) TargetsSeries {
            assertTags(targets, .{ .batch, .variate, .time });

            const mask = targets.isFinite(); // exclude NaN and inf
            const values = mask.select(targets, .scalar(0, targets.dtype()));

            return .{
                .values = assertedTags(values, .{ .batch, .variate, .time }),
                .mask = assertedTags(mask, .{ .batch, .variate, .time }),
            };
        }
    };

    const Rows = struct {
        /// Time series and covariate values.
        values: Tensor,
        /// Validity mask.
        mask: Tensor,
        /// Type of the time series covariate (target, past only, or known future).
        roles: Tensor,
        /// Index in the batch.
        group: Tensor,

        pub fn make(targets: TargetsSeries) Rows {
            assertTags(targets.values, .{ .batch, .variate, .time });
            assertTags(targets.mask, .{ .batch, .variate, .time });

            const n_batch = targets.values.dim(.batch);
            const n_variate = targets.values.dim(.variate);

            const values = ffillMissing(targets.values, targets.mask, .time);
            const roles = Tensor.scalar(@backingInt(Roles.target), .u8)
                .broad(.init(.{ .batch = n_batch, .variate = n_variate }, .u8));
            const group = Tensor.arange(.{ .end = n_batch }, .u32)
                .withTags(.{.batch})
                .broad(roles.shape().withDtype(.u32));

            return .{
                .values = assertedTags(asRows(values), .{ .row, .time }),
                .mask = assertedTags(asRows(targets.mask), .{ .row, .time }),
                .roles = assertedTags(asRows(roles), .{.row}),
                .group = assertedTags(asRows(group), .{.row}),
            };
        }

        pub fn asRows(x: Tensor) Tensor {
            return x.merge(.{ .row = .{ .batch, .variate } });
        }

        /// Forward-fill unobserved positions.
        /// Leading unobserved positions are forward filled with the unobserved value.
        fn ffillMissing(inputs: Tensor, mask: Tensor, axis: anytype) Tensor {
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
        assertTags(targets, .{ .batch, .variate, .time });
        return .make(.make(targets));
    }
};

pub const PatchBuilder = struct {
    pub const Options = struct {
        minimum_scale: f32,
        patch_size: u32,
        scale_factor_exponent: f32 = 10.0,
        stabilize_with_global: bool,
        use_asinh: bool,
    };

    options: Options,

    const PatchRows = struct {
        values: Tensor,
        mask: Tensor,
    };

    const Output = struct {
        rows: PatchRows,
        stats: scaling.LocationScales,
    };

    /// Split the `.time` axis into a `.token` sequence axis and a `.h` patch dimension.
    fn toPatches(self: PatchBuilder, x: Tensor) Tensor {
        return x.splitAxis(.time, .{ .token = .auto, .h = self.options.patch_size });
    }

    /// Transform input time series into tokens.
    ///
    /// Input are expected in shape `{.row, .time}` and the output is `{.row, .h, .token}`.
    pub fn forward(self: PatchBuilder, rows: PreProcessor.Rows) Output {
        assertTags(rows.values, .{ .row, .time });
        assertTags(rows.mask, .{ .row, .time });
        // Scaling locations and scale for all timesteps
        var stats = scaling.computeCausalStatistics(
            rows.values,
            rows.mask,
            .{
                .minimum_scale = self.options.minimum_scale,
                .stabilize_with_global = self.options.stabilize_with_global,
            },
            .time,
        );
        // Use only the last value in each patch for its whole patch
        const ps: u31 = @intCast(self.options.patch_size);
        stats.scales = scaling.patchwiseBroadcastLast(stats.scales, .time, ps);
        stats.locations = scaling.patchwiseBroadcastLast(stats.locations, .time, ps);
        // Scale inputs with locations and scales
        const scaled = scaling.scaleByAffineAsinh(rows.values, rows.mask, stats, self.options.use_asinh);

        return .{
            .rows = .{
                .values = assertedTags(self.toPatches(scaled), .{ .row, .token, .h }),
                .mask = assertedTags(self.toPatches(rows.mask), .{ .row, .token, .h }),
            },
            .stats = stats,
        };
    }
};

const PatchRepresentation = struct {
    patch_proj: layer.ResidualBlock,
    forecast_query: Tensor,

    pub fn init(store: zml.io.TensorStore.View) PatchRepresentation {
        const query = store.createTensor("forecast_query", .{.h}, .replicated, .{ .h = .replicated });
        return .{
            .patch_proj = .init(store.withPrefix("patch_proj")),
            .forecast_query = query,
        };
    }

    pub fn forward(self: *const PatchRepresentation, rows: PatchBuilder.PatchRows) Tensor {
        assertTags(rows.values, .{ .row, .token, .h });
        assertTags(rows.mask, .{ .row, .token, .h });
        const dtype = rows.values.dtype();
        const x: Tensor = .concatenate(&[_]Tensor{ rows.values, rows.mask.not().convert(dtype) }, .h);
        const out = self.patch_proj.forward(x.renameTag(.h, .d)).renameTag(.out, .h);
        return assertedTags(out, .{ .row, .token, .h });
    }

    pub fn forecast_tokens(self: *const PatchRepresentation, rows: u32, n_patches: u32) Tensor {
        assertTags(self.forecast_query, .{.h});
        const token = self.forecast_query.reshape(.{ .batch = 1, .token = 1, .h = .auto });
        return token.broad(token.shape().setDim(.batch, rows).setDim(.token, n_patches));
    }
};

pub const AttentionMasking = struct {
    pub const Options = struct {
        attend_missing_patches: bool,
    };

    options: Options,
    horizon: u32,

    const Masks = struct {
        time: Tensor,
        variate: Tensor,
    };

    pub fn forward(self: *const AttentionMasking, observed: Tensor, group: Tensor) Masks {
        assertTags(observed, .{ .row, .token, .h });
        const context_masks = masking.Context.forward(observed, self.options.attend_missing_patches);
        const time_mask = masking.Window.forward(context_masks.attended, self.horizon);
        const variate_mask = masking.Variate.forward(context_masks.attended, group, self.horizon);
        return .{
            .time = assertedTags(time_mask, .{ .row, .token_q, .token_k }),
            .variate = assertedTags(variate_mask, .{ .token, .row_q, .row_k }),
        };
    }
};

pub const Model = struct {
    patcher: PatchBuilder,
    representation: PatchRepresentation,
    attn_masking: AttentionMasking,

    pub fn forward(self: Model, targets: Tensor) Tensor {
        assertTags(targets, .{ .batch, .variate, .time });
        const preprocessed = PreProcessor.forward(targets);
        const patches = self.patcher.forward(preprocessed);
        const reprs = self.representation.forward(patches.rows);

        const attn_masks = self.attn_masking.forward(patches.rows.mask, preprocessed.group);
        _ = attn_masks;
        return reprs;
    }
};
