const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;

const scaling = @import("./scaling.zig");
const layer = @import("./layer.zig");

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
        /// Input tensor with dimensions {.batch, .variates, .time}
        pub fn make(targets: Tensor) TargetsSeries {
            assertTags(targets, .{ .batch, .variates, .time });
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
            assertTags(targets.values, .{ .batch, .variates, .time });
            assertTags(targets.mask, .{ .batch, .variates, .time });
            const n_batch = targets.values.dim(.batch);
            const n_targets = targets.values.dim(.variates);
            const target_roles = Tensor.scalar(@backingInt(Roles.target), .u8)
                .broad(.init(.{ .variates = n_targets }, .u8));
            return .{
                .values = ffillMissing(targets.values, targets.mask, .time).renameTag(.variates, .d),
                .mask = targets.mask.renameTag(.variates, .d),
                .roles = target_roles.renameTag(.variates, .d).repeat1d(.variates, @intCast(n_batch)),
            };
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
        assertTags(targets, .{ .batch, .variates, .time });
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
        assertTags(rows.values, .{ .batch, .d, .time });
        assertTags(rows.mask, .{ .batch, .d, .time });
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
            .rows = .{ .values = self.toPatches(scaled), .mask = self.toPatches(rows.mask) },
            .stats = stats,
        };
    }
};

pub const Model = struct {
    patcher: PatchBuilder,
    residuals: layer.ResidualBlock,

    pub fn forward(self: Model, targets: Tensor) PatchBuilder.Output {
        assertTags(targets, .{ .batch, .variates, .time });
        const preprocessed = PreProcessor.forward(targets);
        const patches = self.patcher.forward(preprocessed);
        return patches;
    }
};

/// Debug-assert that `x` has exactly the given tags (in any order).
fn assertTags(x: Tensor, comptime tags: anytype) void {
    const n_tags = tags.len;
    const expected = comptime blk: {
        // Braces are doubled because this is spliced into a format string.
        var s: []const u8 = "{{";
        for (tags, 0..) |t, i| {
            s = s ++ (if (i == 0) "" else ", ") ++ "." ++ @tagName(t);
        }
        break :blk s ++ "}}";
    };
    stdx.debug.assert(
        x.rank() == n_tags and x.shape().hasTags(tags),
        "Expected tensor with tags " ++ expected ++ ", got {f}",
        .{x},
    );
}
