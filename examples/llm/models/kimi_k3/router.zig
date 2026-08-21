const std = @import("std");

const zml = @import("zml");

pub const Weights = struct {
    weight: zml.Tensor,
    correction_bias: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Weights {
        return .{
            .weight = store.createTensor(
                "weight",
                .{ .expert, .d },
                .{ .expert = .replicated, .d = .replicated },
            ),
            .correction_bias = store.createTensor(
                "e_score_correction_bias",
                .{.expert},
                .{ .expert = .replicated },
            ),
        };
    }
};

pub const Config = struct {
    top_k: u32,
    num_expert_group: u32 = 1,
    topk_group: u32 = 1,
    routed_scaling_factor: f32 = 1.0,
    renormalize: bool = true,
};

pub const Result = struct {
    logits: zml.Tensor,
    raw_scores: zml.Tensor,
    selection_scores: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_raw_weights: zml.Tensor,
    topk_weights: zml.Tensor,
};

fn groupedSelectionScores(selection: zml.Tensor, config: Config) zml.Tensor {
    if (config.num_expert_group <= 1 or config.num_expert_group <= config.topk_group) {
        return selection;
    }
    const expert_groups: i64 = @intCast(config.num_expert_group);
    std.debug.assert(@rem(selection.dim(.expert), expert_groups) == 0);
    const experts_per_group = @divExact(selection.dim(.expert), expert_groups);
    const grouped = selection.splitAxis(.expert, .{
        .group = config.num_expert_group,
        .member = experts_per_group,
    });
    const group_scores = grouped.topK(.{ .group_top = .member }, 2, .{})
        .values.sum(.group_top).squeeze(.group_top);
    const selected_groups = group_scores.topK(
        .{ .selected_group = .group },
        config.topk_group,
        .{},
    ).indices;
    const batch = selection.dim(.b);
    const sequence = selection.dim(.s);
    const compare_shape = zml.Shape.init(.{
        .b = batch,
        .s = sequence,
        .selected_group = config.topk_group,
        .group = config.num_expert_group,
    }, .i32);
    const selected = selected_groups.reshape(.{
        .b = batch,
        .s = sequence,
        .selected_group = config.topk_group,
        .group = 1,
    }).broad(compare_shape);
    const group_ids = zml.Tensor.arange(.{ .end = config.num_expert_group }, .i32)
        .withTags(.{.group})
        .reshape(.{ .b = 1, .s = 1, .selected_group = 1, .group = config.num_expert_group })
        .broad(compare_shape);
    const enabled_groups = selected.cmp(.EQ, group_ids)
        .convert(.i32).sum(.selected_group).squeeze(.selected_group)
        .cmp(.GT, zml.Tensor.scalar(0, .i32));
    const enabled = enabled_groups.reshape(.{
        .b = batch,
        .s = sequence,
        .group = config.num_expert_group,
        .member = 1,
    }).broad(grouped.shape());
    const masked = enabled.select(
        grouped,
        zml.Tensor.constant(selection.dtype().minValue()).broad(grouped.shape()),
    );
    return masked.merge(.{ .expert = .{ .group, .member } });
}

pub fn forward(hidden: zml.Tensor, weights: Weights, config: Config) Result {
    const logits = hidden.convert(.f32).dotWithPrecision(
        weights.weight.convert(.f32),
        .d,
        .highest,
    );
    const raw_scores = logits.sigmoid();
    const selection_scores = raw_scores.add(
        weights.correction_bias.convert(.f32).broad(raw_scores.shape()),
    );
    const scores_for_choice = groupedSelectionScores(selection_scores, config);
    const topk = scores_for_choice.topK(.{ .route = .expert }, config.top_k, .{});
    const topk_ids = topk.indices.convert(.i64);
    const topk_raw_weights = raw_scores.gather(.{ .expert = topk.indices }, .{});
    const topk_weights = if (config.renormalize and config.top_k > 1) blk: {
        const denominator = topk_raw_weights.sum(.route).addConstant(1e-20);
        break :blk topk_raw_weights.div(denominator.broad(topk_raw_weights.shape()))
            .scale(config.routed_scaling_factor);
    } else topk_raw_weights.scale(config.routed_scaling_factor);
    return .{
        .logits = logits,
        .raw_scores = raw_scores,
        .selection_scores = selection_scores,
        .topk_ids = topk_ids,
        .topk_raw_weights = topk_raw_weights,
        .topk_weights = topk_weights,
    };
}
