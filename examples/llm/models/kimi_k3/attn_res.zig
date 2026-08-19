const std = @import("std");

const zml = @import("zml");

pub const Selection = struct {
    output: zml.Tensor,
    candidates: zml.Tensor,
    scores: zml.Tensor,
    probabilities: zml.Tensor,
};

/// Forward-local depth state.  It is intentionally separate from temporal
/// KDA/MLA generation caches and is reset for every forward/prefix execution.
pub const DepthWorkspace = struct {
    active_blocks: usize = 0,
    prefix_valid: bool = false,

    pub fn reset() DepthWorkspace {
        return .{};
    }

    pub fn beginLayer(self: *DepthWorkspace) void {
        self.prefix_valid = true;
    }

    pub fn appendBoundary(self: *DepthWorkspace, logical_layer: usize, block_size: usize) bool {
        if (logical_layer % block_size != 0) return false;
        self.active_blocks += 1;
        self.prefix_valid = false;
        return true;
    }

    pub fn addBranch(self: *DepthWorkspace) void {
        self.prefix_valid = true;
    }
};

/// Official K3 depth selector. Fixed-capacity block slots are masked so a
/// shorter selected prefix cannot read stale values left by an earlier call.
pub fn select(
    prefix_sum: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    norm_weight: zml.Tensor,
    projection_weight: zml.Tensor,
    eps: f32,
) Selection {
    const dtype = prefix_sum.dtype();
    const prefix_candidate = prefix_sum.reshape(.{
        .token = prefix_sum.dim(.token),
        .source = 1,
        .d = prefix_sum.dim(.d),
    });
    const candidates = zml.Tensor.concatenate(&.{ block_sources, prefix_candidate }, .source);
    const candidates_f32 = candidates.convert(.f32);
    const variance = candidates_f32.powByConst(2).mean(.d);
    const normalized = candidates_f32.mul(
        variance.addConstant(eps).rsqrt().broad(candidates_f32.shape()),
    );
    const direction = norm_weight.convert(.f32).mul(projection_weight.convert(.f32));
    const scores = normalized.mul(direction.broad(normalized.shape())).sum(.d).squeeze(.d);

    const prefix_active = zml.Tensor.scalar(true, .bool).reshape(.{ .source = 1 });
    const candidate_mask = zml.Tensor.concatenate(&.{ active_blocks, prefix_active }, .source);
    const masked_scores = candidate_mask.broad(scores.shape()).select(
        scores,
        zml.Tensor.scalar(-std.math.inf(f32), .f32).broad(scores.shape()),
    );
    const probabilities = masked_scores.softmax(.source);
    const weighted = probabilities.reshape(.{
        .token = probabilities.dim(.token),
        .source = probabilities.dim(.source),
        .d = 1,
    }).broad(candidates_f32.shape()).mul(candidates_f32);
    const output = weighted.sum(.source).squeeze(.source).convert(dtype);
    return .{
        .output = output,
        .candidates = candidates,
        .scores = scores,
        .probabilities = probabilities,
    };
}

pub fn selectEps1e6(
    prefix_sum: zml.Tensor,
    block_sources: zml.Tensor,
    active_blocks: zml.Tensor,
    norm_weight: zml.Tensor,
    projection_weight: zml.Tensor,
) Selection {
    return select(prefix_sum, block_sources, active_blocks, norm_weight, projection_weight, 1e-6);
}
