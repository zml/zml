pub const ActivationMode = enum {
    silu,
    relu,
    quick_gelu_plus_one,
};

const stdx = @import("stdx");
const zml = @import("../zml.zig");
const Tensor = zml.Tensor;

pub const CanonicalInputs = struct {
    b: i64,
    s: i64,
    hidden: Tensor,
    gate_up: Tensor,
    down: Tensor,
    weights: Tensor,
    ids: Tensor,
};

pub fn canonicalizeInputs(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
) CanonicalInputs {
    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    return .{
        .b = b,
        .s = s,
        .hidden = hidden_states.reshape(.{ .token = b * s, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in }),
        .gate_up = w1,
        .down = w2.withTags(.{ .expert, .mid, .out }),
        .weights = topk_weights.reshape(.{ .token = b * s, .topk = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk }),
        .ids = topk_ids.reshape(.{ .token = b * s, .topk = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk }),
    };
}

pub fn normalizeGateUpWeight(w1: Tensor) Tensor {
    return switch (w1.rank()) {
        3 => w1.withTags(.{ .expert, .out, .in }),
        4 => w1.withTags(.{ .expert, .gate_up, .mid, .in }),
        else => stdx.debug.panic("Unsupported gate_up weight rank {}, expected 3 or 4", .{w1.rank()}),
    };
}

pub fn gateUpOutDim(gate_up: Tensor) i64 {
    return if (gate_up.rank() == 4)
        gate_up.dim(.gate_up) * gate_up.dim(.mid)
    else
        gate_up.dim(.out);
}

pub fn flattenGateUpForMatmul(gate_up: Tensor) Tensor {
    if (gate_up.rank() == 3) return gate_up;

    stdx.debug.assert(gate_up.rank() == 4, "Unsupported gate_up local rank {}, expected 4", .{gate_up.rank()});
    return gate_up.reshape(.{
        .expert = gate_up.dim(.expert),
        .out = gate_up.dim(.gate_up) * gate_up.dim(.mid),
        .in = gate_up.dim(.in),
    }).withTags(.{ .expert, .out, .in });
}
