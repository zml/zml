const zml = @import("zml");

const primitives = @import("primitives.zig");

pub const Mxfp4Bank = struct {
    values: zml.Tensor,
    scale: zml.Tensor,
};

pub const ExpertBank = struct {
    w1: Mxfp4Bank,
    w2: Mxfp4Bank,
    w3: Mxfp4Bank,
};

pub const DenseWeights = struct {
    routed_down: zml.Tensor,
    routed_norm: zml.Tensor,
    routed_up: zml.Tensor,
    shared_gate: zml.Tensor,
    shared_up: zml.Tensor,
    shared_down: zml.Tensor,
};

/// Slow correctness-first MXFP4 bank execution. `expert_ids` index the normal
/// expert axis; there is deliberately no global-to-local map parameter.
pub fn bankLinear(input: zml.Tensor, expert_ids: zml.Tensor, bank: Mxfp4Bank) zml.Tensor {
    const packed_values = bank.values.gather(.{ .expert = expert_ids.convert(.i32) }, .{});
    const scale = bank.scale.gather(.{ .expert = expert_ids.convert(.i32) }, .{});
    const weight = primitives.dequantizeMxfp4(packed_values, scale);
    const route_input = if (input.shape().hasTag(.route) != null)
        input
    else
        input.reshape(.{
            .token = input.dim(.token),
            .route = 1,
            .d = input.dim(.d),
        }).broad(zml.Shape.init(.{
            .token = input.dim(.token),
            .route = expert_ids.dim(.route),
            .d = input.dim(.d),
        }, input.dtype()));
    return route_input.convert(.f32).dotWithPrecision(weight, .d, .highest).convert(input.dtype());
}

pub fn probeLinear(input: zml.Tensor, bank: Mxfp4Bank) zml.Tensor {
    const weight = primitives.dequantizeMxfp4(bank.values, bank.scale);
    return input.convert(.f32).dotWithPrecision(weight, .d, .highest);
}

pub fn sharedMlp(input: zml.Tensor, weights: DenseWeights) struct {
    gate: zml.Tensor,
    up: zml.Tensor,
    activated: zml.Tensor,
    output: zml.Tensor,
} {
    const gate = input.dot(weights.shared_gate, .d);
    const up = input.dot(weights.shared_up, .d);
    const activated = primitives.situGlu(gate, up);
    return .{
        .gate = gate,
        .up = up,
        .activated = activated,
        .output = activated.dot(weights.shared_down, .intermediate),
    };
}

pub fn finishRouted(combined: zml.Tensor, weights: DenseWeights) struct {
    normalized: zml.Tensor,
    output: zml.Tensor,
} {
    const normalized = primitives.rmsNorm(
        combined.rename(.{ .latent = .d }),
        weights.routed_norm.rename(.{ .latent = .d }),
        1e-5,
    ).rename(.{ .d = .latent });
    return .{
        .normalized = normalized,
        .output = normalized.dot(weights.routed_up, .latent),
    };
}
