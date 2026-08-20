const zml = @import("zml");

const grouped_mxfp4 = @import("grouped_mxfp4.zig");
const primitives = @import("primitives.zig");
const router = @import("router.zig");

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

/// Native CUDA grouped MXFP4 bank execution. Keep `bankLinear` only as the
/// differential oracle until cleanup milestone M20.
pub fn nativeBankLinear(
    input: zml.Tensor,
    expert_ids: zml.Tensor,
    bank: Mxfp4Bank,
    comptime output_tag: zml.Shape.Tag,
) zml.Tensor {
    return grouped_mxfp4.linear(input, expert_ids, bank.values, bank.scale)
        .rename(.{ .out = output_tag });
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
pub const Weights = struct {
    gate: router.Weights,
    experts: ExpertBank,
    dense: DenseWeights,
};

// KIMI_K3_TEMP_REMOVE_M20: named router/expert boundaries are returned for
// composed layer-family parity and reduced to output plus route telemetry in M20.
pub const Result = struct {
    route: router.Result,
    routed_down: zml.Tensor,
    route_outputs: zml.Tensor,
    combined_latent: zml.Tensor,
    routed_norm: zml.Tensor,
    routed_up: zml.Tensor,
    shared_output: zml.Tensor,
    output: zml.Tensor,
};

/// Production-shaped Stable LatentMoE. Router IDs address the normal global
/// expert axis directly; compact expert maps and injected routes are forbidden.
pub fn forward(hidden: zml.Tensor, weights: Weights, config: router.Config) Result {
    const route = router.forward(hidden, weights.gate, config);
    const token_hidden = hidden.merge(.{ .token = .{ .b, .s } });
    const expert_ids = route.topk_ids.merge(.{ .token = .{ .b, .s } });
    const route_weights = route.topk_weights.merge(.{ .token = .{ .b, .s } });
    const routed_down = token_hidden.dot(weights.dense.routed_down, .d);
    const expert_input = routed_down.rename(.{ .latent = .d });
    const gate = nativeBankLinear(expert_input, expert_ids, weights.experts.w1, zml.Shape.toTag(.intermediate));
    const up = nativeBankLinear(expert_input, expert_ids, weights.experts.w3, zml.Shape.toTag(.intermediate));
    const activated = primitives.situGlu(gate, up);
    const route_outputs = nativeBankLinear(
        activated.rename(.{ .intermediate = .d }),
        expert_ids,
        weights.experts.w2,
        zml.Shape.toTag(.latent),
    );
    const combined_latent = route_outputs.convert(.f32)
        .mul(route_weights.convert(.f32).broad(route_outputs.shape()))
        .sum(.route).squeeze(.route).convert(hidden.dtype());
    const routed = finishRouted(combined_latent, weights.dense);
    const shared = sharedMlp(token_hidden, weights.dense);
    const output = routed.output.add(shared.output).reshape(.{
        .b = hidden.dim(.b),
        .s = hidden.dim(.s),
        .d = hidden.dim(.d),
    });
    return .{
        .route = route,
        .routed_down = routed_down,
        .route_outputs = route_outputs,
        .combined_latent = combined_latent,
        .routed_norm = routed.normalized,
        .routed_up = routed.output,
        .shared_output = shared.output,
        .output = output,
    };
}
