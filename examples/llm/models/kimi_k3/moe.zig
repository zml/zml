const std = @import("std");

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
        .output = primitives.stableLinear(activated, weights.shared_down, .intermediate),
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
        .output = primitives.stableLinear(normalized, weights.routed_up, .latent),
    };
}
pub const Weights = struct {
    gate: router.Weights,
    experts: ExpertBank,
    dense: DenseWeights,
};

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
pub const RoutedExpertResult = struct {
    route_outputs: zml.Tensor,
    combined_latent_f32: zml.Tensor,
};

fn combineRouteOutputs(route_outputs: zml.Tensor, route_weights: zml.Tensor) zml.Tensor {
    return route_outputs.convert(.f32)
        .mul(route_weights.convert(.f32).broad(route_outputs.shape()))
        .sum(.route).squeeze(.route);
}

fn replicatedRoutedExperts(
    expert_input: zml.Tensor,
    expert_ids: zml.Tensor,
    route_weights: zml.Tensor,
    experts: ExpertBank,
) RoutedExpertResult {
    const gate = nativeBankLinear(expert_input, expert_ids, experts.w1, zml.Shape.toTag(.intermediate));
    const up = nativeBankLinear(expert_input, expert_ids, experts.w3, zml.Shape.toTag(.intermediate));
    const activated = primitives.situGlu(gate, up);
    const route_outputs = nativeBankLinear(
        activated.rename(.{ .intermediate = .d }),
        expert_ids,
        experts.w2,
        zml.Shape.toTag(.latent),
    );
    return .{
        .route_outputs = route_outputs,
        .combined_latent_f32 = combineRouteOutputs(route_outputs, route_weights),
    };
}

fn sharedAxisRoutedExperts(
    expert_input: zml.Tensor,
    expert_ids: zml.Tensor,
    route_weights: zml.Tensor,
    experts: ExpertBank,
) RoutedExpertResult {
    const route_shape = zml.Shape.init(.{
        .token = expert_ids.dim(.token),
        .route = expert_ids.dim(.route),
        .latent = experts.w2.values.dim(.latent),
    }, .bf16);
    const combined_shape = zml.Shape.init(.{
        .token = expert_ids.dim(.token),
        .latent = experts.w2.values.dim(.latent),
    }, .f32);

    const outputs = zml.ops.manualComputation(
        .{
            expert_input,
            expert_ids,
            route_weights,
            experts.w1.values,
            experts.w1.scale,
            experts.w2.values,
            experts.w2.scale,
            experts.w3.values,
            experts.w3.scale,
        },
        .{ route_shape, combined_shape },
        {},
        (struct {
            fn body(
                _: void,
                allocator: std.mem.Allocator,
                sharded_inputs: []const zml.Tensor,
                _: []const zml.Shape,
            ) []const zml.Tensor {
                const local_expert_count = sharded_inputs[3].dim(.expert);
                const partition_id = zml.ops.partitionId().convert(.i32);
                const expert_start = partition_id.scale(local_expert_count).convert(.i32);
                const global_ids = sharded_inputs[1].convert(.i32);
                const owned = global_ids.cmp(.GE, expert_start).logical(
                    .AND,
                    global_ids.cmp(.LT, expert_start.addConstant(local_expert_count)),
                );
                const local_ids = owned.select(global_ids.sub(expert_start), zml.Tensor.scalar(-1, .i32));
                const local_experts: ExpertBank = .{
                    .w1 = .{ .values = sharded_inputs[3], .scale = sharded_inputs[4] },
                    .w2 = .{ .values = sharded_inputs[5], .scale = sharded_inputs[6] },
                    .w3 = .{ .values = sharded_inputs[7], .scale = sharded_inputs[8] },
                };
                const gate = nativeBankLinear(sharded_inputs[0], local_ids, local_experts.w1, zml.Shape.toTag(.intermediate));
                const up = nativeBankLinear(sharded_inputs[0], local_ids, local_experts.w3, zml.Shape.toTag(.intermediate));
                const activated = primitives.situGlu(gate, up);
                const local_route_outputs = nativeBankLinear(
                    activated.rename(.{ .intermediate = .d }),
                    local_ids,
                    local_experts.w2,
                    zml.Shape.toTag(.latent),
                );
                const local_combined = combineRouteOutputs(local_route_outputs, sharded_inputs[2]);

                const result = allocator.alloc(zml.Tensor, 2) catch unreachable;
                result[0] = zml.ops.allReduce(local_route_outputs, zml.Tensor.add);
                result[1] = zml.ops.allReduce(local_combined, zml.Tensor.add);
                return result;
            }
        }).body,
    );
    return .{ .route_outputs = outputs[0], .combined_latent_f32 = outputs[1] };
}

/// Conformance entry point for explicit global route IDs. Production calls the
/// same function after its router has selected those IDs.
pub fn routedExpertsForTest(
    expert_input: zml.Tensor,
    expert_ids: zml.Tensor,
    route_weights: zml.Tensor,
    experts: ExpertBank,
) RoutedExpertResult {
    const expert_partition = experts.w1.values.shape().partition(.expert);
    if (!expert_partition.eql(.init(.experts))) {
        return replicatedRoutedExperts(expert_input, expert_ids, route_weights, experts);
    }

    std.debug.assert(experts.w1.scale.shape().partition(.expert).eql(.init(.experts)));
    std.debug.assert(experts.w2.values.shape().partition(.expert).eql(.init(.experts)));
    std.debug.assert(experts.w2.scale.shape().partition(.expert).eql(.init(.experts)));
    std.debug.assert(experts.w3.values.shape().partition(.expert).eql(.init(.experts)));
    std.debug.assert(experts.w3.scale.shape().partition(.expert).eql(.init(.experts)));
    return sharedAxisRoutedExperts(expert_input, expert_ids, route_weights, experts);
}

/// Production-shaped Stable LatentMoE. Router IDs address the normal global
/// expert axis directly; compact expert maps and injected routes are forbidden.
pub fn forward(hidden: zml.Tensor, weights: Weights, config: router.Config) Result {
    const route = router.forward(hidden, weights.gate, config);
    const token_hidden = hidden.merge(.{ .token = .{ .b, .s } });
    const expert_ids = route.topk_ids.merge(.{ .token = .{ .b, .s } });
    const route_weights = route.topk_weights.merge(.{ .token = .{ .b, .s } });
    const routed_down = token_hidden.dot(weights.dense.routed_down, .d);
    const expert_input = routed_down.rename(.{ .latent = .d });
    const routed_experts = routedExpertsForTest(expert_input, expert_ids, route_weights, weights.experts);
    const route_outputs = routed_experts.route_outputs;
    const combined_latent = routed_experts.combined_latent_f32.convert(hidden.dtype());
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
