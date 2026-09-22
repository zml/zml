//! Blackwell MXFP4 MoE backend with persistent CuTe GEMMs.
const zml = @import("../zml.zig");
const triton_mxfp4 = @import("triton_mxfp4.zig");
pub const kernels = @import("cute_kernels/moe.zig");

test {
    _ = kernels;
}

pub const Parameters = triton_mxfp4.Parameters;

/// This backend consumes expert weight scales in the 128x4 layout read by the
/// SM100 scale-factor tensor maps. Loaders apply this once to the stacked
/// `[experts, rows, k/32]` E8M0 scales (gate/up rows interleaved).
pub const packWeightScales = kernels.packWeightScales;

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

pub fn fusedExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
    parameters: Parameters,
) !zml.Tensor {
    if (parameters.activation != .silu) return error.UnsupportedActivation;
    if (input.dtype() != .bf16) return error.UnsupportedDataType;
    if (gate_up.bias != null or down.bias != null) return error.UnsupportedBias;
    const gq = gate_up.quantization orelse return error.UnsupportedQuantization;
    const dq = down.quantization orelse return error.UnsupportedQuantization;
    if (gq.scheme != .mxfp4 or dq.scheme != .mxfp4) return error.UnsupportedQuantization;
    if ((gate_up.weight.dtype() != .u8 and gate_up.weight.dtype() != .i8) or
        (down.weight.dtype() != .u8 and down.weight.dtype() != .i8))
    {
        return error.UnsupportedWeightLayout;
    }

    const expert_parallelism = gate_up.weight.shape().partition(.expert).eql(.init(.experts));
    const hidden = down.weight.dim(1);
    const intermediate = down.weight.dim(2) * 2;
    const tokens: i64 = @intCast(input.count() / @as(usize, @intCast(hidden)));
    if (!kernels.isSupported(tokens, hidden, intermediate)) {
        // Shapes without a CuTe specialization run the Triton backend on the
        // same weights, after restoring the linear scale layout.
        var linear_gate_up = gate_up;
        var linear_down = down;
        linear_gate_up.quantization.?.scales = kernels.unpackWeightScales(gq.scales);
        linear_down.quantization.?.scales = kernels.unpackWeightScales(dq.scales);
        return triton_mxfp4.fusedExperts(input, ids, weights, linear_gate_up, linear_down, options, parameters);
    }
    const activation_threshold = options.activation_threshold orelse return error.UnsupportedActivation;
    if (activation_threshold != 10.0 or options.routing_weight_placement != .before_down) {
        return error.UnsupportedActivation;
    }
    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = gate_up.weight.bitCast(.u8),
        .s1 = gq.scales,
        .w2 = down.weight.bitCast(.u8),
        .s2 = dq.scales,
        .topk = parameters.num_experts_per_tok,
        .expert_parallel = expert_parallelism,
    };
    return if (expert_parallelism)
        zml.ops.manualComputation(Context.body, context, input.shape())
    else
        context.body(input.shape());
}

const Context = struct {
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    w1: zml.Tensor,
    s1: zml.Tensor,
    w2: zml.Tensor,
    s2: zml.Tensor,
    topk: i64,
    expert_parallel: bool,

    fn body(self: Context, _: zml.Shape) zml.Tensor {
        const experts = self.w1.dim(.expert);
        const hidden = self.w2.dim(1);
        const intermediate = self.w2.dim(2) * 2;
        const tokens: i64 = @intCast(self.input.count() / @as(usize, @intCast(hidden)));
        var ids = self.ids.convert(.i32).reshape(.{ tokens, self.topk });
        var routing_weights = self.weights.convert(.f32).reshape(.{ tokens, self.topk });
        if (self.expert_parallel) {
            const partition_id = zml.ops.partitionId().convert(.i32);
            const expert_start = partition_id.scale(experts);
            const expert_end = expert_start.addConstant(experts);
            const local = ids.cmp(.GE, expert_start).logical(.AND, ids.cmp(.LT, expert_end));
            ids = local.select(ids.sub(expert_start), zml.Tensor.scalar(0, .i32));
            routing_weights = local.select(routing_weights, zml.Tensor.scalar(0, .f32));
        }
        const result = kernels.forward(tokens, hidden, intermediate, experts, self.topk, .{
            .x = self.input.reshape(.{ tokens, hidden }),
            .routing_weights = routing_weights,
            .w1 = self.w1,
            .s1 = self.s1,
            .w2 = self.w2,
            .s2 = self.s2,
            .ids = ids,
        }).reshape(self.input.shape().dims()).withTags(self.input.shape());

        return if (self.expert_parallel)
            zml.ops.allReduce(result, zml.Tensor.add)
        else
            result;
    }
};
