//! Blackwell MXFP4 MoE backend with persistent CuTe GEMMs.
const zml = @import("../zml.zig");
const triton_mxfp4 = @import("triton_mxfp4.zig");
pub const kernels = @import("cute_kernels/moe.zig");

// `moe.zig`'s `refAllDecls` reaches this file but not its imports, so the
// kernel emit tests are only collected if `kernels` is referenced here.
test {
    _ = kernels;
}

pub const Parameters = triton_mxfp4.Parameters;

/// Pack E8M0 weight scales in the exact 128x4 swizzle consumed by the TMA
/// scale-factor descriptor. Loaders apply this once at load time.
pub fn packWeightScales(scales: zml.Tensor) zml.Tensor {
    const experts = scales.dim(0);
    const rows = scales.dim(1);
    const groups = scales.dim(2);
    var packed_scales = scales.reshape(.{ experts, @divExact(rows, 128), 4, 32, @divExact(groups, 4), 4 })
        .transpose(.{ 0, 1, 4, 3, 2, 5 });
    return packed_scales.reshape(scales.shape()).withPartitioning(.{ .expert = .experts });
}

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
    // Weight storage for mxfp4 in HF is expressed as u8 or i8
    if ((gate_up.weight.dtype() != .u8 and gate_up.weight.dtype() != .i8) or
        (down.weight.dtype() != .u8 and down.weight.dtype() != .i8))
    {
        return error.UnsupportedWeightLayout;
    }

    const expert_parallelism = gate_up.weight.shape().partition(.expert).eql(.init(.experts));
    const hidden = down.weight.dim(1);
    const intermediate = down.weight.dim(2) * 2;
    // The CuTe GEMMs are specialized on one set of dimensions
    if (!kernels.isSupported(hidden, intermediate)) return error.UnsupportedShape;
    const activation_threshold = options.activation_threshold orelse return error.UnsupportedActivation;
    // The SwiGLU clamp is a kernel parameter, but the routing weight is
    // multiplied in the up epilogue, before the down projection: applying it
    // after would have to move into the down epilogue or the reduction.
    if (options.routing_weight_placement != .before_down) return error.UnsupportedRoutingWeightPlacement;
    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = gate_up.weight.bitCast(.u8),
        .s1 = gq.scales,
        .w2 = down.weight.bitCast(.u8),
        .s2 = dq.scales,
        .topk = parameters.num_experts_per_tok,
        .swiglu_limit = activation_threshold,
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
    swiglu_limit: f32,
    expert_parallel: bool,

    fn body(self: Context, _: zml.Shape) zml.Tensor {
        const experts = self.w1.dim(.expert);
        const hidden = self.w2.dim(1);
        const intermediate = self.w2.dim(2) * 2;
        var ids = self.ids.convert(.i32).reshape(.{ .token = .auto, .topk = self.topk });
        const tokens = ids.dim(.token);
        const routing_weights = self.weights.convert(.f32).reshape(.{ tokens, self.topk });
        if (self.expert_parallel) {
            const partition_id = zml.ops.partitionId().convert(.i32);
            const expert_start = partition_id.scale(experts);
            const expert_end = expert_start.addConstant(experts);
            // Routes of other ranks get expert -1 and are not computed here.
            const local = ids.cmp(.GE, expert_start).logical(.AND, ids.cmp(.LT, expert_end));
            ids = local.select(ids.sub(expert_start), .scalar(-1, .i32));
        }
        const result = kernels.forward(tokens, hidden, intermediate, experts, self.topk, self.swiglu_limit, .{
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
