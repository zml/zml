const std = @import("std");

const zml = @import("../zml.zig");
pub const kernels = @import("triton_kernels/mxfp4.zig");

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

pub const Parameters = struct {
    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: zml.moe.ActivationMode,
    };

    num_experts_per_tok: u32,
    activation: zml.moe.ActivationMode,

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
        };
    }
};

/// Row-major MXFP4 E2M1 weights and linear E8M0 block32 scales.
/// BF16 activations are quantized on the GPU to FP8 with per-32 E8M0 scales.
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
    if ((gate_up.weight.dtype() != .u8 and gate_up.weight.dtype() != .i8) or (down.weight.dtype() != .u8 and down.weight.dtype() != .i8)) return error.UnsupportedWeightLayout;

    const expert_parallelism = gate_up.weight.shape().partition(.expert).eql(.init(.experts));

    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = gate_up.weight.bitCast(.u8),
        .s1 = gq.scales,
        .w2 = down.weight.bitCast(.u8),
        .s2 = dq.scales,
        .global_experts = gate_up.weight.dim(.expert),
        .topk = parameters.num_experts_per_tok,
        .limit = options.activation_threshold orelse 0,
        .routing_weight_placement = options.routing_weight_placement,
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
    global_experts: i64,
    topk: i64,
    limit: f32,
    routing_weight_placement: zml.moe.triton.RoutingWeightPlacement,
    expert_parallel: bool,

    fn body(self: Context, _: zml.Shape) zml.Tensor {
        const experts = self.w1.dim(.expert);
        const hidden = self.w2.dim(1);
        const intermediate = self.w2.dim(2) * 2;
        const tokens: i64 = @intCast(self.input.count() / @as(usize, @intCast(hidden)));

        var ids = self.ids.convert(.i32);
        if (self.expert_parallel) {
            const partition_id = zml.ops.partitionId().convert(.i32);
            ids = ids.sub(partition_id.scale(experts));
        }

        const cfg: kernels.Config = .{
            .tokens = tokens,
            .hidden = hidden,
            .intermediate = intermediate,
            .experts = experts,
            .global_experts = self.global_experts,
            .topk = self.topk,
            .swiglu_limit = self.limit,
            .routing_weight_placement = self.routing_weight_placement,
        };

        const result = kernels.forward(cfg, .{
            .x = self.input.reshape(.{ tokens, hidden }),
            .ids = ids.reshape(.{ tokens, self.topk }),
            .scales = self.weights.convert(.f32).reshape(.{ tokens, self.topk }),
            .w1 = self.w1,
            .s1 = self.s1,
            .w2 = self.w2,
            .s2 = self.s2,
        }).reshape(self.input.shape().dims()).withTags(self.input.shape());

        return if (self.expert_parallel)
            zml.ops.allReduce(result, zml.Tensor.add)
        else
            result;
    }
};
