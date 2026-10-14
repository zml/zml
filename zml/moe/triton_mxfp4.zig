const std = @import("std");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
pub const kernels = @import("triton_kernels/mxfp4.zig");

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

pub fn validateOptions(opts: zml.moe.Options) void {
    stdx.debug.assert(opts.activation == .swiglu, "cute_mxfp4 backend only accepts swiglu activation, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.limit != null, "cute_mxfp4 backend requires swiglu limit to be set", .{});
    stdx.debug.assert(opts.activation.swiglu.bias == null, "cute_mxfp4 backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "cute_mxfp4 backend requires swiglu scale to be null", .{});
}

/// Row-major MXFP4 E2M1 weights and linear E8M0 block32 scales.
/// BF16 activations are quantized on the GPU to FP8 with per-32 E8M0 scales.
pub fn fusedExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
) zml.Tensor {
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "triton_mxfp4 backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "triton_mxfp4 backend expects gate_up bias and down bias to be null", .{});

    const gq = gate_up.quantization orelse @panic("triton_mxfp4 backend requires gate_up quantization to be set");
    const dq = down.quantization orelse @panic("triton_mxfp4 backend requires down quantization to be set");
    stdx.debug.assert(gq.scheme == .mxfp4 and dq.scheme == .mxfp4, "triton_mxfp4 expects gate_up and down quantization scheme to be mxfp4, got {} and {}", .{ gq.scheme, dq.scheme });

    // Weight storage for mxfp4 in HF is expressed as u8 or i8
    stdx.debug.assert(gate_up.weight.dtype() == .u8 or gate_up.weight.dtype() == .i8, "triton_mxfp4 expects gate_up weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});
    stdx.debug.assert(down.weight.dtype() == .u8 or down.weight.dtype() == .i8, "triton_mxfp4 expects down weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});

    const expert_parallelism = gate_up.weight.shape().partition(.expert).isSharded();

    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = gate_up.weight.bitCast(.u8),
        .s1 = gq.scales,
        .w2 = down.weight.bitCast(.u8),
        .s2 = dq.scales,
        .global_experts = gate_up.weight.dim(.expert),
        .topk = ids.dim(.topk),
        .limit = options.activation.swiglu.limit.?,
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
        var ids = self.ids.convert(.i32).reshape(.{ .token = .auto, .topk = self.topk });
        const tokens = ids.dim(.token);
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
            .ids = ids,
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
