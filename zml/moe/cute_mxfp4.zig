//! Blackwell MXFP4 MoE backend with persistent CuTe GEMMs.

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const triton_mxfp4 = @import("triton_mxfp4.zig");
pub const kernels = @import("cute_kernels/moe.zig");

// `moe.zig`'s `refAllDecls` reaches this file but not its imports, so the
// kernel emit tests are only collected if `kernels` is referenced here.
test {
    _ = kernels;
}

pub const Parameters = triton_mxfp4.Parameters;

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

pub fn fusedExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
) zml.Tensor {
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "cute_mxfp4 backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "cute_mxfp4 backend expects gate_up bias and down bias to be null", .{});

    const gq = gate_up.quantization orelse @panic("cute_mxfp4 backend requires gate_up quantization to be set");
    const dq = down.quantization orelse @panic("cute_mxfp4 backend requires down quantization to be set");
    stdx.debug.assert(gq.scheme == .mxfp4 and dq.scheme == .mxfp4, "cute_mxfp4 expects gate_up and down quantization scheme to be mxfp4, got {} and {}", .{ gq.scheme, dq.scheme });

    // Weight storage for mxfp4 in HF is expressed as u8 or i8
    stdx.debug.assert(gate_up.weight.dtype() == .u8 or gate_up.weight.dtype() == .i8, "cute_mxfp4 expects gate_up weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});
    stdx.debug.assert(down.weight.dtype() == .u8 or down.weight.dtype() == .i8, "cute_mxfp4 expects down weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});

    const expert_parallelism = gate_up.weight.shape().partition(.expert).isSharded();
    const hidden = down.weight.dim(1);
    const intermediate = down.weight.dim(2) * 2;

    // The CuTe GEMMs are specialized on one set of dimensions
    kernels.validateShapes(hidden, intermediate);

    // The SwiGLU clamp is a kernel parameter, but the routing weight is
    // multiplied in the up epilogue, before the down projection: applying it
    // after would have to move into the down epilogue or the reduction.
    stdx.debug.assert(options.routing_weight_placement == .before_down, "cute_mxfp4 backend only supports routing_weight_placement = .before_down, got {}", .{options.routing_weight_placement});
    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = gate_up.weight,
        .s1 = gq.scales,
        .w2 = down.weight,
        .s2 = dq.scales,
        .topk = ids.dim(.topk),
        .swiglu_limit = options.activation.swiglu.limit.?,
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
