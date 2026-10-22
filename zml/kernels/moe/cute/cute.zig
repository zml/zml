//! Blackwell MXFP4 MoE backend with persistent CuTe GEMMs.

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
pub const kernels = @import("moe.zig");

// `moe.zig`'s `refAllDecls` reaches this file but not its imports, so the
// kernel emit tests are only collected if `kernels` is referenced here.
test {
    _ = kernels;
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

pub fn validateOptions(opts: zml.moe.Options) void {
    stdx.debug.assert(opts.activation == .swiglu, "cute backend only accepts swiglu activation, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.limit != null, "cute backend requires swiglu limit to be set", .{});
    stdx.debug.assert(opts.activation.swiglu.bias == null, "cute backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "cute backend requires swiglu scale to be null", .{});
}

pub fn fusedExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
) zml.Tensor {
    const plan = kernels.launchPlan(zml.Compiler.current().platform, input.dim(.t), down.weight.dim(1), down.weight.dim(2) * 2, gate_up.weight.dim(.expert), ids.dim(.topk)) catch |err|
        stdx.debug.panic("invalid CuTe MoE launch configuration: {s}", .{@errorName(err)});
    return fusedExpertsWithPlan(input, ids, weights, gate_up, down, options, plan);
}

/// The normal expert-parallel path with an explicitly resolved launch plan.
/// Configuration is baked into the compiled graph, never read from the environment.
pub fn fusedExpertsWithPlan(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
    plan: kernels.LaunchPlan,
) zml.Tensor {
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "cute backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "cute backend expects gate_up bias and down bias to be null", .{});

    const gq = gate_up.quantization orelse @panic("cute backend requires gate_up quantization to be set");
    const dq = down.quantization orelse @panic("cute backend requires down quantization to be set");
    stdx.debug.assert(gq.scheme == .mxfp4 and dq.scheme == .mxfp4, "cute expects gate_up and down quantization scheme to be mxfp4, got {} and {}", .{ gq.scheme, dq.scheme });

    return routedExperts(input, ids, weights, .{
        .w1 = gate_up.weight,
        .s1 = gq.scales,
        .w2 = down.weight,
        .s2 = dq.scales,
        .layout = .cute,
    }, options, plan);
}

/// `fusedExperts` on MegaMoE's routed experts (`zml.moe.megamoe.Weights`), so
/// that one copy of the weights serves both MoE kernels, e.g. MegaMoE for
/// prefill and these GEMMs for decode. The GEMMs read MegaMoE's layout in
/// place (`kernels.persistent.WeightLayout.megamoe`). MegaMoE's fused shared
/// expert is not run. Experts sharded over `.expert` run expert parallel, like
/// `fusedExperts`: rank r holds expert block r.
pub fn fusedMegaMoEExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    experts: zml.moe.megamoe.Weights,
    options: zml.moe.Options,
) zml.Tensor {
    zml.moe.validateRoutingInputs(input, ids, weights);
    const plan = kernels.launchPlan(zml.Compiler.current().platform, input.dim(.t), experts.l2_w.dim(1), experts.l2_w.dim(2) * 2, experts.l1_w.dim(.expert), ids.dim(.topk)) catch |err|
        stdx.debug.panic("invalid CuTe MoE launch configuration: {s}", .{@errorName(err)});
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "cute backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(experts.l1_sf.dtype() == .i32 and experts.l2_sf.dtype() == .i32, "cute expects MegaMoE's i32 scale words, got {} and {}", .{ experts.l1_sf.dtype(), experts.l2_sf.dtype() });
    return routedExperts(input, ids, weights, .{
        .w1 = experts.l1_w,
        .s1 = experts.l1_sf,
        .w2 = experts.l2_w,
        .s2 = experts.l2_sf,
        .layout = .megamoe,
    }, options, plan);
}

/// Routed expert weights `[experts, rows, columns]` and their scales, in `layout`.
const RoutedWeights = struct {
    w1: zml.Tensor,
    s1: zml.Tensor,
    w2: zml.Tensor,
    s2: zml.Tensor,
    layout: kernels.persistent.WeightLayout,
};

fn routedExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    routed: RoutedWeights,
    options: zml.moe.Options,
    plan: kernels.LaunchPlan,
) zml.Tensor {
    // Weight storage for mxfp4 in HF is expressed as u8 or i8
    stdx.debug.assert(routed.w1.dtype() == .u8 or routed.w1.dtype() == .i8, "cute expects gate_up weight dtype to be u8 or i8, got {}", .{routed.w1.dtype()});
    stdx.debug.assert(routed.w2.dtype() == .u8 or routed.w2.dtype() == .i8, "cute expects down weight dtype to be u8 or i8, got {}", .{routed.w2.dtype()});

    const expert_parallelism = routed.w1.shape().partition(.expert).isSharded();
    const hidden = routed.w2.dim(1);
    const intermediate = routed.w2.dim(2) * 2;

    // The CuTe GEMMs are specialized on one set of dimensions
    kernels.validateShapes(hidden, intermediate);
    plan.validate(zml.Compiler.current().platform, input.dim(.t), hidden, intermediate, routed.w1.dim(.expert), ids.dim(.topk)) catch |err|
        stdx.debug.panic("invalid CuTe MoE launch configuration: {s}", .{@errorName(err)});

    // The SwiGLU clamp is a kernel parameter, but the routing weight is
    // multiplied in the up epilogue, before the down projection: applying it
    // after would have to move into the down epilogue or the reduction.
    stdx.debug.assert(options.routing_weight_placement == .before_down, "cute backend only supports routing_weight_placement = .before_down, got {}", .{options.routing_weight_placement});
    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = routed.w1,
        .s1 = routed.s1,
        .w2 = routed.w2,
        .s2 = routed.s2,
        .weight_layout = routed.layout,
        .topk = ids.dim(.topk),
        .swiglu_limit = options.activation.swiglu.limit.?,
        .expert_parallel = expert_parallelism,
        .plan = plan,
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
    weight_layout: kernels.persistent.WeightLayout,
    topk: i64,
    swiglu_limit: f32,
    expert_parallel: bool,
    plan: kernels.LaunchPlan,

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
            .weight_layout = self.weight_layout,
        }, self.plan).reshape(self.input.shape().dims()).withTags(self.input.shape());

        return if (self.expert_parallel)
            zml.ops.allReduce(result, zml.Tensor.add)
        else
            result;
    }
};
