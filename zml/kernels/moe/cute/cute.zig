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
    stdx.debug.assert(opts.activation.swiglu.bias == null, "cute backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "cute backend requires swiglu scale to be null", .{});
}

pub fn fusedExperts(
    input: zml.Tensor,
    routing: zml.moe.LocalRouting,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
) zml.Tensor {
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "cute backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "cute backend expects gate_up bias and down bias to be null", .{});

    const gq = gate_up.quantization orelse @panic("cute backend requires gate_up quantization to be set");
    const dq = down.quantization orelse @panic("cute backend requires down quantization to be set");
    stdx.debug.assert(gq.scheme == .mxfp4 and dq.scheme == .mxfp4, "cute expects gate_up and down quantization scheme to be mxfp4, got {} and {}", .{ gq.scheme, dq.scheme });

    // Weight storage for mxfp4 in HF is expressed as u8 or i8
    stdx.debug.assert(gate_up.weight.dtype() == .u8 or gate_up.weight.dtype() == .i8, "cute expects gate_up weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});
    stdx.debug.assert(down.weight.dtype() == .u8 or down.weight.dtype() == .i8, "cute expects down weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});

    const hidden = down.weight.dim(1);
    const intermediate = down.weight.dim(2) * 2;

    // Check the boundary quantizer and CuTe GEMM tile constraints.
    kernels.validateShapes(hidden, intermediate);

    // The SwiGLU clamp is a kernel parameter, but the routing weight is
    // multiplied in the up epilogue, before the down projection: applying it
    // after would have to move into the down epilogue or the reduction.
    stdx.debug.assert(options.routing_weight_placement == .before_down, "cute backend only supports routing_weight_placement = .before_down, got {}", .{options.routing_weight_placement});
    const topk = routing.ids.dim(.topk);
    const experts = gate_up.weight.dim(.expert);
    const ids = routing.localIds().convert(.i32).reshape(.{ .token = .auto, .topk = topk });
    const tokens = ids.dim(.token);
    const routing_weights = routing.weights.convert(.f32).reshape(.{ tokens, topk });

    return kernels.forward(tokens, hidden, intermediate, experts, topk, options.activation.swiglu.limit, .{
        .x = input.reshape(.{ tokens, hidden }),
        .routing_weights = routing_weights,
        .w1 = gate_up.weight,
        .s1 = gq.scales,
        .w2 = down.weight,
        .s2 = dq.scales,
        .ids = ids,
    }).reshape(input.shape().dims()).withTags(input.shape());
}
