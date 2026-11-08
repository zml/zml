//! Blackwell MXFP4 and NVFP4 MoE backend with persistent CuTe GEMMs.

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const Options = @import("../options.zig").Options;
const RoutingWeightPlacement = @import("../options.zig").RoutingWeightPlacement;
pub const kernels = @import("sm10x/moe.zig");

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

/// The NVFP4 kernels are only validated on SM103.
pub fn isNvfp4Supported(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10 and cc.minor == 3;
}

pub fn validateOptions(opts: Options) void {
    stdx.debug.assert(opts.activation == .swiglu, "cute backend only accepts swiglu activation, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.bias == null, "cute backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "cute backend requires swiglu scale to be null", .{});
}

/// MXFP4 weights are `u8` (or `i8`) packed FP4 with E8M0 block scales. NVFP4
/// weights are packed `u8` with E4M3 block scales; their weight global scale
/// is the per-expert multiplier dequantizing each GEMM output (it folds in
/// the inverse of the input's global scale), and their input scale is the
/// multiplier applied to the GEMM's input before NVFP4 quantization. Both
/// expect block scales in the 128x4 tensor-core layout.
pub fn fusedExperts(
    input: zml.Tensor,
    routing: zml.moe.LocalRouting,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: Options,
) zml.Tensor {
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "cute backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "cute backend expects gate_up bias and down bias to be null", .{});
    const gq = gate_up.quantization orelse @panic("cute backend requires gate_up quantization to be set");
    const dq = down.quantization orelse @panic("cute backend requires down quantization to be set");
    stdx.debug.assert(gq.scheme == dq.scheme, "cute expects gate_up and down quantization scheme to match, got {} and {}", .{ gq.scheme, dq.scheme });

    const format: Format = switch (gq.scheme) {
        .mxfp4 => blk: {
            // Weight storage for mxfp4 in HF is expressed as u8 or i8
            stdx.debug.assert(gate_up.weight.dtype() == .u8 or gate_up.weight.dtype() == .i8, "cute expects gate_up weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});
            stdx.debug.assert(down.weight.dtype() == .u8 or down.weight.dtype() == .i8, "cute expects down weight dtype to be u8 or i8, got {}", .{down.weight.dtype()});
            // Check the boundary quantizer and CuTe GEMM tile constraints.
            kernels.validateMxfp4Shapes(down.weight.dim(1), down.weight.dim(2) * 2);
            // The SwiGLU clamp is a kernel parameter, but the routing weight is
            // multiplied in the up epilogue, before the down projection: applying it
            // after would have to move into the down epilogue or the reduction.
            stdx.debug.assert(options.routing_weight_placement == .before_down, "cute MXFP4 backend only supports routing_weight_placement = .before_down, got {}", .{options.routing_weight_placement});
            break :blk .mxfp4;
        },
        .nvfp4 => blk: {
            stdx.debug.assert(!options.quantize_input, "cute NVFP4 backend always quantizes its inputs to NVFP4", .{});
            stdx.debug.assert(gq.swizzled_scales and dq.swizzled_scales, "cute NVFP4 backend expects swizzled block scales", .{});
            stdx.debug.assert(gate_up.weight.dtype() == .u8 and down.weight.dtype() == .u8, "cute NVFP4 backend expects packed u8 weights, got {} and {}", .{ gate_up.weight.dtype(), down.weight.dtype() });
            const scale = struct {
                fn f(global: ?zml.Quantization.GlobalScale) zml.Tensor {
                    return (global orelse @panic("cute NVFP4 backend requires global and input scales")).asMultiplier();
                }
            }.f;
            break :blk .{ .nvfp4 = .{
                .alpha1 = scale(gq.global_scale),
                .alpha2 = scale(dq.global_scale),
                .act1 = scale(gq.input_scale),
                .act2 = scale(dq.input_scale),
            } };
        },
        else => stdx.debug.panic("cute backend only supports mxfp4 and nvfp4 quantization, got {}", .{gq.scheme}),
    };

    const hidden = down.weight.dim(1);
    const intermediate = down.weight.dim(2) * 2;
    const topk = routing.ids.dim(.topk);
    const experts = gate_up.weight.dim(.expert);
    const ids = routing.localIds().convert(.i32).reshape(.{ .token = .auto, .topk = topk });
    const tokens = ids.dim(.token);
    const routing_weights = routing.weights.convert(.f32).reshape(.{ tokens, topk });

    return switch (format) {
        .mxfp4 => kernels.forwardMxfp4(tokens, hidden, intermediate, experts, topk, options.activation.swiglu.limit, .{
            .x = input.reshape(.{ tokens, hidden }),
            .routing_weights = routing_weights,
            .w1 = gate_up.weight,
            .s1 = gq.scales,
            .w2 = down.weight,
            .s2 = dq.scales,
            .ids = ids,
        }, null),
        .nvfp4 => |f| kernels.forwardNvfp4(tokens, hidden, intermediate, experts, topk, .{
            .x = input.reshape(.{ tokens, hidden }),
            .routing_weights = routing_weights,
            .ids = ids,
            .w1 = gate_up.weight,
            .s1 = gq.scales,
            .w2 = down.weight,
            .s2 = dq.scales,
            // `alpha` is per expert; `act` is one scale shared by every
            // expert, which the kernels read per expert.
            .alpha1 = f.alpha1.reshape(.{experts}),
            .alpha2 = f.alpha2.reshape(.{experts}),
            .act1 = f.act1.reshape(.{}).broad(.init(.{experts}, .f32)),
            .act2 = f.act2.reshape(.{}).broad(.init(.{experts}, .f32)),
        }, .{
            .swiglu_limit = options.activation.swiglu.limit,
            .routing_weight_placement = options.routing_weight_placement,
        }),
    }.reshape(input.shape().dims()).withTags(input.shape());
}

/// The format-specific tensors: NVFP4's per-expert global scales. They stay
/// as given until the sharded body, so they split across ranks like the
/// weights.
const Format = union(enum) {
    mxfp4,
    nvfp4: struct {
        alpha1: zml.Tensor,
        alpha2: zml.Tensor,
        act1: zml.Tensor,
        act2: zml.Tensor,
    },
};
