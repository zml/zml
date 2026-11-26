//! Blackwell MXFP4 and NVFP4 MoE backend with persistent CuTe GEMMs.

const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
const nn = @import("zml/nn");
const Options = @import("zml/kernels/moe/api").Options;
const Quantization = @import("zml/quantization").Quantization;
const RoutingWeightPlacement = @import("zml/kernels/moe/api").RoutingWeightPlacement;
const stdx = @import("stdx");
const zml = @import("zml/core");

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
    switch (opts.activation) {
        .swiglu => |swiglu| {
            stdx.debug.assert(swiglu.bias == null, "cute backend requires swiglu bias to be null", .{});
            stdx.debug.assert(swiglu.scale == null, "cute backend requires swiglu scale to be null", .{});
            stdx.debug.assert(opts.nvfp4_input_scale == .largest_maps_to_six, "cute backend scales calibrated NVFP4 activation blocks by amax/6", .{});
        },
        // NVFP4 only. Activations take calibrated scales (amax/6 blocks) or are quantized row by row.
        .capped_swiglu => stdx.debug.assert(opts.quantize_input or opts.nvfp4_input_scale == .largest_maps_to_six, "cute backend scales calibrated NVFP4 activation blocks by amax/6", .{}),
        else => stdx.debug.panic("cute backend only accepts swiglu and capped swiglu activations, got {}", .{opts.activation}),
    }
}

/// MXFP4 weights are `u8` (or `i8`) packed FP4 with E8M0 block scales. NVFP4
/// weights are packed `u8` with E4M3 block scales; their weight global scale
/// is the per-expert multiplier dequantizing each GEMM output (it folds in
/// the inverse of the input's global scale), and their input scale is the
/// multiplier applied to the GEMM's input before NVFP4 quantization. Both
/// expect block scales in the 128x4 tensor-core layout. MXFP4 gate/up rows
/// interleave by 8 and its scale atoms are MN-major packed i32 words.
pub fn fusedExperts(
    input: zml.Tensor,
    routing: LocalRouting,
    gate_up: nn.Linear,
    down: nn.Linear,
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
            stdx.debug.assert(options.activation == .swiglu, "cute MXFP4 backend only accepts swiglu activation, got {}", .{options.activation});
            kernels.validateMxfp4Shapes(down.weight.dim(1), down.weight.dim(2) * 2);
            // The SwiGLU clamp is a kernel parameter, but the routing weight is
            // multiplied in the up epilogue, before the down projection: applying it
            // after would have to move into the down epilogue or the reduction.
            stdx.debug.assert(options.routing_weight_placement == .before_down, "cute MXFP4 backend only supports routing_weight_placement = .before_down, got {}", .{options.routing_weight_placement});
            break :blk .mxfp4;
        },
        .nvfp4 => blk: {
            stdx.debug.assert(gq.swizzled_scales and dq.swizzled_scales, "cute NVFP4 backend expects swizzled block scales", .{});
            stdx.debug.assert(gate_up.weight.dtype() == .u8 and down.weight.dtype() == .u8, "cute NVFP4 backend expects packed u8 weights, got {} and {}", .{ gate_up.weight.dtype(), down.weight.dtype() });
            const scale = struct {
                fn f(global: ?Quantization.GlobalScale) zml.Tensor {
                    return (global orelse @panic("cute NVFP4 backend requires global and input scales")).asMultiplier();
                }
            }.f;
            // Activations are quantized row by row when asked to; the global scales then only
            // dequantize the weights. Otherwise they take the calibrated input scales.
            if (options.quantize_input) {
                stdx.debug.assert(options.activation == .capped_swiglu, "cute NVFP4 backend quantizes activations row by row around a capped swiglu, got {}", .{options.activation});
                stdx.debug.assert(gq.input_scale == null and dq.input_scale == null, "cute NVFP4 backend quantizes activations row by row without calibrated input scales", .{});
                break :blk .{ .nvfp4 = .{ .alpha1 = scale(gq.global_scale), .alpha2 = scale(dq.global_scale), .act = null } };
            }
            break :blk .{ .nvfp4 = .{
                .alpha1 = scale(gq.global_scale),
                .alpha2 = scale(dq.global_scale),
                .act = .{ scale(gq.input_scale), scale(dq.input_scale) },
            } };
        },
        else => stdx.debug.panic("cute backend only supports mxfp4 and nvfp4 quantization, got {}", .{gq.scheme}),
    };

    const hidden = down.weight.dim(1);
    // The gate and up rows count the intermediate values; the down projection may pad them.
    const intermediate = @divExact(gate_up.weight.dim(1), 2);
    const topk = routing.ids.dim(.topk);
    const experts = gate_up.weight.dim(.expert);
    const ids = routing.localIds().convert(.i32).reshape(.{ .token = .auto, .topk = topk });
    const tokens = ids.dim(.token);
    const routing_weights = routing.weights.convert(.f32).reshape(.{ tokens, topk });

    return switch (format) {
        .mxfp4 => kernels.forwardMxfp4(tokens, hidden, down.weight.dim(2) * 2, experts, topk, options.activation.swiglu.limit, .{
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
            .act1 = if (f.act) |act| act[0].reshape(.{}).broad(.init(.{experts}, .f32)) else f.alpha1.reshape(.{experts}),
            .act2 = if (f.act) |act| act[1].reshape(.{}).broad(.init(.{experts}, .f32)) else f.alpha2.reshape(.{experts}),
        }, switch (options.activation) {
            .swiglu => |swiglu| .{ .swiglu_limit = swiglu.limit, .routing_weight_placement = options.routing_weight_placement },
            .capped_swiglu => |caps| .{
                .routing_weight_placement = options.routing_weight_placement,
                .capped = .{ .gate_cap = caps.gate_cap, .up_cap = caps.up_cap },
                .row_scaled = if (f.act == null) .{ .strategy = options.nvfp4_input_scale, .activation = .{ .gate_cap = caps.gate_cap, .up_cap = caps.up_cap } } else null,
            },
            else => unreachable,
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
        /// The calibrated input scales of the two GEMMs; null quantizes activations row by row.
        act: ?[2]zml.Tensor,
    },
};
