const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../zml.zig");
const stdx = zml.stdx;
pub const stablehlo = @import("stablehlo.zig");
pub const triton_mxfp4 = @import("triton_mxfp4.zig");
pub const cute_mxfp4 = @import("cute_mxfp4.zig");
pub const cutlass_flashinfer = @import("cutlass_flashinfer.zig");
pub const metal = @import("metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu.zig");
pub const triton = @import("triton.zig");
pub const fly = @import("fly_kernels/moe.zig");
const fused_experts = @import("fused_experts.zig");
pub const triton_kernels = @import("triton_kernels/triton_kernels.zig");
pub const ProjectionLayout = fused_experts.ProjectionLayout;

const log = std.log.scoped(.@"zml/moe");

test {
    std.testing.refAllDecls(@This());
}

/// How a backend expects the expert weights to be stored
/// The packer writes this layout and forwardMoe reads it
pub const ExpertsLayout = struct {
    /// Column order of the fused gate/up projection.
    gate_up: ProjectionLayout,
    /// Backend specific transformation of the weights and scales.
    packing: Packing,

    pub const Packing = enum {
        /// Experts stacked as stored in the checkpoint.
        plain,
        /// Block scales swizzled to the 128x4 tensor-core layout (rows tiled by 4x32, columns by 4).
        swizzled_scales,
        /// Weights, block scales and global scales in the FlashInfer CUTLASS NVFP4 layout.
        flashinfer_nvfp4,
    };
};

pub const ActivationKind = enum {
    /// Gelu activation function
    gelu,
    /// ReLU activation function
    relu,
    /// SiLU activation function
    silu,
    /// SwiGLU activation function (including clamped/clipped/scaled/biased variants)
    swiglu,
    /// SwiGLU activation function with step function
    swiglu_step,
    /// GeGlu activation function
    geglu,
    /// GeGlu activation function with tanh
    geglu_tanh,
};

pub const Activation = union(ActivationKind) {
    gelu: void,
    relu: void,
    silu: void,
    swiglu: struct {
        limit: ?f32 = null,
        scale: ?f32 = null,
        bias: ?f32 = null,
    },
    swiglu_step: struct {
        limit: ?f32 = null,
    },
    geglu: void,
    geglu_tanh: void,
};

pub const Backend = enum {
    stablehlo,
    cute_mxfp4,
    triton_mxfp4,
    flashinfer_cutlass,
    triton,
    fly,
    mosaic_tpu,
    metal,

    pub fn auto(platform: *const zml.Platform, scheme: ?zml.Quantization.Scheme, dtype: zml.DataType) !Backend {
        // Keep the dtype as an argument because non scheme-specific backends may depend on it later
        _ = dtype;
        return switch (platform.target) {
            .cuda => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (cute_mxfp4.isAvailable(platform))
                        .cute_mxfp4
                    else if (triton_mxfp4.isAvailable(platform))
                        .triton_mxfp4
                    else
                        .triton,
                    .nvfp4 => if (cutlass_flashinfer.isNvfp4Supported(platform))
                        .flashinfer_cutlass
                    else
                        error.UnsupportedQuantization,
                    .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .triton,
                };
            },
            .rocm => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (zml.platform.rocm.computeCapability(platform) == .gfx942) .fly else .triton,
                    .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .triton,
                    .nvfp4 => error.UnsupportedQuantization,
                };
            },
            .oneapi => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => .triton,
                    .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => error.UnsupportedQuantization,
                };
            },
            .tpu => if (scheme == null) .mosaic_tpu else error.UnsupportedQuantization,
            .metal => b: {
                const s = scheme orelse break :b .metal;
                break :b switch (s) {
                    .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .metal,
                    .mxfp4 => error.UnsupportedQuantization,
                };
            },
            else => error.UnimplementedMoEBackend,
        };
    }

    pub fn isAvailable(backend: Backend, platform: *const zml.Platform) bool {
        return switch (backend) {
            .stablehlo => true,
            .triton_mxfp4 => triton_mxfp4.isAvailable(platform),
            .cute_mxfp4 => cute_mxfp4.isAvailable(platform),
            .flashinfer_cutlass => cutlass_flashinfer.isAvailable(platform),
            .fly => switch (platform.target) {
                .rocm => zml.platform.rocm.computeCapability(platform) == .gfx942,
                else => false,
            },
            .triton => switch (platform.target) {
                .cuda, .rocm, .oneapi => true,
                else => false,
            },
            .mosaic_tpu => platform.target == .tpu,
            .metal => platform.target == .metal,
        };
    }

    /// The layout contract the expert weights must have for a given backend with scheme
    pub fn expertsLayout(backend: Backend, scheme: ?zml.Quantization.Scheme) !ExpertsLayout {
        return switch (backend) {
            .stablehlo => .{ .gate_up = .concatenated, .packing = .plain },
            .cute_mxfp4 => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .swizzled_scales } else error.UnsupportedQuantization,
            .triton_mxfp4 => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .plain } else error.UnsupportedQuantization,
            .flashinfer_cutlass => if (scheme == .nvfp4) .{ .gate_up = .concatenated, .packing = .flashinfer_nvfp4 } else error.UnsupportedQuantization,
            .mosaic_tpu, .metal => .{ .gate_up = .concatenated, .packing = .plain },
            .triton, .fly => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .plain } else .{ .gate_up = .concatenated, .packing = .plain },
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .stablehlo => {},
            .cute_mxfp4, .triton_mxfp4 => {},
            .flashinfer_cutlass => cutlass_flashinfer.register(platform),
            .triton, .fly => {},
            .mosaic_tpu => {},
            .metal => {},
        };
    }
};

pub const Options = struct {
    activation: Activation,
    /// Quantize activations for Triton FP8 GEMMs; false keeps BF16 activations.
    quantize_input: bool,
    /// Where routing weights are applied; FlashInfer, Mosaic and Metal require after_down.
    routing_weight_placement: fused_experts.RoutingWeightPlacement,
};

/// Routing IDs and weights have shape { b, s, topk }; input has shape { b, s, d }.
pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    backend: Backend,
    opts: Options,
) zml.Tensor {
    return switch (backend) {
        .stablehlo => stablehlo.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .cute_mxfp4 => cute_mxfp4.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .triton_mxfp4 => triton_mxfp4.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .flashinfer_cutlass => cutlass_flashinfer.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        inline .triton, .fly => |b| fused_experts.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts, b),
        .mosaic_tpu => mosaic_tpu.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
        .metal => metal.fusedExperts(input, topk_ids, topk_weights, gate_up, down, opts),
    };
}

pub fn unpackedWeight(linear: zml.nn.Linear) zml.Tensor {
    const quantization = linear.quantization orelse return linear.weight;
    return if (zml.nn.isPackedFp4(quantization.scheme, linear.weight.dtype()))
        zml.nn.unpackFp4(linear.weight, linear.tag, linear.tag)
    else
        linear.weight;
}

pub fn applyActivation(input: zml.Tensor, activation: zml.moe.Activation, layout: ProjectionLayout) zml.Tensor {
    const x = input.convert(.f32);
    return switch (activation) {
        .gelu => x.gelu(),
        .relu => x.relu(),
        .silu => x.silu(),
        .swiglu, .swiglu_step, .geglu, .geglu_tanh => b: {
            const mid = @divFloor(x.dim(.out), 2);
            var gate, var up = switch (layout) {
                .concatenated => .{ x.slice(.out, .{ .end = mid }), x.slice(.out, .{ .start = mid }) },
                .interleaved => .{ x.slice(.out, .{ .start = 0, .step = 2 }), x.slice(.out, .{ .start = 1, .step = 2 }) },
            };

            break :b switch (activation) {
                .swiglu => |parameters| {
                    stdx.debug.assert(parameters.scale == null, "triton and fly moe backend don't support swiglu scale", .{});

                    const limit: ?zml.Tensor = if (parameters.limit) |limit| .scalar(limit, x.dtype()) else null;

                    // Apply limit on gate and clamp up
                    gate = if (limit) |l| gate.minimum(l) else gate;
                    up = if (limit) |l| up.clamp(l.negate(), l) else up;

                    // Apply bias
                    up = if (parameters.bias) |bias| up.addConstant(bias) else up;

                    break :b gate.silu().mul(up);
                },
                .swiglu_step => |parameters| {
                    gate = gate.silu();

                    const limit: ?zml.Tensor = if (parameters.limit) |limit| .scalar(limit, x.dtype()) else null;
                    gate = if (limit) |l| gate.minimum(l) else gate;
                    up = if (limit) |l| up.clamp(l.negate(), l) else up;
                    break :b gate.mul(up);
                },
                .geglu_tanh => gate.gelu().mul(up),
                .geglu => {
                    log.warn("The geglu activation function was requested but we only support the tanh approximation", .{});
                    break :b gate.gelu().mul(up);
                },
                else => unreachable, // already treated by the top-level switch
            };
        },
    };
}
