const std = @import("std");

const stdx = @import("stdx");

const Tensor = @import("../../tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/moe");

pub const Options = struct {
    activation: Activation,
    /// Quantize activations for Triton FP8 GEMMs; false keeps BF16 activations.
    quantize_input: bool,
    /// Where routing weights are applied; FlashInfer, Mosaic, Metal and TCL require after_down.
    routing_weight_placement: RoutingWeightPlacement,
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

    pub fn applyActivation(self: Activation, input: Tensor, layout: ProjectionLayout) Tensor {
        const x = input.convert(.f32);
        return switch (self) {
            .gelu => x.gelu(),
            .relu => x.relu(),
            .silu => x.silu(),
            .swiglu, .swiglu_step, .geglu, .geglu_tanh => b: {
                const mid = @divFloor(x.dim(.out), 2);
                var gate, var up = switch (layout) {
                    .concatenated => .{ x.slice(.out, .{ .end = mid }), x.slice(.out, .{ .start = mid }) },
                    .row_interleaved_1 => .{ x.slice(.out, .{ .start = 0, .step = 2 }), x.slice(.out, .{ .start = 1, .step = 2 }) },
                    .row_interleaved_8 => blk: {
                        const grouped = x.splitAxis(.out, .{ .group = @divExact(mid, 8), .pair = 2, .row = 8 });
                        break :blk .{
                            grouped.slice(.pair, .{ .end = 1 }).merge(.{ .out = .{ .group, .pair, .row } }),
                            grouped.slice(.pair, .{ .start = 1 }).merge(.{ .out = .{ .group, .pair, .row } }),
                        };
                    },
                };

                break :b switch (self) {
                    .swiglu => |parameters| {
                        stdx.debug.assert(parameters.scale == null, "triton and fly moe backend don't support swiglu scale", .{});

                        const limit: ?Tensor = if (parameters.limit) |limit| .scalar(limit, x.dtype()) else null;

                        // Apply limit on gate and clamp up
                        gate = if (limit) |l| gate.minimum(l) else gate;
                        up = if (limit) |l| up.clamp(l.negate(), l) else up;

                        // Apply bias
                        up = if (parameters.bias) |bias| up.addConstant(bias) else up;

                        break :b gate.silu().mul(up);
                    },
                    .swiglu_step => |parameters| {
                        gate = gate.silu();

                        const limit: ?Tensor = if (parameters.limit) |limit| .scalar(limit, x.dtype()) else null;
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
};

pub const RoutingWeightPlacement = enum { before_down, after_down };

pub const ProjectionLayout = enum {
    concatenated, // G0 G1 … Gn U0 U1 … Un
    row_interleaved_1, // G0 U0 G1 U1 …
    row_interleaved_8, // G0…G7 U0…U7 G8…G15 U8…U15 …
};

/// How a backend expects the expert weights to be stored
/// The packer writes this layout and forwardMoe reads it
pub const ExpertsLayout = struct {
    /// Gate/up ordering before applying the backend-specific packing.
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
        /// MXFP4 gate/up rows interleaved by 8. E8M0 scales use 512-byte
        /// 128x4 atoms in MN-major order `[expert, k/128, rows/128, 512]`,
        /// represented as `[expert, k/128, rows]` i32 words of four scales.
        mxfp4_interleaved8,
    };
};
