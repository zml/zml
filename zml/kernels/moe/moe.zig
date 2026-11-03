const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;
pub const tcl = @import("tcl/tcl.zig");
pub const stablehlo = @import("stablehlo/stablehlo.zig");
pub const cute = @import("cute/cute.zig");
pub const cutlass_flashinfer = @import("cutlass_flashinfer/cutlass_flashinfer.zig");
pub const metal = @import("metal/metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu/mosaic_tpu.zig");
pub const triton = @import("triton/triton.zig");
pub const fly = @import("fly/fly.zig");

const log = std.log.scoped(.@"zml/moe");

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}

pub const ProjectionLayout = enum { concatenated, interleaved };
pub const RoutingWeightPlacement = enum { before_down, after_down };

/// How a backend expects the expert weights to be stored
/// The packer writes this layout and forwardMoe reads it
pub const ExpertsLayout = struct {
    /// Whether the two projections occupy contiguous halves or alternate.
    gate_up: ProjectionLayout,
    /// Projection order within the concatenated halves or interleaved pairs.
    gate_up_order: enum { gate_up, up_gate } = .gate_up,
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
    cute,
    flashinfer_cutlass,
    triton,
    fly,
    mosaic_tpu,
    metal,
    tcl,

    pub fn auto(platform: *const zml.Platform, scheme: ?zml.Quantization.Scheme, dtype: zml.DataType) !Backend {
        // Keep the dtype as an argument because non scheme-specific backends may depend on it later
        _ = dtype;
        return switch (platform.target) {
            .cuda => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (cute.isAvailable(platform))
                        .cute
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
            .furiosa => if (scheme == null) .tcl else error.UnsupportedQuantization,
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
            .cute => cute.isAvailable(platform),
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
            .tcl => platform.target == .furiosa,
        };
    }

    /// The layout contract the expert weights must have for a given backend with scheme
    pub fn expertsLayout(backend: Backend, scheme: ?zml.Quantization.Scheme) !ExpertsLayout {
        return switch (backend) {
            .stablehlo => .{ .gate_up = .concatenated, .packing = .plain },
            .cute => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .swizzled_scales } else error.UnsupportedQuantization,
            .flashinfer_cutlass => if (scheme) |s| switch (s) {
                .nvfp4 => .{ .gate_up = .concatenated, .packing = .flashinfer_nvfp4 },
                else => error.UnsupportedQuantization,
            } else .{ .gate_up = .concatenated, .gate_up_order = .up_gate, .packing = .plain },
            .mosaic_tpu, .metal, .tcl => .{ .gate_up = .concatenated, .packing = .plain },
            .triton, .fly => if (scheme == .mxfp4) .{ .gate_up = .interleaved, .packing = .plain } else .{ .gate_up = .concatenated, .packing = .plain },
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .stablehlo => {},
            .cute => {},
            .flashinfer_cutlass => cutlass_flashinfer.register(platform),
            .triton, .fly => {},
            .mosaic_tpu => {},
            .metal => {},
            .tcl => {},
        };
    }
};

/// Global routing with the expert range owned by this local computation.
/// Kernels select local ids or a global-to-local map to match their routing algorithm.
/// Kernels return this range's contribution; `forwardMoe` sums contributions.
pub const LocalRouting = struct {
    ids: zml.Tensor,
    weights: zml.Tensor,
    global_num_experts: i64,
    local_num_experts: i64,
    expert_start: ?zml.Tensor = null,

    pub fn localIds(self: LocalRouting) zml.Tensor {
        const start = self.expert_start orelse return self.ids;
        const local = self.ids.cmp(.GE, start).logical(.AND, self.ids.cmp(.LT, start.addConstant(self.local_num_experts)));
        return local.select(self.ids.sub(start), .scalar(-1, .i32));
    }

    pub fn expertMap(self: LocalRouting) ?zml.Tensor {
        const start = self.expert_start orelse return null;
        const ids = zml.Tensor.arange(.{ .end = self.global_num_experts }, .i32).withTags(.{.expert});
        const local = ids.cmp(.GE, start).logical(.AND, ids.cmp(.LT, start.addConstant(self.local_num_experts)));
        return local.select(ids.sub(start), .scalar(-1, .i32));
    }
};

pub const Options = struct {
    activation: Activation,
    /// Quantize activations for Triton FP8 GEMMs; false keeps BF16 activations.
    quantize_input: bool,
    /// Where routing weights are applied; FlashInfer, Mosaic, Metal and TCL require after_down.
    routing_weight_placement: RoutingWeightPlacement,
};

/// Check the shared token-wise contract for activations and routing tensors.
pub fn validateRoutingInputs(input: zml.Tensor, ids: zml.Tensor, weights: zml.Tensor) void {
    stdx.debug.assert(input.rank() == 2 and ids.rank() == 2 and weights.rank() == 2, "MoE expects input {{.t, .d}} and routing {{.t, .topk}}, got {f}, {f}, {f}", .{ input.shape(), ids.shape(), weights.shape() });
    stdx.debug.assert(input.dim(.t) == ids.dim(.t) and input.dim(.t) == weights.dim(.t), "MoE input and routing token counts must match", .{});
    stdx.debug.assert(ids.dim(.topk) == weights.dim(.topk), "MoE routing ids and weights must have matching top-k dimensions", .{});
    stdx.debug.assert(input.shape().hasTag(.t) == 0 and input.shape().hasTag(.d) == 1 and
        ids.shape().hasTag(.t) == 0 and ids.shape().hasTag(.topk) == 1 and
        weights.shape().hasTag(.t) == 0 and weights.shape().hasTag(.topk) == 1, "MoE expects ordered tags {{.t, .d}} for input and {{.t, .topk}} for routing", .{});
}

/// Input has shape {.t, .d}; routing IDs and weights have shape {.t, .topk}.
/// Returns {.t, .d}. Flatten batched input with merge(.{ .t = .{ .b, .s } }).
/// Every backend runs in a manual computation; the shared operator routes to
/// local experts and sums their contributions.
pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    backend: Backend,
    opts: Options,
) zml.Tensor {
    validateRoutingInputs(input, topk_ids, topk_weights);
    const expert_parallel = gate_up.weight.shape().partition(.expert).isSharded();
    if (backend == .metal or backend == .tcl) stdx.debug.assert(!expert_parallel, "This MoE backend does not support expert parallelism", .{});
    const Call = struct {
        input: zml.Tensor,
        ids: zml.Tensor,
        weights: zml.Tensor,
        gate_up: zml.nn.Linear,
        down: zml.nn.Linear,
        backend: Backend,
        opts: Options,
        expert_parallel: bool,
        global_experts: i64,

        fn forward(args: @This(), _: zml.Shape) zml.Tensor {
            const routing: LocalRouting = .{
                .ids = args.ids,
                .weights = args.weights,
                .global_num_experts = args.global_experts,
                .local_num_experts = args.gate_up.weight.dim(.expert),
                .expert_start = if (args.expert_parallel) zml.ops.partitionId().convert(.i32).scale(args.gate_up.weight.dim(.expert)) else null,
            };
            const output = switch (args.backend) {
                .stablehlo => stablehlo.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .cute => cute.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .flashinfer_cutlass => cutlass_flashinfer.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .triton => triton.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .fly => fly.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .mosaic_tpu => mosaic_tpu.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .metal => metal.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .tcl => tcl.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
            };
            return if (args.expert_parallel) zml.ops.allReduce(output, zml.Tensor.add) else output;
        }
    };
    const args: Call = .{
        .input = input,
        .ids = topk_ids,
        .weights = topk_weights,
        .gate_up = gate_up,
        .down = down,
        .backend = backend,
        .opts = opts,
        .expert_parallel = expert_parallel,
        .global_experts = gate_up.weight.dim(.expert),
    };
    const mesh = gate_up.weight.shape()._sharding.mesh orelse zml.Compiler.current().platform.replicated_mesh;
    return zml.ops.manualComputation(Call.forward, args, input.shape().withPartitioning(mesh, .{}));
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
