const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;
const manual = @import("../../ops/manual_computation2.zig");
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
/// `partitioning` declares `mesh`, `in_specs` and `out_specs`.
/// Input specs cover input, ids, weights, gate_up and
/// down; projection specs describe the weight tensor. Bias and quantization
/// scales follow their projection's expert partitioning automatically.
/// A declaration supports token and expert parallelism: token specs must agree,
/// expert weights may split on the expert dimension, and feature/top-k dimensions
/// are replicated. Tokens must be replicated along the expert axis.
/// Every backend runs in a manual region with local experts.
/// The shared operator routes tokens to those experts and sums their contributions.
pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    backend: Backend,
    opts: Options,
    partitioning: anytype,
) zml.Tensor {
    validateRoutingInputs(input, topk_ids, topk_weights);

    const compiler = zml.Compiler.current();
    const mesh: *const zml.Mesh = if (@TypeOf(partitioning.mesh) == @EnumLiteral()) compiler.mesh(partitioning.mesh) else partitioning.mesh;
    const shapes = manual.applySpecs(zml.Shape, compiler.arena.allocator(), mesh, .{
        .input = input.shape(),
        .ids = topk_ids.shape(),
        .weights = topk_weights.shape(),
        .gate_up = gate_up.weight.shape(),
        .down = down.weight.shape(),
    }, partitioning.in_specs) catch |err| stdx.debug.panic("Invalid MoE input partitioning: {s}", .{@errorName(err)});
    const output_shape = manual.applySpecs(zml.Shape, compiler.arena.allocator(), mesh, input.shape(), partitioning.out_specs) catch |err|
        stdx.debug.panic("Invalid MoE output partitioning: {s}", .{@errorName(err)});

    const tokens = shapes.input.partition(.t);
    const experts = shapes.gate_up.partition(.expert);
    stdx.debug.assert(shapes.input.partition(.d) == .replicated and shapes.ids.partition(.topk) == .replicated and shapes.weights.partition(.topk) == .replicated, "MoE feature and top-k dimensions must be replicated", .{});
    stdx.debug.assert(shapes.ids.partition(.t) == tokens and shapes.weights.partition(.t) == tokens and output_shape.partition(.t) == tokens and output_shape.partition(.d) == .replicated, "MoE input, routing and output must have matching token partitioning", .{});
    stdx.debug.assert(shapes.down.partition(.expert) == experts and (!experts.isSharded() or experts != tokens), "MoE projections must agree on expert partitioning; tokens must be replicated along the expert axis", .{});
    for ([_]zml.Shape{ shapes.gate_up, shapes.down }) |shape| {
        for (0..shape.rank()) |axis| {
            stdx.debug.assert(axis == shape.axis(.expert) or shape.partition(axis) == .replicated, "MoE kernel weights can only be partitioned along experts", .{});
        }
    }
    const expert_axis = experts.meshAxis();
    const parallel = if (expert_axis) |axis| mesh.grid.axis_sizes[axis] > 1 else false;
    if (backend == .metal or backend == .tcl) stdx.debug.assert(!parallel, "This MoE backend does not support expert parallelism", .{});

    const Call = struct {
        input: zml.Tensor,
        ids: zml.Tensor,
        weights: zml.Tensor,
        gate_up: zml.nn.Linear,
        down: zml.nn.Linear,
        backend: Backend,
        opts: Options,
        mesh: *const zml.Mesh,
        expert_axis: ?u3,
        global_experts: i64,

        fn forward(args: @This(), _: zml.Shape) zml.Tensor {
            const routing: LocalRouting = .{
                .ids = args.ids,
                .weights = args.weights,
                .global_num_experts = args.global_experts,
                .local_num_experts = args.gate_up.weight.dim(.expert),
                .expert_start = if (args.expert_axis) |axis| zml.ops.axisIndex(args.mesh, args.mesh.logical.axes.get(axis)).convert(.i32).scale(args.gate_up.weight.dim(.expert)) else null,
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
            return if (args.expert_axis) |axis| zml.ops.allReduceOnAxis(output, zml.Tensor.add, args.mesh, args.mesh.logical.axes.get(axis)) else output;
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
        .mesh = mesh,
        .expert_axis = if (parallel) expert_axis else null,
        .global_experts = gate_up.weight.dim(.expert),
    };
    const in_specs = .{
        .input = shapes.input._sharding.partition,
        .ids = shapes.ids._sharding.partition,
        .weights = shapes.weights._sharding.partition,
        .gate_up = linearSpecs(gate_up, shapes.gate_up._sharding.partition),
        .down = linearSpecs(down, shapes.down._sharding.partition),
    };
    return zml.ops.manualComputation2(Call.forward, args, output_shape, .{
        .mesh = mesh,
        .in_specs = in_specs,
        .out_specs = output_shape._sharding.partition,
    });
}

// Packed scales may be untagged. Their first dimension identifies experts;
// scalar scales apply to every expert and remain replicated.
fn auxiliarySpec(tensor: ?zml.Tensor, experts: i64, spec: zml.Sharding.PartitionSpec) zml.Sharding.Partitioning {
    const t = tensor orelse return .replicated(0);
    const replicated = zml.Sharding.Partitioning.replicated(t.rank());
    if (t.count() == 1 or spec == .replicated) return replicated;
    stdx.debug.assert(t.dim(0) == experts, "MoE bias/scale must be scalar or contain {} experts, got {f}", .{ experts, t.shape() });
    return replicated.set(0, spec);
}

fn linearSpecs(linear: zml.nn.Linear, partition: zml.Sharding.Partitioning) struct {
    weight: zml.Sharding.Partitioning,
    bias: zml.Sharding.Partitioning,
    quantization: struct {
        scales: zml.Sharding.Partitioning,
        global_scale: struct { value: zml.Sharding.Partitioning },
        input_scale: struct { value: zml.Sharding.Partitioning },
    },
} {
    const experts = linear.weight.dim(.expert);
    const spec = partition.get(linear.weight.axis(.expert));
    const q = linear.quantization;
    return .{
        .weight = partition,
        .bias = auxiliarySpec(linear.bias, experts, spec),
        .quantization = .{
            .scales = auxiliarySpec(if (q) |v| v.scales else null, experts, spec),
            .global_scale = .{ .value = auxiliarySpec(if (q) |v| if (v.global_scale) |scale| scale.value else null else null, experts, spec) },
            .input_scale = .{ .value = auxiliarySpec(if (q) |v| if (v.input_scale) |scale| scale.value else null else null, experts, spec) },
        },
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
