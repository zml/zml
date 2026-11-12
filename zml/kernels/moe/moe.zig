const std = @import("std");

const platforms = @import("platforms");

const zml = @import("../../zml.zig");
const stdx = zml.stdx;
pub const cute = @import("cute/cute.zig");
pub const ExpertsLayout = @import("options.zig").ExpertsLayout;
pub const flashinfer_cutlass = @import("flashinfer_cutlass/flashinfer_cutlass.zig");
pub const fly = @import("fly/fly.zig");
pub const metal = @import("metal/metal.zig");
pub const mosaic_tpu = @import("mosaic_tpu/mosaic_tpu.zig");
pub const Options = @import("options.zig").Options;
pub const Activation = @import("options.zig").Activation;
pub const ActivationKind = @import("options.zig").ActivationKind;
pub const ProjectionLayout = @import("options.zig").ProjectionLayout;
pub const RoutingWeightPlacement = @import("options.zig").RoutingWeightPlacement;
pub const stablehlo = @import("stablehlo/stablehlo.zig");
pub const tcl = @import("tcl/tcl.zig");
pub const triton = @import("triton/triton.zig");

const log = std.log.scoped(.@"zml/moe");

test {
    std.testing.refAllDecls(@This());
    _ = @import("tests.zig");
}

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
        return switch (platform.target) {
            .cuda => b: {
                const s = scheme orelse break :b .triton;
                break :b switch (s) {
                    .mxfp4 => if (cute.isAvailable(platform))
                        .cute
                    else
                        .triton,
                    .nvfp4 => if (cute.isNvfp4Supported(platform))
                        .cute
                    else if (flashinfer_cutlass.isNvfp4Supported(platform))
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
            .furiosa => b: {
                const s = scheme orelse break :b .tcl;
                break :b switch (s) {
                    .fp8_block128 => if (dtype == .f8e4m3fn) .tcl else error.UnsupportedQuantization,
                    .mxfp4, .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor, .fp8_block32 => error.UnsupportedQuantization,
                };
            },
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
            .flashinfer_cutlass => flashinfer_cutlass.isAvailable(platform),
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
            .cute => if (scheme) |s| switch (s) {
                .mxfp4 => .{ .gate_up = .row_interleaved_8, .packing = .mxfp4_interleaved8 },
                .nvfp4 => .{ .gate_up = .row_interleaved_1, .packing = .swizzled_scales },
                else => error.UnsupportedQuantization,
            } else error.UnsupportedQuantization,
            .flashinfer_cutlass => if (scheme) |s| switch (s) {
                .nvfp4 => .{ .gate_up = .concatenated, .packing = .flashinfer_nvfp4 },
                else => error.UnsupportedQuantization,
            } else error.UnsupportedQuantization,
            .mosaic_tpu, .metal, .tcl => .{ .gate_up = .concatenated, .packing = .plain },
            .triton, .fly => if (scheme == .mxfp4) .{ .gate_up = .row_interleaved_1, .packing = .plain } else .{ .gate_up = .concatenated, .packing = .plain },
        };
    }

    pub fn register(backend: Backend, platform: *zml.Platform) !void {
        return switch (backend) {
            .stablehlo => {},
            .cute => {},
            .flashinfer_cutlass => flashinfer_cutlass.register(platform),
            .triton, .fly => {},
            .mosaic_tpu => {},
            .metal => {},
            .tcl => {},
        };
    }
};

/// Routing for a local token batch and a contiguous range of experts.
/// IDs retain the model's global expert numbering, while the projection weights
/// contain only the local experts. Kernels compute those experts contributions and
/// `forwardMoe` sums them along the expert axis.
pub const LocalRouting = struct {
    /// Global expert IDs for local tokens, with shape {.t, .topk}
    ids: zml.Tensor,
    weights: zml.Tensor,
    global_num_experts: u32,
    local_num_experts: u32,
    /// Scalar i32 containing the first local expert's global ID.
    /// Null when all experts are local and global IDs already index the weights.
    expert_start: ?zml.Tensor = null,

    /// Maps routing IDs to local weight indices, using -1 for nonlocal experts.
    /// For a shard owning experts 4–7, routing IDs [1, 6] become [-1, 2].
    pub fn localIds(self: LocalRouting) zml.Tensor {
        const start = self.expert_start orelse return self.ids;
        const local = self.ids.cmp(.GE, start).logical(.AND, self.ids.cmp(.LT, start.addConstant(self.local_num_experts)));
        return local.select(self.ids.sub(start), .scalar(-1, .i32));
    }

    /// Returns a lookup table from every global expert ID to its local weight
    /// index or -1 if nonlocal. Returns null when every expert is local.
    pub fn expertMap(self: LocalRouting) ?zml.Tensor {
        const start = self.expert_start orelse return null;
        const ids = zml.Tensor.arange(.{ .end = self.global_num_experts }, .i32).withTags(.{.expert});
        const local = ids.cmp(.GE, start).logical(.AND, ids.cmp(.LT, start.addConstant(self.local_num_experts)));
        return local.select(ids.sub(start), .scalar(-1, .i32));
    }
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

pub const Partitioning = struct {
    const Logical = zml.Sharding.Partitioning.Logical;

    in_specs: struct {
        input: Logical,
        ids: Logical,
        weights: Logical,
        gate_up: Logical,
        down: Logical,
    },
    out_specs: Logical,

    /// Shards the `.expert` dimension of both projections over `mesh_axis`.
    /// Other dimensions, input tokens, routing IDs/weights and output are replicated.
    pub fn expert(comptime mesh_axis: @EnumLiteral()) Partitioning {
        const experts: Logical = .parse(.{ .expert = mesh_axis });
        return .{
            .in_specs = .{
                .input = .replicated,
                .ids = .replicated,
                .weights = .replicated,
                .gate_up = experts,
                .down = experts,
            },
            .out_specs = .replicated,
        };
    }
};

/// Input has shape {.t, .d}; routing IDs and weights have shape {.t, .topk}.
/// Returns {.t, .d}. Flatten batched input with merge(.{ .t = .{ .b, .s } }).
pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    backend: Backend,
    mesh: *const zml.Mesh,
    partitioning: Partitioning,
    opts: Options,
) zml.Tensor {
    validateRoutingInputs(input, topk_ids, topk_weights);

    const shapes = .{
        .input = input.shape().withPartitioning(mesh, partitioning.in_specs.input),
        .ids = topk_ids.shape().withPartitioning(mesh, partitioning.in_specs.ids),
        .weights = topk_weights.shape().withPartitioning(mesh, partitioning.in_specs.weights),
        .gate_up = gate_up.weight.shape().withPartitioning(mesh, partitioning.in_specs.gate_up),
        .down = down.weight.shape().withPartitioning(mesh, partitioning.in_specs.down),
    };
    const output_shape = input.shape().withPartitioning(mesh, partitioning.out_specs);

    const tokens = shapes.input.partition(.t);
    const experts = shapes.gate_up.partition(.expert);
    // Local kernels require complete feature vectors and top-k routing lists.
    stdx.debug.assert(shapes.input.partition(.d) == .replicated and shapes.ids.partition(.topk) == .replicated and shapes.weights.partition(.topk) == .replicated, "MoE feature and top-k dimensions must be replicated", .{});
    // The expert-axis reduction must add contributions for the same token rows.
    // Sharding tokens along that axis would mix contributions for different tokens.
    stdx.debug.assert(shapes.ids.partition(.t) == tokens and shapes.weights.partition(.t) == tokens and output_shape.partition(.t) == tokens and output_shape.partition(.d) == .replicated, "MoE input, routing and output must have matching token partitioning", .{});
    stdx.debug.assert(shapes.down.partition(.expert) == experts and (!experts.isSharded() or experts != tokens), "MoE projections must agree on expert partitioning; tokens must be replicated along the expert axis", .{});
    for ([_]zml.Shape{ shapes.gate_up, shapes.down }) |shape| {
        stdx.debug.assert(!shape.containsPartitionSpec(.open), "MoE kernel weight partitioning must be closed", .{});
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
        global_experts: u32,

        fn forward(args: @This(), _: zml.Shape) zml.Tensor {
            const routing: LocalRouting = .{
                .ids = args.ids,
                .weights = args.weights,
                .global_num_experts = args.global_experts,
                .local_num_experts = @intCast(args.gate_up.weight.dim(.expert)),
                .expert_start = if (args.expert_axis) |axis| zml.ops.axisIndex(args.mesh, args.mesh.logical.axes.get(axis)).convert(.i32).scale(args.gate_up.weight.dim(.expert)) else null,
            };
            const output = switch (args.backend) {
                .stablehlo => stablehlo.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .cute => cute.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
                .flashinfer_cutlass => flashinfer_cutlass.fusedExperts(args.input, routing, args.gate_up, args.down, args.opts),
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
        .global_experts = @intCast(gate_up.weight.dim(.expert)),
    };
    const Logical = zml.Sharding.Partitioning.Logical;
    var projection_specs: [2]zml.meta.MapRestrict(zml.Tensor, Logical).map(zml.nn.Linear) = undefined;
    for ([_]zml.nn.Linear{ gate_up, down }, &projection_specs) |linear, *specs| {
        zml.meta.mapAlloc(struct {
            fn spec(ctx: struct { experts: i64, axis: ?zml.Shape.Tag }, tensor: zml.Tensor) Logical {
                const axis = ctx.axis orelse return .replicated;
                if (tensor.count() == 1) return .replicated;
                // Packed scales may be untagged; experts are their leading dimension.
                const dim = tensor.shape().hasTag(.expert) orelse 0;
                stdx.debug.assert(tensor.dim(dim) == ctx.experts, "MoE tensor must contain {} experts, got {f}", .{ ctx.experts, tensor.shape() });
                return .{ .mapping = .{
                    .src = .init(&.{if (tensor.shape().hasTag(.expert) != null) zml.Shape.toTag(.expert) else zml.Shape.toTag(._0)}),
                    .tgt = .init(&.{axis}),
                } };
            }
        }.spec, zml.Compiler.current().arena.allocator(), .{
            .experts = linear.weight.dim(.expert),
            .axis = if (expert_axis) |axis| mesh.logical.axes.get(axis) else null,
        }, linear, specs) catch @panic("OOM");
    }
    return zml.ops.manualComputation2(Call.forward, args, output_shape, mesh, .{
        .in_specs = .{
            .input = partitioning.in_specs.input,
            .ids = partitioning.in_specs.ids,
            .weights = partitioning.in_specs.weights,
            .gate_up = projection_specs[0],
            .down = projection_specs[1],
        },
        .out_specs = partitioning.out_specs,
    });
}
