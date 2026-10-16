const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const DataType = zml.DataType;
const kernels = @import("moe.zig");
const toDType = zml.kernel.triton.from;
const ProjectionLayout = zml.moe.ProjectionLayout;
const RoutingWeightPlacement = zml.moe.RoutingWeightPlacement;

const log = std.log.scoped(.@"zml/moe/triton");

fn numStages(config: LaunchConfig) i32 {
    const platform = zml.Compiler.current().platform;
    if (platform.target != .rocm) return config.num_stages;

    // ROCm buffers num_stages - 1 BF16 operand pairs in LDS. A single stage
    // reuses scratch space between operand layout conversions.
    const bytes_per_stage: i64 = @intCast(2 * config.block_size_k * (config.block_size_m + config.block_size_n));
    var stages = config.num_stages;
    for (platform.devices) |device| {
        const attribute = device.pjrt_desc.attribute(platform.pjrt_api, "shared_memory_per_block_optin");
        const limit = if (attribute) |value| value.int64 else 64 * 1024;
        stages = @intCast(@min(stages, @divTrunc(limit, bytes_per_stage) + 1));
    }
    return stages;
}

test {
    std.testing.refAllDecls(@This());
}

pub fn fusedExperts(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
) zml.Tensor {
    // HACK(Corentin): only rely on the scheme of gate_up for now.

    if (gate_up.quantization) |q| switch (q.scheme) {
        .mxfp4 => if (zml.Compiler.current().platform.target == .cuda) {
            return fusedExpertsMxfp4(input, ids, weights, gate_up, down, options);
        },
        else => {},
    };

    return fusedExpertsBase(input, ids, weights, gate_up, down, options);
}

pub fn fusedExpertsBase(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    opts: zml.moe.Options,
) zml.Tensor {
    const gate_up_scheme: ?zml.Quantization.Scheme = if (gate_up.quantization) |q| q.scheme else null;
    const layout: zml.moe.ExpertsLayout = if (gate_up_scheme == .mxfp4)
        .{ .gate_up = .interleaved, .packing = .plain }
    else
        .{ .gate_up = .concatenated, .packing = .plain };

    const args: FusedExpertsArgs = .{
        .hidden_states = input,
        .gate_up = gate_up,
        .down = down,
        .topk_weights = topk_weights,
        .topk_ids = topk_ids,
        .activation = opts.activation,
        .quantize_input = opts.quantize_input,
        .gate_up_layout = layout.gate_up,
        .routing_weight_placement = opts.routing_weight_placement,
    };
    const expert_partition = gate_up.weight.shape().partition(.expert);

    if (!expert_partition.isSharded()) {
        return fusedExpertsImpl(args);
    }

    return zml.ops.manualComputation(
        (struct {
            args: FusedExpertsArgs,
            global_num_experts: i64,

            fn call(self: @This(), _: zml.Shape) zml.Tensor {
                const local_args = self.args;
                const local_num_experts = local_args.gate_up.weight.dim(.expert);
                const partition_id = zml.ops.partitionId().convert(.i32);
                const expert_start = partition_id.scale(local_num_experts).convert(.i32);
                const global_expert_ids = zml.Tensor.arange(.{ .end = self.global_num_experts }, .i32).withTags(.{.expert});

                // Map global expert ids to local ids, or -1 for experts outside this partition.
                const local_expert_mask = global_expert_ids.cmp(.GE, expert_start)
                    .logical(.AND, global_expert_ids.cmp(.LT, expert_start.addConstant(local_num_experts)));
                var mapped_args = local_args;
                mapped_args.expert_map = local_expert_mask.select(
                    global_expert_ids.sub(expert_start),
                    zml.Tensor.scalar(-1, .i32),
                );

                const local_output = fusedExpertsImpl(mapped_args);
                const local_reshaped = local_output.reshape(local_args.hidden_states.shape()).withTags(.{ .t, .d });
                return zml.ops.allReduce(local_reshaped, zml.Tensor.add);
            }
        }).call,
        .{ .args = args, .global_num_experts = gate_up.weight.dim(.expert) },
        input.shape(),
    );
}

pub const FusedExpertsArgs = struct {
    hidden_states: Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    topk_weights: Tensor,
    topk_ids: Tensor,
    activation: zml.moe.Activation,
    expert_map: ?Tensor = null,
    /// Use FP8 activations for FP8 weights; false keeps BF16 activations.
    quantize_input: bool,
    gate_up_layout: ProjectionLayout,
    routing_weight_placement: RoutingWeightPlacement,
};

pub fn fusedExpertsImpl(opts: FusedExpertsArgs) Tensor {
    const hidden_states = opts.hidden_states;
    const topk_weights = opts.topk_weights;
    const topk_ids = opts.topk_ids;
    const num_tokens = hidden_states.dim(.t);
    var launch_config = launchConfigForTokens(num_tokens);

    const gate_up_scheme = opts.gate_up.quantizationScheme();
    const down_scheme = opts.down.quantizationScheme();
    if (gate_up_scheme) |scheme| switch (scheme) {
        .nvfp4 => stdx.debug.panic("triton backend doesn't support gate_up nvfp4 quantization scheme", .{}),
        .fp8_block128 => launch_config.block_size_k = 128,
        .fp8_block32 => launch_config.block_size_k = 32,
        .mxfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    var down_launch_config = launchConfigForTokens(num_tokens);
    if (down_scheme) |scheme| switch (scheme) {
        .nvfp4 => stdx.debug.panic("triton backend doesn't support down nvfp4 quantization scheme", .{}),
        .fp8_block128 => down_launch_config.block_size_k = 128,
        .fp8_block32 => down_launch_config.block_size_k = 32,
        .mxfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = opts.gate_up.weight.withTags(.{ .expert, .out, .in });
    const down = opts.down.weight.withTags(.{ .expert, .out, .mid });
    const routing_weights = topk_weights.reshape(.{ .token = num_tokens, .in = topk_weights.dim(.topk) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = num_tokens, .in = topk_ids.dim(.topk) }).withTags(.{ .token, .topk });

    stdx.debug.assert(hidden.dtype() == .bf16, "triton backend expected BF16 hidden states, got {}", .{hidden.dtype()});
    if (gate_up_scheme == .mxfp4) {
        stdx.debug.assert(gate_up.dtype() == .u8 or gate_up.dtype() == .i8 or gate_up.dtype() == .f4e2m1, "triton backend expected u8, i8 or f4e2m1 gate_up data type when quantization scheme is mxfp4, got {}", .{gate_up.dtype()});
    } else {
        stdx.debug.assert(gate_up.dtype() == .bf16 or gate_up.dtype() == .f8e4m3fn or gate_up.dtype() == .f8e4m3fnuz, "triton backend expected gate_up data type to be bf16, fp8e4m3fn or f8e4m3fnuz, got {}", .{gate_up.dtype()});
    }
    if (down_scheme == .mxfp4) {
        stdx.debug.assert(down.dtype() == .u8 or down.dtype() == .i8 or down.dtype() == .f4e2m1, "triton backend expected u8, i8 or f4e2m1 down data type when quantization scheme is mxfp4, got {}", .{down.dtype()});
    } else {
        stdx.debug.assert(down.dtype() == .bf16 or down.dtype() == .f8e4m3fn or down.dtype() == .f8e4m3fnuz, "triton backend expected down data type to be bf16, fp8e4m3fn or f8e4m3fnuz, got {}", .{down.dtype()});
    }

    stdx.debug.assert(routing_weights.dtype() == .f32 or routing_weights.dtype() == .bf16, "triton backend expected f32 or bf16 routing weights, got {}", .{routing_weights.dtype()});
    stdx.debug.assert(ids.dtype() == .i32, "triton backend expected i32 expert ids, got {}", .{ids.dtype()});
    const gate_up_k = gate_up.dim(.in) * @as(i64, if (gate_up_scheme == .mxfp4 and gate_up.dtype() != .f4e2m1) 2 else 1);
    const down_k = down.dim(.mid) * @as(i64, if (down_scheme == .mxfp4 and down.dtype() != .f4e2m1) 2 else 1);
    stdx.debug.assert(hidden.dim(.in) == gate_up_k, "triton backend expected hidden width to match gate/up input width, got {} and {}", .{ hidden.dim(.in), gate_up_k });
    const activation_reduction: i64 = switch (opts.activation) {
        .gelu, .relu, .silu => 1,
        .swiglu, .swiglu_step, .geglu, .geglu_tanh => 2,
    };
    stdx.debug.assert(@rem(gate_up.dim(.out), activation_reduction) == 0, "triton backend expected gate/up output width to be divisible by {}, got {}", .{ activation_reduction, gate_up.dim(.out) });
    stdx.debug.assert(down_k == @divFloor(gate_up.dim(.out), activation_reduction), "triton backend expected down input width to match activated width, got {} and {}", .{ down_k, @divFloor(gate_up.dim(.out), activation_reduction) });
    stdx.debug.assert(ids.dim(.token) == hidden.dim(.token) and routing_weights.dim(.token) == hidden.dim(.token), "triton backend expected routing ids and weights to match hidden token count, got {}, {} and {}", .{ ids.dim(.token), routing_weights.dim(.token), hidden.dim(.token) });
    stdx.debug.assert(ids.dim(.topk) == routing_weights.dim(.topk), "triton backend expected routing ids and weights to have matching top-k dimensions, got {} and {}", .{ ids.dim(.topk), routing_weights.dim(.topk) });
    stdx.debug.assert(gate_up.dim(.expert) == down.dim(.expert), "triton backend expected gate/up and down expert counts to match, got {} and {}", .{ gate_up.dim(.expert), down.dim(.expert) });
    if (opts.expert_map) |expert_map|
        stdx.debug.assert(expert_map.dtype() == .i32 and expert_map.rank() == 1, "triton backend expected a rank-1 i32 expert map, got rank {} and dtype {}", .{ expert_map.rank(), expert_map.dtype() });

    const num_experts = if (opts.expert_map) |expert_map| expert_map.dim(.expert) else gate_up.dim(.expert);
    const routing = prepareRouting(ids, num_experts, @intCast(launch_config.block_size_m));

    const expert_ids = if (opts.expert_map) |expert_map|
        expert_map.gather(.{ .expert = routing.expert_ids }, .{}).withTags(.{.g})
    else
        routing.expert_ids;

    const hidden_quant, const input_scale = prepareInput(hidden, gate_up_scheme, gate_up.dtype(), opts.quantize_input);

    const gate_up_out = gemm(.{
        .input = hidden_quant,
        .weight = gate_up,
        .bias = opts.gate_up.bias,
        .input_scale = input_scale,
        .weight_scale = opts.gate_up.quantizationScales(),
        .routing = routing,
        .expert_ids = expert_ids,
        .launch_config = launch_config,
        .quant_scheme = gate_up_scheme,
        .top_k = @intCast(ids.dim(.topk)),
        .output_shape = Shape.init(.{ .token = routing.num_assignments, .out = gate_up.dim(.out) }, .bf16),
    });

    var activated = zml.moe.applyActivation(gate_up_out, opts.activation, opts.gate_up_layout);
    if (opts.routing_weight_placement == .before_down) {
        const weights = routing_weights.reshape(.{ .token = routing.num_assignments }).convert(.f32);
        activated = activated.mul(weights.broad(activated.shape()));
    }
    const activated_quant, const down_input_scale = prepareInput(activated, down_scheme, down.dtype(), opts.quantize_input);

    const down_out = gemm(.{
        .input = activated_quant,
        .weight = down,
        .bias = opts.down.bias,
        .input_scale = down_input_scale,
        .weight_scale = opts.down.quantizationScales(),
        .routing_weights = if (opts.routing_weight_placement == .after_down) routing_weights else null,
        .routing = routing,
        .expert_ids = expert_ids,
        .launch_config = down_launch_config,
        .quant_scheme = down_scheme,
        .top_k = 1,
        .output_shape = Shape.init(.{ .token = num_tokens, .topk = ids.dim(.topk), .out = down.dim(.out) }, .bf16),
    });

    const output = down_out.convert(.f32).sum(.topk).squeeze(.topk).convert(.bf16);

    return output.reshape(.{ .t = num_tokens, .d = down.dim(.out) });
}

pub const GemmOptions = struct {
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor = null,
    input_scale: ?Tensor = null,
    weight_scale: ?Tensor = null,
    routing_weights: ?Tensor = null,
    routing: Routing,
    expert_ids: Tensor,
    launch_config: LaunchConfig,
    quant_scheme: ?zml.Quantization.Scheme = null,
    top_k: usize,
    output_shape: Shape,
};

pub fn gemm(opts: GemmOptions) Tensor {
    // Native FP4 tensors expose logical K; the Triton operand is byte-packed.
    const weight = if (opts.quant_scheme == .mxfp4 and opts.weight.dtype() == .f4e2m1)
        opts.weight.reshape(opts.weight.shape().setDim(2, @divExact(opts.weight.dim(2), 2)).append(.{ .nibble = 2 })).bitCast(.u8)
    else
        opts.weight;
    const weight_k = weight.dim(2) * @as(i64, if (opts.quant_scheme == .mxfp4) 2 else 1);
    stdx.debug.assert(opts.input.dim(1) == weight_k, "input width {} must match logical weight K {}", .{ opts.input.dim(1), weight_k });
    if (opts.quant_scheme == .mxfp4) {
        stdx.debug.assert(@mod(weight_k, 32) == 0 and opts.weight_scale != null, "MXFP4 requires K divisible by 32 and weight scales", .{});
        stdx.debug.assert(opts.input.dtype() == .bf16 and opts.input_scale == null, "MXFP4 requires BF16 activations without input scales", .{});
    }

    if (opts.quant_scheme == .fp8_block32) {
        const scales = opts.weight_scale orelse stdx.debug.panic("block32 FP8 requires weight scales", .{});
        stdx.debug.assert(weight.dtype() == .f8e4m3fn and @mod(weight.dim(1), 32) == 0 and @mod(weight_k, 32) == 0, "block32 FP8 requires E4M3FN weights with N and K divisible by 32", .{});
        stdx.debug.assert(scales.rank() == 3 and scales.dim(0) == weight.dim(0) and scales.dim(1) == @divExact(weight.dim(1), 32) and scales.dim(2) == @divExact(weight_k, 32), "block32 FP8 requires scales shaped [expert, N/32, K/32], got {f}", .{scales.shape()});
        stdx.debug.assert(scales.dtype() == .f8e8m0 or scales.dtype() == .bf16 or scales.dtype() == .f32, "unsupported block32 FP8 scale dtype {}", .{scales.dtype()});
    }

    stdx.debug.assert(opts.quant_scheme != null or (opts.input_scale == null and opts.weight_scale == null), "scales require a quantization scheme", .{});
    for ([_]i64{ weight.dim(1), weight.dim(2), opts.input.dim(1), weight.dim(1) * weight.dim(2), opts.output_shape.dim(-1) }) |dim_or_stride| {
        stdx.debug.assert(dim_or_stride > 0 and @mod(dim_or_stride, 16) == 0, "FusedMoe dimensions and matrix strides must be positive multiples of 16, got {}", .{dim_or_stride});
    }
    const block_size_m: i64 = @intCast(opts.launch_config.block_size_m);
    const block_size_n: i64 = @intCast(opts.launch_config.block_size_n);
    const m_tokens = opts.input.dim(0);
    const em_effective = if (m_tokens < block_size_m)
        @min(opts.routing.max_num_tokens_padded, opts.routing.num_assignments * block_size_m)
    else
        opts.routing.max_num_tokens_padded;
    stdx.debug.assert(@mod(em_effective, block_size_m) == 0, "routing capacity {} must be a multiple of block size {}", .{ em_effective, block_size_m });
    const grid_x =
        @divExact(em_effective, block_size_m) *
        (std.math.divCeil(i64, weight.dim(1), block_size_n) catch unreachable);

    const weight_scale: ?Tensor = if (opts.weight_scale) |scale| blk: {
        const scheme = opts.quant_scheme orelse break :blk scale;
        break :blk switch (scheme) {
            // dot_scaled takes E8M0 encodings as bytes, including native E8M0 tensors.
            .mxfp4, .mxfp8 => scale.bitCast(.u8),
            .fp8_per_channel => scale.reshape(.{ weight.dim(0), weight.dim(1), 1 }),
            .fp8_per_tensor => if (scale.count() == 1) scale else scale.reshape(.{ weight.dim(0), 1, 1 }),
            // Tiled FP8 uses ordinary dot products with FP32 scale arithmetic.
            .fp8_block32 => scale.convert(.f32),
            .fp8_block128, .nvfp4 => scale,
        };
    } else null;

    const stride_asm: i64 = if (opts.input_scale) |scale| (if (scale.rank() == 2) scale.dim(1) else 0) else 0;
    const stride_ask: i64 = if (opts.input_scale) |scale| (if (scale.rank() == 2) 1 else 0) else 0;
    const has_weight_scale_expert_stride = if (weight_scale) |scale|
        !(opts.quant_scheme == .fp8_per_tensor and scale.count() == 1)
    else
        false;
    const stride_bse: i64 = if (weight_scale) |scale| (if (scale.rank() == 3 and has_weight_scale_expert_stride) scale.dim(1) * scale.dim(2) else 0) else 0;
    const stride_bsk: i64 = if (weight_scale) |scale| (if (scale.rank() == 3) 1 else 0) else 0;
    const stride_bsn: i64 = if (weight_scale) |scale| (if (scale.rank() == 3) scale.dim(2) else 0) else 0;

    return kernels.FusedMoe.Kernel.call(
        .{
            .a_ptr = opts.input,
            .b_ptr = weight,
            .b_bias_ptr = opts.bias orelse Tensor.scalar(0, .f32),
            .a_scale_ptr = opts.input_scale orelse Tensor.scalar(1.0, .f32),
            .b_scale_ptr = weight_scale orelse Tensor.scalar(1.0, .f32),
            .routing_weights_ptr = opts.routing_weights orelse Tensor.scalar(1.0, .f32),
            .sorted_token_ids_ptr = opts.routing.sorted_token_ids,
            .expert_ids_ptr = opts.expert_ids,
            .num_tokens_post_padded_ptr = opts.routing.num_tokens_post_padded,
            .N_ptr = Tensor.constant(.{ .i64 = weight.dim(1) }).reshape(.{1}),
            .K_ptr = Tensor.constant(.{ .i64 = weight_k }).reshape(.{1}),
            .EM_ptr = Tensor.constant(.{ .i64 = em_effective }).reshape(.{1}),
            .num_valid_tokens_ptr = Tensor.constant(.{ .i64 = opts.routing.num_assignments }).reshape(.{1}),
            .stride_am_ptr = Tensor.constant(.{ .i64 = opts.input.dim(1) }).reshape(.{1}),
            .stride_be_ptr = Tensor.constant(.{ .i64 = weight.dim(1) * weight.dim(2) }).reshape(.{1}),
            .stride_bn_ptr = Tensor.constant(.{ .i64 = weight.dim(2) }).reshape(.{1}),
            .stride_cm_ptr = Tensor.constant(.{ .i64 = weight.dim(.out) }).reshape(.{1}),
            .stride_asm_ptr = Tensor.constant(.{ .i64 = stride_asm }).reshape(.{1}),
            .stride_ask_ptr = Tensor.constant(.{ .i64 = stride_ask }).reshape(.{1}),
            .stride_bse_ptr = Tensor.constant(.{ .i64 = stride_bse }).reshape(.{1}),
            .stride_bsk_ptr = Tensor.constant(.{ .i64 = stride_bsk }).reshape(.{1}),
            .stride_bsn_ptr = Tensor.constant(.{ .i64 = stride_bsn }).reshape(.{1}),
            .stride_bbe_ptr = Tensor.constant(.{ .i64 = if (opts.bias) |bias| (if (bias.rank() == 2 and bias.dim(0) > 1) bias.dim(1) else 0) else 0 }).reshape(.{1}),
            .stride_bbn_ptr = Tensor.constant(.{ .i64 = if (opts.bias) |bias| (if (bias.rank() > 0 and bias.dim(-1) > 1) 1 else 0) else 0 }).reshape(.{1}),
        },
        .{ .c = opts.output_shape },
        .{
            .cfg = .{
                .a_dtype = toDType(opts.input.dtype()),
                .b_dtype = toDType(weight.dtype()),
                .c_dtype = toDType(opts.output_shape.dtype()),
                .a_scale_dtype = if (opts.input_scale) |scale| toDType(scale.dtype()) else null,
                .b_scale_dtype = if (weight_scale) |scale| toDType(scale.dtype()) else null,
                .b_bias_dtype = if (opts.bias) |bias| toDType(bias.dtype()) else null,
                .routing_weights_dtype = if (opts.routing_weights) |weights| toDType(weights.dtype()) else null,
                .block_size_m = opts.launch_config.block_size_m,
                .block_size_n = opts.launch_config.block_size_n,
                .block_size_k = opts.launch_config.block_size_k,
                .group_size_m = opts.launch_config.group_size_m,
                .top_k = opts.top_k,
                .naive_block_assignment = opts.routing.naive_block_assignment,
                .compute_type = .bf16,
                .quant_scheme = opts.quant_scheme,
            },
            .grid = .{ @intCast(grid_x), 1, 1 },
            .num_warps = opts.launch_config.num_warps,
            .num_stages = numStages(opts.launch_config),
        },
    ).c;
}

pub const Routing = struct {
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    max_num_tokens_padded: i64,
    num_assignments: i64,
    naive_block_assignment: bool,
};

pub fn prepareRouting(topk_ids: Tensor, num_experts: i64, block_size_m: i64) Routing {
    stdx.debug.assert(block_size_m > 0 and @mod(block_size_m, 16) == 0, "routing block size must be a positive multiple of 16, got {}", .{block_size_m});
    const ids = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_assignments = ids.dim(.token) * ids.dim(.topk);
    const sparsity_factor: i64 = 4;
    const naive_block_assignment = num_assignments * sparsity_factor <= num_experts;
    const max_num_tokens_padded = if (naive_block_assignment)
        num_assignments * block_size_m
    else if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        (std.math.divCeil(i64, num_assignments + num_experts * (block_size_m - 1), block_size_m) catch unreachable) * block_size_m;

    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        log.debug("Using naive block assignment for MoE kernels. Num assignments: {d}, Num experts: {d}", .{ num_assignments, num_experts });
        break :blk .{
            Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32)),
            ids.reshape(.{ .g = num_assignments }),
            Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1}),
        };
    } else alignBlockSize(ids, num_experts, block_size_m);

    return .{
        .sorted_token_ids = sorted_token_ids,
        .expert_ids = expert_ids,
        .num_tokens_post_padded = num_tokens_post_padded,
        .max_num_tokens_padded = max_num_tokens_padded,
        .num_assignments = num_assignments,
        .naive_block_assignment = naive_block_assignment,
    };
}

fn alignBlockSize(topk_ids: Tensor, num_experts: i64, block_size_m: i64) struct { Tensor, Tensor, Tensor } {
    const topk_ids_ = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_tokens = topk_ids_.dim(.token);
    const topk = topk_ids_.dim(.topk);
    const num_assignments = num_tokens * topk;
    const max_num_tokens_padded = if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        (std.math.divCeil(i64, num_assignments + num_experts * (block_size_m - 1), block_size_m) catch unreachable) * block_size_m;
    const max_num_m_blocks = @divExact(max_num_tokens_padded, block_size_m);
    const warp_size: i64 = 32;
    const padded_num_experts: i64 = @intCast(std.math.ceilPowerOfTwoAssert(u64, @intCast(@max(num_experts, warp_size))));
    const sort_block_size: i64 = 256;
    const sort_grid_x: i64 = @min(std.math.divCeil(i64, num_assignments, sort_block_size) catch unreachable, 65535);

    const flat_experts = topk_ids_.reshape(.{ .g = num_assignments });
    var cumsums = Tensor.zeroes(Shape.init(.{ .g = num_experts + 1 }, .i32));
    var expert_ids = Tensor.zeroes(Shape.init(.{ .g = max_num_m_blocks }, .i32));
    var sorted_token_ids = Tensor.zeroes(Shape.init(.{ .g = max_num_tokens_padded }, .i32));
    var num_tokens_post_padded = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));

    {
        const align_outs = kernels.MoeAlignBlockSize.Kernel.call(
            .{
                .topk_ids_ptr = flat_experts,
                .sorted_token_ids_ptr = sorted_token_ids,
                .expert_ids_ptr = expert_ids,
                .num_tokens_post_pad_ptr = num_tokens_post_padded,
                .cumsum_ptr = cumsums,
            },
            .{
                .sorted_token_ids = sorted_token_ids.shape(),
                .expert_ids = expert_ids.shape(),
                .num_tokens_post_pad = num_tokens_post_padded.shape(),
                .cumsum = cumsums.shape(),
            },
            .{
                .cfg = .{
                    .numel = @intCast(num_assignments),
                    .num_experts = @intCast(num_experts),
                    .padded_num_experts = @intCast(padded_num_experts),
                    .max_num_tokens_padded = @intCast(max_num_tokens_padded),
                    .max_num_m_blocks = @intCast(max_num_m_blocks),
                    .block_size_m = @intCast(block_size_m),
                    .hist_block = 256,
                },
                .grid = .{ 2, 1, 1 },
                .num_stages = 1,
                .num_warps = 8,
                .output_operand_aliases = .{
                    .sorted_token_ids = .sorted_token_ids_ptr,
                    .expert_ids = .expert_ids_ptr,
                    .num_tokens_post_pad = .num_tokens_post_pad_ptr,
                    .cumsum = .cumsum_ptr,
                },
            },
        );
        sorted_token_ids = align_outs.sorted_token_ids;
        expert_ids = align_outs.expert_ids;
        num_tokens_post_padded = align_outs.num_tokens_post_pad;
        cumsums = align_outs.cumsum;
    }

    {
        const sort_outs = kernels.CountAndSortExpertTokens.Kernel.call(
            .{
                .topk_ids_ptr = flat_experts,
                .sorted_token_ids_ptr = sorted_token_ids,
                .cumsum_ptr = cumsums,
            },
            .{
                .sorted_token_ids = sorted_token_ids.shape(),
                .cumsum = cumsums.shape(),
            },
            .{
                .cfg = .{
                    .numel = @intCast(num_assignments),
                    .num_experts = @intCast(num_experts),
                    .sort_block_size = @intCast(sort_block_size),
                },
                .grid = .{ @intCast(sort_grid_x), 1, 1 },
                .num_stages = 1,
                .num_warps = 4,
                .output_operand_aliases = .{
                    .sorted_token_ids = .sorted_token_ids_ptr,
                    .cumsum = .cumsum_ptr,
                },
            },
        );
        sorted_token_ids = sort_outs.sorted_token_ids;
        cumsums = sort_outs.cumsum;
    }

    return .{ sorted_token_ids, expert_ids, num_tokens_post_padded };
}

pub fn prepareInput(x: Tensor, scheme: ?zml.Quantization.Scheme, output_dtype: zml.DataType, quantize: bool) struct { Tensor, ?Tensor } {
    if (quantize) {
        if (scheme) |scheme_| {
            if (scheme_ != .mxfp4) {
                const input, const scale = quantizeFp8Input(x, scheme_, output_dtype);
                return .{ input, scale };
            }
        }
    }
    return .{ x.convert(.bf16), null };
}

fn quantizeFp8Input(x: Tensor, scheme: zml.Quantization.Scheme, output_dtype: DataType) struct { Tensor, Tensor } {
    const group_size: i64, const scale_dtype: DataType, const absmax_epsilon: f32 = switch (scheme) {
        .mxfp8 => .{ 32, .u8, 1e-10 },
        .fp8_per_channel, .fp8_per_tensor => .{ x.dim(1), .f32, 1e-10 },
        .fp8_block128 => .{ 128, .f32, 1e-10 },
        .fp8_block32 => .{ 32, .f32, 1e-4 },
        .mxfp4, .nvfp4 => unreachable,
    };
    stdx.debug.assert(x.rank() == 2, "expected a rank-2 activation matrix, got {f}", .{x.shape()});
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "activation width must be divisible by group size {d}, got {d}", .{ group_size, x.dim(1) });

    const groups_per_row = @divExact(x.dim(1), group_size);
    const quantized = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .feature = x.dim(1) }, output_dtype));
    const scales = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .group = groups_per_row }, scale_dtype));

    const outs = kernels.PerTokenGroupQuantFp8.Kernel.call(
        .{
            .y_ptr = x,
            .group_size_ptr = Tensor.constant(.{ .i64 = group_size }).reshape(.{1}),
            .y_num_columns_ptr = Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
            .y_row_stride_ptr = Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
            .eps_ptr = Tensor.scalar(absmax_epsilon, .f32),
        },
        .{ .y_q = quantized.shape(), .y_s = scales.shape() },
        .{
            .cfg = .{
                .input_dtype = toDType(x.dtype()),
                .output_dtype = toDType(output_dtype),
                .scale_dtype = toDType(scale_dtype),
                .block = std.math.ceilPowerOfTwoAssert(usize, @intCast(group_size)),
                .quant_scheme = scheme,
            },
            .grid = .{ @intCast(x.dim(0) * groups_per_row), 1, 1 },
            .num_stages = 1,
            .num_warps = 1,
        },
    );

    return .{ outs.y_q, outs.y_s };
}

test "SwiGLU uses FP32 math for concatenated and interleaved BF16 inputs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: Tensor, layout: ProjectionLayout, activation: zml.moe.Activation) Tensor {
            return zml.moe.applyActivation(x, activation, layout);
        }
    };
    const x: Tensor = .init(.{ .token = 1, .out = 6 }, .bf16);
    const gates = [_]f32{ 0.75, -1.25, 3.5 };
    const ups = [_]f32{ 0.875, -2.25, 4.5 };
    for ([_]ProjectionLayout{ .concatenated, .interleaved }) |layout| {
        var values: [6]zml.floats.BFloat16 = undefined;
        for (gates, ups, 0..) |gate, up, i| {
            values[if (layout == .concatenated) i else 2 * i] = .fromF32(gate);
            values[if (layout == .concatenated) i + 3 else 2 * i + 1] = .fromF32(up);
        }
        var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
        defer input.deinit();
        for ([_]?f32{ null, 2 }) |threshold| {
            const activation: zml.moe.Activation = .{ .swiglu = .{ .limit = threshold } };
            var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, layout, activation }, .{});
            defer exe.deinit();
            try zml.testing.expectEqualShapes(Shape.init(.{ .token = 1, .out = 3 }, .f32), exe.output_shapes[0]);
            var output = try exe.eval(allocator, io, .{input});
            defer output.deinit();
            var actual = try output.toSliceAlloc(allocator, io);
            defer actual.free(allocator);
            for (gates, ups, actual.constItems(f32)) |gate, up, value| {
                const g = if (threshold) |limit| @min(gate, limit) else gate;
                const u = if (threshold) |limit| std.math.clamp(up, -limit, limit) else up;
                try std.testing.expectApproxEqAbs(g / (1 + @exp(-g)) * u, value, 1e-6);
            }
        }
    }
}

test "FP8 routed GEMM with bias matches dequantized weights" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;

    const Local = struct {
        const Outputs = struct { actual: Tensor, expected: Tensor };

        fn expandScales(scales: Tensor, shape: Shape) Tensor {
            var expanded = scales.convert(.f32);
            for (0..shape.rank()) |axis| {
                const factor = @divExact(shape.dim(axis), expanded.dim(axis));
                const broad_shape = expanded.shape().insert(axis + 1, .{factor});
                const axes = Shape.range(broad_shape.rank(), .i64).remove(axis + 1);
                expanded = expanded.broadcast(broad_shape, axes.dims())
                    .reshape(expanded.shape().setDim(axis, shape.dim(axis)));
            }
            return expanded.withTags(shape.tags());
        }

        fn forward(x: Tensor, w: Tensor, scales: Tensor, ids: Tensor, weights: Tensor, bias: Tensor, scheme: zml.Quantization.Scheme, mx_native: bool, quantize_input: bool) Outputs {
            const weight = w.convert(.f8e4m3fn);
            var input = x;
            var input_scale: ?Tensor = null;
            var reference_input = x.convert(.f32);
            if (quantize_input) {
                input, input_scale = prepareInput(x, scheme, .f8e4m3fn, true);
                const scale = if (scheme == .mxfp8) input_scale.?.bitCast(.f8e8m0) else input_scale.?;
                reference_input = input.convert(.f32).withTags(x.shape().tags())
                    .mul(expandScales(scale.convert(.f32), x.shape()));
            }
            const weight_scale = if (scheme == .mxfp8) blk: {
                const native = scales.convert(.f8e8m0);
                break :blk if (mx_native) native else native.bitCast(.u8);
            } else scales;
            const config = launchConfigForTokens(8);
            const routing = prepareRouting(ids, w.dim(.expert), @intCast(config.block_size_m));
            const actual = gemm(.{
                .input = input,
                .weight = weight,
                .bias = bias,
                .input_scale = input_scale,
                .weight_scale = weight_scale,
                .routing_weights = weights,
                .routing = routing,
                .expert_ids = routing.expert_ids,
                .launch_config = config,
                .quant_scheme = scheme,
                .top_k = 1,
                .output_shape = Shape.init(.{ .token = x.dim(.token), .topk = 1, .out = w.dim(.out) }, .bf16),
            }).squeeze(.topk);
            const expanded_weight = weight.convert(.f32).mul(expandScales(scales, weight.shape()));
            const selected_weight = expanded_weight.gather(.{ .expert = ids.squeeze(.topk) }, .{});
            const dot = reference_input.dot(selected_weight, .in);
            const selected_bias = if (bias.rank() == 2)
                bias.gather(.{ .expert = ids.squeeze(.topk) }, .{})
            else
                bias;
            const unweighted = dot.add(selected_bias.convert(.f32).broad(dot.shape()));
            const expected = unweighted.mul(weights.squeeze(.topk).convert(.f32).broad(unweighted.shape())).convert(.bf16);
            return .{ .actual = actual, .expected = expected };
        }
    };

    for ([_]zml.Quantization.Scheme{ .mxfp8, .fp8_per_tensor, .fp8_per_channel, .fp8_block128 }) |scheme| {
        for ([_]bool{ false, true }) |quantize_input| {
            for ([_]i64{ 1, 17 }) |tokens| {
                const n: i64 = if (scheme == .fp8_block128) 256 else 48;
                const scale_dims: [3]i64 = switch (scheme) {
                    .mxfp8 => .{ 8, n, 8 },
                    .fp8_per_tensor => if (tokens == 1) .{ 1, 1, 1 } else .{ 8, 1, 1 },
                    .fp8_per_channel => .{ 8, n, 1 },
                    .fp8_block128 => .{ 8, 2, 2 },
                    else => unreachable,
                };
                const x: Tensor = .init(.{ .token = tokens, .in = 256 }, .bf16);
                const w: Tensor = .init(.{ .expert = 8, .out = n, .in = 256 }, .bf16);
                const scales: Tensor = .init(.{ scale_dims[0], scale_dims[1], scale_dims[2] }, .f32);
                const ids: Tensor = .init(.{ .token = tokens, .topk = 1 }, .i32);
                const weights: Tensor = .init(.{ .token = tokens, .topk = 1 }, .bf16);
                const bias: Tensor = if (tokens == 1)
                    .init(.{ .out = n }, .bf16)
                else
                    .init(.{ .expert = 8, .out = n }, .f32);
                var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, w, scales, ids, weights, bias, scheme, tokens == 1, quantize_input }, .{});
                defer exe.deinit();
                var buffers: [6]zml.Buffer = undefined;
                var initialized: usize = 0;
                defer for (buffers[0..initialized]) |*buffer| buffer.deinit();
                for ([_]Shape{ x.shape(), w.shape(), scales.shape(), ids.shape(), weights.shape(), bias.shape() }, 0..) |shape, i| {
                    const slice = try zml.Slice.alloc(allocator, shape);
                    defer slice.free(allocator);
                    for (0..shape.count()) |j| {
                        switch (i) {
                            2 => slice.items(f32)[j] = @as(f32, @floatFromInt(@as(u32, 1) << @as(u5, @intCast(j % 4)))) / 16,
                            3 => slice.items(i32)[j] = @intCast((j * 3 + 5) % 8),
                            5 => {
                                const value = @as(f32, @floatFromInt(@as(i32, @intCast(j % 17)) - 8)) / 2;
                                if (shape.dtype() == .f32) {
                                    slice.items(f32)[j] = value;
                                } else {
                                    slice.items(zml.floats.BFloat16)[j] = .fromF32(value);
                                }
                            },
                            else => slice.items(zml.floats.BFloat16)[j] = .fromF32(if (i == 4)
                                @as(f32, @floatFromInt(1 + j % 3)) / 4
                            else
                                @as(f32, @floatFromInt(@as(i32, @intCast((j * 7 + j / 256) % 13)) - 6)) / 4),
                        }
                    }
                    buffers[i] = try zml.Buffer.fromSlice(io, platform, slice);
                    initialized += 1;
                }
                var output: zml.Bufferized(Local.Outputs) = undefined;
                var runner = try exe.runner(allocator);
                defer runner.deinit(allocator);
                runner.run(io, .{ buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5] }, .{&output}, .{ .wait = true });
                defer zml.Buffer.deinitAll(Local.Outputs, &output);
                zml.testing.expectClose(io, output.expected, output.actual, .{ .absolute_tolerance = 0.03125, .relative_tolerance = 0.01 }) catch |err| {
                    log.warn("FP8 routed GEMM failed for scheme={s}, quantize_input={}, tokens={}", .{ @tagName(scheme), quantize_input, tokens });
                    return err;
                };
            }
        }
    }
}

test "fused experts support BF16 and MXFP4 layouts, bias, and routing weights" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const Local = struct {
        fn forward(x: Tensor, layout: ProjectionLayout, placement: RoutingWeightPlacement, storage_dtype: DataType) Tensor {
            const fp4 = storage_dtype != .bf16;
            const columns = Tensor.arange(.{ .end = 256 }, .i32).withTags(.{.dout});
            const is_gate = switch (layout) {
                .concatenated => columns.cmp(.LT, Tensor.scalar(128, .i32)),
                .interleaved => columns.remainder(Tensor.scalar(2, .i32)).cmp(.EQ, Tensor.scalar(0, .i32)),
            };
            const dtype: DataType = if (fp4) .u8 else .bf16;
            // E2M1 codes 1, 2, 3 encode 0.5, 1, 1.5; each byte holds two values.
            const gate_value = if (fp4) Tensor.scalar(0x11, .u8) else Tensor.scalar(0.5, .bf16);
            const up_value = if (fp4) Tensor.scalar(0x33, .u8) else Tensor.scalar(1.5, .bf16);
            const gate_up_values = is_gate.select(gate_value, up_value);
            var gate_up: zml.nn.Linear = .{
                .weight = gate_up_values.broad(Shape.init(.{ .expert = 8, .dout = 256, .d = @as(i64, if (fp4) 64 else 128) }, dtype)),
                .tag = Shape.toTag(.d),
                .quantization = if (fp4) .{
                    .scheme = .mxfp4,
                    .scales = Tensor.scalar(127, .u8).broad(Shape.init(.{ .expert = 8, .dout = 256, .d = 4 }, .u8)),
                } else null,
            };
            var down: zml.nn.Linear = .{
                .weight = (if (fp4) Tensor.scalar(0x22, .u8) else Tensor.scalar(1, .bf16)).broad(Shape.init(.{ .expert = 8, .d = 128, .dout = @as(i64, if (fp4) 64 else 128) }, dtype)),
                .bias = Tensor.scalar(2, .bf16).broad(Shape.init(.{ .expert = 8, .d = 128 }, .bf16)),
                .tag = Shape.toTag(.dout),
                .quantization = if (fp4) .{
                    .scheme = .mxfp4,
                    .scales = Tensor.scalar(127, .u8).broad(Shape.init(.{ .expert = 8, .d = 128, .dout = 4 }, .u8)),
                } else null,
            };
            if (storage_dtype == .f4e2m1) {
                gate_up.weight = gate_up.weight.bitCast(.f4e2m1).reshape(.{ .expert = 8, .dout = 256, .d = 128 });
                down.weight = down.weight.bitCast(.f4e2m1).reshape(.{ .expert = 8, .d = 128, .dout = 128 });
                gate_up.quantization.?.scales = gate_up.quantization.?.scales.bitCast(.f8e8m0);
                down.quantization.?.scales = down.quantization.?.scales.bitCast(.f8e8m0);
            }
            const route = Tensor.arange(.{ .end = 2 }, .i32).reshape(.{ .t = 1, .topk = 2 })
                .broad(Shape.init(.{ .t = x.dim(.t), .topk = 2 }, .i32));
            return fusedExpertsImpl(.{
                .hidden_states = x,
                .gate_up = gate_up,
                .down = down,
                .topk_ids = route.addConstant(2),
                .topk_weights = route.convert(.f32).addConstant(1).scale(0.25),
                .gate_up_layout = layout,
                .routing_weight_placement = placement,
                .quantize_input = false,
                .activation = .{ .swiglu = .{} },
            });
        }
    };
    for ([_]DataType{ .bf16, .u8, .f4e2m1 }) |storage_dtype| {
        const tokens: i64 = if (storage_dtype == .u8) 17 else 1;
        const x: Tensor = .init(.{ .t = tokens, .d = 128 }, .bf16);
        const host = try zml.Slice.alloc(allocator, x.shape());
        defer host.free(allocator);
        @memset(host.items(zml.floats.BFloat16), .fromF32(1.0 / 128.0));
        var input = try zml.Buffer.fromSlice(io, platform, host);
        defer input.deinit();
        for ([_]ProjectionLayout{ .concatenated, .interleaved }) |layout| {
            for ([_]RoutingWeightPlacement{ .before_down, .after_down }) |placement| {
                var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, layout, placement, storage_dtype }, .{});
                defer exe.deinit();
                var output = try exe.eval(allocator, io, .{input});
                defer output.deinit();
                var actual = try output.toSliceAlloc(allocator, io);
                defer actual.free(allocator);
                const activation: f32 = 0.5 / (1 + @exp(@as(f32, -0.5))) * 1.5;
                // The two routes have weights 0.25 and 0.5. Bias distinguishes placement.
                const bias: f32 = if (placement == .before_down) 4 else 1.5;
                const expected = 128 * activation * 0.75 + bias;
                for (actual.constItems(zml.floats.BFloat16)) |value| {
                    try std.testing.expectApproxEqAbs(expected, value.toF32(), 0.5);
                }
            }
        }
    }
}

// =============================================================================
// Config / validation helpers
// =============================================================================

const LaunchConfig = struct {
    block_size_m: usize,
    block_size_n: usize,
    block_size_k: usize,
    group_size_m: usize,
    num_warps: i32,
    num_stages: i32,
};

const launch_configs = [_]struct { tokens: i64, config: LaunchConfig }{
    .{ .tokens = 1, .config = .{ .block_size_m = 16, .block_size_n = 32, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 4 } },
    .{ .tokens = 2, .config = .{ .block_size_m = 16, .block_size_n = 32, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 4 } },
    .{ .tokens = 4, .config = .{ .block_size_m = 16, .block_size_n = 32, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 3 } },
    .{ .tokens = 8, .config = .{ .block_size_m = 16, .block_size_n = 128, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 3 } },
    .{ .tokens = 16, .config = .{ .block_size_m = 16, .block_size_n = 64, .block_size_k = 64, .group_size_m = 64, .num_warps = 4, .num_stages = 5 } },
    .{ .tokens = 24, .config = .{ .block_size_m = 16, .block_size_n = 64, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 2 } },
    .{ .tokens = 32, .config = .{ .block_size_m = 16, .block_size_n = 32, .block_size_k = 128, .group_size_m = 1, .num_warps = 4, .num_stages = 2 } },
    .{ .tokens = 48, .config = .{ .block_size_m = 16, .block_size_n = 32, .block_size_k = 128, .group_size_m = 64, .num_warps = 4, .num_stages = 2 } },
    .{ .tokens = 64, .config = .{ .block_size_m = 16, .block_size_n = 64, .block_size_k = 128, .group_size_m = 1, .num_warps = 4, .num_stages = 2 } },
    .{ .tokens = 96, .config = .{ .block_size_m = 16, .block_size_n = 128, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 3 } },
    .{ .tokens = 128, .config = .{ .block_size_m = 16, .block_size_n = 256, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 2 } },
    .{ .tokens = 256, .config = .{ .block_size_m = 16, .block_size_n = 256, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 2 } },
    .{ .tokens = 512, .config = .{ .block_size_m = 32, .block_size_n = 128, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 3 } },
    .{ .tokens = 1024, .config = .{ .block_size_m = 64, .block_size_n = 128, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 3 } },
    .{ .tokens = 1536, .config = .{ .block_size_m = 64, .block_size_n = 128, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 3 } },
    .{ .tokens = 2048, .config = .{ .block_size_m = 128, .block_size_n = 128, .block_size_k = 64, .group_size_m = 16, .num_warps = 8, .num_stages = 3 } },
    .{ .tokens = 3072, .config = .{ .block_size_m = 128, .block_size_n = 256, .block_size_k = 64, .group_size_m = 1, .num_warps = 8, .num_stages = 4 } },
    .{ .tokens = 4096, .config = .{ .block_size_m = 128, .block_size_n = 256, .block_size_k = 64, .group_size_m = 16, .num_warps = 8, .num_stages = 4 } },
};

fn launchConfigForTokens(num_tokens: i64) LaunchConfig {
    var best = launch_configs[0];
    for (launch_configs[1..]) |candidate| {
        // The table is ascending, so ties retain the smaller token bucket.
        if (@abs(candidate.tokens - num_tokens) < @abs(best.tokens - num_tokens)) {
            best = candidate;
        }
    }
    return best.config;
}

test "MoE launch config selects the nearest token bucket" {
    try std.testing.expectEqualDeep(launchConfigForTokens(4), launchConfigForTokens(6));
    try std.testing.expectEqualDeep(launchConfigForTokens(8), launchConfigForTokens(7));
    try std.testing.expectEqualDeep(launchConfigForTokens(1), launchConfigForTokens(0));
    try std.testing.expectEqualDeep(launchConfigForTokens(4096), launchConfigForTokens(8192));
    try std.testing.expectEqual(@as(usize, 64), launchConfigForTokens(16).group_size_m);
    try std.testing.expectEqual(@as(i32, 5), launchConfigForTokens(16).num_stages);
}

pub fn isAvailable(platform: *const zml.Platform) bool {
    if (platform.target != .cuda) return false;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return false;
    return cc.major == 10;
}

pub fn validateOptions(opts: zml.moe.Options) void {
    stdx.debug.assert(opts.activation == .swiglu, "triton backend only accepts swiglu activation, got {}", .{opts.activation});
    stdx.debug.assert(opts.activation.swiglu.limit != null, "triton backend requires swiglu limit to be set", .{});
    stdx.debug.assert(opts.activation.swiglu.bias == null, "triton backend requires swiglu bias to be null", .{});
    stdx.debug.assert(opts.activation.swiglu.scale == null, "triton backend requires swiglu scale to be null", .{});
}

/// Row-major MXFP4 E2M1 weights and linear E8M0 block32 scales.
/// BF16 activations are quantized on the GPU to FP8 with per-32 E8M0 scales.
pub fn fusedExpertsMxfp4(
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    options: zml.moe.Options,
) zml.Tensor {
    validateOptions(options);
    stdx.debug.assert(input.dtype() == .bf16, "triton backend only supports bf16 inputs, got {}", .{input.dtype()});
    stdx.debug.assert(gate_up.bias == null and down.bias == null, "triton backend expects gate_up bias and down bias to be null", .{});

    const gq = gate_up.quantization orelse @panic("triton backend requires gate_up quantization to be set");
    const dq = down.quantization orelse @panic("triton backend requires down quantization to be set");
    stdx.debug.assert(gq.scheme == .mxfp4 and dq.scheme == .mxfp4, "triton expects gate_up and down quantization scheme to be mxfp4, got {} and {}", .{ gq.scheme, dq.scheme });

    // Weight storage for mxfp4 in HF is expressed as u8 or i8
    stdx.debug.assert(gate_up.weight.dtype() == .u8 or gate_up.weight.dtype() == .i8, "triton expects gate_up weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});
    stdx.debug.assert(down.weight.dtype() == .u8 or down.weight.dtype() == .i8, "triton expects down weight dtype to be u8 or i8, got {}", .{gate_up.weight.dtype()});

    const expert_parallelism = gate_up.weight.shape().partition(.expert).isSharded();

    const context: Context = .{
        .input = input,
        .ids = ids,
        .weights = weights,
        .w1 = gate_up.weight.bitCast(.u8),
        .s1 = gq.scales,
        .w2 = down.weight.bitCast(.u8),
        .s2 = dq.scales,
        .global_experts = gate_up.weight.dim(.expert),
        .topk = ids.dim(.topk),
        .limit = options.activation.swiglu.limit.?,
        .routing_weight_placement = options.routing_weight_placement,
        .expert_parallel = expert_parallelism,
    };

    return if (expert_parallelism)
        zml.ops.manualComputation(Context.body, context, input.shape())
    else
        context.body(input.shape());
}

pub const Config = struct {
    tokens: i64,
    hidden: i64,
    intermediate: i64,
    experts: i64,
    global_experts: i64,
    topk: i64,
    expert_offset: i64 = 0,
    // Swiglu clamping. Set to 0 to disable.
    swiglu_limit: f32,
    /// Scale SwiGLU before FP8 quantization, or scale the down result before reduction.
    routing_weight_placement: RoutingWeightPlacement = .before_down,

    pub fn validate(c: Config) !void {
        if (c.tokens <= 0 or
            c.experts <= 0 or
            c.topk <= 0 or c.topk > c.global_experts or
            c.expert_offset < 0 or c.expert_offset + c.experts > c.global_experts)
        {
            return error.InvalidRoutingShape;
        }

        if (c.hidden < 128 or @mod(c.hidden, 128) != 0 or
            c.intermediate < 128 or @mod(c.intermediate, 128) != 0)
        {
            return error.InvalidHiddenShape;
        }

        if (!std.math.isFinite(c.swiglu_limit) or c.swiglu_limit < 0) {
            return error.InvalidScale;
        }
    }
};

pub fn validateInputs(
    c: Config,
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    w1: zml.Tensor,
    s1: zml.Tensor,
    w2: zml.Tensor,
    s2: zml.Tensor,
) !void {
    if (s1.rank() != 3 or s2.rank() != 3) return error.InvalidInputShape;
    if (s1.dim(2) < @divTrunc(c.hidden, 32) or
        s2.dim(2) < @divTrunc(c.intermediate, 32)) return error.InvalidInputShape;
    const shapes = .{
        .x = zml.Shape.init(.{ c.tokens, c.hidden }, .bf16),
        .ids = zml.Shape.init(.{ c.tokens, c.topk }, .i32),
        .scales = zml.Shape.init(.{ c.tokens, c.topk }, .f32),
        .w1 = zml.Shape.init(.{ c.experts, 2 * c.intermediate, @divTrunc(c.hidden, 2) }, .u8),
        .s1 = zml.Shape.init(.{ c.experts, 2 * c.intermediate, s1.dim(2) }, .f8e8m0),
        .w2 = zml.Shape.init(.{ c.experts, c.hidden, @divTrunc(c.intermediate, 2) }, .u8),
        .s2 = zml.Shape.init(.{ c.experts, c.hidden, s2.dim(2) }, .f8e8m0),
    };

    if (!input.shape().eql(shapes.x)) return error.InvalidInputShape;
    if (!ids.shape().eql(shapes.ids)) return error.InvalidInputShape;
    if (!weights.shape().eql(shapes.scales)) return error.InvalidInputShape;
    if (!w1.shape().eql(shapes.w1)) return error.InvalidInputShape;
    if (!s1.shape().eql(shapes.s1)) return error.InvalidInputShape;
    if (!w2.shape().eql(shapes.w2)) return error.InvalidInputShape;
    if (!s2.shape().eql(shapes.s2)) return error.InvalidInputShape;
}

const Context = struct {
    input: zml.Tensor,
    ids: zml.Tensor,
    weights: zml.Tensor,
    w1: zml.Tensor,
    s1: zml.Tensor,
    w2: zml.Tensor,
    s2: zml.Tensor,
    global_experts: i64,
    topk: i64,
    limit: f32,
    routing_weight_placement: RoutingWeightPlacement,
    expert_parallel: bool,

    fn body(self: Context, _: zml.Shape) zml.Tensor {
        const experts = self.w1.dim(.expert);
        const hidden = self.w2.dim(1);
        const intermediate = self.w2.dim(2) * 2;
        var ids = self.ids.convert(.i32).reshape(.{ .token = .auto, .topk = self.topk });
        const tokens = ids.dim(.token);
        if (self.expert_parallel) {
            const partition_id = zml.ops.partitionId().convert(.i32);
            ids = ids.sub(partition_id.scale(experts));
        }
        const x = self.input.reshape(.{ tokens, hidden });
        const scales = self.weights.convert(.f32).reshape(.{ tokens, self.topk });

        const cfg: Config = .{
            .tokens = tokens,
            .hidden = hidden,
            .intermediate = intermediate,
            .experts = experts,
            .global_experts = self.global_experts,
            .topk = self.topk,
            .swiglu_limit = self.limit,
            .routing_weight_placement = self.routing_weight_placement,
        };

        cfg.validate() catch @panic("Invalid MXFP4 MoE configuration");
        validateInputs(cfg, x, ids, scales, self.w1, self.s1, self.w2, self.s2) catch @panic("Invalid MXFP4 tensor layout");

        const bn = 16;
        const bk = 512;

        const tile_count = kernels.tiles(cfg.tokens, cfg.topk, cfg.experts, bn);

        // Quantize activations
        const q = kernels.Quant.Kernel.call(
            .{ .x = x },
            .{
                .q = .init(.{ cfg.tokens, cfg.hidden }, .f8e4m3fn),
                .s = .init(.{ cfg.tokens, @divTrunc(cfg.hidden, 32) }, .u8),
            },
            .{
                .cfg = .{ .tokens = cfg.tokens, .hidden = cfg.hidden },
                .grid = .{ @intCast(@divTrunc(cfg.tokens + kernels.quantRows(cfg.tokens) - 1, kernels.quantRows(cfg.tokens))), @intCast(@divTrunc(cfg.hidden, 128)), 1 },
                .num_warps = 4,
                .num_stages = 1,
            },
        );

        const dummy = zml.Tensor.scalar(0, .i32);
        var map = dummy;
        var perm = dummy;
        var sched = dummy;

        if (cfg.tokens != 1) {
            const r = kernels.Route.Kernel.call(
                .{ .ids = ids },
                .{ .counts = .init(.{cfg.experts}, .i32), .pos = .init(.{cfg.tokens * cfg.topk}, .i32) },
                .{ .cfg = .{
                    .tokens = cfg.tokens,
                    .topk = cfg.topk,
                    .expert_offset = cfg.expert_offset,
                }, .grid = .{ @intCast(cfg.experts), 1, 1 }, .num_warps = 4, .num_stages = 1 },
            );

            const s = kernels.Schedule.Kernel.call(
                .{ .ids = ids, .counts = r.counts, .pos = r.pos },
                .{ .map = .init(.{tile_count * bn}, .i32), .perm = .init(.{cfg.tokens * cfg.topk}, .i32), .sched = .init(.{2 * tile_count + 1}, .i32) },
                .{
                    .cfg = .{
                        .tokens = cfg.tokens,
                        .topk = cfg.topk,
                        .expert_offset = cfg.expert_offset,
                        .experts = cfg.experts,
                        .capacity = tile_count,
                        .group = bn,
                    },
                    .grid = .{ @intCast(@divTrunc(cfg.tokens * cfg.topk + 127, 128)), 1, 1 },
                    .num_warps = 4,
                    .num_stages = 1,
                },
            );

            map = s.map;
            perm = s.perm;
            sched = s.sched;
        }

        // Each narrow tile produces 64 activations: two complete scale groups.
        const up_columns: i64 = if (cfg.tokens <= 8) 64 else 128;
        const m = kernels.GateUp.Kernel.call(
            .{ .q = q.q, .s = q.s, .w = self.w1, .ws = self.s1.bitCast(.u8), .ids = ids, .rw = scales, .map = map, .sched = sched },
            .{ .mid = .init(.{ tile_count * bn, cfg.intermediate }, .f8e4m3fn), .ms = .init(.{ tile_count * bn, @divTrunc(cfg.intermediate, 32) }, .u8) },
            .{
                .cfg = .{
                    .tokens = cfg.tokens,
                    .hidden = cfg.hidden,
                    .intermediate = cfg.intermediate,
                    .experts = cfg.experts,
                    .topk = cfg.topk,
                    .expert_offset = cfg.expert_offset,
                    .swiglu_limit = cfg.swiglu_limit,
                    .scale_stride = self.s1.dim(2),
                    .routing_weight_placement = cfg.routing_weight_placement,
                    .bn = bn,
                    .bk = bk,
                },
                .grid = .{ @intCast(@divTrunc(cfg.intermediate, up_columns)), @intCast(tile_count), 1 },
                .num_warps = if (cfg.tokens <= 8) 4 else 8,
                .num_stages = 2,
                .global_scratch_memory_size = @intCast(128 * @divTrunc(cfg.intermediate, up_columns) * tile_count),
            },
        );

        const d = kernels.Down.Kernel.call(
            .{ .mid = m.mid, .ms = m.ms, .w = self.w2, .ws = self.s2.bitCast(.u8), .ids = ids, .sched = sched },
            .{ .d = .init(.{ tile_count * bn, cfg.hidden }, .bf16) },
            .{
                .cfg = .{
                    .tokens = cfg.tokens,
                    .hidden = cfg.hidden,
                    .intermediate = cfg.intermediate,
                    .experts = cfg.experts,
                    .topk = cfg.topk,
                    .expert_offset = cfg.expert_offset,
                    .swiglu_limit = cfg.swiglu_limit,
                    .scale_stride = self.s2.dim(2),
                    .routing_weight_placement = cfg.routing_weight_placement,
                    .bn = bn,
                    .bk = bk,
                },
                .grid = .{ @intCast(@divTrunc(cfg.hidden, 128)), @intCast(tile_count), 1 },
                .num_warps = if (cfg.tokens <= 8) 4 else 8,
                .num_stages = 2,
                .global_scratch_memory_size = @intCast(128 * @divTrunc(cfg.hidden, 128) * tile_count),
            },
        );

        const result = kernels.Combine.Kernel.call(
            .{ .d = d.d, .ids = ids, .perm = perm, .rw = scales },
            .{ .y = x.shape() },
            .{
                .cfg = .{
                    .tokens = cfg.tokens,
                    .hidden = cfg.hidden,
                    .topk = cfg.topk,
                    .expert_offset = cfg.expert_offset,
                    .experts = cfg.experts,
                    .routing_weight_placement = cfg.routing_weight_placement,
                    .bn = bn,
                },
                .grid = .{ @intCast(cfg.tokens), @intCast(@divTrunc(cfg.hidden + kernels.combineBlock(cfg.tokens, cfg.hidden) - 1, kernels.combineBlock(cfg.tokens, cfg.hidden))), 1 },
                .num_warps = 4,
                .num_stages = 1,
            },
        ).y.reshape(self.input.shape().dims()).withTags(self.input.shape());

        return if (self.expert_parallel)
            zml.ops.allReduce(result, zml.Tensor.add)
        else
            result;
    }
};
