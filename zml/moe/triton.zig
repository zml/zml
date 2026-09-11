const std = @import("std");
const stdx = @import("stdx");
const zml = @import("../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const DataType = zml.DataType;
const log = std.log.scoped(.moe);
const kernels = @import("triton_kernels/triton_kernels.zig");
const toDType = zml.kernel.triton.from;
const shared = @import("fused_experts.zig");

pub const Parameters = shared.Parameters;
pub const GateUpLayout = shared.GateUpLayout;
pub const RoutingWeightPlacement = shared.RoutingWeightPlacement;
pub const FusedExpertsArgs = shared.FusedExpertsArgs;

pub fn call(opts: shared.GemmOptions) Tensor {
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
            .num_stages = opts.launch_config.num_stages,
        },
    ).c;
}

pub fn prepareRouting(topk_ids: Tensor, num_experts: i64, block_size_m: i64) shared.Routing {
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
        .fp8_block32 => .{ 32, .f32, 1e-10 },
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
