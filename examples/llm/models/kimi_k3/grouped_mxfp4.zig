const std = @import("std");

const zml = @import("zml");
const stdx = zml.stdx;
const kernel = zml.moe.triton_a16w4_kernel;

// This native grouped MXFP4 path is adapted from the Apache-2.0 donor locked
// at b4f0af76e4c464c0f533420b94fdb1fba838c5e3. The Triton kernel itself is
// byte-for-byte identical; this file keeps only the routing and GEMM wrapper
// needed by Kimi K3. SiTU deliberately remains outside this linear primitive.

const block_m: i64 = 16;
const block_n: i64 = 64;
const block_k: i64 = 64;

const Routing = struct {
    num_tokens: i64,
    num_routes: i64,
    topk: i64,
    gather_divisor: i64,
    grid_m: i64,
    sorted_route_indices: zml.Tensor,
    active_routes: zml.Tensor,
    hist: zml.Tensor,
    offsets: zml.Tensor,
    expert_data: zml.Tensor,
};

/// Execute one native grouped A16W4 linear. `input` is either [token, d] or
/// [token, route, d], `expert_ids` is [token, route], packed values are
/// [expert, out, d/2], and scales are [expert, out, d/32]. The returned routes
/// are restored to the caller's original order and tagged [token, route, out].
pub fn linear(
    input: zml.Tensor,
    expert_ids: zml.Tensor,
    packed_values: zml.Tensor,
    scales: zml.Tensor,
) zml.Tensor {
    stdx.debug.assert(input.shape().hasTag(.token) != null, "grouped MXFP4 input must have a token axis, got {f}", .{input.shape()});
    stdx.debug.assert(input.shape().hasTag(.d) != null, "grouped MXFP4 input must have a d axis, got {f}", .{input.shape()});
    stdx.debug.assert(expert_ids.shape().hasTags(.{ .token, .route }), "grouped MXFP4 expert ids must have token/route axes, got {f}", .{expert_ids.shape()});
    stdx.debug.assert(packed_values.rank() == 3 and scales.rank() == 3, "grouped MXFP4 weights/scales must be rank 3, got {f} and {f}", .{ packed_values.shape(), scales.shape() });

    const num_tokens = expert_ids.dim(.token);
    const topk = expert_ids.dim(.route);
    const num_experts = packed_values.dim(0);
    const n = packed_values.dim(1);
    const packed_k = packed_values.dim(2);
    const scale_k = scales.dim(2);
    const contract_k = input.dim(.d);

    stdx.debug.assert(input.dim(.token) == num_tokens, "input tokens {} do not match route tokens {}", .{ input.dim(.token), num_tokens });
    stdx.debug.assert(scales.dim(0) == num_experts and scales.dim(1) == n, "grouped MXFP4 value/scale expert-output dimensions differ: {f} vs {f}", .{ packed_values.shape(), scales.shape() });
    stdx.debug.assert(packed_k * 2 == contract_k, "packed MXFP4 K {} does not match activation K {}", .{ packed_k, contract_k });
    stdx.debug.assert(scale_k * 32 == contract_k, "MXFP4 scale K {} does not match activation K {}", .{ scale_k, contract_k });

    const has_route_input = input.shape().hasTag(.route) != null;
    if (has_route_input) {
        stdx.debug.assert(input.dim(.route) == topk, "input route width {} does not match routing width {}", .{ input.dim(.route), topk });
    }

    const routing = prepareRouting(expert_ids, num_experts, has_route_input);
    const input_matrix = if (has_route_input)
        input.merge(.{ .row = .{ .token, .route } }).withTags(.{ .row, .k })
    else
        input.withTags(.{ .row, .k });

    const grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
    const even_k = @mod(contract_k, block_k) == 0;
    const output_shape: zml.Shape = .init(.{ .route = routing.num_routes, .out = n }, .bf16);
    const cfg: kernel.Cfg = .{
        .x_dtype = zml.kernel.triton.from(input_matrix.dtype()),
        .w_dtype = packedByteDtype(packed_values.dtype()),
        .w_mx_scale_dtype = packedByteDtype(scales.dtype()),
        .b_dtype = .bf16,
        .gammas_dtype = .f32,
        .y_dtype = .bf16,
        .HAS_B = false,
        .HAS_GAMMAS = false,
        .HAS_GATHER_INDX = true,
        .APPLY_SWIGLU = false,
        .N_EXPTS_ACT = @intCast(routing.gather_divisor),
        .BLOCK_M = @intCast(block_m),
        .BLOCK_N = @intCast(block_n),
        .BLOCK_K = @intCast(block_k),
        .GROUP_M = 1,
        .XCD_SWIZZLE = 1,
        .EVEN_K = even_k,
        .MASK_K_LIMIT = @intCast(if (even_k) block_k else @mod(contract_k, block_k)),
        .SPLIT_K = 1,
        .W_CACHE_MODIFIER = .cg,
    };

    const sorted_output = kernel.Kernel.call(
        .{
            .stride_y_k = scalarI64(0),
            .stride_y_m = scalarI64(n),
            .stride_y_n = scalarI64(1),
            .X = input_matrix,
            .stride_x_m = scalarI64(contract_k),
            .stride_x_k = scalarI64(1),
            .W = packed_values,
            .stride_w_e = scalarI64(n * packed_k),
            .stride_w_k = scalarI64(1),
            .stride_w_n = scalarI64(packed_k),
            .WMxScale = scales,
            .stride_w_mx_e = scalarI64(n * scale_k),
            .stride_w_mx_k = scalarI64(1),
            .stride_w_mx_n = scalarI64(scale_k),
            .B = input_matrix,
            .stride_b_e = scalarI64(0),
            .Gammas = zml.Tensor.scalar(1.0, .f32),
            .N = scalarI64(n),
            .K = scalarI64(contract_k),
            .GatherIndx = routing.sorted_route_indices,
            .ExptHist = routing.hist,
            .ExptOffs = routing.offsets,
            .ExptOffsSum = zml.Tensor.scalar(0, .i32),
            .ExptData = routing.expert_data,
            .grid_m = scalarI64(routing.grid_m),
            .grid_n = scalarI64(grid_n),
            .alpha = scalarF32(1.0),
            .limit = scalarF32(1.0),
        },
        .{ .Y = output_shape },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(routing.grid_m * grid_n), 1, 1 },
            .num_warps = 4,
            .num_stages = 3,
        },
    ).Y;

    const active = routing.active_routes.broad(sorted_output.shape().withDtype(.bool));
    const masked_output = active.select(sorted_output, zml.Tensor.zeroes(sorted_output.shape()));
    const original_order = zml.Tensor.zeroes(output_shape).scatterSlices(
        .{ .route = routing.sorted_route_indices },
        masked_output,
        .{},
    );
    return original_order.reshape(.{ .token = routing.num_tokens, .route = routing.topk, .out = n });
}

fn prepareRouting(expert_ids: zml.Tensor, num_experts: i64, has_route_input: bool) Routing {
    const num_tokens = expert_ids.dim(.token);
    const topk = expert_ids.dim(.route);
    const num_routes = num_tokens * topk;
    const gather_divisor: i64 = if (has_route_input) 1 else topk;
    const grid_m = if (num_routes <= num_experts)
        num_routes
    else
        (std.math.divCeil(i64, num_routes - num_experts + 1, block_m) catch unreachable) + num_experts - 1;

    const route_ids = expert_ids.convert(.i32);
    const valid_route_ids = route_ids.cmp(.GE, zml.Tensor.scalar(0, .i32))
        .logical(.AND, route_ids.cmp(.LT, zml.Tensor.scalar(num_experts, .i32)));
    const routable_ids = valid_route_ids.select(route_ids, zml.Tensor.scalar(num_experts, .i32));
    const sorted = routable_ids.flatten().withTags(.{.route}).sort(.route, .{});
    const sorted_ids = sorted.values.withTags(.{.route}).convert(.i32);
    const sorted_route_indices = sorted.indices.withTags(.{.route}).convert(.i32);
    const active_routes = sorted_ids.cmp(.LT, zml.Tensor.scalar(num_experts, .i32));

    const experts = zml.Tensor.arange(.{ .end = num_experts }, .i32).withTags(.{.expert});
    const route_expert_shape: zml.Shape = .init(.{ .route = num_routes, .expert = num_experts }, .i32);
    const ids_by_expert = sorted_ids.insertAxes(.last, .{.expert}).broad(route_expert_shape);
    const all_expert_ids = experts.insertAxes(0, .{.route}).broad(route_expert_shape);
    const hist = ids_by_expert.cmp(.EQ, all_expert_ids)
        .convert(.i32)
        .sum(.route)
        .squeeze(.route)
        .withTags(.{.expert});
    const offsets = hist.cumulativeSum(.expert).sub(hist).withTags(.{.expert});

    return .{
        .num_tokens = num_tokens,
        .num_routes = num_routes,
        .topk = topk,
        .gather_divisor = gather_divisor,
        .grid_m = grid_m,
        .sorted_route_indices = sorted_route_indices,
        .active_routes = active_routes,
        .hist = hist,
        .offsets = offsets,
        .expert_data = buildExpertBlockMap(hist, num_routes, grid_m),
    };
}

fn buildExpertBlockMap(hist: zml.Tensor, num_routes: i64, grid_m: i64) zml.Tensor {
    const num_experts = hist.dim(.expert);
    const max_blocks_per_expert = std.math.divCeil(i64, num_routes, block_m) catch unreachable;
    const tiles_per_expert = hist.addConstant(block_m - 1).divByConst(block_m).withTags(.{.expert});
    const tile_offsets = tiles_per_expert.cumulativeSum(.expert).sub(tiles_per_expert).withTags(.{.expert});

    const expert_ids = zml.Tensor.arange(.{ .end = num_experts }, .i32).withTags(.{.expert});
    const block_ids = zml.Tensor.arange(.{ .end = max_blocks_per_expert }, .i32).withTags(.{.block});
    const grid_shape: zml.Shape = .init(.{ .expert = num_experts, .block = max_blocks_per_expert }, .i32);
    const block_grid = block_ids.insertAxes(0, .{.expert}).broad(grid_shape);
    const valid = block_grid.cmp(.LT, tiles_per_expert.insertAxes(.last, .{.block}).broad(grid_shape));
    const target_idx = valid.select(
        tile_offsets.insertAxes(.last, .{.block}).broad(grid_shape).add(block_grid),
        zml.Tensor.scalar(0, .i32).broad(grid_shape),
    );
    const packed_data = block_grid.scale(65536).add(expert_ids.insertAxes(.last, .{.block}).broad(grid_shape));
    const updates = valid.select(packed_data, zml.Tensor.scalar(-1, .i32).broad(grid_shape));

    return zml.Tensor.scalar(-1, .i32)
        .broad(zml.Shape.init(.{ .tile = grid_m }, .i32))
        .scatterSlices(.{ .tile = target_idx }, updates, .{ .update_fn = scatterMax });
}

fn scatterMax(values: zml.ops.ScatterArgs) struct { zml.Tensor } {
    return .{values.input.maximum(values.update)};
}

fn packedByteDtype(dtype: zml.DataType) zml.kernel.triton.DType {
    return switch (dtype) {
        .i8, .u8, .f4e2m1, .f8e8m0 => .i8,
        else => zml.kernel.triton.from(dtype),
    };
}

fn scalarI64(value: i64) zml.Tensor {
    return zml.Tensor.constant(.{ .i64 = value }).reshape(.{1});
}

fn scalarF32(value: f32) zml.Tensor {
    return zml.Tensor.constant(.{ .f32 = value }).reshape(.{1});
}
