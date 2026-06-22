const std = @import("std");

const zml = @import("zml");
const gemm = @import("moe_op_gemm_a16w4_kernel.zig");

const stdx = zml.stdx;

const Routing = struct {
    num_tokens: i64,
    num_routes: i64,
    topk: i64,
    block_m: i64,
    grid_m: i64,
    sorted_route_ids: zml.Tensor,
    sorted_weights: zml.Tensor,
    hist: zml.Tensor,
    offsets: zml.Tensor,
    expt_data: zml.Tensor,
};

const GemmOpts = struct {
    routing: Routing,
    weight_contract_tag: zml.Shape.Tag,
    weight_output_tag: zml.Shape.Tag,
    output_shape: zml.Shape,
    gather: ?zml.Tensor = null,
    gammas: ?zml.Tensor = null,
    bias: ?zml.Tensor = null,
    apply_swiglu: bool = false,
    activation_limit: f32 = 1.0,
};

pub fn forwardMoe(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    activation_limit: f32,
    parameters: zml.moe.Parameters,
) !zml.Tensor {
    _ = parameters; // autofix
    stdx.debug.assert(input.shape().hasTags(.{ .batch, .seq, .d }), "expected MoE input tags (.batch, .seq, .d), got {f}", .{input.shape()});
    stdx.debug.assert(topk_ids.shape().hasTags(.{ .batch, .seq, .eid }), "expected topk id tags (.batch, .seq, .eid), got {f}", .{topk_ids.shape()});
    stdx.debug.assert(topk_weights.shape().hasTags(.{ .batch, .seq, .eid }), "expected topk weight tags (.batch, .seq, .eid), got {f}", .{topk_weights.shape()});
    stdx.debug.assert(bias_gate_up == null and bias_down == null, "partitioned A16W4 MoE bias is not wired yet", .{});

    const expert_partition = weights_gate_up.shape().partition(.expert);
    if (!expert_partition.eql(.init(.experts))) {
        return forwardMoeLocal(
            input,
            topk_ids,
            topk_weights,
            weights_gate_up,
            scales_gate_up,
            bias_gate_up,
            weights_down,
            scales_down,
            bias_down,
            activation_limit,
        );
    }

    const gate_up_scale = scales_gate_up orelse stdx.debug.panic("A16W4 MoE GEMM requires gate/up scales", .{});
    const down_scale = scales_down orelse stdx.debug.panic("A16W4 MoE GEMM requires down scales", .{});

    const output = zml.ops.manualComputation(
        .{ input, topk_ids, topk_weights, weights_gate_up, gate_up_scale, weights_down, down_scale },
        input.shape().withPartitioning(.{ .batch = .replicated, .seq = .replicated, .d = .replicated }),
        .{ .activation_limit = activation_limit },
        (struct {
            fn body(ctx: anytype, _: std.mem.Allocator, sharded_inputs: []const zml.Tensor, _: zml.Shape) zml.Tensor {
                const local_num_experts = sharded_inputs[3].dim(.expert);
                const partition_id = zml.ops.partitionId().convert(.i32);
                const expert_start = partition_id.scale(local_num_experts).convert(.i32);

                const local_topk_ids, const local_topk_weights = blk: {
                    const local_ids = sharded_inputs[1].convert(.i32).sub(expert_start);
                    const in_range = local_ids.cmp(.GE, zml.Tensor.scalar(0, .i32))
                        .logical(.AND, local_ids.cmp(.LT, zml.Tensor.scalar(local_num_experts, .i32)));

                    break :blk .{
                        in_range.select(local_ids, zml.Tensor.scalar(0, .i32)),
                        in_range.select(sharded_inputs[2], zml.Tensor.scalar(0, sharded_inputs[2].dtype())),
                    };
                };

                const output = forwardMoeLocal(
                    sharded_inputs[0],
                    local_topk_ids,
                    local_topk_weights,
                    sharded_inputs[3],
                    sharded_inputs[4],
                    null,
                    sharded_inputs[5],
                    sharded_inputs[6],
                    null,
                    ctx.activation_limit,
                );

                const out_reshaped = output.reshape(sharded_inputs[0].shape().dims()).withTags(.{ .batch, .seq, .d });
                return zml.ops.allReduce(out_reshaped, zml.Tensor.add);
            }
        }).body,
    );

    return output;
}

fn forwardMoeLocal(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: ?zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: ?zml.Tensor,
    bias_down: ?zml.Tensor,
    activation_limit: f32,
) zml.Tensor {
    const routing = prepareRouting(topk_ids.merge(.{ .seq = .{ .batch, .seq } }), topk_weights, weights_gate_up.dim(.expert));
    const x = input.reshape(.{ .token = routing.num_tokens, .d = input.dim(.d) });

    const hidden_shape: zml.Shape = .init(.{
        .route = routing.num_routes,
        .dout = @divExact(weights_gate_up.dim(.dout), 2),
    }, .bf16);

    const hidden = runGemm(
        x,
        weights_gate_up,
        scales_gate_up,
        .{
            .routing = routing,
            .weight_contract_tag = zml.Shape.toTag(.d),
            .weight_output_tag = zml.Shape.toTag(.dout),
            .output_shape = hidden_shape,
            .gather = routing.sorted_route_ids,
            .gammas = routing.sorted_weights,
            .bias = bias_gate_up,
            .apply_swiglu = true,
            .activation_limit = activation_limit,
        },
    );

    const routed_shape: zml.Shape = .init(.{
        .route = routing.num_routes,
        .d = weights_down.dim(.d),
    }, .bf16);

    const routed = runGemm(
        hidden,
        weights_down,
        scales_down,
        .{
            .routing = routing,
            .weight_contract_tag = zml.Shape.toTag(.dout),
            .weight_output_tag = zml.Shape.toTag(.d),
            .output_shape = routed_shape,
            .bias = bias_down,
        },
    );

    const token_ids = routing.sorted_route_ids.divByConst(routing.topk).withTags(.{.route});
    const output_flat_shape: zml.Shape = .init(.{ .token = routing.num_tokens, .d = input.dim(.d) }, .f32);
    const output_flat = zml.Tensor.zeroes(output_flat_shape).scatterSlices(
        .{ .token = token_ids },
        routed.convert(.f32),
        .{},
    );

    return output_flat.reshape(input.shape().withDtype(.f32)).convert(input.dtype());
}

fn prepareRouting(topk_ids: zml.Tensor, topk_weights: zml.Tensor, num_experts: i64) Routing {
    const num_tokens = topk_ids.dim(.seq);
    const topk = topk_ids.dim(.eid);
    const num_routes = num_tokens * topk;

    const block_m = blk: {
        const tokens_per_expert = @max(@divFloor(num_routes, num_experts), 1);
        const power = std.math.ceilPowerOfTwoAssert(usize, @intCast(tokens_per_expert));
        break :blk @min(@max(@as(i64, @intCast(power)), 16), 128);
    };

    const grid_m = blk: {
        if (num_routes <= num_experts) break :blk num_routes;
        break :blk (std.math.divCeil(i64, @max(num_routes - num_experts + 1, 0), block_m) catch unreachable) + num_experts - 1;
    };

    const sorted = topk_ids.flatten().withTags(.{.route}).sort(.route, .{});
    const sorted_ids = sorted.values.withTags(.{.route}).convert(.i32);
    const sorted_route_ids = sorted.indices.withTags(.{.route}).convert(.i32);

    const sorted_weights = topk_weights.flatten().withTags(.{.route})
        .gather(.{ .route = sorted_route_ids.rename(.{ .route = .sorted_route }) }, .{})
        .rename(.{ .sorted_route = .route })
        .convert(.f32);

    const experts = zml.Tensor.arange(.{ .end = num_experts }, .i32).withTags(.{.expert});
    const route_expert_shape: zml.Shape = .init(.{ .route = num_routes, .expert = num_experts }, .i32);
    const ids_by_expert = sorted_ids.insertAxes(.last, .{.expert}).broad(route_expert_shape);
    const expert_ids = experts.insertAxes(0, .{.route}).broad(route_expert_shape);

    const hist = ids_by_expert.cmp(.EQ, expert_ids)
        .convert(.i32)
        .sum(.route)
        .squeeze(.route)
        .withTags(.{.expert});

    const offsets = hist.cumulativeSum(.expert).sub(hist).withTags(.{.expert});

    const expert_data = buildExpertBlockMap(hist, num_routes, grid_m, block_m);

    return .{
        .num_tokens = num_tokens,
        .num_routes = num_routes,
        .topk = topk,
        .block_m = block_m,
        .grid_m = grid_m,
        .sorted_route_ids = sorted_route_ids,
        .sorted_weights = sorted_weights,
        .hist = hist,
        .offsets = offsets,
        .expt_data = expert_data,
    };
}

fn buildExpertBlockMap(hist: zml.Tensor, num_routes: i64, grid_m: i64, block_m: i64) zml.Tensor {
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

fn runGemm(
    input: zml.Tensor,
    weights: zml.Tensor,
    scales: ?zml.Tensor,
    opts: GemmOpts,
) zml.Tensor {
    const weight_scales = scales orelse stdx.debug.panic("A16W4 MoE GEMM requires weight scales", .{});
    const input_matrix = input.withTags(.{ .row, .k });
    const contract_k = input_matrix.dim(.k);
    const packed_k = weights.dim(opts.weight_contract_tag);
    const scale_k = weight_scales.dim(opts.weight_contract_tag);
    const n = weights.dim(opts.weight_output_tag);

    stdx.debug.assert(packed_k * 2 == contract_k, "expected packed int4 weight K {} to match activation K {}", .{ packed_k, contract_k });
    stdx.debug.assert(scale_k * 32 == contract_k, "expected MX scale K {} to match activation K {}", .{ scale_k, contract_k });
    const activation_reduction_n: i64 = if (opts.apply_swiglu) 2 else 1;
    stdx.debug.assert(@mod(n, activation_reduction_n) == 0, "invalid GEMM output width {}", .{n});
    stdx.debug.assert(opts.output_shape.dim(-1) == @divExact(n, activation_reduction_n), "output shape {f} does not match GEMM N {}", .{ opts.output_shape, n });

    const block_n, const num_warps = kernelNWarps(opts.routing.block_m, n, opts.routing.grid_m);
    const block_k: i64 = 256;
    const grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
    const has_bias = opts.bias != null;
    const has_gather = opts.gather != null;
    const has_gammas = opts.gammas != null;

    const cfg: gemm.Cfg = .{
        .x_dtype = zml.kernel.triton.from(input_matrix.dtype()),
        .w_dtype = packedByteDtype(weights.dtype()),
        .w_mx_scale_dtype = packedByteDtype(weight_scales.dtype()),
        .b_dtype = zml.kernel.triton.from((opts.bias orelse input_matrix).dtype()),
        .gammas_dtype = zml.kernel.triton.from((opts.gammas orelse zml.Tensor.scalar(1.0, .f32)).dtype()),
        .y_dtype = zml.kernel.triton.from(opts.output_shape.dtype()),
        .HAS_B = has_bias,
        .HAS_GAMMAS = has_gammas,
        .HAS_GATHER_INDX = has_gather,
        .APPLY_SWIGLU = opts.apply_swiglu,
        .ACTIVATION_REDUCTION_N = @intCast(activation_reduction_n),
        .SWIGLU_ADD_RESIDUAL = false,
        .N_EXPTS_ACT = @intCast(opts.routing.topk),
        .BLOCK_M = @intCast(opts.routing.block_m),
        .BLOCK_N = @intCast(block_n),
        .BLOCK_K = @intCast(block_k),
        .GROUP_M = 4,
        .XCD_SWIZZLE = 1,
        .EVEN_K = @mod(contract_k, block_k) == 0,
        .MASK_K_LIMIT = @intCast(if (@mod(contract_k, block_k) == 0) block_k else @mod(contract_k, block_k)),
        .W_CACHE_MODIFIER = if (opts.routing.block_m <= 32) .cg else .none,
    };

    const y = gemm.Kernel.call(
        .{
            .stride_y_k = scalarI64(0),
            .stride_y_m = scalarI64(opts.output_shape.dim(-1)),
            .stride_y_n = scalarI64(1),
            .X = input_matrix,
            .stride_x_m = scalarI64(contract_k),
            .stride_x_k = scalarI64(1),
            .W = weights,
            .stride_w_e = scalarI64(n * packed_k),
            .stride_w_k = scalarI64(1),
            .stride_w_n = scalarI64(packed_k),
            .WMxScale = weight_scales,
            .stride_w_mx_e = scalarI64(n * scale_k),
            .stride_w_mx_k = scalarI64(1),
            .stride_w_mx_n = scalarI64(scale_k),
            .B = opts.bias orelse input_matrix,
            .stride_b_e = scalarI64(if (has_bias) n else 0),
            .Gammas = opts.gammas orelse zml.Tensor.scalar(1.0, .f32),
            .N = scalarI64(n),
            .K = scalarI64(contract_k),
            .GatherIndx = opts.gather orelse opts.routing.sorted_route_ids,
            .ExptHist = opts.routing.hist,
            .ExptOffs = opts.routing.offsets,
            .ExptOffsSum = zml.Tensor.scalar(0, .i32),
            .ExptData = opts.routing.expt_data,
            .grid_m = scalarI64(opts.routing.grid_m),
            .grid_n = scalarI64(grid_n),
            .alpha = scalarF32(1.0),
            .limit = scalarF32(opts.activation_limit),
        },
        .{ .Y = opts.output_shape },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(opts.routing.grid_m * grid_n), 1, 1 },
            .num_warps = @intCast(num_warps),
            .num_stages = 1,
        },
    ).Y;

    return y;
}

fn kernelNWarps(block_m: i64, n: i64, grid_m: i64) struct { i64, i64 } {
    if (block_m == 16) {
        var block_n: i64 = 128;
        const num_warps: i64 = 4;
        var grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
        var grid = grid_m * grid_n;
        while (block_n >= 64 and grid < 256) {
            block_n = @divExact(block_n, 2);
            grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
            grid = grid_m * grid_n;
        }
        return .{ block_n, num_warps };
    }

    if (block_m == 32) {
        if (n <= 1024) return .{ 128, 4 };
        if (n <= 4096) return .{ 256, 8 };
        return .{ 512, 8 };
    }

    return .{ 512, 8 };
}

fn packedByteDtype(dt: zml.DataType) zml.kernel.triton.DType {
    return switch (dt) {
        .i8, .u8, .f4e2m1, .f8e8m0 => .i8,
        else => zml.kernel.triton.from(dt),
    };
}

fn scalarI64(v: i64) zml.Tensor {
    return zml.Tensor.constant(.{ .i64 = v }).reshape(.{1});
}

fn scalarF32(v: f32) zml.Tensor {
    return zml.Tensor.constant(.{ .f32 = v }).reshape(.{1});
}
