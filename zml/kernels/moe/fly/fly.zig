const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../../../zml.zig");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const DataType = zml.DataType;
const fly = @import("moe.zig");
const ProjectionLayout = zml.moe.ProjectionLayout;
const RoutingWeightPlacement = zml.moe.RoutingWeightPlacement;

const log = std.log.scoped(.@"zml/moe/fused_experts");

test {
    std.testing.refAllDecls(@This());
}

pub fn fusedExperts(
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
                const local_reshaped = local_output.reshape(local_args.hidden_states.shape().dims()).withTags(.{ .b, .s, .d });
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
    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    const num_tokens = b * s;
    var launch_config = launchConfigForTokens(num_tokens);

    const gate_up_scheme = opts.gate_up.quantizationScheme();
    const down_scheme = opts.down.quantizationScheme();
    if (gate_up_scheme) |scheme| switch (scheme) {
        .nvfp4 => stdx.debug.panic("fly backend doesn't support gate_up nvfp4 quantization scheme", .{}),
        .fp8_block128 => launch_config.block_size_k = 128,
        .fp8_block32 => launch_config.block_size_k = 32,
        .mxfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    var down_launch_config = launchConfigForTokens(num_tokens);
    if (down_scheme) |scheme| switch (scheme) {
        .nvfp4 => stdx.debug.panic("fly backend doesn't support down nvfp4 quantization scheme", .{}),
        .fp8_block128 => down_launch_config.block_size_k = 128,
        .fp8_block32 => down_launch_config.block_size_k = 32,
        .mxfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = opts.gate_up.weight.withTags(.{ .expert, .out, .in });
    const down = opts.down.weight.withTags(.{ .expert, .out, .mid });
    const routing_weights = topk_weights.reshape(.{ .token = num_tokens, .in = topk_weights.dim(.topk) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = num_tokens, .in = topk_ids.dim(.topk) }).withTags(.{ .token, .topk });

    stdx.debug.assert(hidden.dtype() == .bf16, "fly backend expected BF16 hidden states, got {}", .{hidden.dtype()});
    if (gate_up_scheme == .mxfp4) {
        stdx.debug.assert(gate_up.dtype() == .u8 or gate_up.dtype() == .i8 or gate_up.dtype() == .f4e2m1, "fly backend expected u8, i8 or f4e2m1 gate_up data type when quantization scheme is mxfp4, got {}", .{gate_up.dtype()});
    } else {
        stdx.debug.assert(gate_up.dtype() == .bf16 or gate_up.dtype() == .f8e4m3fn or gate_up.dtype() == .f8e4m3fnuz, "fly backend expected gate_up data type to be bf16, fp8e4m3fn or f8e4m3fnuz, got {}", .{gate_up.dtype()});
    }
    if (down_scheme == .mxfp4) {
        stdx.debug.assert(down.dtype() == .u8 or down.dtype() == .i8 or down.dtype() == .f4e2m1, "fly backend expected u8, i8 or f4e2m1 down data type when quantization scheme is mxfp4, got {}", .{down.dtype()});
    } else {
        stdx.debug.assert(down.dtype() == .bf16 or down.dtype() == .f8e4m3fn or down.dtype() == .f8e4m3fnuz, "fly backend expected down data type to be bf16, fp8e4m3fn or f8e4m3fnuz, got {}", .{down.dtype()});
    }

    stdx.debug.assert(routing_weights.dtype() == .f32 or routing_weights.dtype() == .bf16, "fly backend expected f32 or bf16 routing weights, got {}", .{routing_weights.dtype()});
    stdx.debug.assert(ids.dtype() == .i32, "fly backend expected i32 expert ids, got {}", .{ids.dtype()});
    const gate_up_k = gate_up.dim(.in) * @as(i64, if (gate_up_scheme == .mxfp4 and gate_up.dtype() != .f4e2m1) 2 else 1);
    const down_k = down.dim(.mid) * @as(i64, if (down_scheme == .mxfp4 and down.dtype() != .f4e2m1) 2 else 1);
    stdx.debug.assert(hidden.dim(.in) == gate_up_k, "fly backend expected hidden width to match gate/up input width, got {} and {}", .{ hidden.dim(.in), gate_up_k });
    const activation_reduction: i64 = switch (opts.activation) {
        .gelu, .relu, .silu => 1,
        .swiglu, .swiglu_step, .geglu, .geglu_tanh => 2,
    };
    stdx.debug.assert(@rem(gate_up.dim(.out), activation_reduction) == 0, "fly backend expected gate/up output width to be divisible by {}, got {}", .{ activation_reduction, gate_up.dim(.out) });
    stdx.debug.assert(down_k == @divFloor(gate_up.dim(.out), activation_reduction), "fly backend expected down input width to match activated width, got {} and {}", .{ down_k, @divFloor(gate_up.dim(.out), activation_reduction) });
    stdx.debug.assert(ids.dim(.token) == hidden.dim(.token) and routing_weights.dim(.token) == hidden.dim(.token), "fly backend expected routing ids and weights to match hidden token count, got {}, {} and {}", .{ ids.dim(.token), routing_weights.dim(.token), hidden.dim(.token) });
    stdx.debug.assert(ids.dim(.topk) == routing_weights.dim(.topk), "fly backend expected routing ids and weights to have matching top-k dimensions, got {} and {}", .{ ids.dim(.topk), routing_weights.dim(.topk) });
    stdx.debug.assert(gate_up.dim(.expert) == down.dim(.expert), "fly backend expected gate/up and down expert counts to match, got {} and {}", .{ gate_up.dim(.expert), down.dim(.expert) });
    if (opts.expert_map) |expert_map|
        stdx.debug.assert(expert_map.dtype() == .i32 and expert_map.rank() == 1, "fly backend expected a rank-1 i32 expert map, got rank {} and dtype {}", .{ expert_map.rank(), expert_map.dtype() });

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
        .output_shape = Shape.init(.{ .token = b * s, .topk = ids.dim(.topk), .out = down.dim(.out) }, .bf16),
    });

    const output = down_out.convert(.f32).sum(.topk).squeeze(.topk).convert(.bf16);

    return output.reshape(.{ .b = b, .token = s, .out = down.dim(.out) });
}

// TODO(Corentin): unused for now
pub fn supports(opts: FusedExpertsArgs) bool {
    if (opts.hidden_states.dtype() != .bf16 or opts.gate_up.quantizationScheme() != .mxfp4 or
        opts.down.quantizationScheme() != .mxfp4 or opts.gate_up.bias != null or opts.down.bias != null or
        opts.routing_weight_placement != .before_down) return false;
    const gate_up = opts.gate_up.weight;
    const down = opts.down.weight;
    if (gate_up.rank() != 3 or down.rank() != 3 or opts.hidden_states.dim(.d) != 5120) return false;
    const gate_up_scale = opts.gate_up.quantizationScales() orelse return false;
    const down_scale = opts.down.quantizationScales() orelse return false;
    const experts = if (opts.expert_map) |map| map.dim(.expert) else gate_up.dim(0);
    if (experts < 1 or experts > 512) return false;
    inline for (.{ .{ gate_up, gate_up_scale, 4608, 5120 }, .{ down, down_scale, 5120, 2304 } }) |projection| {
        const weight, const scale, const n, const k = projection;
        const packing_factor: i64 = switch (weight.dtype()) {
            .u8, .i8 => 2,
            .f4e2m1 => 1,
            else => return false,
        };
        if (weight.dim(0) < 1 or weight.dim(0) > 512 or weight.dim(1) != n or weight.dim(2) * packing_factor != k or
            scale.rank() != 3 or scale.dim(0) != weight.dim(0) or scale.dim(1) != n or scale.dim(2) != k / 32) return false;
        switch (scale.dtype()) {
            .u8, .i8, .f8e8m0 => {},
            else => return false,
        }
    }
    const routes = opts.hidden_states.dim(.b) * opts.hidden_states.dim(.s) * opts.topk_ids.dim(.topk);
    return routes > 0 and routes <= std.math.maxInt(i32) / 5120;
}

pub fn prepareInput(x: Tensor, scheme: ?zml.Quantization.Scheme, _: zml.DataType, _: bool) struct { Tensor, ?Tensor } {
    std.debug.assert(scheme == .mxfp4);
    return .{ x.convert(.bf16), null };
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

pub fn gemm(opts: GemmOptions) zml.Tensor {
    zml.stdx.debug.assert(opts.quant_scheme == .mxfp4 and opts.bias == null and opts.routing_weights == null and opts.input_scale == null, "Fly MoE requires MXFP4 weights, BF16 activations and routing weights before the down projection", .{});
    const weight = if (opts.weight.dtype() == .f4e2m1)
        opts.weight.reshape(opts.weight.shape().setDim(2, @divExact(opts.weight.dim(2), 2)).append(.{ .nibble = 2 })).bitCast(.u8)
    else
        opts.weight;
    return fly.Mfma.call(opts.input, weight, opts.weight_scale.?.bitCast(.u8), opts.routing.sorted_token_ids, opts.expert_ids, opts.routing.num_tokens_post_padded, opts.output_shape, opts.top_k, opts.routing.naive_block_assignment, opts.launch_config.block_size_m);
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
    std.debug.assert(num_experts > 0 and num_experts <= 512);
    std.debug.assert(block_size_m > 0 and block_size_m <= 128 and @mod(block_size_m, 16) == 0);
    const ids = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_assignments = ids.dim(.token) * ids.dim(.topk);
    // GEMV handles up to six routes; MFMA shares weights across sorted rows.
    const naive_block_assignment = num_assignments <= 6;
    const max_num_tokens_padded = if (naive_block_assignment or num_assignments < num_experts)
        num_assignments * block_size_m
    else
        (std.math.divCeil(i64, num_assignments + num_experts * (block_size_m - 1), block_size_m) catch unreachable) * block_size_m;
    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        break :blk .{
            Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32)),
            ids.reshape(.{ .g = num_assignments }),
            Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1}),
        };
    } else blk: {
        const flat_experts = ids.reshape(.{ .g = num_assignments });
        const aligned = fly.Align.call(
            flat_experts,
            Tensor.zeroes(Shape.init(.{ .g = max_num_tokens_padded }, .i32)),
            Tensor.zeroes(Shape.init(.{ .g = @divExact(max_num_tokens_padded, block_size_m) }, .i32)),
            Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32)),
            Tensor.zeroes(Shape.init(.{ .g = num_experts + 1 }, .i32)),
            block_size_m,
        );
        const sorted = fly.Sort.call(flat_experts, aligned.sorted_token_ids, aligned.cumsum);
        break :blk .{ sorted.sorted_token_ids, aligned.expert_ids, aligned.num_tokens_post_pad };
    };
    return .{
        .sorted_token_ids = sorted_token_ids,
        .expert_ids = expert_ids,
        .num_tokens_post_padded = num_tokens_post_padded,
        .max_num_tokens_padded = max_num_tokens_padded,
        .num_assignments = num_assignments,
        .naive_block_assignment = naive_block_assignment,
    };
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
