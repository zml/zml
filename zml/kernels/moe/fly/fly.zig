const std = @import("std");

const stdx = @import("stdx");

const nn = @import("zml/nn");
const quantization = @import("zml/quantization");
const Quantization = quantization.Quantization;

const zml = @import("zml/core");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const DataType = zml.DataType;
const ExpertsLayout = @import("zml/kernels/moe/api").ExpertsLayout;
const ProjectionLayout = @import("zml/kernels/moe/api").ProjectionLayout;
const RoutingWeightPlacement = @import("zml/kernels/moe/api").RoutingWeightPlacement;
const Activation = @import("zml/kernels/moe/api").Activation;
const Options = @import("zml/kernels/moe/api").Options;
const LocalRouting = @import("zml/kernels/moe/api").LocalRouting;
const fly = @import("moe.zig");

const log = std.log.scoped(.@"zml/moe/fused_experts");

test {
    std.testing.refAllDecls(@This());
}

pub fn fusedExperts(
    hidden_states: Tensor,
    routing_: LocalRouting,
    gate_up_: nn.Linear,
    down_: nn.Linear,
    opts: Options,
) Tensor {
    const expert_map_ = routing_.expertMap();
    const layout: ProjectionLayout = if (gate_up_.quantizationScheme() == .mxfp4) .row_interleaved_1 else .concatenated;
    const num_tokens = hidden_states.dim(.t);
    var launch_config = launchConfigForTokens(num_tokens);

    const gate_up_scheme = gate_up_.quantizationScheme();
    const down_scheme = down_.quantizationScheme();
    if (gate_up_scheme) |scheme| switch (scheme) {
        .fp8_block128 => launch_config.block_size_k = 128,
        .fp8_block32 => launch_config.block_size_k = 32,
        .mxfp4, .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    var down_launch_config = launchConfigForTokens(num_tokens);
    if (down_scheme) |scheme| switch (scheme) {
        .fp8_block128 => down_launch_config.block_size_k = 128,
        .fp8_block32 => down_launch_config.block_size_k = 32,
        .mxfp4, .nvfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = gate_up_.weight.withTags(.{ .expert, .out, .in });
    const down = down_.weight.withTags(.{ .expert, .out, .mid });
    const routing_weights = routing_.weights.reshape(.{ .token = num_tokens, .in = routing_.weights.dim(.topk) }).withTags(.{ .token, .topk });
    const ids = routing_.ids.reshape(.{ .token = num_tokens, .in = routing_.ids.dim(.topk) }).withTags(.{ .token, .topk });

    stdx.debug.assert(hidden.dtype() == .bf16, "fly backend expected BF16 hidden states, got {}", .{hidden.dtype()});
    if (gate_up_scheme == .mxfp4 or gate_up_scheme == .nvfp4) {
        stdx.debug.assert(gate_up.dtype() == .u8 or gate_up.dtype() == .i8 or gate_up.dtype() == .f4e2m1, "fly backend expected u8, i8 or f4e2m1 gate_up data type when quantization scheme is mxfp4 or nvfp4, got {}", .{gate_up.dtype()});
    } else {
        stdx.debug.assert(gate_up.dtype() == .bf16 or gate_up.dtype() == .f8e4m3fn or gate_up.dtype() == .f8e4m3fnuz, "fly backend expected gate_up data type to be bf16, fp8e4m3fn or f8e4m3fnuz, got {}", .{gate_up.dtype()});
    }
    if (down_scheme == .mxfp4 or down_scheme == .nvfp4) {
        stdx.debug.assert(down.dtype() == .u8 or down.dtype() == .i8 or down.dtype() == .f4e2m1, "fly backend expected u8, i8 or f4e2m1 down data type when quantization scheme is mxfp4 or nvfp4, got {}", .{down.dtype()});
    } else {
        stdx.debug.assert(down.dtype() == .bf16 or down.dtype() == .f8e4m3fn or down.dtype() == .f8e4m3fnuz, "fly backend expected down data type to be bf16, fp8e4m3fn or f8e4m3fnuz, got {}", .{down.dtype()});
    }

    stdx.debug.assert(routing_weights.dtype() == .f32 or routing_weights.dtype() == .bf16, "fly backend expected f32 or bf16 routing weights, got {}", .{routing_weights.dtype()});
    stdx.debug.assert(ids.dtype() == .i32, "fly backend expected i32 expert ids, got {}", .{ids.dtype()});
    const gate_up_k = gate_up.dim(.in) * @as(i64, if (gate_up.dtype() == .u8 or gate_up.dtype() == .i8) 2 else 1);
    const down_k = down.dim(.mid) * @as(i64, if (down.dtype() == .u8 or down.dtype() == .i8) 2 else 1);
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
    if (expert_map_) |expert_map|
        stdx.debug.assert(expert_map.dtype() == .i32 and expert_map.rank() == 1, "fly backend expected a rank-1 i32 expert map, got rank {} and dtype {}", .{ expert_map.rank(), expert_map.dtype() });

    const num_experts = if (expert_map_) |expert_map| expert_map.dim(.expert) else gate_up.dim(.expert);
    const routing = prepareRouting(ids, num_experts, @intCast(launch_config.block_size_m));

    const expert_ids = if (expert_map_) |expert_map|
        expert_map.gather(.{ .expert = routing.expert_ids }, .{}).withTags(.{.g})
    else
        routing.expert_ids;

    const local_ids = routing_.localIds().reshape(ids.shape());

    const gate_up_input, const gate_up_row_scales = prepareInput(hidden, gate_up_, opts, routing);
    const gate_up_out = gemm(.{
        .input = gate_up_input,
        .weight = gate_up,
        .bias = gate_up_.bias,
        .weight_scale = gate_up_.quantizationScales(),
        .routing = routing,
        .expert_ids = expert_ids,
        .launch_config = launch_config,
        .quant_scheme = gate_up_scheme,
        .top_k = @intCast(ids.dim(.topk)),
        .output_shape = Shape.init(.{ .token = routing.num_assignments, .out = gate_up.dim(.out) }, .bf16),
    });

    var activated = opts.activation.applyActivation(scaleRoutes(gate_up_out, gate_up_, local_ids, gate_up_row_scales), layout);
    if (opts.routing_weight_placement == .before_down) {
        const weights = routing_weights.reshape(.{ .token = routing.num_assignments }).convert(.f32);
        activated = activated.mul(weights.broad(activated.shape()));
    }

    const down_input, const down_row_scales = prepareInput(activated, down_, opts, routing);
    const down_out = gemm(.{
        .input = down_input,
        .weight = down,
        .bias = down_.bias,
        .weight_scale = down_.quantizationScales(),
        .routing_weights = if (opts.routing_weight_placement == .after_down) routing_weights else null,
        .routing = routing,
        .expert_ids = expert_ids,
        .launch_config = down_launch_config,
        .quant_scheme = down_scheme,
        .top_k = 1,
        .output_shape = Shape.init(.{ .token = num_tokens, .topk = ids.dim(.topk), .out = down.dim(.out) }, .bf16),
    });

    // Sorted MFMA tiles leave the rows of routes without an expert unwritten.
    const routed = local_ids.cmp(.GE, Tensor.scalar(0, .i32));
    const output = scaleRoutes(down_out, down_, local_ids, down_row_scales).mask(routed, 0).sum(.topk).squeeze(.topk).convert(.bf16);

    return output.reshape(.{ .t = num_tokens, .d = down.dim(.out) });
}

/// Ids and weights of sigmoid top-k routing from gate `logits` {.t, .expert} in one
/// kernel, or null when Fly cannot route them.
pub fn route(logits: Tensor, gate_bias: ?Tensor, opts: @import("zml/kernels/moe/api").RouteOptions) ?[2]Tensor {
    if (logits.rank() != 2 or logits.dim(.expert) > 128 or opts.top_k > 16) return null;
    const bias = if (gate_bias) |b| b.convert(.f32) else Tensor.zeroes(Shape.init(.{logits.dim(.expert)}, .f32));
    const tokens = Shape.init(.{logits.dim(.t)}, .i32);
    const active = if (opts.active) |a| a.convert(.i32).reshape(tokens) else Tensor.constant(.{ .i32 = 1 }).broad(tokens);
    const ids, const weights = fly.Route.call(logits.convert(.f32).transpose(.{ .t, .expert }), bias, active, opts.top_k, opts.renormalize, opts.scale);
    const routed = Shape.init(.{ .t = logits.dim(.t), .topk = opts.top_k }, .i32);
    return .{ ids.reshape(routed), weights.reshape(routed.withDtype(.f32)) };
}

// TODO(Corentin): unused for now
pub fn supports(hidden_states: Tensor, routing: LocalRouting, gate_up_: nn.Linear, down_: nn.Linear, opts: Options) bool {
    if (hidden_states.dtype() != .bf16 or gate_up_.quantizationScheme() != .mxfp4 or
        down_.quantizationScheme() != .mxfp4 or gate_up_.bias != null or down_.bias != null or
        opts.routing_weight_placement != .before_down) return false;
    const gate_up = gate_up_.weight;
    const down = down_.weight;
    if (gate_up.rank() != 3 or down.rank() != 3) return false;
    const gate_up_scale = gate_up_.quantizationScales() orelse return false;
    const down_scale = down_.quantizationScales() orelse return false;
    const experts = routing.global_num_experts;
    if (experts < 1 or experts > 512) return false;
    var max_width: i64 = hidden_states.dim(.d);
    inline for (.{ .{ gate_up, gate_up_scale }, .{ down, down_scale } }) |projection| {
        const weight, const scale = projection;
        const packing_factor: i64 = switch (weight.dtype()) {
            .u8, .i8 => 2,
            .f4e2m1 => 1,
            else => return false,
        };
        const n = weight.dim(1);
        const k = weight.dim(2) * packing_factor;
        if (!fly.Mfma.supportsShape(n, k)) return false;
        max_width = @max(max_width, n);
        if (weight.dim(0) < 1 or weight.dim(0) > 512 or
            scale.rank() != 3 or scale.dim(0) != weight.dim(0) or scale.dim(1) != n or scale.dim(2) != @divExact(k, 32)) return false;
        switch (scale.dtype()) {
            .u8, .i8, .f8e8m0 => {},
            else => return false,
        }
    }
    const routes = hidden_states.dim(.t) * routing.ids.dim(.topk);
    return routes > 0 and routes <= @divTrunc(std.math.maxInt(i32), max_width);
}

/// The GEMM input and, if the GEMM output needs them, the scales of its rows.
/// NVFP4 GEMMs take FP4 activations, but CDNA3 has no FP4 MFMA: activations are
/// rounded to NVFP4 and widened to a format that holds them exactly. Decode
/// GEMVs take them over their row scales in F16, others in BF16.
fn prepareInput(x: Tensor, linear: nn.Linear, opts: Options, routing: Routing) struct { Tensor, ?Tensor } {
    // Activations are rounded to BF16 before quantization, as for BF16 GEMMs.
    const input = x.convert(.bf16);
    return switch (linear.quantizationScheme() orelse return .{ input, null }) {
        .nvfp4 => fly.RoundNvfp4.call(input, opts.nvfp4_input_scale == .four_or_six, if (routing.naive_block_assignment) .f16 else .bf16),
        .mxfp8 => .{ if (opts.quantize_input) fly.RoundMxfp8.call(input) else input, null },
        .mxfp4, .fp8_per_channel, .fp8_per_tensor, .fp8_block128, .fp8_block32 => .{ input, null },
    };
}

/// GEMM outputs `{.token, ..., .out}` in FP32, times each route's input row scale
/// and the NVFP4 per-expert multiplier of its expert in `ids`.
fn scaleRoutes(x: Tensor, linear: nn.Linear, ids: Tensor, row_scales: ?Tensor) Tensor {
    const result = x.convert(.f32);
    // Gate/up input rows are tokens and down input rows are routes: expand both to routes.
    var per_route: ?Tensor = if (row_scales) |scales| scales.reshape(.{ .token = ids.dim(.token), .topk = .auto }).broad(ids.shape().withDtype(.f32)) else null;
    if (linear.quantization) |q| if (q.global_scale) |scale| {
        const global = scale.asMultiplier();
        // Nonlocal routes produce zeros; any valid expert works for them.
        const by_expert = if (global.rank() == 0) global.broad(ids.shape().withDtype(.f32)) else global.withTags(.{.expert}).gather(.{ .expert = ids.maximum(.scalar(0, .i32)) }, .{});
        per_route = if (per_route) |r| r.mul(by_expert) else by_expert;
    };
    const multiplier = per_route orelse return result;
    return result.mul(multiplier.reshape(result.shape().setDim(-1, 1).withDtype(.f32)).broad(result.shape()));
}

pub const GemmOptions = struct {
    input: Tensor,
    weight: Tensor,
    bias: ?Tensor = null,
    weight_scale: ?Tensor = null,
    routing_weights: ?Tensor = null,
    routing: Routing,
    expert_ids: Tensor,
    launch_config: LaunchConfig,
    quant_scheme: ?Quantization.Scheme = null,
    top_k: usize,
    output_shape: Shape,
};

pub fn gemm(opts: GemmOptions) zml.Tensor {
    const scheme_supported = opts.quant_scheme == .mxfp4 or opts.quant_scheme == .nvfp4 or
        (opts.quant_scheme == .mxfp8 and opts.routing.naive_block_assignment);
    stdx.debug.assert(scheme_supported and opts.bias == null and opts.routing_weights == null, "Fly MoE requires MXFP4 or NVFP4 weights, or MXFP8 weights for decode, BF16 activations and routing weights before the down projection", .{});
    const weight = switch (opts.weight.dtype()) {
        .f4e2m1 => opts.weight.reshape(opts.weight.shape().setDim(2, @divExact(opts.weight.dim(2), 2)).append(.{ .nibble = 2 })).bitCast(.u8),
        .f8e4m3fn => opts.weight.bitCast(.u8),
        else => opts.weight,
    };
    // The kernels read one scale word per 32 weights: NVFP4 pairs its 16-weight E4M3 scales.
    const bytes = opts.weight_scale.?.bitCast(.u8);
    const scales = if (opts.quant_scheme == .nvfp4)
        bytes.reshape(bytes.shape().setDim(2, @divExact(bytes.dim(2), 2)).append(.{ .pair = 2 })).bitCast(.i16)
    else
        bytes;
    return fly.Mfma.call(opts.input, weight, scales, opts.routing.sorted_token_ids, opts.expert_ids, opts.routing.num_tokens_post_padded, opts.output_shape, opts.top_k, opts.routing.naive_block_assignment, opts.launch_config.block_size_m);
}

pub const Routing = struct {
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    max_num_tokens_padded: i64,
    num_assignments: i64,
    naive_block_assignment: bool,
};

/// Decode batches of at most this many routes use the grouped GEMV; larger ones
/// share weights across sorted MFMA rows.
pub const max_gemv_routes = 256;

/// Whether Fly runs these weights for `routes` routes: MXFP8 runs only as a decode GEMV.
pub fn runs(gate_up: nn.Linear, routes: i64) bool {
    return gate_up.quantizationScheme() != .mxfp8 or routes <= max_gemv_routes;
}

pub fn prepareRouting(topk_ids: Tensor, num_experts: i64, block_size_m: i64) Routing {
    std.debug.assert(num_experts > 0 and num_experts <= 512);
    std.debug.assert(block_size_m > 0 and block_size_m <= 128 and @mod(block_size_m, 16) == 0);
    const ids = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_assignments = ids.dim(.token) * ids.dim(.topk);
    const naive_block_assignment = num_assignments <= max_gemv_routes;
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

test "RoundMxfp8 rounds to E4M3 over power-of-two block scales" {
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;
    const Local = struct {
        fn forward(x: Tensor) [2]Tensor {
            const blocks = x.convert(.f32).splitAxis(.in, .{ .block = -1, .value = 32 });
            const scales = quantization.ceilPowerOfTwo(blocks.abs().max(.value).maximum(.scalar(1e-10, .f32)).scale(1.0 / 448.0));
            const values = blocks.div(scales.broad(blocks.shape())).convert(.f8e4m3fn).convert(.f32);
            const expected = values.mul(scales.broad(values.shape())).merge(.{ .in = .{ .block, .value } }).convert(.bf16);
            return .{ expected, fly.RoundMxfp8.call(x) };
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Magnitudes vary by row and within blocks, reaching E4M3 subnormals.
    const x: Tensor = .init(.{ .t = 13, .in = 2304 }, .bf16);
    var host = try zml.Slice.alloc(allocator, x.shape());
    defer host.free(allocator);
    for (host.items(zml.floats.BFloat16), 0..) |*value, i| {
        const row: i32 = @intCast(i / 2304);
        const pattern: f32 = @floatFromInt(@as(i32, @intCast((i * 37 + i / 7) % 97)) - 48);
        const spread = std.math.pow(f32, 2, @floatFromInt(@mod(@as(i32, @intCast(i)), 13) - 6));
        value.* = .fromF32(pattern * spread * std.math.pow(f32, 2, @floatFromInt(@mod(row * 5, 11) - 5)));
    }
    var input = try zml.Buffer.fromSlice(io, platform, host);
    defer input.deinit();
    var exe = try platform.compileFn(allocator, io, Local.forward, .{x}, .{});
    defer exe.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var outputs: [2]zml.Buffer = undefined;
    runner.run(io, .{input}, .{&outputs}, .{ .wait = true });
    defer for (&outputs) |*output| output.deinit();
    try zml.testing.expectClose(io, outputs[0], outputs[1], .{ .absolute_tolerance = 0, .relative_tolerance = 0, .minimum_close_fraction = 1 });
}

test "RoundNvfp4 matches quantizeNvfp4Rows" {
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;
    const Outputs = struct { bf16: Tensor, f16: Tensor, reference: Tensor };
    const Local = struct {
        fn forward(x: Tensor, four_or_six: bool) Outputs {
            const quantized = quantization.quantizeNvfp4Rows(x.convert(.f32), .in, four_or_six);
            const normalized, const row_scales = fly.RoundNvfp4.call(x, four_or_six, .f16);
            return .{
                .bf16 = fly.RoundNvfp4.call(x, four_or_six, .bf16)[0],
                .f16 = normalized.convert(.f32).mul(row_scales.?.withTags(.{.t}).broad(normalized.shape().withDtype(.f32))).convert(.bf16),
                .reference = quantization.dequantizeNvfp4(quantized, .in).convert(.bf16),
            };
        }
    };
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Rows of 144 blocks leave part of each 256-thread block idle; magnitudes vary by row.
    const x: Tensor = .init(.{ .t = 13, .in = 2304 }, .bf16);
    var host = try zml.Slice.alloc(allocator, x.shape());
    defer host.free(allocator);
    for (host.items(zml.floats.BFloat16), 0..) |*value, i| {
        const row: i32 = @intCast(i / 2304);
        const pattern: f32 = @floatFromInt(@as(i32, @intCast((i * 37 + i / 7) % 97)) - 48);
        value.* = .fromF32(pattern / 16 * std.math.pow(f32, 2, @floatFromInt(@mod(row * 5, 11) - 5)));
    }
    var input = try zml.Buffer.fromSlice(io, platform, host);
    defer input.deinit();
    for ([_]bool{ false, true }) |four_or_six| {
        var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, four_or_six }, .{});
        defer exe.deinit();
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var outputs: zml.Bufferized(Outputs) = undefined;
        runner.run(io, .{input}, .{&outputs}, .{ .wait = true });
        defer zml.Buffer.deinitAll(Outputs, &outputs);
        const exact: zml.testing.CompareOpts = .{ .absolute_tolerance = 0, .relative_tolerance = 0, .minimum_close_fraction = 1 };
        try zml.testing.expectClose(io, outputs.reference, outputs.bf16, exact);
        try zml.testing.expectClose(io, outputs.reference, outputs.f16, exact);
    }
}
