const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const tri = zml.kernel.triton;
const toDType = tri.from;
const kernels = @import("triton_kernels/triton_kernels.zig");

const log = std.log.scoped(.moe_triton);

test {
    std.testing.refAllDecls(@This());
}

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: ActivationMode,

    pub const ActivationMode = enum {
        silu,
        relu,
        gelu,
    };

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: ActivationMode,
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
        };
    }
};

pub const GateUpLayout = enum { split, interleaved };
pub const RoutingWeightPlacement = enum { before_down, after_down };

pub const FusedExpertsArgs = struct {
    hidden_states: Tensor,
    gate_up: zml.nn.Linear,
    down: zml.nn.Linear,
    topk_weights: Tensor,
    topk_ids: Tensor,
    activation: Parameters.ActivationMode = .silu,
    expert_map: ?Tensor = null,
    activation_threshold: ?f32 = null,
    /// Use FP8 activations for FP8 weights; false keeps BF16 activations.
    quantize_input: bool,
    gate_up_layout: GateUpLayout,
    routing_weight_placement: RoutingWeightPlacement,
};

pub fn fusedExpertsImpl(opts: FusedExpertsArgs) !Tensor {
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
        .nvfp4 => return error.UnsupportedQuantization,
        .fp8_block128 => launch_config.block_size_k = 128,
        .mxfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    var down_launch_config = launchConfigForTokens(num_tokens);
    if (down_scheme) |scheme| switch (scheme) {
        .fp8_block128 => down_launch_config.block_size_k = 128,
        .mxfp4, .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
        .nvfp4 => return error.UnsupportedQuantization,
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = opts.gate_up.weight.withTags(.{ .expert, .out, .in });
    const down = opts.down.weight.withTags(.{ .expert, .out, .mid });
    const routing_weights = topk_weights.reshape(.{ .token = num_tokens, .in = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = num_tokens, .in = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    stdx.debug.assert(hidden.dtype() == .bf16, "expected BF16 hidden states, got {}", .{hidden.dtype()});
    stdx.debug.assert(if (gate_up_scheme == .mxfp4) gate_up.dtype() == .u8 or gate_up.dtype() == .i8 or gate_up.dtype() == .f4e2m1 else gate_up.dtype() == .bf16 or gate_up.dtype() == .f8e4m3fn or gate_up.dtype() == .f8e4m3fnuz, "unsupported gate/up weight dtype {}", .{gate_up.dtype()});
    stdx.debug.assert(if (down_scheme == .mxfp4) down.dtype() == .u8 or down.dtype() == .i8 or down.dtype() == .f4e2m1 else down.dtype() == .bf16 or down.dtype() == .f8e4m3fn or down.dtype() == .f8e4m3fnuz, "unsupported down weight dtype {}", .{down.dtype()});
    stdx.debug.assert(routing_weights.dtype() == .f32 or routing_weights.dtype() == .bf16, "expected FP32 or BF16 routing weights, got {}", .{routing_weights.dtype()});
    stdx.debug.assert(ids.dtype() == .i32, "expected I32 expert ids, got {}", .{ids.dtype()});
    const gate_up_k = gate_up.dim(.in) * @as(i64, if (gate_up_scheme == .mxfp4 and gate_up.dtype() != .f4e2m1) 2 else 1);
    const down_k = down.dim(.mid) * @as(i64, if (down_scheme == .mxfp4 and down.dtype() != .f4e2m1) 2 else 1);
    stdx.debug.assert(hidden.dim(.in) == gate_up_k, "hidden width {} must match gate/up input width {}", .{ hidden.dim(.in), gate_up_k });
    const activation_reduction: i64 = if (opts.activation == .relu) 1 else 2;
    stdx.debug.assert(@rem(gate_up.dim(.out), activation_reduction) == 0, "gate/up output width {} must be divisible by {}", .{ gate_up.dim(.out), activation_reduction });
    stdx.debug.assert(down_k == @divFloor(gate_up.dim(.out), activation_reduction), "down input width {} must match activated width {}", .{ down_k, @divFloor(gate_up.dim(.out), activation_reduction) });
    stdx.debug.assert(ids.dim(.token) == hidden.dim(.token) and routing_weights.dim(.token) == hidden.dim(.token), "routing ids and weights must match hidden token count {}, got {} and {}", .{ hidden.dim(.token), ids.dim(.token), routing_weights.dim(.token) });
    stdx.debug.assert(ids.dim(.topk) == routing_weights.dim(.topk), "routing ids and weights must have matching top-k dimensions, got {} and {}", .{ ids.dim(.topk), routing_weights.dim(.topk) });
    stdx.debug.assert(gate_up.dim(.expert) == down.dim(.expert), "gate/up and down expert counts must match, got {} and {}", .{ gate_up.dim(.expert), down.dim(.expert) });
    if (opts.expert_map) |expert_map|
        stdx.debug.assert(expert_map.dtype() == .i32 and expert_map.rank() == 1, "expected a rank-1 I32 expert map, got rank {} and dtype {}", .{ expert_map.rank(), expert_map.dtype() });

    const num_experts = if (opts.expert_map) |expert_map| expert_map.dim(.expert) else gate_up.dim(.expert);
    const routing = prepareRouting(ids, num_experts, @intCast(launch_config.block_size_m));

    const expert_ids = if (opts.expert_map) |expert_map|
        expert_map.gather(.{ .expert = routing.expert_ids }, .{}).withTags(.{.g})
    else
        routing.expert_ids;

    var hidden_quant = hidden;
    var input_scale: ?Tensor = null;

    if (opts.quantize_input and gate_up_scheme != .mxfp4) {
        if (gate_up_scheme) |scheme| hidden_quant, input_scale = quantizeFp8Input(hidden, scheme, gate_up.dtype());
    }

    const gate_up_out = callFusedMoe(.{
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

    var activated = applyExpertActivation(gate_up_out, opts.activation, opts.activation_threshold, opts.gate_up_layout);
    if (opts.routing_weight_placement == .before_down) {
        const weights = routing_weights.reshape(.{ .token = routing.num_assignments }).convert(.f32);
        activated = activated.mul(weights.broad(activated.shape()));
    }
    var activated_quant = activated.convert(.bf16);
    input_scale = null;
    if (opts.quantize_input and down_scheme != .mxfp4) {
        if (down_scheme) |scheme| activated_quant, input_scale = quantizeFp8Input(activated, scheme, down.dtype());
    }

    const down_out = callFusedMoe(.{
        .input = activated_quant,
        .weight = down,
        .bias = opts.down.bias,
        .input_scale = input_scale,
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

fn applyExpertActivation(input: Tensor, mode: Parameters.ActivationMode, activation_threshold: ?f32, layout: GateUpLayout) Tensor {
    const x = input.convert(.f32);
    if (mode == .relu) {
        const clipped = if (activation_threshold) |limit| x.minimum(Tensor.scalar(limit, x.dtype())) else x;
        return clipped.relu().powByConst(2);
    }

    const mid = @divFloor(x.dim(.out), 2);
    var gate, var up = switch (layout) {
        .split => .{ x.slice(.out, .{ .end = mid }), x.slice(.out, .{ .start = mid }) },
        .interleaved => .{ x.slice(.out, .{ .start = 0, .step = 2 }), x.slice(.out, .{ .start = 1, .step = 2 }) },
    };
    if (activation_threshold) |limit_| {
        const limit = Tensor.scalar(limit_, x.dtype());
        gate = gate.minimum(limit);
        up = up.clamp(limit.negate(), limit);
    }
    return switch (mode) {
        .silu => gate.silu().mul(up),
        .relu => unreachable,
        .gelu => gate.gelu().mul(up),
    };
}

test "SwiGLU uses FP32 math for split and interleaved BF16 inputs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: Tensor, layout: GateUpLayout, threshold: ?f32) Tensor {
            return applyExpertActivation(x, .silu, threshold, layout);
        }
    };
    const x: Tensor = .init(.{ .token = 1, .out = 6 }, .bf16);
    const gates = [_]f32{ 0.75, -1.25, 3.5 };
    const ups = [_]f32{ 0.875, -2.25, 4.5 };
    for ([_]GateUpLayout{ .split, .interleaved }) |layout| {
        var values: [6]zml.floats.BFloat16 = undefined;
        for (gates, ups, 0..) |gate, up, i| {
            values[if (layout == .split) i else 2 * i] = .fromF32(gate);
            values[if (layout == .split) i + 3 else 2 * i + 1] = .fromF32(up);
        }
        var input = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, std.mem.asBytes(&values));
        defer input.deinit();
        for ([_]?f32{ null, 2 }) |threshold| {
            var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, layout, threshold }, .{});
            defer exe.deinit();
            try zml.testing.expectEqualShapes(Shape.init(.{ .token = 1, .out = 3 }, .f32), exe.output_shapes[0]);
            var output = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{input});
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

test "ReLU squared activation preserves width and applies threshold before squaring" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: Tensor, threshold: ?f32) Tensor {
            return applyExpertActivation(x, .relu, threshold, .split);
        }
    };
    const x: Tensor = .init(.{ .token = 1, .out = 5 }, .f32);
    const values = [_]f32{ -3, 0, 1, 2, 4 };
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, std.mem.asBytes(&values));
    defer input.deinit();
    for ([_]?f32{ null, 2 }) |threshold| {
        var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, threshold }, .{});
        defer exe.deinit();
        try zml.testing.expectEqualShapes(x.shape(), exe.output_shapes[0]);
        var output = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{input});
        defer output.deinit();
        var actual = try output.toSliceAlloc(allocator, io);
        defer actual.free(allocator);
        const expected = [_]f32{ 0, 0, 1, 4, if (threshold != null) 4 else 16 };
        try std.testing.expectEqualSlices(f32, &expected, actual.constItems(f32));
    }
}

/// Build the inputs tuple for FusedMoe and invoke it via `K.call(...)`.
fn callFusedMoe(opts: struct {
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
}) Tensor {
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
                input, input_scale = quantizeFp8Input(x, scheme, .f8e4m3fn);
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
            const actual = callFusedMoe(.{
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
                    buffers[i] = try zml.Buffer.fromSlice(io, platform, slice, .replicated);
                    initialized += 1;
                }
                var output = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{ buffers[0], buffers[1], buffers[2], buffers[3], buffers[4], buffers[5] });
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
        fn forward(x: Tensor, layout: GateUpLayout, placement: RoutingWeightPlacement, storage_dtype: DataType) Tensor {
            const fp4 = storage_dtype != .bf16;
            const columns = Tensor.arange(.{ .end = 256 }, .i32).withTags(.{.dout});
            const is_gate = switch (layout) {
                .split => columns.cmp(.LT, Tensor.scalar(128, .i32)),
                .interleaved => columns.remainder(Tensor.scalar(2, .i32)).cmp(.EQ, Tensor.scalar(0, .i32)),
            };
            const dtype: DataType = if (fp4) .u8 else .bf16;
            // E2M1 codes 1, 2, 3 encode 0.5, 1, 1.5; each byte holds two values.
            const gate_up_values = is_gate.select(Tensor.scalar(@as(f32, if (fp4) 0x11 else 0.5), dtype), Tensor.scalar(@as(f32, if (fp4) 0x33 else 1.5), dtype));
            var gate_up: zml.nn.Linear = .{
                .weight = gate_up_values.broad(Shape.init(.{ .expert = 8, .dout = 256, .d = @as(i64, if (fp4) 64 else 128) }, dtype)),
                .tag = Shape.toTag(.d),
                .quantization = if (fp4) .{
                    .scheme = .mxfp4,
                    .scales = Tensor.scalar(127, .u8).broad(Shape.init(.{ .expert = 8, .dout = 256, .d = 4 }, .u8)),
                } else null,
            };
            var down: zml.nn.Linear = .{
                .weight = Tensor.scalar(@as(f32, if (fp4) 0x22 else 1), dtype).broad(Shape.init(.{ .expert = 8, .d = 128, .dout = @as(i64, if (fp4) 64 else 128) }, dtype)),
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
            const route = Tensor.arange(.{ .end = 2 }, .i32).reshape(.{ .b = 1, .s = 1, .top_expert = 2 })
                .broad(Shape.init(.{ .b = 1, .s = x.dim(.s), .top_expert = 2 }, .i32));
            return fusedExpertsImpl(.{
                .hidden_states = x,
                .gate_up = gate_up,
                .down = down,
                .topk_ids = route.addConstant(2),
                .topk_weights = route.convert(.f32).addConstant(1).scale(0.25),
                .gate_up_layout = layout,
                .routing_weight_placement = placement,
                .quantize_input = false,
            }) catch unreachable;
        }
    };
    for ([_]DataType{ .bf16, .u8, .f4e2m1 }) |storage_dtype| {
        const tokens: i64 = if (storage_dtype == .u8) 17 else 1;
        const x: Tensor = .init(.{ .b = 1, .s = tokens, .d = 128 }, .bf16);
        const host = try zml.Slice.alloc(allocator, x.shape());
        defer host.free(allocator);
        @memset(host.items(zml.floats.BFloat16), .fromF32(1.0 / 128.0));
        var input = try zml.Buffer.fromSlice(io, platform, host, .replicated);
        defer input.deinit();
        for ([_]GateUpLayout{ .split, .interleaved }) |layout| {
            for ([_]RoutingWeightPlacement{ .before_down, .after_down }) |placement| {
                var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, layout, placement, storage_dtype }, .{});
                defer exe.deinit();
                var output = try zml.testing.autoCall(allocator, io, &exe, Local.forward, .{input});
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

const Routing = struct {
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    max_num_tokens_padded: i64,
    num_assignments: i64,
    naive_block_assignment: bool,
};

fn prepareRouting(topk_ids: Tensor, num_experts: i64, block_size_m: i64) Routing {
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
    log.debug("Using triton kernels to sort and align tokens to experts with block size {d}", .{block_size_m});
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

fn quantizeFp8Input(x: Tensor, scheme: zml.Quantization.Scheme, output_dtype: DataType) struct { Tensor, Tensor } {
    const group_size: i64, const scale_dtype: DataType, const absmax_epsilon: f32 = switch (scheme) {
        .mxfp8 => .{ 32, .u8, 1e-10 },
        .fp8_per_channel, .fp8_per_tensor => .{ x.dim(1), .f32, 1e-10 },
        .fp8_block128 => .{ 128, .f32, 1e-10 },
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
