const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const tri = zml.kernel.triton;
const toDType = tri.from;
const a16w4_kernel = @import("triton_kernels/a16w4_kernel.zig");
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
        .mxfp4 => {
            if (down_scheme != .mxfp4) return error.UnsupportedQuantization;
            const local_topk_ids, const local_topk_weights = if (opts.expert_map) |expert_map| blk: {
                const local_num_experts = opts.gate_up.weight.dim(.expert);
                const mapped_ids = expert_map
                    .gather(.{ .expert = topk_ids.convert(.i32) }, .{})
                    .withTags(topk_ids.shape().tags());
                const in_range = mapped_ids.cmp(.GE, Tensor.scalar(0, .i32))
                    .logical(.AND, mapped_ids.cmp(.LT, Tensor.scalar(local_num_experts, .i32)));

                break :blk .{
                    in_range.select(mapped_ids, Tensor.scalar(local_num_experts, .i32)),
                    in_range.select(topk_weights, Tensor.scalar(0, topk_weights.dtype())),
                };
            } else .{ topk_ids, topk_weights };

            var local_args = opts;
            local_args.topk_ids = local_topk_ids;
            local_args.topk_weights = local_topk_weights;
            return fusedExpertsImpl_fp4(local_args);
        },
        .nvfp4 => return error.UnsupportedQuantization,
        .fp8_block128 => launch_config.block_size_k = 128,
        .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
    };

    var down_launch_config = launchConfigForTokens(num_tokens);
    if (down_scheme) |scheme| switch (scheme) {
        .fp8_block128 => down_launch_config.block_size_k = 128,
        .mxfp8, .fp8_per_channel, .fp8_per_tensor => {},
        .mxfp4, .nvfp4 => return error.UnsupportedQuantization,
    };

    const hidden = hidden_states.reshape(.{ .token = num_tokens, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = opts.gate_up.weight.withTags(.{ .expert, .out, .in });
    const down = opts.down.weight.withTags(.{ .expert, .out, .mid });
    const routing_weights = topk_weights.reshape(.{ .token = num_tokens, .in = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = num_tokens, .in = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    stdx.debug.assert(hidden.dtype() == .bf16, "expected BF16 hidden states, got {}", .{hidden.dtype()});
    stdx.debug.assert(gate_up.dtype() == .bf16 or gate_up.dtype() == .f8e4m3fn or gate_up.dtype() == .f8e4m3fnuz, "expected BF16 or FP8 E4M3FN/FNUZ gate/up weights, got {}", .{gate_up.dtype()});
    stdx.debug.assert(down.dtype() == .bf16 or down.dtype() == .f8e4m3fn or down.dtype() == .f8e4m3fnuz, "expected BF16 or FP8 E4M3FN/FNUZ down weights, got {}", .{down.dtype()});
    stdx.debug.assert(routing_weights.dtype() == .f32 or routing_weights.dtype() == .bf16, "expected FP32 or BF16 routing weights, got {}", .{routing_weights.dtype()});
    stdx.debug.assert(ids.dtype() == .i32, "expected I32 expert ids, got {}", .{ids.dtype()});
    stdx.debug.assert(hidden.dim(.in) == gate_up.dim(.in), "hidden width {} must match gate/up input width {}", .{ hidden.dim(.in), gate_up.dim(.in) });
    const activation_reduction: i64 = if (opts.activation == .relu) 1 else 2;
    stdx.debug.assert(@rem(gate_up.dim(.out), activation_reduction) == 0, "gate/up output width {} must be divisible by {}", .{ gate_up.dim(.out), activation_reduction });
    stdx.debug.assert(down.dim(.mid) == @divFloor(gate_up.dim(.out), activation_reduction), "down input width {} must match activated width {}", .{ down.dim(.mid), @divFloor(gate_up.dim(.out), activation_reduction) });
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

    if (opts.quantize_input) {
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

    const activated = applyExpertActivation(gate_up_out, opts.activation, opts.activation_threshold);
    var activated_quant = activated.convert(.bf16);
    input_scale = null;
    if (opts.quantize_input) {
        if (down_scheme) |scheme| activated_quant, input_scale = quantizeFp8Input(activated, scheme, down.dtype());
    }

    const down_out = callFusedMoe(.{
        .input = activated_quant,
        .weight = down,
        .bias = opts.down.bias,
        .input_scale = input_scale,
        .weight_scale = opts.down.quantizationScales(),
        .routing_weights = routing_weights,
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

fn applyExpertActivation(input: Tensor, mode: Parameters.ActivationMode, activation_threshold: ?f32) Tensor {
    const x = input.convert(.f32);
    if (mode == .relu) {
        const clipped = if (activation_threshold) |limit| x.minimum(Tensor.scalar(limit, x.dtype())) else x;
        return clipped.relu().powByConst(2);
    }

    const mid = @divFloor(x.dim(.out), 2);
    var gate = x.slice(.out, .{ .end = mid });
    var up = x.slice(.out, .{ .start = mid });
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

test "SwiGLU uses FP32 math for BF16 inputs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: Tensor, threshold: ?f32) Tensor {
            return applyExpertActivation(x, .silu, threshold);
        }
    };
    const x: Tensor = .init(.{ .token = 1, .out = 6 }, .bf16);
    const gates = [_]f32{ 0.75, -1.25, 3.5 };
    const ups = [_]f32{ 0.875, -2.25, 4.5 };
    var values: [6]zml.floats.BFloat16 = undefined;
    for (gates, ups, 0..) |gate, up, i| {
        values[i] = .fromF32(gate);
        values[i + 3] = .fromF32(up);
    }
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), .replicated, std.mem.asBytes(&values));
    defer input.deinit();
    for ([_]?f32{ null, 2 }) |threshold| {
        var exe = try platform.compileFn(allocator, io, Local.forward, .{ x, threshold }, .{});
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

test "ReLU squared activation preserves width and applies threshold before squaring" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Local = struct {
        fn forward(x: Tensor, threshold: ?f32) Tensor {
            return applyExpertActivation(x, .relu, threshold);
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
    stdx.debug.assert(opts.quant_scheme != null or (opts.input_scale == null and opts.weight_scale == null), "scales require a quantization scheme", .{});
    for ([_]i64{ opts.weight.dim(1), opts.weight.dim(2), opts.input.dim(1), opts.weight.dim(1) * opts.weight.dim(2), opts.output_shape.dim(-1) }) |dim_or_stride| {
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
        (std.math.divCeil(i64, opts.weight.dim(1), block_size_n) catch unreachable);

    const weight_scale: ?Tensor = if (opts.weight_scale) |scale| blk: {
        const scheme = opts.quant_scheme orelse break :blk scale;
        break :blk switch (scheme) {
            // dot_scaled takes E8M0 encodings as bytes, including native E8M0 tensors.
            .mxfp8 => scale.bitCast(.u8),
            .fp8_per_channel => scale.reshape(.{ opts.weight.dim(0), opts.weight.dim(1), 1 }),
            .fp8_per_tensor => if (scale.count() == 1) scale else scale.reshape(.{ opts.weight.dim(0), 1, 1 }),
            .fp8_block128, .mxfp4, .nvfp4 => scale,
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
            .b_ptr = opts.weight,
            .b_bias_ptr = opts.bias orelse Tensor.scalar(0, .f32),
            .a_scale_ptr = opts.input_scale orelse Tensor.scalar(1.0, .f32),
            .b_scale_ptr = weight_scale orelse Tensor.scalar(1.0, .f32),
            .routing_weights_ptr = opts.routing_weights orelse Tensor.scalar(1.0, .f32),
            .sorted_token_ids_ptr = opts.routing.sorted_token_ids,
            .expert_ids_ptr = opts.expert_ids,
            .num_tokens_post_padded_ptr = opts.routing.num_tokens_post_padded,
            .N_ptr = Tensor.constant(.{ .i64 = opts.weight.dim(1) }).reshape(.{1}),
            .K_ptr = Tensor.constant(.{ .i64 = opts.weight.dim(2) }).reshape(.{1}),
            .EM_ptr = Tensor.constant(.{ .i64 = em_effective }).reshape(.{1}),
            .num_valid_tokens_ptr = Tensor.constant(.{ .i64 = opts.routing.num_assignments }).reshape(.{1}),
            .stride_am_ptr = Tensor.constant(.{ .i64 = opts.input.dim(1) }).reshape(.{1}),
            .stride_be_ptr = Tensor.constant(.{ .i64 = opts.weight.dim(1) * opts.weight.dim(2) }).reshape(.{1}),
            .stride_bn_ptr = Tensor.constant(.{ .i64 = opts.weight.dim(2) }).reshape(.{1}),
            .stride_cm_ptr = Tensor.constant(.{ .i64 = opts.weight.dim(.out) }).reshape(.{1}),
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
                .b_dtype = toDType(opts.weight.dtype()),
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

// =====
// A16W4
// =====
pub fn fusedExpertsImpl_fp4(args: FusedExpertsArgs) !zml.Tensor {
    const input = args.hidden_states;
    const topk_ids = args.topk_ids;
    const topk_weights = args.topk_weights;
    const x = input.reshape(.{
        .token = @divExact(@as(i64, @intCast(input.count())), input.dim(.d)),
        .d = input.dim(.d),
    });
    const num_tokens = x.dim(.token);
    const num_routes: i64 = @intCast(topk_ids.count());
    stdx.debug.assert(@mod(num_routes, num_tokens) == 0, "expected {} routing ids to be divisible by {} tokens", .{ num_routes, num_tokens });
    stdx.debug.assert(topk_weights.count() == topk_ids.count(), "expected matching routing id and weight counts, got {} and {}", .{ topk_ids.count(), topk_weights.count() });
    const topk = @divExact(num_routes, num_tokens);
    const flat_topk_ids = topk_ids.reshape(.{ .token = num_tokens, .topk = topk });
    const flat_topk_weights = topk_weights.reshape(.{ .token = num_tokens, .topk = topk });
    const kernel_cfg = getBestConfig(
        @intCast(num_tokens),
        @intCast(topk),
        @intCast(args.gate_up.weight.dim(.expert)),
    );
    const num_experts = args.gate_up.weight.dim(.expert);
    const aligned_routing = prepareRouting(flat_topk_ids, num_experts, @intCast(kernel_cfg.block_m));
    const routing = prepareFp4Routing(
        aligned_routing,
        flat_topk_ids,
        flat_topk_weights,
        num_tokens,
        num_experts,
        @intCast(kernel_cfg.block_m),
    );

    const hidden_shape: zml.Shape = .init(.{
        .route = routing.num_rows,
        .dout = @divExact(args.gate_up.weight.dim(.dout), 2),
    }, .bf16);

    const hidden = try runGemm(
        x,
        args.gate_up.weight,
        args.gate_up.quantization.?.scales,
        .{
            .routing = routing,
            .weight_contract_tag = zml.Shape.toTag(.d),
            .weight_output_tag = zml.Shape.toTag(.dout),
            .output_shape = hidden_shape,
            .gather = routing.sorted_route_ids,
            .gammas = routing.sorted_weights,
            .bias = args.gate_up.bias,
            .apply_swiglu = true,
            .activation_limit = args.activation_threshold,
            .block_m = kernel_cfg.block_m,
            .block_n = kernel_cfg.block_n,
            .block_k = kernel_cfg.block_k,
            .group_m = kernel_cfg.group_m,
            .num_warps = kernel_cfg.num_warps,
            .num_stages = kernel_cfg.num_stages,
        },
    );

    const routed_shape: zml.Shape = .init(.{
        .route = routing.num_rows,
        .d = args.down.weight.dim(.d),
    }, .bf16);

    const routed = try runGemm(
        hidden,
        args.down.weight,
        args.down.quantization.?.scales,
        .{
            .routing = routing,
            .weight_contract_tag = zml.Shape.toTag(.dout),
            .weight_output_tag = zml.Shape.toTag(.d),
            .output_shape = routed_shape,
            .bias = args.down.bias,
            .apply_swiglu = false,
            .activation_limit = 1.0,
            .block_m = kernel_cfg.block_m,
            .block_n = kernel_cfg.block_n,
            .block_k = kernel_cfg.block_k,
            .group_m = kernel_cfg.group_m,
            .num_warps = kernel_cfg.num_warps,
            .num_stages = kernel_cfg.num_stages,
        },
    );

    const active_routed = routing.active_routes.broad(routed.shape().withDtype(.bool)).select(
        routed,
        zml.Tensor.zeroes(routed.shape()),
    );
    const token_ids = routing.sorted_route_ids.divByConst(routing.topk).withTags(.{.route});
    const output_flat_shape: zml.Shape = .init(.{ .token = routing.num_tokens, .d = input.dim(.d) }, .f32);
    const output_flat = zml.Tensor.zeroes(output_flat_shape).scatterSlices(
        .{ .token = token_ids },
        active_routed.convert(.f32),
        .{},
    );

    return output_flat.reshape(input.shape().withDtype(.f32)).convert(input.dtype());
}

const KernelConf = struct {
    block_m: u32,
    block_n: u32,
    block_k: u32,
    group_m: u32,
    num_warps: u32,
    num_stages: u32,
};

const kernel_config_token_buckets = [_]u32{
    1,  2,   4,   8,   16,   24,   32,   48,   64,
    96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
};

fn configForTokenBucket(num_tokens: u32) KernelConf {
    return switch (num_tokens) {
        1 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 4,
        },
        2 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 4,
        },
        4 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 3,
        },
        8 => .{
            .block_m = 16,
            .block_n = 128,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 3,
        },
        16 => .{
            .block_m = 16,
            .block_n = 64,
            .block_k = 64,
            .group_m = 64,
            .num_warps = 4,
            .num_stages = 5,
        },
        24 => .{
            .block_m = 16,
            .block_n = 64,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 2,
        },
        32 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 2,
        },
        48 => .{
            .block_m = 16,
            .block_n = 32,
            .block_k = 128,
            .group_m = 64,
            .num_warps = 4,
            .num_stages = 2,
        },
        64 => .{
            .block_m = 16,
            .block_n = 64,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 2,
        },
        96 => .{
            .block_m = 16,
            .block_n = 128,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 3,
        },
        128 => .{
            .block_m = 16,
            .block_n = 256,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 2,
        },
        256 => .{
            .block_m = 16,
            .block_n = 256,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 2,
        },
        512 => .{
            .block_m = 32,
            .block_n = 128,
            .block_k = 128,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 3,
        },
        1024 => .{
            .block_m = 64,
            .block_n = 128,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 3,
        },
        1536 => .{
            .block_m = 64,
            .block_n = 128,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 4,
            .num_stages = 3,
        },
        2048 => .{
            .block_m = 128,
            .block_n = 128,
            .block_k = 64,
            .group_m = 16,
            .num_warps = 8,
            .num_stages = 3,
        },
        3072 => .{
            .block_m = 128,
            .block_n = 256,
            .block_k = 64,
            .group_m = 1,
            .num_warps = 8,
            .num_stages = 4,
        },
        4096 => .{
            .block_m = 128,
            .block_n = 256,
            .block_k = 64,
            .group_m = 16,
            .num_warps = 8,
            .num_stages = 4,
        },
        else => unreachable,
    };
}

fn getBestConfig(num_tokens: u32, topk: u32, num_experts: u32) KernelConf {
    const num_routes = std.math.mul(u32, num_tokens, topk) catch std.math.maxInt(u32);
    var config = getBestTokenBucketConfig(num_routes);

    if (num_tokens <= 32 and num_routes <= 256 and num_experts <= 64) {
        config.block_m = 16;
        config.block_n = 256;
        config.block_k = 128;
        config.group_m = 1;
        config.num_warps = 4;
        config.num_stages = 2;
    } else if (num_tokens <= 64 and num_routes <= 512 and num_experts <= 64) {
        config.block_m = 16;
        config.block_n = 128;
        config.block_k = 128;
        config.group_m = 1;
        config.num_warps = 4;
        config.num_stages = 2;
    }

    return config;
}

fn getBestTokenBucketConfig(num_tokens: u32) KernelConf {
    var best_num_tokens = kernel_config_token_buckets[0];
    var best_distance = tokenDistance(num_tokens, best_num_tokens);

    for (kernel_config_token_buckets[1..]) |candidate| {
        const distance = tokenDistance(num_tokens, candidate);
        if (distance < best_distance or (distance == best_distance and candidate < best_num_tokens)) {
            best_num_tokens = candidate;
            best_distance = distance;
        }
    }

    return configForTokenBucket(best_num_tokens);
}

fn tokenDistance(a: u32, b: u32) u32 {
    return if (a >= b) a - b else b - a;
}

const Fp4Routing = struct {
    num_tokens: i64,
    num_rows: i64,
    topk: i64,
    gather_divisor: i64,
    grid_m: i64,
    sorted_route_ids: zml.Tensor,
    sorted_weights: zml.Tensor,
    active_routes: zml.Tensor,
    tile_experts: zml.Tensor,
    tile_starts: zml.Tensor,
    tile_ends: zml.Tensor,
};

fn prepareFp4Routing(
    aligned: Routing,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    num_tokens: i64,
    num_experts: i64,
    block_m: i64,
) Fp4Routing {
    const topk = topk_ids.dim(.topk);
    const num_routes = aligned.num_assignments;
    const num_rows = if (aligned.naive_block_assignment)
        num_routes
    else
        aligned.max_num_tokens_padded;
    const grid_m = if (aligned.naive_block_assignment)
        num_routes
    else
        std.math.divCeil(i64, num_rows, block_m) catch unreachable;

    const sorted_route_candidates = if (aligned.naive_block_assignment)
        zml.Tensor.arange(.{ .end = num_routes }, .i32).withTags(.{.route})
    else
        aligned.sorted_token_ids.withTags(.{.route});
    const route_index_valid = sorted_route_candidates.cmp(.GE, zml.Tensor.scalar(0, .i32))
        .logical(.AND, sorted_route_candidates.cmp(.LT, zml.Tensor.scalar(num_routes, .i32)));
    const sorted_route_ids = route_index_valid.select(
        sorted_route_candidates,
        zml.Tensor.zeroes(sorted_route_candidates.shape()),
    );
    const gather_indices = sorted_route_ids.rename(.{ .route = .sorted_route });

    const flat_expert_ids = topk_ids.flatten().withTags(.{.route}).convert(.i32);
    const sorted_expert_ids = flat_expert_ids
        .gather(.{ .route = gather_indices }, .{})
        .rename(.{ .sorted_route = .route });
    const active_routes = route_index_valid
        .logical(.AND, sorted_expert_ids.cmp(.GE, zml.Tensor.scalar(0, .i32)))
        .logical(.AND, sorted_expert_ids.cmp(.LT, zml.Tensor.scalar(num_experts, .i32)));

    const flat_weights = topk_weights.flatten().withTags(.{.route});
    const gathered_weights = flat_weights
        .gather(.{ .route = gather_indices }, .{})
        .rename(.{ .sorted_route = .route })
        .convert(.f32);
    const sorted_weights = active_routes.select(gathered_weights, zml.Tensor.zeroes(gathered_weights.shape()));

    const raw_tile_experts = aligned.expert_ids.withTags(.{.tile}).convert(.i32);
    const valid_tile_experts = raw_tile_experts.cmp(.GE, zml.Tensor.scalar(0, .i32))
        .logical(.AND, raw_tile_experts.cmp(.LT, zml.Tensor.scalar(num_experts, .i32)));
    const tile_experts = valid_tile_experts.select(
        raw_tile_experts,
        zml.Tensor.zeroes(raw_tile_experts.shape()),
    );
    const tile_starts = if (aligned.naive_block_assignment)
        zml.Tensor.arange(.{ .end = grid_m }, .i64).withTags(.{.tile})
    else
        zml.Tensor.arange(.{ .end = grid_m }, .i64).withTags(.{.tile}).scale(block_m);
    const tile_ends = if (aligned.naive_block_assignment)
        valid_tile_experts.select(tile_starts.addConstant(1), tile_starts)
    else blk: {
        const num_tokens_post_padded = aligned.num_tokens_post_padded
            .withTags(.{.tile})
            .convert(.i64)
            .broad(tile_starts.shape());
        const active_tiles = valid_tile_experts.logical(.AND, tile_starts.cmp(.LT, num_tokens_post_padded));
        break :blk active_tiles.select(
            tile_starts.addConstant(block_m).minimum(num_tokens_post_padded),
            tile_starts,
        );
    };

    return .{
        .num_tokens = num_tokens,
        .num_rows = num_rows,
        .topk = topk,
        .gather_divisor = topk,
        .grid_m = grid_m,
        .sorted_route_ids = sorted_route_ids,
        .sorted_weights = sorted_weights,
        .active_routes = active_routes,
        .tile_experts = tile_experts,
        .tile_starts = tile_starts,
        .tile_ends = tile_ends,
    };
}

const GemmOpts = struct {
    routing: Fp4Routing,
    weight_contract_tag: zml.Shape.Tag,
    weight_output_tag: zml.Shape.Tag,
    output_shape: zml.Shape,
    gather: ?zml.Tensor = null,
    gammas: ?zml.Tensor = null,
    bias: ?zml.Tensor = null,
    apply_swiglu: bool = false,
    activation_limit: ?f32 = null,
    block_m: u32,
    block_n: u32,
    block_k: u32,
    group_m: u32,
    num_warps: u32,
    num_stages: u32,
};
fn runGemm(
    input: zml.Tensor,
    weights: zml.Tensor,
    scales: zml.Tensor,
    opts: GemmOpts,
) !zml.Tensor {
    const input_matrix = input.withTags(.{ .row, .k });
    const contract_k = input_matrix.dim(.k);
    const packed_k = weights.dim(opts.weight_contract_tag);
    const scale_k = scales.dim(opts.weight_contract_tag);
    const n = weights.dim(opts.weight_output_tag);

    stdx.debug.assert(packed_k * 2 == contract_k, "expected packed int4 weight K {} to match activation K {}", .{ packed_k, contract_k });
    stdx.debug.assert(scale_k * 32 == contract_k, "expected MX scale K {} to match activation K {}", .{ scale_k, contract_k });
    const activation_reduction_n: i64 = if (opts.apply_swiglu) 2 else 1;
    stdx.debug.assert(@mod(n, activation_reduction_n) == 0, "invalid GEMM output width {}", .{n});
    stdx.debug.assert(opts.output_shape.dim(-1) == @divExact(n, activation_reduction_n), "output shape {f} does not match GEMM N {}", .{ opts.output_shape, n });
    stdx.debug.assert(opts.bias == null, "MXFP4 Triton MoE GEMM bias is not wired yet", .{});

    const block_m: i32 = @intCast(opts.block_m);
    const block_n: i32 = @intCast(opts.block_n);
    const block_k: i32 = @intCast(opts.block_k);
    // TODO: update the kernel to support uneven K.
    if (@mod(contract_k, block_k) != 0) return error.InvalidShape;
    const grid_n = std.math.divCeil(i64, n, block_n) catch unreachable;
    const has_gammas = opts.gammas != null;
    const gathered_input = if (opts.gather) |gather| blk: {
        const token_ids = gather.divByConst(opts.routing.gather_divisor).withTags(.{.route});
        break :blk input_matrix.gather(.{ .row = token_ids }, .{}).rename(.{ .route = .row });
    } else input_matrix;
    const raw_output_shape = if (opts.apply_swiglu)
        opts.output_shape.set(-1, n)
    else
        opts.output_shape;

    const cfg: a16w4_kernel.Cfg = .{
        .a_dtype = zml.kernel.triton.from(gathered_input.dtype()),
        .wp_dtype = packedByteDtype(weights.dtype()),
        .ws_dtype = packedByteDtype(scales.dtype()),
        .c_dtype = zml.kernel.triton.from(raw_output_shape.dtype()),
        .BLOCK_M = block_m,
        .BLOCK_N = block_n,
        .BLOCK_K = block_k,
        .SPLIT_K = 1,
        .GROUP_M = @intCast(opts.group_m),
        .num_warps = @intCast(opts.num_warps),
        .num_stages = @intCast(opts.num_stages),
    };

    var y = a16w4_kernel.Kernel.call(
        .{
            .a_ptr = gathered_input,
            .wp_ptr = weights,
            .ws_ptr = scales,
            .tile_expert_ptr = opts.routing.tile_experts,
            .tile_mstart_ptr = opts.routing.tile_starts,
            .tile_mend_ptr = opts.routing.tile_ends,
            .NUM_M_TILES_ptr = scalarI64(opts.routing.grid_m),
            .N_ptr = scalarI64(n),
            .K_ptr = scalarI64(contract_k),
            .stride_am_ptr = scalarI64(contract_k),
            .stride_ak_ptr = scalarI64(1),
            .stride_we_ptr = scalarI64(n * packed_k),
            .stride_wk_ptr = scalarI64(1),
            .stride_wn_ptr = scalarI64(packed_k),
            .stride_se_ptr = scalarI64(n * scale_k),
            .stride_sk_ptr = scalarI64(1),
            .stride_sn_ptr = scalarI64(scale_k),
            .stride_cm_ptr = scalarI64(raw_output_shape.dim(-1)),
            .stride_cn_ptr = scalarI64(1),
        },
        .{ .c = raw_output_shape },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(opts.routing.grid_m * grid_n), 1, 1 },
            .num_warps = @intCast(opts.num_warps),
            .num_stages = @intCast(opts.num_stages),
        },
    ).c;

    if (opts.apply_swiglu) {
        y = applySwiGlu(y.convert(.f32), opts.activation_limit).convert(opts.output_shape.dtype());
    }

    if (has_gammas) {
        const gammas = opts.gammas.?.convert(.f32).appendAxes(.{.dout}).broad(opts.output_shape.withDtype(.f32));
        y = y.convert(.f32).mul(gammas).convert(opts.output_shape.dtype());
    }

    return y;
}

fn applySwiGlu(input: zml.Tensor, activation_limit: ?f32) zml.Tensor {
    var gate = input.slice(.dout, .{ .start = 0, .step = 2 });
    var up = input.slice(.dout, .{ .start = 1, .step = 2 });

    if (activation_limit) |limit| {
        const threshold = zml.Tensor.scalar(limit, .f32);
        gate = gate.minimum(threshold);
        up = up.clamp(threshold.negate(), threshold);
    }

    return gate.silu().mul(up);
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
