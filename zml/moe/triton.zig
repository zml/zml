const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const tri = zml.kernel.triton;
const DType = tri.DType;
const toDType = tri.from;
const kernels = @import("triton_kernels/triton_kernels.zig");
const a16w4_kernel = @import("triton_kernels/a16w4_kernel.zig");

const log = std.log.scoped(.moe_triton);

// =============================================================================
// Public API
// =============================================================================

pub const Options = struct {
    inplace: bool = false,
    activation: Parameters.ActivationMode = .silu,
    apply_router_weight_on_input: bool = false,
    quant_scheme: ?zml.nn.QuantScheme = null,
    global_num_experts: i64 = -1,
    expert_map: ?Tensor = null,
    w1_scale: ?Tensor = null,
    w2_scale: ?Tensor = null,
    w1_zp: ?Tensor = null,
    w2_zp: ?Tensor = null,
    a1_scale: ?Tensor = null,
    a2_scale: ?Tensor = null,
    block_shape: ?[]const i64 = null,
    w1_bias: ?Tensor = null,
    w2_bias: ?Tensor = null,
    block_size_m: i64 = 16,
    block_size_n: i64 = 64,
    block_size_k: i64 = 32,
    group_size_m: i64 = 1,
    num_warps: i64 = 8,
    num_stages: i64 = 4,
    dynamic_launch_by_num_tokens: bool = true,
    activation_threshold: ?f32 = null,
};

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

pub const Metadata = struct {
    w1_zero_bias: ?Tensor = null,
    w2_zero_bias: ?Tensor = null,

    pub const InitOptions = struct {
        w1_zero_bias_shape: ?Shape = null,
        w2_zero_bias_shape: ?Shape = null,
    };

    pub fn init(opts: InitOptions) Metadata {
        return .{
            .w1_zero_bias = if (opts.w1_zero_bias_shape) |shape| Tensor.fromShape(shape) else null,
            .w2_zero_bias = if (opts.w2_zero_bias_shape) |shape| Tensor.fromShape(shape) else null,
        };
    }

    pub fn initBuffer(self: Metadata, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(Metadata) {
        const replicated_sharding = platform.replicated_sharding;
        return .{
            .w1_zero_bias = if (self.w1_zero_bias) |tensor| try initZeroBiasBuffer(io, platform, replicated_sharding, tensor.shape()) else null,
            .w2_zero_bias = if (self.w2_zero_bias) |tensor| try initZeroBiasBuffer(io, platform, replicated_sharding, tensor.shape()) else null,
        };
    }
};

pub fn deinitBuffer(bufferized: *zml.Bufferized(Metadata)) void {
    if (bufferized.w1_zero_bias) |*buffer| buffer.deinit();
    if (bufferized.w2_zero_bias) |*buffer| buffer.deinit();
}

fn initZeroBiasBuffer(io: std.Io, platform: *const zml.Platform, sharding: zml.Sharding, shape: Shape) !zml.Buffer {
    var zero_slice: zml.Slice = try .alloc(std.heap.c_allocator, shape);
    defer zero_slice.free(std.heap.c_allocator);
    @memset(zero_slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, zero_slice, sharding);
}

fn applyActivation(x: Tensor, mode: Parameters.ActivationMode) Tensor {
    const mid = @divFloor(x.dim(.out), 2);
    const gate = x.slice1d(.out, .{ .end = mid });
    const up = x.slice1d(.out, .{ .start = mid });
    return switch (mode) {
        .silu => gate.silu().mul(up),
        .relu => x.relu().powByConst(2),
        .gelu => gate.gelu().mul(up),
    };
}

// =============================================================================
// Top-level entry point
// =============================================================================

fn isMxFp8(quant_scheme: ?zml.nn.QuantScheme) bool {
    return quant_scheme != null and quant_scheme.? == .mxfp8;
}

pub fn fusedExpertsImpl(
    hidden_states: Tensor,
    w1: Tensor,
    w2: Tensor,
    topk_weights: Tensor,
    topk_ids: Tensor,
    metadata: Metadata,
    opts: Options,
) !Tensor {
    try validateOptions(opts);

    if (opts.quant_scheme != null and opts.quant_scheme.? == .mxfp4) {
        const local_topk_ids, const local_topk_weights = if (opts.expert_map) |expert_map| blk: {
            const local_num_experts = w1.dim(.expert);
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

        return fusedExpertsImpl_fp4(
            hidden_states,
            local_topk_ids,
            local_topk_weights,
            w1,
            opts.w1_scale.?,
            opts.w1_bias,
            w2,
            opts.w2_scale.?,
            opts.w2_bias,
            opts.activation_threshold,
        );
    }

    const options = applyJsonTokenConfig(opts, hidden_states.dim(0)) catch |err| fallback: {
        log.warn("Failed to load MoE launch config from JSON ({}), falling back to built-in token heuristic", .{err});
        break :fallback applyDefaultTokenConfig(opts, hidden_states.dim(0), w1.dim(0));
    };
    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);

    const hidden = hidden_states.reshape(.{ .token = b * s, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = w1.withTags(.{ .expert, .out, .in });
    const down = w2.withTags(.{ .expert, .out, .mid });
    const weights = topk_weights.reshape(.{ .token = b * s, .in = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = b * s, .in = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    try validateInputs(hidden, gate_up, down, weights, ids);

    const block_size_m = options.block_size_m;
    const num_experts = if (opts.global_num_experts != -1) opts.global_num_experts else gate_up.dim(.expert);
    if (opts.expert_map) |expert_map| {
        if (expert_map.dtype() != .i32) return error.UnsupportedType;
        if (expert_map.rank() != 1 or expert_map.dim(.expert) != num_experts) return error.InvalidShape;
    }
    const routing = prepareRouting(ids, num_experts, block_size_m);

    const expert_ids = if (opts.expert_map) |expert_map|
        expert_map.gather(.{ .expert = routing.expert_ids }, .{}).withTags(.{.g})
    else
        routing.expert_ids;

    var hidden_quant = hidden;
    var a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);

    if (gate_up.dtype() == .f8e4m3fn) {
        hidden_quant, a_scale = quantizePerTokenGroupFp8(hidden, fp8ActivationGroupSize(hidden));
    }

    const first_cfg = makeFusedMoeConfig(
        hidden_quant,
        gate_up,
        options,
        routing.naive_block_assignment,
        ids.dim(.topk),
        false,
        false,
        .bf16,
    );

    const b_bias_1 =
        opts.w1_bias orelse
        metadata.w1_zero_bias orelse
        Tensor.zeroes(Shape.init(.{ .expert = gate_up.dim(.expert), .out = gate_up.dim(.out) }, .bf16));

    const b_scale_1 = opts.w1_scale orelse Tensor.scalar(1.0, .f32);

    const first_out = callFusedMoe(
        hidden_quant,
        gate_up,
        b_bias_1,
        a_scale,
        b_scale_1,
        weights,
        routing.sorted_token_ids,
        expert_ids,
        routing.num_tokens_post_padded,
        first_cfg,
        options,
        routing.max_num_tokens_padded,
        routing.num_assignments,
        Shape.init(.{ .token = routing.num_assignments, .out = gate_up.dim(.out) }, .bf16),
    );

    const activated = applyActivation(first_out, options.activation);

    var activated_quant = activated;
    a_scale = opts.a2_scale orelse Tensor.scalar(1.0, .f32);
    if (down.dtype() == .f8e4m3fn) {
        activated_quant, a_scale = quantizePerTokenGroupFp8(activated, fp8ActivationGroupSize(activated));
    }

    const second_cfg = makeFusedMoeConfig(
        activated_quant,
        down,
        options,
        routing.naive_block_assignment,
        1,
        true,
        false,
        .bf16,
    );

    const b_bias_2 =
        opts.w2_bias orelse
        metadata.w2_zero_bias orelse
        Tensor.zeroes(Shape.init(.{ .expert = down.dim(.expert), .out = down.dim(.out) }, .bf16));

    const b_scale_2 = opts.w2_scale orelse Tensor.scalar(1.0, .f32);

    const second_out = callFusedMoe(
        activated_quant,
        down,
        b_bias_2,
        a_scale,
        b_scale_2,
        weights,
        routing.sorted_token_ids,
        expert_ids,
        routing.num_tokens_post_padded,
        second_cfg,
        options,
        routing.max_num_tokens_padded,
        routing.num_assignments,
        Shape.init(.{ .token = b * s, .topk = ids.dim(.topk), .out = down.dim(.out) }, .bf16),
    );

    const output = second_out.sum(.topk).squeeze(.topk);

    return output.reshape(.{ .b = b, .token = s, .out = down.dim(.out) });
}

/// Build the inputs tuple for FusedMoe and invoke it via `K.call(...)`.
fn callFusedMoe(
    a: Tensor,
    b: Tensor,
    b_bias: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    cfg: kernels.FusedMoe.Cfg,
    options: Options,
    max_num_tokens_padded: i64,
    num_valid_tokens: i64,
    output_shape: Shape,
) Tensor {
    const block_size_m: i64 = @intCast(cfg.block_size_m);
    const block_size_n: i64 = @intCast(cfg.block_size_n);
    const m_tokens = a.dim(0);
    const em_effective = if (m_tokens < block_size_m)
        @min(max_num_tokens_padded, num_valid_tokens * block_size_m)
    else
        max_num_tokens_padded;
    const grid_x =
        (std.math.divCeil(i64, em_effective, block_size_m) catch unreachable) *
        (std.math.divCeil(i64, b.dim(1), block_size_n) catch unreachable);

    const stride_asm: i64 = if (cfg.b_scale_dtype != null and a_scale.rank() == 2) a_scale.dim(1) else 0;
    const stride_ask: i64 = if (cfg.b_scale_dtype != null and a_scale.rank() == 2) 1 else 0;
    const stride_bse: i64 = if (cfg.b_scale_dtype != null and b_scale.rank() == 3)
        b_scale.dim(1) * b_scale.dim(2)
    else
        0;
    const stride_bsk: i64 = if (cfg.b_scale_dtype != null and b_scale.rank() == 3) 1 else 0;
    const stride_bsn: i64 = if (cfg.b_scale_dtype != null and b_scale.rank() == 3) b_scale.dim(2) else 0;

    return kernels.FusedMoe.Kernel.call(
        .{
            .a_ptr = a,
            .b_ptr = b,
            .b_bias_ptr = b_bias,
            .a_scale_ptr = a_scale,
            .b_scale_ptr = b_scale,
            .topk_weights_ptr = topk_weights,
            .sorted_token_ids_ptr = sorted_token_ids,
            .expert_ids_ptr = expert_ids,
            .num_tokens_post_padded_ptr = num_tokens_post_padded,
            .N_ptr = Tensor.constant(.{ .i64 = b.dim(1) }).reshape(.{1}),
            .K_ptr = Tensor.constant(.{ .i64 = b.dim(2) }).reshape(.{1}),
            .EM_ptr = Tensor.constant(.{ .i64 = em_effective }).reshape(.{1}),
            .num_valid_tokens_ptr = Tensor.constant(.{ .i64 = num_valid_tokens }).reshape(.{1}),
            .stride_am_ptr = Tensor.constant(.{ .i64 = a.dim(1) }).reshape(.{1}),
            .stride_be_ptr = Tensor.constant(.{ .i64 = b.dim(1) * b.dim(2) }).reshape(.{1}),
            .stride_bn_ptr = Tensor.constant(.{ .i64 = b.dim(2) }).reshape(.{1}),
            .stride_cm_ptr = Tensor.constant(.{ .i64 = b.dim(.out) }).reshape(.{1}),
            .stride_asm_ptr = Tensor.constant(.{ .i64 = stride_asm }).reshape(.{1}),
            .stride_ask_ptr = Tensor.constant(.{ .i64 = stride_ask }).reshape(.{1}),
            .stride_bse_ptr = Tensor.constant(.{ .i64 = stride_bse }).reshape(.{1}),
            .stride_bsk_ptr = Tensor.constant(.{ .i64 = stride_bsk }).reshape(.{1}),
            .stride_bsn_ptr = Tensor.constant(.{ .i64 = stride_bsn }).reshape(.{1}),
            .stride_bbe_ptr = Tensor.constant(.{ .i64 = 0 }).reshape(.{1}),
            .stride_bbn_ptr = Tensor.constant(.{ .i64 = 0 }).reshape(.{1}),
        },
        .{ .c = output_shape },
        .{
            .cfg = cfg,
            .grid = .{ @intCast(grid_x), 1, 1 },
            .num_warps = @intCast(options.num_warps),
            .num_stages = @intCast(options.num_stages),
        },
    ).c;
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
    const ids = topk_ids.withTags(.{ .token, .topk }).convert(.i32);
    const num_assignments = ids.dim(.token) * ids.dim(.topk);
    const sparsity_factor: i64 = 4;
    const naive_block_assignment = num_assignments * sparsity_factor <= num_experts;
    const max_num_tokens_padded = if (naive_block_assignment)
        num_assignments * block_size_m
    else if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        num_assignments + num_experts * (block_size_m - 1);

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
        num_assignments + num_experts * (block_size_m - 1);
    const max_num_m_blocks = std.math.divCeil(i64, max_num_tokens_padded, block_size_m) catch unreachable;
    const warp_size: i64 = 32;
    const padded_num_experts = (std.math.divCeil(i64, num_experts, warp_size) catch unreachable) * warp_size;
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

fn quantizePerTokenGroupFp8(x: Tensor, group_size: i64) struct { Tensor, Tensor } {
    stdx.debug.assert(x.rank() == 2, "expected a rank-2 activation matrix, got {f}", .{x.shape()});
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "activation width must be divisible by group size {d}, got {d}", .{ group_size, x.dim(1) });

    const groups_per_row = @divExact(x.dim(1), group_size);
    const quantized = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .feature = x.dim(1) }, .f8e4m3fn));
    const scales = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .group = groups_per_row }, .bf16));

    const outs = kernels.PerTokenGroupQuantFp8.Kernel.call(
        .{
            .y_ptr = x,
            .group_size_ptr = Tensor.constant(.{ .i64 = group_size }).reshape(.{1}),
            .y_num_columns_ptr = Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
            .y_row_stride_ptr = Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
            .eps_ptr = Tensor.scalar(1e-6, .f32),
        },
        .{ .y_q = quantized.shape(), .y_s = scales.shape() },
        .{
            .cfg = .{
                .input_dtype = toDType(x.dtype()),
                .output_dtype = .f8e4m3fn,
                .scale_dtype = .bf16,
                .block = @intCast(group_size),
                .fp8_min = -448.0,
                .fp8_max = 448.0,
                .use_ue8m0 = false,
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

fn makeFusedMoeConfig(
    a: Tensor,
    b: Tensor,
    opts: Options,
    naive_block_assignment: bool,
    top_k: i64,
    mul_routed_weight: bool,
    has_bias: bool,
    output_dtype: DataType,
) kernels.FusedMoe.Cfg {
    var use_fp8 = isMxFp8(opts.quant_scheme);
    if (b.dtype() == .f8e4m3fn) use_fp8 = true;
    return .{
        .a_dtype = toDType(a.dtype()),
        .b_dtype = toDType(b.dtype()),
        .c_dtype = toDType(output_dtype),
        .a_scale_dtype = if (use_fp8) .bf16 else null,
        .b_scale_dtype = if (use_fp8) .bf16 else null,
        .b_bias_dtype = null,
        .topk_weights_dtype = null,
        .block_size_m = @intCast(opts.block_size_m),
        .block_size_n = @intCast(opts.block_size_n),
        .block_size_k = @intCast(opts.block_size_k),
        .group_size_m = @intCast(opts.group_size_m),
        .top_k = @intCast(top_k),
        .naive_block_assignment = naive_block_assignment,
        .mul_routed_weight = mul_routed_weight,
        .compute_type = .bf16,
        .use_fp8_w8a8 = use_fp8,
        .use_int8_w8a8 = false,
        .use_int8_w8a16 = false,
        .per_channel_quant = false,
        .has_bias = has_bias,
    };
}

const DefaultTokenBucket = struct {
    tokens: i64,
    block_size_m: i64,
    block_size_n: i64,
    block_size_k: i64,
    group_size_m: i64,
    num_warps: i64,
    num_stages: i64,
};

fn applyDefaultTokenConfig(opts: Options, num_tokens: i64, num_experts: i64) Options {
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

    // General default policy for NVIDIA bf16/fp16 and fp8 per-tensor.
    // Tile sizes scale with batch size: small batches are more memory-bound,
    // while larger batches benefit from wider M/N tiles and more warps.
    if (num_tokens <= 32) {
        out.block_size_m = 16;
    } else if (num_tokens <= 96) {
        out.block_size_m = 32;
    } else if (num_tokens <= 512) {
        out.block_size_m = 64;
    } else {
        out.block_size_m = 128;
    }

    out.block_size_n = if (num_tokens <= 64) 64 else 128;
    out.block_size_k = if (isMxFp8(opts.quant_scheme) or num_tokens <= 64) 128 else 64;

    const tokens_per_expert = @divFloor(num_tokens, @max(num_experts, 1));
    out.group_size_m = if (tokens_per_expert > 128) 16 else 1;
    out.num_warps = if (num_tokens <= 128) 4 else 8;
    out.num_stages = if (num_tokens <= 32) 4 else 3;

    return out;
}

fn getLaunchConfigJsonPath(allocator: std.mem.Allocator) ![]const u8 {
    const runfiles = bazel.runfiles(bazel_builtin.current_repository) catch |err| {
        log.err("Failed to initialize runfiles for MoE launch config: {}", .{err});
        return err;
    };

    var config_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const config_path =
        runfiles.rlocation("zml/zml/moe/triton_kernels/config.json", &config_path_buf) catch null orelse
        runfiles.rlocation("zml/zml/moe/triton/triton_kernels/config.json", &config_path_buf) catch |err| {
            log.err("Failed to resolve MoE launch config in runfiles: {}", .{err});
            return err;
        };

    const config_json = config_path orelse {
        log.warn("MoE launch config is missing from runfiles at both zml/zml/moe/triton_kernels/config.json and zml/zml/moe/triton/triton_kernels/config.json", .{});
        return error.MissingLaunchConfigRunfile;
    };

    return try allocator.dupe(u8, config_json);
}

const JsonConfigFields = struct {
    BLOCK_SIZE_M: i64,
    BLOCK_SIZE_N: i64,
    BLOCK_SIZE_K: i64,
    GROUP_SIZE_M: i64,
    num_warps: i64,
    num_stages: i64,
};

fn applyJsonTokenConfig(opts: Options, num_tokens: i64) !Options {
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

    const compilation_context = zml.module.CompilationContext.current();
    const io = compilation_context.io;
    const allocator = compilation_context.allocator;

    const config_path = try getLaunchConfigJsonPath(allocator);
    defer allocator.free(config_path);
    const config_json = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .unlimited);
    defer allocator.free(config_json);
    const parsed = try std.json.parseFromSlice(
        std.json.ArrayHashMap(JsonConfigFields),
        allocator,
        config_json,
        .{},
    );
    defer parsed.deinit();

    var best_diff: u64 = std.math.maxInt(u64);
    var best_m: ?i64 = null;
    var best_config: ?JsonConfigFields = null;

    var it = parsed.value.map.iterator();
    while (it.next()) |entry| {
        const m = try std.fmt.parseInt(i64, entry.key_ptr.*, 10);

        const diff: u64 = @intCast(@abs(m - num_tokens));
        if (best_m == null or diff < best_diff or (diff == best_diff and m < best_m.?)) {
            best_m = m;
            best_config = entry.value_ptr.*;
            best_diff = diff;
        }
    }

    const cfg = best_config orelse return error.NoMatchingLaunchConfig;
    out.block_size_m = cfg.BLOCK_SIZE_M;
    out.block_size_n = cfg.BLOCK_SIZE_N;
    out.block_size_k = cfg.BLOCK_SIZE_K;
    out.group_size_m = cfg.GROUP_SIZE_M;
    out.num_warps = cfg.num_warps;
    out.num_stages = cfg.num_stages;

    return out;
}

fn fp8ActivationGroupSize(x: Tensor) i64 {
    const group_size: i64 = 128;
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "FP8 activation width must be divisible by {d}, got {d}", .{ group_size, x.dim(1) });
    return group_size;
}

fn validateOptions(opts: Options) !void {
    if (opts.inplace) return error.Unimplemented;
    if (opts.apply_router_weight_on_input) return error.UnsupportedOption;
    if (opts.quant_scheme != null and opts.quant_scheme.? != .mxfp4) return error.UnsupportedQuantization;
    if (opts.expert_map != null and opts.global_num_experts == -1) return error.InvalidShape;
    if (opts.w1_zp != null or opts.w2_zp != null) return error.UnsupportedOption;
    if (opts.a1_scale != null or opts.a2_scale != null or opts.block_shape != null) return error.UnsupportedOption;
    if (opts.w1_bias != null or opts.w2_bias != null) return error.UnsupportedOption;
}

fn validateInputs(hidden: Tensor, gate_up: Tensor, down: Tensor, weights: Tensor, ids: Tensor) !void {
    if (hidden.dtype() != .bf16) return error.UnsupportedType;
    if (gate_up.dtype() != .bf16 and gate_up.dtype() != .f8e4m3fn) return error.UnsupportedType;
    if (down.dtype() != .bf16 and down.dtype() != .f8e4m3fn) return error.UnsupportedType;
    if (weights.dtype() != .f32 and weights.dtype() != .bf16) return error.UnsupportedType;
    if (ids.dtype() != .i32) return error.UnsupportedType;
    if (hidden.dim(.in) != gate_up.dim(.in)) return error.InvalidShape;
    if (@rem(gate_up.dim(.out), 2) != 0) return error.InvalidShape;
    if (down.dim(.mid) != @divFloor(gate_up.dim(.out), 2)) return error.InvalidShape;
    if (ids.dim(.token) != hidden.dim(.token) or weights.dim(.token) != hidden.dim(.token)) return error.InvalidShape;
    if (ids.dim(.topk) != weights.dim(.topk)) return error.InvalidShape;
    if (gate_up.dim(.expert) != down.dim(.expert)) return error.InvalidShape;
}

// =====
// A16W4
// =====
pub fn fusedExpertsImpl_fp4(
    input: zml.Tensor,
    topk_ids: zml.Tensor,
    topk_weights: zml.Tensor,
    weights_gate_up: zml.Tensor,
    scales_gate_up: zml.Tensor,
    bias_gate_up: ?zml.Tensor,
    weights_down: zml.Tensor,
    scales_down: zml.Tensor,
    bias_down: ?zml.Tensor,
    activation_limit: ?f32,
) zml.Tensor {
    const x = input.reshape(.{
        .token = @divExact(@as(i64, @intCast(input.count())), input.dim(.d)),
        .d = input.dim(.d),
    });
    const kernel_cfg = getBestConfig(
        @intCast(topk_ids.dim(.b)),
        @intCast(topk_ids.dim(.eid)),
        @intCast(weights_gate_up.dim(.expert)),
    );
    const num_experts = weights_gate_up.dim(.expert);
    const aligned_routing = prepareRouting(topk_ids, num_experts, @intCast(kernel_cfg.block_m));
    const routing = prepareFp4Routing(
        aligned_routing,
        topk_ids,
        topk_weights,
        x.dim(.token),
        num_experts,
        @intCast(kernel_cfg.block_m),
    );

    const hidden_shape: zml.Shape = .init(.{
        .route = routing.num_rows,
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
            .activation_limit = activation_limit orelse 1.0,
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
    const topk = topk_ids.dim(.eid);
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
    activation_limit: f32 = 1.0,
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
) zml.Tensor {
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

fn applySwiGlu(input: zml.Tensor, activation_limit: f32) zml.Tensor {
    const threshold = zml.Tensor.scalar(activation_limit, .f32);
    const gate = input.slice1d(.dout, .{ .start = 0, .step = 2 }).minimum(threshold);
    const up = input.slice1d(.dout, .{ .start = 1, .step = 2 }).clamp(threshold.negate(), threshold);
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
