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

const log = std.log.scoped(.moe_triton);

// =============================================================================
// Public API
// =============================================================================

pub const Options = struct {
    inplace: bool = false,
    activation: Parameters.ActivationMode = .silu,
    activation_limit: f32 = std.math.inf(f32),
    apply_router_weight_on_input: bool = false,
    use_fp8_w8a8: bool = false,
    use_int8_w8a8: bool = false,
    use_int8_w8a16: bool = false,
    use_int4_w4a16: bool = false,
    ocp_mx_scheme: ?[]const u8 = null,
    per_channel_quant: bool = false,
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
};

pub const Parameters = struct {
    num_experts_per_tok: u32,
    activation: ActivationMode,
    activation_limit: f32,

    pub const ActivationMode = enum {
        silu,
        relu,
        gelu,
    };

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
        activation: ActivationMode,
        activation_limit: f32 = std.math.inf(f32),
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
            .activation = opts.activation,
            .activation_limit = opts.activation_limit,
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

fn applyActivation(x: Tensor, mode: Parameters.ActivationMode, activation_limit: f32) Tensor {
    const mid = @divFloor(x.dim(.out), 2);
    var gate = x.slice1d(.out, .{ .end = mid });
    var up = x.slice1d(.out, .{ .start = mid });
    if (std.math.isFinite(activation_limit)) {
        const limit = Tensor.scalar(activation_limit, x.dtype());
        gate = Tensor.select(gate.cmp(.GT, limit), limit, gate);
        up = up.clamp(limit.negate(), limit);
    }
    return switch (mode) {
        .silu => gate.silu().mul(up),
        .relu => x.relu().powByConst(2),
        .gelu => gate.gelu().mul(up),
    };
}

fn ckFusedExpertsImpl(
    hidden: Tensor,
    gate_up: Tensor,
    down: Tensor,
    weights: Tensor,
    ids: Tensor,
    opts: Options,
) !Tensor {
    if (opts.activation != .silu) return error.UnsupportedActivation;
    if (std.math.isFinite(opts.activation_limit)) return error.UnsupportedActivationLimit;
    if (opts.w1_bias != null or opts.w2_bias != null) return error.UnsupportedBias;
    const w1_scale = (opts.w1_scale orelse return error.MissingWeightScale).convert(.f32);
    const w2_scale = (opts.w2_scale orelse return error.MissingWeightScale).convert(.f32);

    const tokens = hidden.dim(.token);
    const topk = ids.dim(.topk);
    const num_assignments = tokens * topk;
    const local_ids = if (opts.expert_map) |expert_map|
        expert_map.gather(.{ .expert = ids }, .{}).withTags(.{ .token, .topk })
    else
        ids;
    const sorted_flat_ids, const sorted_expert_ids, const num_valid_ids =
        alignBlockSize(local_ids, gate_up.dim(.expert), 32);

    // AITER encodes the route lane in the high byte and the token in the
    // low 24 bits. ZML's sorter carries a flattened token*topk+route index.
    const flat_sorted = sorted_flat_ids.withTags(.{.sorted});
    const valid = flat_sorted.cmp(.LT, Tensor.scalar(num_assignments, .i32));
    const safe_flat = valid.select(flat_sorted, Tensor.scalar(num_assignments, .i32));
    const token_ids = safe_flat.divByConst(topk);
    const route_ids = safe_flat.remainderConst(topk);
    const encoded_valid = token_ids.logical(
        .OR,
        route_ids.shiftLeft(Tensor.scalar(24, .i32)),
    );
    const encoded_ids = valid.select(encoded_valid, Tensor.scalar(tokens, .i32));

    // AITER indexes route weights by sorted position. Add one zero sentinel so
    // padded sorter entries can be gathered without an out-of-bounds index.
    const flat_weights = weights.convert(.f32).reshape(.{ .assignment = num_assignments });
    const padded_weights = Tensor.concatenate(&.{
        flat_weights,
        Tensor.zeroes(Shape.init(.{ .assignment = 1 }, .f32)),
    }, .assignment);
    const sorted_weights = padded_weights.gather(
        .{ .assignment = safe_flat },
        .{},
    ).withTags(.{.sorted});

    const hidden_quant, const a1_scale = quantizePerTokenGroupFp8(hidden, 128, true);
    const intermediate_size = @divExact(gate_up.dim(.out), 2);
    const output = zml.fp8.ck.moe(
        encoded_ids,
        sorted_expert_ids,
        num_valid_ids,
        sorted_weights,
        hidden_quant,
        gate_up,
        down,
        a1_scale.transpose(.{ .group, .token }),
        w1_scale,
        w2_scale,
        Shape.init(.{ .token = tokens, .out = down.dim(.out) }, .bf16),
        .{
            .tokens = tokens,
            .experts = gate_up.dim(.expert),
            .topk = topk,
            .sorted_blocks = sorted_expert_ids.dim(0),
            .hidden_size = hidden.dim(.in),
            .intermediate_size = intermediate_size,
        },
    );
    return output;
}

// =============================================================================
// Top-level entry point
// =============================================================================

fn hasBlock128Scale(weight: Tensor, maybe_scale: ?Tensor) bool {
    if (weight.rank() != 3 or
        (weight.dtype() != .f8e4m3fn and weight.dtype() != .f8e4m3fnuz))
    {
        return false;
    }

    const scale = maybe_scale orelse return false;
    if (scale.rank() != 3 or scale.dim(0) != weight.dim(0)) return false;

    const n_blocks = std.math.divCeil(i64, weight.dim(1), 128) catch unreachable;
    const k_blocks = std.math.divCeil(i64, weight.dim(2), 128) catch unreachable;
    return scale.dim(1) == n_blocks and scale.dim(2) == k_blocks;
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
    var options = applyJsonTokenConfig(opts, hidden_states.dim(0)) catch |err| fallback: {
        log.warn("Failed to load MoE launch config from JSON ({}), falling back to built-in token heuristic", .{err});
        break :fallback applyDefaultTokenConfig(opts, hidden_states.dim(0), w1.dim(0));
    };
    const block_fp8 = hasBlock128Scale(w1, opts.w1_scale) and hasBlock128Scale(w2, opts.w2_scale);
    if (block_fp8) options.block_size_k = 128;
    const rocm_block_fp8 = block_fp8 and zml.module.CompilationContext.current().platform.target == .rocm;
    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);
    if (rocm_block_fp8 and b * s <= 16) {
        // The generic JSON table was tuned for other backends and is notably
        // underperformant for small-M block-FP8 MoE on gfx942. This tile was
        // checked at 1, 8, and 16 tokens; larger batches retain the table.
        options.block_size_m = 16;
        options.block_size_n = 64;
        options.block_size_k = 128;
        options.group_size_m = 1;
        options.num_warps = 4;
        options.num_stages = 3;
    }

    const hidden = hidden_states.reshape(.{ .token = b * s, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up_fn = w1.withTags(.{ .expert, .out, .in });
    const down_fn = w2.withTags(.{ .expert, .out, .mid });
    const gate_up = if (rocm_block_fp8 and gate_up_fn.dtype() == .f8e4m3fn) gate_up_fn.bitCast(.f8e4m3fnuz) else gate_up_fn;
    const down = if (rocm_block_fp8 and down_fn.dtype() == .f8e4m3fn) down_fn.bitCast(.f8e4m3fnuz) else down_fn;
    const weights = topk_weights.reshape(.{ .token = b * s, .in = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = b * s, .in = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    try validateInputs(hidden, gate_up, down, weights, ids);

    if (rocm_block_fp8 and zml.fp8.Backend.selected() == .ck) {
        const output = try ckFusedExpertsImpl(hidden, gate_up, down, weights, ids, opts);
        return output.reshape(.{ .b = b, .token = s, .out = down.dim(.out) });
    }

    const block_size_m = options.block_size_m;
    const num_experts = if (opts.global_num_experts != -1) opts.global_num_experts else gate_up.dim(.expert);
    if (opts.expert_map) |expert_map| {
        if (expert_map.dtype() != .i32) return error.UnsupportedType;
        if (expert_map.rank() != 1 or expert_map.dim(.expert) != num_experts) return error.InvalidShape;
    }
    const num_assignments = hidden.dim(.token) * ids.dim(.topk);
    const sparsity_factor: i64 = 4;
    const naive_block_assignment = num_assignments * sparsity_factor <= num_experts;

    const max_num_tokens_padded = if (naive_block_assignment)
        num_assignments * block_size_m
    else if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        num_assignments + num_experts * (block_size_m - 1);

    const sorted_token_ids, const expert_ids_global, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        log.debug("Using naive block assignment for MoE kernels. Num assignments: {d}, Num experts: {d}", .{ num_assignments, num_experts });
        const naive_sorted_ids = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));
        const naive_expert_ids = ids.reshape(.{ .g = num_assignments });
        const naive_num_tokens_post_padded = Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1});
        break :blk .{ naive_sorted_ids, naive_expert_ids, naive_num_tokens_post_padded };
    } else alignBlockSize(ids, num_experts, block_size_m);

    const expert_ids = if (opts.expert_map) |expert_map|
        expert_map.gather(.{ .expert = expert_ids_global }, .{}).withTags(.{.g})
    else
        expert_ids_global;

    var hidden_quant = hidden;
    var a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);

    if (gate_up.dtype() == .f8e4m3fn or gate_up.dtype() == .f8e4m3fnuz) {
        hidden_quant, a_scale = quantizePerTokenGroupFp8(hidden, fp8ActivationGroupSize(hidden), gate_up.dtype() == .f8e4m3fnuz);
    }

    const first_cfg = makeFusedMoeConfig(
        hidden_quant,
        gate_up,
        options,
        naive_block_assignment,
        ids.dim(.topk),
        false,
        false,
        .bf16,
    );

    const b_bias_1 =
        opts.w1_bias orelse
        metadata.w1_zero_bias orelse
        Tensor.zeroes(Shape.init(.{ .expert = gate_up.dim(.expert), .out = gate_up.dim(.out) }, .bf16));

    const b_scale_1 = if (gate_up.dtype() == .f8e4m3fnuz)
        (opts.w1_scale orelse return error.MissingWeightScale).convert(.f32)
    else
        opts.w1_scale orelse Tensor.scalar(1.0, .f32);

    const first_out = callFusedMoe(
        hidden_quant,
        gate_up,
        b_bias_1,
        a_scale,
        b_scale_1,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        first_cfg,
        options,
        max_num_tokens_padded,
        num_assignments,
        Shape.init(.{ .token = num_assignments, .out = gate_up.dim(.out) }, .bf16),
    );

    const activated_quant, const a2_scale = if (rocm_block_fp8 and
        options.activation == .silu and
        down.dtype() == .f8e4m3fnuz)
        siluAndQuantizePerTokenGroupFp8(first_out, 128, true, options.activation_limit)
    else blk: {
        const activated = applyActivation(first_out, options.activation, options.activation_limit);
        var quantized = activated;
        var scale = opts.a2_scale orelse Tensor.scalar(1.0, .f32);
        if (down.dtype() == .f8e4m3fn or down.dtype() == .f8e4m3fnuz) {
            quantized, scale = quantizePerTokenGroupFp8(activated, fp8ActivationGroupSize(activated), down.dtype() == .f8e4m3fnuz);
        }
        break :blk .{ quantized, scale };
    };

    const second_cfg = makeFusedMoeConfig(
        activated_quant,
        down,
        options,
        naive_block_assignment,
        1,
        true,
        false,
        .bf16,
    );

    const b_bias_2 =
        opts.w2_bias orelse
        metadata.w2_zero_bias orelse
        Tensor.zeroes(Shape.init(.{ .expert = down.dim(.expert), .out = down.dim(.out) }, .bf16));

    const b_scale_2 = if (down.dtype() == .f8e4m3fnuz)
        (opts.w2_scale orelse return error.MissingWeightScale).convert(.f32)
    else
        opts.w2_scale orelse Tensor.scalar(1.0, .f32);

    const second_out = callFusedMoe(
        activated_quant,
        down,
        b_bias_2,
        a2_scale,
        b_scale_2,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        second_cfg,
        options,
        max_num_tokens_padded,
        num_assignments,
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
    // Triton ranges (and the histogram result built over this range) must have a
    // power-of-two width.  Rounding only to a warp multiple breaks models such
    // as GLM-5.3, which has 288 experts.
    const padded_num_experts: i64 = @intCast(std.math.ceilPowerOfTwoAssert(usize, @intCast(num_experts)));
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

fn quantizePerTokenGroupFp8(x: Tensor, group_size: i64, fnuz: bool) struct { Tensor, Tensor } {
    stdx.debug.assert(x.rank() == 2, "expected a rank-2 activation matrix, got {f}", .{x.shape()});
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "activation width must be divisible by group size {d}, got {d}", .{ group_size, x.dim(1) });

    const groups_per_row = @divExact(x.dim(1), group_size);
    const output_dtype: DataType = if (fnuz) .f8e4m3fnuz else .f8e4m3fn;
    const scale_dtype: DataType = if (fnuz) .f32 else .bf16;
    const fp8_max: f32 = if (fnuz) 240.0 else 448.0;
    const quantized = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .feature = x.dim(1) }, output_dtype));
    const scales = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .group = groups_per_row }, scale_dtype));

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
                .output_dtype = toDType(output_dtype),
                .scale_dtype = toDType(scale_dtype),
                .block = @intCast(group_size),
                .fp8_min = -fp8_max,
                .fp8_max = fp8_max,
                .use_ue8m0 = false,
            },
            .grid = .{ @intCast(x.dim(0) * groups_per_row), 1, 1 },
            .num_stages = 1,
            .num_warps = 1,
        },
    );

    return .{ outs.y_q, outs.y_s };
}

fn siluAndQuantizePerTokenGroupFp8(x: Tensor, group_size: i64, fnuz: bool, activation_limit: f32) struct { Tensor, Tensor } {
    stdx.debug.assert(x.rank() == 2, "expected a rank-2 SwiGLU input, got {f}", .{x.shape()});
    const output_columns = @divExact(x.dim(1), 2);
    stdx.debug.assert(@mod(output_columns, group_size) == 0, "SwiGLU output width must be divisible by group size {d}, got {d}", .{ group_size, output_columns });

    const groups_per_row = @divExact(output_columns, group_size);
    const output_dtype: DataType = if (fnuz) .f8e4m3fnuz else .f8e4m3fn;
    const scale_dtype: DataType = if (fnuz) .f32 else .bf16;
    const fp8_max: f32 = if (fnuz) 240.0 else 448.0;
    const outs = kernels.SiluAndQuantizePerTokenGroupFp8.Kernel.call(
        .{ .x = x },
        .{
            .q = Shape.init(.{ .token = x.dim(0), .feature = output_columns }, output_dtype),
            .scale = Shape.init(.{ .token = x.dim(0), .group = groups_per_row }, scale_dtype),
        },
        .{
            .cfg = .{
                .input_dtype = toDType(x.dtype()),
                .output_dtype = toDType(output_dtype),
                .scale_dtype = toDType(scale_dtype),
                .input_columns = @intCast(x.dim(1)),
                .output_columns = @intCast(output_columns),
                .block = @intCast(group_size),
                .fp8_min = -fp8_max,
                .fp8_max = fp8_max,
                .activation_limit = activation_limit,
            },
            .grid = .{ @intCast(x.dim(0) * groups_per_row), 1, 1 },
            .num_stages = 1,
            .num_warps = 1,
        },
    );
    return .{ outs.q, outs.scale };
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
    var use_fp8 = opts.use_fp8_w8a8;
    if (b.dtype() == .f8e4m3fn or b.dtype() == .f8e4m3fnuz) use_fp8 = true;
    const fp8_scale_dtype: ?DType = if (!use_fp8)
        null
    else if (b.dtype() == .f8e4m3fnuz)
        .f32
    else
        .bf16;
    return .{
        .a_dtype = toDType(a.dtype()),
        .b_dtype = toDType(b.dtype()),
        .c_dtype = toDType(output_dtype),
        .a_scale_dtype = fp8_scale_dtype,
        .b_scale_dtype = fp8_scale_dtype,
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
    out.block_size_k = if (opts.use_fp8_w8a8 or num_tokens <= 64) 128 else 64;

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
    if (opts.use_fp8_w8a8 or opts.use_int8_w8a8 or opts.use_int8_w8a16 or opts.use_int4_w4a16) return error.UnsupportedQuantization;
    if (opts.ocp_mx_scheme != null or opts.per_channel_quant) return error.UnsupportedOption;
    if (opts.expert_map != null and opts.global_num_experts == -1) return error.InvalidShape;
    if (opts.w1_zp != null or opts.w2_zp != null) return error.UnsupportedOption;
    if (opts.a1_scale != null or opts.a2_scale != null or opts.block_shape != null) return error.UnsupportedOption;
    if (opts.w1_bias != null or opts.w2_bias != null) return error.UnsupportedOption;
}

fn validateInputs(hidden: Tensor, gate_up: Tensor, down: Tensor, weights: Tensor, ids: Tensor) !void {
    if (hidden.dtype() != .bf16) return error.UnsupportedType;
    if (gate_up.dtype() != .bf16 and gate_up.dtype() != .f8e4m3fn and gate_up.dtype() != .f8e4m3fnuz) return error.UnsupportedType;
    if (down.dtype() != .bf16 and down.dtype() != .f8e4m3fn and down.dtype() != .f8e4m3fnuz) return error.UnsupportedType;
    if (weights.dtype() != .f32 and weights.dtype() != .bf16) return error.UnsupportedType;
    if (ids.dtype() != .i32) return error.UnsupportedType;
    if (hidden.dim(.in) != gate_up.dim(.in)) return error.InvalidShape;
    if (@rem(gate_up.dim(.out), 2) != 0) return error.InvalidShape;
    if (down.dim(.mid) != @divFloor(gate_up.dim(.out), 2)) return error.InvalidShape;
    if (ids.dim(.token) != hidden.dim(.token) or weights.dim(.token) != hidden.dim(.token)) return error.InvalidShape;
    if (ids.dim(.topk) != weights.dim(.topk)) return error.InvalidShape;
    if (gate_up.dim(.expert) != down.dim(.expert)) return error.InvalidShape;
}
