//! Triton MoE backend. Public entry point + orchestration: builds per-call
//! configs and invokes the kernels via `K.call(...)`. Each kernel handles
//! its own TTIR emission + `stablehlo.custom_call` insertion.
//!
//! Scope: matches `validateOptions` below — bf16 weights, no per-channel
//! quantization, no bias, no int8/int4. The fp8-quant path is wired so the
//! signature compiles, but only the bf16 matmul path has been exercised
//! end-to-end.

const std = @import("std");

const stdx = @import("stdx");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const tri = @import("zml/triton");
const DType = tri.DType;

const kernels = @import("triton_kernels.zig");

/// Map a zml `DataType` to the closest Triton DSL `DType`. The Triton set is
/// narrower (e.g. several fp8 variants collapse to `.f8e4m3fn`), so this is
/// intentionally lossy.
fn toDType(dt: DataType) DType {
    return switch (dt) {
        .bool => .i1,
        .i8, .u8 => .i8,
        .i16, .u16 => .i16,
        .i32, .u32 => .i32,
        .i64, .u64 => .i64,
        .f16 => .f16,
        .bf16 => .bf16,
        .f32 => .f32,
        .f64 => .f64,
        .f8e4m3fn, .f8e4m3b11fnuz, .f8e4m3fnuz => .f8e4m3fn,
        .f8e5m2, .f8e5m2fnuz => .f8e5m2,
        else => .i8,
    };
}

const log = std.log.scoped(.moe_triton);

// =============================================================================
// Public API
// =============================================================================

pub const Options = struct {
    inplace: bool = false,
    activation: []const u8 = "silu",
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

    pub const InitOptions = struct {
        num_experts_per_tok: u32,
    };

    pub fn init(opts: InitOptions) Parameters {
        return .{
            .num_experts_per_tok = opts.num_experts_per_tok,
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
        const replicated_sharding = try zml.sharding.replicatedSharding(platform);
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

fn initZeroBiasBuffer(io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding, shape: Shape) !zml.Buffer {
    var zero_slice: zml.Slice = try .alloc(std.heap.c_allocator, shape);
    defer zero_slice.free(std.heap.c_allocator);
    @memset(zero_slice.data(), 0);
    return zml.Buffer.fromSlice(io, platform, zero_slice, sharding);
}

// =============================================================================
// Top-level entry point
// =============================================================================

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
    const options = applyDefaultTokenConfig(opts, hidden_states.dim(0), w1.dim(0));

    const b = hidden_states.dim(.b);
    const s = hidden_states.dim(.s);

    const hidden = hidden_states.reshape(.{ .token = b * s, .in = hidden_states.dim(.d) }).withTags(.{ .token, .in });
    const gate_up = w1.withTags(.{ .expert, .out, .in });
    const down = w2.withTags(.{ .expert, .out, .mid });
    const weights = topk_weights.reshape(.{ .token = b * s, .in = topk_weights.dim(.top_expert) }).withTags(.{ .token, .topk });
    const ids = topk_ids.reshape(.{ .token = b * s, .in = topk_ids.dim(.top_expert) }).withTags(.{ .token, .topk });

    try validateInputs(hidden, gate_up, down, weights, ids);

    const block_size_m = options.block_size_m;
    const num_experts = gate_up.dim(.expert);
    const num_assignments = hidden.dim(.token) * ids.dim(.topk);
    const sparsity_factor: i64 = 4;
    const naive_block_assignment = num_assignments * sparsity_factor <= num_experts;

    const max_num_tokens_padded = if (naive_block_assignment)
        num_assignments * block_size_m
    else if (num_assignments < num_experts)
        num_assignments * block_size_m
    else
        num_assignments + num_experts * (block_size_m - 1);

    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        log.info("Using naive block assignment for MoE kernels. Num assignments: {d}, Num experts: {d}", .{ num_assignments, num_experts });
        const naive_sorted_ids = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));
        const naive_expert_ids = ids.reshape(.{ .g = num_assignments });
        const naive_num_tokens_post_padded = Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1});
        break :blk .{ naive_sorted_ids, naive_expert_ids, naive_num_tokens_post_padded };
    } else alignBlockSize(ids, num_experts, block_size_m);

    var hidden_quant = hidden;
    var a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);

    if (gate_up.dtype() == .f8e4m3fn) {
        hidden_quant, a_scale = quantizePerTokenGroupFp8(hidden, fp8ActivationGroupSize(hidden));
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

    const b_bias_1 = opts.w1_bias orelse metadata.w1_zero_bias.?;
    const b_scale_1 = opts.w1_scale orelse Tensor.scalar(1.0, .f32);

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

    const gate, const up = zml.nn.splitRealImg(first_out, .sequential);
    const activated = gate.silu().mul(up);

    var activated_quant = activated;
    a_scale = opts.a2_scale orelse Tensor.scalar(1.0, .f32);
    if (down.dtype() == .f8e4m3fn) {
        activated_quant, a_scale = quantizePerTokenGroupFp8(activated, fp8ActivationGroupSize(activated));
    }

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

    const b_bias_2 = opts.w2_bias orelse metadata.w2_zero_bias.?;
    const b_scale_2 = opts.w2_scale orelse Tensor.scalar(1.0, .f32);

    const second_out = callFusedMoe(
        activated_quant,
        down,
        b_bias_2,
        a_scale,
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
    cfg: kernels.FusedMoe.Config,
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

    return kernels.FusedMoe.call(.{
        .inputs = .{
            a,
            b,
            b_bias,
            a_scale,
            b_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            Tensor.constant(.{ .i64 = b.dim(1) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = b.dim(2) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = em_effective }).reshape(.{1}),
            Tensor.constant(.{ .i64 = num_valid_tokens }).reshape(.{1}),
            Tensor.constant(.{ .i64 = a.dim(1) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = b.dim(1) * b.dim(2) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = b.dim(2) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = b.dim(.out) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = stride_asm }).reshape(.{1}),
            Tensor.constant(.{ .i64 = stride_ask }).reshape(.{1}),
            Tensor.constant(.{ .i64 = stride_bse }).reshape(.{1}),
            Tensor.constant(.{ .i64 = stride_bsk }).reshape(.{1}),
            Tensor.constant(.{ .i64 = stride_bsn }).reshape(.{1}),
            Tensor.constant(.{ .i64 = 0 }).reshape(.{1}),
            Tensor.constant(.{ .i64 = 0 }).reshape(.{1}),
        },
        .outputs = .{output_shape},
        .cfg = cfg,
        .grid = .{ @intCast(grid_x), 1, 1 },
        .num_warps = @intCast(options.num_warps),
        .num_stages = @intCast(options.num_stages),
    })[0];
}

fn alignBlockSize(topk_ids: Tensor, num_experts: i64, block_size_m: i64) struct { Tensor, Tensor, Tensor } {
    log.info("Using triton kernels to sort and align tokens to experts with block size {d}", .{block_size_m});
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
        const align_outs = kernels.MoeAlignBlockSize.call(.{
            .inputs = .{
                flat_experts,
                sorted_token_ids,
                expert_ids,
                num_tokens_post_padded,
                cumsums,
            },
            .outputs = .{
                sorted_token_ids.shape(),
                expert_ids.shape(),
                num_tokens_post_padded.shape(),
                cumsums.shape(),
            },
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
            .output_operand_aliases = &.{
                .{ .output_index = 0, .operand_index = 1 },
                .{ .output_index = 1, .operand_index = 2 },
                .{ .output_index = 2, .operand_index = 3 },
                .{ .output_index = 3, .operand_index = 4 },
            },
        });
        sorted_token_ids = align_outs[0];
        expert_ids = align_outs[1];
        num_tokens_post_padded = align_outs[2];
        cumsums = align_outs[3];
    }

    {
        const sort_outs = kernels.CountAndSortExpertTokens.call(.{
            .inputs = .{
                flat_experts,
                sorted_token_ids,
                cumsums,
            },
            .outputs = .{ sorted_token_ids.shape(), cumsums.shape() },
            .cfg = .{
                .numel = @intCast(num_assignments),
                .num_experts = @intCast(num_experts),
                .sort_block_size = @intCast(sort_block_size),
            },
            .grid = .{ @intCast(sort_grid_x), 1, 1 },
            .num_stages = 1,
            .num_warps = 4,
            .output_operand_aliases = &.{
                .{ .output_index = 0, .operand_index = 1 },
                .{ .output_index = 1, .operand_index = 2 },
            },
        });
        sorted_token_ids = sort_outs[0];
        cumsums = sort_outs[1];
    }

    return .{ sorted_token_ids, expert_ids, num_tokens_post_padded };
}

fn quantizePerTokenGroupFp8(x: Tensor, group_size: i64) struct { Tensor, Tensor } {
    stdx.debug.assert(x.rank() == 2, "expected a rank-2 activation matrix, got {f}", .{x.shape()});
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "activation width must be divisible by group size {d}, got {d}", .{ group_size, x.dim(1) });

    const groups_per_row = @divExact(x.dim(1), group_size);
    const quantized = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .feature = x.dim(1) }, .f8e4m3fn));
    const scales = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .group = groups_per_row }, .bf16));

    const outs = kernels.PerTokenGroupQuantFp8.call(.{
        .inputs = .{
            x,
            Tensor.constant(.{ .i64 = group_size }).reshape(.{1}),
            Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
            Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
            Tensor.scalar(1e-6, .f32),
        },
        .outputs = .{ quantized.shape(), scales.shape() },
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
    });

    return .{ outs[0], outs[1] };
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
) kernels.FusedMoe.Config {
    var use_fp8 = opts.use_fp8_w8a8;
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

const default_token_buckets = [_]DefaultTokenBucket{
    .{ .tokens = 1, .block_size_m = 16, .block_size_n = 32, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 4 },
    .{ .tokens = 2, .block_size_m = 16, .block_size_n = 32, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 4 },
    .{ .tokens = 4, .block_size_m = 16, .block_size_n = 32, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 3 },
    .{ .tokens = 8, .block_size_m = 16, .block_size_n = 128, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 3 },
    .{ .tokens = 16, .block_size_m = 16, .block_size_n = 64, .block_size_k = 64, .group_size_m = 64, .num_warps = 4, .num_stages = 5 },
    .{ .tokens = 24, .block_size_m = 16, .block_size_n = 64, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 2 },
    .{ .tokens = 32, .block_size_m = 16, .block_size_n = 32, .block_size_k = 128, .group_size_m = 1, .num_warps = 4, .num_stages = 2 },
    .{ .tokens = 48, .block_size_m = 16, .block_size_n = 32, .block_size_k = 128, .group_size_m = 64, .num_warps = 4, .num_stages = 2 },
    .{ .tokens = 64, .block_size_m = 16, .block_size_n = 64, .block_size_k = 128, .group_size_m = 1, .num_warps = 4, .num_stages = 2 },
    .{ .tokens = 96, .block_size_m = 16, .block_size_n = 128, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 3 },
    .{ .tokens = 128, .block_size_m = 16, .block_size_n = 256, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 2 },
    .{ .tokens = 256, .block_size_m = 16, .block_size_n = 256, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 2 },
    .{ .tokens = 512, .block_size_m = 32, .block_size_n = 128, .block_size_k = 128, .group_size_m = 1, .num_warps = 8, .num_stages = 3 },
    .{ .tokens = 1024, .block_size_m = 64, .block_size_n = 128, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 3 },
    .{ .tokens = 1536, .block_size_m = 64, .block_size_n = 128, .block_size_k = 64, .group_size_m = 1, .num_warps = 4, .num_stages = 3 },
    .{ .tokens = 2048, .block_size_m = 128, .block_size_n = 128, .block_size_k = 64, .group_size_m = 16, .num_warps = 8, .num_stages = 3 },
    .{ .tokens = 3072, .block_size_m = 128, .block_size_n = 256, .block_size_k = 64, .group_size_m = 1, .num_warps = 8, .num_stages = 4 },
    .{ .tokens = 4096, .block_size_m = 128, .block_size_n = 256, .block_size_k = 64, .group_size_m = 16, .num_warps = 8, .num_stages = 4 },
};

fn applyDefaultTokenConfig(opts: Options, num_tokens: i64, num_experts: i64) Options {
    _ = num_experts;
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

    const bucket = bucket: {
        for (default_token_buckets) |b| {
            if (b.tokens >= num_tokens) break :bucket b;
        }
        break :bucket default_token_buckets[default_token_buckets.len - 1];
    };

    out.block_size_m = bucket.block_size_m;
    out.block_size_n = bucket.block_size_n;
    out.block_size_k = bucket.block_size_k;
    out.group_size_m = bucket.group_size_m;
    out.num_warps = bucket.num_warps;
    out.num_stages = bucket.num_stages;

    return out;
}

fn fp8ActivationGroupSize(x: Tensor) i64 {
    const group_size: i64 = 128;
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "FP8 activation width must be divisible by {d}, got {d}", .{ group_size, x.dim(1) });
    return group_size;
}

fn validateOptions(opts: Options) !void {
    if (opts.inplace) return error.Unimplemented;
    if (!std.mem.eql(u8, opts.activation, "silu")) return error.UnsupportedActivation;
    if (opts.apply_router_weight_on_input) return error.UnsupportedOption;
    if (opts.use_fp8_w8a8 or opts.use_int8_w8a8 or opts.use_int8_w8a16 or opts.use_int4_w4a16) return error.UnsupportedQuantization;
    if (opts.ocp_mx_scheme != null or opts.per_channel_quant) return error.UnsupportedOption;
    if (opts.global_num_experts != -1) return error.UnsupportedOption;
    if (opts.expert_map != null) return error.UnsupportedOption;
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
