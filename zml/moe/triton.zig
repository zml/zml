//! Triton MoE backend. Mirrors Louis's Python-driven version at
//! `zml_louis-qwen3_5_moe/zml/moe/triton.zig`, but replaces the Python
//! subprocess (which drove Triton's frontend to emit TTIR text) with our
//! in-tree Zig Triton DSL in `zml/triton/kernel.zig`.
//!
//! Each `generate*Ttir` function constructs the kernel body op-by-op so the
//! TTIR string handed to `zml.ops.triton(...)` is functionally equivalent to
//! what the upstream Triton frontend produces from `triton_kernels/moe.py`.
//!
//! Scope: matches `validateOptions` below — bf16 weights, no per-channel
//! quantization, no bias, no int8/int4. The fp8-quant path is wired so the
//! signature compiles, but only the bf16 matmul path has been exercised
//! end-to-end.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const ops = zml.ops;
const kernel_dsl = @import("zml/triton");

const Kernel = kernel_dsl.Kernel;
const DslValue = kernel_dsl.Value;
const DslArgSpec = kernel_dsl.ArgSpec;

const log = std.log.scoped(.moe_triton);

// =============================================================================
// Public API — same shape as Louis's version so the rest of the model
// (examples/llm/models/qwen3_5_moe/*) compiles unchanged.
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
// Generation config — kept for parity with Louis's version (the legacy JSON
// launch-config lookup still reads these field names).
// =============================================================================

pub const GenerationConfig = struct {
    a_dtype: DataType,
    b_dtype: DataType,
    c_dtype: DataType,
    a_scale_dtype: ?DataType = null,
    b_scale_dtype: ?DataType = null,
    b_bias_dtype: ?DataType = null,
    topk_weights_dtype: ?DataType = null,
    num_tokens: usize,
    top_k: usize,
    num_experts: usize,
    out_features: usize,
    in_features: usize,
    max_num_tokens_padded: usize,
    num_valid_tokens: usize,
    block_size_m: usize,
    block_size_n: usize,
    block_size_k: usize,
    group_size_m: usize,
    split_k: usize = 1,
    group_n: usize = 0,
    group_k: usize = 0,
    naive_block_assignment: bool = false,
    mul_routed_weight: bool = false,
    compute_type: DataType = .bf16,
    use_fp8_w8a8: bool = false,
    use_int8_w8a8: bool = false,
    use_int8_w8a16: bool = false,
    per_channel_quant: bool = false,
    has_bias: bool = false,
    num_warps: usize,
    num_stages: usize,
};

const AlignBlockSizeKernel = enum {
    align_block_size,
    count_and_sort,

    fn kernelName(self: @This()) []const u8 {
        return switch (self) {
            .align_block_size => "moe_align_block_size_kernel",
            .count_and_sort => "count_and_sort_expert_tokens_kernel",
        };
    }
};

const AlignBlockSizeGenerationConfig = struct {
    kernel_name: []const u8,
    numel: usize,
    num_experts: usize,
    padded_num_experts: usize,
    max_num_tokens_padded: usize,
    max_num_m_blocks: usize,
    block_size_m: usize,
    experts_per_warp: usize,
    hist_block: usize,
    sort_block_size: usize,
    sort_grid_x: usize,
};

const QuantGenerationConfig = struct {
    num_rows: usize,
    num_columns: usize,
    group_size: usize,
    block: usize,
    input_dtype: DataType,
    output_dtype: DataType,
    scale_dtype: DataType = .bf16,
    eps: f32 = 1e-6,
    fp8_min: f32,
    fp8_max: f32,
    use_ue8m0: bool = false,
};

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

    const allocator = zml.module.CompilationContext.current().allocator;

    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        log.info("Using naive block assignment for MoE kernels. Num assignments: {d}, Num experts: {d}", .{ num_assignments, num_experts });
        const naive_sorted_ids = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));
        const naive_expert_ids = ids.reshape(.{ .g = num_assignments });
        const naive_num_tokens_post_padded = Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1});
        break :blk .{ naive_sorted_ids, naive_expert_ids, naive_num_tokens_post_padded };
    } else try alignBlockSize(allocator, ids, num_experts, block_size_m);

    var hidden_quant = hidden;
    var a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);

    if (gate_up.dtype() == .f8e4m3fn) {
        hidden_quant, a_scale = try quantizePerTokenGroupFp8(allocator, hidden, fp8ActivationGroupSize(hidden));
    }

    const first_generation_config = makeGenerationConfig(
        hidden_quant,
        gate_up,
        max_num_tokens_padded,
        num_assignments,
        options,
        naive_block_assignment,
        ids.dim(.topk),
        false,
        false,
        .bf16,
    );

    const ttir_first_matmul = try generateFusedMoeKernelTtir(allocator, first_generation_config);
    defer allocator.free(ttir_first_matmul);

    var b_bias = opts.w1_bias orelse metadata.w1_zero_bias.?;
    var b_scale = opts.w1_scale orelse Tensor.scalar(1.0, .f32);

    const first_out = callFusedKernel(
        hidden_quant,
        gate_up,
        b_bias,
        a_scale,
        b_scale,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        ttir_first_matmul,
        first_generation_config,
        max_num_tokens_padded,
        num_assignments,
        Shape.init(.{ .token = num_assignments, .out = gate_up.dim(.out) }, .bf16),
    );

    const gate, const up = zml.nn.splitRealImg(first_out, .sequential);
    const activated = gate.silu().mul(up);

    var activated_quant = activated;
    a_scale = opts.a2_scale orelse Tensor.scalar(1.0, .f32);
    if (down.dtype() == .f8e4m3fn) {
        activated_quant, a_scale = try quantizePerTokenGroupFp8(allocator, activated, fp8ActivationGroupSize(activated));
    }

    const second_generation_config = makeGenerationConfig(
        activated_quant,
        down,
        max_num_tokens_padded,
        num_assignments,
        options,
        naive_block_assignment,
        1,
        true,
        false,
        .bf16,
    );
    const ttir_second_matmul = try generateFusedMoeKernelTtir(allocator, second_generation_config);
    defer allocator.free(ttir_second_matmul);
    b_bias = opts.w2_bias orelse metadata.w2_zero_bias.?;
    b_scale = opts.w2_scale orelse Tensor.scalar(1.0, .f32);

    const second_out = callFusedKernel(
        activated_quant,
        down,
        b_bias,
        a_scale,
        b_scale,
        weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        ttir_second_matmul,
        second_generation_config,
        max_num_tokens_padded,
        num_assignments,
        Shape.init(.{ .token = b * s, .topk = ids.dim(.topk), .out = down.dim(.out) }, .bf16),
    );

    const output = second_out.sum(.topk).squeeze(.topk);

    return output.reshape(.{ .b = b, .token = s, .out = down.dim(.out) });
}

fn callFusedKernel(
    a: Tensor,
    b: Tensor,
    b_bias: Tensor,
    a_scale: Tensor,
    b_scale: Tensor,
    topk_weights: Tensor,
    sorted_token_ids: Tensor,
    expert_ids: Tensor,
    num_tokens_post_padded: Tensor,
    ir: [:0]const u8,
    config: GenerationConfig,
    max_num_tokens_padded: i64,
    num_valid_tokens: i64,
    output_shape: Shape,
) Tensor {
    const block_size_m: i64 = @intCast(config.block_size_m);
    const block_size_n: i64 = @intCast(config.block_size_n);
    const m_tokens = a.dim(0);
    const em_effective = if (m_tokens < block_size_m)
        @min(max_num_tokens_padded, num_valid_tokens * block_size_m)
    else
        max_num_tokens_padded;
    const grid_x =
        (std.math.divCeil(i64, em_effective, block_size_m) catch unreachable) *
        (std.math.divCeil(i64, b.dim(1), block_size_n) catch unreachable);

    const stride_asm: i64 = if (config.group_k > 0 and a_scale.rank() == 2) a_scale.dim(1) else 0;
    const stride_ask: i64 = if (config.group_k > 0 and a_scale.rank() == 2) 1 else 0;
    const stride_bse: i64 = if (config.group_k > 0 and config.group_n > 0 and b_scale.rank() == 3)
        b_scale.dim(1) * b_scale.dim(2)
    else
        0;
    const stride_bsk: i64 = if (config.group_k > 0 and config.group_n > 0 and b_scale.rank() == 3) 1 else 0;
    const stride_bsn: i64 = if (config.group_k > 0 and config.group_n > 0 and b_scale.rank() == 3) b_scale.dim(2) else 0;

    const inputs = .{
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
    };
    const outputs = ops.triton(inputs, .{output_shape}, .{
        .name = "fused_moe_kernel",
        .ir = ir,
        .grid = .{
            @intCast(grid_x),
            1,
            1,
        },
        .num_stages = @intCast(config.num_stages),
        .num_warps = @intCast(config.num_warps),
        .output_operand_aliases = &.{},
    });
    return outputs[0];
}

fn alignBlockSize(allocator: std.mem.Allocator, topk_ids: Tensor, num_experts: i64, block_size_m: i64) !struct { Tensor, Tensor, Tensor } {
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
    const experts_per_warp: i64 = warp_size;
    const hist_block: i64 = 256;
    const sort_block_size: i64 = 256;
    const sort_grid_x: i64 = @min(std.math.divCeil(i64, num_assignments, sort_block_size) catch unreachable, 65535);

    const align_config: AlignBlockSizeGenerationConfig = .{
        .kernel_name = AlignBlockSizeKernel.align_block_size.kernelName(),
        .numel = @intCast(num_assignments),
        .num_experts = @intCast(num_experts),
        .padded_num_experts = @intCast(padded_num_experts),
        .max_num_tokens_padded = @intCast(max_num_tokens_padded),
        .max_num_m_blocks = @intCast(max_num_m_blocks),
        .block_size_m = @intCast(block_size_m),
        .experts_per_warp = @intCast(experts_per_warp),
        .hist_block = @intCast(hist_block),
        .sort_block_size = @intCast(sort_block_size),
        .sort_grid_x = @intCast(sort_grid_x),
    };
    const ttir_align = try generateAlignBlockSizeKernelTtir(allocator, align_config);
    defer allocator.free(ttir_align);

    var count_sort_config = align_config;
    count_sort_config.kernel_name = AlignBlockSizeKernel.count_and_sort.kernelName();
    const ttir_count_sort = try generateCountAndSortKernelTtir(allocator, count_sort_config);
    defer allocator.free(ttir_count_sort);

    const flat_experts = topk_ids_.reshape(.{ .g = num_assignments });
    var cumsums = Tensor.zeroes(Shape.init(.{ .g = num_experts + 1 }, .i32));
    var expert_ids = Tensor.zeroes(Shape.init(.{ .g = max_num_m_blocks }, .i32));
    var sorted_token_ids = Tensor.zeroes(Shape.init(.{ .g = max_num_tokens_padded }, .i32));
    var num_tokens_post_padded = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));

    {
        const inputs = .{
            flat_experts,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            cumsums,
        };
        const outputs = ops.triton(inputs, .{ sorted_token_ids.shape(), expert_ids.shape(), num_tokens_post_padded.shape(), cumsums.shape() }, .{
            .name = "moe_align_block_size_kernel",
            .ir = ttir_align,
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
        sorted_token_ids = outputs[0];
        expert_ids = outputs[1];
        num_tokens_post_padded = outputs[2];
        cumsums = outputs[3];
    }

    {
        const inputs = .{
            flat_experts,
            sorted_token_ids,
            cumsums,
        };
        const outputs = ops.triton(inputs, .{ sorted_token_ids.shape(), cumsums.shape() }, .{
            .name = "count_and_sort_expert_tokens_kernel",
            .ir = ttir_count_sort,
            .grid = .{ @intCast(sort_grid_x), 1, 1 },
            .num_stages = 1,
            .num_warps = 4,
            .output_operand_aliases = &.{
                .{ .output_index = 0, .operand_index = 1 },
                .{ .output_index = 1, .operand_index = 2 },
            },
        });
        sorted_token_ids = outputs[0];
        cumsums = outputs[1];
    }

    return .{ sorted_token_ids, expert_ids, num_tokens_post_padded };
}

fn quantizePerTokenGroupFp8(
    allocator: std.mem.Allocator,
    x: Tensor,
    group_size: i64,
) !struct { Tensor, Tensor } {
    stdx.debug.assert(x.rank() == 2, "expected a rank-2 activation matrix, got {f}", .{x.shape()});
    stdx.debug.assert(@mod(x.dim(1), group_size) == 0, "activation width must be divisible by group size {d}, got {d}", .{ group_size, x.dim(1) });

    const config = QuantGenerationConfig{
        .num_rows = @intCast(x.dim(0)),
        .num_columns = @intCast(x.dim(1)),
        .group_size = @intCast(group_size),
        .block = @intCast(group_size),
        .input_dtype = x.dtype(),
        .output_dtype = .f8e4m3fn,
        .scale_dtype = .bf16,
        .eps = 1e-6,
        .fp8_min = -448.0,
        .fp8_max = 448.0,
        .use_ue8m0 = false,
    };

    const ir = try generatePerTokenGroupQuantFp8KernelTtir(allocator, config);
    defer allocator.free(ir);

    const groups_per_row = @divExact(x.dim(1), group_size);
    const quantized = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .feature = x.dim(1) }, .f8e4m3fn));
    const scales = Tensor.zeroes(Shape.init(.{ .token = x.dim(0), .group = groups_per_row }, .bf16));

    const inputs = .{
        x,
        Tensor.constant(.{ .i64 = group_size }).reshape(.{1}),
        Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
        Tensor.constant(.{ .i64 = x.dim(1) }).reshape(.{1}),
        Tensor.scalar(1e-6, .f32),
    };

    const outputs = ops.triton(inputs, .{ quantized.shape(), scales.shape() }, .{
        .name = "per_token_group_quant_fp8",
        .ir = ir,
        .grid = .{ @intCast(x.dim(0) * groups_per_row), 1, 1 },
        .num_stages = 1,
        .num_warps = 1,
        .output_operand_aliases = &.{},
    });

    return .{ outputs[0], outputs[1] };
}

// =============================================================================
// TTIR generation
// =============================================================================

fn dslDType(dt: DataType) kernel_dsl.DType {
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

fn ptrArg(name: []const u8, dtype: DataType) DslArgSpec {
    return .{ .name = name, .kind = .{ .ptr = .{ .dtype = dslDType(dtype) } } };
}

fn currentMlirCtx() *mlir.Context {
    return zml.module.CompilationContext.current().mlir_ctx;
}

// -----------------------------------------------------------------------------
// Helpers — small idioms used by every kernel body. All return DSL Values.
// -----------------------------------------------------------------------------

/// Splat `scalar` to a tensor of `shape` (element type taken from `scalar`).
inline fn splatTo(k: *Kernel, scalar: DslValue, shape: []const i64) DslValue {
    return k.splat(scalar, shape);
}

/// Splat a constant i32 scalar to a tensor.
inline fn splatI32(k: *Kernel, value: i32, shape: []const i64) DslValue {
    return k.splat(k.constI32(value), shape);
}

/// Splat a constant i64 scalar to a tensor.
inline fn splatI64(k: *Kernel, value: i64, shape: []const i64) DslValue {
    return k.splat(k.constI64(value), shape);
}

/// Signed `<` between a tensor and a scalar (broadcast on rhs).
inline fn cmpiTensorLtScalar(k: *Kernel, lhs: DslValue, scalar: DslValue, shape: []const i64) DslValue {
    return k.cmpi(.slt, lhs, k.splat(scalar, shape));
}

/// `tt.expand_dims` then `tt.broadcast` to grow a 1-D tensor to a 2-D
/// `[m, n]` tensor where `axis=0` means the value lives in the column axis
/// (i.e. tensor was shape `[n]` and is broadcast over `m`).
inline fn broadcast2d(k: *Kernel, vec: DslValue, axis: i32, m: i64, n: i64) DslValue {
    const expanded_shape: [2]i64 = if (axis == 0) .{ 1, n } else .{ m, 1 };
    const exp = k.expandDims(vec, axis, &expanded_shape);
    const target_shape: [2]i64 = .{ m, n };
    return k.broadcast(exp, &target_shape);
}

/// Convenience: load an i64 scalar from a `!tt.ptr<i64>` arg.
inline fn loadI64(k: *Kernel, ptr: DslValue) DslValue {
    return k.load(ptr, k.scalarTy(.i64), .{});
}

/// Convenience: load an i32 scalar from a `!tt.ptr<i32>` arg.
inline fn loadI32(k: *Kernel, ptr: DslValue) DslValue {
    return k.load(ptr, k.scalarTy(.i32), .{});
}

inline fn zeroI32(k: *Kernel) DslValue {
    return k.constI32(0);
}

inline fn zeroI64(k: *Kernel) DslValue {
    return k.constI64(0);
}

// =============================================================================
// count_and_sort_expert_tokens_kernel
// =============================================================================

fn generateCountAndSortKernelTtir(allocator: std.mem.Allocator, config: AlignBlockSizeGenerationConfig) ![:0]const u8 {
    const ctx = currentMlirCtx();

    // Signature: 3 input ptrs + 2 output ptrs (output_operand_aliases ties
    // out0->sorted_token_ids and out1->cumsum, so the trailing args are unused
    // by the body — they exist purely for XLA's custom_call binding).
    const args = [_]DslArgSpec{
        ptrArg("topk_ids_ptr", .i32),
        ptrArg("sorted_token_ids_ptr", .i32),
        ptrArg("cumsum_ptr", .i32),
        ptrArg("out0_ptr", .i32),
        ptrArg("out1_ptr", .i32),
    };

    var kernel = try Kernel.init(allocator, ctx, "count_and_sort_expert_tokens_kernel", &args, &.{});
    defer kernel.deinit();

    const k = &kernel;
    const block_size: i64 = @intCast(config.sort_block_size);
    const numel: i32 = @intCast(config.numel);
    const num_experts: i32 = @intCast(config.num_experts);

    const topk_ids = k.arg(0);
    const sorted_token_ids = k.arg(1);
    const cumsum = k.arg(2);
    _ = k.arg(3);
    _ = k.arg(4);

    const pid = k.programId(.x);
    const num_progs = k.numPrograms(.x);
    const block_size_c = k.constI32(@intCast(block_size));
    const numel_c = k.constI32(numel);
    const num_experts_c = k.constI32(num_experts);
    const one_i32 = k.constI32(1);

    // token_offs = arange(0, BLOCK_SIZE)
    const token_offs = k.makeRange(0, @intCast(block_size));

    // token_start_init = pid * BLOCK_SIZE
    const token_start_init = k.muli(pid, block_size_c);

    // step = num_progs * BLOCK_SIZE
    const step = k.muli(num_progs, block_size_c);

    const Ctx = struct {
        topk_ids: DslValue,
        sorted_token_ids: DslValue,
        cumsum: DslValue,
        token_offs: DslValue,
        numel_c: DslValue,
        num_experts_c: DslValue,
        one_i32: DslValue,
        step: DslValue,
        block_size: i64,
    };
    const c: Ctx = .{
        .topk_ids = topk_ids,
        .sorted_token_ids = sorted_token_ids,
        .cumsum = cumsum,
        .token_offs = token_offs,
        .numel_c = numel_c,
        .num_experts_c = num_experts_c,
        .one_i32 = one_i32,
        .step = step,
        .block_size = block_size,
    };

    const before = struct {
        fn call(kk: *Kernel, args_: []const DslValue, ctx_: Ctx) struct { cond: DslValue, forwarded: []const DslValue } {
            const ts = args_[0];
            const cond = kk.cmpi(.slt, ts, ctx_.numel_c);
            const out = kk.arena.allocator().alloc(DslValue, 1) catch @panic("OOM");
            out[0] = ts;
            return .{ .cond = cond, .forwarded = out };
        }
    }.call;

    const after = struct {
        fn call(kk: *Kernel, args_: []const DslValue, ctx_: Ctx) []const DslValue {
            const ts = args_[0];

            // offs = ts + token_offs
            const ts_splat = kk.splat(ts, &.{ctx_.block_size});
            const offs = kk.addi(ts_splat, ctx_.token_offs);

            // mask = offs < NUMEL
            const numel_splat = kk.splat(ctx_.numel_c, &.{ctx_.block_size});
            const mask = kk.cmpi(.slt, offs, numel_splat);

            // expert_vals = load(topk_ids + offs, mask=mask, other=NUM_EXPERTS) (i32)
            const topk_ptrs = kk.addptr(kk.splat(ctx_.topk_ids, &.{ctx_.block_size}), offs);
            const num_experts_splat = kk.splat(ctx_.num_experts_c, &.{ctx_.block_size});
            const i32_tensor_ty = kk.tensorTy(&.{ctx_.block_size}, .i32);
            const expert_vals = kk.load(topk_ptrs, i32_tensor_ty, .{
                .mask = mask.inner,
                .other = num_experts_splat.inner,
            });

            // valid = mask & (expert_vals < NUM_EXPERTS)
            const lt = kk.cmpi(.slt, expert_vals, num_experts_splat);
            const valid = kk.andi(mask, lt);

            // rank = atomic_add(cumsum + expert_vals, 1, mask=valid)
            const cumsum_ptrs = kk.addptr(kk.splat(ctx_.cumsum, &.{ctx_.block_size}), expert_vals);
            const ones_tensor = kk.splat(ctx_.one_i32, &.{ctx_.block_size});
            const rank = kk.atomicRmw(.add, cumsum_ptrs, ones_tensor, .{
                .mask = valid.inner,
                .sem = .relaxed,
                .scope = .gpu,
            });

            // store(sorted_token_ids + rank, offs, mask=valid)
            const sorted_ptrs = kk.addptr(kk.splat(ctx_.sorted_token_ids, &.{ctx_.block_size}), rank);
            kk.store(sorted_ptrs, offs, .{ .mask = valid.inner });

            // ts += step
            const next = kk.addi(ts, ctx_.step);
            const out = kk.arena.allocator().alloc(DslValue, 1) catch @panic("OOM");
            out[0] = next;
            return out;
        }
    }.call;

    _ = k.whileLoop(&.{token_start_init}, &.{k.scalarTy(.i32)}, before, after, c);

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize count_and_sort_expert_tokens_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// per_token_group_quant_fp8
// =============================================================================

fn generatePerTokenGroupQuantFp8KernelTtir(allocator: std.mem.Allocator, config: QuantGenerationConfig) ![:0]const u8 {
    const ctx = currentMlirCtx();
    const args = [_]DslArgSpec{
        ptrArg("y_ptr", config.input_dtype),
        ptrArg("group_size_ptr", .i64),
        ptrArg("y_num_columns_ptr", .i64),
        ptrArg("y_row_stride_ptr", .i64),
        ptrArg("eps_ptr", .f32),
        ptrArg("y_q_ptr", config.output_dtype),
        ptrArg("y_s_ptr", config.scale_dtype),
    };
    var kernel = try Kernel.init(allocator, ctx, "per_token_group_quant_fp8", &args, &.{});
    defer kernel.deinit();

    const k = &kernel;
    const block: i64 = @intCast(config.block);

    const y_ptr = k.arg(0);
    const group_size_ptr = k.arg(1);
    const y_num_columns_ptr = k.arg(2);
    const y_row_stride_ptr = k.arg(3);
    const eps_ptr = k.arg(4);
    const y_q_ptr = k.arg(5);
    const y_s_ptr = k.arg(6);

    // Load runtime scalars.
    const group_size = loadI64(k, group_size_ptr);
    const y_num_columns = loadI64(k, y_num_columns_ptr);
    const y_row_stride = loadI64(k, y_row_stride_ptr);
    const eps = k.load(eps_ptr, k.scalarTy(.f32), .{});

    // groups_per_row = y_num_columns // group_size  (scalar i64)
    const groups_per_row = k.divsi(y_num_columns, group_size);

    // g_id = program_id (i32) → cast to i64 for offset math
    const g_id_i32 = k.programId(.x);
    const g_id = k.extsi(g_id_i32, k.scalarTy(.i64));

    // row = g_id // groups_per_row, row_g_id = g_id % groups_per_row
    const row = k.divsi(g_id, groups_per_row);
    const row_g_id = k.remsi(g_id, groups_per_row);

    // y_ptr_offset = row*y_row_stride + row_g_id*group_size
    const row_off = k.muli(row, y_row_stride);
    const grp_off = k.muli(row_g_id, group_size);
    const y_ptr_offset = k.addi(row_off, grp_off);
    const y_ptr_shifted = k.addptr(y_ptr, y_ptr_offset);

    // y_q_ptr_offset = g_id*group_size
    const y_q_offset = k.muli(g_id, group_size);
    const y_q_ptr_shifted = k.addptr(y_q_ptr, y_q_offset);

    // y_s_ptr += g_id (single scalar)
    const y_s_ptr_shifted = k.addptr(y_s_ptr, g_id);

    // cols = arange(0, BLOCK)  (tensor<BLOCK x i32>)
    const cols = k.makeRange(0, @intCast(block));
    // Promote cols → i64 for ptr offsets.
    const cols_i64 = k.extsi(cols, k.tensorTy(&.{block}, .i64));

    // mask = cols < group_size  (need group_size as i32 splat for cmp)
    // group_size is i64; truncate for comparison with cols (i32).
    const group_size_i32 = k.trunci(group_size, k.scalarTy(.i32));
    const group_size_splat = k.splat(group_size_i32, &.{block});
    const mask = k.cmpi(.slt, cols, group_size_splat);

    // y_load_ptr = y_ptr_shifted + cols (tensor of ptrs), load tensor of input dtype.
    const y_load_ptrs = k.addptr(k.splat(y_ptr_shifted, &.{block}), cols_i64);
    const input_dt = dslDType(config.input_dtype);
    const input_elem = k.scalarTy(input_dt);
    const input_tensor_ty = k.tensorTy(&.{block}, input_dt);
    const f32_tensor_ty = k.tensorTy(&.{block}, .f32);

    // Zero "other" value, in input element type, splatted to the block tensor.
    const zero_f32_scalar = k.constF32(0.0);
    const zero_input_scalar = if (input_dt == .f32)
        zero_f32_scalar
    else
        k.fpToFp(zero_f32_scalar, input_elem, .{ .rounding = .rtne });
    const zero_input_tensor = k.splat(zero_input_scalar, &.{block});

    const y_loaded = k.load(y_load_ptrs, input_tensor_ty, .{
        .mask = mask.inner,
        .other = zero_input_tensor.inner,
    });

    // y_f32 = y_loaded.to(f32)  (extf if bf16/f16, identity if f32)
    const y_f32 = if (input_dt == .f32)
        y_loaded
    else
        k.extf(y_loaded, f32_tensor_ty);

    // _absmax = max(reduce_max(abs(y)), eps)
    const abs_y = k.absf(y_f32);
    const f32_ty = k.scalarTy(.f32);
    const max_combine = struct {
        fn combine(kk: *Kernel, lhs: DslValue, rhs: DslValue, _: void) DslValue {
            return kk.maximumf(lhs, rhs);
        }
    }.combine;
    const absmax_raw = k.reduce(abs_y, 0, f32_ty, f32_ty, max_combine, {});
    const absmax = k.maximumf(absmax_raw, eps);

    // scale_raw = absmax * (1.0 / fp8_max)
    const inv_fp8_max = k.constF32(1.0 / config.fp8_max);
    const scale_raw = k.mulf(absmax, inv_fp8_max);

    // y_s = use_ue8m0 ? exp2(ceil(log2(scale_raw))) : scale_raw
    const y_s_scalar = if (config.use_ue8m0)
        k.exp2(k.ceil(k.log2(scale_raw)))
    else
        scale_raw;

    // y_q = clamp(y_f32 / y_s, fp8_min, fp8_max).to(output_dtype)
    const y_s_splat = k.splat(y_s_scalar, &.{block});
    const y_div = k.divf(y_f32, y_s_splat);
    const fp8_min_splat = k.splat(k.constF32(config.fp8_min), &.{block});
    const fp8_max_splat = k.splat(k.constF32(config.fp8_max), &.{block});
    const y_clamped = k.clampf(y_div, fp8_min_splat, fp8_max_splat, .none);
    const out_dt = dslDType(config.output_dtype);
    const out_tensor_ty = k.tensorTy(&.{block}, out_dt);
    const y_q = k.fpToFp(y_clamped, out_tensor_ty, .{ .rounding = .rtne });

    // store(y_q_ptr_shifted + cols, y_q, mask)
    const y_q_ptrs = k.addptr(k.splat(y_q_ptr_shifted, &.{block}), cols_i64);
    k.store(y_q_ptrs, y_q, .{ .mask = mask.inner });

    // store(y_s_ptr_shifted, y_s) — single element, possibly cast to scale_dtype.
    const scale_dt = dslDType(config.scale_dtype);
    const y_s_to_store = if (scale_dt == .f32)
        y_s_scalar
    else
        k.fpToFp(y_s_scalar, k.scalarTy(scale_dt), .{ .rounding = .rtne });
    k.store(y_s_ptr_shifted, y_s_to_store, .{});

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize per_token_group_quant_fp8 TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// moe_align_block_size_kernel
// =============================================================================

fn generateAlignBlockSizeKernelTtir(allocator: std.mem.Allocator, config: AlignBlockSizeGenerationConfig) ![:0]const u8 {
    const ctx = currentMlirCtx();

    const args = [_]DslArgSpec{
        ptrArg("topk_ids_ptr", .i32),
        ptrArg("sorted_token_ids_ptr", .i32),
        ptrArg("expert_ids_ptr", .i32),
        ptrArg("num_tokens_post_pad_ptr", .i32),
        ptrArg("cumsum_ptr", .i32),
        ptrArg("out0_ptr", .i32),
        ptrArg("out1_ptr", .i32),
        ptrArg("out2_ptr", .i32),
        ptrArg("out3_ptr", .i32),
    };

    var kernel = try Kernel.init(allocator, ctx, "moe_align_block_size_kernel", &args, &.{});
    defer kernel.deinit();
    const k = &kernel;

    const block_size_m: i64 = @intCast(config.block_size_m);
    const numel: i32 = @intCast(config.numel);
    const num_experts: i32 = @intCast(config.num_experts);
    const padded_num_experts: i64 = @intCast(config.padded_num_experts);
    const max_num_tokens_padded: i32 = @intCast(config.max_num_tokens_padded);
    const max_num_m_blocks: i64 = @intCast(config.max_num_m_blocks);
    const hist_block: i64 = @intCast(config.hist_block);

    const topk_ids = k.arg(0);
    const sorted_token_ids = k.arg(1);
    const expert_ids = k.arg(2);
    const num_tokens_post_pad = k.arg(3);
    const cumsum = k.arg(4);
    _ = k.arg(5);
    _ = k.arg(6);
    _ = k.arg(7);
    _ = k.arg(8);

    const pid = k.programId(.x);
    const c1 = k.constI32(1);
    const cond_pid_eq_1 = k.cmpi(.eq, pid, c1);

    // pid==1 branch: fill sorted_token_ids with NUMEL using a hist_block-strided loop.
    // pid==0 branch: histogram + cumsum + assignment.
    //
    // We can't return early in TTIR; both branches must yield matching types.
    // We model this as: scf.if (pid==1) { fill_branch; } else { compute_branch; }
    // with no result types. Both branches share the same `body_ctx`.

    const AlignCtx = struct {
        topk_ids: DslValue,
        sorted_token_ids: DslValue,
        expert_ids: DslValue,
        num_tokens_post_pad: DslValue,
        cumsum: DslValue,
        block_size_m: i64,
        numel: i32,
        num_experts: i32,
        padded_num_experts: i64,
        max_num_tokens_padded: i32,
        max_num_m_blocks: i64,
        hist_block: i64,
    };

    const ax: AlignCtx = .{
        .topk_ids = topk_ids,
        .sorted_token_ids = sorted_token_ids,
        .expert_ids = expert_ids,
        .num_tokens_post_pad = num_tokens_post_pad,
        .cumsum = cumsum,
        .block_size_m = block_size_m,
        .numel = numel,
        .num_experts = num_experts,
        .padded_num_experts = padded_num_experts,
        .max_num_tokens_padded = max_num_tokens_padded,
        .max_num_m_blocks = max_num_m_blocks,
        .hist_block = hist_block,
    };

    const fill_then = struct {
        fn call(kk: *Kernel, c: AlignCtx) []const DslValue {
            const block = c.hist_block;
            const fill_offs = kk.makeRange(0, @intCast(block));
            const max_padded_c = kk.constI32(c.max_num_tokens_padded);
            const numel_c2 = kk.constI32(c.numel);
            const block_c = kk.constI32(@intCast(block));

            // for start in range(0, MAX_NUM_TOKENS_PADDED, HIST_BLOCK)
            const lb = kk.constI32(0);
            const ub = max_padded_c;
            const Inner = struct {
                sorted_token_ids: DslValue,
                fill_offs: DslValue,
                max_padded_c: DslValue,
                numel_c: DslValue,
                block: i64,
            };
            const inner: Inner = .{
                .sorted_token_ids = c.sorted_token_ids,
                .fill_offs = fill_offs,
                .max_padded_c = max_padded_c,
                .numel_c = numel_c2,
                .block = block,
            };
            const body = struct {
                fn b(kkk: *Kernel, iv: DslValue, _: []const DslValue, ic: Inner) []const DslValue {
                    const start_splat = kkk.splat(iv, &.{ic.block});
                    const offs = kkk.addi(start_splat, ic.fill_offs);
                    const max_padded_splat = kkk.splat(ic.max_padded_c, &.{ic.block});
                    const mask = kkk.cmpi(.slt, offs, max_padded_splat);
                    const numel_splat = kkk.splat(ic.numel_c, &.{ic.block});
                    const sorted_ptrs = kkk.addptr(kkk.splat(ic.sorted_token_ids, &.{ic.block}), offs);
                    kkk.store(sorted_ptrs, numel_splat, .{ .mask = mask.inner });
                    return &.{};
                }
            }.b;
            _ = kk.forLoop(lb, ub, block_c, &.{}, body, inner);
            return &.{};
        }
    }.call;

    const compute_else = struct {
        fn call(kk: *Kernel, c: AlignCtx) []const DslValue {
            const i32_ty_ = kk.scalarTy(.i32);

            // expert_offs = arange(0, PADDED_NUM_EXPERTS) (i32)
            const expert_offs = kk.makeRange(0, @intCast(c.padded_num_experts));
            // token_offs = arange(0, HIST_BLOCK)
            const token_offs = kk.makeRange(0, @intCast(c.hist_block));

            // expert_mask = expert_offs < NUM_EXPERTS
            const num_experts_splat = kk.splat(kk.constI32(c.num_experts), &.{c.padded_num_experts});
            const expert_mask = kk.cmpi(.slt, expert_offs, num_experts_splat);

            // Histogram accumulator: counts: tensor<PADDED_NUM_EXPERTS x i32>, init 0.
            const counts_init = kk.splat(kk.constI32(0), &.{c.padded_num_experts});

            // for token_start in range(0, NUMEL, HIST_BLOCK):
            const numel_c = kk.constI32(c.numel);
            const hist_block_c = kk.constI32(@intCast(c.hist_block));
            const zero = kk.constI32(0);

            const HistCtx = struct {
                topk_ids: DslValue,
                token_offs: DslValue,
                numel: i32,
                num_experts: i32,
                hist_block: i64,
                padded_num_experts: i64,
            };
            const hist_ctx: HistCtx = .{
                .topk_ids = c.topk_ids,
                .token_offs = token_offs,
                .numel = c.numel,
                .num_experts = c.num_experts,
                .hist_block = c.hist_block,
                .padded_num_experts = c.padded_num_experts,
            };

            const hist_body = struct {
                fn b(kkk: *Kernel, iv: DslValue, iter: []const DslValue, hc: HistCtx) []const DslValue {
                    const counts = iter[0];
                    const offs = kkk.addi(kkk.splat(iv, &.{hc.hist_block}), hc.token_offs);
                    const mask = kkk.cmpi(.slt, offs, kkk.splat(kkk.constI32(hc.numel), &.{hc.hist_block}));
                    const topk_ptrs = kkk.addptr(kkk.splat(hc.topk_ids, &.{hc.hist_block}), offs);
                    const num_experts_other = kkk.splat(kkk.constI32(hc.num_experts), &.{hc.hist_block});
                    const i32_tensor = kkk.tensorTy(&.{hc.hist_block}, .i32);
                    const expert_vals = kkk.load(topk_ptrs, i32_tensor, .{
                        .mask = mask.inner,
                        .other = num_experts_other.inner,
                    });
                    const lt = kkk.cmpi(.slt, expert_vals, num_experts_other);
                    const valid = kkk.andi(mask, lt);
                    // tt.histogram returns tensor<PADDED_NUM_EXPERTS x i32>.
                    const hist_ty = kkk.tensorTy(&.{hc.padded_num_experts}, .i32);
                    const h = kkk.histogram(expert_vals, valid, hist_ty);
                    const new_counts = kkk.addi(counts, h);
                    const out = kkk.arena.allocator().alloc(DslValue, 1) catch @panic("OOM");
                    out[0] = new_counts;
                    return out;
                }
            }.b;

            const counts_results = kk.forLoop(zero, numel_c, hist_block_c, &.{counts_init}, hist_body, hist_ctx);
            const counts = counts_results[0];

            // padded_counts = where(expert_mask, cdiv(counts, BLOCK_SIZE_M)*BLOCK_SIZE_M, 0)
            const block_size_m_splat = kk.splat(kk.constI32(@intCast(c.block_size_m)), &.{c.padded_num_experts});
            // cdiv(a, b) = (a + b - 1) / b  → but Triton uses ceildivsi. We have ceildivsi.
            const padded_counts_full = kk.muli(kk.ceildivsi(counts, block_size_m_splat), block_size_m_splat);
            const zero_tensor = kk.splat(kk.constI32(0), &.{c.padded_num_experts});
            const padded_counts = kk.select(expert_mask, padded_counts_full, zero_tensor);

            // padded_cumsum = cumsum(padded_counts) via tt.scan add along axis 0
            const addi_combine = struct {
                fn combine(kkk: *Kernel, lhs: DslValue, rhs: DslValue, _: void) DslValue {
                    return kkk.addi(lhs, rhs);
                }
            }.combine;
            const scan_result_ty = kk.tensorTy(&.{c.padded_num_experts}, .i32);
            const padded_cumsum = kk.scan(padded_counts, 0, false, i32_ty_, scan_result_ty, addi_combine, {});

            // starts = padded_cumsum - padded_counts
            const starts = kk.subi(padded_cumsum, padded_counts);

            // total_tokens_post_pad = sum(padded_counts)
            const total = kk.reduce(padded_counts, 0, i32_ty_, i32_ty_, addi_combine, {});

            // store(cumsum + expert_offs, starts, mask=expert_mask)
            const cumsum_ptrs = kk.addptr(kk.splat(c.cumsum, &.{c.padded_num_experts}), expert_offs);
            kk.store(cumsum_ptrs, starts, .{ .mask = expert_mask.inner });

            // store(cumsum + NUM_EXPERTS, total)
            const cumsum_ne_ptr = kk.addptr(c.cumsum, kk.constI32(c.num_experts));
            kk.store(cumsum_ne_ptr, total, .{});

            // store(num_tokens_post_pad, total)
            kk.store(c.num_tokens_post_pad, total, .{});

            // Block-to-expert assignment: for block_start in 0..MAX_NUM_M_BLOCKS step HIST_BLOCK:
            const block_offs = kk.makeRange(0, @intCast(c.hist_block));
            const max_blocks_c = kk.constI32(@intCast(c.max_num_m_blocks));

            const AssignCtx = struct {
                cumsum: DslValue,
                expert_ids: DslValue,
                block_offs: DslValue,
                block_size_m: i64,
                num_experts: i32,
                hist_block: i64,
                max_num_m_blocks: i64,
            };
            const assign_ctx: AssignCtx = .{
                .cumsum = c.cumsum,
                .expert_ids = c.expert_ids,
                .block_offs = block_offs,
                .block_size_m = c.block_size_m,
                .num_experts = c.num_experts,
                .hist_block = c.hist_block,
                .max_num_m_blocks = c.max_num_m_blocks,
            };
            const assign_body = struct {
                fn b(kkk: *Kernel, block_start: DslValue, _: []const DslValue, ac: AssignCtx) []const DslValue {
                    const block_ids = kkk.addi(kkk.splat(block_start, &.{ac.hist_block}), ac.block_offs);
                    const max_blocks_splat = kkk.splat(kkk.constI32(@intCast(ac.max_num_m_blocks)), &.{ac.hist_block});
                    const block_mask = kkk.cmpi(.slt, block_ids, max_blocks_splat);
                    const bsm_splat = kkk.splat(kkk.constI32(@intCast(ac.block_size_m)), &.{ac.hist_block});
                    const block_offsets = kkk.muli(block_ids, bsm_splat);
                    const minus_one_splat = kkk.splat(kkk.constI32(-1), &.{ac.hist_block});

                    // Inner expert loop — runtime scf.for over [0, NUM_EXPERTS).
                    const ExpCtx = struct {
                        cumsum: DslValue,
                        block_offsets: DslValue,
                        block_mask: DslValue,
                        hist_block: i64,
                    };
                    const ec: ExpCtx = .{
                        .cumsum = ac.cumsum,
                        .block_offsets = block_offsets,
                        .block_mask = block_mask,
                        .hist_block = ac.hist_block,
                    };
                    const exp_body = struct {
                        fn eb(kkkk: *Kernel, iv: DslValue, iter: []const DslValue, ee: ExpCtx) []const DslValue {
                            const cur = iter[0];
                            const start_ptr = kkkk.addptr(ee.cumsum, iv);
                            const start_v = loadI32(kkkk, start_ptr);
                            const next_iv = kkkk.addi(iv, kkkk.constI32(1));
                            const end_ptr = kkkk.addptr(ee.cumsum, next_iv);
                            const end_v = loadI32(kkkk, end_ptr);
                            const start_splat = kkkk.splat(start_v, &.{ee.hist_block});
                            const end_splat = kkkk.splat(end_v, &.{ee.hist_block});
                            const ge = kkkk.cmpi(.sge, ee.block_offsets, start_splat);
                            const lt = kkkk.cmpi(.slt, ee.block_offsets, end_splat);
                            const in_range = kkkk.andi(kkkk.andi(ee.block_mask, ge), lt);
                            const new_cur = kkkk.select(in_range, kkkk.splat(iv, &.{ee.hist_block}), cur);
                            const out = kkkk.arena.allocator().alloc(DslValue, 1) catch @panic("OOM");
                            out[0] = new_cur;
                            return out;
                        }
                    }.eb;
                    const lb = kkk.constI32(0);
                    const ub = kkk.constI32(ac.num_experts);
                    const step = kkk.constI32(1);
                    const exp_results = kkk.forLoop(lb, ub, step, &.{minus_one_splat}, exp_body, ec);
                    const block_expert = exp_results[0];

                    const expert_ptrs = kkk.addptr(kkk.splat(ac.expert_ids, &.{ac.hist_block}), block_ids);
                    kkk.store(expert_ptrs, block_expert, .{ .mask = block_mask.inner });
                    return &.{};
                }
            }.b;

            const block_step = kk.constI32(@intCast(c.hist_block));
            const block_lb = kk.constI32(0);
            _ = kk.forLoop(block_lb, max_blocks_c, block_step, &.{}, assign_body, assign_ctx);
            return &.{};
        }
    }.call;

    _ = k.ifThenElse(cond_pid_eq_1, &.{}, fill_then, compute_else, ax);

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize moe_align_block_size_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// fused_moe_kernel
// =============================================================================

fn generateFusedMoeKernelTtir(allocator: std.mem.Allocator, config: GenerationConfig) ![:0]const u8 {
    const ctx = currentMlirCtx();

    // Currently only the no-quantization, no-bias path is supported by the
    // surrounding `validateOptions` check. Refuse early if the config wanders
    // outside that scope so we don't emit half-configured TTIR.
    if (config.use_fp8_w8a8 or config.use_int8_w8a8 or config.use_int8_w8a16 or config.has_bias or config.per_channel_quant) {
        log.err("fused_moe_kernel: unsupported config (fp8/int8/bias/per_channel)", .{});
        return error.TritonTtirGenerationFailed;
    }

    const args = [_]DslArgSpec{
        ptrArg("a_ptr", config.a_dtype),
        ptrArg("b_ptr", config.b_dtype),
        ptrArg("b_bias_ptr", config.b_bias_dtype orelse config.c_dtype),
        ptrArg("a_scale_ptr", config.a_scale_dtype orelse .f32),
        ptrArg("b_scale_ptr", config.b_scale_dtype orelse .f32),
        ptrArg("topk_weights_ptr", config.topk_weights_dtype orelse .f32),
        ptrArg("sorted_token_ids_ptr", .i32),
        ptrArg("expert_ids_ptr", .i32),
        ptrArg("num_tokens_post_padded_ptr", .i32),
        ptrArg("N_ptr", .i64),
        ptrArg("K_ptr", .i64),
        ptrArg("EM_ptr", .i64),
        ptrArg("num_valid_tokens_ptr", .i64),
        ptrArg("stride_am_ptr", .i64),
        ptrArg("stride_be_ptr", .i64),
        ptrArg("stride_bn_ptr", .i64),
        ptrArg("stride_cm_ptr", .i64),
        ptrArg("stride_asm_ptr", .i64),
        ptrArg("stride_ask_ptr", .i64),
        ptrArg("stride_bse_ptr", .i64),
        ptrArg("stride_bsk_ptr", .i64),
        ptrArg("stride_bsn_ptr", .i64),
        ptrArg("stride_bbe_ptr", .i64),
        ptrArg("stride_bbn_ptr", .i64),
        ptrArg("c_ptr", config.c_dtype),
    };

    var kernel = try Kernel.init(allocator, ctx, "fused_moe_kernel", &args, &.{});
    defer kernel.deinit();
    const k = &kernel;

    // Constexpr parameters baked from config.
    const block_m: i64 = @intCast(config.block_size_m);
    const block_n: i64 = @intCast(config.block_size_n);
    const block_k: i64 = @intCast(config.block_size_k);
    const group_size_m: i64 = @intCast(config.group_size_m);
    const top_k: i64 = @intCast(config.top_k);

    const i64_ty = k.scalarTy(.i64);

    const a_ptr = k.arg(0);
    const b_ptr = k.arg(1);
    _ = k.arg(2); // b_bias_ptr — unused (HAS_BIAS=false enforced)
    _ = k.arg(3); // a_scale_ptr — unused (no quant)
    _ = k.arg(4); // b_scale_ptr — unused (no quant)
    const topk_weights_ptr = k.arg(5);
    const sorted_token_ids_ptr = k.arg(6);
    const expert_ids_ptr = k.arg(7);
    const num_tokens_post_padded_ptr = k.arg(8);
    const n_ptr = k.arg(9);
    const k_ptr = k.arg(10);
    const em_ptr = k.arg(11);
    const num_valid_tokens_ptr = k.arg(12);
    const stride_am_ptr = k.arg(13);
    const stride_be_ptr = k.arg(14);
    const stride_bn_ptr = k.arg(15);
    const stride_cm_ptr = k.arg(16);
    _ = k.arg(17); // stride_asm — unused (no quant)
    _ = k.arg(18); // stride_ask — unused (no quant)
    _ = k.arg(19); // stride_bse — unused (no quant)
    _ = k.arg(20); // stride_bsk — unused (no quant)
    _ = k.arg(21); // stride_bsn — unused (no quant)
    _ = k.arg(22); // stride_bbe — unused (no bias)
    _ = k.arg(23); // stride_bbn — unused (no bias)
    const c_ptr = k.arg(24);

    // Load runtime scalars and clamp to multiples of 16 (matches Python).
    const sixteen = k.constI64(16);
    const block_floor = struct {
        fn f(kk: *Kernel, v: DslValue, sixteen_: DslValue) DslValue {
            return kk.muli(kk.divsi(v, sixteen_), sixteen_);
        }
    }.f;

    const n_raw = loadI64(k, n_ptr);
    const k_raw = loadI64(k, k_ptr);
    const em_raw = loadI64(k, em_ptr);
    const num_valid_tokens = loadI64(k, num_valid_tokens_ptr);
    const stride_am_raw = loadI64(k, stride_am_ptr);
    const stride_be_raw = loadI64(k, stride_be_ptr);
    const stride_bn_raw = loadI64(k, stride_bn_ptr);
    const stride_cm_raw = loadI64(k, stride_cm_ptr);

    const n_block = block_floor(k, n_raw, sixteen);
    const k_block = block_floor(k, k_raw, sixteen);
    const em_block = block_floor(k, em_raw, sixteen);
    const stride_am_block = block_floor(k, stride_am_raw, sixteen);
    const stride_be_block = block_floor(k, stride_be_raw, sixteen);
    const stride_bn_block = block_floor(k, stride_bn_raw, sixteen);
    const stride_cm_block = block_floor(k, stride_cm_raw, sixteen);
    // stride_ak / stride_bk / stride_cn are constexpr 1 in the original Python.
    const stride_ak_block = k.constI64(1);
    const stride_bk_block = k.constI64(1);
    const stride_cn_block = k.constI64(1);

    // pid grouping math (all i64).
    const pid_i32 = k.programId(.x);
    const pid = k.extsi(pid_i32, i64_ty);
    const block_m_c = k.constI64(block_m);
    const block_n_c = k.constI64(block_n);
    const block_k_c = k.constI64(block_k);
    const group_size_m_c = k.constI64(group_size_m);
    const num_pid_m = k.ceildivsi(em_block, block_m_c);
    const num_pid_n = k.ceildivsi(n_block, block_n_c);
    const num_pid_in_group = k.muli(group_size_m_c, num_pid_n);
    const group_id = k.divsi(pid, num_pid_in_group);
    const first_pid_m = k.muli(group_id, group_size_m_c);
    const left = k.subi(num_pid_m, first_pid_m);
    const gsm_actual = k.minsi(left, group_size_m_c);
    const pid_mod_in_group = k.remsi(pid, num_pid_in_group);
    const pid_m = k.addi(first_pid_m, k.remsi(pid_mod_in_group, gsm_actual));
    const pid_n = k.divsi(pid_mod_in_group, gsm_actual);

    // num_tokens_post_padded (load as i32 then sext to i64).
    const ntpp_i32 = loadI32(k, num_tokens_post_padded_ptr);
    const ntpp = k.extsi(ntpp_i32, i64_ty);
    const pid_m_times_block = k.muli(pid_m, block_m_c);
    const out_of_range = k.cmpi(.sge, pid_m_times_block, ntpp);

    // We model the Python `if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded:
    // return` as `scf.if out_of_range { /* noop */ } else { compute }`.
    const InRangeCtx = struct {
        a_ptr: DslValue,
        b_ptr: DslValue,
        c_ptr: DslValue,
        topk_weights_ptr: DslValue,
        sorted_token_ids_ptr: DslValue,
        expert_ids_ptr: DslValue,

        // i64 scalar values
        num_valid_tokens: DslValue,
        n_block: DslValue,
        k_block: DslValue,
        stride_am_block: DslValue,
        stride_ak_block: DslValue,
        stride_be_block: DslValue,
        stride_bk_block: DslValue,
        stride_bn_block: DslValue,
        stride_cm_block: DslValue,
        stride_cn_block: DslValue,
        block_m_c: DslValue,
        block_n_c: DslValue,
        block_k_c: DslValue,
        pid_m: DslValue,
        pid_n: DslValue,

        // constexpr
        block_m: i64,
        block_n: i64,
        block_k: i64,
        top_k: i64,
        naive: bool,
        mul_routed: bool,

        config: GenerationConfig,
    };

    const ic: InRangeCtx = .{
        .a_ptr = a_ptr,
        .b_ptr = b_ptr,
        .c_ptr = c_ptr,
        .topk_weights_ptr = topk_weights_ptr,
        .sorted_token_ids_ptr = sorted_token_ids_ptr,
        .expert_ids_ptr = expert_ids_ptr,
        .num_valid_tokens = num_valid_tokens,
        .n_block = n_block,
        .k_block = k_block,
        .stride_am_block = stride_am_block,
        .stride_ak_block = stride_ak_block,
        .stride_be_block = stride_be_block,
        .stride_bk_block = stride_bk_block,
        .stride_bn_block = stride_bn_block,
        .stride_cm_block = stride_cm_block,
        .stride_cn_block = stride_cn_block,
        .block_m_c = block_m_c,
        .block_n_c = block_n_c,
        .block_k_c = block_k_c,
        .pid_m = pid_m,
        .pid_n = pid_n,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .top_k = top_k,
        .naive = config.naive_block_assignment,
        .mul_routed = config.mul_routed_weight,
        .config = config,
    };

    const noop = struct {
        fn n(_: *Kernel, _: InRangeCtx) []const DslValue {
            return &.{};
        }
    }.n;

    const compute = struct {
        fn cmp(kk: *Kernel, c: InRangeCtx) []const DslValue {
            buildFusedKernelMain(kk, c);
            return &.{};
        }
    }.cmp;

    _ = k.ifThenElse(out_of_range, &.{}, noop, compute, ic);

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize fused_moe_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

/// Body of the fused_moe_kernel inside the `pid_m * BLOCK_M < ntpp` guard.
/// Translates the bf16 / no-bias / no-quant path of `fused_moe_kernel` from
/// `triton_kernels/moe.py` into TTIR ops.
fn buildFusedKernelMain(k: *Kernel, ic: anytype) void {
    const i32_ty = k.scalarTy(.i32);
    const i64_ty = k.scalarTy(.i64);

    const block_m = ic.block_m;
    const block_n = ic.block_n;
    const block_k = ic.block_k;
    const top_k = ic.top_k;
    const naive = ic.naive;
    const mul_routed = ic.mul_routed;

    // offs = arange(0, BLOCK_M).to(i64)
    const offs_i32 = k.makeRange(0, @intCast(block_m));
    const offs_i64_ty = k.tensorTy(&.{block_m}, .i64);
    const offs = k.extsi(offs_i32, offs_i64_ty);

    // offs_token: depends on naive_block_assignment.
    const offs_token = blk: {
        if (naive) {
            // offs_token = where(offs == 0, pid_m, num_valid_tokens)
            const zero_splat = k.splat(k.constI64(0), &.{block_m});
            const eq = k.cmpi(.eq, offs, zero_splat);
            const pid_m_splat = k.splat(ic.pid_m, &.{block_m});
            const nvt_splat = k.splat(ic.num_valid_tokens, &.{block_m});
            break :blk k.select(eq, pid_m_splat, nvt_splat);
        } else {
            // offs_token_id = pid_m * BLOCK_M + offs
            const pid_m_block = k.muli(ic.pid_m, ic.block_m_c);
            const pid_m_block_splat = k.splat(pid_m_block, &.{block_m});
            const offs_token_id = k.addi(pid_m_block_splat, offs);

            // sorted_token_ids[offs_token_id] (i32 → sext to i64)
            const sorted_ptrs = k.addptr(k.splat(ic.sorted_token_ids_ptr, &.{block_m}), offs_token_id);
            const i32_tensor = k.tensorTy(&.{block_m}, .i32);
            const loaded = k.load(sorted_ptrs, i32_tensor, .{});
            break :blk k.extsi(loaded, offs_i64_ty);
        }
    };

    // token_mask = offs_token < num_valid_tokens
    const nvt_splat = k.splat(ic.num_valid_tokens, &.{block_m});
    const token_mask = k.cmpi(.slt, offs_token, nvt_splat);

    // off_experts = expert_ids[pid_m] (i32 → sext i64)
    const expert_id_ptr = k.addptr(ic.expert_ids_ptr, k.trunci(ic.pid_m, i32_ty));
    const off_experts_i32 = loadI32(k, expert_id_ptr);
    const off_experts = k.extsi(off_experts_i32, i64_ty);

    // We branch on (off_experts == -1) to write zeros and skip.
    const minus_one = k.constI64(-1);
    const is_dead = k.cmpi(.eq, off_experts, minus_one);

    // Build offs_bn := (pid_n * BLOCK_N + arange(0, BLOCK_N).to(i64)) % n_block
    const arange_n_i32 = k.makeRange(0, @intCast(block_n));
    const arange_n = k.extsi(arange_n_i32, k.tensorTy(&.{block_n}, .i64));
    const pid_n_block = k.muli(ic.pid_n, ic.block_n_c);
    const pid_n_block_splat = k.splat(pid_n_block, &.{block_n});
    const offs_bn_pre = k.addi(pid_n_block_splat, arange_n);
    const n_block_splat = k.splat(ic.n_block, &.{block_n});
    const offs_bn = k.remsi(offs_bn_pre, n_block_splat);

    // offs_k = arange(0, BLOCK_K)  (kept i32 — matches Python which doesn't sext)
    const offs_k_i32 = k.makeRange(0, @intCast(block_k));
    const offs_k = k.extsi(offs_k_i32, k.tensorTy(&.{block_k}, .i64));

    // c_ptrs: c_ptr + stride_cm_block * offs_token[:, None] + stride_cn_block * offs_cn[None, :]
    // offs_cn = pid_n * BLOCK_N + arange(0, BLOCK_N)  (i32, used only for c address)
    const offs_cn_i32 = k.addi(k.splat(k.trunci(pid_n_block, i32_ty), &.{block_n}), arange_n_i32);
    const offs_cn = k.extsi(offs_cn_i32, k.tensorTy(&.{block_n}, .i64));

    // Both branches share the same ctx type because `ifThenElse` only takes a
    // single `body_ctx`. AliveCtx carries every value either branch needs.
    const AliveCtx = struct {
        a_ptr: kernel_dsl.Value,
        b_ptr: kernel_dsl.Value,
        c_ptr: kernel_dsl.Value,
        topk_weights_ptr: kernel_dsl.Value,
        offs_token: kernel_dsl.Value,
        token_mask: kernel_dsl.Value,
        offs_bn: kernel_dsl.Value,
        offs_cn: kernel_dsl.Value,
        offs_k: kernel_dsl.Value,
        off_experts: kernel_dsl.Value,
        k_block: kernel_dsl.Value,
        n_block: kernel_dsl.Value,
        stride_am_block: kernel_dsl.Value,
        stride_ak_block: kernel_dsl.Value,
        stride_be_block: kernel_dsl.Value,
        stride_bk_block: kernel_dsl.Value,
        stride_bn_block: kernel_dsl.Value,
        stride_cm_block: kernel_dsl.Value,
        stride_cn_block: kernel_dsl.Value,
        block_m: i64,
        block_n: i64,
        block_k: i64,
        top_k: i64,
        mul_routed: bool,
        a_dtype: kernel_dsl.DType,
        b_dtype: kernel_dsl.DType,
        compute_dtype: kernel_dsl.DType,
    };
    const ac: AliveCtx = .{
        .a_ptr = ic.a_ptr,
        .b_ptr = ic.b_ptr,
        .c_ptr = ic.c_ptr,
        .topk_weights_ptr = ic.topk_weights_ptr,
        .offs_token = offs_token,
        .token_mask = token_mask,
        .offs_bn = offs_bn,
        .offs_cn = offs_cn,
        .offs_k = offs_k,
        .off_experts = off_experts,
        .k_block = ic.k_block,
        .n_block = ic.n_block,
        .stride_am_block = ic.stride_am_block,
        .stride_ak_block = ic.stride_ak_block,
        .stride_be_block = ic.stride_be_block,
        .stride_bk_block = ic.stride_bk_block,
        .stride_bn_block = ic.stride_bn_block,
        .stride_cm_block = ic.stride_cm_block,
        .stride_cn_block = ic.stride_cn_block,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .top_k = top_k,
        .mul_routed = mul_routed,
        .a_dtype = dslDType(ic.config.a_dtype),
        .b_dtype = dslDType(ic.config.b_dtype),
        .compute_dtype = dslDType(ic.config.compute_type),
    };
    const zero_branch = struct {
        fn z(kk: *Kernel, c: AliveCtx) []const DslValue {
            const zero_acc_scalar = kk.constF32(0.0);
            const compute_elem = c.compute_dtype.toMlir(kk.ctx);
            const acc_elem = if (c.compute_dtype == .f32)
                zero_acc_scalar
            else
                kk.fpToFp(zero_acc_scalar, compute_elem, .{ .rounding = .rtne });
            const acc = kk.splat(acc_elem, &.{ c.block_m, c.block_n });

            // c_ptrs = c_ptr + stride_cm * offs_token[:,None] + stride_cn * offs_cn[None,:]
            const cm_off = kk.muli(broadcast2d(kk, c.offs_token, 1, c.block_m, c.block_n), kk.splat(c.stride_cm_block, &.{ c.block_m, c.block_n }));
            const cn_off = kk.muli(broadcast2d(kk, c.offs_cn, 0, c.block_m, c.block_n), kk.splat(c.stride_cn_block, &.{ c.block_m, c.block_n }));
            const off_total = kk.addi(cm_off, cn_off);
            const c_base_splat = kk.splat(c.c_ptr, &.{ c.block_m, c.block_n });
            const c_ptrs = kk.addptr(c_base_splat, off_total);

            // c_mask = token_mask[:,None] & (offs_cn[None,:] < n_block)
            const tm_2d = broadcast2d(kk, c.token_mask, 1, c.block_m, c.block_n);
            const n_block_splat2 = kk.splat(c.n_block, &.{c.block_n});
            const cn_lt_n = kk.cmpi(.slt, c.offs_cn, n_block_splat2);
            const cn_lt_n_2d = broadcast2d(kk, cn_lt_n, 0, c.block_m, c.block_n);
            const c_mask = kk.andi(tm_2d, cn_lt_n_2d);
            kk.store(c_ptrs, acc, .{ .mask = c_mask.inner });
            return &.{};
        }
    }.z;

    const alive_branch = struct {
        fn a(kk: *Kernel, c: AliveCtx) []const DslValue {
            buildAliveBody(kk, c);
            return &.{};
        }
    }.a;

    _ = k.ifThenElse(is_dead, &.{}, zero_branch, alive_branch, ac);
}

fn buildAliveBody(k: *Kernel, ac: anytype) void {
    const block_m = ac.block_m;
    const block_n = ac.block_n;
    const block_k = ac.block_k;
    const top_k = ac.top_k;

    // a_ptrs = a_ptr + (offs_token // top_k) * stride_am_block + offs_k * stride_ak_block
    const top_k_splat_m = k.splat(k.constI64(top_k), &.{block_m});
    const offs_token_div_top_k = k.divsi(ac.offs_token, top_k_splat_m);
    const am_2d_lhs = broadcast2d(k, offs_token_div_top_k, 1, block_m, block_k);
    const stride_am_2d = k.splat(ac.stride_am_block, &.{ block_m, block_k });
    const am_term = k.muli(am_2d_lhs, stride_am_2d);
    const ak_2d_rhs = broadcast2d(k, ac.offs_k, 0, block_m, block_k);
    const stride_ak_2d = k.splat(ac.stride_ak_block, &.{ block_m, block_k });
    const ak_term = k.muli(ak_2d_rhs, stride_ak_2d);
    const a_off = k.addi(am_term, ak_term);
    const a_base_splat = k.splat(ac.a_ptr, &.{ block_m, block_k });
    const a_ptrs_init = k.addptr(a_base_splat, a_off);

    // b_ptrs = b_ptr + off_experts*stride_be_block + offs_k[:,None]*stride_bk_block + offs_bn[None,:]*stride_bn_block
    const off_exp_x_be = k.muli(ac.off_experts, ac.stride_be_block);
    const off_exp_splat = k.splat(off_exp_x_be, &.{ block_k, block_n });
    const bk_lhs = broadcast2d(k, ac.offs_k, 1, block_k, block_n);
    const stride_bk_2d = k.splat(ac.stride_bk_block, &.{ block_k, block_n });
    const bk_term = k.muli(bk_lhs, stride_bk_2d);
    const bn_rhs = broadcast2d(k, ac.offs_bn, 0, block_k, block_n);
    const stride_bn_2d = k.splat(ac.stride_bn_block, &.{ block_k, block_n });
    const bn_term = k.muli(bn_rhs, stride_bn_2d);
    const b_off_inner = k.addi(bk_term, bn_term);
    const b_off = k.addi(off_exp_splat, b_off_inner);
    const b_base_splat = k.splat(ac.b_ptr, &.{ block_k, block_n });
    const b_ptrs_init = k.addptr(b_base_splat, b_off);

    // accumulator: tensor<BLOCK_M x BLOCK_N x f32> = 0
    const acc_init = k.splat(k.constF32(0.0), &.{ block_m, block_n });

    // Loop bounds: for k_iter in 0..cdiv(k_block, BLOCK_K).
    const num_k_iters = k.ceildivsi(ac.k_block, k.constI64(block_k));
    const num_k_iters_i32 = k.trunci(num_k_iters, k.scalarTy(.i32));
    const lb = k.constI32(0);
    const step = k.constI32(1);

    const KCtx = struct {
        a_ptrs_base: kernel_dsl.Value,
        b_ptrs_base: kernel_dsl.Value,
        offs_k: kernel_dsl.Value,
        token_mask: kernel_dsl.Value,
        k_block: kernel_dsl.Value,
        block_k: i64,
        block_m: i64,
        block_n: i64,
        stride_ak_block: kernel_dsl.Value,
        stride_bk_block: kernel_dsl.Value,
        a_dtype: kernel_dsl.DType,
        b_dtype: kernel_dsl.DType,
    };
    const kctx: KCtx = .{
        .a_ptrs_base = a_ptrs_init,
        .b_ptrs_base = b_ptrs_init,
        .offs_k = ac.offs_k,
        .token_mask = ac.token_mask,
        .k_block = ac.k_block,
        .block_k = block_k,
        .block_m = block_m,
        .block_n = block_n,
        .stride_ak_block = ac.stride_ak_block,
        .stride_bk_block = ac.stride_bk_block,
        .a_dtype = ac.a_dtype,
        .b_dtype = ac.b_dtype,
    };

    const k_body = struct {
        fn b(kk: *Kernel, k_iter: kernel_dsl.Value, iter: []const kernel_dsl.Value, kc: KCtx) []const kernel_dsl.Value {
            const acc = iter[0];
            const a_ptrs = iter[1];
            const b_ptrs = iter[2];

            // k_remaining = k_block - k_iter * BLOCK_K  (i64)
            const k_iter_i64 = kk.extsi(k_iter, kk.scalarTy(.i64));
            const k_iter_x_block = kk.muli(k_iter_i64, kk.constI64(kc.block_k));
            const k_remaining = kk.subi(kc.k_block, k_iter_x_block);

            // mask_a = token_mask[:, None] & (offs_k[None, :] < k_remaining)
            const k_rem_splat = kk.splat(k_remaining, &.{kc.block_k});
            const offs_k_lt = kk.cmpi(.slt, kc.offs_k, k_rem_splat);
            const offs_k_lt_2d_a = broadcast2d(kk, offs_k_lt, 0, kc.block_m, kc.block_k);
            const tm_2d = broadcast2d(kk, kc.token_mask, 1, kc.block_m, kc.block_k);
            const mask_a = kk.andi(tm_2d, offs_k_lt_2d_a);

            // a = load(a_ptrs, mask=mask_a, other=0)
            const a_elem = kk.scalarTy(kc.a_dtype);
            const a_tensor_ty = kk.tensorTy(&.{ kc.block_m, kc.block_k }, kc.a_dtype);
            const a_zero_scalar = kk.constF32(0.0);
            const a_zero_input = if (kc.a_dtype == .f32)
                a_zero_scalar
            else
                kk.fpToFp(a_zero_scalar, a_elem, .{ .rounding = .rtne });
            const a_zero_tensor = kk.splat(a_zero_input, &.{ kc.block_m, kc.block_k });
            const a_val = kk.load(a_ptrs, a_tensor_ty, .{
                .mask = mask_a.inner,
                .other = a_zero_tensor.inner,
            });

            // mask_b = (offs_k[:, None] < k_remaining)
            const offs_k_lt_2d_b = broadcast2d(kk, offs_k_lt, 1, kc.block_k, kc.block_n);
            const b_elem = kk.scalarTy(kc.b_dtype);
            const b_tensor_ty = kk.tensorTy(&.{ kc.block_k, kc.block_n }, kc.b_dtype);
            const b_zero_scalar = kk.constF32(0.0);
            const b_zero_input = if (kc.b_dtype == .f32)
                b_zero_scalar
            else
                kk.fpToFp(b_zero_scalar, b_elem, .{ .rounding = .rtne });
            const b_zero_tensor = kk.splat(b_zero_input, &.{ kc.block_k, kc.block_n });
            const b_val = kk.load(b_ptrs, b_tensor_ty, .{
                .mask = offs_k_lt_2d_b.inner,
                .other = b_zero_tensor.inner,
            });

            // accumulator = tt.dot(a, b, acc)
            const acc_ty = kk.tensorTy(&.{ kc.block_m, kc.block_n }, .f32);
            const new_acc = kk.dot(a_val, b_val, acc, acc_ty, .{ .input_precision = .ieee, .max_num_imprecise_acc = 0 });

            // a_ptrs += BLOCK_K * stride_ak_block, b_ptrs += BLOCK_K * stride_bk_block
            const ak_stride_x_block = kk.muli(kk.constI64(kc.block_k), kc.stride_ak_block);
            const ak_stride_2d = kk.splat(ak_stride_x_block, &.{ kc.block_m, kc.block_k });
            const new_a_ptrs = kk.addptr(a_ptrs, ak_stride_2d);
            const bk_stride_x_block = kk.muli(kk.constI64(kc.block_k), kc.stride_bk_block);
            const bk_stride_2d = kk.splat(bk_stride_x_block, &.{ kc.block_k, kc.block_n });
            const new_b_ptrs = kk.addptr(b_ptrs, bk_stride_2d);

            const out = kk.arena.allocator().alloc(kernel_dsl.Value, 3) catch @panic("OOM");
            out[0] = new_acc;
            out[1] = new_a_ptrs;
            out[2] = new_b_ptrs;
            return out;
        }
    }.b;

    const loop_results = k.forLoop(lb, num_k_iters_i32, step, &.{ acc_init, a_ptrs_init, b_ptrs_init }, k_body, kctx);
    var acc_final = loop_results[0];

    // Optional: MUL_ROUTED_WEIGHT
    if (ac.mul_routed) {
        const tw_ptrs = k.addptr(k.splat(ac.topk_weights_ptr, &.{block_m}), ac.offs_token);
        const tw_zero = k.constF32(0.0);
        const tw_zero_splat = k.splat(tw_zero, &.{block_m});
        const tw_loaded = k.load(tw_ptrs, k.tensorTy(&.{block_m}, .f32), .{
            .mask = ac.token_mask.inner,
            .other = tw_zero_splat.inner,
        });
        const tw_2d = broadcast2d(k, tw_loaded, 1, block_m, block_n);
        acc_final = k.mulf(acc_final, tw_2d);
    }

    // Cast accumulator to compute_type and store.
    const compute_dt = ac.compute_dtype;
    const acc_cast = if (compute_dt == .f32)
        acc_final
    else
        k.fpToFp(acc_final, k.tensorTy(&.{ block_m, block_n }, compute_dt), .{ .rounding = .rtne });

    // c_ptrs = c_ptr + stride_cm * offs_token[:,None] + stride_cn * offs_cn[None,:]
    const cm_2d = broadcast2d(k, ac.offs_token, 1, block_m, block_n);
    const cm_off = k.muli(cm_2d, k.splat(ac.stride_cm_block, &.{ block_m, block_n }));
    const cn_2d = broadcast2d(k, ac.offs_cn, 0, block_m, block_n);
    const cn_off = k.muli(cn_2d, k.splat(ac.stride_cn_block, &.{ block_m, block_n }));
    const c_off = k.addi(cm_off, cn_off);
    const c_ptrs = k.addptr(k.splat(ac.c_ptr, &.{ block_m, block_n }), c_off);

    // c_mask = token_mask[:,None] & (offs_cn[None,:] < n_block)
    const tm_2d_c = broadcast2d(k, ac.token_mask, 1, block_m, block_n);
    const cn_lt_n = k.cmpi(.slt, ac.offs_cn, k.splat(ac.n_block, &.{block_n}));
    const cn_lt_n_2d = broadcast2d(k, cn_lt_n, 0, block_m, block_n);
    const c_mask = k.andi(tm_2d_c, cn_lt_n_2d);

    k.store(c_ptrs, acc_cast, .{ .mask = c_mask.inner });
}

// =============================================================================
// Config / validation helpers
// =============================================================================

fn makeGenerationConfig(
    a: Tensor,
    b: Tensor,
    max_num_tokens_padded: i64,
    num_valid_tokens: i64,
    opts: Options,
    naive_block_assignment: bool,
    top_k: i64,
    mul_routed_weight: bool,
    has_bias: bool,
    output_dtype: DataType,
) GenerationConfig {
    var use_fp8 = opts.use_fp8_w8a8;
    if (b.dtype() == .f8e4m3fn) use_fp8 = true;
    return .{
        .a_dtype = a.dtype(),
        .b_dtype = b.dtype(),
        .c_dtype = output_dtype,
        .num_tokens = @intCast(a.dim(0)),
        .top_k = @intCast(top_k),
        .num_experts = @intCast(b.dim(0)),
        .out_features = @intCast(b.dim(1)),
        .in_features = @intCast(b.dim(2)),
        .max_num_tokens_padded = @intCast(max_num_tokens_padded),
        .num_valid_tokens = @intCast(num_valid_tokens),
        .block_size_m = @intCast(opts.block_size_m),
        .block_size_n = @intCast(opts.block_size_n),
        .block_size_k = @intCast(opts.block_size_k),
        .group_size_m = @intCast(opts.group_size_m),
        .group_n = if (use_fp8) 128 else 0,
        .group_k = if (use_fp8) 128 else 0,
        .naive_block_assignment = naive_block_assignment,
        .a_scale_dtype = if (use_fp8) .bf16 else null,
        .b_scale_dtype = if (use_fp8) .bf16 else null,
        .compute_type = .bf16,
        .use_fp8_w8a8 = use_fp8,
        .num_warps = @intCast(opts.num_warps),
        .num_stages = @intCast(opts.num_stages),
        .mul_routed_weight = mul_routed_weight,
        .has_bias = has_bias,
    };
}

fn applyDefaultTokenConfig(opts: Options, num_tokens: i64, num_experts: i64) Options {
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

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
