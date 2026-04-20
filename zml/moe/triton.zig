const std = @import("std");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const ops = zml.ops;
const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const stdx = @import("stdx");

const log = std.log.scoped(.moe_triton);

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
    // Retrieve the kernel config from json or default
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

    var threaded_io: std.Io.Threaded = .init_single_threaded;
    threaded_io.allocator = std.heap.c_allocator;
    defer threaded_io.deinit();

    const io = threaded_io.io();

    const sorted_token_ids, const expert_ids, const num_tokens_post_padded = if (naive_block_assignment) blk: {
        // In the naive path each M-block corresponds to exactly one (token, topk) assignment.
        log.info("Using naive block assignment for MoE kernels, the tokens repartition across experts is sparse. Num assignments: {d}, Num experts: {d}", .{ num_assignments, num_experts });
        const naive_sorted_ids = Tensor.zeroes(Shape.init(.{ .g = 1 }, .i32));
        const naive_expert_ids = ids.reshape(.{ .g = num_assignments });
        const naive_num_tokens_post_padded = Tensor.constant(.{ .i32 = @as(i32, @intCast(max_num_tokens_padded)) }).reshape(.{1});
        break :blk .{ naive_sorted_ids, naive_expert_ids, naive_num_tokens_post_padded };
    } else try alignBlockSize(std.heap.c_allocator, io, ids, num_experts, block_size_m);

    var hidden_quant = hidden;
    var a_scale = opts.a1_scale orelse Tensor.scalar(1.0, .f32);

    // If fp8 quantized model, quantize activations to fp8 with per-token-group quantization
    if (gate_up.dtype() == .f8e4m3fn) {
        hidden_quant, a_scale = try quantizePerTokenGroupFp8(std.heap.c_allocator, io, hidden, fp8ActivationGroupSize(hidden));
    }

    // Build the moe matmul kernels configs
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

    const ttir_first_matmul = generateTtir(std.heap.c_allocator, io, "moe", "matmul_config", .{ .matmul_config = first_generation_config }) catch |err| {
        log.err("Failed to generate TTIR for first MoE matmul: {}", .{err});
        return err;
    };

    defer std.heap.c_allocator.free(ttir_first_matmul);
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
        activated_quant, a_scale = try quantizePerTokenGroupFp8(std.heap.c_allocator, io, activated, fp8ActivationGroupSize(activated));
    }

    const second_generation_config = makeGenerationConfig(
        activated_quant,
        down,
        max_num_tokens_padded,
        num_assignments,
        options,
        naive_block_assignment,
        1,
        true, // is second matmul
        false,
        .bf16,
    );
    const ttir_second_matmul = generateTtir(std.heap.c_allocator, io, "moe", "matmul_config", .{ .matmul_config = second_generation_config }) catch |err| {
        log.err("Failed to generate TTIR for second MoE matmul: {}", .{err});
        return err;
    };
    defer std.heap.c_allocator.free(ttir_second_matmul);
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
    ttir: [:0]const u8,
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
        .ir = ttir,
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

// Here the padding is made so that each token is aligned "in front of" its assigned experts and an expert process a contiguous block of tokens (based on block size m)
fn alignBlockSize(allocator: std.mem.Allocator, io: std.Io, topk_ids: Tensor, num_experts: i64, block_size_m: i64) !struct { Tensor, Tensor, Tensor } {
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

    const ttir_align = try generateTtir(allocator, io, "moe", "align_block", .{ .align_block = .{
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
    } });
    defer allocator.free(ttir_align);
    const ttir_count_sort = try generateTtir(
        allocator,
        io,
        "moe",
        "align_block",
        .{ .align_block = .{
            .kernel_name = AlignBlockSizeKernel.count_and_sort.kernelName(),
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
        } },
    );
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
    io: std.Io,
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

    const ttir = try generateTtir(allocator, io, "moe", "quant_config", .{ .quant_config = config });
    defer allocator.free(ttir);

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
        .ir = ttir,
        .grid = .{ @intCast(x.dim(0) * groups_per_row), 1, 1 },
        .num_stages = 1,
        .num_warps = 1,
        .output_operand_aliases = &.{},
    });

    return .{ outputs[0], outputs[1] };
}

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

pub const Config = union(enum) {
    matmul_config: GenerationConfig,
    quant_config: QuantGenerationConfig,
    align_block: AlignBlockSizeGenerationConfig,
};

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

fn generateTtir(
    allocator: std.mem.Allocator,
    io: std.Io,
    backend: []const u8,
    kernel: []const u8,
    config: Config,
) ![:0]const u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const writer_buffer = try arena.allocator().alloc(u8, 4096);
    const reader_buffer = try arena.allocator().alloc(u8, 4096);

    const compilation_context = zml.module.CompilationContext.current();
    const platform = compilation_context.platform;
    const triton_runtime = platform.triton_runtime.?;
    const request_json = try std.fmt.allocPrint(arena.allocator(), "{f}", .{std.json.fmt(.{
        .backend = backend,
        .kernel = kernel,
        .config = config,
    }, .{ .emit_null_optional_fields = false })});

    var allocating: std.Io.Writer.Allocating = .init(arena.allocator());
    {
        try triton_runtime.process_mutex.lock(compilation_context.io);
        defer triton_runtime.process_mutex.unlock(compilation_context.io);

        var writer = triton_runtime.process.stdin.?.writer(io, writer_buffer);
        writer.interface.print("{s}\n", .{request_json}) catch |err| {
            log.err("Failed to write Triton request backend={s} kernel={s}: {}", .{ backend, kernel, err });
            return err;
        };
        writer.interface.flush() catch |err| {
            log.err("Failed to flush Triton request backend={s} kernel={s}: {}", .{ backend, kernel, err });
            return err;
        };

        var reader = triton_runtime.process.stdout.?.reader(io, reader_buffer);
        _ = reader.interface.streamDelimiter(&allocating.writer, '\n') catch |err| {
            log.err("Failed to read Triton response backend={s} kernel={s}: {}", .{ backend, kernel, err });
            return err;
        };
        _ = reader.interface.discardShort(1) catch |err| {
            log.err("Failed to finalize Triton response read backend={s} kernel={s}: {}", .{ backend, kernel, err });
            return err;
        };
    }

    const raw_response = allocating.written();
    const response: std.json.Value = std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), raw_response, .{}) catch |err| {
        log.err("Failed to parse Triton response backend={s} kernel={s}: {}", .{ backend, kernel, err });
        log.err("Raw Triton response: {s}", .{raw_response});
        return err;
    };
    if (response.object.get("ok").?.bool) {
        return try allocator.dupeZ(u8, response.object.get("result").?.string);
    }

    const error_msg = response.object.get("error").?.string;
    log.err("Triton TTIR generation failed for backend={s} kernel={s}: {s}", .{ backend, kernel, error_msg });
    return error.GenerationFailed;
}

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
    const config_path = runfiles.rlocation("zml/zml/moe/triton/triton_kernels/config.json", &config_path_buf) catch |err| {
        log.err("Failed to resolve MoE launch config in runfiles: {}", .{err});
        return err;
    };

    const config_json = config_path orelse {
        log.warn("MoE launch config is missing from runfiles: zml/zml/moe/triton/triton_kernels/config.json", .{});
        return error.MissingLaunchConfigRunfile;
    };

    return try allocator.dupe(u8, config_json);
}

fn applyJsonTokenConfig(opts: Options, num_tokens: i64) !Options {
    var out = opts;
    if (!opts.dynamic_launch_by_num_tokens) return out;

    var threaded_io: std.Io.Threaded = .init_single_threaded;
    threaded_io.allocator = std.heap.c_allocator;
    defer threaded_io.deinit();
    const io = threaded_io.io();

    var arena: std.heap.ArenaAllocator = .init(std.heap.c_allocator);
    defer arena.deinit();

    const allocator = arena.allocator();
    const config_path = try getLaunchConfigJsonPath(allocator);
    const cwd = std.Io.Dir.cwd();
    var file = try cwd.openFile(io, config_path, .{ .mode = .read_only });
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const config_json = try reader.interface.readAlloc(allocator, try file.length(io));
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, config_json, .{});

    if (parsed.value != .object) return error.InvalidLaunchConfigShape;

    var it = parsed.value.object.iterator();
    var best_m: ?i64 = null;
    var best_diff: u64 = std.math.maxInt(u64);
    var best_config: ?std.json.Value = null;

    while (it.next()) |entry| {
        const m = std.fmt.parseInt(i64, entry.key_ptr.*, 10) catch continue;
        if (entry.value_ptr.* != .object) continue;

        const diff: u64 = if (m >= num_tokens)
            @intCast(m - num_tokens)
        else
            @intCast(num_tokens - m);

        if (best_m == null or diff < best_diff or (diff == best_diff and m < best_m.?)) {
            best_m = m;
            best_diff = diff;
            best_config = entry.value_ptr.*;
        }
    }

    if (best_config == null) return error.NoMatchingLaunchConfig;

    const cfg = best_config.?.object;
    const block_size_m = cfg.get("BLOCK_SIZE_M") orelse return error.MissingLaunchConfigField;
    const block_size_n = cfg.get("BLOCK_SIZE_N") orelse return error.MissingLaunchConfigField;
    const block_size_k = cfg.get("BLOCK_SIZE_K") orelse return error.MissingLaunchConfigField;
    const group_size_m = cfg.get("GROUP_SIZE_M") orelse return error.MissingLaunchConfigField;
    const num_warps = cfg.get("num_warps") orelse return error.MissingLaunchConfigField;
    const num_stages = cfg.get("num_stages") orelse return error.MissingLaunchConfigField;

    out.block_size_m = block_size_m.integer;
    out.block_size_n = block_size_n.integer;
    out.block_size_k = block_size_k.integer;
    out.group_size_m = group_size_m.integer;
    out.num_warps = num_warps.integer;
    out.num_stages = num_stages.integer;

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
