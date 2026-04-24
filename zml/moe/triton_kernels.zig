//! TTIR generation for MoE kernels — direct ports of the `@triton.jit`
//! functions in `triton_kernels/moe.py`. Each top-level function here
//! mirrors a Python kernel one-to-one: same name (snake_case → camelCase),
//! same parameter intent, same body flow. The generated IR is handed to
//! `zml.ops.triton(...)` by `zml/moe/triton.zig`.

const std = @import("std");

const mlir = @import("mlir");

const zml = @import("../zml.zig");
const tri = @import("zml/triton");
const Kernel = tri.Kernel;
const Value = tri.Value;
const DType = tri.DType;

const log = std.log.scoped(.moe_triton);

// =============================================================================
// Per-kernel generation configs
// =============================================================================

pub const GenerationConfig = struct {
    a_dtype: DType,
    b_dtype: DType,
    c_dtype: DType,
    a_scale_dtype: ?DType = null,
    b_scale_dtype: ?DType = null,
    b_bias_dtype: ?DType = null,
    topk_weights_dtype: ?DType = null,
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
    compute_type: DType = .bf16,
    use_fp8_w8a8: bool = false,
    use_int8_w8a8: bool = false,
    use_int8_w8a16: bool = false,
    per_channel_quant: bool = false,
    has_bias: bool = false,
    num_warps: usize,
    num_stages: usize,
};

pub const AlignBlockSizeKernel = enum {
    align_block_size,
    count_and_sort,

    pub fn kernelName(self: @This()) []const u8 {
        return switch (self) {
            .align_block_size => "moe_align_block_size_kernel",
            .count_and_sort => "count_and_sort_expert_tokens_kernel",
        };
    }
};

pub const AlignBlockSizeGenerationConfig = struct {
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

pub const QuantGenerationConfig = struct {
    num_rows: usize,
    num_columns: usize,
    group_size: usize,
    block: usize,
    input_dtype: DType,
    output_dtype: DType,
    scale_dtype: DType = .bf16,
    eps: f32 = 1e-6,
    fp8_min: f32,
    fp8_max: f32,
    use_ue8m0: bool = false,
};

// =============================================================================
// Helpers
// =============================================================================

fn ctx() *mlir.Context {
    return zml.module.CompilationContext.current().mlir_ctx;
}

/// Floor to a multiple of 16 — matches the Python `(v // 16) * 16` guards
/// that keep dynamic strides aligned for tt.load/store.
fn blockFloor16(v: Value) Value {
    return v.div(@as(i64, 16)).mul(@as(i64, 16));
}

// =============================================================================
// per_token_group_quant_fp8
// =============================================================================

/// Direct port of Python `per_token_group_quant_fp8` — per-token-group
/// float8 quantization.
pub fn perTokenGroupQuantFp8(allocator: std.mem.Allocator, config: QuantGenerationConfig) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "per_token_group_quant_fp8", .{
        .y_ptr = .{ .ptr = config.input_dtype },
        .group_size_ptr = .{ .ptr = .i64 },
        .y_num_columns_ptr = .{ .ptr = .i64 },
        .y_row_stride_ptr = .{ .ptr = .i64 },
        .eps_ptr = .{ .ptr = .f32 },
        .y_q_ptr = .{ .ptr = config.output_dtype },
        .y_s_ptr = .{ .ptr = config.scale_dtype },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const block: i64 = @intCast(config.block);
    const out_dt = config.output_dtype;
    const scale_dt = config.scale_dtype;

    const group_size = k.load(a.group_size_ptr);
    const y_num_columns = k.load(a.y_num_columns_ptr);
    const y_row_stride = k.load(a.y_row_stride_ptr);
    const eps = k.load(a.eps_ptr);

    const groups_per_row = y_num_columns.div(group_size);
    const g_id = k.programId(.x).to(.i64);

    // y_ptr += (g_id // groups_per_row) * y_row_stride + (g_id % groups_per_row) * group_size
    const row_off = g_id.div(groups_per_row).mul(y_row_stride);
    const grp_off = g_id.rem(groups_per_row).mul(group_size);
    const y_ptr_shifted = a.y_ptr.addPtr(row_off.add(grp_off));
    const y_q_ptr_shifted = a.y_q_ptr.addPtr(g_id.mul(group_size));
    const y_s_ptr_shifted = a.y_s_ptr.addPtr(g_id);

    // cols = arange(0, BLOCK) — i64 for pointer offsets, i32 for mask.
    const cols_i32 = k.arange(0, block, .i32);
    const cols_i64 = cols_i32.to(.i64);
    const mask = cols_i32.lt(group_size.to(.i32));

    const y_load_ptrs = y_ptr_shifted.splatTo(&.{block}).addPtr(cols_i64);
    const y = k.loadMasked(y_load_ptrs, mask).to(.f32);

    // _absmax = max(tl.max(tl.abs(y)), eps)
    const absmax = k.maximumf(k.max(k.absf(y)), eps);

    // scale_raw = _absmax * (1.0 / fp8_max)
    const scale_raw = absmax.mul(@as(f32, 1.0 / config.fp8_max));
    const y_s = if (config.use_ue8m0) k.exp2(k.ceil(k.log2(scale_raw))) else scale_raw;

    // y_q = clamp(y / y_s, fp8_min, fp8_max).to(output_dtype)
    const y_div = y.div(y_s.splatTo(&.{block}));
    const clamped = k.clampf(
        y_div,
        k.splat(config.fp8_min, &.{block}),
        k.splat(config.fp8_max, &.{block}),
    );
    const y_q = clamped.to(out_dt);

    const y_q_ptrs = y_q_ptr_shifted.splatTo(&.{block}).addPtr(cols_i64);
    k.storeOpts(y_q_ptrs, y_q, .{ .mask = mask });
    k.store(y_s_ptr_shifted, y_s.to(scale_dt));

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize per_token_group_quant_fp8 TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// write_zeros_to_output — helper called by fused_moe_kernel
// =============================================================================

/// Direct port of Python `write_zeros_to_output`. Stores a masked zero-tile
/// for blocks whose expert is `-1` (no expert assigned).
fn writeZerosToOutput(
    k: *Kernel,
    c_ptr: Value,
    stride_cm: Value,
    stride_cn: Value,
    pid_n: Value,
    n: Value,
    offs_token: Value,
    token_mask: Value,
    block_size_m: i64,
    block_size_n: i64,
    compute_type: DType,
) void {
    const accumulator = k.zeros(&.{ block_size_m, block_size_n }, compute_type);
    const offs_cn = pid_n.to(.i32)
        .mul(@as(i32, @intCast(block_size_n))).splatTo(&.{block_size_n})
        .add(k.arange(0, block_size_n, .i32))
        .to(.i64);
    const cm_off = offs_token.broadcast2d(1, block_size_m, block_size_n)
        .mul(stride_cm.splatTo(&.{ block_size_m, block_size_n }));
    const cn_off = offs_cn.broadcast2d(0, block_size_m, block_size_n)
        .mul(stride_cn.splatTo(&.{ block_size_m, block_size_n }));
    const c_ptrs = c_ptr.splatTo(&.{ block_size_m, block_size_n }).addPtr(cm_off.add(cn_off));
    const c_mask = k.mask2d(token_mask, offs_cn.lt(n), block_size_m, block_size_n);
    k.storeOpts(c_ptrs, accumulator, .{ .mask = c_mask });
}

// =============================================================================
// fused_moe_kernel
// =============================================================================

/// Direct port of Python `fused_moe_kernel`. Bf16 / no-quant / no-bias path
/// only — errors out for the feature flags that aren't implemented yet.
pub fn fusedMoeKernel(allocator: std.mem.Allocator, config: GenerationConfig) ![:0]const u8 {
    if (config.use_fp8_w8a8 or config.use_int8_w8a8 or config.use_int8_w8a16 or config.has_bias or config.per_channel_quant) {
        log.err("fused_moe_kernel: unsupported config (fp8/int8/bias/per_channel)", .{});
        return error.TritonTtirGenerationFailed;
    }

    var spec = try Kernel.build(allocator, ctx(), "fused_moe_kernel", .{
        .a_ptr = .{ .ptr = config.a_dtype },
        .b_ptr = .{ .ptr = config.b_dtype },
        .b_bias_ptr = .{ .ptr = config.b_bias_dtype orelse config.c_dtype },
        .a_scale_ptr = .{ .ptr = config.a_scale_dtype orelse .f32 },
        .b_scale_ptr = .{ .ptr = config.b_scale_dtype orelse .f32 },
        .topk_weights_ptr = .{ .ptr = config.topk_weights_dtype orelse .f32 },
        .sorted_token_ids_ptr = .{ .ptr = .i32 },
        .expert_ids_ptr = .{ .ptr = .i32 },
        .num_tokens_post_padded_ptr = .{ .ptr = .i32 },
        .N_ptr = .{ .ptr = .i64 },
        .K_ptr = .{ .ptr = .i64 },
        .EM_ptr = .{ .ptr = .i64 },
        .num_valid_tokens_ptr = .{ .ptr = .i64 },
        .stride_am_ptr = .{ .ptr = .i64 },
        .stride_be_ptr = .{ .ptr = .i64 },
        .stride_bn_ptr = .{ .ptr = .i64 },
        .stride_cm_ptr = .{ .ptr = .i64 },
        .stride_asm_ptr = .{ .ptr = .i64 },
        .stride_ask_ptr = .{ .ptr = .i64 },
        .stride_bse_ptr = .{ .ptr = .i64 },
        .stride_bsk_ptr = .{ .ptr = .i64 },
        .stride_bsn_ptr = .{ .ptr = .i64 },
        .stride_bbe_ptr = .{ .ptr = .i64 },
        .stride_bbn_ptr = .{ .ptr = .i64 },
        .c_ptr = .{ .ptr = config.c_dtype },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const block_size_m: i64 = @intCast(config.block_size_m);
    const block_size_n: i64 = @intCast(config.block_size_n);
    const block_size_k: i64 = @intCast(config.block_size_k);
    const group_size_m: i64 = @intCast(config.group_size_m);
    const top_k: i64 = @intCast(config.top_k);
    const compute_type = config.compute_type;

    // Runtime scalars — floored to multiples of 16 for alignment.
    const n_block = blockFloor16(k.load(a.N_ptr));
    const k_block = blockFloor16(k.load(a.K_ptr));
    const em_block = blockFloor16(k.load(a.EM_ptr));
    const num_valid_tokens = k.load(a.num_valid_tokens_ptr);
    const stride_am_block = blockFloor16(k.load(a.stride_am_ptr));
    const stride_be_block = blockFloor16(k.load(a.stride_be_ptr));
    const stride_bn_block = blockFloor16(k.load(a.stride_bn_ptr));
    const stride_cm_block = blockFloor16(k.load(a.stride_cm_ptr));
    const stride_ak_block = k.lift(@as(i64, 1));
    const stride_bk_block = k.lift(@as(i64, 1));
    const stride_cn_block = k.lift(@as(i64, 1));

    // pid grouping (all i64).
    const pid = k.programId(.x).to(.i64);
    const num_pid_m = em_block.cdiv(block_size_m);
    const num_pid_n = n_block.cdiv(block_size_n);
    const num_pid_in_group = num_pid_n.mul(group_size_m);
    const group_id = pid.div(num_pid_in_group);
    const first_pid_m = group_id.mul(group_size_m);
    const gsm_actual = num_pid_m.sub(first_pid_m).minimum(group_size_m);
    const pid_mod_in_group = pid.rem(num_pid_in_group);
    const pid_m = first_pid_m.add(pid_mod_in_group.rem(gsm_actual));
    const pid_n = pid_mod_in_group.div(gsm_actual);

    // Python: if pid_m * BLOCK_SIZE_M >= num_tokens_post_padded: return
    const num_tokens_post_padded = k.load(a.num_tokens_post_padded_ptr).to(.i64);
    const out_of_range = pid_m.mul(block_size_m).ge(num_tokens_post_padded);
    k.returnIf(out_of_range, .{});

    // offs_token, token_mask — two paths based on naive_block_assignment.
    const offs = k.arange(0, block_size_m, .i64);
    const offs_token = if (config.naive_block_assignment)
        k.select(
            offs.eq(@as(i64, 0)),
            pid_m.splatTo(&.{block_size_m}),
            num_valid_tokens.splatTo(&.{block_size_m}),
        )
    else off: {
        const ids_ptrs = a.sorted_token_ids_ptr.splatTo(&.{block_size_m})
            .addPtr(pid_m.mul(block_size_m).splatTo(&.{block_size_m}).add(offs));
        break :off k.load(ids_ptrs).to(.i64);
    };
    const token_mask = offs_token.lt(num_valid_tokens);

    // off_experts = expert_ids[pid_m].to(i64)
    const off_experts = k.load(a.expert_ids_ptr.addPtr(pid_m.to(.i32))).to(.i64);
    const is_dead = off_experts.eq(@as(i64, -1));

    // Python: if off_experts == -1: write_zeros_to_output(...); return
    //
    // The write-zeros helper runs *before* the return, so we can't use a
    // plain `returnIf` — we need a then-branch with side-effects that then
    // returns. Model this as: skip the gemm path with an `scf.if` on
    // `!is_dead`, but inline the write-zeros in the dead branch and emit
    // `returnIf(is_dead)` after it. Simpler: predicate the rest of the
    // kernel on `!is_dead` after side-effecting on `is_dead`.
    var dead_if = k.openIf(is_dead);
    {
        writeZerosToOutput(
            k,
            a.c_ptr,
            stride_cm_block,
            stride_cn_block,
            pid_n,
            n_block,
            offs_token,
            token_mask,
            block_size_m,
            block_size_n,
            compute_type,
        );
        dead_if.yieldThen(.{});
    }
    k.returnIf(is_dead, .{});

    // Alive body — inlined from Python fused_moe_kernel (bf16 path).

    // offs_bn = (pid_n * BLOCK_SIZE_N + arange(BLOCK_SIZE_N).to(i64)) % n_block
    const offs_bn = pid_n.mul(block_size_n).splatTo(&.{block_size_n})
        .add(k.arange(0, block_size_n, .i64))
        .rem(n_block.splatTo(&.{block_size_n}));
    const offs_k = k.arange(0, block_size_k, .i64);

    // a_ptrs = a_ptr + (offs_token // top_k) * stride_am + offs_k * stride_ak   [m,k]
    const am_term = offs_token.div(top_k).broadcast2d(1, block_size_m, block_size_k)
        .mul(stride_am_block.splatTo(&.{ block_size_m, block_size_k }));
    const ak_term = offs_k.broadcast2d(0, block_size_m, block_size_k)
        .mul(stride_ak_block.splatTo(&.{ block_size_m, block_size_k }));
    const a_ptrs_init = a.a_ptr.splatTo(&.{ block_size_m, block_size_k }).addPtr(am_term.add(ak_term));

    // b_ptrs = b_ptr + off_experts*stride_be + offs_k[:,None]*stride_bk + offs_bn[None,:]*stride_bn   [k,n]
    const be_term = off_experts.mul(stride_be_block).splatTo(&.{ block_size_k, block_size_n });
    const bk_term = offs_k.broadcast2d(1, block_size_k, block_size_n)
        .mul(stride_bk_block.splatTo(&.{ block_size_k, block_size_n }));
    const bn_term = offs_bn.broadcast2d(0, block_size_k, block_size_n)
        .mul(stride_bn_block.splatTo(&.{ block_size_k, block_size_n }));
    const b_ptrs_init = a.b_ptr.splatTo(&.{ block_size_k, block_size_n }).addPtr(be_term.add(bk_term.add(bn_term)));

    // accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    const acc_init = k.zeros(&.{ block_size_m, block_size_n }, .f32);

    // for k in range(0, tl.cdiv(k_block, BLOCK_SIZE_K)):
    const num_k_iters = k_block.cdiv(block_size_k).to(.i32);
    var loop = k.openFor(0, num_k_iters, 1, .{ acc_init, a_ptrs_init, b_ptrs_init });
    {
        const k_iter = loop.iv;
        const acc = loop.carried[0];
        const a_ptrs = loop.carried[1];
        const b_ptrs = loop.carried[2];

        // k_remaining = k_block - k_iter * BLOCK_SIZE_K
        const k_remaining = k_block.sub(k_iter.to(.i64).mul(block_size_k));
        const offs_k_lt = offs_k.lt(k_remaining);

        // a = tl.load(a_ptrs, mask=token_mask[:, None] & (offs_k[None, :] < k_remaining), other=0.0)
        const mask_a = k.mask2d(token_mask, offs_k_lt, block_size_m, block_size_k);
        const a_val = k.loadMasked(a_ptrs, mask_a);

        // b = tl.load(b_ptrs, mask=offs_k[:, None] < k_remaining, other=0.0)
        const mask_b = offs_k_lt.broadcast2d(1, block_size_k, block_size_n);
        const b_val = k.loadMasked(b_ptrs, mask_b);

        // accumulator += tl.dot(a, b)
        const new_acc = k.dotOpts(a_val, b_val, acc, .{ .input_precision = .tf32, .max_num_imprecise_acc = 0 });

        // a_ptrs += BLOCK_SIZE_K * stride_ak_block; b_ptrs += BLOCK_SIZE_K * stride_bk_block
        const new_a_ptrs = a_ptrs.addPtr(stride_ak_block.mul(block_size_k).splatTo(&.{ block_size_m, block_size_k }));
        const new_b_ptrs = b_ptrs.addPtr(stride_bk_block.mul(block_size_k).splatTo(&.{ block_size_k, block_size_n }));

        loop.yield(.{ new_acc, new_a_ptrs, new_b_ptrs });
    }
    var accumulator = loop.results[0];

    // if MUL_ROUTED_WEIGHT: accumulator *= topk_weights[offs_token][:, None]
    if (config.mul_routed_weight) {
        const tw_ptrs = a.topk_weights_ptr.splatTo(&.{block_size_m}).addPtr(offs_token);
        const tw = k.loadMasked(tw_ptrs, token_mask);
        accumulator = accumulator.mul(tw.broadcast2d(1, block_size_m, block_size_n));
    }

    // accumulator = accumulator.to(compute_type)
    accumulator = accumulator.to(compute_type);

    // Write back:
    //   offs_cn = pid_n * BLOCK_SIZE_N + arange(BLOCK_SIZE_N)   (i32 → i64 for ptr math)
    //   c_ptrs  = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    //   c_mask  = token_mask[:, None] & (offs_cn[None, :] < n_block)
    const offs_cn = pid_n.to(.i32)
        .mul(@as(i32, @intCast(block_size_n))).splatTo(&.{block_size_n})
        .add(k.arange(0, block_size_n, .i32))
        .to(.i64);
    const cm_off = offs_token.broadcast2d(1, block_size_m, block_size_n)
        .mul(stride_cm_block.splatTo(&.{ block_size_m, block_size_n }));
    const cn_off = offs_cn.broadcast2d(0, block_size_m, block_size_n)
        .mul(stride_cn_block.splatTo(&.{ block_size_m, block_size_n }));
    const c_ptrs = a.c_ptr.splatTo(&.{ block_size_m, block_size_n }).addPtr(cm_off.add(cn_off));
    const c_mask = k.mask2d(token_mask, offs_cn.lt(n_block), block_size_m, block_size_n);
    k.storeOpts(c_ptrs, accumulator, .{ .mask = c_mask });

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize fused_moe_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// moe_align_block_size_kernel
// =============================================================================

/// Direct port of Python `moe_align_block_size_kernel`. pid==0 runs the
/// histogram + cumsum + block-to-expert assignment pass; pid==1 fills
/// sorted_token_ids with NUMEL.
pub fn moeAlignBlockSizeKernel(allocator: std.mem.Allocator, config: AlignBlockSizeGenerationConfig) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "moe_align_block_size_kernel", .{
        .topk_ids_ptr = .{ .ptr = .i32 },
        .sorted_token_ids_ptr = .{ .ptr = .i32 },
        .expert_ids_ptr = .{ .ptr = .i32 },
        .num_tokens_post_pad_ptr = .{ .ptr = .i32 },
        .cumsum_ptr = .{ .ptr = .i32 },
        .out0_ptr = .{ .ptr = .i32 },
        .out1_ptr = .{ .ptr = .i32 },
        .out2_ptr = .{ .ptr = .i32 },
        .out3_ptr = .{ .ptr = .i32 },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const block_size_m: i64 = @intCast(config.block_size_m);
    const numel: i32 = @intCast(config.numel);
    const num_experts: i32 = @intCast(config.num_experts);
    const padded_num_experts: i64 = @intCast(config.padded_num_experts);
    const max_num_tokens_padded: i32 = @intCast(config.max_num_tokens_padded);
    const max_num_m_blocks: i64 = @intCast(config.max_num_m_blocks);
    const hist_block: i64 = @intCast(config.hist_block);

    const pid = k.programId(.x);

    var pid_if = k.openIfElse(pid.eq(1), .{});
    {
        // pid == 1: fill sorted_token_ids with NUMEL.
        var fill_loop = k.openFor(0, max_num_tokens_padded, hist_block, .{});
        {
            const iv = fill_loop.iv;
            const offs = iv.splatTo(&.{hist_block}).add(k.arange(0, hist_block, .i32));
            const mask = offs.lt(max_num_tokens_padded);
            const sorted_ptrs = a.sorted_token_ids_ptr.splatTo(&.{hist_block}).addPtr(offs);
            k.storeOpts(sorted_ptrs, k.splat(numel, &.{hist_block}), .{ .mask = mask });
            fill_loop.yield(.{});
        }
        pid_if.yieldThen(.{});
    }
    {
        // pid == 0: histogram + cumsum + block assignment.
        const expert_offs = k.arange(0, padded_num_experts, .i32);
        const expert_mask = expert_offs.lt(num_experts);
        const counts_init = k.zeros(&.{padded_num_experts}, .i32);

        // counts[e] = #tokens with topk_id == e
        var hist_loop = k.openFor(0, numel, hist_block, .{counts_init});
        {
            const iv = hist_loop.iv;
            const acc = hist_loop.carried[0];
            const offs = iv.splatTo(&.{hist_block}).add(k.arange(0, hist_block, .i32));
            const mask = offs.lt(numel);
            const topk_ptrs = a.topk_ids_ptr.splatTo(&.{hist_block}).addPtr(offs);
            const expert_vals = k.loadOpts(topk_ptrs, .{
                .mask = mask,
                .other = k.splat(num_experts, &.{hist_block}),
            });
            const valid = mask.bitAnd(expert_vals.lt(num_experts));
            const h = k.histogramOpts(expert_vals, padded_num_experts, .{ .mask = valid });
            hist_loop.yield(.{acc.add(h)});
        }
        const counts = hist_loop.results[0];

        // padded_counts = where(expert_mask, cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M, 0)
        const block_m_v = k.splat(@as(i32, @intCast(block_size_m)), &.{padded_num_experts});
        const padded_counts_full = counts.cdiv(block_m_v).mul(block_m_v);
        const padded_counts = k.select(expert_mask, padded_counts_full, k.zeros(&.{padded_num_experts}, .i32));

        const padded_cumsum = k.cumsum(padded_counts);
        const starts = padded_cumsum.sub(padded_counts);
        const total = k.sum(padded_counts);

        // tl.store(cumsum + expert_offs, starts, mask=expert_mask)
        k.storeOpts(a.cumsum_ptr.splatTo(&.{padded_num_experts}).addPtr(expert_offs), starts, .{ .mask = expert_mask });
        // tl.store(cumsum + NUM_EXPERTS, total); tl.store(num_tokens_post_pad_ptr, total)
        k.store(a.cumsum_ptr.addPtr(num_experts), total);
        k.store(a.num_tokens_post_pad_ptr, total);

        // Block → expert assignment.
        var assign_loop = k.openFor(0, max_num_m_blocks, hist_block, .{});
        {
            const block_start = assign_loop.iv;
            const block_offs = k.arange(0, hist_block, .i32);
            const block_ids = block_start.splatTo(&.{hist_block}).add(block_offs);
            const block_mask = block_ids.lt(@as(i32, @intCast(max_num_m_blocks)));
            const block_offsets = block_ids.mul(@as(i32, @intCast(block_size_m)));
            const init = k.splat(@as(i32, -1), &.{hist_block});

            var exp_loop = k.openFor(0, num_experts, 1, .{init});
            {
                const e_iv = exp_loop.iv;
                const e_acc = exp_loop.carried[0];
                const start_v = k.load(a.cumsum_ptr.addPtr(e_iv));
                const end_v = k.load(a.cumsum_ptr.addPtr(e_iv.add(1)));
                const in_range = block_mask
                    .bitAnd(block_offsets.ge(start_v.splatTo(&.{hist_block})))
                    .bitAnd(block_offsets.lt(end_v.splatTo(&.{hist_block})));
                exp_loop.yield(.{k.select(in_range, e_iv.splatTo(&.{hist_block}), e_acc)});
            }
            const block_expert = exp_loop.results[0];
            k.storeOpts(a.expert_ids_ptr.splatTo(&.{hist_block}).addPtr(block_ids), block_expert, .{ .mask = block_mask });
            assign_loop.yield(.{});
        }
        pid_if.yieldElse(.{});
    }

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize moe_align_block_size_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// count_and_sort_expert_tokens_kernel
// =============================================================================

/// Direct port of Python `count_and_sort_expert_tokens_kernel`. Each program
/// atomically bumps `cumsum[expert]`, writing its block's token offsets into
/// `sorted_token_ids` at the returned rank.
pub fn countAndSortExpertTokensKernel(allocator: std.mem.Allocator, config: AlignBlockSizeGenerationConfig) ![:0]const u8 {
    var spec = try Kernel.build(allocator, ctx(), "count_and_sort_expert_tokens_kernel", .{
        .topk_ids_ptr = .{ .ptr = .i32 },
        .sorted_token_ids_ptr = .{ .ptr = .i32 },
        .cumsum_ptr = .{ .ptr = .i32 },
        .out0_ptr = .{ .ptr = .i32 },
        .out1_ptr = .{ .ptr = .i32 },
    }, &.{});
    defer spec.deinit();
    const k = &spec.kernel;
    const a = spec.args;

    const block: i64 = @intCast(config.sort_block_size);
    const numel: i32 = @intCast(config.numel);
    const num_experts: i32 = @intCast(config.num_experts);
    const block_i32: i32 = @intCast(block);

    const pid = k.programId(.x);
    const num_progs = k.numPrograms(.x);
    const token_offs = k.makeRange(0, block);

    const token_start_init = pid.mul(block_i32);
    const step = num_progs.mul(block_i32);

    // while token_start < NUMEL:
    var w = k.openWhile(.{token_start_init}, .{k.scalarTy(.i32)});
    {
        const ts = w.before_carried[0];
        w.yieldBefore(ts.lt(numel), .{ts});
    }
    {
        const ts = w.after_carried[0];

        // offs = ts + token_offs; mask = offs < NUMEL
        const offs = ts.splatTo(&.{block}).add(token_offs);
        const mask = offs.lt(numel);

        // expert_vals = load(topk_ids + offs, mask=mask, other=NUM_EXPERTS)
        const topk_ptrs = a.topk_ids_ptr.splatTo(&.{block}).addPtr(offs);
        const expert_vals = k.loadOpts(topk_ptrs, .{
            .mask = mask,
            .other = k.splat(num_experts, &.{block}),
        });
        const valid = mask.bitAnd(expert_vals.lt(num_experts));

        // rank = atomic_add(cumsum + expert_vals, 1, mask=valid, sem="relaxed")
        const cumsum_ptrs = a.cumsum_ptr.splatTo(&.{block}).addPtr(expert_vals);
        const rank = k.atomicRmwOpts(.add, cumsum_ptrs, k.ones(&.{block}, .i32), .{
            .mask = valid,
            .sem = .relaxed,
            .scope = .gpu,
        });

        // store(sorted_token_ids + rank, offs, mask=valid)
        const sorted_ptrs = a.sorted_token_ids_ptr.splatTo(&.{block}).addPtr(rank);
        k.storeOpts(sorted_ptrs, offs, .{ .mask = valid });

        w.yieldAfter(.{ts.add(step)});
    }

    return k.finish(&.{}) catch |err| {
        log.err("failed to finalize count_and_sort_expert_tokens_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}
