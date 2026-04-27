//! Triton MoE kernel declarations — direct ports of the `@triton.jit`
//! functions in `triton_kernels/moe.py`. Each kernel here is a
//! `zml.Kernel(...)` declaration with its own inline config struct (no
//! defaults — every field is required at call time). Production callers
//! invoke these via `K.call(inputs, outputs, opts)`; offline tooling reaches the TTIR
//! string via `K.emit(allocator, ctx, cfg)`.

const std = @import("std");

const zml = @import("../zml.zig");
const tri = @import("zml/triton");
const Builder = tri.Builder;
const Value = tri.Value;
const DType = tri.DType;

const log = std.log.scoped(.moe_triton);

/// Floor to a multiple of 16 — matches the Python `(v // 16) * 16` guards
/// that keep dynamic strides aligned for tt.load/store.
fn blockFloor16(v: Value) Value {
    return v.div(16).mul(16);
}

// =============================================================================
// per_token_group_quant_fp8
// =============================================================================

/// Direct port of Python `per_token_group_quant_fp8` — per-token-group
/// float8 quantization.
pub const PerTokenGroupQuantFp8 = zml.Kernel(.{
    .name = "per_token_group_quant_fp8",
    .config = struct {
        input_dtype: DType,
        output_dtype: DType,
        scale_dtype: DType,
        block: usize,
        fp8_min: f32,
        fp8_max: f32,
        use_ue8m0: bool,
    },
}, struct {
    pub fn run(b: *Builder, cfg: anytype) !void {
        const a = try b.declareArgs(.{
            .y_ptr = .{ .ptr = cfg.input_dtype },
            .group_size_ptr = .{ .ptr = .i64 },
            .y_num_columns_ptr = .{ .ptr = .i64 },
            .y_row_stride_ptr = .{ .ptr = .i64 },
            .eps_ptr = .{ .ptr = .f32 },
            .y_q_ptr = .{ .ptr = cfg.output_dtype },
            .y_s_ptr = .{ .ptr = cfg.scale_dtype },
        });

        const block: i64 = @intCast(cfg.block);
        const out_dt = cfg.output_dtype;
        const scale_dt = cfg.scale_dtype;

        const group_size = b.load(a.group_size_ptr);
        const y_num_columns = b.load(a.y_num_columns_ptr);
        const y_row_stride = b.load(a.y_row_stride_ptr);
        const eps = b.load(a.eps_ptr);

        const groups_per_row = y_num_columns.div(group_size);
        // g_id is i32 from `tl.program_id`; the i64 promotion only happens
        // inside the `(row|row_g_id).to(tl.int64)` casts in the Python source.
        // y_s_ptr_shifted reuses the raw i32 `g_id` directly (no extsi).
        const pid_i32 = b.programId(.x);
        const g_id = pid_i32.to(.i64);

        // y_ptr += (g_id // groups_per_row) * y_row_stride + (g_id % groups_per_row) * group_size
        // Python emit order: divsi, remsi, muli (row * stride), muli (row_g_id * group_size), addi.
        const row = g_id.div(groups_per_row);
        const row_g_id = g_id.rem(groups_per_row);
        const row_off = row.mul(y_row_stride);
        const grp_off = row_g_id.mul(group_size);
        const y_ptr_shifted = a.y_ptr.addPtr(row_off.add(grp_off));
        const y_q_ptr_shifted = a.y_q_ptr.addPtr(g_id.mul(group_size));
        // y_s_ptr += g_id (raw i32, no cast — matches `y_s_ptr += g_id` in Python).
        const y_s_ptr_shifted = a.y_s_ptr.addPtr(pid_i32);

        // cols = arange(0, BLOCK) — i64 for both pointer offsets and mask
        // comparison against the i64 `group_size` (matches Python's
        // `cols < group_size` after Triton's auto-promotion).
        const cols_i32 = b.arange(0, block, .i32);
        const cols_i64 = cols_i32.to(.i64);
        const mask = cols_i64.lt(group_size);

        // Python: `tl.load(y_ptr + cols, mask=mask, other=0.0)` — explicit
        // 3-operand form. Use `.other` to materialize the const-zero tensor
        // (without it, `tt.load` is 2-operand and Python's emit doesn't match).
        const y_load_ptrs = y_ptr_shifted.addPtr(cols_i64);
        const y_other = b.zeros(&.{block}, cfg.input_dtype);
        const y = b.loadOpts(y_load_ptrs, .{ .mask = mask, .other = y_other }).to(.f32);

        // _absmax = max(tl.max(tl.abs(y)), eps) — `b.max(...).maximum(eps)`
        // resolves to `arith.maxnumf` (matches Python's `tl.maximum`).
        const absmax = b.max(b.absf(y)).maximum(eps);

        // scale_raw = _absmax * (1.0 / fp8_max)
        const scale_raw = absmax.mul(1.0 / cfg.fp8_max);
        const y_s = if (cfg.use_ue8m0) b.exp2(b.ceil(b.log2(scale_raw))) else scale_raw;

        // y_q = clamp(y / y_s, fp8_min, fp8_max).to(output_dtype)
        const y_div = y.div(y_s);
        const clamped = b.clampf(
            y_div,
            b.splat(cfg.fp8_min, &.{block}),
            b.splat(cfg.fp8_max, &.{block}),
        );
        const y_q = clamped.to(out_dt);

        const y_q_ptrs = y_q_ptr_shifted.addPtr(cols_i64);
        b.storeOpts(y_q_ptrs, y_q, .{ .mask = mask });
        b.store(y_s_ptr_shifted, y_s.to(scale_dt));
    }
});

// =============================================================================
// write_zeros_to_output — helper called by FusedMoe
// =============================================================================

/// Direct port of Python `write_zeros_to_output`. Stores a masked zero-tile
/// for blocks whose expert is `-1` (no expert assigned). `stride_cn` is
/// hardcoded to 1 (contiguous last dim), so the N-axis offset is just
/// `offs_cn` itself.
fn writeZerosToOutput(
    b: *Builder,
    c_ptr: Value,
    stride_cm: Value,
    pid_n: Value,
    n: Value,
    offs_token: Value,
    token_mask: Value,
    block_size_m: i64,
    block_size_n: i64,
    compute_type: DType,
) void {
    const accumulator = b.zeros(&.{ block_size_m, block_size_n }, compute_type);
    const offs_cn = pid_n.mul(block_size_n).add(b.arange(0, block_size_n, .i64));

    // c_ptrs in two addptrs to match Python's `c_ptr + cm[:, None] + cn[None, :]`
    // left-to-right evaluation. `stride_cm * offs_token[:, None]` puts the
    // splatted stride on the LHS of arith.muli (Python source order).
    const offs_token_col = b.expandDims(offs_token, 1);
    const cm_col = stride_cm.mul(offs_token_col);
    const c_ptrs_col = b.splat(c_ptr, &.{ block_size_m, 1 }).addPtr(cm_col);
    const offs_cn_row = b.expandDims(offs_cn, 0);
    const c_ptrs_2d = b.broadcastTo(c_ptrs_col, &.{ block_size_m, block_size_n });
    const cn_off_2d = b.broadcastTo(offs_cn_row, &.{ block_size_m, block_size_n });
    const c_ptrs = c_ptrs_2d.addPtr(cn_off_2d);

    // c_mask: Python emit order — expand_dims, splat, cmpi, broadcast, broadcast, andi.
    const token_mask_col = b.expandDims(token_mask, 1);
    const n_splat = b.splat(n, &.{ 1, block_size_n });
    const offs_cn_lt_2d = offs_cn_row.lt(n_splat);
    const token_mask_full = b.broadcastTo(token_mask_col, &.{ block_size_m, block_size_n });
    const offs_cn_lt_full = b.broadcastTo(offs_cn_lt_2d, &.{ block_size_m, block_size_n });
    b.storeOpts(c_ptrs, accumulator, .{ .mask = token_mask_full.bitAnd(offs_cn_lt_full) });
}

// =============================================================================
// fused_moe_kernel
// =============================================================================

/// Direct port of Python `fused_moe_kernel`. Bf16 / no-quant / no-bias path
/// only — errors out for the feature flags that aren't implemented yet.
pub const FusedMoe = zml.Kernel(.{
    .name = "fused_moe_kernel",
    .config = struct {
        a_dtype: DType,
        b_dtype: DType,
        c_dtype: DType,
        a_scale_dtype: ?DType,
        b_scale_dtype: ?DType,
        b_bias_dtype: ?DType,
        topk_weights_dtype: ?DType,
        block_size_m: usize,
        block_size_n: usize,
        block_size_k: usize,
        group_size_m: usize,
        top_k: usize,
        naive_block_assignment: bool,
        mul_routed_weight: bool,
        compute_type: DType,
        // Validation flags — the kernel rejects configs that set any of
        // these to true (the body only implements the bf16 / no-quant /
        // no-bias path).
        use_fp8_w8a8: bool,
        use_int8_w8a8: bool,
        use_int8_w8a16: bool,
        per_channel_quant: bool,
        has_bias: bool,
    },
}, struct {
    pub fn run(b: *Builder, cfg: anytype) !void {
        if (cfg.use_fp8_w8a8 or cfg.use_int8_w8a8 or cfg.use_int8_w8a16 or cfg.has_bias or cfg.per_channel_quant) {
            log.err("fused_moe_kernel: unsupported config (fp8/int8/bias/per_channel)", .{});
            return error.TritonTtirGenerationFailed;
        }

        const a = try b.declareArgs(.{
            .a_ptr = .{ .ptr = cfg.a_dtype },
            .b_ptr = .{ .ptr = cfg.b_dtype },
            .b_bias_ptr = .{ .ptr = cfg.b_bias_dtype orelse cfg.c_dtype },
            .a_scale_ptr = .{ .ptr = cfg.a_scale_dtype orelse .f32 },
            .b_scale_ptr = .{ .ptr = cfg.b_scale_dtype orelse .f32 },
            .topk_weights_ptr = .{ .ptr = cfg.topk_weights_dtype orelse .f32 },
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
            .c_ptr = .{ .ptr = cfg.c_dtype },
        });

        const block_size_m: i64 = @intCast(cfg.block_size_m);
        const block_size_n: i64 = @intCast(cfg.block_size_n);
        const block_size_k: i64 = @intCast(cfg.block_size_k);
        const group_size_m: i64 = @intCast(cfg.group_size_m);
        const top_k: i64 = @intCast(cfg.top_k);
        const compute_type = cfg.compute_type;

        // Runtime scalars — load all first, then floor to multiples of 16
        // (Python loads them as a group then runs the // 16 * 16 sequence).
        const n_raw = b.load(a.N_ptr);
        const k_raw = b.load(a.K_ptr);
        const em_raw = b.load(a.EM_ptr);
        const num_valid_tokens = b.load(a.num_valid_tokens_ptr);
        const stride_am_raw = b.load(a.stride_am_ptr);
        const stride_be_raw = b.load(a.stride_be_ptr);
        const stride_bn_raw = b.load(a.stride_bn_ptr);
        const stride_cm_raw = b.load(a.stride_cm_ptr);
        const n_block = blockFloor16(n_raw);
        const k_block = blockFloor16(k_raw);
        const em_block = blockFloor16(em_raw);
        const stride_am_block = blockFloor16(stride_am_raw);
        const stride_be_block = blockFloor16(stride_be_raw);
        const stride_bn_block = blockFloor16(stride_bn_raw);
        const stride_cm_block = blockFloor16(stride_cm_raw);
        // stride_ak / stride_bk / stride_cn are all 1 in this port — the A/B/C
        // inner dimension is contiguous, so those mul-by-1 terms are elided.

        // pid grouping. Python keeps `pid` as i32 and lets the divsi auto-
        // promote at the use site — emit the cdivs first, then extsi at the
        // first i64 use. Manual order keeps `extsi pid` after the cdivs in TTIR.
        const pid_i32 = b.programId(.x);
        const num_pid_m = em_block.cdiv(block_size_m);
        const num_pid_n = n_block.cdiv(block_size_n);
        const num_pid_in_group = num_pid_n.mul(group_size_m);
        const pid = pid_i32.to(.i64);
        const group_id = pid.div(num_pid_in_group);
        const first_pid_m = group_id.mul(group_size_m);
        const gsm_actual = num_pid_m.sub(first_pid_m).minimum(group_size_m);
        const pid_mod_in_group = pid.rem(num_pid_in_group);
        const pid_m = first_pid_m.add(pid_mod_in_group.rem(gsm_actual));
        const pid_n = pid_mod_in_group.div(gsm_actual);

        // Python source order: arange first, then load + early-return.
        // Match emit order so the make_range op lands before the cf.cond_br.
        const offs = b.arange(0, block_size_m, .i64);
        // Order: load (i32), muli, extsi (Python emits the muli before the
        // i32→i64 extsi at the cmpi).
        const num_tokens_post_padded_i32 = b.load(a.num_tokens_post_padded_ptr);
        const pid_m_block = pid_m.mul(block_size_m);
        const num_tokens_post_padded = num_tokens_post_padded_i32.to(.i64);
        const out_of_range = pid_m_block.ge(num_tokens_post_padded);
        b.returnIf(out_of_range, .{});
        const offs_token = if (cfg.naive_block_assignment)
            b.select(
                offs.eq(0),
                pid_m.splatTo(&.{block_size_m}),
                num_valid_tokens.splatTo(&.{block_size_m}),
            )
        else off: {
            const ids_ptrs = a.sorted_token_ids_ptr
                .addPtr(pid_m.mul(block_size_m).add(offs));
            break :off b.load(ids_ptrs).to(.i64);
        };
        const token_mask = offs_token.lt(num_valid_tokens);

        // off_experts = expert_ids[pid_m].to(i64). pid_m is i64; Python's
        // `expert_ids_ptr + pid_m` keeps the offset as i64.
        const off_experts = b.load(a.expert_ids_ptr.addPtr(pid_m)).to(.i64);
        const is_dead = off_experts.eq(-1);

        // Python: if off_experts == -1: write_zeros_to_output(...); return
        var dead_ret = b.openReturnIf(is_dead);
        {
            writeZerosToOutput(
                b,
                a.c_ptr,
                stride_cm_block,
                pid_n,
                n_block,
                offs_token,
                token_mask,
                block_size_m,
                block_size_n,
                compute_type,
            );
            dead_ret.yieldReturn(.{});
        }

        // CSE `pid_n * BLOCK_SIZE_N` so it reuses across offs_bn / offs_cn.
        const pid_n_block_n = pid_n.mul(block_size_n);
        const offs_bn = pid_n_block_n.add(b.arange(0, block_size_n, .i64)).rem(n_block);

        // offs_k stays i32 through expand_dims; extsi to i64 happens on the 2D form.
        const offs_k_i32 = b.arange(0, block_size_k, .i32);

        // a_ptrs: `offs_token[:, None] // top_k * stride_am + offs_k[None, :]`.
        // (stride_ak == 1 here, so `* stride_ak` folds away.) Build with explicit
        // broadcastTo to match Python's emit shape.
        const offs_token_col = b.expandDims(offs_token, 1);
        const am_col = offs_token_col.div(top_k).mul(stride_am_block);
        const offs_k_row = b.expandDims(offs_k_i32, 0).to(.i64);
        const am_term = b.broadcastTo(am_col, &.{ block_size_m, block_size_k });
        const ak_term = b.broadcastTo(offs_k_row, &.{ block_size_m, block_size_k });
        const a_ptrs_init = a.a_ptr.addPtr(am_term.add(ak_term));

        // b_ptrs: `b_ptr + off_experts * stride_be` (scalar addptr) THEN
        // `+ (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)` (2D
        // addptr) — matches Python's left-to-right evaluation. Defer extsi
        // for offs_k to after offs_bn's expand_dims/mul (Python emit order).
        const b_ptr_shifted = a.b_ptr.addPtr(off_experts.mul(stride_be_block));
        const offs_k_col_i32 = b.expandDims(offs_k_i32, 1);
        const offs_bn_row = b.expandDims(offs_bn, 0);
        const bn_row = offs_bn_row.mul(stride_bn_block);
        const offs_k_col = offs_k_col_i32.to(.i64);
        const bk_term = b.broadcastTo(offs_k_col, &.{ block_size_k, block_size_n });
        const bn_term = b.broadcastTo(bn_row, &.{ block_size_k, block_size_n });
        const b_ptrs_init = b_ptr_shifted.addPtr(bk_term.add(bn_term));

        const acc_init = b.zeros(&.{ block_size_m, block_size_n }, .f32);

        // Loop bounds stay i64 because `k_block` is i64. iter_args order
        // matches Python's TTIR (a_ptrs, b_ptrs, accumulator).
        const num_k_iters = k_block.cdiv(block_size_k);
        var loop = b.openFor(@as(i64, 0), num_k_iters, @as(i64, 1), .{ a_ptrs_init, b_ptrs_init, acc_init });
        {
            const k_iter = loop.iv;
            const a_ptrs = loop.carried[0];
            const b_ptrs = loop.carried[1];
            const acc = loop.carried[2];

            const k_remaining = k_block.sub(k_iter.mul(block_size_k));

            // mask_a = token_mask[:, None] & (offs_k[None, :] < k_remaining).
            const token_mask_col_a = b.expandDims(token_mask, 1);
            const offs_k_row_a = b.expandDims(offs_k_i32, 0).to(.i64);
            const offs_k_lt_a = offs_k_row_a.lt(b.splat(k_remaining, &.{ 1, block_size_k }));
            const token_mask_full_a = b.broadcastTo(token_mask_col_a, &.{ block_size_m, block_size_k });
            const offs_k_lt_a_full = b.broadcastTo(offs_k_lt_a, &.{ block_size_m, block_size_k });
            const mask_a = token_mask_full_a.bitAnd(offs_k_lt_a_full);

            const a_val = b.loadOpts(a_ptrs, .{
                .mask = mask_a,
                .other = b.zeros(&.{ block_size_m, block_size_k }, cfg.a_dtype),
            });

            // mask_b = offs_k[:, None] < k_remaining (broadcast to KxN).
            const offs_k_col_b = b.expandDims(offs_k_i32, 1).to(.i64);
            const offs_k_lt_b = offs_k_col_b.lt(b.splat(k_remaining, &.{ block_size_k, 1 }));
            const mask_b = b.broadcastTo(offs_k_lt_b, &.{ block_size_k, block_size_n });

            const b_val = b.loadOpts(b_ptrs, .{
                .mask = mask_b,
                .other = b.zeros(&.{ block_size_k, block_size_n }, cfg.b_dtype),
            });

            const new_acc = b.dotOpts(a_val, b_val, acc, .{
                .input_precision = .tf32,
                .max_num_imprecise_acc = 0,
            });

            // BLOCK_SIZE_K is `tl.constexpr` → i32 dense splat (stride_ak == 1).
            const bsk_i32: i32 = @intCast(cfg.block_size_k);
            const new_a_ptrs = a_ptrs.addPtr(b.splat(bsk_i32, &.{ block_size_m, block_size_k }));
            const new_b_ptrs = b_ptrs.addPtr(b.splat(bsk_i32, &.{ block_size_k, block_size_n }));

            loop.yield(.{ new_a_ptrs, new_b_ptrs, new_acc });
        }
        var accumulator = loop.results[2];

        if (cfg.mul_routed_weight) {
            const tw_dtype = cfg.topk_weights_dtype orelse .f32;
            const tw_other = b.zeros(&.{block_size_m}, tw_dtype);
            const tw = b.loadOpts(a.topk_weights_ptr.addPtr(offs_token), .{ .mask = token_mask, .other = tw_other });
            accumulator = accumulator.mul(tw.broadcast2d(1, block_size_m, block_size_n));
        }

        accumulator = accumulator.to(compute_type);

        // c_ptrs: two addptrs to match Python's `c_ptr + cm[:, None] + cn[None, :]`
        // left-to-right eval. `stride_cm * offs_token[:, None]` puts the splatted
        // stride on the LHS of arith.muli (Python source order).
        const offs_cn = pid_n_block_n.add(b.arange(0, block_size_n, .i64));
        const offs_token_col_c = b.expandDims(offs_token, 1);
        const cm_col = stride_cm_block.mul(offs_token_col_c);
        const c_ptrs_col = b.splat(a.c_ptr, &.{ block_size_m, 1 }).addPtr(cm_col);
        const offs_cn_row = b.expandDims(offs_cn, 0);
        const c_ptrs_2d = b.broadcastTo(c_ptrs_col, &.{ block_size_m, block_size_n });
        const c_ptrs = c_ptrs_2d.addPtr(b.broadcastTo(offs_cn_row, &.{ block_size_m, block_size_n }));

        // c_mask = token_mask[:, None] & (offs_cn[None, :] < n_block).
        const token_mask_col_c = b.expandDims(token_mask, 1);
        const offs_cn_lt_2d = offs_cn_row.lt(b.splat(n_block, &.{ 1, block_size_n }));
        const token_mask_full_c = b.broadcastTo(token_mask_col_c, &.{ block_size_m, block_size_n });
        const offs_cn_lt_full = b.broadcastTo(offs_cn_lt_2d, &.{ block_size_m, block_size_n });
        b.storeOpts(c_ptrs, accumulator, .{ .mask = token_mask_full_c.bitAnd(offs_cn_lt_full) });
    }
});

// =============================================================================
// moe_align_block_size_kernel
// =============================================================================

/// Direct port of Python `moe_align_block_size_kernel`. pid==0 runs the
/// histogram + cumsum + block-to-expert assignment pass; pid==1 fills
/// sorted_token_ids with NUMEL.
pub const MoeAlignBlockSize = zml.Kernel(.{
    .name = "moe_align_block_size_kernel",
    .config = struct {
        numel: usize,
        num_experts: usize,
        padded_num_experts: usize,
        max_num_tokens_padded: usize,
        max_num_m_blocks: usize,
        block_size_m: usize,
        hist_block: usize,
    },
}, struct {
    pub fn run(b: *Builder, cfg: anytype) !void {
        const a = try b.declareArgs(.{
            .topk_ids_ptr = .{ .ptr = .i32 },
            .sorted_token_ids_ptr = .{ .ptr = .i32 },
            .expert_ids_ptr = .{ .ptr = .i32 },
            .num_tokens_post_pad_ptr = .{ .ptr = .i32 },
            .cumsum_ptr = .{ .ptr = .i32 },
            .out0_ptr = .{ .ptr = .i32 },
            .out1_ptr = .{ .ptr = .i32 },
            .out2_ptr = .{ .ptr = .i32 },
            .out3_ptr = .{ .ptr = .i32 },
        });

        const block_size_m: i32 = @intCast(cfg.block_size_m);
        const numel: i32 = @intCast(cfg.numel);
        const num_experts: i32 = @intCast(cfg.num_experts);
        const padded_num_experts: i64 = @intCast(cfg.padded_num_experts);
        const max_num_tokens_padded: i32 = @intCast(cfg.max_num_tokens_padded);
        const max_num_m_blocks: i64 = @intCast(cfg.max_num_m_blocks);
        const hist_block: i64 = @intCast(cfg.hist_block);

        const pid = b.programId(.x);
        const fill_offs = b.arange(0, hist_block, .i32);

        // if pid == 1: fill sorted_token_ids with NUMEL; return
        var fill_branch = b.openReturnIf(pid.eq(1));
        {
            var fill_loop = b.openFor(0, max_num_tokens_padded, hist_block, .{});
            {
                const offs = fill_loop.iv.add(fill_offs);
                const mask = offs.lt(max_num_tokens_padded);
                b.storeOpts(
                    a.sorted_token_ids_ptr.addPtr(offs),
                    b.splat(numel, &.{hist_block}),
                    .{ .mask = mask },
                );
                fill_loop.yield(.{});
            }
            fill_branch.yieldReturn(.{});
        }

        // pid != 1: histogram + cumsum + block assignment.
        const expert_offs = b.arange(0, padded_num_experts, .i32);
        const token_offs = b.arange(0, hist_block, .i32);
        const expert_mask = expert_offs.lt(num_experts);
        const counts_init = b.zeros(&.{padded_num_experts}, .i32);

        var hist_loop = b.openFor(0, numel, hist_block, .{counts_init});
        {
            const offs = hist_loop.iv.add(token_offs);
            const mask = offs.lt(numel);
            const expert_vals = b.loadOpts(a.topk_ids_ptr.addPtr(offs), .{
                .mask = mask,
                .other = b.splat(num_experts, &.{hist_block}),
            });
            const valid = mask.bitAnd(expert_vals.lt(num_experts));
            const h = b.histogramOpts(expert_vals, padded_num_experts, .{ .mask = valid });
            hist_loop.yield(.{hist_loop.carried[0].add(h)});
        }
        const counts = hist_loop.results[0];

        // padded_counts = where(expert_mask, cdiv(counts, BLOCK_SIZE_M) * BLOCK_SIZE_M, 0)
        const padded_counts = b.select(
            expert_mask,
            counts.cdiv(block_size_m).mul(block_size_m),
            b.zeros(&.{padded_num_experts}, .i32),
        );
        const padded_cumsum = b.cumsumOpts(padded_counts, .{ .axis = 0 });
        const starts = padded_cumsum.sub(padded_counts);
        const total_tokens_post_pad = b.sumOpts(padded_counts, .{ .axis = 0 });

        b.storeOpts(a.cumsum_ptr.addPtr(expert_offs), starts, .{ .mask = expert_mask });
        b.store(a.cumsum_ptr.addPtr(num_experts), total_tokens_post_pad);
        b.store(a.num_tokens_post_pad_ptr, total_tokens_post_pad);

        const block_offs = b.arange(0, hist_block, .i32);
        var assign_loop = b.openFor(0, max_num_m_blocks, hist_block, .{});
        {
            const block_ids = assign_loop.iv.add(block_offs);
            const block_mask = block_ids.lt(@as(i32, @intCast(max_num_m_blocks)));
            const block_offsets = block_ids.mul(block_size_m);
            const block_expert_init = b.full(&.{hist_block}, -1, .i32);

            var exp_loop = b.openFor(0, num_experts, 1, .{block_expert_init});
            {
                const e_iv = exp_loop.iv;
                const start_v = b.load(a.cumsum_ptr.addPtr(e_iv));
                const end_v = b.load(a.cumsum_ptr.addPtr(e_iv.add(1)));
                const in_range = block_mask
                    .bitAnd(block_offsets.ge(start_v))
                    .bitAnd(block_offsets.lt(end_v));
                exp_loop.yield(.{
                    b.select(in_range, e_iv.splatTo(&.{hist_block}), exp_loop.carried[0]),
                });
            }
            b.storeOpts(
                a.expert_ids_ptr.addPtr(block_ids),
                exp_loop.results[0],
                .{ .mask = block_mask },
            );
            assign_loop.yield(.{});
        }
    }
});

// =============================================================================
// count_and_sort_expert_tokens_kernel
// =============================================================================

/// Direct port of Python `count_and_sort_expert_tokens_kernel`. Each program
/// atomically bumps `cumsum[expert]`, writing its block's token offsets into
/// `sorted_token_ids` at the returned rank.
pub const CountAndSortExpertTokens = zml.Kernel(.{
    .name = "count_and_sort_expert_tokens_kernel",
    .config = struct {
        numel: usize,
        num_experts: usize,
        sort_block_size: usize,
    },
}, struct {
    pub fn run(b: *Builder, cfg: anytype) !void {
        const a = try b.declareArgs(.{
            .topk_ids_ptr = .{ .ptr = .i32 },
            .sorted_token_ids_ptr = .{ .ptr = .i32 },
            .cumsum_ptr = .{ .ptr = .i32 },
            .out0_ptr = .{ .ptr = .i32 },
            .out1_ptr = .{ .ptr = .i32 },
        });

        const block: i64 = @intCast(cfg.sort_block_size);
        const numel: i32 = @intCast(cfg.numel);
        const num_experts: i32 = @intCast(cfg.num_experts);
        const block_i32: i32 = @intCast(block);

        const pid = b.programId(.x);
        const num_progs = b.numPrograms(.x);
        const token_offs = b.makeRange(0, block);

        const token_start_init = pid.mul(block_i32);
        const step = num_progs.mul(block_i32);

        // while token_start < NUMEL:
        var w = b.openWhile(.{token_start_init}, .{b.scalarTy(.i32)});
        {
            const ts = w.before_carried[0];
            w.yieldBefore(ts.lt(numel), .{ts});
        }
        {
            const ts = w.after_carried[0];

            // offs = ts + token_offs; mask = offs < NUMEL
            const offs = ts.add(token_offs);
            const mask = offs.lt(numel);

            // expert_vals = load(topk_ids + offs, mask=mask, other=NUM_EXPERTS)
            const topk_ptrs = a.topk_ids_ptr.addPtr(offs);
            const expert_vals = b.loadOpts(topk_ptrs, .{
                .mask = mask,
                .other = b.splat(num_experts, &.{block}),
            });
            const valid = mask.bitAnd(expert_vals.lt(num_experts));

            // rank = atomic_add(cumsum + expert_vals, 1, mask=valid, sem="relaxed")
            const cumsum_ptrs = a.cumsum_ptr.addPtr(expert_vals);
            const rank = b.atomicRmwOpts(.add, cumsum_ptrs, b.ones(&.{block}, .i32), .{
                .mask = valid,
                .sem = .relaxed,
                .scope = .gpu,
            });

            // store(sorted_token_ids + rank, offs, mask=valid)
            const sorted_ptrs = a.sorted_token_ids_ptr.addPtr(rank);
            b.storeOpts(sorted_ptrs, offs, .{ .mask = valid });

            w.yieldAfter(.{ts.add(step)});
        }
    }
});
