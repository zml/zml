//! TTIR generation for MoE kernels. Each `generate*Ttir` builds a Triton IR
//! string via the `zml/triton/kernel.zig` DSL; the orchestration in
//! `zml/moe/triton.zig` picks which kernel to run and hands the IR to
//! `zml.ops.triton(...)`.

const std = @import("std");

const mlir = @import("mlir");
const stdx = @import("stdx");

const zml = @import("../zml.zig");
const DataType = zml.DataType;
const tri = @import("zml/triton");
const Kernel = tri.Kernel;
const Value = tri.Value;
const DType = tri.DType;
const Arg = tri.ArgSpec;

const log = std.log.scoped(.moe_triton);

// =============================================================================
// Per-kernel generation configs
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
    input_dtype: DataType,
    output_dtype: DataType,
    scale_dtype: DataType = .bf16,
    eps: f32 = 1e-6,
    fp8_min: f32,
    fp8_max: f32,
    use_ue8m0: bool = false,
};

// =============================================================================
// Helpers
// =============================================================================

fn dsl(dt: DataType) DType {
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

fn ctx() *mlir.Context {
    return zml.module.CompilationContext.current().mlir_ctx;
}

// =============================================================================
// count_and_sort_expert_tokens_kernel
// =============================================================================

pub fn generateCountAndSortKernelTtir(allocator: std.mem.Allocator, config: AlignBlockSizeGenerationConfig) ![:0]const u8 {
    const args = [_]Arg{
        Arg.ptr("topk_ids_ptr", .i32),
        Arg.ptr("sorted_token_ids_ptr", .i32),
        Arg.ptr("cumsum_ptr", .i32),
        Arg.ptr("out0_ptr", .i32),
        Arg.ptr("out1_ptr", .i32),
    };

    var kernel = try Kernel.init(allocator, ctx(), "count_and_sort_expert_tokens_kernel", &args, &.{});
    defer kernel.deinit();
    const k = &kernel;

    const block: i64 = @intCast(config.sort_block_size);
    const numel: i32 = @intCast(config.numel);
    const num_experts: i32 = @intCast(config.num_experts);

    const topk_ids = k.arg(0);
    const sorted_token_ids = k.arg(1);
    const cumsum = k.arg(2);

    const pid = k.programId(.x);
    const num_progs = k.numPrograms(.x);
    const token_offs = k.makeRange(0, @intCast(block));
    const block_i32: i32 = @intCast(block);
    const token_start_init = pid.mul(block_i32);
    const step = num_progs.mul(block_i32);

    const Ctx = struct {
        topk_ids: Value,
        sorted_token_ids: Value,
        cumsum: Value,
        token_offs: Value,
        numel: i32,
        num_experts: i32,
        step: Value,
        block: i64,
    };
    const c: Ctx = .{
        .topk_ids = topk_ids,
        .sorted_token_ids = sorted_token_ids,
        .cumsum = cumsum,
        .token_offs = token_offs,
        .numel = numel,
        .num_experts = num_experts,
        .step = step,
        .block = block,
    };

    const before = struct {
        fn call(kk: *Kernel, a: []const Value, cc: Ctx) tri.WhileBefore {
            const ts = a[0];
            return .{ .cond = ts.lt(cc.numel), .forwarded = kk.yield(.{ts}) };
        }
    }.call;

    const after = struct {
        fn call(kk: *Kernel, a: []const Value, cc: Ctx) []const Value {
            const ts = a[0];
            // offs = ts + token_offs  (broadcast)
            const offs = ts.splatTo(&.{cc.block}).add(cc.token_offs);
            // mask = offs < NUMEL
            const mask = offs.lt(cc.numel);
            // expert_vals = load(topk_ids + offs, mask=mask, other=NUM_EXPERTS)
            const topk_ptrs = cc.topk_ids.splatTo(&.{cc.block}).addPtr(offs);
            const num_experts_splat = kk.splat(cc.num_experts, &.{cc.block});
            const expert_vals = kk.loadOpts(topk_ptrs, .{
                .mask = mask,
                .other = num_experts_splat,
            });
            // valid = mask & (expert_vals < NUM_EXPERTS)
            const valid = mask.bitAnd(expert_vals.lt(cc.num_experts));
            // rank = atomic_add(cumsum + expert_vals, 1, mask=valid)
            const cumsum_ptrs = cc.cumsum.splatTo(&.{cc.block}).addPtr(expert_vals);
            const rank = kk.atomicRmwOpts(.add, cumsum_ptrs, kk.ones(&.{cc.block}, .i32), .{
                .mask = valid,
                .sem = .relaxed,
                .scope = .gpu,
            });
            // store(sorted_token_ids + rank, offs, mask=valid)
            const sorted_ptrs = cc.sorted_token_ids.splatTo(&.{cc.block}).addPtr(rank);
            kk.storeOpts(sorted_ptrs, offs, .{ .mask = valid });

            return kk.yield(.{ts.add(cc.step)});
        }
    }.call;

    _ = k.whileLoop(c, .{
        .inits = &.{token_start_init},
        .after_types = &.{k.scalarTy(.i32)},
        .before = before,
        .after = after,
    });

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize count_and_sort_expert_tokens_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// per_token_group_quant_fp8
// =============================================================================

pub fn generatePerTokenGroupQuantFp8KernelTtir(allocator: std.mem.Allocator, config: QuantGenerationConfig) ![:0]const u8 {
    const args = [_]Arg{
        Arg.ptr("y_ptr", dsl(config.input_dtype)),
        Arg.ptr("group_size_ptr", .i64),
        Arg.ptr("y_num_columns_ptr", .i64),
        Arg.ptr("y_row_stride_ptr", .i64),
        Arg.ptr("eps_ptr", .f32),
        Arg.ptr("y_q_ptr", dsl(config.output_dtype)),
        Arg.ptr("y_s_ptr", dsl(config.scale_dtype)),
    };
    var kernel = try Kernel.init(allocator, ctx(), "per_token_group_quant_fp8", &args, &.{});
    defer kernel.deinit();
    const k = &kernel;
    const block: i64 = @intCast(config.block);
    const out_dt = dsl(config.output_dtype);
    const scale_dt = dsl(config.scale_dtype);

    const y_ptr = k.arg(0);
    const group_size = k.load(k.arg(1));
    const y_num_columns = k.load(k.arg(2));
    const y_row_stride = k.load(k.arg(3));
    const eps = k.load(k.arg(4));
    const y_q_ptr = k.arg(5);
    const y_s_ptr = k.arg(6);

    const groups_per_row = y_num_columns.div(group_size);
    const g_id = k.programId(.x).to(.i64);

    // y_ptr += (g_id // groups_per_row) * y_row_stride + (g_id % groups_per_row) * group_size
    const row_off = g_id.div(groups_per_row).mul(y_row_stride);
    const grp_off = g_id.rem(groups_per_row).mul(group_size);
    const y_ptr_shifted = y_ptr.addPtr(row_off.add(grp_off));
    const y_q_ptr_shifted = y_q_ptr.addPtr(g_id.mul(group_size));
    const y_s_ptr_shifted = y_s_ptr.addPtr(g_id);

    // cols = arange(0, BLOCK) — in i64 for pointer offsets, in i32 for mask.
    const cols_i32 = k.arange(0, @intCast(block), .i32);
    const cols_i64 = cols_i32.to(.i64);
    const mask = cols_i32.lt(group_size.to(.i32));

    // Masked load with zero — the DSL auto-creates the zero-other tensor.
    const y_load_ptrs = y_ptr_shifted.splatTo(&.{block}).addPtr(cols_i64);
    const y_loaded = k.loadMasked(y_load_ptrs, mask);
    const y_f32 = y_loaded.to(.f32);

    // _absmax = max(reduce_max(abs(y)), eps)
    const abs_y = k.absf(y_f32);
    const absmax_raw = k.reduceMax(abs_y, 0);
    const absmax = k.maximumf(absmax_raw, eps);

    // scale_raw = absmax / fp8_max
    const scale_raw = absmax.mul(@as(f32, 1.0 / config.fp8_max));
    const y_s_scalar = if (config.use_ue8m0) k.exp2(k.ceil(k.log2(scale_raw))) else scale_raw;

    // y_q = clamp(y_f32 / y_s, fp8_min, fp8_max).to(output_dtype)
    const y_div = y_f32.div(y_s_scalar.splatTo(&.{block}));
    const clamped = k.clampf(
        y_div,
        k.splat(k.lift(@as(f32, config.fp8_min)), &.{block}),
        k.splat(k.lift(@as(f32, config.fp8_max)), &.{block}),
    );
    const y_q = clamped.to(out_dt);

    // Stores: tensor store for y_q (masked), scalar store for y_s.
    const y_q_ptrs = y_q_ptr_shifted.splatTo(&.{block}).addPtr(cols_i64);
    k.storeOpts(y_q_ptrs, y_q, .{ .mask = mask });
    k.store(y_s_ptr_shifted, y_s_scalar.to(scale_dt));

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize per_token_group_quant_fp8 TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// moe_align_block_size_kernel
// =============================================================================

pub fn generateAlignBlockSizeKernelTtir(allocator: std.mem.Allocator, config: AlignBlockSizeGenerationConfig) ![:0]const u8 {
    const args = [_]Arg{
        Arg.ptr("topk_ids_ptr", .i32),
        Arg.ptr("sorted_token_ids_ptr", .i32),
        Arg.ptr("expert_ids_ptr", .i32),
        Arg.ptr("num_tokens_post_pad_ptr", .i32),
        Arg.ptr("cumsum_ptr", .i32),
        Arg.ptr("out0_ptr", .i32),
        Arg.ptr("out1_ptr", .i32),
        Arg.ptr("out2_ptr", .i32),
        Arg.ptr("out3_ptr", .i32),
    };

    var kernel = try Kernel.init(allocator, ctx(), "moe_align_block_size_kernel", &args, &.{});
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

    const pid = k.programId(.x);

    const AlignCtx = struct {
        topk_ids: Value,
        sorted_token_ids: Value,
        expert_ids: Value,
        num_tokens_post_pad: Value,
        cumsum: Value,
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

    // pid==1 branch: fill sorted_token_ids with NUMEL.
    const fill_then = struct {
        fn call(kk: *Kernel, c: AlignCtx) []const Value {
            const FillCtx = struct { sorted: Value, block: i64, numel: i32, max_padded: i32 };
            const fc: FillCtx = .{ .sorted = c.sorted_token_ids, .block = c.hist_block, .numel = c.numel, .max_padded = c.max_num_tokens_padded };
            const body = struct {
                fn b(kkk: *Kernel, iv: Value, _: []const Value, ic: FillCtx) []const Value {
                    const offs = iv.splatTo(&.{ic.block}).add(kkk.arange(0, @intCast(ic.block), .i32));
                    const mask = offs.lt(ic.max_padded);
                    const sorted_ptrs = ic.sorted.splatTo(&.{ic.block}).addPtr(offs);
                    kkk.storeOpts(sorted_ptrs, kkk.splat(ic.numel, &.{ic.block}), .{ .mask = mask });
                    return &.{};
                }
            }.b;
            _ = kk.forLoop(fc, 0, fc.max_padded, @as(i32, @intCast(fc.block)), .{
                .body = body,
            });
            return &.{};
        }
    }.call;

    // pid==0 branch: histogram + cumsum + assignment.
    const compute_else = struct {
        fn call(kk: *Kernel, c: AlignCtx) []const Value {
            const expert_offs = kk.arange(0, @intCast(c.padded_num_experts), .i32);
            const expert_mask = expert_offs.lt(c.num_experts);
            const counts_init = kk.zeros(&.{c.padded_num_experts}, .i32);

            // Histogram loop over token chunks.
            const HistCtx = struct { topk_ids: Value, numel: i32, num_experts: i32, hist_block: i64, padded: i64 };
            const hctx: HistCtx = .{ .topk_ids = c.topk_ids, .numel = c.numel, .num_experts = c.num_experts, .hist_block = c.hist_block, .padded = c.padded_num_experts };
            const hist_body = struct {
                fn b(kkk: *Kernel, iv: Value, iter: []const Value, hc: HistCtx) []const Value {
                    const offs = iv.splatTo(&.{hc.hist_block}).add(kkk.arange(0, @intCast(hc.hist_block), .i32));
                    const mask = offs.lt(hc.numel);
                    const topk_ptrs = hc.topk_ids.splatTo(&.{hc.hist_block}).addPtr(offs);
                    const num_experts_other = kkk.splat(hc.num_experts, &.{hc.hist_block});
                    const expert_vals = kkk.loadOpts(topk_ptrs, .{
                        .mask = mask,
                        .other = num_experts_other,
                    });
                    const valid = mask.bitAnd(expert_vals.lt(hc.num_experts));
                    const h = kkk.histogramOpts(expert_vals, hc.padded, .{ .mask = valid });
                    return kkk.yield(.{iter[0].add(h)});
                }
            }.b;
            const counts_results = kk.forLoop(hctx, 0, c.numel, @as(i32, @intCast(c.hist_block)), .{
                .inits = &.{counts_init},
                .body = hist_body,
            });
            const counts = counts_results[0];

            // padded_counts = where(expert_mask, cdiv(counts, BLOCK_SIZE_M)*BLOCK_SIZE_M, 0)
            const block_size_m_v = kk.splat(@as(i32, @intCast(c.block_size_m)), &.{c.padded_num_experts});
            const padded_counts_full = counts.cdiv(block_size_m_v).mul(block_size_m_v);
            const padded_counts = kk.select(expert_mask, padded_counts_full, kk.zeros(&.{c.padded_num_experts}, .i32));

            const padded_cumsum = kk.scanSum(padded_counts, 0);
            const starts = padded_cumsum.sub(padded_counts);
            const total = kk.reduceSum(padded_counts, 0);

            // store(cumsum + expert_offs, starts, mask=expert_mask)
            kk.storeOpts(c.cumsum.splatTo(&.{c.padded_num_experts}).addPtr(expert_offs), starts, .{ .mask = expert_mask });
            // store(cumsum + NUM_EXPERTS, total)
            kk.store(c.cumsum.addPtr(c.num_experts), total);
            kk.store(c.num_tokens_post_pad, total);

            // Block-to-expert assignment.
            const AssignCtx = struct { cumsum: Value, expert_ids: Value, block_size_m: i64, num_experts: i32, hist_block: i64, max_num_m_blocks: i64 };
            const actx: AssignCtx = .{ .cumsum = c.cumsum, .expert_ids = c.expert_ids, .block_size_m = c.block_size_m, .num_experts = c.num_experts, .hist_block = c.hist_block, .max_num_m_blocks = c.max_num_m_blocks };
            const assign_body = struct {
                fn b(kkk: *Kernel, block_start: Value, _: []const Value, ac: AssignCtx) []const Value {
                    const block_offs = kkk.arange(0, @intCast(ac.hist_block), .i32);
                    const block_ids = block_start.splatTo(&.{ac.hist_block}).add(block_offs);
                    const block_mask = block_ids.lt(@as(i32, @intCast(ac.max_num_m_blocks)));
                    const block_offsets = block_ids.mul(@as(i32, @intCast(ac.block_size_m)));
                    const init = kkk.splat(@as(i32, -1), &.{ac.hist_block});

                    // Inner loop over experts.
                    const ExpCtx = struct { cumsum: Value, block_offsets: Value, block_mask: Value, hist_block: i64 };
                    const ec: ExpCtx = .{ .cumsum = ac.cumsum, .block_offsets = block_offsets, .block_mask = block_mask, .hist_block = ac.hist_block };
                    const exp_body = struct {
                        fn eb(kkkk: *Kernel, iv: Value, iter: []const Value, ee: ExpCtx) []const Value {
                            const start_v = kkkk.load(ee.cumsum.addPtr(iv));
                            const end_v = kkkk.load(ee.cumsum.addPtr(iv.add(1)));
                            const in_range = ee.block_mask
                                .bitAnd(ee.block_offsets.ge(start_v.splatTo(&.{ee.hist_block})))
                                .bitAnd(ee.block_offsets.lt(end_v.splatTo(&.{ee.hist_block})));
                            return kkkk.yield(.{kkkk.select(in_range, iv.splatTo(&.{ee.hist_block}), iter[0])});
                        }
                    }.eb;
                    const exp_results = kkk.forLoop(ec, 0, ac.num_experts, 1, .{
                        .inits = &.{init},
                        .body = exp_body,
                    });
                    const block_expert = exp_results[0];
                    kkk.storeOpts(ac.expert_ids.splatTo(&.{ac.hist_block}).addPtr(block_ids), block_expert, .{ .mask = block_mask });
                    return &.{};
                }
            }.b;
            _ = kk.forLoop(actx, 0, @as(i32, @intCast(c.max_num_m_blocks)), @as(i32, @intCast(c.hist_block)), .{
                .body = assign_body,
            });
            return &.{};
        }
    }.call;

    _ = k.ifThenElse(ax, .{
        .cond = pid.eq(1),
        .then_ = fill_then,
        .else_ = compute_else,
    });

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize moe_align_block_size_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

// =============================================================================
// fused_moe_kernel
// =============================================================================

pub fn generateFusedMoeKernelTtir(allocator: std.mem.Allocator, config: GenerationConfig) ![:0]const u8 {
    if (config.use_fp8_w8a8 or config.use_int8_w8a8 or config.use_int8_w8a16 or config.has_bias or config.per_channel_quant) {
        log.err("fused_moe_kernel: unsupported config (fp8/int8/bias/per_channel)", .{});
        return error.TritonTtirGenerationFailed;
    }

    const args = [_]Arg{
        Arg.ptr("a_ptr", dsl(config.a_dtype)),
        Arg.ptr("b_ptr", dsl(config.b_dtype)),
        Arg.ptr("b_bias_ptr", dsl(config.b_bias_dtype orelse config.c_dtype)),
        Arg.ptr("a_scale_ptr", dsl(config.a_scale_dtype orelse .f32)),
        Arg.ptr("b_scale_ptr", dsl(config.b_scale_dtype orelse .f32)),
        Arg.ptr("topk_weights_ptr", dsl(config.topk_weights_dtype orelse .f32)),
        Arg.ptr("sorted_token_ids_ptr", .i32),
        Arg.ptr("expert_ids_ptr", .i32),
        Arg.ptr("num_tokens_post_padded_ptr", .i32),
        Arg.ptr("N_ptr", .i64),
        Arg.ptr("K_ptr", .i64),
        Arg.ptr("EM_ptr", .i64),
        Arg.ptr("num_valid_tokens_ptr", .i64),
        Arg.ptr("stride_am_ptr", .i64),
        Arg.ptr("stride_be_ptr", .i64),
        Arg.ptr("stride_bn_ptr", .i64),
        Arg.ptr("stride_cm_ptr", .i64),
        Arg.ptr("stride_asm_ptr", .i64),
        Arg.ptr("stride_ask_ptr", .i64),
        Arg.ptr("stride_bse_ptr", .i64),
        Arg.ptr("stride_bsk_ptr", .i64),
        Arg.ptr("stride_bsn_ptr", .i64),
        Arg.ptr("stride_bbe_ptr", .i64),
        Arg.ptr("stride_bbn_ptr", .i64),
        Arg.ptr("c_ptr", dsl(config.c_dtype)),
    };

    var kernel = try Kernel.init(allocator, ctx(), "fused_moe_kernel", &args, &.{});
    defer kernel.deinit();
    const k = &kernel;

    const block_m: i64 = @intCast(config.block_size_m);
    const block_n: i64 = @intCast(config.block_size_n);
    const block_k: i64 = @intCast(config.block_size_k);
    const group_size_m: i64 = @intCast(config.group_size_m);
    const top_k: i64 = @intCast(config.top_k);

    const a_ptr = k.arg(0);
    const b_ptr = k.arg(1);
    const topk_weights_ptr = k.arg(5);
    const sorted_token_ids_ptr = k.arg(6);
    const expert_ids_ptr = k.arg(7);
    const num_tokens_post_padded_ptr = k.arg(8);
    const c_ptr = k.arg(24);

    // Runtime scalars, each clamped to a multiple of 16.
    const block_floor = struct {
        fn f(v: Value) Value {
            return v.div(@as(i64, 16)).mul(@as(i64, 16));
        }
    }.f;
    const n_block = block_floor(k.load(k.arg(9)));
    const k_block = block_floor(k.load(k.arg(10)));
    const em_block = block_floor(k.load(k.arg(11)));
    const num_valid_tokens = k.load(k.arg(12));
    const stride_am_block = block_floor(k.load(k.arg(13)));
    const stride_be_block = block_floor(k.load(k.arg(14)));
    const stride_bn_block = block_floor(k.load(k.arg(15)));
    const stride_cm_block = block_floor(k.load(k.arg(16)));
    const stride_ak_block = k.lift(@as(i64, 1));
    const stride_bk_block = k.lift(@as(i64, 1));
    const stride_cn_block = k.lift(@as(i64, 1));

    // pid grouping (all in i64).
    const pid = k.programId(.x).to(.i64);
    const num_pid_m = em_block.cdiv(block_m);
    const num_pid_n = n_block.cdiv(block_n);
    const num_pid_in_group = num_pid_n.mul(group_size_m);
    const group_id = pid.div(num_pid_in_group);
    const first_pid_m = group_id.mul(group_size_m);
    const gsm_actual = num_pid_m.sub(first_pid_m).min(group_size_m);
    const pid_mod_in_group = pid.rem(num_pid_in_group);
    const pid_m = first_pid_m.add(pid_mod_in_group.rem(gsm_actual));
    const pid_n = pid_mod_in_group.div(gsm_actual);

    // Early-return guard: only run when pid_m * BLOCK_M < num_tokens_post_padded.
    const ntpp = k.load(num_tokens_post_padded_ptr).to(.i64);
    const in_range = pid_m.mul(block_m).lt(ntpp);

    const InRangeCtx = struct {
        a_ptr: Value,
        b_ptr: Value,
        c_ptr: Value,
        topk_weights_ptr: Value,
        sorted_token_ids_ptr: Value,
        expert_ids_ptr: Value,
        num_valid_tokens: Value,
        n_block: Value,
        k_block: Value,
        stride_am_block: Value,
        stride_ak_block: Value,
        stride_be_block: Value,
        stride_bk_block: Value,
        stride_bn_block: Value,
        stride_cm_block: Value,
        stride_cn_block: Value,
        pid_m: Value,
        pid_n: Value,
        block_m: i64,
        block_n: i64,
        block_k: i64,
        top_k: i64,
        naive: bool,
        mul_routed: bool,
        a_dtype: DType,
        b_dtype: DType,
        compute_dtype: DType,
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
        .pid_m = pid_m,
        .pid_n = pid_n,
        .block_m = block_m,
        .block_n = block_n,
        .block_k = block_k,
        .top_k = top_k,
        .naive = config.naive_block_assignment,
        .mul_routed = config.mul_routed_weight,
        .a_dtype = dsl(config.a_dtype),
        .b_dtype = dsl(config.b_dtype),
        .compute_dtype = dsl(config.compute_type),
    };

    const compute = struct {
        fn cmp(kk: *Kernel, c: InRangeCtx) []const Value {
            buildFusedMain(kk, c);
            return &.{};
        }
    }.cmp;

    _ = k.ifThenElse(ic, .{
        .cond = in_range,
        .then_ = compute,
    });

    return kernel.finish(&.{}, allocator) catch |err| {
        log.err("failed to finalize fused_moe_kernel TTIR: {}", .{err});
        return error.TritonTtirGenerationFailed;
    };
}

/// Body of fused_moe_kernel inside the `pid_m * BLOCK_M < ntpp` guard.
/// Bf16 / no-bias / no-quant path only.
fn buildFusedMain(k: *Kernel, ic: anytype) void {
    const block_m = ic.block_m;
    const block_n = ic.block_n;
    const block_k = ic.block_k;
    const top_k = ic.top_k;

    // offs_token: depends on naive_block_assignment.
    const offs = k.arange(0, @intCast(block_m), .i64);
    const offs_token = blk: {
        if (ic.naive) {
            // offs_token = where(offs == 0, pid_m, num_valid_tokens)
            break :blk k.select(offs.eq(@as(i64, 0)), ic.pid_m.splatTo(&.{block_m}), ic.num_valid_tokens.splatTo(&.{block_m}));
        } else {
            // offs_token = load(sorted_token_ids_ptr + pid_m * BLOCK_M + offs)
            const ids_ptrs = ic.sorted_token_ids_ptr.splatTo(&.{block_m}).addPtr(ic.pid_m.mul(block_m).splatTo(&.{block_m}).add(offs));
            const loaded = k.load(ids_ptrs);
            break :blk loaded.to(.i64);
        }
    };
    const token_mask = offs_token.lt(ic.num_valid_tokens);

    // off_experts = expert_ids[pid_m] (i32 → i64)
    const off_experts = k.load(ic.expert_ids_ptr.addPtr(ic.pid_m.to(.i32))).to(.i64);
    const is_dead = off_experts.eq(@as(i64, -1));

    // offs_bn = (pid_n * BLOCK_N + arange(BLOCK_N).to(i64)) % n_block
    const arange_n = k.arange(0, @intCast(block_n), .i64);
    const offs_bn = ic.pid_n.mul(block_n).splatTo(&.{block_n}).add(arange_n).rem(ic.n_block.splatTo(&.{block_n}));
    // offs_cn = pid_n * BLOCK_N + arange_n (for c-ptr addressing; i64)
    const offs_cn = ic.pid_n.to(.i32).mul(@as(i32, @intCast(block_n))).splatTo(&.{block_n}).add(k.arange(0, @intCast(block_n), .i32)).to(.i64);
    const offs_k = k.arange(0, @intCast(block_k), .i64);

    const AliveCtx = struct {
        a_ptr: Value,
        b_ptr: Value,
        c_ptr: Value,
        topk_weights_ptr: Value,
        offs_token: Value,
        token_mask: Value,
        offs_bn: Value,
        offs_cn: Value,
        offs_k: Value,
        off_experts: Value,
        k_block: Value,
        n_block: Value,
        stride_am_block: Value,
        stride_ak_block: Value,
        stride_be_block: Value,
        stride_bk_block: Value,
        stride_bn_block: Value,
        stride_cm_block: Value,
        stride_cn_block: Value,
        block_m: i64,
        block_n: i64,
        block_k: i64,
        top_k: i64,
        mul_routed: bool,
        a_dtype: DType,
        b_dtype: DType,
        compute_dtype: DType,
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
        .mul_routed = ic.mul_routed,
        .a_dtype = ic.a_dtype,
        .b_dtype = ic.b_dtype,
        .compute_dtype = ic.compute_dtype,
    };

    const zero_branch = struct {
        fn z(kk: *Kernel, c: AliveCtx) []const Value {
            const zero = kk.zeros(&.{ c.block_m, c.block_n }, c.compute_dtype);
            writeBlockC(kk, c, zero);
            return &.{};
        }
    }.z;

    const alive_branch = struct {
        fn a(kk: *Kernel, c: AliveCtx) []const Value {
            buildAliveBody(kk, c);
            return &.{};
        }
    }.a;

    _ = k.ifThenElse(ac, .{
        .cond = is_dead,
        .then_ = zero_branch,
        .else_ = alive_branch,
    });
}

/// Final `c_ptrs` + `c_mask` build + masked store. Shared by the zero-write
/// and alive branches.
fn writeBlockC(k: *Kernel, c: anytype, accumulator: Value) void {
    const m = c.block_m;
    const n = c.block_n;
    // c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    const cm_off = c.offs_token.broadcast2d(1, m, n).mul(c.stride_cm_block.splatTo(&.{ m, n }));
    const cn_off = c.offs_cn.broadcast2d(0, m, n).mul(c.stride_cn_block.splatTo(&.{ m, n }));
    const c_ptrs = c.c_ptr.splatTo(&.{ m, n }).addPtr(cm_off.add(cn_off));
    // c_mask = token_mask[:, None] & (offs_cn[None, :] < n_block)
    const c_mask = k.mask2d(c.token_mask, c.offs_cn.lt(c.n_block), m, n);
    k.storeOpts(c_ptrs, accumulator, .{ .mask = c_mask });
}

fn buildAliveBody(k: *Kernel, ac: anytype) void {
    const block_m = ac.block_m;
    const block_n = ac.block_n;
    const block_k = ac.block_k;
    const top_k = ac.top_k;

    // a_ptrs = a_ptr + (offs_token // top_k) * stride_am + offs_k * stride_ak   (shape [m,k])
    const am_term = ac.offs_token.div(top_k).broadcast2d(1, block_m, block_k).mul(ac.stride_am_block.splatTo(&.{ block_m, block_k }));
    const ak_term = ac.offs_k.broadcast2d(0, block_m, block_k).mul(ac.stride_ak_block.splatTo(&.{ block_m, block_k }));
    const a_ptrs_init = ac.a_ptr.splatTo(&.{ block_m, block_k }).addPtr(am_term.add(ak_term));

    // b_ptrs = b_ptr + off_exp*stride_be + offs_k[:,None]*stride_bk + offs_bn[None,:]*stride_bn   (shape [k,n])
    const be_term = ac.off_experts.mul(ac.stride_be_block).splatTo(&.{ block_k, block_n });
    const bk_term = ac.offs_k.broadcast2d(1, block_k, block_n).mul(ac.stride_bk_block.splatTo(&.{ block_k, block_n }));
    const bn_term = ac.offs_bn.broadcast2d(0, block_k, block_n).mul(ac.stride_bn_block.splatTo(&.{ block_k, block_n }));
    const b_ptrs_init = ac.b_ptr.splatTo(&.{ block_k, block_n }).addPtr(be_term.add(bk_term.add(bn_term)));

    // acc = zeros(block_m, block_n, f32)
    const acc_init = k.zeros(&.{ block_m, block_n }, .f32);

    const num_k_iters = ac.k_block.cdiv(block_k).to(.i32);

    const KCtx = struct {
        offs_k: Value,
        token_mask: Value,
        k_block: Value,
        block_k: i64,
        block_m: i64,
        block_n: i64,
        stride_ak_block: Value,
        stride_bk_block: Value,
        a_dtype: DType,
        b_dtype: DType,
    };
    const kctx: KCtx = .{
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
        fn b(kk: *Kernel, k_iter: Value, iter: []const Value, kc: KCtx) []const Value {
            const acc = iter[0];
            const a_ptrs = iter[1];
            const b_ptrs = iter[2];

            // k_remaining = k_block - k_iter * BLOCK_K   (i64)
            const k_remaining = kc.k_block.sub(k_iter.to(.i64).mul(kc.block_k));

            // mask_a = token_mask[:, None] & (offs_k[None, :] < k_remaining)
            const offs_k_lt = kc.offs_k.lt(k_remaining);
            const mask_a = kk.mask2d(kc.token_mask, offs_k_lt, kc.block_m, kc.block_k);
            const a_val = kk.loadMasked(a_ptrs, mask_a);

            // mask_b = (offs_k[:, None] < k_remaining)   — broadcast along N axis.
            const mask_b = offs_k_lt.broadcast2d(1, kc.block_k, kc.block_n);
            const b_val = kk.loadMasked(b_ptrs, mask_b);

            // acc = tt.dot(a, b, acc)
            const new_acc = kk.dotOpts(a_val, b_val, acc, .{ .input_precision = .ieee, .max_num_imprecise_acc = 0 });

            // Advance pointers along K.
            const new_a_ptrs = a_ptrs.addPtr(kc.stride_ak_block.mul(kc.block_k).splatTo(&.{ kc.block_m, kc.block_k }));
            const new_b_ptrs = b_ptrs.addPtr(kc.stride_bk_block.mul(kc.block_k).splatTo(&.{ kc.block_k, kc.block_n }));

            return kk.yield(.{ new_acc, new_a_ptrs, new_b_ptrs });
        }
    }.b;

    const loop_results = k.forLoop(kctx, 0, num_k_iters, 1, .{
        .inits = &.{ acc_init, a_ptrs_init, b_ptrs_init },
        .body = k_body,
    });
    var acc_final = loop_results[0];

    if (ac.mul_routed) {
        const tw_ptrs = ac.topk_weights_ptr.splatTo(&.{block_m}).addPtr(ac.offs_token);
        const tw = k.loadMasked(tw_ptrs, ac.token_mask);
        acc_final = acc_final.mul(tw.broadcast2d(1, block_m, block_n));
    }

    const acc_cast = acc_final.to(ac.compute_dtype);
    writeBlockC(k, ac, acc_cast);
}
