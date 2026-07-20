//! Harness registration for `_gemm_a16w8_blockscale`.
//!
//! Runtime scalar inputs are passed as pointers, and the output pointer is
//! declared last to match the XLA-compatible ABI expected by the harness.

const std = @import("std");

const dialects = @import("mlir/dialects");
const ttir_dialect = dialects.ttir;

const zml = @import("../zml.zig");
const tri = zml.kernel.triton;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const Cfg = struct {
    a_dtype: tri.DType = .bf16,
    b_dtype: tri.DType = .i8,
    c_dtype: tri.DType = .bf16,
    b_scale_dtype: tri.DType = .f32,

    GROUP_K: i32 = 128,
    GROUP_N: i32 = 128,
    BLOCK_SIZE_M: i32 = 16,
    BLOCK_SIZE_N: i32 = 64,
    BLOCK_SIZE_K: i32 = 64,
    GROUP_SIZE_M: i32 = 1,
    NUM_KSPLIT: i32 = 1,
    SPLITK_BLOCK_SIZE: i32 = 128,
    EVEN_K: bool = true,
    GRID_MN: i32 = 64,
    PREQUANT: bool = false,
    DTYPE_MAX: f32 = 127.0,
    DTYPE_MIN: f32 = -128.0,
    num_warps: i32 = 4,
    num_stages: i32 = 2,
    waves_per_eu: i32 = 0,
    matrix_instr_nonkdim: i32 = 16,
    cache_modifier: ttir_dialect.CacheModifier = .none,
};

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_gemm_a16w8_blockscale",
    .inputs = &.{
        "a_ptr",
        "b_ptr",
        "b_scale_ptr",
        "M_ptr",
        "N_ptr",
        "K_ptr",
        "stride_am_ptr",
        "stride_ak_ptr",
        "stride_bk_ptr",
        "stride_bn_ptr",
        "stride_ck_ptr",
        "stride_cm_ptr",
        "stride_cn_ptr",
        "stride_bscale_k_ptr",
        "stride_bscale_n_ptr",
    },
    .outputs = &.{"c"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .a_ptr = .{ .ptr = cfg.a_dtype },
        .b_ptr = .{ .ptr = cfg.b_dtype },
        .b_scale_ptr = .{ .ptr = cfg.b_scale_dtype },
        .M_ptr = .{ .ptr = .i64 },
        .N_ptr = .{ .ptr = .i64 },
        .K_ptr = .{ .ptr = .i64 },
        .stride_am_ptr = .{ .ptr = .i64 },
        .stride_ak_ptr = .{ .ptr = .i64 },
        .stride_bk_ptr = .{ .ptr = .i64 },
        .stride_bn_ptr = .{ .ptr = .i64 },
        .stride_ck_ptr = .{ .ptr = .i64 },
        .stride_cm_ptr = .{ .ptr = .i64 },
        .stride_cn_ptr = .{ .ptr = .i64 },
        .stride_bscale_k_ptr = .{ .ptr = .i64 },
        .stride_bscale_n_ptr = .{ .ptr = .i64 },
        .c_ptr = .{ .ptr = cfg.c_dtype },
    });

    const M_val = b.load(a.M_ptr);
    const N_val = b.load(a.N_ptr);
    const K_val = b.load(a.K_ptr);
    const stride_am = b.load(a.stride_am_ptr);
    const stride_ak = b.load(a.stride_ak_ptr);
    const stride_bk = b.load(a.stride_bk_ptr);
    const stride_bn = b.load(a.stride_bn_ptr);
    const stride_ck = b.load(a.stride_ck_ptr);
    const stride_cm = b.load(a.stride_cm_ptr);
    const stride_cn = b.load(a.stride_cn_ptr);
    const stride_bscale_k = b.load(a.stride_bscale_k_ptr);
    const stride_bscale_n = b.load(a.stride_bscale_n_ptr);

    b.assume(stride_am.gt(0));
    b.assume(stride_ak.gt(0));
    b.assume(stride_bk.gt(0));
    b.assume(stride_bn.gt(0));
    b.assume(stride_ck.gt(0));
    b.assume(stride_cm.gt(0));
    b.assume(stride_cn.gt(0));
    b.assume(stride_bscale_k.gt(0));
    b.assume(stride_bscale_n.gt(0));

    // -----------------------------------------------------------
    // Map program ids `pid` to the block of C it should compute.
    // This is done in a grouped ordering to promote L2 data reuse.
    const pid_unified = b.programId(.x);
    const pid_k = pid_unified.rem(cfg.NUM_KSPLIT);
    const pid = pid_unified.div(cfg.NUM_KSPLIT);
    const num_pid_m = M_val.cdiv(cfg.BLOCK_SIZE_M);
    const num_pid_n = N_val.cdiv(cfg.BLOCK_SIZE_N);

    const pid_m, const pid_n = pidGrid(pid, num_pid_m, num_pid_n, cfg.GROUP_SIZE_M);

    b.assume(pid_m.ge(0));
    b.assume(pid_n.ge(0));
    b.assume(pid_k.ge(0));

    var body = b.openIf(pid_k.mul(cfg.SPLITK_BLOCK_SIZE).lt(K_val));
    {
        // SPLITK_BLOCK_SIZE = tl.cdiv(K, NUM_KSPLIT)
        const num_k_iter = @divTrunc(cfg.SPLITK_BLOCK_SIZE, cfg.BLOCK_SIZE_K);

        // Create pointers for first block of A and B input matrices
        const offs_k = b.arange(0, cfg.BLOCK_SIZE_K, .i32);
        const offs_k_split = pid_k.mul(cfg.SPLITK_BLOCK_SIZE).add(offs_k);
        const offs_am = pid_m.mul(cfg.BLOCK_SIZE_M).add(b.arange(0, cfg.BLOCK_SIZE_M, .i32)).rem(M_val);
        const offs_bn = pid_n.mul(cfg.BLOCK_SIZE_N).add(b.arange(0, cfg.BLOCK_SIZE_N, .i32)).rem(N_val);
        const a_ptrs = a.a_ptr.addPtr(
            offs_am.expandDims(1).mul(stride_am).add(offs_k_split.expandDims(0).mul(stride_ak)),
        );
        const b_ptrs = a.b_ptr.addPtr(
            offs_k_split.expandDims(1).mul(stride_bk).add(offs_bn.expandDims(0).mul(stride_bn)),
        );

        // Create pointers for the scales
        const offs_ks = pid_k.mul(cfg.SPLITK_BLOCK_SIZE).div(cfg.GROUP_K);
        const offs_bsn = offs_bn.div(cfg.GROUP_N);
        const b_scale_ptrs = a.b_scale_ptr
            .addPtr(offs_ks.mul(stride_bscale_k))
            .addPtr(offs_bsn.mul(stride_bscale_n));

        const acc_dtype: tri.DType = if (cfg.c_dtype == .i8) .i32 else .f32;
        const accumulator = b.zeros(&.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_N }, acc_dtype);

        var loop = b.openFor(
            pid_k.mul(num_k_iter),
            pid_k.add(1).mul(num_k_iter),
            1,
            .{ a_ptrs, b_ptrs, accumulator },
        );
        loop.opts.num_stages = cfg.num_stages;
        {
            const k = loop.iv;
            const a_ptrs_loop = loop.carried[0];
            const b_ptrs_loop = loop.carried[1];
            const accumulator_loop = loop.carried[2];

            // Load the next block of A and B, generate a mask by checking the K dimension.
            // If it is out of bounds, set it to 0.
            const a_val = if (cfg.EVEN_K)
                b.load(a_ptrs_loop)
            else
                b.loadOpts(a_ptrs_loop, .{
                    .mask = offs_k.expandDims(0).lt(K_val.sub(k.mul(cfg.BLOCK_SIZE_K))),
                    .other = b.zeros(&.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_K }, cfg.a_dtype),
                });
            const b_val_raw = if (cfg.EVEN_K)
                b.loadOpts(b_ptrs_loop, .{ .cache_modifier = cfg.cache_modifier })
            else
                b.loadOpts(b_ptrs_loop, .{
                    .mask = offs_k.expandDims(1).lt(K_val.sub(k.mul(cfg.BLOCK_SIZE_K))),
                    .other = b.zeros(&.{ cfg.BLOCK_SIZE_K, cfg.BLOCK_SIZE_N }, cfg.b_dtype),
                });

            const b_scale = b.load(b_scale_ptrs);

            const new_accumulator = if (cfg.PREQUANT) prequant: {
                const a_quant, const a_scale = fp8QuantOp(b, a_val, cfg);
                const a_quant_2d = b.reshape(a_quant.to(cfg.b_dtype), &.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_K });
                const a_scale_1d = b.reshape(a_scale, &.{cfg.BLOCK_SIZE_M});
                const prequant_acc_dtype: tri.DType = if (cfg.b_dtype == .f8e4m3fn or cfg.b_dtype == .f8e5m2) .f32 else .i32;
                const prequant_dot = b.dotOpts(a_quant_2d, b_val_raw, b.zeros(&.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_N }, prequant_acc_dtype), .{
                    .input_precision = .tf32,
                    .max_num_imprecise_acc = 0,
                });
                const a_scale_2d = a_scale_1d.expandDims(1);
                const prequant_dot_f32 = prequant_dot.to(.f32);
                break :prequant accumulator_loop.add(prequant_dot_f32.mul(a_scale_2d).mul(b_scale.expandDims(0)));
            } else no_prequant: {
                const b_val = b_val_raw.to(cfg.a_dtype);
                const no_prequant_dot = b.dotOpts(a_val, b_val, b.zeros(&.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_N }, acc_dtype), .{
                    .input_precision = .tf32,
                    .max_num_imprecise_acc = 0,
                });
                break :no_prequant accumulator_loop.add(no_prequant_dot.mul(b_scale.expandDims(0)));
            };

            // Advance the ptrs to the next K block.
            const block_size_k = b.liftAs(cfg.BLOCK_SIZE_K, .i64);
            const new_a_ptrs = a_ptrs_loop.addPtr(block_size_k.mul(stride_ak));
            const new_b_ptrs = b_ptrs_loop.addPtr(block_size_k.mul(stride_bk));

            loop.yield(.{ new_a_ptrs, new_b_ptrs, new_accumulator });
        }

        const c = loop.results[2].to(cfg.c_dtype);

        // Write back the block of the output matrix C with masks.
        const offs_cm = pid_m.mul(cfg.BLOCK_SIZE_M).add(b.arange(0, cfg.BLOCK_SIZE_M, .i32).to(.i64));
        const offs_cn = pid_n.mul(cfg.BLOCK_SIZE_N).add(b.arange(0, cfg.BLOCK_SIZE_N, .i32).to(.i64));
        const c_ptrs_col = a.c_ptr.addPtr(stride_cm.mul(offs_cm.expandDims(1)));
        const offs_cn_row = offs_cn.expandDims(0);
        const cn_off = stride_cn.mul(offs_cn_row);
        const c_ptrs_2d = b.broadcastTo(c_ptrs_col, &.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_N });
        const cn_off_full = b.broadcastTo(cn_off, &.{ cfg.BLOCK_SIZE_M, cfg.BLOCK_SIZE_N });
        const c_ptrs = c_ptrs_2d
            .addPtr(cn_off_full)
            .addPtr(pid_k.mul(stride_ck));
        const c_mask = offs_cm.expandDims(1).lt(M_val).bitAnd(offs_cn_row.lt(N_val));
        b.storeOpts(c_ptrs, c, .{ .mask = c_mask });

        body.yieldThen(.{});
    }
}

fn fp8QuantOp(b: *tri.Builder, x: tri.Value, cfg: Cfg) struct { tri.Value, tri.Value } {
    const NUM_QUANT_BLOCKS = @divTrunc(cfg.BLOCK_SIZE_K, cfg.BLOCK_SIZE_K);
    const x_3d = b.reshape(x, &.{ cfg.BLOCK_SIZE_M, NUM_QUANT_BLOCKS, cfg.BLOCK_SIZE_K });
    const m = b.maxOpts(b.abs(x_3d).to(.f32), .{ .axis = 2 }).maximum(1e-10);
    const scale_out = m.to(.f32).div(cfg.DTYPE_MAX);
    const scale_recip = b.liftAs(1.0, .f32).div(b.reshape(scale_out, &.{ cfg.BLOCK_SIZE_M, NUM_QUANT_BLOCKS, 1 }));
    const x_scaled = x_3d.to(.f32).mul(scale_recip);
    const lo = b.splat(b.liftAs(cfg.DTYPE_MIN, .f32), &.{ cfg.BLOCK_SIZE_M, NUM_QUANT_BLOCKS, cfg.BLOCK_SIZE_K });
    const hi = b.splat(b.liftAs(cfg.DTYPE_MAX, .f32), &.{ cfg.BLOCK_SIZE_M, NUM_QUANT_BLOCKS, cfg.BLOCK_SIZE_K });
    const x_clamped = b.clampf(x_scaled, lo, hi);

    return .{ x_clamped, scale_out };
}

fn pidGrid(
    pid: tri.Value,
    num_pid_m: tri.Value,
    num_pid_n: tri.Value,
    GROUP_SIZE_M: i32,
) struct { tri.Value, tri.Value } {
    if (GROUP_SIZE_M == 1) {
        const pid_m = pid.div(num_pid_n);
        const pid_n = pid.rem(num_pid_n);
        return .{ pid_m, pid_n };
    } else {
        const num_pid_in_group = num_pid_n.mul(GROUP_SIZE_M);
        const group_id = pid.div(num_pid_in_group);
        const first_pid_m = group_id.mul(GROUP_SIZE_M);
        const group_size_m = num_pid_m.sub(first_pid_m).minimum(GROUP_SIZE_M);
        const pid_m = first_pid_m.add(pid.rem(group_size_m));
        const pid_n = pid.rem(num_pid_in_group).div(group_size_m);
        return .{ pid_m, pid_n };
    }
}
