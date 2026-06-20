//! Harness registration for the BF16/INT4 MoE GEMM kernel.

const std = @import("std");

const dialects = @import("mlir/dialects");
const cf = dialects.cf;
const harness = @import("harness");
const mlir = @import("mlir");
const zml = @import("zml");
const ttir = dialects.ttir;
const tri = zml.kernel.triton;
const ops = zml.ops;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const Value = tri.Value;

pub const Cfg = struct {
    x_dtype: tri.DType = .bf16,
    w_dtype: tri.DType = .i8,
    w_mx_scale_dtype: tri.DType = .i8,
    b_dtype: tri.DType = .bf16,
    gammas_dtype: tri.DType = .f32,
    y_dtype: tri.DType = .bf16,

    HAS_B: bool = false,
    HAS_GAMMAS: bool = false,
    HAS_GATHER_INDX: bool = false,
    HAS_EXPT_OFFS_SUM: bool = false,
    APPLY_SWIGLU: bool = false,
    ACTIVATION_REDUCTION_N: i32 = 1,
    SWIGLU_ADD_RESIDUAL: bool = false,
    N_EXPTS_ACT: i32 = 2,
    BLOCK_M: i32 = 16,
    BLOCK_N: i32 = 64,
    BLOCK_K: i32 = 64,
    GROUP_M: i32 = 1,
    XCD_SWIZZLE: i32 = 1,
    SWIZZLE_MX_SCALE: enum { none, cdna4_scale } = .none,
    EVEN_K: bool = true,
    MASK_K_LIMIT: i32 = 64,
    SPLIT_K: i32 = 1,
    W_CACHE_MODIFIER: enum { none, cg, ca } = .none,
    UPCAST_INDICES: bool = false,
};

fn cacheModifier(cfg: Cfg) ttir.CacheModifier {
    return switch (cfg.W_CACHE_MODIFIER) {
        .none => .none,
        .cg => .cg,
        .ca => .ca,
    };
}

fn pidGrid(pid: Value, num_pid_m: Value, num_pid_n: Value, group_size_m: i32) struct { Value, Value } {
    if (group_size_m == 1) {
        const pid_m = pid.div(num_pid_n);
        const pid_n = pid.rem(num_pid_n);
        return .{ pid_m, pid_n };
    } else {
        const num_pid_in_group = num_pid_n.mul(group_size_m);
        const group_id = pid.div(num_pid_in_group);
        const first_pid_m = group_id.mul(group_size_m);
        const group_size_m_actual = num_pid_m.sub(first_pid_m).minimum(group_size_m);
        const pid_m = first_pid_m.add(pid.rem(group_size_m_actual));
        const pid_n = pid.rem(num_pid_in_group).div(group_size_m_actual);
        return .{ pid_m, pid_n };
    }
}

fn xcdSwizzlePidI32(pid: Value, domain_size: Value, XCD_SWIZZLE: i32) Value {
    const pids_per_group = domain_size.div(XCD_SWIZZLE);
    const extra_pid_groups = domain_size.rem(XCD_SWIZZLE);
    const group_i32 = pid.rem(XCD_SWIZZLE);
    const local_pid_i32 = pid.div(XCD_SWIZZLE);
    const group = group_i32.to(.i64);
    const group_scaled = group.mul(pids_per_group);
    const group_min = group.minimum(extra_pid_groups);
    const local_pid = local_pid_i32.to(.i64);
    return group_scaled.add(group_min).add(local_pid);
}

fn unswizzleMxScaleCdna4(b: *tri.Builder, x: Value, BLOCK_N: i32, MX_SCALE_BLOCK_K: i32) Value {
    const N_PRESHUFFLE_FACTOR: i32 = 32;
    const reshaped = b.reshape(x, &.{
        @divExact(BLOCK_N, N_PRESHUFFLE_FACTOR),
        @divExact(MX_SCALE_BLOCK_K, 8),
        4,
        16,
        2,
        2,
        1,
    });
    const permuted = b.permute(reshaped, &.{ 0, 5, 3, 1, 4, 2, 6 });
    return b.reshape(permuted, &.{ BLOCK_N, MX_SCALE_BLOCK_K });
}

fn clip(b: *tri.Builder, x: Value, limit: Value, comptime clip_lower: bool) Value {
    var res = x.minimum(limit);
    if (clip_lower) {
        res = res.maximum(b.liftAs(0.0, .f32).sub(limit));
    }
    return res;
}

fn swiglu(
    b: *tri.Builder,
    input: Value,
    alpha: Value,
    limit: Value,
    BLOCK_M: i32,
    OUT_BLOCK_N: i32,
    ADD_RESIDUAL: bool,
) Value {
    const split_input = b.split(b.reshape(input, &.{ BLOCK_M, OUT_BLOCK_N, 2 }));
    const gelu = clip(b, split_input[0].to(.f32), limit, false);
    const linear = clip(b, split_input[1].to(.f32), limit, true);
    const exp_arg = alpha.mul(-1.44269504089).mul(gelu);
    const s = gelu.div(b.exp2(exp_arg).add(1.0));
    return if (ADD_RESIDUAL)
        b.fma(s, linear, s)
    else
        s.mul(linear);
}

fn returnIfTritonOrder(b: *tri.Builder, cond: Value) void {
    const current = b.currentBlock();
    const region = current.parentRegion();
    const cont_block = mlir.Block.init(&.{}, &.{});
    region.appendOwnedBlock(cont_block);

    const ret_block = b.exit_block orelse blk: {
        const rb = mlir.Block.init(&.{}, &.{});
        const ret_operands: [0]*const mlir.Value = .{};
        _ = ttir.return_(b.ctx, &ret_operands, b.loc()).appendTo(rb);
        b.exit_block = rb;
        break :blk rb;
    };

    _ = cf.cond_br(
        b.ctx,
        cond.inner,
        ret_block,
        &.{},
        cont_block,
        &.{},
        null,
        b.loc(),
    ).appendTo(current);

    if (b.block_stack.items.len == 0) {
        b.pushBlock(cont_block);
    } else {
        b.block_stack.items[b.block_stack.items.len - 1] = cont_block;
    }
}

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_moe_gemm_a16w4",
    .inputs = &.{
        "stride_y_k",
        "stride_y_m",
        "stride_y_n",
        "X",
        "stride_x_m",
        "stride_x_k",
        "W",
        "stride_w_e",
        "stride_w_k",
        "stride_w_n",
        "WMxScale",
        "stride_w_mx_e",
        "stride_w_mx_k",
        "stride_w_mx_n",
        "B",
        "stride_b_e",
        "Gammas",
        "N",
        "K",
        "GatherIndx",
        "ExptHist",
        "ExptOffs",
        "ExptOffsSum",
        "ExptData",
        "grid_m",
        "grid_n",
        "alpha",
        "limit",
    },
    .outputs = &.{"Y"},
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .stride_y_k = .{ .ptr = .i64 },
        .stride_y_m = .{ .ptr = .i64 },
        .stride_y_n = .{ .ptr = .i64 },
        .X = .{ .ptr = cfg.x_dtype },
        .stride_x_m = .{ .ptr = .i64 },
        .stride_x_k = .{ .ptr = .i64 },
        .W = .{ .ptr = cfg.w_dtype },
        .stride_w_e = .{ .ptr = .i64 },
        .stride_w_k = .{ .ptr = .i64 },
        .stride_w_n = .{ .ptr = .i64 },
        .WMxScale = .{ .ptr = cfg.w_mx_scale_dtype },
        .stride_w_mx_e = .{ .ptr = .i64 },
        .stride_w_mx_k = .{ .ptr = .i64 },
        .stride_w_mx_n = .{ .ptr = .i64 },
        .B = .{ .ptr = cfg.b_dtype },
        .stride_b_e = .{ .ptr = .i64 },
        .Gammas = .{ .ptr = cfg.gammas_dtype },
        .N = .{ .ptr = .i64 },
        .K = .{ .ptr = .i64 },
        .GatherIndx = .{ .ptr = .i32 },
        .ExptHist = .{ .ptr = .i32 },
        .ExptOffs = .{ .ptr = .i32 },
        .ExptOffsSum = .{ .ptr = .i32 },
        .ExptData = .{ .ptr = .i32 },
        .grid_m = .{ .ptr = .i64 },
        .grid_n = .{ .ptr = .i64 },
        .alpha = .{ .ptr = .f32 },
        .limit = .{ .ptr = .f32 },
        .Y = .{ .ptr = cfg.y_dtype },
    });

    const stride_y_k = b.load(a.stride_y_k);
    const stride_y_m = b.load(a.stride_y_m);
    const stride_y_n = b.load(a.stride_y_n);
    const stride_x_m = b.load(a.stride_x_m);
    const stride_x_k = b.load(a.stride_x_k);
    const stride_w_e = b.load(a.stride_w_e);
    const stride_w_k = b.load(a.stride_w_k);
    const stride_w_n = b.load(a.stride_w_n);
    const stride_w_mx_e = b.load(a.stride_w_mx_e);
    const stride_w_mx_k = b.load(a.stride_w_mx_k);
    const stride_w_mx_n = b.load(a.stride_w_mx_n);
    const stride_b_e = if (cfg.HAS_B) b.load(a.stride_b_e) else undefined;
    const N = b.load(a.N);
    const K = b.load(a.K);
    const grid_m = b.load(a.grid_m);
    const grid_n = b.load(a.grid_n);
    const alpha = b.load(a.alpha);
    const limit = b.load(a.limit);

    const zero_i64 = b.liftAs(0, .i64);
    const minus_one_i32 = b.liftAs(-1, .i32);
    const mask_expt_id = b.liftAs(0x0000FFFF, .i32);
    const block_shift = b.liftAs(16, .i32);

    b.assume(stride_y_k.ge(zero_i64));
    b.assume(stride_y_m.ge(zero_i64));
    b.assume(stride_y_n.ge(zero_i64));
    b.assume(stride_x_m.ge(zero_i64));
    b.assume(stride_x_k.ge(zero_i64));
    b.assume(stride_w_e.ge(zero_i64));
    b.assume(stride_w_k.ge(zero_i64));
    b.assume(stride_w_n.ge(zero_i64));
    b.assume(stride_w_mx_e.ge(zero_i64));
    b.assume(stride_w_mx_k.ge(zero_i64));
    b.assume(stride_w_mx_n.ge(zero_i64));
    if (cfg.HAS_B) {
        b.assume(stride_b_e.ge(zero_i64));
    }
    b.assume(grid_m.ge(zero_i64));
    b.assume(grid_n.ge(zero_i64));

    const MX_PACK_DIVISOR: i32 = 32;
    const OUT_BLOCK_N: i32 = @divExact(cfg.BLOCK_N, cfg.ACTIVATION_REDUCTION_N);
    const yN = N.div(cfg.ACTIVATION_REDUCTION_N);

    const pid = b.programId(.x);
    const unpadded_m = if (cfg.HAS_EXPT_OFFS_SUM and cfg.XCD_SWIZZLE > 1) blk: {
        const padding_m = grid_m.sub(b.load(a.ExptOffsSum));
        const value = grid_m.sub(padding_m);
        b.assume(value.ge(zero_i64));
        const total_actual_tiles = value.mul(grid_n).mul(cfg.SPLIT_K);
        returnIfTritonOrder(b, padding_m.gt(zero_i64).bitAnd(pid.to(.i64).ge(total_actual_tiles)));
        break :blk value;
    } else grid_m;
    if (!(cfg.HAS_EXPT_OFFS_SUM and cfg.XCD_SWIZZLE > 1)) {
        b.assume(unpadded_m.ge(zero_i64));
    }
    const total_actual_tiles = unpadded_m.mul(grid_n).mul(cfg.SPLIT_K);

    // swizzle program ids
    const pid_emnk = if (cfg.XCD_SWIZZLE != 1)
        xcdSwizzlePidI32(pid, total_actual_tiles, cfg.XCD_SWIZZLE)
    else
        pid.to(.i64);
    const pid_mnk = pid_emnk.rem(total_actual_tiles);
    var pid_k = if (cfg.SPLIT_K == 1)
        zero_i64
    else
        pid_mnk.rem(cfg.SPLIT_K);
    const pid_mn = if (cfg.SPLIT_K == 1)
        pid_mnk
    else
        pid_mnk.div(cfg.SPLIT_K);
    const pg = pidGrid(pid_mn, unpadded_m, grid_n, cfg.GROUP_M);
    const pid_m = pg[0];
    var pid_n = pg[1];
    // For split-k, advance to the output k slice
    var Y = a.Y;
    if (cfg.SPLIT_K > 1) {
        Y = Y.addPtr(pid_k.to(if (cfg.UPCAST_INDICES) .i64 else .i32).mul(stride_y_k));
    }
    // unpack expert data
    const expt_data = b.load(a.ExptData.addPtr(pid_m));
    returnIfTritonOrder(b, expt_data.eq(minus_one_i32));
    var expt_id = expt_data.bitAnd(mask_expt_id);
    var block_id = b.shrsi(expt_data, block_shift);
    const expert_m = b.load(a.ExptHist.addPtr(expt_id));
    var start_m = b.load(a.ExptOffs.addPtr(expt_id));
    const index_type: tri.DType = if (cfg.UPCAST_INDICES) .i64 else .i32;
    expt_id = expt_id.to(index_type);
    block_id = block_id.to(index_type);
    start_m = start_m.to(index_type);
    pid_n = pid_n.to(index_type);
    pid_k = pid_k.to(index_type);

    // X pointers
    var offs_x_m = block_id.mul(cfg.BLOCK_M).add(b.arange(0, cfg.BLOCK_M, .i32));
    offs_x_m = offs_x_m.rem(expert_m).maximum(offs_x_m.rem(expert_m));
    var X = a.X;
    if (cfg.HAS_GATHER_INDX) {
        const GatherIndx = a.GatherIndx.addPtr(start_m);
        // no needs to bounds-check here because `offs_x_m` wraps around M dim
        offs_x_m = b.load(GatherIndx.addPtr(offs_x_m)).div(cfg.N_EXPTS_ACT);
    } else {
        X = X.addPtr(start_m.mul(stride_x_m));
    }
    const offs_x_k = pid_k.mul(cfg.BLOCK_K).add(b.arange(0, cfg.BLOCK_K, .i32));
    const XPtrRows = X.addPtr(
        b.expandDims(offs_x_m.to(index_type), 1).mul(stride_x_m),
    );
    const XOffsK = b.expandDims(offs_x_k.to(index_type), 0).mul(stride_x_k);
    const XPtrs = b.broadcastTo(XPtrRows, &.{ cfg.BLOCK_M, cfg.BLOCK_K }).addPtr(b.broadcastTo(XOffsK, &.{ cfg.BLOCK_M, cfg.BLOCK_K }));

    const W_K_DIVISOR: i32 = 2;
    const W_N_DIVISOR: i32 = 1;
    const PACKED_BLOCK_K_W: i32 = @divExact(cfg.BLOCK_K, W_K_DIVISOR);
    const PACKED_BLOCK_N_W: i32 = @divExact(cfg.BLOCK_N, W_N_DIVISOR);
    const MX_SCALE_BLOCK_K: i32 = @divExact(cfg.BLOCK_K, MX_PACK_DIVISOR);
    const NON_K_PRESHUFFLE_BLOCK_SIZE: i32 = 32;
    const PACKED_MX_BLOCK: i32 = if (cfg.SWIZZLE_MX_SCALE == .cdna4_scale)
        MX_SCALE_BLOCK_K * NON_K_PRESHUFFLE_BLOCK_SIZE
    else
        MX_SCALE_BLOCK_K;
    const SCALE_BLOCK_N: i32 = if (cfg.SWIZZLE_MX_SCALE == .cdna4_scale)
        @divExact(cfg.BLOCK_N, NON_K_PRESHUFFLE_BLOCK_SIZE)
    else
        cfg.BLOCK_N;

    const WMxScale = a.WMxScale.addPtr(expt_id.mul(stride_w_mx_e));
    var offs_w_n_scale = pid_n.mul(SCALE_BLOCK_N).add(b.arange(0, SCALE_BLOCK_N, .i32)).rem(N);
    offs_w_n_scale = offs_w_n_scale.maximum(offs_w_n_scale);
    // K dimension must be the last dimension for the scales
    const offs_w_k_scale = pid_k.mul(PACKED_MX_BLOCK).add(b.arange(0, PACKED_MX_BLOCK, .i32));
    const WMxScalePtrKs = WMxScale.addPtr(
        b.expandDims(offs_w_k_scale.to(index_type), 0).mul(stride_w_mx_k),
    );
    const WMxScaleOffsN = b.expandDims(offs_w_n_scale.to(index_type), 1).mul(stride_w_mx_n);
    const WMxScalePtrs = b.broadcastTo(WMxScalePtrKs, &.{ SCALE_BLOCK_N, PACKED_MX_BLOCK }).addPtr(b.broadcastTo(WMxScaleOffsN, &.{ SCALE_BLOCK_N, PACKED_MX_BLOCK }));

    // W pointers
    var offs_w_n = pid_n.mul(PACKED_BLOCK_N_W).add(b.arange(0, PACKED_BLOCK_N_W, .i32));
    offs_w_n = offs_w_n.rem(N.div(W_N_DIVISOR)).maximum(offs_w_n.rem(N.div(W_N_DIVISOR)));
    const offs_w_k = b.arange(0, PACKED_BLOCK_K_W, .i32).add(pid_k.mul(PACKED_BLOCK_K_W));
    const W = a.W.addPtr(expt_id.mul(stride_w_e));
    const WPtrs = W.addPtr(
        b.expandDims(offs_w_k.to(index_type), 1).mul(stride_w_k)
            .add(b.expandDims(offs_w_n.to(index_type), 0).mul(stride_w_n)),
    );

    var num_k_iter = K.cdiv(cfg.BLOCK_K * cfg.SPLIT_K);
    if (!cfg.EVEN_K) {
        num_k_iter = num_k_iter.sub(1);
    }

    // compute output
    const acc = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_N }, .f32);
    var loop = b.openFor(@as(i64, 0), num_k_iter, @as(i64, 1), .{ XPtrs, WMxScalePtrs, WPtrs, acc });
    {
        const x_ptrs = loop.carried[0];
        const w_mx_scale_ptrs = loop.carried[1];
        const w_ptrs = loop.carried[2];
        const acc_in = loop.carried[3];

        const x = b.load(x_ptrs);
        const w = b.loadOpts(w_ptrs, .{ .cache_modifier = cacheModifier(cfg) });

        const w_scales_loaded = b.loadOpts(w_mx_scale_ptrs, .{ .cache_modifier = cacheModifier(cfg) });
        const w_scales = if (cfg.SWIZZLE_MX_SCALE == .cdna4_scale)
            unswizzleMxScaleCdna4(b, w_scales_loaded, cfg.BLOCK_N, MX_SCALE_BLOCK_K)
        else
            w_scales_loaded;

        const acc_out = b.dotScaledOpts(x, w, acc_in, null, w_scales, .bf16, .e2m1, .{ .fast_math = true });

        const new_w_mx_scale_ptrs = w_mx_scale_ptrs.addPtr(stride_w_mx_k.mul(PACKED_MX_BLOCK * cfg.SPLIT_K));
        const new_x_ptrs = x_ptrs.addPtr(stride_x_k.mul(cfg.BLOCK_K * cfg.SPLIT_K));
        const new_w_ptrs = w_ptrs.addPtr(stride_w_k.mul(PACKED_BLOCK_K_W * cfg.SPLIT_K));

        loop.yield(.{ new_x_ptrs, new_w_mx_scale_ptrs, new_w_ptrs, acc_out });
    }
    var acc_out = loop.results[3];

    if (!cfg.EVEN_K) {
        const x_ptrs = loop.results[0];
        const w_mx_scale_ptrs = loop.results[1];
        const w_ptrs = loop.results[2];

        const mask_x_k = offs_x_k.lt(cfg.MASK_K_LIMIT);
        const mask_w_k = offs_w_k.lt(@divTrunc(cfg.MASK_K_LIMIT, W_K_DIVISOR));
        const mask_w_k_scale = offs_w_k_scale.mul(MX_PACK_DIVISOR).lt(cfg.MASK_K_LIMIT);

        const x = b.loadOpts(x_ptrs, .{
            .mask = b.broadcastTo(b.expandDims(mask_x_k, 0), &.{ cfg.BLOCK_M, cfg.BLOCK_K }),
            .other = b.zeros(&.{ cfg.BLOCK_M, cfg.BLOCK_K }, cfg.x_dtype),
        });
        const w = b.loadOpts(w_ptrs, .{
            .mask = b.broadcastTo(b.expandDims(mask_w_k, 1), &.{ PACKED_BLOCK_K_W, cfg.BLOCK_N }),
            .other = b.zeros(&.{ PACKED_BLOCK_K_W, cfg.BLOCK_N }, cfg.w_dtype),
            .cache_modifier = cacheModifier(cfg),
        });

        const w_scales = if (cfg.SWIZZLE_MX_SCALE == .cdna4_scale) blk: {
            const loaded = b.loadOpts(w_mx_scale_ptrs, .{ .cache_modifier = cacheModifier(cfg) });
            break :blk unswizzleMxScaleCdna4(b, loaded, cfg.BLOCK_N, MX_SCALE_BLOCK_K);
        } else b.loadOpts(w_mx_scale_ptrs, .{
            .mask = b.broadcastTo(b.expandDims(mask_w_k_scale, 0), &.{ cfg.BLOCK_N, MX_SCALE_BLOCK_K }),
        });

        acc_out = b.dotScaledOpts(x, w, acc_out, null, w_scales, .bf16, .e2m1, .{ .fast_math = true });
    }

    const offs_m = b.arange(0, cfg.BLOCK_M, .i32).add(block_id.mul(cfg.BLOCK_M));
    var offs_y_n = pid_n.mul(cfg.BLOCK_N).add(b.arange(0, cfg.BLOCK_N, .i32));
    const mask_m = offs_m.lt(expert_m);
    var mask_n = offs_y_n.lt(N);
    if (cfg.HAS_B) {
        const BBase = a.B.addPtr(expt_id.mul(stride_b_e));
        const BPtrs = BBase.addPtr(offs_y_n.to(index_type));
        const bias = if (cfg.SPLIT_K == 1) b.loadOpts(BPtrs, .{
            .mask = mask_n,
            .other = b.zeros(&.{cfg.BLOCK_N}, cfg.b_dtype),
            .cache_modifier = cacheModifier(cfg),
        }).to(.f32) else blk: {
            var if_bias = b.openIfElse(pid_k.eq(0), .{b.tensorTy(&.{cfg.BLOCK_N}, .f32)});
            {
                const loaded = b.loadOpts(BPtrs, .{
                    .mask = mask_n,
                    .other = b.zeros(&.{cfg.BLOCK_N}, cfg.b_dtype),
                    .cache_modifier = cacheModifier(cfg),
                }).to(.f32);
                if_bias.yieldThen(.{loaded});
            }
            {
                if_bias.yieldElse(.{b.zeros(&.{cfg.BLOCK_N}, .f32)});
            }
            break :blk if_bias.results[0];
        };
        acc_out = acc_out.add(b.expandDims(bias, 0));
    }

    var out = acc_out;
    if (cfg.APPLY_SWIGLU and cfg.SPLIT_K == 1) {
        out = swiglu(b, acc_out, alpha, limit, cfg.BLOCK_M, OUT_BLOCK_N, cfg.SWIGLU_ADD_RESIDUAL);
        offs_y_n = pid_n.mul(OUT_BLOCK_N).add(b.arange(0, OUT_BLOCK_N, .i32));
        mask_n = offs_y_n.lt(yN);
    } else {
        out = out;
    }
    if (cfg.HAS_GAMMAS) {
        const gammas = b.loadOpts(a.Gammas.addPtr(start_m.add(offs_m.to(index_type))), .{
            .mask = mask_m,
            .other = b.zeros(&.{cfg.BLOCK_M}, cfg.gammas_dtype),
        }).to(.f32);
        out = out.mul(b.expandDims(gammas, 1));
    }

    // write-back
    Y = Y.addPtr(start_m.mul(stride_y_m));
    const offs_y_m = offs_m;
    const STORE_BLOCK_N: i32 = if (cfg.APPLY_SWIGLU and cfg.SPLIT_K == 1) OUT_BLOCK_N else cfg.BLOCK_N;
    const YPtrRows = Y.addPtr(
        b.expandDims(offs_y_m.to(index_type), 1).mul(stride_y_m),
    );
    const YOffsN = b.expandDims(offs_y_n.to(index_type), 0).mul(stride_y_n);
    const YPtrs = b.broadcastTo(YPtrRows, &.{ cfg.BLOCK_M, STORE_BLOCK_N }).addPtr(b.broadcastTo(YOffsN, &.{ cfg.BLOCK_M, STORE_BLOCK_N }));
    const mask = b.expandDims(mask_m, 1).bitAnd(b.expandDims(mask_n, 0));
    b.storeOpts(YPtrs, out.to(cfg.y_dtype), .{ .mask = mask });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "basic", .cfg = .{} },
};

// =============================================================================
// XLA-pipeline driver. Shapes are synthetic; XLA lowers the custom call but never
// launches it. Keep `Y` as the final function argument to match the TTIR ABI.
// =============================================================================

const M: i64 = 128;
const N_: i64 = 128;
const K_: i64 = 128;
const E: i64 = 4;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(blob: [:0]const u8) void {
    active_ttir = blob;
}

pub fn forward(
    stride_y_k: Tensor,
    stride_y_m: Tensor,
    stride_y_n: Tensor,
    X: Tensor,
    stride_x_m: Tensor,
    stride_x_k: Tensor,
    W: Tensor,
    stride_w_e: Tensor,
    stride_w_k: Tensor,
    stride_w_n: Tensor,
    WMxScale: Tensor,
    stride_w_mx_e: Tensor,
    stride_w_mx_k: Tensor,
    stride_w_mx_n: Tensor,
    B: Tensor,
    stride_b_e: Tensor,
    Gammas: Tensor,
    N: Tensor,
    K: Tensor,
    GatherIndx: Tensor,
    ExptHist: Tensor,
    ExptOffs: Tensor,
    ExptOffsSum: Tensor,
    ExptData: Tensor,
    grid_m: Tensor,
    grid_n: Tensor,
    alpha: Tensor,
    limit: Tensor,
    _: Tensor,
) Tensor {
    return ops.triton(
        .{
            stride_y_k,
            stride_y_m,
            stride_y_n,
            X,
            stride_x_m,
            stride_x_k,
            W,
            stride_w_e,
            stride_w_k,
            stride_w_n,
            WMxScale,
            stride_w_mx_e,
            stride_w_mx_k,
            stride_w_mx_n,
            B,
            stride_b_e,
            Gammas,
            N,
            K,
            GatherIndx,
            ExptHist,
            ExptOffs,
            ExptOffsSum,
            ExptData,
            grid_m,
            grid_n,
            alpha,
            limit,
        },
        .{Shape.init(.{ M, N_ }, .bf16)},
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ 16, 1, 1 },
            .num_warps = 4,
            .num_stages = 2,
        },
    )[0];
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    const p = struct {
        fn t() Tensor {
            return Tensor.init(.{1}, .i64);
        }
    }.t;

    return .{
        p(), // stride_y_k
        p(), // stride_y_m
        p(), // stride_y_n
        Tensor.init(.{ M, K_ }, .bf16), // X
        p(), // stride_x_m
        p(), // stride_x_k
        Tensor.init(.{ E, K_ / 2, N_ }, .i8), // W
        p(), // stride_w_e
        p(), // stride_w_k
        p(), // stride_w_n
        Tensor.init(.{ E, N_, K_ / 32 }, .i8), // WMxScale
        p(), // stride_w_mx_e
        p(), // stride_w_mx_k
        p(), // stride_w_mx_n
        Tensor.init(.{ E, N_ }, .bf16), // B
        p(), // stride_b_e
        Tensor.init(.{M}, .f32), // Gammas
        p(), // N
        p(), // K
        Tensor.init(.{M}, .i32), // GatherIndx
        Tensor.init(.{E}, .i32), // ExptHist
        Tensor.init(.{E}, .i32), // ExptOffs
        Tensor.init(.{1}, .i32), // ExptOffsSum
        Tensor.init(.{M}, .i32), // ExptData
        p(), // grid_m
        p(), // grid_n
        Tensor.init(.{1}, .f32), // alpha
        Tensor.init(.{1}, .f32), // limit
        Tensor.init(.{ M, N_ }, .bf16), // Y placeholder
    };
}

test "emit TTIR basic sweep" {
    const ttir_blob = try Kernel.emit(std.testing.allocator, SWEEPS[0].cfg);
    defer std.testing.allocator.free(ttir_blob);

    try std.testing.expect(std.mem.indexOf(u8, ttir_blob, "_moe_gemm_a16w4") != null);
}
