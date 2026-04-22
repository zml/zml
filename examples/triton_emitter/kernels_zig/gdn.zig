//! Zig port of `kernels_py/gdn.py:fused_recurrent_gated_delta_rule_fwd_kernel_ptr`.
//!
//! Pinned to the constexpr config used by `monorepo/llmd/models/qwen3_5.zig`
//! (`TritonGatedDeltaNetHelper.forward`):
//!   USE_G=true, USE_GK=false, USE_GV=false, USE_QK_L2NORM_IN_KERNEL=true,
//!   IS_BETA_HEADWISE=true, USE_INITIAL_STATE=true, STORE_FINAL_STATE=true,
//!   USE_EXP2=false, TRANSPOSE_STATE=false, IS_VARLEN=true.
//!
//! The Python wrapper kernel passes `gk` and `gv` as `tl.constexpr=None`, so
//! the gk/gv pointer args do not appear in the TTIR signature. We mirror that
//! by simply not declaring them.

const std = @import("std");

const tri = @import("zml/triton");
const zml = @import("zml");

pub const FusedRecurrentGatedDeltaRule = zml.Kernel(.{
    .name = "fused_recurrent_gated_delta_rule_fwd_kernel_ptr",
    .config = struct {
        // Pointer dtypes — match the monorepo call (q/k/v/beta/o = bf16, g/h0/ht = f32).
        q_dtype: tri.DType = .bf16,
        k_dtype: tri.DType = .bf16,
        v_dtype: tri.DType = .bf16,
        g_dtype: tri.DType = .f32,
        beta_dtype: tri.DType = .bf16,
        h_dtype: tri.DType = .f32,
        o_dtype: tri.DType = .bf16,
        // tl.constexpr fields (Python's constexpr ints / float / bools).
        scale: f32,
        T: i32,
        H: i32,
        HV: i32,
        K: i32,
        V: i32,
        BK: i32,
        BV: i32,
        USE_G: bool,
        USE_GK: bool,
        USE_GV: bool,
        USE_QK_L2NORM_IN_KERNEL: bool,
        IS_BETA_HEADWISE: bool,
        USE_INITIAL_STATE: bool,
        STORE_FINAL_STATE: bool,
        USE_EXP2: bool,
        TRANSPOSE_STATE: bool,
        IS_VARLEN: bool,
    },
}, struct {
    pub fn run(b: *tri.Builder, cfg: anytype) !void {
        // This port is pinned to the monorepo config. Other combos change the
        // loop's iter_args arity (and Python's emit order); they're a separate
        // exercise. Asserts run at TTIR-emit time, before any IR is built.
        std.debug.assert(cfg.USE_G);
        std.debug.assert(!cfg.USE_GK);
        std.debug.assert(!cfg.USE_GV);
        std.debug.assert(cfg.USE_QK_L2NORM_IN_KERNEL);
        std.debug.assert(cfg.IS_BETA_HEADWISE);
        std.debug.assert(cfg.USE_INITIAL_STATE);
        std.debug.assert(cfg.STORE_FINAL_STATE);
        std.debug.assert(!cfg.USE_EXP2);
        std.debug.assert(!cfg.TRANSPOSE_STATE);
        std.debug.assert(cfg.IS_VARLEN);

        const a = try b.declareArgs(.{
            .q_ptr = .{ .ptr = cfg.q_dtype },
            .k_ptr = .{ .ptr = cfg.k_dtype },
            .v_ptr = .{ .ptr = cfg.v_dtype },
            .g_ptr = .{ .ptr = cfg.g_dtype },
            .beta_ptr = .{ .ptr = cfg.beta_dtype },
            .h0_ptr = .{ .ptr = cfg.h_dtype },
            .cu_seqlens_ptr = .{ .ptr = .i32 },
            .o_ptr = .{ .ptr = cfg.o_dtype },
            .ht_ptr = .{ .ptr = cfg.h_dtype },
        });

        const H = cfg.H;
        const HV = cfg.HV;
        const K = cfg.K;
        const V = cfg.V;
        const BK = cfg.BK;
        const BV = cfg.BV;
        const HV_per_H: i32 = @divTrunc(HV, H);
        const BK_i64: i64 = @intCast(BK);
        const BV_i64: i64 = @intCast(BV);

        // i_v, i_nh = tl.program_id(0), tl.program_id(1)
        const i_v = b.programId(.x);
        const i_nh = b.programId(.y);
        // i_n, i_hv = i_nh // HV, i_nh % HV
        const i_n = i_nh.div(HV);
        const i_hv = i_nh.rem(HV);
        // i_h = i_hv // (HV // H)
        const i_h = i_hv.div(HV_per_H);

        // if IS_VARLEN:
        //     bos, eos = (
        //         tl.load(cu_seqlens + i_n).to(tl.int64),
        //         tl.load(cu_seqlens + i_n + 1).to(tl.int64),
        //     )
        //     T = eos - bos
        const bos = b.load(a.cu_seqlens_ptr.addPtr(i_n)).to(.i64);
        const eos = b.load(a.cu_seqlens_ptr.addPtr(i_n).addPtr(1)).to(.i64);
        const T = eos.sub(bos);

        // o_k = tl.arange(0, BK)
        const o_k = b.arange(0, BK, .i32);
        // o_v = i_v * BV + tl.arange(0, BV)
        const o_v = i_v.mul(BV).add(b.arange(0, BV, .i32));

        // p_q = q + (bos * H + i_h) * K + o_k
        const p_q_init = a.q_ptr.addPtr(bos.mul(H).add(i_h).mul(K)).addPtr(o_k);
        // p_k = k + (bos * H + i_h) * K + o_k
        const p_k_init = a.k_ptr.addPtr(bos.mul(H).add(i_h).mul(K)).addPtr(o_k);
        // p_v = v + (bos * HV + i_hv) * V + o_v
        const p_v_init = a.v_ptr.addPtr(bos.mul(HV).add(i_hv).mul(V)).addPtr(o_v);
        // if USE_G: p_g = g + bos * HV + i_hv
        const p_g_init = a.g_ptr.addPtr(bos.mul(HV)).addPtr(i_hv);
        // if IS_BETA_HEADWISE: p_beta = beta + bos * HV + i_hv
        const p_beta_init = a.beta_ptr.addPtr(bos.mul(HV)).addPtr(i_hv);
        // p_o = o + (bos * HV + i_hv) * V + o_v
        const p_o_init = a.o_ptr.addPtr(bos.mul(HV).add(i_hv).mul(V)).addPtr(o_v);

        // mask_k = o_k < K
        const mask_k = o_k.lt(K);
        // mask_v = o_v < V
        const mask_v = o_v.lt(V);
        // mask_h = mask_k[:, None] & mask_v[None, :]   (TRANSPOSE_STATE=false)
        const mask_h = b.expandDims(mask_k, 1).bitAnd(b.expandDims(mask_v, 0));

        // b_h = tl.zeros([BK, BV], dtype=tl.float32)   (TRANSPOSE_STATE=false)
        var b_h_init = b.zeros(&.{ BK_i64, BV_i64 }, .f32);
        // if USE_INITIAL_STATE:
        //     p_h0 = h0 + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        //     b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
        // Three left-to-right addptrs (Python's source order). The BKx1 → BKxBV
        // step needs explicit broadcasts on both sides since `addPtr` doesn't
        // auto-broadcast tensor-of-ptrs against a tensor offset.
        const p_h0_col = a.h0_ptr.addPtr(i_nh.mul(K).mul(V)).addPtr(b.expandDims(o_k, 1).mul(V));
        // Emit expand_dims before broadcast(p_h0_col) so the IR op-order
        // matches Python (expand_dims → broadcast(ptr) → broadcast(dim)).
        const o_v_2d = b.expandDims(o_v, 0);
        const p_h0 = b.broadcastTo(p_h0_col, &.{ BK_i64, BV_i64 })
            .addPtr(b.broadcastTo(o_v_2d, &.{ BK_i64, BV_i64 }));
        const h0_loaded = b.loadOpts(p_h0, .{
            .mask = mask_h,
            .other = b.zeros(&.{ BK_i64, BV_i64 }, cfg.h_dtype),
        }).to(.f32);
        b_h_init = b_h_init.add(h0_loaded);

        // for _ in tl.range(0, T):
        //   iter_args (declaration order): p_q, p_k, p_v, p_g, p_beta, p_o, b_h
        var loop = b.openFor(@as(i64, 0), T, @as(i64, 1), .{
            p_q_init, p_k_init, p_v_init, p_g_init, p_beta_init, p_o_init, b_h_init,
        });
        {
            const p_q = loop.carried[0];
            const p_k = loop.carried[1];
            const p_v = loop.carried[2];
            const p_g = loop.carried[3];
            const p_beta = loop.carried[4];
            const p_o = loop.carried[5];
            var b_h = loop.carried[6];

            // b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
            var b_q = b.loadOpts(p_q, .{
                .mask = mask_k,
                .other = b.zeros(&.{BK_i64}, cfg.q_dtype),
            }).to(.f32);
            // b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
            var b_k = b.loadOpts(p_k, .{
                .mask = mask_k,
                .other = b.zeros(&.{BK_i64}, cfg.k_dtype),
            }).to(.f32);
            // b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
            var b_v = b.loadOpts(p_v, .{
                .mask = mask_v,
                .other = b.zeros(&.{BV_i64}, cfg.v_dtype),
            }).to(.f32);
            // if USE_QK_L2NORM_IN_KERNEL:
            //     b_q = b_q / tl.sqrt(tl.sum(b_q * b_q) + 1e-6)
            //     b_k = b_k / tl.sqrt(tl.sum(b_k * b_k) + 1e-6)
            b_q = b_q.div(b.sqrt(b.sum(b_q.mul(b_q)).add(1e-6)));
            b_k = b_k.div(b.sqrt(b.sum(b_k.mul(b_k)).add(1e-6)));
            // b_q = b_q * scale
            b_q = b_q.mul(cfg.scale);
            // if IS_BETA_HEADWISE: b_beta = tl.load(p_beta).to(tl.float32)
            const b_beta = b.load(p_beta).to(.f32);

            // if USE_G:
            //     b_g = tl.load(p_g).to(tl.float32)
            //     # (USE_EXP2=false)
            //     b_h *= exp(b_g)
            const b_g = b.load(p_g).to(.f32);
            b_h = b_h.mul(b.exp(b_g));

            // (TRANSPOSE_STATE=false)
            //     b_v = b_beta * (b_v - tl.sum(b_h * b_k[:, None], 0))
            //     b_h += b_k[:, None] * b_v
            //     b_o = tl.sum(b_h * b_q[:, None], 0)
            const sum_hk = b.sumOpts(b_h.mul(b.expandDims(b_k, 1)), .{ .axis = 0 });
            b_v = b_beta.mul(b_v.sub(sum_hk));
            b_h = b_h.add(b.expandDims(b_k, 1).mul(b_v));
            const b_o = b.sumOpts(b_h.mul(b.expandDims(b_q, 1)), .{ .axis = 0 });

            // tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
            b.storeOpts(p_o, b_o.to(cfg.o_dtype), .{ .mask = mask_v });

            // p_q += H * K
            // p_k += H * K
            // p_v += HV * V
            // if USE_G: p_g += HV
            // p_beta += HV * (1 if IS_BETA_HEADWISE else V)   → HV
            // p_o += HV * V
            const new_p_q = p_q.addPtr(b.splat(H * K, &.{BK_i64}));
            const new_p_k = p_k.addPtr(b.splat(H * K, &.{BK_i64}));
            const new_p_v = p_v.addPtr(b.splat(HV * V, &.{BV_i64}));
            const new_p_g = p_g.addPtr(HV);
            const new_p_beta = p_beta.addPtr(HV);
            const new_p_o = p_o.addPtr(b.splat(HV * V, &.{BV_i64}));

            loop.yield(.{ new_p_q, new_p_k, new_p_v, new_p_g, new_p_beta, new_p_o, b_h });
        }
        const b_h_final = loop.results[6];

        // if STORE_FINAL_STATE:
        //     p_ht = ht + i_nh * K * V + o_k[:, None] * V + o_v[None, :]
        //     tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), mask=mask_h)
        const p_ht_col = a.ht_ptr.addPtr(i_nh.mul(K).mul(V)).addPtr(b.expandDims(o_k, 1).mul(V));
        const p_ht = b.broadcastTo(p_ht_col, &.{ BK_i64, BV_i64 })
            .addPtr(b.broadcastTo(b.expandDims(o_v, 0), &.{ BK_i64, BV_i64 }));
        b.storeOpts(p_ht, b_h_final.to(cfg.h_dtype), .{ .mask = mask_h });
    }
});
