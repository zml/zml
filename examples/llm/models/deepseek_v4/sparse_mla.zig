//! Harness registration for the XPU MLA sparse prefill kernel.

const std = @import("std");

const harness = @import("harness");
const zml = @import("zml");
const ops = zml.ops;
const tri = zml.kernel.triton;
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const Value = tri.Value;

pub const Cfg = struct {
    q_dtype: tri.DType = .bf16,
    k_dtype: tri.DType = .bf16,
    v_dtype: tri.DType = .bf16,
    out_dtype: tri.DType = .bf16,

    kv_group_num: i32 = 16,
    index_topk: i32 = 16,
    BLOCK_H: i32 = 16,
    BLOCK_M: i32 = 32,
    BLOCK_N: i32 = 16,
    BLOCK_DV: i32 = 512,
    BLOCK_DMODEL: i32 = 512,
    BLOCK_DPE: i32 = 64,
    LOGE2: f64 = 0.6931471805599453,
};

pub const Kernel = tri.Kernel(Cfg, .{
    .name = "_bf16_mla_sparse_kernel",
    .inputs = &.{
        "q_buffer",
        "k_buffer",
        "v_buffer",
        "indices_ptr",
        "seq_q_ptr",
        "seq_kv_ptr",
        "h_q_ptr",
        "dim_qk_ptr",
        "dim_v_ptr",
        "stride_q_token_ptr",
        "stride_q_head_ptr",
        "stride_k_token_ptr",
        "stride_k_head_ptr",
        "stride_v_token_ptr",
        "stride_v_head_ptr",
        "stride_out_token_ptr",
        "stride_out_head_ptr",
        "stride_lse_ptr",
        "stride_indices_token_ptr",
        "stride_indices_head_ptr",
        "sm_scale_ptr",
    },
    .outputs = &.{ "out", "softmax_lse", "max_logits" },
    .run = run,
});

fn run(b: *tri.Builder, cfg: Cfg) tri.FinishError!void {
    const a = try b.declareArgs(.{
        .q_buffer = .{ .ptr = cfg.q_dtype },
        .k_buffer = .{ .ptr = cfg.k_dtype },
        .v_buffer = .{ .ptr = cfg.v_dtype },
        .indices_ptr = .{ .ptr = .i32 },
        .seq_q_ptr = .{ .ptr = .i64 },
        .seq_kv_ptr = .{ .ptr = .i64 },
        .h_q_ptr = .{ .ptr = .i64 },
        .dim_qk_ptr = .{ .ptr = .i64 },
        .dim_v_ptr = .{ .ptr = .i64 },
        .stride_q_token_ptr = .{ .ptr = .i64 },
        .stride_q_head_ptr = .{ .ptr = .i64 },
        .stride_k_token_ptr = .{ .ptr = .i64 },
        .stride_k_head_ptr = .{ .ptr = .i64 },
        .stride_v_token_ptr = .{ .ptr = .i64 },
        .stride_v_head_ptr = .{ .ptr = .i64 },
        .stride_out_token_ptr = .{ .ptr = .i64 },
        .stride_out_head_ptr = .{ .ptr = .i64 },
        .stride_lse_ptr = .{ .ptr = .i64 },
        .stride_indices_token_ptr = .{ .ptr = .i64 },
        .stride_indices_head_ptr = .{ .ptr = .i64 },
        .sm_scale_ptr = .{ .ptr = .f32 },
        .out_ptr = .{ .ptr = cfg.out_dtype },
        .softmax_lse_ptr = .{ .ptr = .f32 },
        .max_logits_ptr = .{ .ptr = .f32 },
    });

    const seq_kv = b.load(a.seq_kv_ptr);
    const h_q = b.load(a.h_q_ptr);
    const dim_qk = b.load(a.dim_qk_ptr);
    const dim_v = b.load(a.dim_v_ptr);
    const stride_q_token = b.load(a.stride_q_token_ptr);
    const stride_q_head = b.load(a.stride_q_head_ptr);
    const stride_k_token = b.load(a.stride_k_token_ptr);
    const stride_k_head = b.load(a.stride_k_head_ptr);
    const stride_v_token = b.load(a.stride_v_token_ptr);
    const stride_v_head = b.load(a.stride_v_head_ptr);
    const stride_out_token = b.load(a.stride_out_token_ptr);
    const stride_out_head = b.load(a.stride_out_head_ptr);
    const stride_lse = b.load(a.stride_lse_ptr);
    const stride_indices_token = b.load(a.stride_indices_token_ptr);
    const stride_indices_head = b.load(a.stride_indices_head_ptr);
    const sm_scale = b.load(a.sm_scale_ptr);

    const cur_q = b.programId(.x);
    const cur_head_id = b.programId(.y);
    const cur_kv_head_id = cur_head_id.div(@divTrunc(cfg.kv_group_num + cfg.BLOCK_H - 1, cfg.BLOCK_H));

    const VALID_BLOCK_H: i32 = if (cfg.kv_group_num > cfg.BLOCK_H) cfg.BLOCK_H else cfg.kv_group_num;
    const cur_head = cur_head_id.mul(VALID_BLOCK_H).add(b.arange(0, cfg.BLOCK_H, .i32));
    var mask_h = cur_head.lt(cur_head_id.add(1).mul(VALID_BLOCK_H));
    mask_h = mask_h.bitAnd(cur_head.to(.i64).lt(h_q));

    const offs_d = b.arange(0, cfg.BLOCK_DMODEL, .i32);
    const offs_dv = b.arange(0, cfg.BLOCK_DV, .i32);

    const off_q = cur_q.to(.i64).mul(stride_q_token)
        .add(cur_head.expandDims(1).to(.i64).mul(stride_q_head))
        .add(offs_d.expandDims(0).to(.i64));
    const mask_dmodel = offs_d.lt(cfg.BLOCK_DMODEL);
    const mask_q = mask_h.expandDims(1).bitAnd(mask_dmodel.expandDims(0));
    const q = b.loadOpts(a.q_buffer.addPtr(off_q), .{
        .mask = mask_q,
        .other = b.zeros(&.{ cfg.BLOCK_H, cfg.BLOCK_DMODEL }, cfg.q_dtype),
    });

    var offs_dpe: Value = undefined;
    var qpe: Value = undefined;
    if (cfg.BLOCK_DPE > 0) {
        offs_dpe = b.arange(0, cfg.BLOCK_DPE, .i32).add(cfg.BLOCK_DMODEL);
        const off_qpe = cur_q.to(.i64).mul(stride_q_token)
            .add(cur_head.expandDims(1).to(.i64).mul(stride_q_head))
            .add(offs_dpe.expandDims(0).to(.i64));
        // assume dim_qk == BLOCK_DMODEL + BLOCK_DPE
        const mask_dpe = offs_dpe.to(.i64).lt(dim_qk);
        const mask_qpe = mask_h.expandDims(1).bitAnd(mask_dpe.expandDims(0));
        qpe = b.loadOpts(a.q_buffer.addPtr(off_qpe), .{
            .mask = mask_qpe,
            .other = b.zeros(&.{ cfg.BLOCK_H, cfg.BLOCK_DPE }, cfg.q_dtype),
        });
    }

    var e_max = b.zeros(&.{cfg.BLOCK_H}, .f32).sub(std.math.inf(f32));
    var e_sum = b.zeros(&.{cfg.BLOCK_H}, .f32);
    var acc = b.zeros(&.{ cfg.BLOCK_H, cfg.BLOCK_DV }, .f32);

    var start_indice: i32 = 0;
    while (start_indice < cfg.index_topk) : (start_indice += cfg.BLOCK_N) {
        const offs_indice = b.arange(0, cfg.BLOCK_N, .i32).add(start_indice);
        const mask_indice = offs_indice.lt(cfg.index_topk);
        const indices = b.loadOpts(
            a.indices_ptr.addPtr(
                cur_q.to(.i64).mul(stride_indices_token)
                    .add(cur_kv_head_id.to(.i64).mul(stride_indices_head))
                    .add(offs_indice.to(.i64)),
            ),
            .{
                .mask = mask_indice,
                .other = b.full(&.{cfg.BLOCK_N}, -1, .i32),
            },
        );

        const mask_kv = indices.ge(0).bitAnd(indices.to(.i64).lt(seq_kv));
        const mask_kv_d = mask_dmodel;
        const offs_k = indices.expandDims(0).to(.i64).mul(stride_k_token)
            .add(cur_kv_head_id.to(.i64).mul(stride_k_head))
            .add(offs_d.expandDims(1).to(.i64));

        // q_nope @ k_nope
        const mask_k = mask_kv.expandDims(0).bitAnd(mask_kv_d.expandDims(1));
        const k = b.loadOpts(a.k_buffer.addPtr(offs_k), .{
            .mask = mask_k,
            .other = b.zeros(&.{ cfg.BLOCK_DMODEL, cfg.BLOCK_N }, cfg.k_dtype),
        });
        var qk = b.dot(q, k.to(cfg.q_dtype), b.zeros(&.{ cfg.BLOCK_H, cfg.BLOCK_N }, .f32));

        if (cfg.BLOCK_DPE > 0) {
            // q_rope @ k_rope
            const offs_kpe = indices.expandDims(0).to(.i64).mul(stride_k_token)
                .add(cur_kv_head_id.to(.i64).mul(stride_k_head))
                .add(offs_dpe.expandDims(1).to(.i64));
            const mask_k_dpe = offs_dpe.to(.i64).lt(dim_qk);
            const mask_kpe = mask_kv.expandDims(0).bitAnd(mask_k_dpe.expandDims(1));
            const kpe = b.loadOpts(a.k_buffer.addPtr(offs_kpe), .{
                .mask = mask_kpe,
                .other = b.zeros(&.{ cfg.BLOCK_DPE, cfg.BLOCK_N }, cfg.k_dtype),
            });
            qk = b.dot(qpe, kpe.to(cfg.q_dtype), qk);
        }

        // apply scaling
        qk = qk.mul(sm_scale);
        qk = b.where(
            mask_h.expandDims(1).bitAnd(mask_kv.expandDims(0)),
            qk,
            b.full(&.{ cfg.BLOCK_H, cfg.BLOCK_N }, -std.math.inf(f32), .f32),
        );

        // load v
        const mask_v_d = offs_dv.to(.i64).lt(dim_v);
        const offs_v = indices.expandDims(1).to(.i64).mul(stride_v_token)
            .add(cur_kv_head_id.to(.i64).mul(stride_v_head))
            .add(offs_dv.expandDims(0).to(.i64));
        const mask_v = mask_kv.expandDims(1).bitAnd(mask_v_d.expandDims(0));
        const v = b.loadOpts(a.v_buffer.addPtr(offs_v), .{
            .mask = mask_v,
            .other = b.zeros(&.{ cfg.BLOCK_N, cfg.BLOCK_DV }, cfg.v_dtype),
        });

        // online softmax
        const n_e_max = b.maxOpts(qk, .{ .axis = 1 }).maximum(e_max);
        const re_scale = b.exp2(e_max.sub(n_e_max));
        const p = b.exp2(qk.sub(n_e_max.expandDims(1)));
        acc = if (start_indice == 0)
            b.broadcastTo(re_scale.expandDims(1).mul(b.zeros(&.{ cfg.BLOCK_H, 1 }, .f32)), &.{ cfg.BLOCK_H, cfg.BLOCK_DV })
        else
            re_scale.expandDims(1).mul(acc);

        // score @ v
        acc = b.dot(p.to(cfg.v_dtype), v, acc);

        // update global sum and max
        e_sum = e_sum.mul(re_scale).add(b.sumOpts(p, .{ .axis = 1 }));
        e_max = n_e_max;
    }

    // rescaling
    acc = acc.div(e_sum.expandDims(1));

    const max_logits = e_max.mul(b.full(&.{cfg.BLOCK_H}, cfg.LOGE2, .f32));
    // calculate lse
    const lse = max_logits.add(b.log2(e_sum).mul(b.full(&.{cfg.BLOCK_H}, cfg.LOGE2, .f32)));

    // write output
    const offs_o = cur_q.to(.i64).mul(stride_out_token)
        .add(cur_head.expandDims(1).to(.i64).mul(stride_out_head))
        .add(offs_dv.expandDims(0).to(.i64));
    const mask_out_d = offs_dv.to(.i64).lt(dim_v);
    const mask_out = mask_h.expandDims(1).bitAnd(mask_out_d.expandDims(0));
    b.storeOpts(
        a.out_ptr.addPtr(offs_o),
        acc.to(.bf16),
        .{ .mask = mask_out },
    );

    const offs_lse = cur_q.to(.i64).mul(stride_lse).add(cur_head.to(.i64));
    b.storeOpts(a.softmax_lse_ptr.addPtr(offs_lse), lse, .{ .mask = mask_h });
    b.storeOpts(a.max_logits_ptr.addPtr(offs_lse), max_logits, .{ .mask = mask_h });
}

pub const SWEEPS: []const harness.Sweep(Kernel.Config) = &.{
    .{ .label = "basic", .cfg = .{} },
};

const NUM_TOKENS: i64 = 2;
const NUM_HEADS_Q: i64 = 16;
const NUM_HEADS_KV: i64 = 1;
const DIM_QK: i64 = 576;
const DIM_V: i64 = 512;
const INDEX_TOPK: i64 = 16;

threadlocal var active_ttir: [:0]const u8 = "";

pub fn setActiveTtir(ttir: [:0]const u8) void {
    active_ttir = ttir;
}

pub fn forward(
    q_buffer: Tensor,
    k_buffer: Tensor,
    v_buffer: Tensor,
    indices_ptr: Tensor,
    seq_q_ptr: Tensor,
    seq_kv_ptr: Tensor,
    h_q_ptr: Tensor,
    dim_qk_ptr: Tensor,
    dim_v_ptr: Tensor,
    stride_q_token_ptr: Tensor,
    stride_q_head_ptr: Tensor,
    stride_k_token_ptr: Tensor,
    stride_k_head_ptr: Tensor,
    stride_v_token_ptr: Tensor,
    stride_v_head_ptr: Tensor,
    stride_out_token_ptr: Tensor,
    stride_out_head_ptr: Tensor,
    stride_lse_ptr: Tensor,
    stride_indices_token_ptr: Tensor,
    stride_indices_head_ptr: Tensor,
    sm_scale_ptr: Tensor,
    _: Tensor,
    _: Tensor,
    _: Tensor,
) struct { Tensor, Tensor, Tensor } {
    const out = ops.triton(
        .{
            q_buffer,
            k_buffer,
            v_buffer,
            indices_ptr,
            seq_q_ptr,
            seq_kv_ptr,
            h_q_ptr,
            dim_qk_ptr,
            dim_v_ptr,
            stride_q_token_ptr,
            stride_q_head_ptr,
            stride_k_token_ptr,
            stride_k_head_ptr,
            stride_v_token_ptr,
            stride_v_head_ptr,
            stride_out_token_ptr,
            stride_out_head_ptr,
            stride_lse_ptr,
            stride_indices_token_ptr,
            stride_indices_head_ptr,
            sm_scale_ptr,
        },
        .{
            Shape.init(.{ NUM_TOKENS * NUM_HEADS_Q * DIM_V }, .bf16),
            Shape.init(.{ NUM_TOKENS * NUM_HEADS_Q }, .f32),
            Shape.init(.{ NUM_TOKENS * NUM_HEADS_Q }, .f32),
        },
        .{
            .name = Kernel.name,
            .ir = active_ttir,
            .grid = .{ NUM_TOKENS, 1, 1 },
            .num_warps = 4,
            .num_stages = 3,
        },
    );
    return .{ out[0], out[1], out[2] };
}

pub fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
    const p_i64 = struct {
        fn t() Tensor {
            return Tensor.init(.{1}, .i64);
        }
    }.t;
    const p_f32 = struct {
        fn t() Tensor {
            return Tensor.init(.{1}, .f32);
        }
    }.t;

    return .{
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_Q * DIM_QK }, .bf16), // q_buffer
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_KV * DIM_QK }, .bf16), // k_buffer
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_KV * DIM_V }, .bf16), // v_buffer
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_KV * INDEX_TOPK }, .i32), // indices_ptr
        p_i64(), // seq_q_ptr
        p_i64(), // seq_kv_ptr
        p_i64(), // h_q_ptr
        p_i64(), // dim_qk_ptr
        p_i64(), // dim_v_ptr
        p_i64(), // stride_q_token_ptr
        p_i64(), // stride_q_head_ptr
        p_i64(), // stride_k_token_ptr
        p_i64(), // stride_k_head_ptr
        p_i64(), // stride_v_token_ptr
        p_i64(), // stride_v_head_ptr
        p_i64(), // stride_out_token_ptr
        p_i64(), // stride_out_head_ptr
        p_i64(), // stride_lse_ptr
        p_i64(), // stride_indices_token_ptr
        p_i64(), // stride_indices_head_ptr
        p_f32(), // sm_scale_ptr
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_Q * DIM_V }, .bf16), // out
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_Q }, .f32), // softmax_lse
        Tensor.init(.{ NUM_TOKENS * NUM_HEADS_Q }, .f32), // max_logits
    };
}

test "emit TTIR sweeps" {
    inline for (SWEEPS) |sweep| {
        const ttir_blob = try Kernel.emit(std.testing.allocator, sweep.cfg);
        defer std.testing.allocator.free(ttir_blob);

        try std.testing.expect(std.mem.indexOf(u8, ttir_blob, "_bf16_mla_sparse_kernel") != null);
    }
}
