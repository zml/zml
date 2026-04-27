//! Push every TTIR file under `--in-dir` through ZML's real compile path so
//! XLA's Triton pipeline (`MakeTTIR → MakeTTGIR → MakeLLIR`, see
//! `xla/backends/gpu/codegen/triton/compilation_pipeline_cuda.cc`) runs end
//! to end. The flags `xla_dump_to=<out-dir>` + `xla_dump_emitter_re=triton-to-llvm`
//! make XLA write one IR snapshot per pass into
//! `<out-dir>/<program>.<kernel>.triton-to-llvm.txt`.
//!
//! Use the same binary on both `py_ir/` and `zig_ir/` to get apples-to-apples
//! post-pipeline IR that `extract_xla_dump.py` can slice into per-stage files.
//!
//! Requires CUDA or ROCm (XLA's Triton pipeline is GPU-only).

const std = @import("std");

const mlir = @import("mlir");
const zml = @import("zml");
const Tensor = zml.Tensor;
const Shape = zml.Shape;
const ops = zml.ops;
const stdx = zml.stdx;

pub const std_options: std.Options = .{ .log_level = .info };
const log = std.log.scoped(.dump_via_xla);

const CliArgs = struct {
    pub const help =
        \\dump_via_xla --in-dir=<dir> [--out-dir=xla_dump] [--kernel=NAME]
        \\
        \\For each `<NAME>.ttir` under `--in-dir`, compile a synthetic ZML
        \\module that wraps the IR in a `__gpu$xla.gpu.triton` custom_call
        \\and dump XLA's per-pass IR snapshots into `--out-dir`.
        \\
    ;
    @"in-dir": []const u8,
    @"out-dir": []const u8 = "xla_dump",
    kernel: []const u8 = "",
};

/// Side-channel for the active kernel's TTIR — set just before each compile,
/// read by every per-kernel `forward` via `ops.triton(... .ir = active_ttir ...)`.
var active_ttir: [:0]const u8 = "";

// =============================================================================
// Per-kernel forwards. Each one wraps `ops.triton` with a synthetic Tensor
// signature matching the kernel's TTIR `tt.func`. The shapes don't have to be
// runtime-correct — XLA never actually launches the kernel, it just runs the
// codegen pipeline.
// =============================================================================

fn ten1d(comptime dt: zml.DataType, n: i64) Tensor {
    return Tensor.init(.{n}, dt);
}
fn scalar(comptime dt: zml.DataType) Tensor {
    return Tensor.init(.{}, dt);
}

fn run(
    comptime name: [:0]const u8,
    inputs: anytype,
    output: Shape,
    grid: [3]i32,
    num_warps: i32,
    num_stages: i32,
) Tensor {
    return ops.triton(inputs, .{output}, .{
        .name = name,
        .ir = active_ttir,
        .grid = grid,
        .num_warps = num_warps,
        .num_stages = num_stages,
    })[0];
}

const VectorAdd = struct {
    pub fn forward(x: Tensor, y: Tensor, _: Tensor, n: Tensor) Tensor {
        return run("triton_add_kernel", .{ x, y, n }, x.shape(), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.f32, 1024), ten1d(.f32, 1024), ten1d(.f32, 1024), scalar(.i32) };
    }
};

const VectorExpFwd = struct {
    pub fn forward(x: Tensor, _: Tensor, n: Tensor) Tensor {
        return run("triton_exp_kernel", .{ x, n }, x.shape(), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.f32, 1024), ten1d(.f32, 1024), scalar(.i32) };
    }
};

const VectorExpBwd = struct {
    pub fn forward(grad_out: Tensor, out: Tensor, _: Tensor, n: Tensor) Tensor {
        return run("triton_exp_backward_kernel", .{ grad_out, out, n }, grad_out.shape(), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.f32, 1024), ten1d(.f32, 1024), ten1d(.f32, 1024), scalar(.i32) };
    }
};

const LowMemDropout = struct {
    pub fn forward(x: Tensor, x_keep: Tensor, _: Tensor, n: Tensor, p: Tensor) Tensor {
        return run("_triton_dropout", .{ x, x_keep, n, p }, x.shape(), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.f32, 1024), ten1d(.f32, 1024), ten1d(.f32, 1024), scalar(.i32), scalar(.f32) };
    }
};

const SumScalar = struct {
    pub fn forward(input: Tensor, _: Tensor, m: Tensor) Tensor {
        return run("triton_sum_kernel_scalar_result", .{ input, m }, Shape.init(.{}, .f32), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.f32, 1024), ten1d(.f32, 1), scalar(.i32) };
    }
};

// `softmax_kernel` puts `output_ptr` at TTIR position 0; we declare it last
// so the custom_call output binds correctly. The kernel never runs from this
// binary, so the runtime mis-wiring is harmless.
const Softmax = struct {
    pub fn forward(input: Tensor, in_stride: Tensor, out_stride: Tensor, n_cols: Tensor, _: Tensor) Tensor {
        return run("softmax_kernel", .{ input, in_stride, out_stride, n_cols }, input.shape(), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.f32, 1024 * 64), scalar(.i32), scalar(.i32), scalar(.i32), ten1d(.f32, 1024 * 64) };
    }
};

const PerTokenGroupQuantFp8 = struct {
    pub fn forward(y: Tensor, gs: Tensor, ncol: Tensor, rstride: Tensor, eps: Tensor, _: Tensor, ys: Tensor) Tensor {
        return run("per_token_group_quant_fp8", .{ y, gs, ncol, rstride, eps, ys }, Shape.init(.{ 64, 1024 }, .f8e5m2), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{
            ten1d(.bf16, 64 * 1024), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.f32, 1),
            ten1d(.f8e5m2, 64 * 1024), ten1d(.bf16, 64 * 8),
        };
    }
};

const FusedMoe = struct {
    pub fn forward(
        a: Tensor, b: Tensor, b_bias: Tensor, a_scale: Tensor, b_scale: Tensor,
        topk_w: Tensor, sorted_ids: Tensor, expert_ids: Tensor, num_post: Tensor,
        n: Tensor, k: Tensor, em: Tensor, num_valid: Tensor,
        s_am: Tensor, s_be: Tensor, s_bn: Tensor, s_cm: Tensor, s_asm: Tensor, s_ask: Tensor,
        s_bse: Tensor, s_bsk: Tensor, s_bsn: Tensor, s_bbe: Tensor, s_bbn: Tensor,
        _: Tensor,
    ) Tensor {
        return run(
            "fused_moe_kernel",
            .{ a, b, b_bias, a_scale, b_scale, topk_w, sorted_ids, expert_ids, num_post,
               n, k, em, num_valid, s_am, s_be, s_bn, s_cm, s_asm, s_ask, s_bse, s_bsk, s_bsn, s_bbe, s_bbn },
            Shape.init(.{ 256, 1024 }, .bf16),
            .{ 16, 1, 1 }, 4, 2,
        );
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{
            ten1d(.bf16, 256 * 1024), ten1d(.bf16, 8 * 1024 * 1024), ten1d(.bf16, 8 * 1024),
            ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 256), ten1d(.i32, 256), ten1d(.i32, 4), ten1d(.i32, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.bf16, 256 * 1024),
        };
    }
};

const MoeAlign = struct {
    pub fn forward(topk: Tensor, sorted: Tensor, experts: Tensor, num_post: Tensor, cumsum: Tensor, _: Tensor, _: Tensor, _: Tensor, out3: Tensor) Tensor {
        return run("moe_align_block_size_kernel", .{ topk, sorted, experts, num_post, cumsum, out3 }, Shape.init(.{1024}, .i32), .{ 2, 1, 1 }, 8, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{
            ten1d(.i32, 1024), ten1d(.i32, 2048), ten1d(.i32, 32), ten1d(.i32, 1), ten1d(.i32, 9),
            ten1d(.i32, 64), ten1d(.i32, 64), ten1d(.i32, 64), ten1d(.i32, 1024),
        };
    }
};

const CountAndSort = struct {
    pub fn forward(topk: Tensor, sorted: Tensor, cumsum: Tensor, _: Tensor, out1: Tensor) Tensor {
        return run("count_and_sort_expert_tokens_kernel", .{ topk, sorted, cumsum, out1 }, Shape.init(.{1024}, .i32), .{ 1, 1, 1 }, 4, 1);
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{ ten1d(.i32, 1024), ten1d(.i32, 2048), ten1d(.i32, 9), ten1d(.i32, 1024), ten1d(.i32, 1024) };
    }
};

// Unified-attention `_ptr` kernels — arg order matches the `tt.func` signature
// emitted by `zml/attention/triton_kernels/unified_attention.zig` (which is the production
// version we're comparing against). Tensor sizes are dummies; XLA never
// launches these — it just runs the per-pass codegen pipeline. The shapes
// below are oversized (max num_query_heads=64, head_size_padded=256,
// num_segments_per_seq=128) so they cover every variant in the fuzzer.
const _MAX_NUM_TOKENS: i64 = 64;
const _MAX_NUM_QUERY_HEADS: i64 = 64;
const _MAX_HEAD_SIZE_PADDED: i64 = 256;
const _MAX_NUM_BLOCKS: i64 = 64;
const _MAX_NUM_KV_HEADS: i64 = 16;
const _MAX_BLOCK_SIZE: i64 = 16;
const _MAX_NUM_SEGMENTS: i64 = 128;
const _MAX_Q_BUF: i64 = _MAX_NUM_TOKENS * _MAX_NUM_QUERY_HEADS * _MAX_HEAD_SIZE_PADDED;
const _MAX_KV_BUF: i64 = _MAX_NUM_BLOCKS * _MAX_NUM_KV_HEADS * _MAX_BLOCK_SIZE * _MAX_HEAD_SIZE_PADDED;
const _MAX_SEGM_BASE: i64 = _MAX_NUM_TOKENS * _MAX_NUM_QUERY_HEADS * _MAX_NUM_SEGMENTS;

const KernelUnifiedAttention2dPtr = struct {
    pub fn forward(
        query: Tensor, key_cache: Tensor, value_cache: Tensor, sink: Tensor,
        block_tables: Tensor, seq_lens: Tensor, alibi: Tensor, qq_bias: Tensor,
        scale: Tensor, k_scale: Tensor, v_scale: Tensor, out_scale: Tensor, softcap: Tensor,
        bt_stride: Tensor, q_s0: Tensor, q_s1: Tensor, o_s0: Tensor, o_s1: Tensor, qqb_s0: Tensor,
        k_s0: Tensor, k_s1: Tensor, k_s2: Tensor, v_s0: Tensor, v_s1: Tensor, v_s2: Tensor,
        qsl: Tensor, num_seqs: Tensor, _: Tensor,
    ) Tensor {
        return run(
            "kernel_unified_attention_2d_ptr",
            .{ query, key_cache, value_cache, sink, block_tables, seq_lens, alibi, qq_bias,
               scale, k_scale, v_scale, out_scale, softcap,
               bt_stride, q_s0, q_s1, o_s0, o_s1, qqb_s0,
               k_s0, k_s1, k_s2, v_s0, v_s1, v_s2,
               qsl, num_seqs },
            query.shape(),
            .{ 8, 1, 1 }, 4, 2,
        );
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{
            ten1d(.bf16, _MAX_Q_BUF), ten1d(.bf16, _MAX_KV_BUF), ten1d(.bf16, _MAX_KV_BUF),
            ten1d(.f32, _MAX_NUM_QUERY_HEADS), ten1d(.i32, _MAX_NUM_BLOCKS), ten1d(.i32, 1), ten1d(.f32, _MAX_NUM_QUERY_HEADS), ten1d(.f32, 1),
            ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i32, 2), ten1d(.i32, 1), ten1d(.bf16, _MAX_Q_BUF),
        };
    }
};

const KernelUnifiedAttention3dPtr = struct {
    pub fn forward(
        query: Tensor, key_cache: Tensor, value_cache: Tensor, sink: Tensor,
        block_tables: Tensor, seq_lens: Tensor, alibi: Tensor, qq_bias: Tensor,
        scale: Tensor, k_scale: Tensor, v_scale: Tensor, softcap: Tensor,
        bt_stride: Tensor, q_s0: Tensor, q_s1: Tensor, qqb_s0: Tensor,
        k_s0: Tensor, k_s1: Tensor, k_s2: Tensor, v_s0: Tensor, v_s1: Tensor, v_s2: Tensor,
        qsl: Tensor, num_seqs: Tensor, _: Tensor, _: Tensor, _: Tensor,
    ) Tensor {
        const segm = Shape.init(.{ 64 * 32 * 4 }, .f32);
        return run(
            "kernel_unified_attention_3d_ptr",
            .{ query, key_cache, value_cache, sink, block_tables, seq_lens, alibi, qq_bias,
               scale, k_scale, v_scale, softcap,
               bt_stride, q_s0, q_s1, qqb_s0,
               k_s0, k_s1, k_s2, v_s0, v_s1, v_s2,
               qsl, num_seqs },
            segm,
            .{ 1, 8, 4 }, 4, 2,
        );
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{
            ten1d(.bf16, _MAX_Q_BUF), ten1d(.bf16, _MAX_KV_BUF), ten1d(.bf16, _MAX_KV_BUF),
            ten1d(.f32, _MAX_NUM_QUERY_HEADS), ten1d(.i32, _MAX_NUM_BLOCKS), ten1d(.i32, 1), ten1d(.f32, _MAX_NUM_QUERY_HEADS), ten1d(.f32, 1),
            ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1), ten1d(.f32, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1),
            ten1d(.i32, 2), ten1d(.i32, 1),
            ten1d(.f32, _MAX_SEGM_BASE * _MAX_HEAD_SIZE_PADDED), ten1d(.f32, _MAX_SEGM_BASE), ten1d(.f32, _MAX_SEGM_BASE),
        };
    }
};

const ReduceSegmentsPtr = struct {
    pub fn forward(
        segm_out: Tensor, segm_max: Tensor, segm_expsum: Tensor,
        seq_lens: Tensor, num_seqs: Tensor, out_scale_inv: Tensor,
        o_s0: Tensor, o_s1: Tensor, bt_stride: Tensor, qsl: Tensor, _: Tensor,
    ) Tensor {
        return run(
            "reduce_segments_ptr",
            .{ segm_out, segm_max, segm_expsum, seq_lens, num_seqs, out_scale_inv,
               o_s0, o_s1, bt_stride, qsl },
            Shape.init(.{ _MAX_Q_BUF }, .bf16),
            .{ 64, 32, 1 }, 4, 1,
        );
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        return .{
            ten1d(.f32, _MAX_SEGM_BASE * _MAX_HEAD_SIZE_PADDED), ten1d(.f32, _MAX_SEGM_BASE), ten1d(.f32, _MAX_SEGM_BASE),
            ten1d(.i32, 1), ten1d(.i32, 1), ten1d(.f32, 1),
            ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i64, 1), ten1d(.i32, 2),
            ten1d(.bf16, _MAX_Q_BUF),
        };
    }
};

// Gated Delta Net recurrent forward — `_ptr` wrapper kernel from
// `kernels_py/gdn.py`. Two outputs (o, ht), so we return them as a tuple
// rather than going through the single-output `run` helper.
const FusedRecurrentGatedDeltaRule = struct {
    pub fn forward(
        q: Tensor,
        k: Tensor,
        v: Tensor,
        g: Tensor,
        beta: Tensor,
        h0: Tensor,
        cu_seqlens: Tensor,
        _: Tensor,
        _: Tensor,
    ) struct { Tensor, Tensor } {
        const out = ops.triton(
            .{ q, k, v, g, beta, h0, cu_seqlens },
            .{ v.shape(), h0.shape() },
            .{
                .name = "fused_recurrent_gated_delta_rule_fwd_kernel_ptr",
                .ir = active_ttir,
                .grid = .{ 8, 32, 1 },
                .num_warps = 1,
                .num_stages = 1,
            },
        );
        return .{ out[0], out[1] };
    }
    fn args() std.meta.ArgsTuple(@TypeOf(forward)) {
        // Shapes match the Python kwargs in monorepo's `compile_gated_delta_net_kernel`:
        // num_tokens=64, num_qk_heads=4, num_v_heads=16, key_dim=32, value_dim=64, num_sequences=2.
        return .{
            ten1d(.bf16, 1 * 64 * 4 * 32), // q_ptr      (1, T, H, K)
            ten1d(.bf16, 1 * 64 * 4 * 32), // k_ptr      (1, T, H, K)
            ten1d(.bf16, 1 * 64 * 16 * 64), // v_ptr      (1, T, HV, V)
            ten1d(.f32, 1 * 64 * 16), // g_ptr      (1, T, HV)
            ten1d(.bf16, 1 * 64 * 16), // beta_ptr   (1, T, HV)
            ten1d(.f32, 2 * 16 * 32 * 64), // h0_ptr     (NS, HV, K, V)
            ten1d(.i32, 3), // cu_seqlens_ptr (NS+1,)
            ten1d(.bf16, 1 * 64 * 16 * 64), // o_ptr      (output)
            ten1d(.f32, 2 * 16 * 32 * 64), // ht_ptr     (output)
        };
    }
};

// `name` must equal the kernel's `tt.func` symbol (matches `<name>.ttir`).
const KERNELS = .{
    .{ "triton_add_kernel", VectorAdd },
    .{ "triton_exp_kernel", VectorExpFwd },
    .{ "triton_exp_backward_kernel", VectorExpBwd },
    .{ "_triton_dropout", LowMemDropout },
    .{ "triton_sum_kernel_scalar_result", SumScalar },
    .{ "softmax_kernel", Softmax },
    .{ "per_token_group_quant_fp8", PerTokenGroupQuantFp8 },
    .{ "fused_moe_kernel", FusedMoe },
    .{ "moe_align_block_size_kernel", MoeAlign },
    .{ "count_and_sort_expert_tokens_kernel", CountAndSort },
    .{ "kernel_unified_attention_2d_ptr", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_3d_ptr", KernelUnifiedAttention3dPtr },
    .{ "reduce_segments_ptr", ReduceSegmentsPtr },
    // Unified-attention fuzzer variants — each label-suffixed file contains
    // a `tt.func @kernel_unified_attention_*_ptr` body with a different
    // Config2D / Config3D / ConfigReduce baked in. Tensor signatures are
    // identical across variants in a family, so we reuse the same Mod
    // struct (the dummy shapes are oversized to cover every variant).
    .{ "kernel_unified_attention_2d_ptr__dec_h128_g4", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_2d_ptr__pre_h128_g4", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_2d_ptr__pre_h128_g8", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_2d_ptr__pre_h128_g4_long", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_2d_ptr__dec_h256_swa", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_2d_ptr__pre_h256_swa", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_2d_ptr__dec_h64_g1", KernelUnifiedAttention2dPtr },
    .{ "kernel_unified_attention_3d_ptr__pre_h128_g4_seg16", KernelUnifiedAttention3dPtr },
    .{ "kernel_unified_attention_3d_ptr__pre_h128_g8_seg32", KernelUnifiedAttention3dPtr },
    .{ "kernel_unified_attention_3d_ptr__dec_h128_g4_seg64", KernelUnifiedAttention3dPtr },
    .{ "kernel_unified_attention_3d_ptr__pre_h256_seg16", KernelUnifiedAttention3dPtr },
    .{ "reduce_segments_ptr__h128_qh32_seg16", ReduceSegmentsPtr },
    .{ "reduce_segments_ptr__h128_qh64_seg32", ReduceSegmentsPtr },
    .{ "reduce_segments_ptr__h256_qh32_seg16", ReduceSegmentsPtr },
    .{ "fused_recurrent_gated_delta_rule_fwd_kernel_ptr", FusedRecurrentGatedDeltaRule },
};

fn readTtir(allocator: std.mem.Allocator, io: std.Io, in_dir: []const u8, name: []const u8) ![:0]const u8 {
    const path = try std.fmt.allocPrint(allocator, "{s}/{s}.ttir", .{ in_dir, name });
    defer allocator.free(path);
    return std.Io.Dir.cwd().readFileAllocOptions(io, path, allocator, .unlimited, .of(u8), 0);
}

fn compileKernel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    program_name: []const u8,
    out_dir: []const u8,
    comptime forwardFn: anytype,
    args: std.meta.ArgsTuple(@TypeOf(forwardFn)),
) !void {
    const replicated = try zml.sharding.replicatedSharding(platform);
    var exe = try zml.module.compile(allocator, io, forwardFn, args, platform, .{
        .program_name = program_name,
        .shardings = &.{replicated},
        .xla_dump_to = out_dir,
        .xla_dump_emitter_re = "triton-to-llvm",
    });
    exe.deinit();
    log.info("{s}: compiled", .{program_name});
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const platform = try zml.Platform.auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    if (platform.target != .cuda and platform.target != .rocm) {
        log.err("dump_via_xla needs CUDA or ROCm; got {s}", .{@tagName(platform.target)});
        return error.NoGpuPlatform;
    }

    var out_dir_handle = try std.Io.Dir.createDirPathOpen(.cwd(), io, cli.@"out-dir", .{});
    defer out_dir_handle.close(io);
    const out_dir_abs = try std.Io.Dir.cwd().realPathFileAlloc(io, cli.@"out-dir", allocator);
    defer allocator.free(out_dir_abs);
    const in_dir_abs = try std.Io.Dir.cwd().realPathFileAlloc(io, cli.@"in-dir", allocator);
    defer allocator.free(in_dir_abs);

    log.info("reading TTIR from {s}", .{in_dir_abs});

    inline for (KERNELS) |entry| {
        const name = entry[0];
        const Mod = entry[1];
        if (cli.kernel.len == 0 or std.mem.eql(u8, cli.kernel, name)) {
            const ttir = try readTtir(allocator, io, in_dir_abs, name);
            defer allocator.free(ttir);
            active_ttir = ttir;
            try compileKernel(allocator, io, platform, name, out_dir_abs, Mod.forward, Mod.args());
        }
    }

    log.info("done — XLA per-pass dumps in {s}", .{out_dir_abs});
}
