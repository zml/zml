//! Emission test of the MegaMoE kernel (`kernel.zig`): the `cuda.kernel` of vLLM's production
//! instantiation and of a single-rank one must build and verify, and carry DeepGEMM's roles.
//! (Compilation through the CuTe compiler and ptxas is `gpu_test.zig`'s job.)
const std = @import("std");

const mlir = @import("mlir");
const cute = @import("kernels/cute/builder");

const kernel = @import("kernel.zig");
const layout = @import("layout.zig");

const B = cute.Builder;

/// A MegaMoE instantiation (`program.Instance`).
pub const Instance = @import("program.zig").Instance;

/// vLLM's DSv4.1 instantiation (EP2, `max_num_batched_tokens` 8192, 4096 tokens per rank).
pub const vllm_dsv41: Instance = .{};
/// Single rank, the same model, 8192 tokens (BLOCK_M 192 with 384 local experts).
pub const single_rank: Instance = .{ .num_ranks = 1, .num_tokens = 8192 };

fn testContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (cute.dialects_needed) |d| mlir.DialectHandle.fromString(d).insertDialect(registry);
    mlir.registerFuncExtensions(registry);
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

fn expectContains(haystack: []const u8, needle: []const u8) !void {
    if (std.mem.indexOf(u8, haystack, needle) == null) {
        std.debug.print("missing {s}\n", .{needle});
        return error.TestExpectedContains;
    }
}

fn emitAndCheck(inst: Instance, expected_block_m: u32) !void {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try B.openProgram(std.testing.allocator, ctx, "megamoe");
    defer b.deinit();
    const cfg = inst.config();
    try std.testing.expectEqual(expected_block_m, cfg.kernel.cfg.block_m);
    try kernel.emit(&b, cfg, "megamoe_kernel");
    const ir = try b.finishProgram();
    defer std.testing.allocator.free(ir);
    const k = cfg.kernel;
    var buf: [128]u8 = undefined;
    for ([_][]const u8{
        "griddepcontrol.wait;",
        "tcgen05.alloc.cta_group::2.sync.aligned.shared::cta.b32 [$0], $1;",
        try std.fmt.bufPrint(&buf, "setmaxnreg.dec.sync.aligned.u32 {d};", .{k.numDispatchRegisters()}),
        "atom.shared.cta.add.u32",
        "atom.release.gpu.global.add.u32",
        "red.release.sys.global.add.s32",
        "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint",
        "cp.async.bulk.global.shared::cta.bulk_group.L2::cache_hint",
        "redux.sync.min.u32",
        "fns.b32",
        "st.async.shared::cluster.mbarrier::complete_tx::bytes.u32.v4",
        "cp.async.bulk.tensor.2d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint",
        "cp.async.bulk.tensor.4d.cta_group::2.shared::cluster.global.mbarrier::complete_tx::bytes.L2::cache_hint",
        "red.release.gpu.global.xor.b64",
        "ld.acquire.gpu.global.u64",
        "tcgen05.cp.cta_group::2.32x128b.warpx4",
        "tcgen05.mma.cta_group::2.kind::mxf8f6f4.block_scale",
        "tcgen05.commit.cta_group::2.mbarrier::arrive::one.shared::cluster.multicast::cluster.b64",
        "tcgen05.dealloc.cta_group::2.sync.aligned.b32",
        "st.release.sys.global.u64",
        "barrier.sync $0, $1;",
        // Epilogue and combine
        "tcgen05.ld.sync.aligned.16x256b.x1.b32",
        "tcgen05.wait::ld.sync.aligned;",
        "min.bf16x2",
        "ex2.approx.f32",
        "rcp.approx.ftz.f32",
        "mul.rn.f32x2",
        "cvt.rn.satfinite.e4m3x2.f32",
        "stmatrix.sync.aligned.m16n8.x1.trans.shared.b8",
        "cp.async.bulk.tensor.2d.global.shared::cta.bulk_group",
        "stmatrix.sync.aligned.x4.m8n8.shared.b16.trans",
        "ld.shared.v4.f32",
        "ld.acquire.sys.global.b64",
        "vote.sync.all.pred",
        "add.rn.f32.bf16",
        "st.shared.v4.u32",
    }) |needle| try expectContains(ir, needle);
}

test "vLLM's DSv4.1 MegaMoE kernel emits and verifies" {
    try emitAndCheck(vllm_dsv41, 192);
}

test "single-rank MegaMoE kernel emits and verifies" {
    try emitAndCheck(single_rank, 192);
}

test "small-BLOCK_M MegaMoE kernels emit and verify" {
    // BLOCK_M 16 / 32 / 64 / 128 / 240 instantiations (different TMEM sizes, register split,
    // ring blocks), single rank and without shared expert.
    try emitAndCheck(.{ .num_tokens = 256 }, 16);
    try emitAndCheck(.{ .num_tokens = 768 }, 32);
    try emitAndCheck(.{ .num_tokens = 1024, .num_shared_experts = 0 }, 64);
    try emitAndCheck(.{ .num_ranks = 1, .num_tokens = 4096 }, 128);
    try emitAndCheck(.{ .num_experts = 128, .num_topk = 3, .num_tokens = 4096 }, 240);
}

fn emitProgramAndCheck(spec: @import("program.zig").Spec, needles: []const []const u8) !void {
    const ctx = try testContext();
    defer ctx.deinit();
    var b = try B.openProgram(std.testing.allocator, ctx, "megamoe");
    defer b.deinit();
    try @import("program.zig").emitProgram(&b, spec, "megamoe");
    const ir = try b.finishProgram();
    defer std.testing.allocator.free(ir);
    for (needles) |needle| try expectContains(ir, needle);
}

test "MegaMoE programs (staging kernel, MegaMoE kernel, host launches) emit and verify" {
    // One rank, staged inputs, workspace operand: 13 operands + y + workspace_out.
    try emitProgramAndCheck(.{ .instance = .{ .num_ranks = 1, .num_tokens = 512 }, .inputs = .staged, .workspace = .operand }, &.{
        "megamoe_staging_kernel",
        "cuda.launch_ex",
    });
    // Two ranks, bf16 inputs, symmetric workspace: 12 operands + y, then 2 workspaces and the rank.
    try emitProgramAndCheck(.{ .instance = .{}, .inputs = .bf16, .workspace = .symmetric }, &.{
        "megamoe_staging_kernel",
        "div.rn.f32",
        "cvt.rn.satfinite.e4m3x2.f32",
        "shfl.sync.bfly.b32",
        "!cute.ptr<bf16, gmem, align<16>>, !cute.ptr<i8, gmem, align<16>>, !cute.ptr<i8, gmem, align<16>>, i32) -> i32",
    });
    // No shared expert, BLOCK_M 64.
    try emitProgramAndCheck(.{ .instance = .{ .num_ranks = 1, .num_tokens = 1024, .num_shared_experts = 0 }, .inputs = .bf16, .workspace = .operand }, &.{"megamoe_staging_kernel"});
}
