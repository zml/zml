//! CuTe top-k reduction behind the persistent MoE GEMMs: sums each token's
//! BF16 route rows.
const std = @import("std");

const zml = @import("../../../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;
const launchBoundary = @import("common.zig").launchBoundary;

pub const Config = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    columns_per_cta: i64,
};

// A program rather than a bare kernel so its launch carries the
// programmatic-dependent-launch attribute (see `mxfp4.zig`).
const Topk = cute.Program(Config, .{
    .name = "moe_cute_reduce_topk",
    .inputs = &.{ "d", "routes" },
    .outputs = &.{"y"},
    .run = topkProgram,
});

const topk_kernel_name = "moe_cute_reduce_topk_kernel";

/// Amortize launch overhead for large prefills while retaining enough CTAs
/// to saturate the GPU for decode and small batches.
pub fn columnsPerCta(tokens: i64, hidden: i64) i64 {
    const elements = tokens * hidden;
    if (tokens > 1 and elements >= 512 * 2048) return 2048;
    if (tokens > 1 and elements >= 512 * 1024) return 1024;
    // Two columns per thread is the smallest tile: 128 threads, one 32-bit
    // word each.
    return 256;
}

/// Sum the `[tokens * topk, hidden]` BF16 route rows of each token into BF16.
/// Routing weights were already applied before the down projection.
/// `routes` is negative for routes this rank does not compute (expert
/// parallelism): their rows were never written.
pub fn topk(d: zml.Tensor, routes: zml.Tensor, cfg: Config) zml.Tensor {
    return Topk.call(
        .{ .d = d, .routes = routes },
        .{ .y = .init(.{ cfg.tokens, cfg.hidden }, .bf16) },
        .{ .cfg = cfg },
    ).y;
}

/// Each thread owns two adjacent BF16 columns, read and written as one
/// 32-bit word: a warp then moves a full 128-byte line per access.
fn topkProgram(b: *B, cfg: Config) cute.FinishError!void {
    b.beginFunction(topk_kernel_name, .cuda_kernel);
    try topkKernel(b, cfg);
    b.endFunction(.{ 128, 1, 1 });

    b.beginFunction(Topk.name, .host);
    // The kernel reads and writes two BF16 columns per 32-bit word.
    const host = try b.declareArgs(.{
        .d = .{ .ptr = cute.DType.bf16 },
        .routes = .{ .ptr = cute.DType.i32 },
        .y = .{ .ptr = cute.DType.bf16 },
    });
    try launchBoundary(b, topk_kernel_name, .{
        @intCast(@divTrunc(cfg.hidden + cfg.columns_per_cta - 1, cfg.columns_per_cta)),
        @intCast(cfg.tokens),
        1,
    }, .{ b.recastPointer(host.d, .i32, .gmem, 16), host.routes, b.recastPointer(host.y, .i32, .gmem, 16) });
    b.endFunction(null);
}

fn topkKernel(b: *B, cfg: Config) cute.FinishError!void {
    const pairs = @divExact(cfg.hidden, 2);
    const a = try b.declareArgs(.{
        .d = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.tokens * cfg.topk, pairs } } },
        .routes = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.tokens * cfg.topk} } },
        .y = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.tokens, pairs } } },
    });

    // The down projection's rows arrive from a grid this one may overlap.
    b.waitForDependency();
    const token = b.blockIdx().y;
    const pairs_per_cta = @divExact(cfg.columns_per_cta, 2);
    const first_pair = b.blockIdx().x.mul(pairs_per_cta).add(b.threadIdx().x);
    for (0..@intCast(@divExact(pairs_per_cta, 128))) |chunk| {
        const pair = first_pair.add(chunk * 128);
        var in_bounds = b.openIf(pair.lt(pairs));
        var low = b.cst(.f32, 0);
        var high = b.cst(.f32, 0);
        for (0..@intCast(cfg.topk)) |slot| {
            const route = token.mul(cfg.topk).add(slot);
            // Routes of other expert-parallel ranks were never written.
            var computed = b.openIfElse(a.routes.get(.{route}).ge(0), .{ cute.DType.f32.toMlir(b.ctx), cute.DType.f32.toMlir(b.ctx) });
            const word = a.d.get(.{ route, pair });
            computed.yieldThen(.{
                low.add(word.shl(16).bitCast(.f32)),
                high.add(word.bitAnd(@as(i32, @bitCast(@as(u32, 0xffff0000)))).bitCast(.f32)),
            });
            computed.yieldElse(.{ low, high });
            low = computed.results[0];
            high = computed.results[1];
        }
        const low_bits = low.to(.bf16).to(.f32).bitCast(.i32).shrLogical(16);
        const high_bits = high.to(.bf16).to(.f32).bitCast(.i32).bitAnd(@as(i32, @bitCast(@as(u32, 0xffff0000))));
        a.y.set(.{ token, pair }, low_bits.bitOr(high_bits));
        in_bounds.yieldThen(.{});
    }
    b.launchDependents();
}

test "CuTe top-k reduction emits for decode and prefill batches" {
    inline for (.{ @as(i64, 1), 6, 96, 16_384 }) |tokens| {
        const ir = try Topk.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .columns_per_cta = columnsPerCta(tokens, 5120),
        });
        defer std.testing.allocator.free(ir);
        try std.testing.expect(std.mem.indexOf(u8, ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, ir, "scf.if") != null);
    }
}
