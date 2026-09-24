//! CuTe boundary kernels around the persistent MXFP4 GEMMs of
//! `persistent_mxfp4.zig`: routed MXFP8 input quantization and the top-k
//! reduction.
const std = @import("std");

const zml = @import("../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;

pub const GroupedQuantConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    /// Rows stored per expert group.
    group_size: i64,
    capacity: i64,
    /// Route `r` goes to row `r * group_size`; `route_map` is not read.
    direct: bool = false,
};

pub const CombineConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    columns_per_cta: i64,
};

// Both boundary kernels are programs rather than bare kernels so their launch
// carries the programmatic-dependent-launch attribute: they then start while
// the previous kernel drains, and wait on `griddepcontrol` before their first
// read. A bare kernel is launched by XLA without that attribute.
const GroupedQuantize = cute.Program(GroupedQuantConfig, .{
    .name = "mxfp4_cute_grouped_quantize",
    .inputs = &.{ "x", "route_map" },
    .outputs = &.{ "q", "s" },
    .run = groupedQuantizeProgram,
});

const Combine = cute.Program(CombineConfig, .{
    .name = "mxfp4_cute_combine",
    .inputs = &.{ "d", "routes" },
    .outputs = &.{"y"},
    .run = combineProgram,
});

const quantize_kernel_name = "mxfp4_cute_grouped_quantize_kernel";
const combine_kernel_name = "mxfp4_cute_combine_kernel";

/// Host launch of a boundary kernel: forward the buffer pointers, with PDL.
fn launchBoundary(b: *B, kernel_name: [:0]const u8, grid: [3]i32, args: anytype) cute.FinishError!void {
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, grid[0]), b.cst(.i32, grid[1]), b.cst(.i32, grid[2]) },
        .block = .{ b.cst(.i32, 128), one, one },
        .dynamic_smem = b.kernelSmemSize(kernel_name),
        .stream = b.cudaStream(),
        .cluster = .{ one, one, one },
        .use_pdl = true,
    });
    const launch = b.launchEx(kernel_name, config, args);
    b.returnHostStatus(b.cudaResultStatus(launch));
}

/// Quantize each token once, then scatter it to the physical rows assigned by
/// the routing. Scale bytes use the 128x4 layout consumed by the SM100
/// block-scaled MMA scale-factor tensor map.
pub fn groupedQuantize(x: zml.Tensor, route_map: zml.Tensor, cfg: GroupedQuantConfig) struct { q: zml.Tensor, s: zml.Tensor } {
    std.debug.assert(cfg.group_size == 8 or cfg.group_size == 16 or cfg.group_size == 32 or cfg.group_size == 64 or cfg.group_size == 128);
    const result = GroupedQuantize.call(
        .{ .x = x, .route_map = route_map },
        .{
            .q = .init(.{ cfg.capacity * cfg.group_size, cfg.hidden }, .f8e4m3fn),
            .s = .init(.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 }, .u8),
        },
        .{ .cfg = cfg },
    );
    return .{ .q = result.q, .s = result.s };
}

/// Amortize launch overhead for large prefills while retaining enough CTAs
/// to saturate the GPU for decode and small batches.
pub fn combineColumns(tokens: i64, hidden: i64) i64 {
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
pub fn combine(d: zml.Tensor, routes: zml.Tensor, cfg: CombineConfig) zml.Tensor {
    return Combine.call(
        .{ .d = d, .routes = routes },
        .{ .y = .init(.{ cfg.tokens, cfg.hidden }, .bf16) },
        .{ .cfg = cfg },
    ).y;
}

/// Python `prepare_input_vector`: each thread quantizes four consecutive
/// values and stores them as one packed word, so eight lanes cover one
/// 32-value MX block.
fn groupedQuantizeProgram(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    b.beginFunction(quantize_kernel_name, .cuda_kernel);
    try groupedQuantizeKernel(b, cfg);
    b.endFunction(.{ 128, 1, 1 });

    b.beginFunction(GroupedQuantize.name, .host);
    // Host parameters carry the custom call's element types; the kernel reads
    // four FP8 values per 32-bit word.
    const host = try b.declareArgs(.{
        .x = .{ .ptr = cute.DType.bf16 },
        .route_map = .{ .ptr = cute.DType.i32 },
        .q = .{ .ptr = cute.DType.f8e4m3fn },
        .s = .{ .ptr = cute.DType.i8 },
    });
    try launchBoundary(b, quantize_kernel_name, .{ @intCast(@divExact(cfg.hidden, 4 * 128)), @intCast(cfg.tokens), 1 }, .{
        host.x, host.route_map, b.recastPointer(host.q, .i32, .gmem, 16), host.s,
    });
    b.endFunction(null);
}

fn groupedQuantizeKernel(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    std.debug.assert(@rem(cfg.hidden, 4 * 128) == 0);
    const a = try b.declareArgs(.{
        .x = .{ .tensor = .{ .dtype = .bf16, .shape = &.{ cfg.tokens, cfg.hidden } } },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.tokens * cfg.topk} } },
        // Four FP8 values per word.
        .q = .{ .tensor = .{ .dtype = .i32, .shape = &.{ cfg.capacity * cfg.group_size, @divExact(cfg.hidden, 4) } } },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 } } },
    });

    const tid = b.threadIdx().x;
    // This grid may have started before its producer finished: the routing and
    // the previous layer write what it reads below.
    b.waitForDependency();
    const token = b.blockIdx().y;
    const word = b.blockIdx().x.mul(128).add(tid);
    var x: [4]cute.Value = undefined;
    for (&x, 0..) |*value, i| value.* = a.x.get(.{ token, word.mul(4).add(@as(i32, @intCast(i))) }).to(.f32);

    var maximum = x[0].abs().maximum(x[1].abs()).maximum(x[2].abs().maximum(x[3].abs()));
    inline for (.{ 1, 2, 4 }) |offset| {
        maximum = maximum.maximum(b.shuffleXor(maximum, offset));
    }
    const raw_bits = maximum.maximum(1e-4).mul(1.0 / 448.0).bitCast(.i32);
    const exponent = raw_bits.shrLogical(23).add(raw_bits.bitAnd(0x7fffff).ne(0).to(.i32));
    const inverse_scale = b.cst(.i32, 254).sub(exponent).shl(23).bitCast(.f32);
    const packed_word = b.packFp8x4(x[0].mul(inverse_scale), x[1].mul(inverse_scale), x[2].mul(inverse_scale), x[3].mul(inverse_scale));

    const scale_col = word.div(8);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = token.mul(cfg.topk).add(slot);
        const row = if (cfg.direct) route.mul(cfg.group_size) else a.route_map.get(.{route});
        // Routes of other expert-parallel ranks are not scheduled.
        var scheduled = if (cfg.direct) null else b.openIf(row.ge(0));
        a.q.set(.{ row, word }, packed_word);

        var leader = b.openIf(tid.rem(8).eq(0));
        const scale_row = row.div(cfg.group_size);
        const row_in_group = row.rem(cfg.group_size);
        const swizzled_col = scale_col.div(4).mul(512)
            .add(row_in_group.rem(32).mul(16))
            .add(row_in_group.div(32).mul(4))
            .add(scale_col.rem(4));
        a.s.set(.{ scale_row, swizzled_col }, exponent.to(.i8));
        leader.yieldThen(.{});
        if (scheduled) |*s| s.yieldThen(.{});
    }
    // Only now are the rows this CTA owns complete: release the persistent
    // GEMM, which waits on this signal before its first TMA read.
    b.launchDependents();
}

/// Each thread owns two adjacent BF16 columns, read and written as one
/// 32-bit word: a warp then moves a full 128-byte line per access.
fn combineProgram(b: *B, cfg: CombineConfig) cute.FinishError!void {
    b.beginFunction(combine_kernel_name, .cuda_kernel);
    try combineKernel(b, cfg);
    b.endFunction(.{ 128, 1, 1 });

    b.beginFunction(Combine.name, .host);
    // The kernel reads and writes two BF16 columns per 32-bit word.
    const host = try b.declareArgs(.{
        .d = .{ .ptr = cute.DType.bf16 },
        .routes = .{ .ptr = cute.DType.i32 },
        .y = .{ .ptr = cute.DType.bf16 },
    });
    try launchBoundary(b, combine_kernel_name, .{
        @intCast(@divTrunc(cfg.hidden + cfg.columns_per_cta - 1, cfg.columns_per_cta)),
        @intCast(cfg.tokens),
        1,
    }, .{ b.recastPointer(host.d, .i32, .gmem, 16), host.routes, b.recastPointer(host.y, .i32, .gmem, 16) });
    b.endFunction(null);
}

fn combineKernel(b: *B, cfg: CombineConfig) cute.FinishError!void {
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

test "MXFP4 CuTe boundary kernels emit for decode and prefill batches" {
    inline for (.{ @as(i64, 1), 6, 96, 16_384 }) |tokens| {
        const quant_ir = try GroupedQuantize.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .group_size = 16,
            .capacity = @divTrunc(tokens * 6 + 15, 16) + 191,
        });
        defer std.testing.allocator.free(quant_ir);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "griddepcontrol.launch_dependents;") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cvt.rn.satfinite.e4m3x2.f32") != null);

        const combine_ir = try Combine.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
            .columns_per_cta = combineColumns(tokens, 5120),
        });
        defer std.testing.allocator.free(combine_ir);
        try std.testing.expect(std.mem.indexOf(u8, combine_ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, combine_ir, "scf.if") != null);
    }
}
