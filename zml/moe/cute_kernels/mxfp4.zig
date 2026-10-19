//! CuTe boundary kernels around the persistent MXFP4 GEMMs of
//! `persistent_mxfp4.zig`: routed MXFP8 input quantization and the top-k
//! reduction.
const std = @import("std");

const zml = @import("../../zml.zig");
const cute = zml.kernel.cute;
const B = cute.Builder;

const persistent = @import("persistent_mxfp4.zig");

pub const GroupedQuantConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
    /// Rows stored per expert group.
    group_size: i64,
    capacity: i64,
    /// Route `r` goes to row `r * group_size`; `route_map` is not read.
    direct: bool = false,
    /// Threads sharing one 32-value MX block, 1 or 4. One thread per block
    /// keeps the most bytes in flight for prefills; four spread a small
    /// batch over more threads, which shortens its latency.
    lanes_per_block: i64 = 1,

    fn valuesPerThread(self: GroupedQuantConfig) i64 {
        return @divExact(32, self.lanes_per_block);
    }

    fn threadsPerToken(self: GroupedQuantConfig) i64 {
        return @divExact(self.hidden, self.valuesPerThread());
    }

    /// Threads of one CTA: a whole token with one thread per block, else 128.
    fn ctaThreads(self: GroupedQuantConfig) i64 {
        return if (self.lanes_per_block == 1) self.threadsPerToken() else 128;
    }
};

/// `GroupedQuantConfig.lanes_per_block` for a batch.
pub fn quantizeLanes(tokens: i64) i64 {
    return if (tokens <= 256) 4 else 1;
}

pub const CombineConfig = struct {
    tokens: i64,
    hidden: i64,
    topk: i64,
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
fn launchBoundary(b: *B, kernel_name: [:0]const u8, grid: [3]i32, threads: i32, args: anytype) cute.FinishError!void {
    const one = b.cst(.i32, 1);
    const config = b.makeLaunchConfig(.{
        .grid = .{ b.cst(.i32, grid[0]), b.cst(.i32, grid[1]), b.cst(.i32, grid[2]) },
        .block = .{ b.cst(.i32, threads), one, one },
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

/// Each thread quantizes 32 / `lanes_per_block` consecutive values of a
/// token with 16-byte loads, so each SM keeps enough bytes in flight to
/// stream the activations, then stores its FP8 values to every row the token
/// is routed to with 8- or 16-byte stores.
fn groupedQuantizeProgram(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    const cta_threads: i32 = @intCast(cfg.ctaThreads());
    b.beginFunction(quantize_kernel_name, .cuda_kernel);
    try groupedQuantizeKernel(b, cfg);
    b.endFunction(.{ cta_threads, 1, 1 });

    b.beginFunction(GroupedQuantize.name, .host);
    // Host parameters carry the custom call's element types; the kernel moves
    // 32-bit words.
    const host = try b.declareArgs(.{
        .x = .{ .ptr = cute.DType.bf16 },
        .route_map = .{ .ptr = cute.DType.i32 },
        .q = .{ .ptr = cute.DType.f8e4m3fn },
        .s = .{ .ptr = cute.DType.i8 },
    });
    try launchBoundary(b, quantize_kernel_name, .{ @intCast(@divExact(cfg.threadsPerToken(), cfg.ctaThreads())), @intCast(cfg.tokens), 1 }, cta_threads, .{
        b.recastPointer(host.x, .i32, .gmem, 16), host.route_map, b.recastPointer(host.q, .i32, .gmem, 16), host.s,
    });
    b.endFunction(null);
}

fn groupedQuantizeKernel(b: *B, cfg: GroupedQuantConfig) cute.FinishError!void {
    std.debug.assert(cfg.lanes_per_block == 1 or cfg.lanes_per_block == 4);
    std.debug.assert(@rem(cfg.threadsPerToken(), cfg.ctaThreads()) == 0 and cfg.ctaThreads() <= 1024);
    const a = try b.declareArgs(.{
        // BF16 pairs.
        .x = .{ .ptr = cute.DType.i32 },
        .route_map = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.tokens * cfg.topk} } },
        // Four FP8 values per word.
        .q = .{ .ptr = cute.DType.i32 },
        .s = .{ .tensor = .{ .dtype = .i8, .shape = &.{ cfg.capacity, @divExact(cfg.hidden, 32) * 128 } } },
    });

    const values: usize = @intCast(cfg.valuesPerThread());
    // The thread's values within its token, and its MX block.
    const first = b.blockIdx().x.mul(@as(i32, @intCast(cfg.ctaThreads()))).add(b.threadIdx().x);
    const block = first.div(@as(i32, @intCast(cfg.lanes_per_block)));
    // This grid may have started before its producer finished: the routing and
    // the previous layer write what it reads below.
    b.waitForDependency();
    const token = b.blockIdx().y;
    const x_address = b.ptrToInt(a.x, .i64).add(token.to(.i64).mul(cfg.hidden * 2)).add(first.to(.i64).mul(@as(i64, @intCast(values * 2))));
    var pairs: [16]cute.Value = undefined;
    for (0..values / 8) |i| {
        const words = b.loadWords(4, x_address.add(@as(i64, @intCast(16 * i))), .global, .{ .invariant = true });
        for (words, 0..) |word, j| pairs[4 * i + j] = word;
    }
    var x: [32]cute.Value = undefined;
    for (pairs[0 .. values / 2], 0..) |pair, i| x[2 * i ..][0..2].* = b.unpackBf16x2(pair);
    var maximum = x[0].abs();
    for (x[1..values]) |value| maximum = maximum.maximum(value.abs());
    var offset: i32 = 1;
    while (offset < cfg.lanes_per_block) : (offset *= 2) maximum = maximum.maximum(b.shuffleXor(maximum, offset));
    const raw_bits = maximum.maximum(1e-4).mul(1.0 / 448.0).bitCast(.i32);
    const exponent = raw_bits.shrLogical(23).add(raw_bits.bitAnd(0x7fffff).ne(0).to(.i32));
    const inverse_scale = b.cst(.i32, 254).sub(exponent).shl(23).bitCast(.f32);
    var packed_words: [8]cute.Value = undefined;
    for (packed_words[0 .. values / 4], 0..) |*word, i| {
        word.* = b.packFp8x4(x[4 * i].mul(inverse_scale), x[4 * i + 1].mul(inverse_scale), x[4 * i + 2].mul(inverse_scale), x[4 * i + 3].mul(inverse_scale));
    }

    const q_base = b.ptrToInt(a.q, .i64);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = token.mul(cfg.topk).add(slot);
        const row = if (cfg.direct) route.mul(cfg.group_size) else a.route_map.get(.{route});
        // Routes of other expert-parallel ranks are not scheduled.
        var scheduled = if (cfg.direct) null else b.openIf(row.ge(0));
        const q_address = q_base.add(row.to(.i64).mul(cfg.hidden)).add(first.to(.i64).mul(@as(i64, @intCast(values))));
        if (values == 8) {
            b.storeWords(2, q_address, .global, packed_words[0..2].*);
        } else {
            b.storeWords(4, q_address, .global, packed_words[0..4].*);
            b.storeWords(4, q_address.add(16), .global, packed_words[4..8].*);
        }

        // One lane per block stores its scale.
        var leader = if (cfg.lanes_per_block == 1) null else b.openIf(first.rem(@as(i32, @intCast(cfg.lanes_per_block))).eq(0));
        const scale_row = row.div(cfg.group_size);
        const row_in_group = row.rem(cfg.group_size);
        a.s.set(.{ scale_row, persistent.scaleColumn(block, row_in_group.rem(32), row_in_group.div(32)) }, exponent.to(.i8));
        if (leader) |*l| l.yieldThen(.{});
        if (scheduled) |*s| s.yieldThen(.{});
    }
    // Only now are the rows this CTA owns complete: release the persistent
    // GEMM, which waits on this signal before its first TMA read.
    b.launchDependents();
}

/// Each thread owns eight adjacent BF16 columns of one token, read and
/// written 16 bytes at a time.
const combine_columns = 8;
const combine_threads = 128;

fn combineProgram(b: *B, cfg: CombineConfig) cute.FinishError!void {
    b.beginFunction(combine_kernel_name, .cuda_kernel);
    try combineKernel(b, cfg);
    b.endFunction(.{ combine_threads, 1, 1 });

    b.beginFunction(Combine.name, .host);
    // The kernel reads and writes BF16 pairs as 32-bit words.
    const host = try b.declareArgs(.{
        .d = .{ .ptr = cute.DType.bf16 },
        .routes = .{ .ptr = cute.DType.i32 },
        .y = .{ .ptr = cute.DType.bf16 },
    });
    const chunks = @divExact(cfg.hidden, combine_columns);
    try launchBoundary(b, combine_kernel_name, .{
        @intCast(@divTrunc(chunks + combine_threads - 1, combine_threads)),
        @intCast(cfg.tokens),
        1,
    }, combine_threads, .{ b.recastPointer(host.d, .i32, .gmem, 16), host.routes, b.recastPointer(host.y, .i32, .gmem, 16) });
    b.endFunction(null);
}

fn combineKernel(b: *B, cfg: CombineConfig) cute.FinishError!void {
    const chunks = @divExact(cfg.hidden, combine_columns);
    const a = try b.declareArgs(.{
        .d = .{ .ptr = cute.DType.i32 },
        .routes = .{ .tensor = .{ .dtype = .i32, .shape = &.{cfg.tokens * cfg.topk} } },
        .y = .{ .ptr = cute.DType.i32 },
    });

    // The down projection's rows arrive from a grid this one may overlap.
    b.waitForDependency();
    const token = b.blockIdx().y;
    const chunk = b.blockIdx().x.mul(combine_threads).add(b.threadIdx().x);
    var in_bounds = b.openIf(chunk.lt(chunks));
    const d_base = b.ptrToInt(a.d, .i64).add(chunk.to(.i64).mul(16));
    const i32_type = cute.DType.i32.toMlir(b.ctx);
    // Load every route first so the loads overlap; routes of other
    // expert-parallel ranks were never written and contribute zeros.
    var loaded: [16][4]cute.Value = undefined;
    std.debug.assert(cfg.topk <= loaded.len);
    for (0..@intCast(cfg.topk)) |slot| {
        const route = token.mul(cfg.topk).add(slot);
        var computed = b.openIfElse(a.routes.get(.{route}).ge(0), .{ i32_type, i32_type, i32_type, i32_type });
        const words = b.loadWords(4, d_base.add(route.to(.i64).mul(cfg.hidden * 2)), .global, .{ .invariant = true });
        computed.yieldThen(.{ words[0], words[1], words[2], words[3] });
        const zero = b.cst(.i32, 0);
        computed.yieldElse(.{ zero, zero, zero, zero });
        for (&loaded[slot], computed.results[0..4]) |*word, result| word.* = result;
    }
    // Sum in route order, as before: adding the +0 of a skipped route leaves
    // a sum that starts at +0 unchanged.
    var sums: [combine_columns]cute.Value = @splat(b.cst(.f32, 0));
    for (loaded[0..@intCast(cfg.topk)]) |words| {
        for (words, 0..) |word, i| {
            const pair = b.unpackBf16x2(word);
            sums[2 * i] = sums[2 * i].add(pair[0]);
            sums[2 * i + 1] = sums[2 * i + 1].add(pair[1]);
        }
    }
    var out: [4]cute.Value = undefined;
    for (&out, 0..) |*word, i| word.* = b.packBf16x2(sums[2 * i], sums[2 * i + 1]);
    b.storeWords(4, b.ptrToInt(a.y, .i64).add(token.to(.i64).mul(cfg.hidden * 2)).add(chunk.to(.i64).mul(16)), .global, out);
    in_bounds.yieldThen(.{});
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
            .lanes_per_block = quantizeLanes(tokens),
        });
        defer std.testing.allocator.free(quant_ir);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "griddepcontrol.launch_dependents;") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, quant_ir, "cvt.rn.satfinite.e4m3x2.f32") != null);

        const combine_ir = try Combine.emit(std.testing.allocator, .{
            .tokens = tokens,
            .hidden = 5120,
            .topk = 6,
        });
        defer std.testing.allocator.free(combine_ir);
        try std.testing.expect(std.mem.indexOf(u8, combine_ir, "cuda.launch_ex") != null);
        try std.testing.expect(std.mem.indexOf(u8, combine_ir, "scf.if") != null);
    }
}

test "CuTe quantization matches scalar FP8 rounding, scales, and route scatter" {
    const platform = zml.testing.env();
    if (!@import("../cute_mxfp4.zig").isAvailable(platform)) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const hidden = 5120;
    const Local = struct {
        fn call(x: zml.Tensor, routes: zml.Tensor, cfg: GroupedQuantConfig) [2]zml.Tensor {
            const result = groupedQuantize(x, routes, cfg);
            return .{ result.q.bitCast(.u8), result.s };
        }

        // Enumerate the finite E4M3 codes: independent scalar nearest-even
        // rounding, including subnormals and signed zero.
        fn fp8(x: f32) u8 {
            var best: u8 = 0;
            var distance = std.math.inf(f32);
            for (0..127) |code| {
                const exponent: i32 = @intCast(code / 8);
                const mantissa: f32 = @floatFromInt(code % 8);
                const value = if (exponent == 0) mantissa * 0x1p-9 else (1 + mantissa / 8) * std.math.pow(f32, 2, @floatFromInt(exponent - 7));
                const delta = @abs(@abs(x) - value);
                if (delta < distance or (delta == distance and code % 2 == 0)) {
                    best = @intCast(code);
                    distance = delta;
                }
            }
            return best | (if (std.math.signbit(x)) @as(u8, 128) else 0);
        }
    };
    for ([_]bool{ true, false }) |direct| {
        const tokens: i64 = if (direct) 1 else 3;
        const group: i64 = if (direct) 8 else 16;
        var host = try zml.Slice.alloc(allocator, .init(.{ tokens, hidden }, .bf16));
        defer host.free(allocator);
        const pattern = [_]f32{ 0, -0.0, 1.0625, 1.1875, -1.0625, -1.1875, 0x1p-10, 448 };
        for (host.items(zml.floats.BFloat16), 0..) |*value, i| {
            var v = pattern[i % pattern.len];
            v = switch ((i / 32) % 5) {
                0 => 0,
                1 => v * 0x1p-16,
                2 => v,
                3 => v * 128,
                else => if (i % 32 == 31) 450 else v,
            };
            value.* = .fromF32(v);
        }
        const routes = [_]i32{ 17, 0, -1, -1, 15, 31 };
        const ids = routes[0..@intCast(tokens * 2)];
        var input = try zml.Buffer.fromSlice(io, platform, host);
        defer input.deinit();
        var map = try zml.Buffer.fromBytes(io, platform, .init(.{tokens * 2}, .i32), std.mem.sliceAsBytes(ids));
        defer map.deinit();
        for ([_]i64{ 1, 4 }) |lanes| {
            const cfg: GroupedQuantConfig = .{ .tokens = tokens, .hidden = hidden, .topk = 2, .group_size = group, .capacity = 2, .direct = direct, .lanes_per_block = lanes };
            var exe = try platform.compileFn(allocator, io, Local.call, .{ zml.Tensor.fromShape(host.shape), zml.Tensor.init(.{tokens * 2}, .i32), cfg }, .{});
            defer exe.deinit();
            var runner = try exe.runner(allocator);
            defer runner.deinit(allocator);
            var output: [2]zml.Buffer = undefined;
            runner.run(io, .{ input, map }, &output, .{ .wait = true });
            defer for (&output) |*buffer| buffer.deinit();
            const q = try output[0].toSliceAlloc(allocator, io);
            defer q.free(allocator);
            const scales = try output[1].toSliceAlloc(allocator, io);
            defer scales.free(allocator);
            for (ids, 0..) |mapped, route| {
                if (!direct and mapped < 0) continue;
                const row: usize = if (direct) route * @as(usize, @intCast(group)) else @intCast(mapped);
                for (0..hidden / 32) |block| {
                    const values = host.constItems(zml.floats.BFloat16)[(route / 2) * hidden + block * 32 ..][0..32];
                    var maximum: f32 = 1e-4;
                    for (values) |value| maximum = @max(maximum, @abs(value.toF32()));
                    var scale: f32 = 0x1p-22;
                    var exponent: u8 = 105;
                    while (maximum > 448 * scale) : (exponent += 1) scale *= 2;
                    const group_size: usize = @intCast(group);
                    const local_row = row % group_size;
                    const scale_index = (row / group_size) * (hidden / 32 * 128) + (block / 4) * 512 + (local_row % 32) * 16 + (local_row / 32) * 4 + block % 4;
                    try std.testing.expectEqual(exponent, scales.constItems(u8)[scale_index]);
                    for (values, 0..) |value, column| try std.testing.expectEqual(Local.fp8(value.toF32() / scale), q.constItems(u8)[row * hidden + block * 32 + column]);
                }
            }
        }
    }
}

test "CuTe combine ignores poisoned remote rows and handles partial blocks" {
    const platform = zml.testing.env();
    if (!@import("../cute_mxfp4.zig").isAvailable(platform)) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const tokens = 3;
    const topk = 3;
    const hidden = 24;
    const routes = [_]i32{ 0, -1, 2, -1, -1, -1, 0, 1, -1 };
    var values: [tokens * topk * hidden]zml.floats.BFloat16 = undefined;
    for (&values, 0..) |*value, i| {
        value.* = .fromF32(if (routes[i / hidden] < 0) std.math.nan(f32) else @as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) / 8);
    }
    const Local = struct {
        fn call(x: zml.Tensor, ids: zml.Tensor) zml.Tensor {
            return combine(x, ids, .{ .tokens = tokens, .hidden = hidden, .topk = topk });
        }
    };
    const x: zml.Tensor = .init(.{ tokens * topk, hidden }, .bf16);
    const ids: zml.Tensor = .init(.{tokens * topk}, .i32);
    var exe = try platform.compileFn(allocator, io, Local.call, .{ x, ids }, .{});
    defer exe.deinit();
    var input = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
    defer input.deinit();
    var route_buffer = try zml.Buffer.fromBytes(io, platform, ids.shape(), std.mem.asBytes(&routes));
    defer route_buffer.deinit();
    var output = try exe.eval(allocator, io, .{ input, route_buffer });
    defer output.deinit();
    const actual = try output.toSliceAlloc(allocator, io);
    defer actual.free(allocator);
    for (actual.constItems(zml.floats.BFloat16), 0..) |value, i| {
        var sum: f32 = 0;
        for (0..topk) |slot| {
            const route = (i / hidden) * topk + slot;
            if (routes[route] >= 0) sum += values[route * hidden + i % hidden].toF32();
        }
        try std.testing.expectEqual(@as(u16, @bitCast(zml.floats.BFloat16.fromF32(sum))), @as(u16, @bitCast(value)));
    }
}
