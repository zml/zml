const reference = @import("reference.zig");
const std = @import("std");
const zml = @import("zml");
const cases = @import("cases.zig");
const fixtures = @import("fixtures.zig");
const Inputs = fixtures.Inputs;

test "CPU reference self checks" {
    _ = @import("reference.zig");
}

test "paged attention correctness" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;

    const a = std.testing.allocator;
    const io = std.testing.io;

    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.UnknownComputeCapability;
    std.debug.print("PAGED_ATTENTION dtype=bf16 page_size=16 seed=42 cc={d}.{d}\n{f}\n", .{
        cc.major, cc.minor, platform.fmtVerbose(),
    });

    var failures: usize = 0;
    for (cases.correctness) |c| {
        std.debug.print("CASE name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} mask={s} backend=cute\n", .{
            c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, @tagName(c.mask),
        });

        runCorrectness(a, io, platform, c, .cute, .bf16) catch |err| {
            failures += 1;
            std.debug.print("FAIL error={s}\n", .{@errorName(err)});
        };
    }

    try std.testing.expectEqual(@as(usize, 0), failures);
}

test "FP16 paged attention" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;
    for ([_]cases.Case{
        .{ .name = "fp16_decode", .head_dim = 64, .kv_len = 257 },
        .{ .name = "fp16_mixed", .prefill = 1, .mask = .non_causal },
        .{ .name = "fp16_window_sink", .head_dim = 256, .prefill = 1, .mask = .window, .sink = true },
        .{ .name = "fp16_split_sink", .head_dim = 512, .batch = 1, .kv_len = 2049, .sink = true, .shuffle_pages = true },
    }) |c| try runCorrectness(std.testing.allocator, std.testing.io, platform, c, .cute, .f16);
}

test "indexed paged attention BF16" {
    try runSparseCorrectness(.bf16);
}

test "indexed paged attention FP16" {
    try runSparseCorrectness(.f16);
}

test "indexed long selections handle sinks tails and inactive capacity" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;
    const suite = [_]cases.Case{
        .{ .name = "sparse_wide_sink_tail", .head_dim = 512, .batch = 1, .kv_heads = 1, .group = 16, .kv_len = 2049, .mask = .indices, .topk = 1025, .sink = true, .shuffle_pages = true },
        .{ .name = "sparse_wide_empty_capacity", .head_dim = 512, .batch = 3, .kv_heads = 1, .group = 5, .inactive = 1, .token_capacity = 6, .mask = .indices, .topk = 513, .empty_indices = true, .sink = true, .sink_value = -std.math.inf(f32) },
        .{ .name = "sparse_wide_prefill", .head_dim = 512, .group = 3, .prefill = 1, .chunk = 3, .kv_len = 2049, .mask = .indices, .topk = 513, .sink = true, .shuffle_pages = true },
    };
    for ([_]zml.DataType{ .bf16, .f16 }) |dtype| for (suite) |c| for (fixtures.backends(c)) |backend| {
        std.debug.print("CASE name={s} backend={s} compute_dtype={s}\n", .{ c.name, @tagName(backend), @tagName(dtype) });
        try runCorrectness(std.testing.allocator, std.testing.io, platform, c, backend, dtype);
    };
}

test "indexed attention handles tails and packed capacity" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;
    const suite = [_]cases.Case{
        .{ .name = "sparse_eight_head_mixed_tail", .head_dim = 512, .batch = 3, .kv_heads = 1, .group = 19, .prefill = 1, .inactive = 1, .chunk = 3, .token_capacity = 9, .kv_len = 2049, .mask = .indices, .topk = 513, .sink = true, .shuffle_pages = true },
        .{ .name = "sparse_eight_head_small_tail", .head_dim = 64, .batch = 4, .group = 5, .prefill = 1, .inactive = 1, .chunk = 3, .token_capacity = 9, .mask = .indices, .topk = 65, .sink = true, .shuffle_pages = true },
        .{ .name = "sparse_eight_head_empty_capacity", .head_dim = 512, .batch = 3, .kv_heads = 1, .group = 16, .inactive = 1, .token_capacity = 6, .mask = .indices, .topk = 513, .empty_indices = true, .sink = true, .sink_value = -std.math.inf(f32) },
    };
    for ([_]zml.DataType{ .bf16, .f16 }) |dtype| for (suite) |c| {
        std.debug.print("CASE name={s} backend=cute compute_dtype={s}\n", .{ c.name, @tagName(dtype) });
        try runCorrectness(std.testing.allocator, std.testing.io, platform, c, .cute, dtype);
    };
}

fn runSparseCorrectness(dtype: zml.DataType) !void {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.CudaRequired;
    var failures: usize = 0;
    for (cases.sparseCorrectness) |c| {
        for (fixtures.backends(c)) |backend| {
            std.debug.print("CASE name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} backend={s} compute_dtype={s}\n", .{
                c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(backend), @tagName(dtype),
            });
            runCorrectness(std.testing.allocator, std.testing.io, platform, c, backend, dtype) catch |err| {
                failures += 1;
                std.debug.print("FAIL error={s}\n", .{@errorName(err)});
            };
        }
    }
    try std.testing.expectEqual(@as(usize, 0), failures);
}

fn runCorrectness(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, c: cases.Case, backend: fixtures.Backend, dtype: zml.DataType) !void {
    const inputs = Inputs.initDType(c, dtype);
    const exe = try platform.compileFn(a, io, fixtures.forwardBackend, .{ inputs, c, backend }, .{ .program_name = "paged_attention_harness" });
    defer exe.deinit();

    var runner = try exe.runner(a);
    defer runner.deinit(a);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const fixture = try fixtures.Fixture.init(arena.allocator(), c);
    if (dtype == .f16) fixture.quantizeF16();

    var buffers = try fixtures.upload(a, io, platform, inputs, fixture);
    defer zml.Buffer.deinitAll(Inputs, &buffers);

    var output: zml.Buffer = undefined;
    runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
    defer output.deinit();

    var host = try output.toSliceAlloc(a, io);
    defer host.free(a);

    const max_error = if (dtype == .f16)
        try reference.check(a, c, fixture, host.items(f16))
    else
        try reference.check(a, c, fixture, host.items(zml.floats.BFloat16));
    if (dtype == .f16) {
        try reference.checkZeroTail(c, fixture, host.items(f16));
    } else {
        try reference.checkZeroTail(c, fixture, host.items(zml.floats.BFloat16));
    }
    std.debug.print("RESULT name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} topk={d} backend={s} compute_dtype={s} check=all_active_outputs inactive_tail=zero_checked max_abs={d:.6}\n", .{
        c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, c.topk, @tagName(backend), @tagName(dtype), max_error,
    });
}
