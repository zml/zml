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

        runCorrectness(a, io, platform, c) catch |err| {
            failures += 1;
            std.debug.print("FAIL error={s}\n", .{@errorName(err)});
        };
    }

    try std.testing.expectEqual(@as(usize, 0), failures);
}

fn runCorrectness(a: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, c: cases.Case) !void {
    const exe = try platform.compileFn(a, io, fixtures.forward, .{ Inputs.init(c), c, zml.paged_attention.Backend.cute }, .{ .program_name = "paged_attention_harness" });
    defer exe.deinit();

    var runner = try exe.runner(a);
    defer runner.deinit(a);

    var arena = std.heap.ArenaAllocator.init(a);
    defer arena.deinit();

    const fixture = try fixtures.Fixture.init(arena.allocator(), c);

    var buffers = try fixtures.upload(io, platform, Inputs.init(c), fixture);
    defer zml.Buffer.deinitAll(Inputs, &buffers);

    var output: zml.Buffer = undefined;
    runner.run(io, .{buffers}, .{&output}, .{ .wait = true });
    defer output.deinit();

    var host = try output.toSliceAlloc(a, io);
    defer host.free(a);

    const max_error = try reference.check(a, c, fixture, host.items(zml.floats.BFloat16));
    std.debug.print("RESULT name={s} d={d} b={d} hkv={d} g={d} prefill={d} chunk={d} kv={d} backend=cute check=all_outputs max_abs={d:.6}\n", .{
        c.name, c.head_dim, c.batch, c.kv_heads, c.group, c.prefill, if (c.prefill > 0) c.chunk else 1, c.kv_len, max_error,
    });
}
