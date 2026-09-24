const std = @import("std");
const zml = @import("zml");

// Exercise the exact attention entry point used by Llama, with deterministic
// host inputs so the test does not depend on accelerator random-number support.
pub fn main(init: std.process.Init) !void {
    const args = zml.stdx.flags.parse(init.minimal.args, struct {
        dtype: ?zml.DataType = null,
        pub const help = "Compare Llama vanilla SDPA with CPU; optionally --dtype=f32 or --dtype=bf16";
    });
    if (args.dtype) |dtype| {
        if (dtype != .f32 and dtype != .bf16) return error.UnsupportedTestDtype;
    }
    const allocator = init.gpa;
    const io = init.io;
    const platform = try zml.Platform.init(allocator, io, .furiosa, .{});
    defer platform.deinit(allocator, io);
    const reference = try zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } });
    defer reference.deinit(allocator, io);

    if (zml.attention.Backend.auto(platform) != .vanilla) return error.ExpectedVanillaAttention;
    for ([_]zml.DataType{ .f32, .bf16 }) |dtype| {
        if (args.dtype != null and args.dtype.? != dtype) continue;
        // Prefill, a chunk at a nonzero cache offset, and single-token decode.
        for ([_]struct { queries: i64, offset: u32 }{
            .{ .queries = 8, .offset = 0 },
            .{ .queries = 4, .offset = 7 },
            .{ .queries = 1, .offset = 15 },
        }) |case| {
            try run(allocator, io, platform, reference, dtype, case.queries, case.offset);
        }
    }
}

fn vanilla(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, offset: zml.Tensor) zml.Tensor {
    return zml.attention.attention(q, k, v, offset, .vanilla, .vanilla);
}

fn input(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, shape: zml.Shape, seed: usize) !zml.Buffer {
    const values = try allocator.alloc(f32, @intCast(shape.count()));
    defer allocator.free(values);
    for (values, 0..) |*value, i| {
        value.* = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + seed * 13) % 113)) - 56)) / 64.0;
    }
    if (shape.dtype() == .f32) return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(values));
    const bits = try allocator.alloc(u16, values.len);
    defer allocator.free(bits);
    // All inputs above are exactly representable in BF16.
    for (values, bits) |value, *bit| bit.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(bits));
}

fn execute(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, dtype: zml.DataType, queries: i64, offset: u32) !zml.Buffer {
    const q_shape = zml.Shape.init(.{ .q = queries, .h = 4, .hd = 32 }, dtype);
    const kv_shape = zml.Shape.init(.{ .k = 16, .h = 2, .hd = 32 }, dtype);
    const offset_shape = zml.Shape.init(.{}, .u32);
    const exe = try platform.compileFn(allocator, io, vanilla, .{
        zml.Tensor.fromShape(q_shape),  zml.Tensor.fromShape(kv_shape),
        zml.Tensor.fromShape(kv_shape), zml.Tensor.fromShape(offset_shape),
    }, .{ .program_name = "llama_vanilla_attention", .shardings = &.{platform.replicated_sharding} });
    defer exe.deinit();
    var q = try input(allocator, io, platform, q_shape, 1);
    defer q.deinit();
    var k = try input(allocator, io, platform, kv_shape, 2);
    defer k.deinit();
    var v = try input(allocator, io, platform, kv_shape, 3);
    defer v.deinit();
    var index = try zml.Buffer.fromBytes(io, platform, offset_shape, .replicated, std.mem.asBytes(&offset));
    defer index.deinit();
    return zml.testing.autoCall(allocator, io, &exe, vanilla, .{ q, k, v, index });
}

fn run(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, reference: *const zml.Platform, dtype: zml.DataType, queries: i64, offset: u32) !void {
    std.log.info("Llama vanilla SDPA: {t}, queries={}, offset={}", .{ dtype, queries, offset });
    var expected = try execute(allocator, io, reference, dtype, queries, offset);
    defer expected.deinit();
    var actual = try execute(allocator, io, platform, dtype, queries, offset);
    defer actual.deinit();
    try zml.testing.expectClose(io, actual, expected, .{
        .absolute_tolerance = if (dtype == .bf16) 4e-3 else 1e-4,
        .relative_tolerance = if (dtype == .bf16) 2e-2 else 1e-3,
        .minimum_close_fraction = 1.0,
    });
    std.log.info("PASS: Llama vanilla SDPA {t}, queries={}, offset={}", .{ dtype, queries, offset });
}
