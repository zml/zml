const std = @import("std");
const zml = @import("zml");

// Run separately from the generic suite: PE topology is fixed per process.
// XLA_FURIOSA_VISIBLE_DEVICES=0,1 also exercises two-chip sharding.
test "eight PE writer preserves pinned placement" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .furiosa, .{ .furiosa = .{ .pe_count = 8 } });
    defer platform.deinit(allocator, io);
    const shape = zml.Shape.init(.{32}, .f32);
    const values: [32]f32 = @splat(3.25);
    var input: zml.Buffer = undefined;
    var writer = try zml.io.MemoryWriter.init(allocator, io, platform, &.{}, &.{}, 0, shape, .replicated, &input, .host_pinned);
    defer writer.deinit(allocator);
    try writer.interface().writeAll(std.mem.sliceAsBytes(&values));
    try writer.interface().flush();
    defer input.deinit();
    for (input._shards.constSlice()) |shard| {
        try std.testing.expectEqual(.host_pinned, shard.memory(platform.pjrt_api).kind(platform.pjrt_api));
    }
    var result = try input.toSliceAlloc(allocator, io);
    defer result.free(allocator);
    try std.testing.expectEqualSlices(f32, &values, result.items(f32));
}

test "eight PE sharded execution preserves input ownership" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .furiosa, .{ .furiosa = .{ .pe_count = 8 } });
    defer platform.deinit(allocator, io);
    const sharding = try platform.registerSharding("placement_mesh", .mesh(.{ .m = .low_bandwidth }));
    const shape = zml.Shape.init(.{ .m = 32 }, .f32).withPartitioning(.{ .m = .m });
    const values: [32]f32 = @splat(3.25);
    const expected: [32]f32 = @splat(-3.25);
    var input = try zml.Buffer.fromBytes(io, platform, shape, sharding, std.mem.sliceAsBytes(&values));
    defer input.deinit();
    const Graph = struct {
        fn run(x: zml.Tensor) zml.Tensor {
            return x.negate();
        }
    };
    const exe = try platform.compileFn(allocator, io, Graph.run, .{zml.Tensor.fromShape(shape)}, .{ .shardings = &.{sharding} });
    defer exe.deinit();
    for (0..2) |_| {
        var output = try zml.testing.autoCall(allocator, io, &exe, Graph.run, .{input});
        defer output.deinit();
        var result = try output.toSliceAlloc(allocator, io);
        defer result.free(allocator);
        try std.testing.expectEqualSlices(f32, &expected, result.items(f32));
    }
    var original = try input.toSliceAlloc(allocator, io);
    defer original.free(allocator);
    try std.testing.expectEqualSlices(f32, &values, original.items(f32));
}
