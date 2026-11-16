const std = @import("std");

const cuda_tile = @import("cuda_tile.zig");

test "cuda_tile kernel emits a module XLA can parse" {
    const Cfg = struct { n: i64 };
    const AddOne = cuda_tile.Kernel(Cfg, .{
        .name = "add_one",
        .inputs = &.{"x"},
        .outputs = &.{"out"},
        .run = struct {
            fn run(b: *cuda_tile.Builder, cfg: Cfg) cuda_tile.FinishError!void {
                const a = try b.declareArgs(.{ .x = .{ .ptr = .f32 }, .out = .{ .ptr = .f32 } });
                const vx = b.partitionView(b.tensorView(a.x, &.{cfg.n}, &.{1}), &.{128}, .{});
                const vo = b.partitionView(b.tensorView(a.out, &.{cfg.n}, &.{1}), &.{128}, .{});
                const bid = b.tileBlockId();
                _ = b.store(b.load(vx, &.{bid.x}).add(1.0), vo, &.{bid.x});
            }
        }.run,
    });

    const ir = try AddOne.emit(std.testing.allocator, .{ .n = 1024 });
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "cuda_tile.module @add_one") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "entry @add_one(") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "load_view_tko") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "store_view_tko") != null);
}
