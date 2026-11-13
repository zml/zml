const std = @import("std");

const cute = @import("cute.zig");

test "cute kernel emits the module the CuTe compiler takes" {
    const Cfg = struct { n: i64 };
    const AddOne = cute.Kernel(Cfg, .{
        .name = "add_one",
        .inputs = &.{"x"},
        .outputs = &.{"out"},
        .run = struct {
            fn run(b: *cute.Builder, cfg: Cfg) cute.FinishError!void {
                const a = try b.declareArgs(.{
                    .x = .{ .tensor = .{ .dtype = .f32, .shape = &.{cfg.n} } },
                    .out = .{ .tensor = .{ .dtype = .f32, .shape = &.{cfg.n} } },
                });
                const i = b.blockIdx().x.mul(b.blockDim().x).add(b.threadIdx().x);
                var guard = b.openIf(i.lt(cfg.n));
                {
                    a.out.set(.{i}, a.x.get(.{i}).add(1.0));
                    guard.close();
                }
            }
        }.run,
    });

    const ir = try AddOne.emit(std.testing.allocator, .{ .n = 1000 }, .{ 128, 1, 1 });
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "\"func.func\"() <{function_type = (") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "sym_name = \"add_one\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "nvvm.reqntid = array<i32: 128, 1, 1>") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "cute.memref.load") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "gpu.module") == null);
}
