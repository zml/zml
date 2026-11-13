const std = @import("std");

const metal = @import("metal.zig");
const zml = @import("../../core.zig");

/// `out[i] = a[i] * scale + b[i]` over `n` floats.
const MetalAxpy = struct {
    const Cfg = struct {};
    const K = metal.Kernel(Cfg, .{
        .name = "axpy",
        .inputs = &.{ "a", "b" },
        .outputs = &.{"out"},
        .constants = &.{ "scale", "n" },
        .run = run,
    });

    fn run(b: *metal.Builder, _: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .a = .{ .buffer = .{ .type = .float, .const_ = true } },
            .b = .{ .buffer = .{ .type = .float, .const_ = true } },
            .out = .{ .buffer = .{ .type = .float } },
            .scale = .{ .constant = .float },
            .n = .{ .constant = .uint },
            .tid = .{ .thread_position_in_grid = .uint },
        });
        b.returnIf(a.tid.ge(a.n));
        a.out.store(a.tid, b.fma(a.a.load(a.tid), a.scale, a.b.load(a.tid)));
    }
};

test "metal kernels run on metal" {
    const allocator = std.testing.allocator;

    const ir = try MetalAxpy.K.emit(allocator, .{});
    defer allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "air.fma") != null);

    const platform = zml.testing.env();
    if (platform.target != .metal) return error.SkipZigTest;

    const n = 1000;
    const Mod = struct {
        pub fn forward(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
            return MetalAxpy.K.call(.{ .a = x, .b = y }, .{ .out = x.shape() }, .{
                .cfg = .{},
                .grid = .{ (n + 255) / 256, 1, 1 },
                .threadgroup = .{ 256, 1, 1 },
                .constants = .{ .scale = .of(@as(f32, 0.5)), .n = .of(@as(u32, n)) },
            }).out;
        }
    };
    const x: zml.Tensor = .init(.{n}, .f32);
    var exe = try zml.module.compile(allocator, std.testing.io, Mod.forward, .{ x, zml.Tensor.init(.{n}, .f32) }, platform, .{});
    defer exe.deinit();

    var hx: [n]f32 = undefined;
    var hy: [n]f32 = undefined;
    for (&hx, &hy, 0..) |*a, *b, i| {
        a.* = @floatFromInt(i);
        b.* = 1;
    }
    var bx: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&hx));
    defer bx.deinit();
    var by: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&hy));
    defer by.deinit();
    var result = try exe.eval(allocator, std.testing.io, .{ bx, by });
    defer result.deinit();
    var host = try result.toSliceAlloc(allocator, std.testing.io);
    defer host.free(allocator);
    for (host.items(f32), hx, hy) |out, a, b| try std.testing.expectEqual(a * 0.5 + b, out);
}
