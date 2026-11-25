//! Runs one matmul2d on a simdgroup and checks the per-lane element layout the paged attention kernel assumes.

const std = @import("std");

const zml = @import("zml/core");
const metal = zml.kernel.metal;

const Builder = metal.Builder;
const paged_attn = @import("paged_attn.zig");

const i32Of = paged_attn.i32Of;

const Probe = struct {
    const Cfg = struct {};
    const K = metal.Kernel(Cfg, .{
        .name = "nax_probe",
        .inputs = &.{ "a", "b" },
        .outputs = &.{"out"},
        .constants = &.{"unused"},
        .run = run,
    });

    fn run(b: *Builder, _: Cfg) metal.FinishError!void {
        const a = try b.declareArgs(.{
            .a = .{ .buffer = .{ .type = .half, .const_ = true } },
            .b = .{ .buffer = .{ .type = .half, .const_ = true } },
            .out = .{ .buffer = .{ .type = .float } },
            .unused = .{ .constant = .uint },
            .lane = .{ .thread_index_in_simdgroup = .uint },
        });
        const lane = a.lane.to(.int);
        const fragment = paged_attn.tensorFragmentCoord(lane);
        const fm = fragment.row;
        const fn_ = fragment.column;

        const op: Builder.Matmul2d = .{ .m = 16, .n = 32, .k = 16, .transpose_left = false, .transpose_right = true, .left = .half, .right = .half, .destination = .float };
        const left = b.cooperativeTensor(op, .left);
        const right = b.cooperativeTensor(op, .right);
        const dest = b.cooperativeTensor(op, .destination);

        inline for (0..2) |rh| {
            inline for (0..4) |j| {
                const row = fm.add(i32Of(rh * 8));
                left.element(rh * 4 + j).store(0, a.a.load(row.mul(16).add(fn_).add(i32Of(j))));
            }
        }
        inline for (0..2) |blk| {
            inline for (0..2) |rh| {
                inline for (0..4) |j| {
                    const key = fm.add(i32Of(blk * 16 + rh * 8));
                    right.element(blk * 8 + rh * 4 + j).store(0, a.b.load(key.mul(16).add(fn_).add(i32Of(j))));
                }
            }
        }
        inline for (0..16) |e| {
            dest.element(e).store(0, 0.0);
        }
        b.matmul2dRun(op, left, right, dest);
        inline for (0..16) |e| {
            a.out.store(lane.mul(16).add(i32Of(e)), dest.element(e).load(0));
        }
    }
};

test "matmul2d element layout" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    if (platform.target != .metal) return error.SkipZigTest;
    if (!paged_attn.tensorOpsSupported(platform.devices[0].pjrt_desc.kind(platform.pjrt_api))) return error.SkipZigTest;

    const Mod = struct {
        pub fn forward(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
            return Probe.K.call(.{ .a = x, .b = y }, .{ .out = zml.Shape.init(.{32 * 16}, .f32) }, .{
                .cfg = .{},
                .grid = .{ 1, 1, 1 },
                .threadgroup = .{ 32, 1, 1 },
                .constants = .{ .unused = .of(@as(u32, 0)) },
            }).out;
        }
    };
    const a_shape = zml.Shape.init(.{16 * 16}, .f16);
    const b_shape = zml.Shape.init(.{32 * 16}, .f16);
    var exe = try zml.module.compile(allocator, std.testing.io, Mod.forward, .{ zml.Tensor.fromShape(a_shape), zml.Tensor.fromShape(b_shape) }, platform, .{});
    defer exe.deinit();

    var host_a: [16 * 16]f16 = undefined;
    var host_b: [32 * 16]f16 = undefined;
    for (&host_a, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 7) % 11)) - 5);
    for (&host_b, 0..) |*v, i| v.* = @floatFromInt(@as(i32, @intCast((i * 5) % 13)) - 6);
    var buffer_a: zml.Buffer = try .fromBytes(std.testing.io, platform, a_shape, std.mem.sliceAsBytes(&host_a));
    defer buffer_a.deinit();
    var buffer_b: zml.Buffer = try .fromBytes(std.testing.io, platform, b_shape, std.mem.sliceAsBytes(&host_b));
    defer buffer_b.deinit();
    var result = try exe.eval(allocator, std.testing.io, .{ buffer_a, buffer_b });
    defer result.deinit();
    var host = try result.toSliceAlloc(allocator, std.testing.io);
    defer host.free(allocator);

    var mismatches: usize = 0;
    for (0..32) |lane| {
        const quad = lane / 4;
        const fm = (quad & 4) | ((lane >> 1) & 3);
        const fn_ = ((quad & 2) | (lane & 1)) * 4;
        for (0..16) |e| {
            const blk = e / 8;
            const i = e % 8;
            const row = fm + (i / 4) * 8;
            const col = blk * 16 + fn_ + (i % 4);
            var expected: f32 = 0;
            for (0..16) |k| {
                expected += @as(f32, @floatCast(host_a[row * 16 + k])) * @as(f32, @floatCast(host_b[col * 16 + k]));
            }
            const got = host.items(f32)[lane * 16 + e];
            if (got != expected) {
                mismatches += 1;
                if (mismatches <= 12) std.debug.print("lane {d} element {d}: got {d} expected {d} (row {d} col {d})\n", .{ lane, e, got, expected, row, col });
            }
        }
    }
    try std.testing.expectEqual(0, mismatches);
}
