const std = @import("std");

const fly_builder = @import("dsl/fly/builder");

const fly = @import("fly.zig");
const zml = @import("../../zml.zig");

/// C = A + B over (M, N) f32, 128-bit vectorized and predicated on the
/// ragged edge.
const FlyVectorAdd = struct {
    const Cfg = struct {
        thr: fly.Layout = fly.ordered(.{ 8, 16 }, .{ 1, 0 }),
        val: fly.Layout = fly.ordered(.{ 1, 4 }, .{ 0, 1 }),
    };

    const K = fly.Kernel(Cfg, .{
        .name = "vector_add",
        .inputs = &.{ "a", "b" },
        .outputs = &.{"c"},
        .run = run,
    });

    fn run(b: *fly.Builder, cfg: Cfg) fly.FinishError!void {
        const a = K.args(b);
        fly_builder.emitVectorAdd(b, cfg.thr, cfg.val, a.a, a.b, a.c);
    }
};

/// C = A @ B^T with one MFMA-tiled block.
const FlyTiledMma = struct {
    const Cfg = struct { flavor: fly.MmaFlavor };
    const K = fly.Kernel(Cfg, .{
        .name = "tiled_mma",
        .inputs = &.{ "a", "b" },
        .outputs = &.{"c"},
        .run = run,
    });

    fn run(b: *fly.Builder, cfg: Cfg) fly.FinishError!void {
        const a = K.args(b);
        fly_builder.emitTiledMma(b, cfg.flavor, a.a, a.b, a.c);
    }
};

test "fly kernel emits a module XLA can parse" {
    const shapes = [_]zml.Shape{ .init(.{ 100, 1000 }, .f32), .init(.{ 100, 1000 }, .f32), .init(.{ 100, 1000 }, .f32) };
    const ir = try FlyVectorAdd.K.emit(std.testing.allocator, .{}, &shapes);
    defer std.testing.allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "gpu.func @vector_add(%arg0: !fly.ptr<f32, global>, %arg1: !fly.ptr<f32, global>, %arg2: !fly.ptr<f32, global>) kernel") != null);
    try std.testing.expect(std.mem.indexOf(u8, ir, "fly.static : !fly.tile<[8|64]>") != null);
}

test "fly kernels run on rocm" {
    const platform = zml.testing.env();
    if (platform.target != .rocm) return error.SkipZigTest;

    const allocator = std.testing.allocator;

    // Compile `forward`, upload both inputs, run, and bring the result back.
    const call = struct {
        fn f(comptime forward: anytype, p: *const zml.Platform, ta: zml.Tensor, ha: anytype, tb: zml.Tensor, hb: anytype) !zml.Slice {
            var exe = try zml.module.compile(std.testing.allocator, std.testing.io, forward, .{ ta, tb }, p, .{});
            defer exe.deinit();
            var ba: zml.Buffer = try .fromBytes(std.testing.io, p, ta.shape(), std.mem.sliceAsBytes(ha));
            defer ba.deinit();
            var bb: zml.Buffer = try .fromBytes(std.testing.io, p, tb.shape(), std.mem.sliceAsBytes(hb));
            defer bb.deinit();
            var out = try exe.eval(std.testing.allocator, std.testing.io, .{ ba, bb });
            defer out.deinit();
            return out.toSliceAlloc(std.testing.allocator, std.testing.io);
        }
    }.f;

    // vector add: elementwise over a ragged (100, 1000), predicated at the edge.
    {
        const M, const N = .{ 100, 1000 };
        const Mod = struct {
            pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
                return FlyVectorAdd.K.call(.{ .a = a, .b = b }, .{ .c = a.shape() }, .{
                    .cfg = .{},
                    .grid = .{ (M + 8 - 1) / 8, (N + 64 - 1) / 64, 1 },
                    .threads = 8 * 16,
                }).c;
            }
        };

        const ha = try allocator.alloc(f32, M * N);
        defer allocator.free(ha);
        const hb = try allocator.alloc(f32, M * N);
        defer allocator.free(hb);
        for (ha, hb, 0..) |*va, *vb, i| {
            va.* = @floatFromInt(i % 977);
            vb.* = @floatFromInt(1000 + (i % 313));
        }

        var host = try call(Mod.forward, platform, .init(.{ M, N }, .f32), ha, .init(.{ M, N }, .f32), hb);
        defer host.free(allocator);
        for (host.items(f32), 0..) |v, i| try std.testing.expectEqual(ha[i] + hb[i], v);
    }

    // tiled mma: C = A @ B^T in one block of 256 threads, over whatever matrix
    // atom this device has. CDNA multiplies f32 through MFMA; RDNA3+ has WMMA,
    // whose verifier rejects f32 operands, so the operand type and K come from
    // the flavour rather than being written into the test.
    if (fly.mmaFlavor(platform)) |flavor| switch (flavor) {
        inline else => |f| {
            const Elem = switch (comptime f.operandDType()) {
                .f32 => f32,
                .f16 => f16,
                else => @compileError("unhandled MMA operand type"),
            };
            const M, const N = .{ 64, 64 };
            const Kd = comptime f.blockK();
            const operand_dtype: zml.DataType = switch (comptime f.operandDType()) {
                .f32 => .f32,
                .f16 => .f16,
                else => unreachable,
            };

            const Mod = struct {
                pub fn forward(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
                    return FlyTiledMma.K.call(.{ .a = a, .b = b }, .{ .c = zml.Shape.init(.{ M, N }, .f32) }, .{
                        .cfg = .{ .flavor = f },
                        .grid = .{ 1, 1, 1 },
                        .threads = comptime f.blockThreads(),
                    }).c;
                }
            };

            var ha: [M * Kd]Elem = undefined;
            var hb: [N * Kd]Elem = undefined;
            for (&ha, 0..) |*v, i| v.* = @as(Elem, @floatFromInt((i * 7) % 13)) - 6;
            for (&hb, 0..) |*v, i| v.* = @as(Elem, @floatFromInt((i * 5) % 11)) - 5;

            var host = try call(Mod.forward, platform, .init(.{ M, Kd }, operand_dtype), &ha, .init(.{ N, Kd }, operand_dtype), &hb);
            defer host.free(allocator);

            // f16 operands accumulate in f32, so the error is set by the input
            // rounding rather than the sum.
            const tol: f32 = if (Elem == f16) 1e-2 else 1e-3;
            const out = host.items(f32);
            for (0..M) |m| for (0..N) |n| {
                var acc: f32 = 0;
                for (0..Kd) |k| acc += @as(f32, ha[m * Kd + k]) * @as(f32, hb[n * Kd + k]);
                try std.testing.expectApproxEqAbs(acc, out[m * N + n], tol);
            };
        },
    };
}
