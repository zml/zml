const std = @import("std");

const zml = @import("../../zml.zig");
const tcl = @import("tcl.zig");

/// `furiosa.tcl.examples.softmax` over f32.
const TclSoftmax = struct {
    const Cfg = struct { b: i64, s: i64, d: i64 };
    const K = tcl.Kernel(Cfg, .{
        .name = "softmax_kernel",
        .inputs = &.{"x"},
        .outputs = &.{"y"},
        .run = run,
    });

    fn run(b: *tcl.Builder, cfg: Cfg) tcl.FinishError!void {
        const B = b.axis("B", cfg.b);
        const S = b.axis("S", cfg.s);
        const D = b.axis("D", cfg.d);
        const t = try b.declareArgs(.{ .x = .{ .dtype = .f32, .axes = &.{ B, S, D } } });

        const op = b.tensorOperation(.{});
        const x = op.fetch(t.x, .{});
        const e = x.subf(x.reduce(&.{D}, .maxf)).exp();
        b.ret(&.{try op.commit(e.divf(e.reduce(&.{D}, .addf)), .{})});
    }
};

test "tcl kernels run on furiosa" {
    const allocator = std.testing.allocator;
    const cfg: TclSoftmax.Cfg = .{ .b = 2, .s = 16, .d = 256 };

    const ir = try TclSoftmax.K.emit(allocator, cfg);
    defer allocator.free(ir);
    try std.testing.expect(std.mem.indexOf(u8, ir, "func.func @softmax_kernel(") != null);

    const platform = zml.testing.env();
    if (platform.target != .furiosa) return error.SkipZigTest;

    const Mod = struct {
        pub fn forward(x: zml.Tensor) zml.Tensor {
            return TclSoftmax.K.call(.{ .x = x }, .{ .y = x.shape() }, .{ .cfg = cfg }).y;
        }
    };
    const x: zml.Tensor = .init(.{ cfg.b, cfg.s, cfg.d }, .f32);
    var exe = try zml.module.compile(allocator, std.testing.io, Mod.forward, .{x}, platform, .{});
    defer exe.deinit();

    var input: [cfg.b * cfg.s * cfg.d]f32 = undefined;
    for (&input, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 37)) * 0.25 - 4;
    var x_buffer: zml.Buffer = try .fromBytes(std.testing.io, platform, x.shape(), std.mem.sliceAsBytes(&input));
    defer x_buffer.deinit();
    var result = try exe.eval(allocator, std.testing.io, .{x_buffer});
    defer result.deinit();
    var host = try result.toSliceAlloc(allocator, std.testing.io);
    defer host.free(allocator);

    const out = host.items(f32);
    var row: usize = 0;
    while (row < cfg.b * cfg.s) : (row += 1) {
        const in = input[row * cfg.d ..][0..cfg.d];
        var max: f32 = in[0];
        for (in) |v| max = @max(max, v);
        var sum: f32 = 0;
        for (in) |v| sum += @exp(v - max);
        for (in, out[row * cfg.d ..][0..cfg.d]) |v, y| try std.testing.expectApproxEqAbs(@exp(v - max) / sum, y, 1e-4);
    }
}

/// exp over a tensor split across two chips in one launch, returning the
/// split result and its all_gather.
const TclSplitExp = struct {
    const Cfg = struct {};
    const K = tcl.Kernel(Cfg, .{ .name = "split_exp", .inputs = &.{"x"}, .outputs = &.{ "y", "all" }, .run = run });
    fn run(b: *tcl.Builder, _: Cfg) tcl.FinishError!void {
        const C = b.axis("C", 2);
        const A = b.axis("A", 1024);
        const t = try b.declareArgs(.{ .x = .{ .dtype = .f32, .axes = &.{ C, A }, .dram = tcl.Dram{ .chip = &.{.{ .axis = C }} } } });
        const op = b.tensorOperation(.{ .context = .{ .layout = .{ .chip = .{ .axis = C } } } });
        const local = try op.commit(op.fetch(t.x, .{}).exp(), .{});
        const all = try b.allGather(local, C, .{});
        b.ret(&.{ try b.asDram(local, .{ .chip = &.{.{ .axis = C }} }), try b.asDram(all, .{}) });
    }
    fn body(ctx: struct { x: zml.Tensor }, shapes: [2]zml.Shape) [2]zml.Tensor {
        const r = K.call(.{ .x = ctx.x }, .{ .y = shapes[0], .all = shapes[1] }, .{ .cfg = .{} });
        return .{ r.y, r.all };
    }
};

// Needs FURIOSA_VISIBLE_DEVICES=0,1.
test "tcl multi-chip kernels run on two furiosa chips" {
    const allocator = std.testing.allocator;
    const platform = zml.testing.env();
    if (platform.target != .furiosa or platform.devices.len != 2) return error.SkipZigTest;
    const tp = zml.testing.tpMesh();
    const x_shape = zml.Shape.init(.{ .c = 2, .a = 1024 }, .f32).withPartitioning(tp, .{ .c = .model });
    const Mod = struct {
        pub fn forward(x: zml.Tensor, mesh: *const zml.Sharding.Mesh) [2]zml.Tensor {
            const r = zml.ops.manualComputation(TclSplitExp.body, .{ .x = x }, .{ x.shape(), zml.Shape.init(.{ .c = 2, .a = 1024 }, .f32) }, mesh, .{
                .in_specs = .{ .x = .parse(.{ .c = .model }) },
                .out_specs = .{ .parse(.{ .c = .model }), .replicated },
            });
            return .{ r[0], r[1] };
        }
    };
    const x: zml.Tensor = .fromShape(x_shape);
    var exe = try zml.module.compile(allocator, std.testing.io, Mod.forward, .{ x, tp }, platform, .{ .meshes = &.{tp} });
    defer exe.deinit();
    var hx: [2 * 1024]f32 = undefined;
    for (&hx, 0..) |*v, i| v.* = @as(f32, @floatFromInt(i % 13)) * 0.125 - 0.5;
    var bx: zml.Buffer = try .fromBytes(std.testing.io, platform, x_shape, std.mem.sliceAsBytes(&hx));
    defer bx.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var res: [2]zml.Buffer = undefined;
    runner.run(std.testing.io, .{bx}, .{ &res[0], &res[1] }, .{ .wait = true });
    defer for (&res) |*r| r.deinit();
    for (res) |r| {
        var host = try r.toSliceAlloc(allocator, std.testing.io);
        defer host.free(allocator);
        for (host.items(f32), 0..) |v, i| try std.testing.expectApproxEqRel(@exp(hx[i]), v, 1e-4);
    }
}
