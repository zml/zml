const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const Cul = struct {};

pub fn main(init: std.process.Init) !void {
    _ = init; // autofix

    const bufferized: zml.Bufferized(zml.nn.Linear) = .{ .weight = undefined, .bias = @as(zml.Buffer, undefined) };
    _ = bufferized; // autofix
    const cul: zml.meta.MapRestrict(zml.Tensor, zml.Buffer).map(zml.nn.Linear) = .{ .weight = undefined, .bias = @as(?zml.Buffer, undefined) };

    //const bufferized_manual: struct { weight: zml.Buffer, bias: ?zml.Buffer } = .{ .weight = undefined, .bias = @as(zml.Buffer, undefined) };

    const LocalContext = struct {
        index: usize = 0,
    };
    var ctx: LocalContext = .{};
    _ = &ctx; // autofix
    //zml.meta.visit(struct {
    //    fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
    //        _ = b; // autofix
    //        ctx_.index += 1;
    //    }
    //}.call, &ctx, &bufferized);
    zml.meta.visit(struct {
        fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
            _ = b; // autofix
            ctx_.index += 1;
        }
    }.call, &ctx, &cul);

    //zml.meta.visit(struct {
    //    fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
    //        _ = b; // autofix
    //        ctx_.index += 1;
    //    }
    //}.call, &ctx, &bufferized_manual);
}
