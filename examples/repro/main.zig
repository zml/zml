const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;
const config = @import("config");


    // log.info("config.DefinedEmpty: {}", .{config.DefinedEmpty});
    // if (comptime config.DefinedEmpty) {
    //     const EmptyStruct = struct {};
    //     const map_buffer: zml.meta.MapRestrict(zml.Tensor, EmptyStruct).map(zml.nn.Linear) = .{ .weight = undefined, .bias = undefined };
    //     zml.meta.visit(struct {
    //         fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
    //             _ = b; // autofix
    //             ctx_.index += 1;
    //         }
    //     }.call, &ctx, &map_buffer);
    // } else {
    //     const map_buffer: zml.meta.MapRestrict(zml.Tensor, zml.Buffer).map(zml.nn.Linear) = .{ .weight = undefined, .bias = @as(?zml.Buffer, undefined) };
    //     zml.meta.visit(struct {
    //         fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
    //             _ = b; // autofix
    //             ctx_.index += 1;
    //         }
    //     }.call, &ctx, &map_buffer);
    // }


pub fn main(init: std.process.Init) !void {
    _ = init; // autofix

    const bufferized: zml.Bufferized(zml.nn.Linear) = .{ .weight = undefined, .bias = @as(zml.Buffer, undefined) };
    _ = bufferized; // autofix

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
    log.info("config.DefinedEmpty: {}", .{config.DefinedEmpty});
    if (comptime config.DefinedEmpty) {
        const EmptyStruct = struct {};
        const map_buffer: zml.meta.MapRestrict(zml.Tensor, EmptyStruct).map(zml.nn.Linear) = .{ .weight = undefined, .bias = undefined };
        zml.meta.visit(struct {
            fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
                _ = b; // autofix
                ctx_.index += 1;
            }
        }.call, &ctx, &map_buffer);
    } else {
        const map_buffer: zml.meta.MapRestrict(zml.Tensor, zml.Buffer).map(zml.nn.Linear) = .{ .weight = undefined, .bias = @as(?zml.Buffer, undefined) };
        zml.meta.visit(struct {
            fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
                _ = b; // autofix
                ctx_.index += 1;
            }
        }.call, &ctx, &map_buffer);
    }

    //zml.meta.visit(struct {
    //    fn call(ctx_: *LocalContext, b: *const zml.Buffer) void {
    //        _ = b; // autofix
    //        ctx_.index += 1;
    //    }
    //}.call, &ctx, &bufferized_manual);
}
