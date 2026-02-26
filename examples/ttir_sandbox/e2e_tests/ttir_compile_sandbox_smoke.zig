const std = @import("std");

const ttir_compile_sandbox = @import("../ttir_compile_sandbox.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args_json = "{}";

    const ttir_2d = try ttir_compile_sandbox.get2dUnifiedAttentionTtir(allocator, io, args_json);
    defer allocator.free(ttir_2d);

    const ttir_3d = try ttir_compile_sandbox.get3dUnifiedAttentionTtir(allocator, io, args_json);
    defer allocator.free(ttir_3d);

    const ttir_reduce = try ttir_compile_sandbox.get3dReduceSegmentsTtir(allocator, io, args_json);
    defer allocator.free(ttir_reduce);

    const ttir_decode = try ttir_compile_sandbox.getDecodeAttentionStage1Ttir(allocator, io, args_json);
    defer allocator.free(ttir_decode);

    const ttir_prefill = try ttir_compile_sandbox.getPrefillAttentionTtir(allocator, io, args_json);
    defer allocator.free(ttir_prefill);

    const ttir_hello_world = try ttir_compile_sandbox.getHelloWorldMatmulTtir(allocator, io, args_json);
    defer allocator.free(ttir_hello_world);

    std.debug.print("2d unified attention TTIR bytes: {d}\n", .{ttir_2d.len});
    std.debug.print("3d unified attention TTIR bytes: {d}\n", .{ttir_3d.len});
    std.debug.print("3d reduce segments TTIR bytes: {d}\n", .{ttir_reduce.len});
    std.debug.print("decode stage1 TTIR bytes: {d}\n", .{ttir_decode.len});
    std.debug.print("prefill TTIR bytes: {d}\n", .{ttir_prefill.len});
    std.debug.print("hello_world matmul TTIR bytes: {d}\n", .{ttir_hello_world.len});
}
