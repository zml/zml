const std = @import("std");

pub const hlo_rewriter = @import("hlo_rewriter.zig");

pub fn materializeEmbeddedKernels(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp_dir: std.Io.Dir,
    hlo_code: []const u8,
    target: []const u8,
) ![]const u8 {
    const rewritten_hlo = try hlo_rewriter.rewriteCustomCalls(allocator, io, tmp_dir, hlo_code, target);
    return rewritten_hlo orelse hlo_code;
}
