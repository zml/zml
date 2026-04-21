const std = @import("std");

pub const hlo_rewriter = @import("hlo_rewriter.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/nki");

pub fn materializeEmbeddedKernels(
    allocator: std.mem.Allocator,
    io: std.Io,
    hlo_code: []const u8,
    tmp_dir: []const u8,
    target: []const u8,
) ![]const u8 {
    const rewritten_hlo = try hlo_rewriter.rewriteCustomCalls(allocator, io, hlo_code, tmp_dir, target);
    if (rewritten_hlo != null) {
        log.info("Materialized embedded ZML NKI kernels into Neuron custom-calls", .{});
    }
    return rewritten_hlo orelse hlo_code;
}

test {
    std.testing.refAllDecls(@This());
}
