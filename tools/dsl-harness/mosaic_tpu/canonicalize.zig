//! Run `canonicalize, cse, canonicalize` on an MLIR string and return
//! the printed result. The Zig DSL already runs these passes; this is
//! only invoked on the Pallas side to make the diff apples-to-apples.

const std = @import("std");

const mlir = @import("mlir");

const Allocator = std.mem.Allocator;

pub fn canonicalize(
    arena: Allocator,
    ctx: *mlir.Context,
    src: []const u8,
) ![]u8 {
    // mlir.Module.parse needs a null-terminated buffer.
    const src_z = try arena.dupeZ(u8, src);
    const module = try mlir.Module.parse(ctx, src_z);
    defer module.deinit();

    const pm = mlir.PassManager.init(ctx);
    defer pm.deinit();
    const opm = pm.asOpPassManager();
    inline for (.{ "canonicalize", "cse", "canonicalize" }) |pass| {
        try opm.addPipeline(pass);
    }
    try pm.runOnOp(module.operation());

    var aw = std.Io.Writer.Allocating.init(arena);
    errdefer aw.deinit();
    try aw.writer.print("{f}", .{module.operation().fmt(.{ .debug_info = false })});
    return aw.toOwnedSlice();
}
