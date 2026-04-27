//! Run the same `canonicalize,cse,canonicalize` pipeline on a given
//! `.mlir` file so we can apples-to-apples compare with what our DSL
//! emits. Used for diffing Pallas's `debug=True` output against the Zig
//! port.
//!
//! Usage:
//!   bazel run //examples/mosaic_ragged_paged:canonicalize -- /tmp/rpa_compare/pallas_mosaic.mlir
const std = @import("std");
const mlir = @import("mlir");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    if (args.len < 2) {
        var stderr_buf: [256]u8 = undefined;
        var stderr = std.Io.File.stderr().writer(io, &stderr_buf);
        defer stderr.interface.flush() catch {};
        try stderr.interface.print("usage: canonicalize <path-to-mlir>\n", .{});
        return error.MissingPath;
    }

    var file = try std.Io.Dir.openFile(.cwd(), io, args[1], .{ .mode = .read_only });
    defer file.close(io);
    const read_buf = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(read_buf);
    var reader = file.reader(io, read_buf);
    var allocating: std.Io.Writer.Allocating = .init(allocator);
    defer allocating.deinit();
    _ = try reader.interface.streamRemaining(&allocating.writer);
    const src = try allocator.dupeZ(u8, allocating.written());
    defer allocator.free(src);

    const registry = try mlir.DialectRegistry.init();
    defer registry.deinit();
    inline for (.{ "func", "arith", "scf", "math", "memref", "vector", "tpu" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
    mlir.registerPasses("Transforms");

    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    defer ctx.deinit();
    ctx.loadAllAvailableDialects();

    const module = try mlir.Module.parse(ctx, src);
    defer module.deinit();

    const pm = mlir.PassManager.init(ctx);
    defer pm.deinit();
    const opm = pm.asOpPassManager();
    inline for (.{ "canonicalize", "cse", "canonicalize" }) |pass| {
        try opm.addPipeline(pass);
    }
    try pm.runOnOp(module.operation());

    var stdout_buf: [256 * 1024]u8 = undefined;
    var stdout = std.Io.File.stdout().writer(io, &stdout_buf);
    defer stdout.interface.flush() catch {};
    try stdout.interface.print("{f}\n", .{module.operation().fmt(.{ .debug_info = false })});
}
