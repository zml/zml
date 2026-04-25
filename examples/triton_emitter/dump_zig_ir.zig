//! Dump the Zig DSL's raw TTIR for every kernel in `kernels_zig.KERNELS`.
//!
//! Pair with `dump_python_ir.py` (Python's TTIR) and `dump_via_xla` (XLA
//! pipeline) to compare the two frontends at every IR stage.

const std = @import("std");

const mlir = @import("mlir");

const zml = @import("zml");
const stdx = zml.stdx;

const kernels = @import("kernels_zig.zig");

pub const std_options: std.Options = .{ .log_level = .info };
const log = std.log.scoped(.dump_zig);

const CliArgs = struct {
    pub const help =
        \\dump_zig --out-dir=zig_ir [--kernel=NAME]
        \\
    ;
    @"out-dir": []const u8 = "zig_ir",
    kernel: []const u8 = "",
};

fn setupContext() !*mlir.Context {
    const registry = try mlir.DialectRegistry.init();
    inline for (.{ "func", "arith", "scf", "math", "tt" }) |d| {
        mlir.DialectHandle.fromString(d).insertDialect(registry);
    }
    const ctx = try mlir.Context.init(.{ .registry = registry, .threading = false });
    ctx.loadAllAvailableDialects();
    return ctx;
}

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const cli: CliArgs = stdx.flags.parse(init.minimal.args, CliArgs);

    const ctx = try setupContext();
    defer ctx.deinit();

    var out_dir = try std.Io.Dir.createDirPathOpen(.cwd(), io, cli.@"out-dir", .{});
    defer out_dir.close(io);

    var write_buf: [16 * 1024]u8 = undefined;

    for (kernels.KERNELS) |entry| {
        if (cli.kernel.len > 0 and !std.mem.eql(u8, cli.kernel, entry.name)) continue;

        const ir = try entry.emit(allocator, ctx);
        defer allocator.free(ir);

        const filename = try std.fmt.allocPrint(allocator, "{s}.ttir", .{entry.name});
        defer allocator.free(filename);

        const file = try out_dir.createFile(io, filename, .{});
        defer file.close(io);

        var writer = file.writer(io, &write_buf);
        try writer.interface.writeAll(ir);
        try writer.interface.flush();

        log.info("wrote {s}/{s} ({d} bytes)", .{ cli.@"out-dir", filename, ir.len });
    }
}
