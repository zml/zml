const std = @import("std");

pub fn main(init: std.process.Init) !void {
    const arena = init.arena.allocator();
    const args = try init.minimal.args.toSlice(arena);
    const executable = try std.process.executablePathAlloc(init.io, arena);
    const bin = std.fs.path.dirname(executable) orelse return error.InvalidPath;
    const root = std.fs.path.dirname(bin) orelse return error.InvalidPath;

    var argv: std.ArrayList([]const u8) = .empty;
    if (std.mem.eql(u8, std.fs.path.basename(executable), "aarch64-linux-gnu-gcc")) {
        const toolchain = try std.fmt.allocPrint(arena, "{s}/toolchain", .{root});
        try argv.appendSlice(arena, &.{
            try std.fmt.allocPrint(arena, "{s}/usr/bin/aarch64-linux-gnu-gcc-13", .{toolchain}),
            try std.fmt.allocPrint(arena, "--sysroot={s}", .{toolchain}),
            try std.fmt.allocPrint(arena, "-B{s}/usr/libexec/gcc-cross/aarch64-linux-gnu/13/", .{toolchain}),
            try std.fmt.allocPrint(arena, "-B{s}/usr/lib/gcc-cross/aarch64-linux-gnu/13/", .{toolchain}),
            try std.fmt.allocPrint(arena, "-B{s}/usr/aarch64-linux-gnu/bin/", .{toolchain}),
        });
    } else {
        // These settings affect only the compiler process and its children.
        try init.environ_map.put("PATH", try std.fmt.allocPrint(arena, "{s}/bin:{s}/toolchain/usr/bin", .{ root, root }));
        try init.environ_map.put("LD_LIBRARY_PATH", try std.fmt.allocPrint(arena, "{s}/toolchain/usr/lib/x86_64-linux-gnu:{s}/toolchain/lib/x86_64-linux-gnu", .{ root, root }));
        for ([_][]const u8{
            "LD_PRELOAD", "GCC_EXEC_PREFIX", "COMPILER_PATH",      "LIBRARY_PATH",
            "CPATH",      "C_INCLUDE_PATH",  "CPLUS_INCLUDE_PATH", "OBJC_INCLUDE_PATH",
        }) |key| {
            _ = init.environ_map.swapRemove(key);
        }
        try argv.append(arena, try std.fmt.allocPrint(arena, "{s}/libexec/furiosa-tcc", .{root}));
    }
    try argv.appendSlice(arena, args[1..]);
    return std.process.replace(init.io, .{
        .argv = argv.items,
        .environ_map = init.environ_map,
    });
}
