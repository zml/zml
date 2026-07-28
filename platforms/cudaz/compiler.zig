const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const runfiles = @import("runfiles");

const log = std.log.scoped(.@"zml/platforms/cudaz/compiler");
const zig_runtime_config = "zml/platforms/cudaz/zig_runtime.txt";

pub const Error = error{
    RunfilesUnavailable,
    ToolchainConfigUnavailable,
    ZigExecutableUnavailable,
    ZigLibUnavailable,
    KernelSourceUnavailable,
    ZigCompilationFailed,
    InvalidKernelIr,
};

pub fn compile(allocator: std.mem.Allocator) ![:0]u8 {
    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();
    const io = threaded.io();

    const executable_path = try std.process.executablePathAlloc(io, allocator);
    defer allocator.free(executable_path);

    var runfiles_instance = try runfiles.Runfiles.create(.{
        .allocator = allocator,
        .io = io,
        .argv0 = executable_path,
        .directory = if (std.c.getenv("RUNFILES_DIR")) |value| std.mem.span(value) else null,
        .manifest = if (std.c.getenv("RUNFILES_MANIFEST_FILE")) |value| std.mem.span(value) else null,
    }) orelse return error.RunfilesUnavailable;
    defer runfiles_instance.deinit(allocator);
    const r = runfiles_instance.withSourceRepo(bazel_builtin.current_repository);

    const config_path = try r.rlocationAlloc(allocator, zig_runtime_config) orelse
        return error.ToolchainConfigUnavailable;
    defer allocator.free(config_path);
    const config = try std.Io.Dir.cwd().readFileAlloc(io, config_path, allocator, .limited(16 * 1024));
    defer allocator.free(config);
    var config_lines = std.mem.tokenizeScalar(u8, config, '\n');
    const zig_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;
    const zig_lib_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;
    const kernel_runfile = config_lines.next() orelse return error.ToolchainConfigUnavailable;

    const zig_path = try r.rlocationAlloc(allocator, zig_runfile) orelse
        return error.ZigExecutableUnavailable;
    defer allocator.free(zig_path);
    const zig_lib_path = try r.rlocationAlloc(allocator, zig_lib_runfile) orelse
        return error.ZigLibUnavailable;
    defer allocator.free(zig_lib_path);
    const kernel_path = try r.rlocationAlloc(allocator, kernel_runfile) orelse
        return error.KernelSourceUnavailable;
    defer allocator.free(kernel_path);

    const tmp_root_path = if (std.c.getenv("TMPDIR")) |value| std.mem.span(value) else "/tmp";
    var tmp_root = try std.Io.Dir.openDir(.cwd(), io, tmp_root_path, .{});
    defer tmp_root.close(io);

    var dir_name_buffer: [128]u8 = undefined;
    const dir_name = try std.fmt.bufPrint(
        &dir_name_buffer,
        "zml-cudaz-{d}",
        .{std.Io.Timestamp.now(io, .real).nanoseconds},
    );
    var tmp_dir = try tmp_root.createDirPathOpen(io, dir_name, .{ .permissions = .fromMode(0o700) });
    defer tmp_root.deleteTree(io, dir_name) catch {};
    defer tmp_dir.close(io);

    var tmp_path_buffer: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tmp_path_len = try tmp_dir.realPath(io, &tmp_path_buffer);
    const tmp_path = tmp_path_buffer[0..tmp_path_len];

    try runZig(allocator, io, tmp_path, &.{
        zig_path,
        "build-obj",
        kernel_path,
        "-target",
        "nvptx64-cuda",
        "-mcpu",
        "sm_50+ptx63",
        "-O",
        "ReleaseFast",
        "-fno-incremental",
        "-fno-entry",
        "-fno-emit-bin",
        "-femit-llvm-ir=kernel.ll",
        "--zig-lib-dir",
        zig_lib_path,
        "--cache-dir",
        "cache",
        "--global-cache-dir",
        "global-cache",
    });

    const llvm_ir = try tmp_dir.readFileAlloc(io, "kernel.ll", allocator, .limited(4 * 1024 * 1024));
    defer allocator.free(llvm_ir);
    const fixed_llvm_ir = try rewriteKernelIr(allocator, llvm_ir);
    defer allocator.free(fixed_llvm_ir);

    const fixed_ir_file = try tmp_dir.createFile(io, "kernel.fixed.ll", .{});
    defer fixed_ir_file.close(io);
    try fixed_ir_file.writePositionalAll(io, fixed_llvm_ir, 0);

    try runZig(allocator, io, tmp_path, &.{
        zig_path,
        "cc",
        "-target",
        "nvptx64-cuda",
        "-Xclang",
        "-target-cpu",
        "-Xclang",
        "sm_50",
        "-S",
        "kernel.fixed.ll",
        "-o",
        "kernel.ptx",
    });

    return try tmp_dir.readFileAllocOptions(
        io,
        "kernel.ptx",
        allocator,
        .limited(4 * 1024 * 1024),
        .of(u8),
        0,
    );
}

fn runZig(
    allocator: std.mem.Allocator,
    io: std.Io,
    cwd: []const u8,
    argv: []const []const u8,
) !void {
    const local_cache = try std.Io.Dir.path.join(allocator, &.{ cwd, "cache" });
    defer allocator.free(local_cache);
    const global_cache = try std.Io.Dir.path.join(allocator, &.{ cwd, "global-cache" });
    defer allocator.free(global_cache);
    var environ = std.process.Environ.Map.init(allocator);
    defer environ.deinit();
    try environ.put("ZIG_LOCAL_CACHE_DIR", local_cache);
    try environ.put("ZIG_GLOBAL_CACHE_DIR", global_cache);

    const result = std.process.run(allocator, io, .{
        .argv = argv,
        .cwd = .{ .path = cwd },
        .environ_map = &environ,
        .stdout_limit = .limited(1024 * 1024),
        .stderr_limit = .limited(1024 * 1024),
    }) catch return error.ZigCompilationFailed;
    defer allocator.free(result.stdout);
    defer allocator.free(result.stderr);

    switch (result.term) {
        .exited => |code| if (code == 0) return,
        else => {},
    }
    log.err("Zig PTX compilation failed:\n{s}", .{result.stderr});
    return error.ZigCompilationFailed;
}

fn rewriteKernelIr(allocator: std.mem.Allocator, llvm_ir: []const u8) ![]u8 {
    const alias_start = std.mem.indexOf(u8, llvm_ir, "@main = alias") orelse
        return error.InvalidKernelIr;
    const alias_end = std.mem.indexOfPos(u8, llvm_ir, alias_start, "\n") orelse
        return error.InvalidKernelIr;
    const kernel_declaration = "define private ptx_kernel void @kernel.main";
    if (std.mem.indexOf(u8, llvm_ir, kernel_declaration) == null) return error.InvalidKernelIr;

    const without_alias = try std.mem.concat(allocator, u8, &.{
        llvm_ir[0..alias_start],
        llvm_ir[alias_end + 1 ..],
    });
    defer allocator.free(without_alias);
    const renamed = try std.mem.replaceOwned(
        u8,
        allocator,
        without_alias,
        kernel_declaration,
        "define ptx_kernel void @main",
    );
    defer allocator.free(renamed);
    return try std.mem.replaceOwned(u8, allocator, renamed, " unnamed_addr", "");
}

test "rewrite exported Zig kernel IR for the NVPTX backend" {
    const input =
        \\@main = alias void (ptr, i64), ptr @kernel.main
        \\
        \\define private ptx_kernel void @kernel.main(ptr %0, i64 %1) unnamed_addr {
        \\  ret void
        \\}
    ;
    const actual = try rewriteKernelIr(std.testing.allocator, input);
    defer std.testing.allocator.free(actual);
    try std.testing.expectEqualStrings(
        \\
        \\define ptx_kernel void @main(ptr %0, i64 %1) {
        \\  ret void
        \\}
    ,
        actual,
    );
}

test "Bazel Zig toolchain compiles the cudaz kernel to PTX" {
    const ptx = try compile(std.testing.allocator);
    defer std.testing.allocator.free(ptx);
    try std.testing.expect(std.mem.indexOf(u8, ptx, ".visible .entry main(") != null);
}
