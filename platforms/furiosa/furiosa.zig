const std = @import("std");
const builtin = @import("builtin");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const pjrt = @import("pjrt");
const platforms_options = @import("platforms/options");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/platforms/furiosa");

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
extern fn mkdtemp(template: [*:0]u8) ?[*:0]u8;

fn setEnv(name: [*:0]const u8, value: [:0]const u8, overwrite: c_int) !void {
    if (setenv(name, value, overwrite) != 0) return error.SetEnvFailed;
}

fn setupFuriosaEnv(io: std.Io, root: []const u8) !void {
    var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    try setEnv("XLA_FURIOSA_COMPILER", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ root, "bin", "furiosa-tcc" }), 1);
    try setEnv("XLA_FURIOSA_RUNTIME_LIBRARY", try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ root, "lib", "libdevice_runtime.so" }), 1);
    try setEnv("XLA_FURIOSA_VISIBLE_DEVICES", "0", 0);

    if (std.c.getenv("XLA_FURIOSA_COMPILER_CACHE") == null) {
        var base_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const base = if (std.c.getenv("TEST_TMPDIR")) |path|
            std.mem.span(path)
        else if (std.c.getenv("XDG_CACHE_HOME")) |path|
            std.mem.span(path)
        else if (std.c.getenv("HOME")) |path|
            try stdx.Io.Dir.path.bufJoin(&base_buf, &.{ std.mem.span(path), ".cache" })
        else blk: {
            const template = try stdx.Io.Dir.path.bufJoinZ(&base_buf, &.{
                if (std.c.getenv("TMPDIR")) |path| std.mem.span(path) else "/tmp",
                "zml-furiosa-XXXXXX",
            });
            break :blk std.mem.span(mkdtemp(template) orelse return error.CreateCacheFailed);
        };
        // Bump when changing the pinned toolchain or runtime/image contract.
        const cache = try stdx.Io.Dir.path.bufJoinZ(&buf, &.{ base, "zml/furiosa/tcc-2026.3.0-bridge19-ir7-gcc13-v1" });
        _ = try std.Io.Dir.cwd().createDirPathStatus(io, cache, .fromMode(0o700));
        try setEnv("XLA_FURIOSA_COMPILER_CACHE", cache, 0);
    }
}

pub fn isEnabled() bool {
    return platforms_options.furiosa_enabled;
}

pub fn load(allocator: std.mem.Allocator, io: std.Io) !*const pjrt.Api {
    if (comptime !isEnabled() or builtin.os.tag != .linux or builtin.cpu.arch != .x86_64) return error.Unavailable;

    const r = try bazel.runfiles(bazel_builtin.current_repository);
    var path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox = try r.rlocation("zml/platforms/furiosa/sandbox", &path_buf) orelse {
        log.err("Missing Furiosa sandbox runfile", .{});
        return error.FileNotFound;
    };
    var lib_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const runfile = try stdx.Io.Dir.path.bufJoin(&lib_path_buf, &.{ sandbox, "lib", "libpjrt_c_api_furiosa_plugin.so" });
    // Tests can expose individual runfiles as symlinks. Keep SDK discovery
    // relative to the actual assembled bundle containing the plugin.
    const resolved = try std.Io.Dir.cwd().realPathFileAlloc(io, runfile, allocator);
    defer allocator.free(resolved);
    const library_dir = std.fs.path.dirname(resolved) orelse return error.InvalidPath;
    const root = std.fs.path.dirname(library_dir) orelse return error.InvalidPath;
    try setupFuriosaEnv(io, root);
    const library = try stdx.Io.Dir.path.bufJoinZ(&lib_path_buf, &.{resolved});
    log.info("Loading Furiosa plugin: {s}", .{library});
    return .loadFrom(library);
}
