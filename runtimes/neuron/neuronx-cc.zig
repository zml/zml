const std = @import("std");
const builtin = @import("builtin");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

fn pyErrorOrExit(status: c.PyStatus) void {
    if (c.PyStatus_Exception(status) != 0) {
        if (c.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c.Py_ExitStatusException(status);
    }
}

pub fn toPosixPathW(file_path: []const u8) error{NameTooLong}![std.posix.PATH_MAX - 1:0]c.wchar_t {
    if (file_path.len >= std.posix.PATH_MAX) return error.NameTooLong;

    var path_with_null: [std.posix.PATH_MAX - 1:0]c.wchar_t = undefined;
    const len = c.mbstowcs(&path_with_null, file_path.ptr, file_path.len);
    path_with_null[len] = 0;
    return path_with_null;
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    {
        var preconfig: c.PyPreConfig = undefined;
        c.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyErrorOrExit(c.Py_PreInitialize(&preconfig));
    }

    var config: c.PyConfig = undefined;
    c.PyConfig_InitIsolatedConfig(&config);
    _ = c.PyConfig_SetBytesArgv(&config, @intCast(std.os.argv.len), @ptrCast(std.os.argv));
    defer c.PyConfig_Clear(&config);

    var self_exe_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const self_exe_dir = try std.fs.selfExeDirPath(&self_exe_dir_buf);

    {
        var bufZ: [std.fs.max_path_bytes:0]u8 = undefined;
        pyErrorOrExit(c.PyConfig_SetBytesString(&config, &config.home, try stdx.fs.path.bufJoinZ(&bufZ, &.{
            self_exe_dir,
            "..",
            "lib",
            "python3.11",
        })));
    }

    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        config.module_search_paths_set = 1;
        pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, &(try toPosixPathW(try stdx.fs.path.bufJoin(&buf, &.{
            self_exe_dir,
            "..",
            "lib",
            "python3.11",
        })))));
        pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, &(try toPosixPathW(try stdx.fs.path.bufJoin(&buf, &.{
            self_exe_dir,
            "..",
            "site-packages",
        })))));
    }

    std.debug.print(">>>> CORENTIN\n", .{});

    pyErrorOrExit(c.Py_InitializeFromConfig(&config));
    defer c.Py_Finalize();

    const neuronxcc_main = blk: {
        const module = c.PyImport_ImportModule("neuronxcc.driver.CommandDriver");
        std.debug.print(">>> MODULE: {any}\n", .{module});
        defer c.Py_DecRef(module);
        break :blk c.PyObject_GetAttrString(module, "main");
    };
    defer c.Py_DecRef(neuronxcc_main);

    const result = c.PyObject_CallNoArgs(neuronxcc_main);
    defer c.Py_DecRef(result);
}
