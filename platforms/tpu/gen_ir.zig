const std = @import("std");

const c_interface = @import("c");

fn pyStatusCheck(status: c_interface.PyStatus) void {
    if (c_interface.PyStatus_Exception(status) != 0) {
        if (c_interface.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c_interface.Py_ExitStatusException(status);
    }
}

fn checkPythonError() void {
    if (c_interface.PyErr_Occurred() != null) {
        std.debug.print("Python error occurred:\n", .{});
        c_interface.PyErr_Print();
    }
}

pub fn toPosixPathW(file_path: []const u8) error{NameTooLong}![std.posix.PATH_MAX - 1:0]c_interface.wchar_t {
    if (file_path.len >= std.posix.PATH_MAX) return error.NameTooLong;

    var path_with_null: [std.posix.PATH_MAX - 1:0]c_interface.wchar_t = undefined;
    const len = c_interface.mbstowcs(&path_with_null, file_path.ptr, file_path.len);
    path_with_null[len] = 0;
    return path_with_null;
}

// ./bazel.sh build //platforms/tpu:sandbox && bazel-bin/platforms/tpu/sandbox/bin/gen_ir_zig

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    _ = allocator; // autofix
    const arena = init.arena;
    _ = arena; // autofix
    const io = init.io;
    const values = init.minimal.args.vector;
    _ = values; // autofix
    {
        var preconfig: c_interface.PyPreConfig = undefined;
        c_interface.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyStatusCheck(c_interface.Py_PreInitialize(&preconfig));
    }

    var config: c_interface.PyConfig = undefined;
    c_interface.PyConfig_InitIsolatedConfig(&config);
    defer c_interface.PyConfig_Clear(&config);

    config.module_search_paths_set = 1;
    config.optimization_level = 2;
    config.write_bytecode = 0;

    _ = c_interface.PyConfig_SetBytesArgv(&config, @intCast(init.minimal.args.vector.len), @ptrCast(init.minimal.args.vector.ptr));

    var self_exe_dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const self_exe_dir = blk: {
        const n = try std.process.executableDirPath(io, &self_exe_dir_buf);
        break :blk self_exe_dir_buf[0..n];
    };

    {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const home = try std.fmt.bufPrintZ(&buf, "{f}{d}.{d}", .{
            std.Io.Dir.path.fmtJoin(&.{
                self_exe_dir,
                "..",
                "lib",
                "python",
            }),
            c_interface.PY_MAJOR_VERSION,
            c_interface.PY_MINOR_VERSION,
        });
        pyStatusCheck(c_interface.PyConfig_SetBytesString(&config, &config.home, home));
        pyStatusCheck(c_interface.PyWideStringList_Append(&config.module_search_paths, &try toPosixPathW(home)));
    }

    {
        var buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
        const site_packages = try std.fmt.bufPrint(&buf, "{f}", .{
            std.Io.Dir.path.fmtJoin(&.{
                self_exe_dir,
                "..",
                "site-packages",
            }),
        });
        pyStatusCheck(c_interface.PyWideStringList_Append(&config.module_search_paths, &try toPosixPathW(site_packages)));
    }

    pyStatusCheck(c_interface.Py_InitializeFromConfig(&config));
    defer c_interface.Py_Finalize();

    const genir_main = blk: {
        const module = c_interface.PyImport_ImportModule("gen_ir") orelse {
            return error.PythonModuleImportFailed;
        };
        defer c_interface.Py_DecRef(module);
        break :blk c_interface.PyObject_GetAttrString(module, "main");
    };
    defer c_interface.Py_DecRef(genir_main);

    const result = c_interface.PyObject_CallNoArgs(genir_main);
    checkPythonError();

    defer c_interface.Py_DecRef(result);
}
