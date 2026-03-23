const std = @import("std");

const c = @import("c");

fn pyStatusCheck(status: c.PyStatus) void {
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

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const io = init.io;

    {
        var preconfig: c.PyPreConfig = undefined;
        c.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyStatusCheck(c.Py_PreInitialize(&preconfig));
    }

    var config: c.PyConfig = undefined;
    c.PyConfig_InitIsolatedConfig(&config);
    defer c.PyConfig_Clear(&config);

    config.module_search_paths_set = 1;
    config.optimization_level = 2;
    config.write_bytecode = 0;

    const args = try init.minimal.args.toSlice(arena.allocator());
    const cArgs = try arena.allocator().alloc([*:0]const u8, args.len);
    for (args, 0..) |arg, i| {
        cArgs[i] = (try arena.allocator().dupeZ(u8, arg)).ptr;
    }
    pyStatusCheck(c.PyConfig_SetBytesArgv(&config, @intCast(cArgs.len), @ptrCast(cArgs.ptr)));

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
            c.PY_MAJOR_VERSION,
            c.PY_MINOR_VERSION,
        });
        pyStatusCheck(c.PyConfig_SetBytesString(&config, &config.home, home));
        pyStatusCheck(c.PyWideStringList_Append(&config.module_search_paths, &try toPosixPathW(home)));
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
        pyStatusCheck(c.PyWideStringList_Append(&config.module_search_paths, &try toPosixPathW(site_packages)));
    }

    pyStatusCheck(c.Py_InitializeFromConfig(&config));
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
