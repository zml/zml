const std = @import("std");

const c = @import("c");
const runfiles = @import("runfiles");

fn pyStatusCheck(status: c.PyStatus) void {
    if (c.PyStatus_Exception(status) != 0) {
        if (c.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c.Py_ExitStatusException(status);
    }
}

pub fn toPosixPathW(file_path: []const u8) error{ InvalidUtf8, NameTooLong, CodepointTooLarge }![std.posix.PATH_MAX - 1:0]c.wchar_t {
    var view = std.unicode.Utf8View.init(file_path) catch return error.InvalidUtf8;
    var it = view.iterator();

    var path_with_null: [std.posix.PATH_MAX - 1:0]c.wchar_t = undefined;
    var len: usize = 0;
    while (it.nextCodepoint()) |cp| {
        if (len + 1 >= path_with_null.len) return error.NameTooLong;
        path_with_null[len] = std.math.cast(c.wchar_t, cp) orelse return error.CodepointTooLarge;
        len += 1;
    }
    path_with_null[len] = 0;
    return path_with_null;
}

pub fn main(init: std.process.Init.Minimal) !void {
    const allocator = std.heap.c_allocator;

    var threaded: std.Io.Threaded = .init(allocator, .{});
    defer threaded.deinit();

    const io = threaded.io();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

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

    pyStatusCheck(c.PyConfig_SetBytesArgv(&config, @intCast(init.args.vector.len), @ptrCast(init.args.vector.ptr)));

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

    const module = c.PyImport_ImportModule("generate");
    if (module == null) {
        c.PyErr_Print();
        return;
    }
    defer c.Py_DecRef(module);

    const func = c.PyObject_GetAttrString(module, "main");
    defer c.Py_DecRef(func);

    const result = c.PyObject_CallNoArgs(func);
    if (result == null) {
        c.PyErr_Print();
        return;
    }
    defer c.Py_DecRef(result);
}
