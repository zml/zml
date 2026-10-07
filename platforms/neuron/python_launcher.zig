const std = @import("std");

const c = @import("c");

// Shared isolated-Python launcher used by Neuron helper binaries. The sandbox
// contains the Python stdlib, site-packages, and these tiny Zig executables; the
// helpers configure Python from their own location so they are hermetic.

pub const ModuleEntrypoint = struct {
    module_name: [:0]const u8,
    attr_name: [:0]const u8 = "main",
};

// Run `module.attr()` inside the sandboxed Python interpreter.
pub fn runModuleEntrypoint(init: std.process.Init, entrypoint: ModuleEntrypoint) !void {
    var config: c.PyConfig = undefined;
    _ = try configureInterpreter(init, &config);
    defer c.PyConfig_Clear(&config);

    pyStatusCheck(c.Py_InitializeFromConfig(&config));
    defer c.Py_Finalize();

    const module = c.PyImport_ImportModule(entrypoint.module_name.ptr) orelse {
        std.log.err("Failed to import {s}", .{entrypoint.module_name});
        c.PyErr_Print();
        return error.FailedToImportModule;
    };
    defer c.Py_DecRef(module);

    const main_fn = c.PyObject_GetAttrString(module, entrypoint.attr_name.ptr) orelse {
        std.log.err("Failed to resolve {s}.{s}", .{ entrypoint.module_name, entrypoint.attr_name });
        c.PyErr_Print();
        return error.FailedToResolveEntrypoint;
    };
    defer c.Py_DecRef(main_fn);

    const result = c.PyObject_CallNoArgs(main_fn);
    if (result == null) {
        c.PyErr_Print();
        std.process.exit(1);
    }
    defer c.Py_DecRef(result);
}

// Run a Python file relative to the helper binary and call its `main()`.
pub fn runScriptMain(init: std.process.Init, relative_script_path: []const []const u8) !void {
    const arena = init.arena;

    var config: c.PyConfig = undefined;
    const self_exe_dir = try configureInterpreter(init, &config);
    defer c.PyConfig_Clear(&config);

    pyStatusCheck(c.Py_InitializeFromConfig(&config));
    defer c.Py_Finalize();

    const script_path = try std.Io.Dir.path.join(arena.allocator(), relative_script_path);
    const full_script_path = try std.Io.Dir.path.join(arena.allocator(), &.{ self_exe_dir, script_path });
    const bootstrap = try std.fmt.allocPrint(arena.allocator(),
        \\import importlib.util
        \\spec = importlib.util.spec_from_file_location("_zml_neuron_python_launcher", "{s}")
        \\module = importlib.util.module_from_spec(spec)
        \\assert spec.loader is not None
        \\spec.loader.exec_module(module)
        \\module.main()
    , .{full_script_path});

    if (c.PyRun_SimpleStringFlags(bootstrap.ptr, null) != 0) {
        c.PyErr_Print();
        return error.CompilerExecutionFailed;
    }
}

fn pyStatusCheck(status: c.PyStatus) void {
    if (c.PyStatus_Exception(status) != 0) {
        if (c.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c.Py_ExitStatusException(status);
    }
}

fn toPosixPathW(file_path: []const u8) error{ InvalidUtf8, NameTooLong, CodepointTooLarge }![std.posix.PATH_MAX - 1:0]c.wchar_t {
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

fn configureInterpreter(init: std.process.Init, config: *c.PyConfig) ![]const u8 {
    const arena = init.arena;
    const io = init.io;

    {
        var preconfig: c.PyPreConfig = undefined;
        c.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyStatusCheck(c.Py_PreInitialize(&preconfig));
    }

    c.PyConfig_InitIsolatedConfig(config);
    config.module_search_paths_set = 1;
    config.optimization_level = 2;
    config.write_bytecode = 0;

    const args = try init.minimal.args.toSlice(arena.allocator());
    const c_args = try arena.allocator().alloc([*:0]const u8, args.len);
    for (args, 0..) |arg, i| {
        c_args[i] = (try arena.allocator().dupeZ(u8, arg)).ptr;
    }
    pyStatusCheck(c.PyConfig_SetBytesArgv(config, @intCast(c_args.len), @ptrCast(c_args.ptr)));

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
        pyStatusCheck(c.PyConfig_SetBytesString(config, &config.home, home));
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

    return self_exe_dir;
}
