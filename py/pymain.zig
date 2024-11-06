const std = @import("std");
const c = @import("c");
const pyoptions = @import("pyoptions");

fn toWchar(str: []const u8, out: [:0]c.wchar_t) [:0]c.wchar_t {
    const len = c.mbstowcs(out.ptr, str.ptr, str.len);
    out[len] = 0;
    return out[0..len :0];
}

fn pyErrorOrExit(status: c.PyStatus) void {
    if (c.PyStatus_Exception(status) != 0) {
        if (c.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c.Py_ExitStatusException(status);
    }
}

pub fn main() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var preconfig: c.PyPreConfig = undefined;
    c.PyPreConfig_InitIsolatedConfig(&preconfig);
    preconfig.utf8_mode = 1;
    pyErrorOrExit(c.Py_PreInitialize(&preconfig));

    var config: c.PyConfig = undefined;
    c.PyConfig_InitIsolatedConfig(&config);
    defer c.PyConfig_Clear(&config);

    config.optimization_level = 2;

    pyErrorOrExit(c.PyConfig_SetBytesString(&config, &config.home, pyoptions.home));
    {
        config.module_search_paths_set = 1;
        var wbuf: [std.fs.max_path_bytes:0]c.wchar_t = undefined;
        inline for (pyoptions.modules) |module| {
            const wline = toWchar(module, &wbuf);
            pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, wline.ptr));
        }
    }

    pyErrorOrExit(c.PyConfig_SetBytesArgv(&config, @intCast(std.os.argv.len), std.os.argv.ptr));

    pyErrorOrExit(c.PyConfig_SetBytesString(&config, &config.run_filename, blk: {
        std.fs.cwd().access(pyoptions.main, .{ .mode = .read_only }) catch {
            break :blk pyoptions.main_compiled;
        };
        break :blk pyoptions.main;
    }));

    pyErrorOrExit(c.Py_InitializeFromConfig(&config));
    std.process.exit(@intCast(c.Py_RunMain()));
}
