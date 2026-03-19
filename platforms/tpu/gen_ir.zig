const std = @import("std");

const bazel = @import("bazel");
const bazel_builtin = @import("bazel_builtin");
const c_interface = @import("c");

const log = std.log.scoped(.@"zml/platforms/tpu/gen_ir");

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

extern "c" fn mbstowcs(dest: [*]c_interface.wchar_t, src: [*]const u8, n: usize) usize;
pub fn toPosixPathW(file_path: []const u8) error{NameTooLong}![std.posix.PATH_MAX - 1:0]c_interface.wchar_t {
    if (file_path.len >= std.posix.PATH_MAX) return error.NameTooLong;

    var path_with_null: [std.posix.PATH_MAX - 1:0]c_interface.wchar_t = undefined;
    const len = mbstowcs(&path_with_null, file_path.ptr, file_path.len);
    path_with_null[len] = 0;
    return path_with_null;
}

fn getGenerateBinPath(allocator: std.mem.Allocator) ![]const u8 {
    const runfiles = try bazel.runfiles(bazel_builtin.current_repository);

    var sandbox_path_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try runfiles.rlocation("libpjrt_tpu/sandbox", &sandbox_path_buf) orelse return error.FileNotFound;

    return std.fs.path.join(allocator, &.{ sandbox_path, "bin", "gen_ir_zig" });
}

pub const Runtime = struct {
    process: std.process.Child,
    request_mutex: std.Io.Mutex = .init,

    pub fn init(allocator: std.mem.Allocator, io: std.Io) !Runtime {
        const path = try getGenerateBinPath(allocator);
        defer allocator.free(path);

        var process = try std.process.spawn(io, .{
            .argv = &.{path},
            .stdin = .pipe,
            .stdout = .pipe,
        });
        errdefer process.kill(io);

        return .{ .process = process };
    }

    pub fn request(self: *Runtime, allocator: std.mem.Allocator, io: std.Io, request_json: []const u8) ![]u8 {
        self.request_mutex.lockUncancelable(io);
        defer self.request_mutex.unlock(io);

        var arena = std.heap.ArenaAllocator.init(allocator);
        defer arena.deinit();

        const writer_buffer = try arena.allocator().alloc(u8, 4096);
        var writer = self.process.stdin.?.writer(io, writer_buffer);
        try writer.interface.print("{s}\n", .{request_json});
        try writer.interface.flush();

        const reader_buffer = try arena.allocator().alloc(u8, 4096);
        var reader = self.process.stdout.?.reader(io, reader_buffer);
        var allocating: std.Io.Writer.Allocating = .init(arena.allocator());
        _ = try reader.interface.streamDelimiter(&allocating.writer, '\n');
        _ = try reader.interface.discardShort(1);

        const response: std.json.Value = try std.json.parseFromSliceLeaky(std.json.Value, arena.allocator(), allocating.written(), .{});
        const ok = response.object.get("ok") orelse {
            log.err("TPU backend config response is missing `ok`: {s}", .{allocating.written()});
            return error.InvalidResponse;
        };
        const ok_bool = switch (ok) {
            .bool => |value| value,
            else => {
                log.err("TPU backend config response has non-bool `ok`: {s}", .{allocating.written()});
                return error.InvalidResponse;
            },
        };

        if (ok_bool) {
            const result = response.object.get("result") orelse {
                log.err("TPU backend config response is missing `result`: {s}", .{allocating.written()});
                return error.InvalidResponse;
            };
            return switch (result) {
                .string => |value| try allocator.dupe(u8, value),
                else => {
                    log.err("TPU backend config response has non-string `result`: {s}", .{allocating.written()});
                    return error.InvalidResponse;
                },
            };
        }

        if (response.object.get("error")) |value| {
            switch (value) {
                .string => |message| log.err("TPU backend config generation failed: {s}", .{message}),
                else => log.err("TPU backend config generation failed with invalid error payload: {s}", .{allocating.written()}),
            }
        } else {
            log.err("TPU backend config generation failed without an error payload: {s}", .{allocating.written()});
        }
        return error.GenerationFailed;
    }

    pub fn deinit(self: *Runtime, io: std.Io) void {
        _ = std.c.kill(self.process.id.?, .INT);
        _ = self.process.wait(io) catch unreachable;
    }
};

// ./bazel.sh build //platforms/tpu:sandbox && bazel-bin/platforms/tpu/sandbox/bin/gen_ir_zig
pub fn main(init: std.process.Init) !void {
    const io = init.io;

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
    if (result == null) {
        checkPythonError();
        return error.PythonExecutionFailed;
    }
    defer c_interface.Py_DecRef(result);
}
