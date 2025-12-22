const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtimes/neuron");

fn findFreeTcpPort(io: std.Io) !u16 {
    var address: std.Io.net.IpAddress = .{ .ip4 = .{ .bytes = .{ 127, 0, 0, 1 }, .port = 0 } };
    var socket: std.Io.net.Socket = try address.bind(io, .{ .ip6_only = false, .mode = .stream, .protocol = .tcp });
    defer socket.close(io);
    return socket.address.getPort();
}

pub export fn zmlxneuron_dlopen(filename: [*c]const u8, flags: c_int) ?*anyopaque {
    const replacements: std.StaticStringMap([:0]const u8) = .initComptime(.{
        .{ "libnccom.so", "libnccom.so.2" },
        .{ "libnrt.so", "libnrt.so.1" },
        .{ "libncfw.so", "libncfw.so.2" },
    });

    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const new_filename: [*c]const u8 = if (filename) |f| blk: {
        const replacement = replacements.get(std.fs.path.basename(std.mem.span(f))) orelse break :blk f;
        break :blk stdx.fs.path.bufJoinZ(&buf, &.{
            stdx.fs.selfSharedObjectDirPath(),
            replacement,
        }) catch unreachable;
    } else null;

    return std.c.dlopen(new_filename, @bitCast(flags));
}

extern fn setenv(name: [*:0]const u8, value: [*:0]const u8, overwrite: c_int) c_int;
fn setupNeuronEnv(io: std.Io) !void {
    var buf: [256]u8 = undefined;
    _ = setenv(
        "NEURON_RT_ROOT_COMM_ID",
        try std.fmt.bufPrintZ(&buf, "127.0.0.1:{d}", .{try findFreeTcpPort(io)}),
        1,
    );
    _ = setenv(
        "NEURON_INTERNAL_PJRT_C_API_VERSION",
        std.fmt.comptimePrint("{d}.{d}", .{
            c.PJRT_API_MAJOR,
            c.PJRT_API_MINOR,
        }),
        1,
    );
    _ = setenv(
        "NEURON_RT_STOCHASTIC_ROUNDING_EN",
        "1",
        1,
    );
}

fn pyStatusCheck(status: c.PyStatus) void {
    if (c.PyStatus_Exception(status) != 0) {
        if (c.PyStatus_IsExit(status) != 0) {
            std.process.exit(@intCast(status.exitcode));
        }
        c.Py_ExitStatusException(status);
    }
}

fn toPosixPathW(file_path: []const u8) error{NameTooLong}![std.posix.PATH_MAX - 1:0]c.wchar_t {
    if (file_path.len >= std.posix.PATH_MAX) return error.NameTooLong;

    var path_with_null: [std.posix.PATH_MAX - 1:0]c.wchar_t = undefined;
    const len = c.mbstowcs(&path_with_null, file_path.ptr, file_path.len);
    path_with_null[len] = 0;
    return path_with_null;
}

fn setupPythonEnv(sandbox_path: []const u8) !void {
    const Static = struct {
        var py_config: c.PyConfig = undefined;
    };

    {
        var preconfig: c.PyPreConfig = undefined;
        c.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyStatusCheck(c.Py_PreInitialize(&preconfig));
    }

    c.PyConfig_InitIsolatedConfig(&Static.py_config);

    Static.py_config.module_search_paths_set = 1;
    Static.py_config.optimization_level = 2;
    Static.py_config.write_bytecode = 0;

    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const home = try std.fmt.bufPrintZ(&buf, "{f}{d}.{d}", .{
            std.fs.path.fmtJoin(&.{
                sandbox_path,
                "lib",
                "python",
            }),
            c.PY_MAJOR_VERSION,
            c.PY_MINOR_VERSION,
        });
        pyStatusCheck(c.PyConfig_SetBytesString(&Static.py_config, &Static.py_config.home, home));
        pyStatusCheck(c.PyWideStringList_Append(&Static.py_config.module_search_paths, &try toPosixPathW(home)));
    }

    {
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        const site_packages = try stdx.fs.path.bufJoin(&buf, &.{
            sandbox_path,
            "site-packages",
        });
        pyStatusCheck(c.PyWideStringList_Append(&Static.py_config.module_search_paths, &try toPosixPathW(site_packages)));
    }

    pyStatusCheck(c.Py_InitializeFromConfig(&Static.py_config));

    // release the GIL
    _ = c.PyEval_SaveThread();
}

// Duplicates a PJRT Api object while being careful about struct size differences
fn dupePjrtApi(api: *c.PJRT_Api) c.PJRT_Api {
    var ret: c.PJRT_Api = undefined;
    const struct_size = @min(@sizeOf(c.PJRT_Api), api.struct_size);
    @memcpy(
        std.mem.asBytes(&ret)[0..struct_size],
        std.mem.asBytes(api)[0..struct_size],
    );
    return ret;
}

fn getPjrtApi() !*c.PJRT_Api {
    const Static = struct {
        var inner: *c.PJRT_Api = undefined;
        var proxy: c.PJRT_Api = undefined;
    };

    var sandbox_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = try stdx.fs.path.bufJoin(&sandbox_path_buf, &.{
        stdx.fs.selfSharedObjectDirPath(),
        "..",
    });

    var io_: std.Io.Threaded = .init_single_threaded;

    try setupNeuronEnv(io_.io());
    try setupPythonEnv(sandbox_path);

    Static.inner = blk: {
        const GetPjrtApi_inner = GetPjrtApi_blk: {
            var lib: std.DynLib = .{
                .inner = .{
                    .handle = handle_blk: {
                        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
                        const library = try stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libneuronpjrt.so" });
                        break :handle_blk std.c.dlopen(library, .{ .LAZY = true, .NODELETE = true }) orelse {
                            log.err("Unable to dlopen plugin: {?s}", .{std.mem.span(std.c.dlerror())});
                            return error.DlOpenFailed;
                        };
                    },
                },
            };

            break :GetPjrtApi_blk lib.lookup(*const fn () callconv(.c) *c.PJRT_Api, "GetPjrtApi") orelse {
                log.err("Unable to find symbol GetPjrtApi in plugin: {?s}", .{std.mem.span(std.c.dlerror())});
                return error.SymbolNotFound;
            };
        };

        break :blk GetPjrtApi_inner();
    };

    Static.proxy = dupePjrtApi(Static.inner);
    // Setup the API proxy functions
    Static.proxy.PJRT_Plugin_Attributes = &struct {
        const STRUCT_SIZE = 24; // according to the failing assertion

        fn call(args: [*c]c.PJRT_Plugin_Attributes_Args) callconv(.c) ?*c.PJRT_Error {
            var new_args = args.*;
            new_args.struct_size = @min(new_args.struct_size, STRUCT_SIZE);
            return Static.inner.PJRT_Plugin_Attributes.?(&new_args);
        }
    }.call;

    return &Static.proxy;
}

pub export fn GetPjrtApi() ?*c.PJRT_Api {
    return getPjrtApi() catch |err| {
        log.err("Failed to get PJRT API: {}", .{err});
        return null;
    };
}
