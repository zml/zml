const std = @import("std");

const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtimes/neuron");

fn comptimeStrJoin(comptime separator: [:0]const u8, comptime slices: []const [:0]const u8) [:0]const u8 {
    comptime var ret = slices[0];
    inline for (slices[1..]) |slice| {
        ret = ret ++ separator ++ slice;
    }
    return ret;
}

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

pub fn setNeuronCCFlags() void {
    // See neuronxcc reference:
    // https://awsdocs-neuron.readthedocs-hosted.com/en/latest/compiler/neuronx-cc/api-reference-guide/neuron-compiler-cli-reference-guide.html#neuron-compiler-cli-reference-guide
    _ = c.setenv("NEURON_CC_FLAGS", comptimeStrJoin(" ", &.{
        // 30% faster, no visible speed difference on llama
        "--optlevel=1",
        // generic is the default, but it fails on transformers, force it
        "--model-type=transformer",
        // disable it, we do our own
        "--auto-cast=none",
        "--enable-fast-loading-neuron-binaries",
    }), 1);

    // Enable stochastic rounding
    // https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/rounding-modes.html
    _ = c.setenv("NEURON_RT_STOCHASTIC_ROUNDING_EN", "1", 1);
}

fn setupPythonEnv(sandbox_path: []const u8) !void {
    {
        var preconfig: c.PyPreConfig = undefined;
        c.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyErrorOrExit(c.Py_PreInitialize(&preconfig));
    }

    var config: c.PyConfig = undefined;
    c.PyConfig_InitIsolatedConfig(&config);
    defer c.PyConfig_Clear(&config);

    {
        var bufZ: [std.fs.max_path_bytes:0]u8 = undefined;
        pyErrorOrExit(c.PyConfig_SetBytesString(&config, &config.home, try stdx.fs.path.bufJoinZ(&bufZ, &.{
            sandbox_path,
            "lib",
            "python3.11",
        })));
    }

    {
        config.module_search_paths_set = 1;
        var buf: [std.fs.max_path_bytes]u8 = undefined;
        pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, &(try toPosixPathW(try stdx.fs.path.bufJoin(&buf, &.{
            sandbox_path,
            "lib",
            "python3.11",
        })))));
        pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, &(try toPosixPathW(try stdx.fs.path.bufJoin(&buf, &.{
            sandbox_path,
            "site-packages",
        })))));
    }

    pyErrorOrExit(c.Py_InitializeFromConfig(&config));

    // release the GIL
    _ = c.PyEval_SaveThread();
}

fn setupBinPath(allocator: std.mem.Allocator, sandbox_path: []const u8) !void {
    var buf: [std.fs.max_path_bytes]u8 = undefined;
    const bin_path = try stdx.fs.path.bufJoin(&buf, &.{ sandbox_path, "bin" });
    std.debug.print(">>>>>>>>> {s}\n\n", .{bin_path});
    const old_path = std.posix.getenv("PATH") orelse "";
    const new_path = try std.fmt.allocPrintZ(allocator, "{s}:{s}", .{ bin_path, old_path });
    _ = c.setenv("PATH", new_path, 1);
}

pub export fn GetPjrtApi() ?*anyopaque {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) catch |err| {
        stdx.debug.panic("Unable to find runfiles: {}", .{err});
    } orelse stdx.debug.panic("Runfiles not availeabwewefle", .{});

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var sandbox_path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = r.rlocation("libpjrt_neuron/sandbox", &sandbox_path_buf) catch |err| {
        stdx.debug.panic("Failed to find sandbox path for NEURON runtime: {}", .{err});
    } orelse stdx.debug.panic("No NEURON sandbox path found", .{});

    {
        setNeuronCCFlags();
        setupBinPath(arena.allocator(), sandbox_path) catch |err| {
            log.err("Failed to setup bin path: {}", .{err});
            return null;
        };
        setupPythonEnv(sandbox_path) catch |err| {
            log.err("Failed to setup Python environment: {}", .{err});
            return null;
        };
    }

    const GetPjrtApi_inner = blk: {
        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const library = stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libneuronpjrt.so" }) catch unreachable;

        var lib: std.DynLib = blk_lib: {
            const handle = std.c.dlopen(library, .{ .LAZY = true, .GLOBAL = true, .NODELETE = true }) orelse {
                log.err("Unable to dlopen plugin: {s}", .{library});
                return null;
            };
            break :blk_lib .{ .inner = .{ .handle = handle } };
        };

        break :blk lib.lookup(*const fn () callconv(.C) *anyopaque, "GetPjrtApi") orelse {
            log.err("Unable to find symbol GetPjrtApi in plugin: {s}", .{library});
            return null;
        };
    };

    return GetPjrtApi_inner();
}
