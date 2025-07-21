const builtin = @import("builtin");
const std = @import("std");

const asynk = @import("async");
const bazel_builtin = @import("bazel_builtin");
const c = @import("c");
const libneuronxla_pyenv = @import("libneuronxla_pyenv");
const pjrt = @import("pjrt");
const runfiles = @import("runfiles");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtime/neuron");

pub fn isEnabled() bool {
    return @hasDecl(c, "ZML_RUNTIME_NEURON");
}

fn hasNeuronDevice() bool {
    asynk.File.access("/dev/neuron0", .{ .mode = .read_only }) catch return false;
    return true;
}

fn isRunningOnEC2() !bool {
    const AmazonEC2 = "Amazon EC2";

    var f = try asynk.File.open("/sys/devices/virtual/dmi/id/sys_vendor", .{ .mode = .read_only });
    defer f.close() catch {};

    var buf = [_]u8{0} ** AmazonEC2.len;
    _ = try f.reader().readAll(&buf);

    return std.mem.eql(u8, &buf, AmazonEC2);
}

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

fn initialize() !void {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    const allocator = arena.allocator();
    defer arena.deinit();

    {
        var preconfig: c.PyPreConfig = undefined;
        c.PyPreConfig_InitIsolatedConfig(&preconfig);
        preconfig.utf8_mode = 1;
        pyErrorOrExit(c.Py_PreInitialize(&preconfig));
    }

    var config: c.PyConfig = undefined;
    c.PyConfig_InitIsolatedConfig(&config);
    defer c.PyConfig_Clear(&config);

    var r_ = try runfiles.Runfiles.create(.{ .allocator = allocator }) orelse return error.Unavailable;
    const r = r_.withSourceRepo(bazel_builtin.current_repository);

    var buf: [std.fs.max_path_bytes]u8 = undefined;
    var wbuf: [std.fs.max_path_bytes:0]c.wchar_t = undefined;

    {
        const path = (try r.rlocation(libneuronxla_pyenv.home, &buf)).?;
        const wpath = toWchar(std.fs.path.dirname(path).?, &wbuf);
        pyErrorOrExit(c.PyConfig_SetString(&config, &config.home, wpath.ptr));
    }

    {
        config.module_search_paths_set = 1;
        for (libneuronxla_pyenv.modules) |module| {
            const path = (try r.rlocation(module, &buf)).?;
            const wline = toWchar(std.fs.path.dirname(path).?, &wbuf);
            pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, wline.ptr));
        }
    }

    {
        const neuronx_cc = (try r.rlocation("zml/runtimes/neuron/neuronx-cc/neuronx-cc", &buf)).?;
        const neuronx_cc_path = std.fs.path.dirname(neuronx_cc).?;
        const path = std.posix.getenv("PATH") orelse "";
        const new_path = try std.fmt.allocPrintZ(allocator, "{s}:{s}", .{ neuronx_cc_path, path });
        _ = c.setenv("PATH", new_path.ptr, 1);
    }

    pyErrorOrExit(c.Py_InitializeFromConfig(&config));

    // release the GIL
    _ = c.PyEval_SaveThread();
}

fn comptimeStrJoin(comptime separator: [:0]const u8, comptime slices: []const [:0]const u8) [:0]const u8 {
    comptime var ret = slices[0];
    inline for (slices[1..]) |slice| {
        ret = ret ++ separator ++ slice;
    }
    return ret;
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

pub fn load() !*const pjrt.Api {
    if (comptime !isEnabled()) {
        return error.Unavailable;
    }
    if (comptime builtin.os.tag != .linux) {
        return error.Unavailable;
    }
    if (!(isRunningOnEC2() catch false)) {
        return error.Unavailable;
    }
    if (!hasNeuronDevice()) {
        return error.Unavailable;
    }

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    var r_ = try runfiles.Runfiles.create(.{ .allocator = arena.allocator() }) orelse {
        stdx.debug.panic("Unable to find runfiles", .{});
    };

    const source_repo = bazel_builtin.current_repository;
    const r = r_.withSourceRepo(source_repo);

    var path_buf: [std.fs.max_path_bytes]u8 = undefined;
    const sandbox_path = try r.rlocation("zml/runtimes/neuron/sandbox", &path_buf) orelse {
        log.err("Failed to find sandbox path for NEURON runtime", .{});
        return error.FileNotFound;
    };

    setNeuronCCFlags();
    try initialize();


    return blk: {
        var lib_path_buf: [std.fs.max_path_bytes]u8 = undefined;
        const path = try stdx.fs.path.bufJoinZ(&lib_path_buf, &.{ sandbox_path, "lib", "libpjrt_neuron.so" });
        break :blk asynk.callBlocking(pjrt.Api.loadFrom, .{path});
    };
}
