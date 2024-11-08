const builtin = @import("builtin");
const asynk = @import("async");
const pjrt = @import("pjrt");
const c = @import("c");
const std = @import("std");
const runfiles = @import("runfiles");
const bazel_builtin = @import("bazel_builtin");

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
        const neuronx_cc = (try r.rlocation("zml/runtimes/neuron/neuronx-cc", &buf)).?;
        const neuronx_cc_path = std.fs.path.dirname(neuronx_cc).?;
        const path = std.posix.getenv("PATH") orelse "";
        const new_path = try std.fmt.allocPrintZ(allocator, "{s}:{s}", .{ neuronx_cc_path, path });
        _ = c.setenv("PATH", new_path.ptr, 1);
    }

    {
        config.module_search_paths_set = 1;

        const python_bootstrap_path = (try r.rlocation("zml/runtimes/neuron/python_bootstrap.txt", &buf)).?;
        const python_bootstrap = try asynk.File.open(python_bootstrap_path, .{ .mode = .read_only });
        defer python_bootstrap.close() catch {};

        var step: usize = 0;
        while (try python_bootstrap.reader().readUntilDelimiterOrEof(&buf, '\n')) |line| {
            const wline = toWchar(line, &wbuf);
            pyErrorOrExit(c.PyWideStringList_Append(&config.module_search_paths, wline.ptr));
            if (step == 0) {
                pyErrorOrExit(c.PyConfig_SetString(&config, &config.home, wline.ptr));
            }
            step += 1;
        }
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

    setNeuronCCFlags();
    try initialize();
    return try pjrt.Api.loadFrom("libpjrt_neuron.so");
}
