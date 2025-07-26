const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const log = std.log.scoped(.@"zml/runtimes/neuron/libneuronxla");

pub fn makeTempDir(buf: []u8, prefix: []const u8) ![]const u8 {
    const tmp_dir = std.posix.getenv("TMPDIR") orelse "/tmp";
    const ret = try std.fmt.bufPrint(buf, "{s}{s}{s}{d}", .{
        tmp_dir,
        std.fs.path.sep_str_posix,
        prefix,
        std.time.microTimestamp(),
    });
    try std.fs.makeDirAbsolute(ret);
    return ret;
}

var module_def: c.PyModuleDef = .{
    .m_base = .{},
    .m_name = "libneuronxla",
    .m_doc = "Example module written in Zig.",
    .m_size = 0,
    .m_methods = @constCast(&[_]c.PyMethodDef{
        .{
            .ml_name = "hook",
            .ml_meth = @ptrCast(&hook),
            .ml_flags = c.METH_NOARGS,
            .ml_doc = "Return a greeting from Zig.",
        },
        .{
            .ml_name = "neuronx_cc",
            .ml_meth = @ptrCast(&neuronx_cc),
            .ml_flags = c.METH_FASTCALL,
            .ml_doc = "Return a greeting from Zig.",
        },
        .{},
    }),
    .m_slots = @constCast(&[_]c.PyModuleDef_Slot{
        .{ .slot = c.Py_mod_exec, .value = @constCast(@ptrCast(&module_exec)) },
        .{},
    }),
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

fn module_exec(module: ?*c.PyObject) callconv(.c) c_int {
    _ = module;
    return 0; // 0 = success, -1 = failure
}

fn hook(self: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = self;
    _ = args;
    std.debug.print("Hello from Zig!\n", .{});
    const none = c.Py_None();
    defer c.Py_IncRef(none);
    return none;
}

pub fn PyBytes_AsStringAndSize(object: *c.PyObject) []u8 {
    var buf: [*c]u8 = undefined;
    var len: c.Py_ssize_t = undefined;
    _ = c.PyBytes_AsStringAndSize(object, &buf, &len);
    return buf[0..@intCast(len)];
}

fn neuronx_cc_(self: ?*c.PyObject, args_: [*c]*c.PyObject, nargs_: c.Py_ssize_t) !?*c.PyObject {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    _ = self;
    const args = args_[0..@intCast(nargs_)];

    const code = PyBytes_AsStringAndSize(args[0]);
    const platform_version = PyBytes_AsStringAndSize(args[2]);

    const target: []const u8 = blk: {
        if (std.mem.eql(u8, platform_version, "1.0")) {
            break :blk "--target=inf1";
        } else if (std.mem.eql(u8, platform_version, "2.0")) {
            break :blk "--target=trn1";
        } else if (std.mem.eql(u8, platform_version, "3.0")) {
            break :blk "--target=trn2";
        } else {
            log.err("Unknown platform version: {s}\n", .{platform_version});
            return null;
        }
    };

    var tmp_dir_buf: [std.fs.max_path_bytes]u8 = undefined;
    const tmp_dir = try makeTempDir(&tmp_dir_buf, "zml-neuronxcc-");

    const code_file = try std.fs.path.join(arena.allocator(), &.{ tmp_dir, "file.code" });
    {
        const file = try std.fs.cwd().createFile(code_file, .{ .truncate = true });
        defer file.close();
        try file.writeAll(code);
    }

    const wrapped_neff_hlo_file = try std.fs.path.join(arena.allocator(), &.{ tmp_dir, "wrapped_neff.hlo" });
    _ = wrapped_neff_hlo_file; // autofix
    const neff_file = try std.fs.path.join(arena.allocator(), &.{ tmp_dir, "file.neff" });

    var child = std.process.Child.init(&.{
        "neuronx-cc",
        "compile",
        "--framework=XLA",
        target,
        "--verbose=35",
        "--enable-internal-neff-wrapper",
        "--output",
        neff_file,
        "--optlevel=1",
        // generic is the default, but it fails on transformers, force it
        "--model-type=transformer",
        // disable it, we do our own
        "--auto-cast=none",
        "--enable-fast-loading-neuron-binaries",
        code_file,
    }, arena.allocator());
    child.stdin_behavior = .Ignore;
    child.stdout_behavior = .Inherit;
    child.stderr_behavior = .Inherit;
    _ = try child.spawnAndWait();

    std.debug.print(">>>> {s}\n", .{tmp_dir});

    const none = c.Py_None();
    c.Py_IncRef(none);

    // var arg1: [*c]const u8 = undefined;
    // var arg2: c_int = undefined;
    // c.Py
    // const arg_parse_result = c.PyArg_ParseTuple(args, "s:neuronx_cc", &arg1, &arg2);

    // if (arg_parse_result == 0) {
    //     std.debug.print("Error parsing arguments\n", .{});
    //     c.Py_DecRef(none);
    //     return null;
    // }

    // std.debug.print("arg1: {s}\n", .{arg1});

    const tuple = c.PyTuple_New(2) orelse {
        c.Py_DecRef(none);
        return null;
    };
    const empty_bytes = c.PyBytes_FromStringAndSize(null, 0) orelse {
        c.Py_DecRef(none);
        c.Py_DecRef(tuple);
        return null;
    };
    const py_long = c.PyLong_FromLongLong(0) orelse {
        c.Py_DecRef(none);
        c.Py_DecRef(tuple);
        return null;
    };

    _ = c.PyTuple_SetItem(tuple, 0, py_long);
    _ = c.PyTuple_SetItem(tuple, 1, empty_bytes);

    return tuple;
}

fn neuronx_cc(self: ?*c.PyObject, args_: [*c]*c.PyObject, nargs_: c.Py_ssize_t) callconv(.c) ?*c.PyObject {
    return neuronx_cc_(self, args_, nargs_) catch |err| {
        log.err("Error in neuronx_cc: {}\n", .{err});

        const none = c.Py_None();
        c.Py_IncRef(none);
        const tuple = c.PyTuple_New(2) orelse {
            c.Py_DecRef(none);
            return null;
        };
        _ = c.PyTuple_SetItem(tuple, 0, c.PyLong_FromLongLong(400).?);
        _ = c.PyTuple_SetItem(tuple, 1, none);
        return tuple;
    };
}

pub export fn PyInit_libneuronxla() callconv(.c) ?*c.PyObject {
    return c.PyModuleDef_Init(&module_def);
}
