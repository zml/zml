const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");

const hlo_neff = @import("hlo_neff.zig");
const nki_embed = @import("nki_embed.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/libneuronxla");

pub fn makeTempDir(io: std.Io, buf: []u8, prefix: []const u8) ![]const u8 {
    const tmp_dir = std.c.getenv("TMPDIR") orelse "/tmp";
    const ret = try std.fmt.bufPrint(buf, "{s}{s}{s}{d}", .{
        tmp_dir,
        std.Io.Dir.path.sep_str_posix,
        prefix,
        std.Io.Clock.now(.real, io).toNanoseconds(),
    });
    try std.Io.Dir.createDir(.cwd(), io, ret, .fromMode(0o700));
    return ret;
}

const ModuleState = struct {
    threaded: std.Io.Threaded,
};

var module_def: c.PyModuleDef = .{
    .m_base = std.mem.zeroes(c.PyModuleDef_Base),
    .m_doc = "Zig bindings for libneuronxla",
    .m_name = "libneuronxla",
    .m_size = @sizeOf(ModuleState),
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
        std.mem.zeroes(c.PyMethodDef),
    }),
    .m_slots = @constCast(&[_]c.PyModuleDef_Slot{
        .{ .slot = c.Py_mod_exec, .value = @ptrCast(@constCast(&module_exec)) },
        std.mem.zeroes(c.PyModuleDef_Slot),
    }),
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

fn module_exec(module: ?*c.PyObject) callconv(.c) c_int {
    const state: *ModuleState = @ptrCast(@alignCast(c.PyModule_GetState(module)));
    state.* = .{
        .threaded = .init(std.heap.c_allocator, .{}),
    };
    return 0;
}

fn hook(self: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = self;
    _ = args;
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
    const state: *ModuleState = @ptrCast(@alignCast(c.PyModule_GetState(self)));
    const io = state.threaded.io();

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    const args = args_[0..@intCast(nargs_)];

    const code = PyBytes_AsStringAndSize(args[0]);
    const platform_version = PyBytes_AsStringAndSize(args[2]);

    const target = std.StaticStringMap([]const u8).initComptime(.{
        .{ "1.0", "inf1" },
        .{ "2.0", "trn1" },
        .{ "3.0", "trn2" },
    }).get(platform_version) orelse {
        log.err("Unknown platform version: {s}\n", .{platform_version});
        return error.UnknownPlatformVersion;
    };

    var tmp_dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tmp_dir = try makeTempDir(io, &tmp_dir_buf, "zml-neuronxcc-");
    var preserve_tmp = false;
    const may_have_nki = std.mem.indexOf(u8, code, nki_embed.zml_neuron_nki_target) != null;
    if (may_have_nki) {
        preserve_tmp = true;
    }
    defer if (!preserve_tmp) {
        std.Io.Dir.deleteTree(.cwd(), io, tmp_dir) catch |err| {
            log.err("Error deleting temporary directory {s}: {}\n", .{ tmp_dir, err });
        };
    };

    const rewritten_hlo = try nki_embed.rewriteCustomCalls(arena.allocator(), io, code, tmp_dir, target);
    const compile_hlo = rewritten_hlo orelse code;

    const code_file = try std.Io.Dir.path.join(arena.allocator(), &.{ tmp_dir, "file.code" });
    try std.Io.Dir.writeFile(.cwd(), io, .{
        .data = compile_hlo,
        .sub_path = code_file,
        .flags = .{ .truncate = true },
    });

    const neff_file = try std.Io.Dir.path.join(arena.allocator(), &.{ tmp_dir, "file.neff" });

    var neuronx_cc_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const neuronx_cc_path = try stdx.Io.Dir.path.bufJoin(&neuronx_cc_buf, &.{
        stdx.process.selfSharedObjectDirPath(),
        "..",
        "bin",
        "neuronx-cc",
    });

    {
        const gil_state = c.PyEval_SaveThread();
        defer c.PyEval_RestoreThread(gil_state);

        var child = try std.process.spawn(io, .{
            .argv = &.{
                neuronx_cc_path,
                "compile",
                "--framework=XLA",
                "--target",
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
            },
            .stdin = .ignore,
            .stdout = .inherit,
            .stderr = .inherit,
            .cwd = .{ .path = tmp_dir },
        });
        const term = try child.wait(io);
        if (term.exited != 0) {
            log.err("neuronx-cc exited with code {}", .{term.exited});
            return error.NeuronxCcFailed;
        }
    }

    // neuronx-cc exits 0 even on compilation failure
    std.Io.Dir.access(.cwd(), io, neff_file, .{}) catch |err| {
        log.err("neuronx-cc did not produce output NEFF {s}: {}", .{ neff_file, err });
        return error.NeuronxCcFailed;
    };

    const neff_hlo_bytes = hlo_neff.wrapNeffAsCustomCall(arena.allocator(), io, code, neff_file) catch |err| {
        log.err("Error wrapping NEFF as custom call: {}\n", .{err});
        return err;
    };

    return c.PyTuple_Pack(
        2,
        c.PyLong_FromLongLong(0),
        c.PyBytes_FromStringAndSize(@ptrCast(neff_hlo_bytes), @intCast(neff_hlo_bytes.len)),
    );
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
