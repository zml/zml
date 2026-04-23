const std = @import("std");

const c = @import("c");
const embedded_kernel_rewriter = @import("nki/embedded_kernel_rewriter.zig");
const target_resolution = @import("target_resolution.zig");
const whole_program_compiler = @import("whole_program_compiler.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/libneuronxla");

// `libneuronxla` is the whole-program Neuron compile hook loaded by the
// upstream Neuron Python bridge. Its responsibilities stop at the outer
// StableHLO program boundary:
// - accept the bridge callback arguments
// - let `platforms/neuron/nki` materialize embedded custom kernels, if any
// - invoke `neuronx-cc` on the resulting StableHLO module
// - wrap the produced NEFF back into a single `AwsNeuronNeff` custom-call
//
// Embedded-kernel lowering lives under `platforms/neuron/nki`; this file owns
// only the top-level program compile path.

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
            .ml_doc = "No-op callback expected by the Neuron Python bridge.",
        },
        .{
            .ml_name = "neuronx_cc",
            .ml_meth = @ptrCast(&neuronx_cc),
            .ml_flags = c.METH_FASTCALL,
            .ml_doc = "Compile HLO to NEFF and return a rewritten custom-call HLO module.",
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

// The upstream Neuron Python bridge imports this symbol during module setup and
// expects it to exist even though ZML does not use it for work.
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

// Entry point used by the upstream Neuron bridge. It lowers embedded NKI
// kernels, compiles the whole StableHLO program with `neuronx-cc`, and returns
// a rewritten HLO module containing a single top-level `AwsNeuronNeff` call.
fn neuronx_cc_(self: ?*c.PyObject, args_: [*c]*c.PyObject, nargs_: c.Py_ssize_t) !?*c.PyObject {
    const state: *ModuleState = @ptrCast(@alignCast(c.PyModule_GetState(self)));
    const io = state.threaded.io();

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    const args = args_[0..@intCast(nargs_)];

    const code = PyBytes_AsStringAndSize(args[0]);
    const platform_version = PyBytes_AsStringAndSize(args[2]);

    const target = target_resolution.resolveTarget(platform_version) catch {
        log.err("Unknown platform version: {s}\n", .{platform_version});
        return error.UnknownPlatformVersion;
    };

    var tmp_dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const sandbox_path = try std.fmt.bufPrint(&tmp_dir_buf, "{s}{d}", .{
        "zml-neuronxcc-",
        std.Io.Clock.now(.real, io).toNanoseconds(),
    });

    const tmp_root = if (std.c.getenv("TMPDIR")) |tmpdir|
        std.mem.span(tmpdir)
    else
        "/tmp";
    const tmp_root_dir = try std.Io.Dir.openDir(.cwd(), io, tmp_root, .{});
    defer tmp_root_dir.close(io);

    const tmp_dir = try tmp_root_dir.createDirPathOpen(io, sandbox_path, .{ .permissions = .fromMode(0o700) });

    defer {
        tmp_root_dir.deleteTree(io, sandbox_path) catch |err| {
            log.err("Error deleting temporary directory: {}", .{err});
        };
    }

    // Lower any embedded NKI kernels in-place first. If the module contains no
    // synthetic ZML NKI custom-calls, the original StableHLO bytes are passed
    // through unchanged.
    const compile_hlo = (try embedded_kernel_rewriter.rewriteCustomCalls(
        arena.allocator(),
        io,
        tmp_dir,
        code,
        target,
    )) orelse code;
    const neff_file = try whole_program_compiler.compileHloToNeff(io, tmp_dir, compile_hlo, target);
    const neff_hlo_bytes = try whole_program_compiler.wrapNeffAsCustomCall(arena.allocator(), io, code, neff_file);

    return c.PyTuple_Pack(
        2,
        c.PyLong_FromLongLong(0),
        c.PyBytes_FromStringAndSize(@ptrCast(neff_hlo_bytes), @intCast(neff_hlo_bytes.len)),
    );
}

// Convert Zig errors into the tuple-shaped Python result expected by the
// upstream Neuron bridge.
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

// Python extension module initializer.
pub export fn PyInit_libneuronxla() callconv(.c) ?*c.PyObject {
    return c.PyModuleDef_Init(&module_def);
}
