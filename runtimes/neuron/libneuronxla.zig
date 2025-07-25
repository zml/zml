const std = @import("std");
const c = @import("c");

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
            .ml_flags = c.METH_VARARGS,
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

fn neuronx_cc(self: ?*c.PyObject, args: ?*c.PyObject) callconv(.c) ?*c.PyObject {
    _ = self;
    _ = args;
    std.debug.print("Hello from Zig! 2222222\n", .{});

    const none = c.Py_None();
    c.Py_IncRef(none);

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


pub export fn PyInit_libneuronxla() callconv(.c) ?*c.PyObject {
    return c.PyModuleDef_Init(&module_def);
}
