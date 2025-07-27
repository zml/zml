const std = @import("std");

const c = @import("c");
const hlo_proto = @import("hlo_proto");
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

fn wrapNeffAsCustomCall(allocator: std.mem.Allocator, hlo_code: []const u8, neff_file_path: []const u8) ![]const u8 {
    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const hlo_module = try hlo_proto.HloModuleProto.decode(hlo_code, arena.allocator());
    var entry = blk: {
        for (hlo_module.computations.items) |*comp| {
            if (comp.id == hlo_module.entry_computation_id) {
                break :blk comp;
            }
        } else return error.ComputationNotFound;
    };

    var fused_root: hlo_proto.HloInstructionProto = blk: {
        for (entry.instructions.items) |inst| {
            if (inst.id == entry.root_id) {
                break :blk try inst.dupe(arena.allocator());
            }
        } else return error.CustomCallNotFound;
    };

    fused_root.opcode = .{ .Const = "custom-call" };
    fused_root.custom_call_target = .{ .Const = "AwsNeuronNeff" };
    fused_root.backend_config = .{ .Owned = blk: {
        const neff_file = try std.fs.openFileAbsolute(neff_file_path, .{});
        defer neff_file.close();
        const stat = try neff_file.stat();
        const neff_buf = try arena.allocator().alloc(u8, @intCast(stat.size));
        _ = try neff_file.readAll(neff_buf);
        break :blk neff_buf;
    } };

    const parameters_len = entry.program_shape.?.parameters.items.len;
    var new_entry_instructions: std.ArrayListUnmanaged(hlo_proto.HloInstructionProto) = try .initCapacity(
        arena.allocator(),
        parameters_len + 1, // params + fused_root
    );

    fused_root.operand_ids.clearRetainingCapacity();
    for (entry.instructions.items) |inst| {
        if (std.mem.eql(u8, inst.opcode.getSlice(), "parameter")) {
            try fused_root.operand_ids.append(arena.allocator(), inst.id);
            try new_entry_instructions.append(arena.allocator(), inst);
        }
    }

    try fused_root.frontend_attributes.?.map.append(arena.allocator(), .{
        .key = .{ .Const = "valid_inputs" },
        .value = .{ .Owned = blk: {
            const valid_inputs_value = try arena.allocator().alloc(u8, parameters_len * 2 - 1);
            for (valid_inputs_value, 0..) |*char, i| {
                char.* = if (i % 2 == 0) '1' else ',';
            }
            break :blk valid_inputs_value;
        } },
    });
    try new_entry_instructions.append(arena.allocator(), fused_root);
    entry.instructions = new_entry_instructions;

    return hlo_module.encode(allocator);
}

fn neuronx_cc_(self: ?*c.PyObject, args_: [*c]*c.PyObject, nargs_: c.Py_ssize_t) !?*c.PyObject {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    _ = self;
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
        "--target",
        target,
        "--verbose=debug",
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
    child.cwd = tmp_dir;
    _ = try child.spawnAndWait();

    std.debug.print(">>>> {s}\n", .{tmp_dir});

    const neff_hlo_bytes = wrapNeffAsCustomCall(arena.allocator(), code, neff_file) catch |err| {
        log.err("Error wrapping NEFF as custom call: {}\n", .{err});
        return err;
    };

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
