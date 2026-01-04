const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");
const upb = @import("upb");

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
        .{ .slot = c.Py_mod_exec, .value = @ptrCast(@constCast(&module_exec)) },
        .{},
    }),
    .m_traverse = null,
    .m_clear = null,
    .m_free = null,
};

fn module_exec(module: ?*c.PyObject) callconv(.c) c_int {
    _ = module;
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

fn wrapNeffAsCustomCall(allocator: std.mem.Allocator, hlo_code: []const u8, neff_file_path: []const u8) ![]const u8 {
    var upb_alloc: upb.Allocator = .init(allocator);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());

    const hlo_module = try upb.parse(c.xla_HloModuleProto, upb_arena, hlo_code);

    const entry = blk: {
        var size: usize = undefined;
        const computations = c.xla_HloModuleProto_mutable_computations(hlo_module, &size)[0..size];
        for (computations) |comp| {
            if (c.xla_HloComputationProto_id(comp) == c.xla_HloModuleProto_entry_computation_id(hlo_module)) {
                break :blk comp;
            }
        } else return error.ComputationNotFound;
    };

    const entry_instructions = blk: {
        var size: usize = undefined;
        break :blk c.xla_HloComputationProto_instructions(entry, &size)[0..size];
    };
    c.xla_HloComputationProto_clear_instructions(entry);

    const fused_root = blk: {
        for (entry_instructions) |instruction| {
            if (c.xla_HloInstructionProto_id(instruction) == c.xla_HloComputationProto_root_id(entry)) {
                break :blk try upb.shallowClone(c.xla_HloInstructionProto, upb_arena, instruction);
            }
        } else return error.ComputationNotFound;
    };

    c.xla_HloInstructionProto_set_opcode(fused_root, upb.stringView("custom-call"));
    c.xla_HloInstructionProto_set_custom_call_target(fused_root, upb.stringView("AwsNeuronNeff"));
    c.xla_HloInstructionProto_set_backend_config(fused_root, blk: {
        const neff_file = try std.fs.openFileAbsolute(neff_file_path, .{});
        defer neff_file.close();
        const stat = try neff_file.stat();
        const neff_buf = try allocator.alloc(u8, stat.size);
        const size = try neff_file.readAll(neff_buf);
        break :blk upb.stringView(neff_buf[0..size]);
    });

    const parameters_len = blk: {
        var size: usize = undefined;
        _ = c.xla_ProgramShapeProto_parameters(
            c.xla_HloComputationProto_program_shape(entry),
            &size,
        );
        break :blk size;
    };

    {
        var operand_ids: std.ArrayList(i64) = .initBuffer(c.xla_HloInstructionProto_resize_operand_ids(fused_root, parameters_len + 1, upb_arena)[0 .. parameters_len + 1]);
        var new_instructions: std.ArrayList(*const c.xla_HloInstructionProto) = .initBuffer(@ptrCast(c.xla_HloComputationProto_resize_instructions(entry, parameters_len + 1, upb_arena)[0 .. parameters_len + 1]));
        for (entry_instructions) |instruction| {
            if (std.mem.eql(u8, upb.slice(c.xla_HloInstructionProto_opcode(instruction)) orelse continue, "parameter")) {
                const id = c.xla_HloInstructionProto_id(instruction);
                operand_ids.appendAssumeCapacity(id);
                new_instructions.appendAssumeCapacity(instruction);
            }
        }
        new_instructions.appendAssumeCapacity(fused_root);
    }

    {
        const fa = c.xla_HloInstructionProto_mutable_frontend_attributes(fused_root, upb_arena);
        const map = c._xla_FrontendAttributes_map_mutable_upb_map(fa, upb_arena);
        _ = c.upb_Map_Set(
            map,
            .{ .str_val = upb.stringView("valid_inputs") },
            .{ .str_val = blk: {
                const valid_inputs_value = try allocator.alloc(u8, parameters_len * 2 - 1);
                for (valid_inputs_value, 0..) |*char, i| {
                    char.* = if (i % 2 == 0) '1' else ',';
                }
                break :blk upb.stringView(valid_inputs_value);
            } },
            upb_arena,
        );
    }

    return try upb.serialize(hlo_module, upb_arena);
}

fn neuronx_cc_(self: ?*c.PyObject, args_: [*c]*c.PyObject, nargs_: c.Py_ssize_t) !?*c.PyObject {
    _ = self;

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
    const tmp_dir = try makeTempDir(&tmp_dir_buf, "zml-neuronxcc-");
    defer std.fs.deleteTreeAbsolute(tmp_dir) catch |err| {
        log.err("Error deleting temporary directory {s}: {}\n", .{ tmp_dir, err });
    };

    const code_file = try std.fs.path.join(arena.allocator(), &.{ tmp_dir, "file.code" });
    {
        const file = try std.fs.cwd().createFile(code_file, .{ .truncate = true });
        defer file.close();
        try file.writeAll(code);
    }

    const neff_file = try std.fs.path.join(arena.allocator(), &.{ tmp_dir, "file.neff" });

    var neuronx_cc_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    var child = std.process.Child.init(&.{
        try stdx.fs.path.bufJoin(&neuronx_cc_buf, &.{
            stdx.fs.selfSharedObjectDirPath(),
            "..",
            "bin",
            "neuronx-cc",
        }),
        "compile",
        "--framework=XLA",
        "--target",
        target,
        "--verbose=info",
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
