const std = @import("std");

const c = @import("c");
const neuron_nki = @import("platforms/neuron/nki");
const stdx = @import("stdx");
const upb = @import("upb");

const log = std.log.scoped(.@"zml/platforms/neuron/libneuronxla");
const aws_neuron_neff_target = "AwsNeuronNeff";

const Request = struct {
    hlo_code: []const u8,
    platform_version: []const u8,
    target_hint: ?[]const u8 = null,
};

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

fn logNeuronxCcArgs(args: []const *c.PyObject) void {
    for (args, 0..) |arg, i| {
        if (c.PyBytes_Check(arg) != 0) {
            const bytes = PyBytes_AsStringAndSize(arg);
            if (i == 0) {
                log.info("neuronx_cc arg[{d}] = bytes(len={d})", .{ i, bytes.len });
            } else {
                log.info("neuronx_cc arg[{d}] = bytes({s})", .{ i, bytes });
            }
            continue;
        }
        if (c.PyUnicode_Check(arg) != 0) {
            var len: c.Py_ssize_t = undefined;
            const ptr = c.PyUnicode_AsUTF8AndSize(arg, &len) orelse {
                log.info("neuronx_cc arg[{d}] = unicode(<decode-failed>)", .{i});
                continue;
            };
            log.info("neuronx_cc arg[{d}] = unicode({s})", .{ i, ptr[0..@intCast(len)] });
            continue;
        }

        const type_obj = c.Py_TYPE(arg);
        const type_name = if (type_obj != null and type_obj.*.tp_name != null)
            std.mem.span(type_obj.*.tp_name)
        else
            "<unknown>";
        log.info("neuronx_cc arg[{d}] = <{s}>", .{ i, type_name });
    }
}

fn pyStringArgSlice(object: *c.PyObject) ?[]const u8 {
    if (c.PyBytes_Check(object) != 0) {
        return PyBytes_AsStringAndSize(object);
    }
    if (c.PyUnicode_Check(object) != 0) {
        var len: c.Py_ssize_t = undefined;
        const ptr = c.PyUnicode_AsUTF8AndSize(object, &len) orelse return null;
        return ptr[0..@intCast(len)];
    }
    return null;
}

fn makeTempDir(io: std.Io, buf: []u8, prefix: []const u8) ![]const u8 {
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

fn isSupportedNeuronTarget(target: []const u8) bool {
    return std.mem.eql(u8, target, "inf1") or
        std.mem.eql(u8, target, "inf2") or
        std.mem.eql(u8, target, "trn1") or
        std.mem.eql(u8, target, "trn2");
}

fn inferNeuronTargetFromPlatformVersion(platform_version: []const u8) ?[]const u8 {
    return std.StaticStringMap([]const u8).initComptime(.{
        .{ "1.0", "inf1" },
        .{ "2.0", "trn1" },
        .{ "3.0", "trn2" },
    }).get(platform_version);
}

fn resolveNeuronTarget(platform_version: []const u8, maybe_target: ?[]const u8) ![]const u8 {
    if (maybe_target) |target| {
        if (isSupportedNeuronTarget(target)) {
            if (inferNeuronTargetFromPlatformVersion(platform_version)) |inferred_target| {
                if (!std.mem.eql(u8, inferred_target, target)) {
                    log.warn(
                        "Neuron target hint {s} overrides platform version {s} inference ({s})",
                        .{ target, platform_version, inferred_target },
                    );
                }
            }
            return target;
        }
        log.debug("Ignoring non-target secondary neuronx_cc argument: {s}", .{target});
    }

    return inferNeuronTargetFromPlatformVersion(platform_version) orelse {
        log.err("Unknown platform version: {s}", .{platform_version});
        return error.UnknownPlatformVersion;
    };
}

fn neuronCcVerboseLevel() []const u8 {
    return switch (std.options.log_level) {
        .debug => "debug",
        .info => "info",
        .warn => "warning",
        .err => "error",
    };
}

fn setInstructionFrontendAttribute(
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    key: []const u8,
    value: []const u8,
) void {
    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(instruction, upb_arena);
    const map = c._xla_FrontendAttributes_map_mutable_upb_map(attrs, upb_arena);
    _ = c.upb_Map_Set(
        map,
        .{ .str_val = upb.stringView(key) },
        .{ .str_val = upb.stringView(value) },
        upb_arena,
    );
}

fn setValidInputs(
    allocator: std.mem.Allocator,
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    operand_count: usize,
) !void {
    const valid_inputs_value: []const u8 = if (operand_count == 0)
        &[_]u8{}
    else blk: {
        const out = try allocator.alloc(u8, operand_count * 2 - 1);
        for (out, 0..) |*char, i| {
            char.* = if (i % 2 == 0) '1' else ',';
        }
        break :blk out;
    };
    setInstructionFrontendAttribute(upb_arena, instruction, "valid_inputs", valid_inputs_value);
}

fn wrapNeffAsCustomCall(
    allocator: std.mem.Allocator,
    io: std.Io,
    hlo_code: []const u8,
    neff_file_path: []const u8,
) ![]const u8 {
    // Rebuild the entry computation so the runtime sees a single
    // `AwsNeuronNeff` custom-call with the original parameters preserved.
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
    c.xla_HloInstructionProto_set_custom_call_target(fused_root, upb.stringView(aws_neuron_neff_target));
    c.xla_HloInstructionProto_set_backend_config(
        fused_root,
        upb.stringView(
            try stdx.Io.Dir.readFileAlloc(.cwd(), io, neff_file_path, allocator, .unlimited),
        ),
    );

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

    try setValidInputs(allocator, upb_arena, fused_root, parameters_len);
    log.info("Wrapped {s} as AwsNeuronNeff with {d} parameter(s)", .{ neff_file_path, parameters_len });

    return try upb.serialize(hlo_module, upb_arena);
}

fn compileHloToNeff(
    allocator: std.mem.Allocator,
    io: std.Io,
    hlo_code: []const u8,
    tmp_dir: []const u8,
    target: []const u8,
) ![]const u8 {
    const code_file = try std.Io.Dir.path.join(allocator, &.{ tmp_dir, "file.code" });
    try std.Io.Dir.writeFile(.cwd(), io, .{
        .data = hlo_code,
        .sub_path = code_file,
        .flags = .{ .truncate = true },
    });

    const neff_file = try std.Io.Dir.path.join(allocator, &.{ tmp_dir, "file.neff" });

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

        const cc_verbose = neuronCcVerboseLevel();
        log.info("Launching neuronx-cc with --verbose={s}", .{cc_verbose});

        var child = try std.process.spawn(io, .{
            .argv = &.{
                neuronx_cc_path,
                "compile",
                "--framework=XLA",
                "--target",
                target,
                try std.fmt.allocPrint(allocator, "--verbose={s}", .{cc_verbose}),
                "--enable-internal-neff-wrapper",
                "--output",
                neff_file,
                "--optlevel=1",
                "--model-type=transformer",
                "--auto-cast=none",
                "--enable-fast-loading-neuron-binaries",
                "--logfile",
                try std.Io.Dir.path.join(allocator, &.{ tmp_dir, "log-neuron-cc.txt" }),
                try std.fmt.allocPrint(allocator, "--logfile-verbose={s}", .{cc_verbose}),
                code_file,
            },
            .stdin = .ignore,
            .stdout = .inherit,
            .stderr = .inherit,
            .cwd = .{ .path = tmp_dir },
        });
        const term = try child.wait(io);
        switch (term) {
            .exited => |exit_code| {
                if (exit_code != 0) {
                    log.err("neuronx-cc exited with code {}", .{exit_code});
                    return error.NeuronxCcFailed;
                }
            },
            .signal => |sig| {
                log.err("neuronx-cc terminated by signal {}", .{sig});
                return error.NeuronxCcFailed;
            },
            else => |status| {
                log.err("neuronx-cc terminated unexpectedly: {}", .{status});
                return error.NeuronxCcFailed;
            },
        }
    }

    std.Io.Dir.access(.cwd(), io, neff_file, .{}) catch |err| {
        log.err("neuronx-cc did not produce output NEFF {s}: {}", .{ neff_file, err });
        return error.NeuronxCcFailed;
    };

    return neff_file;
}

fn compileToWrappedNeff(
    allocator: std.mem.Allocator,
    io: std.Io,
    request: Request,
) ![]const u8 {
    const target = try resolveNeuronTarget(request.platform_version, request.target_hint);
    log.info("Compiling Neuron HLO for target {s} (platform version {s})", .{ target, request.platform_version });

    var tmp_dir_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const tmp_dir = try makeTempDir(io, &tmp_dir_buf, "zml-neuronxcc-");
    defer {
        std.Io.Dir.deleteTree(.cwd(), io, tmp_dir) catch |err| {
            log.err("Error deleting temporary directory {s}: {}", .{ tmp_dir, err });
        };
    }

    // Embedded NKI kernels are materialized into AwsNeuronCustomNativeKernel
    // custom-calls before the whole StableHLO program is compiled by neuronx-cc.
    const compile_hlo = try neuron_nki.materializeEmbeddedKernels(
        allocator,
        io,
        request.hlo_code,
        tmp_dir,
        target,
    );
    const neff_file = try compileHloToNeff(allocator, io, compile_hlo, tmp_dir, target);
    return try wrapNeffAsCustomCall(allocator, io, request.hlo_code, neff_file);
}

fn neuronx_cc_(self: ?*c.PyObject, args_: [*c]*c.PyObject, nargs_: c.Py_ssize_t) !?*c.PyObject {
    const state: *ModuleState = @ptrCast(@alignCast(c.PyModule_GetState(self)));
    const io = state.threaded.io();

    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();

    const args = args_[0..@intCast(nargs_)];
    log.info("neuronx_cc called with {d} arg(s)", .{args.len});
    logNeuronxCcArgs(args);

    const code = PyBytes_AsStringAndSize(args[0]);
    const platform_version = PyBytes_AsStringAndSize(args[2]);
    const target_hint = if (args.len > 1) pyStringArgSlice(args[1]) else null;

    const neff_hlo_bytes = try compileToWrappedNeff(arena.allocator(), io, .{
        .hlo_code = code,
        .platform_version = platform_version,
        .target_hint = target_hint,
    });

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
