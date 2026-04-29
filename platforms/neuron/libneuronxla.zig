const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");
const upb = @import("upb");

const custom_call_config = @import("nki_custom_call_config.zig");
const custom_call_mutator = @import("nki_custom_call_mutator.zig");
const kernel_compiler = @import("nki_compiler_client.zig");

const log = std.log.scoped(.@"zml/platforms/neuron/libneuronxla");
const aws_neuron_neff_target = "AwsNeuronNeff";
const zml_neuron_nki_target = "zml$neuron$nki";

// Python extension loaded by the upstream Neuron bridge. This file owns the
// complete hook flow: lower ZML NKI custom-calls, invoke neuronx-cc for the
// whole StableHLO program, then wrap the produced NEFF for PJRT.

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

pub export fn PyInit_libneuronxla() callconv(.c) ?*c.PyObject {
    return c.PyModuleDef_Init(&module_def);
}

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
// kernels, compiles the resulting whole StableHLO program, and returns a new
// HLO module that contains the compiled NEFF as one `AwsNeuronNeff` custom-call.
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

    // defer {
    //     tmp_root_dir.deleteTree(io, sandbox_path) catch |err| {
    //         log.err("Error deleting temporary directory: {}", .{err});
    //     };
    // }

    const compile_hlo = (try rewriteCustomCalls(
        arena.allocator(),
        io,
        tmp_dir,
        code,
        target,
    )) orelse code;
    const neff_file = try compileHloToNeff(io, tmp_dir, compile_hlo, target);
    const neff_hlo_bytes = try wrapNeffAsCustomCall(arena.allocator(), io, code, neff_file);

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

// Compile the StableHLO module that neuronxla handed to this hook into a NEFF
// file inside the caller-owned scratch directory.
fn compileHloToNeff(
    io: std.Io,
    tmp_dir: std.Io.Dir,
    hlo_code: []const u8,
    target: []const u8,
) !std.Io.File {
    const code_file = try tmp_dir.createFile(io, "file.code", .{});
    try code_file.writePositionalAll(io, hlo_code, 0);

    var code_file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const code_file_path = try code_file.realPath(io, &code_file_buf);

    var neuronx_cc_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const neuronx_cc_path = try stdx.Io.Dir.path.bufJoin(&neuronx_cc_buf, &.{
        stdx.process.selfSharedObjectDirPath(),
        "..",
        "bin",
        "neuronx-cc",
    });

    var cwd_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const cwd = try tmp_dir.realPath(io, &cwd_buf);

    var neff_file_buf: [std.Io.Dir.max_path_bytes]u8 = undefined;
    const neff_file_path = try stdx.Io.Dir.path.bufJoin(&neff_file_buf, &.{ cwd_buf[0..cwd], "file.neff" });

    const log_level = if (std.c.getenv("NEURON_RT_LOG_LEVEL")) |level| std.mem.span(level) else "error";

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
                "--enable-internal-neff-wrapper",
                "--output",
                neff_file_path,
                "--optlevel=1",
                "--model-type=transformer",
                "--auto-cast=none",
                "--enable-fast-loading-neuron-binaries",
                "--verbose",
                log_level,
                "--logfile-verbose",
                log_level,
                "--logfile=./log-neuron-cc.txt",
                code_file_buf[0..code_file_path],
            },
            .stdin = .ignore,
            .stdout = if (std.c.getenv("NEURON_RT_LOG_LEVEL")) |_| .inherit else .ignore,
            .stderr = if (std.c.getenv("NEURON_RT_LOG_LEVEL")) |_| .inherit else .ignore,
            .cwd = .{ .path = cwd_buf[0..cwd] },
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

    return try tmp_dir.openFile(io, "file.neff", .{});
}

// Rewrite each synthetic ZML NKI custom-call into the
// AwsNeuronCustomNativeKernel form consumed by neuronx-cc. Returns null when
// the module contains no embedded kernels, so callers can keep the original HLO.
fn rewriteCustomCalls(
    allocator: std.mem.Allocator,
    io: std.Io,
    tmp_dir: std.Io.Dir,
    hlo_code: []const u8,
    target: []const u8,
) !?[]const u8 {
    var upb_alloc: upb.Allocator = .init(allocator);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());

    const hlo_module = try upb.parse(c.xla_HloModuleProto, upb_arena, hlo_code);

    var rewritten: usize = 0;
    var kernel_index: usize = 0;

    var computations_len: usize = undefined;
    const computations = c.xla_HloModuleProto_mutable_computations(hlo_module, &computations_len)[0..computations_len];
    for (computations) |comp| {
        var instructions_len: usize = undefined;
        const instructions = c.xla_HloComputationProto_mutable_instructions(comp, &instructions_len)[0..instructions_len];
        for (instructions) |instruction| {
            const opcode = upb.slice(c.xla_HloInstructionProto_opcode(instruction)) orelse continue;
            if (!std.mem.eql(u8, opcode, "custom-call")) continue;

            const call_target = upb.slice(c.xla_HloInstructionProto_custom_call_target(instruction)) orelse continue;
            if (!std.mem.eql(u8, call_target, zml_neuron_nki_target)) continue;

            const encoded_backend_config = upb.slice(c.xla_HloInstructionProto_backend_config(instruction)) orelse return error.InvalidNkiBackendConfig;
            const kernel_config = try custom_call_config.parseKernelConfig(allocator, encoded_backend_config);
            const compiled_kernel = try kernel_compiler.compileKernel(allocator, io, tmp_dir, kernel_config, target, kernel_index);
            custom_call_mutator.rewriteInstructionAsCustomNativeKernel(instruction, compiled_kernel.backend_config_bytes);
            custom_call_mutator.setOuterCustomNativeKernelIoAttributes(
                allocator,
                upb_arena,
                comp,
                instructions,
                instruction,
                compiled_kernel.input_names,
                compiled_kernel.output_names,
            );

            rewritten += 1;
            kernel_index += 1;
        }
    }

    if (rewritten == 0) return null;
    return try upb.serialize(hlo_module, upb_arena);
}

// Replace the entry computation with one AwsNeuronNeff custom-call carrying
// the compiled NEFF bytes while preserving the original module parameters.
fn wrapNeffAsCustomCall(
    allocator: std.mem.Allocator,
    io: std.Io,
    hlo_code: []const u8,
    neff_file: std.Io.File,
) ![]const u8 {
    var upb_alloc: upb.Allocator = .init(allocator);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());

    const hlo_module = try upb.parse(c.xla_HloModuleProto, upb_arena, hlo_code);

    const neff_file_data = blk: {
        const stat = try neff_file.stat(io);
        const buf = try allocator.alloc(u8, stat.size);
        _ = try neff_file.readPositionalAll(io, buf, 0);
        break :blk buf;
    };

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
    c.xla_HloInstructionProto_set_backend_config(fused_root, upb.stringView(neff_file_data));

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
                operand_ids.appendAssumeCapacity(c.xla_HloInstructionProto_id(instruction));
                new_instructions.appendAssumeCapacity(instruction);
            }
        }
        new_instructions.appendAssumeCapacity(fused_root);
    }

    const valid_inputs_value: []const u8 = if (parameters_len == 0)
        &[_]u8{}
    else blk: {
        const out = try allocator.alloc(u8, parameters_len * 2 - 1);
        for (out, 0..) |*char, i| {
            char.* = if (i % 2 == 0) '1' else ',';
        }
        break :blk out;
    };

    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(fused_root, upb_arena);
    const map = c._xla_FrontendAttributes_map_mutable_upb_map(attrs, upb_arena);
    _ = c.upb_Map_Set(
        map,
        .{ .str_val = upb.stringView("valid_inputs") },
        .{ .str_val = upb.stringView(valid_inputs_value) },
        upb_arena,
    );

    return try upb.serialize(hlo_module, upb_arena);
}
