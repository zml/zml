const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");
const upb = @import("upb");

const log = std.log.scoped(.@"zml/platforms/neuron/whole_program_compiler");
const aws_neuron_neff_target = "AwsNeuronNeff";

// This module owns the whole-program StableHLO -> NEFF -> AwsNeuronNeff flow.
// `libneuronxla.zig` stays the Python hook boundary; this file owns the
// mechanics of invoking `neuronx-cc` and rebuilding the final wrapped HLO.

// Compile a whole StableHLO module to a NEFF inside the caller-owned Neuron
// scratch directory.
pub fn compileHloToNeff(
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

    const verbosity = @import("target_resolution.zig").compilerVerbosity();

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
                verbosity,
                "--logfile-verbose",
                verbosity,
                "--logfile=./log-neuron-cc.txt",
                code_file_buf[0..code_file_path],
            },
            .stdin = .ignore,
            .stdout = .inherit,
            .stderr = .inherit,
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

// Replace the entry computation with a single `AwsNeuronNeff` custom-call that
// carries the compiled NEFF bytes while preserving the original parameters.
pub fn wrapNeffAsCustomCall(
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
