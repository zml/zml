const std = @import("std");

const c = @import("c");
const stdx = @import("stdx");
const upb = @import("upb");

pub const aws_neuron_neff_target = "AwsNeuronNeff";

pub fn setFrontendAttribute(
    upb_arena: *c.upb_Arena,
    attrs: *c.xla_FrontendAttributes,
    key: []const u8,
    value: []const u8,
) void {
    const map = c._xla_FrontendAttributes_map_mutable_upb_map(attrs, upb_arena);
    _ = c.upb_Map_Set(
        map,
        .{ .str_val = upb.stringView(key) },
        .{ .str_val = upb.stringView(value) },
        upb_arena,
    );
}

pub fn setInstructionFrontendAttribute(
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    key: []const u8,
    value: []const u8,
) void {
    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(instruction, upb_arena);
    setFrontendAttribute(upb_arena, attrs, key, value);
}

pub fn setValidInputs(
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

pub fn rewriteInstructionAsNeuronNeff(
    allocator: std.mem.Allocator,
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    neff_bytes: []const u8,
) !void {
    c.xla_HloInstructionProto_set_opcode(instruction, upb.stringView("custom-call"));
    c.xla_HloInstructionProto_set_custom_call_target(instruction, upb.stringView(aws_neuron_neff_target));
    c.xla_HloInstructionProto_set_backend_config(instruction, upb.stringView(neff_bytes));

    var operand_count: usize = undefined;
    _ = c.xla_HloInstructionProto_operand_ids(instruction, &operand_count);
    try setValidInputs(allocator, upb_arena, instruction, operand_count);
}

pub fn wrapNeffAsCustomCall(
    allocator: std.mem.Allocator,
    io: std.Io,
    hlo_code: []const u8,
    neff_file_path: []const u8,
) ![]const u8 {
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

    return try upb.serialize(hlo_module, upb_arena);
}
