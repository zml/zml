const std = @import("std");

const c = @import("c");
const upb = @import("upb");

const aws_neuron_custom_native_kernel_target = "AwsNeuronCustomNativeKernel";
const neff_input_name_attr = "neff_input_name";
const neff_input_names_attr = "neff_input_names";
const neff_output_names_attr = "neff_output_names";

// HLO mutation helpers for the embedded NKI lowering stage. The compiler
// client returns a backend_config and IO names; this file applies those values
// to the existing custom-call and its surrounding entry computation.

// Rewrite the synthetic ZML custom-call into the Neuron-native target consumed
// by neuronx-cc.
pub fn rewriteInstructionAsCustomNativeKernel(
    instruction: *c.xla_HloInstructionProto,
    backend_config_bytes: []const u8,
) void {
    c.xla_HloInstructionProto_set_opcode(instruction, upb.stringView("custom-call"));
    c.xla_HloInstructionProto_set_custom_call_target(instruction, upb.stringView(aws_neuron_custom_native_kernel_target));
    c.xla_HloInstructionProto_set_custom_call_api_version(instruction, c.xla_API_VERSION_UNSPECIFIED);
    c.xla_HloInstructionProto_set_backend_config(instruction, upb.stringView(backend_config_bytes));
}

// Propagate the compiled kernel IO names onto the surrounding StableHLO
// parameters and tuple root so the outer program and runtime agree on names.
pub fn setOuterCustomNativeKernelIoAttributes(
    allocator: std.mem.Allocator,
    upb_arena: *c.upb_Arena,
    comp: *c.xla_HloComputationProto,
    instructions: []const [*c]c.xla_HloInstructionProto,
    instruction: *c.xla_HloInstructionProto,
    input_names: []const []const u8,
    output_names: []const []const u8,
) void {
    const root_instruction = findInstructionById(instructions, c.xla_HloComputationProto_root_id(comp)) orelse return;
    const root_opcode = upb.slice(c.xla_HloInstructionProto_opcode(root_instruction)) orelse return;
    if (!std.mem.eql(u8, root_opcode, "tuple")) return;

    var root_operand_count: usize = 0;
    const root_operand_ids = c.xla_HloInstructionProto_operand_ids(root_instruction, &root_operand_count)[0..root_operand_count];
    if (root_operand_ids.len != 1 or root_operand_ids[0] != c.xla_HloInstructionProto_id(instruction)) return;

    var operand_count: usize = 0;
    const operand_ids = c.xla_HloInstructionProto_operand_ids(instruction, &operand_count)[0..operand_count];
    for (operand_ids, 0..) |operand_id, i| {
        if (i >= input_names.len) break;
        const operand = findInstructionById(instructions, operand_id) orelse continue;
        const opcode = upb.slice(c.xla_HloInstructionProto_opcode(operand)) orelse continue;
        if (!std.mem.eql(u8, opcode, "parameter")) continue;
        deleteInstructionFrontendAttribute(upb_arena, operand, neff_input_names_attr);
        setInstructionFrontendAttribute(upb_arena, operand, neff_input_name_attr, input_names[i]);
    }

    if (output_names.len == 0) return;

    const output_names_value = if (output_names.len == 1)
        output_names[0]
    else
        std.mem.join(allocator, ",", output_names) catch return;
    setInstructionFrontendAttribute(upb_arena, root_instruction, neff_output_names_attr, output_names_value);
}

fn findInstructionById(
    instructions: []const [*c]c.xla_HloInstructionProto,
    id: i64,
) ?*c.xla_HloInstructionProto {
    for (instructions) |instruction| {
        if (c.xla_HloInstructionProto_id(instruction) == id) return instruction;
    }
    return null;
}

fn deleteInstructionFrontendAttribute(
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    key: []const u8,
) void {
    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(instruction, upb_arena);
    _ = c.xla_FrontendAttributes_map_delete(attrs, upb.stringView(key));
}

fn setFrontendAttribute(
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

fn setInstructionFrontendAttribute(
    upb_arena: *c.upb_Arena,
    instruction: *c.xla_HloInstructionProto,
    key: []const u8,
    value: []const u8,
) void {
    const attrs = c.xla_HloInstructionProto_mutable_frontend_attributes(instruction, upb_arena);
    setFrontendAttribute(upb_arena, attrs, key, value);
}
