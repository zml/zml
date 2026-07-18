const std = @import("std");

const ProvenanceRecord = struct {
    stable_op_id: u64,
    file: []const u8,
    line: u32,
    column: u32,
};

const SourceMapping = struct {
    id: []const u8,
    file: []const u8,
    line: u32,
    column: u32,
    stable_op_ids: []const []const u8,
};

const StableOpMapping = struct {
    id: []const u8,
    source_id: []const u8,
    operation: ?[]const u8,
    stablehlo_lines: []const usize,
    hlo_instruction_ids: []const []const u8,
};

const HloInstructionMapping = struct {
    id: []const u8,
    name: ?[]const u8,
    opcode: ?[]const u8,
    hlo_lines: []const usize,
    stable_op_id: []const u8,
    mapping: []const u8,
};

const Mapping = struct {
    version: u32 = 1,
    sources: []const SourceMapping,
    stable_ops: []const StableOpMapping,
    hlo_instructions: []const HloInstructionMapping,
};

const SourceBuilder = struct {
    id: []const u8,
    file: []const u8,
    line: u32,
    column: u32,
    stable_op_ids: std.ArrayList([]const u8) = .empty,
};

const StableOpBuilder = struct {
    id: []const u8,
    source_id: []const u8,
    operation: ?[]const u8 = null,
    result_name: ?[]const u8 = null,
    operands: std.ArrayList([]const u8) = .empty,
    stablehlo_lines: std.ArrayList(usize) = .empty,
    hlo_instruction_ids: std.ArrayList([]const u8) = .empty,
};

const HloInstructionBuilder = struct {
    name: []const u8,
    opcode: []const u8,
    line: usize,
    operands: []const []const u8,
    stable_op_id: ?[]const u8,
};

pub fn writeFile(io: std.Io, output_dir: []const u8, name: []const u8, contents: []const u8) !void {
    var path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const path = try std.fmt.bufPrint(&path_buffer, "{s}/{s}", .{ output_dir, name });
    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, contents, 0);
}

pub fn finalize(
    allocator: std.mem.Allocator,
    io: std.Io,
    output_dir: []const u8,
    xla_dump_dir: []const u8,
) !void {
    const hlo_text_name = try findDumpFile(allocator, io, xla_dump_dir, ".before_optimizations.txt");
    defer allocator.free(hlo_text_name);
    try copyDump(io, xla_dump_dir, hlo_text_name, output_dir, "hlo.before_optimizations.txt");

    const text_extension = ".txt";
    std.debug.assert(std.mem.endsWith(u8, hlo_text_name, text_extension));
    const hlo_proto_name = try std.mem.concat(allocator, u8, &.{
        hlo_text_name[0 .. hlo_text_name.len - text_extension.len],
        ".hlo.pb",
    });
    defer allocator.free(hlo_proto_name);
    try copyDump(io, xla_dump_dir, hlo_proto_name, output_dir, "hlo.before_optimizations.pb");

    var provenance_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const provenance_path = try std.fmt.bufPrint(&provenance_path_buffer, "{s}/provenance.json", .{output_dir});
    const provenance_json = try std.Io.Dir.cwd().readFileAlloc(io, provenance_path, allocator, .unlimited);
    defer allocator.free(provenance_json);

    var stablehlo_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const stablehlo_path = try std.fmt.bufPrint(&stablehlo_path_buffer, "{s}/stablehlo.mlir", .{output_dir});
    const stablehlo = try std.Io.Dir.cwd().readFileAlloc(io, stablehlo_path, allocator, .unlimited);
    defer allocator.free(stablehlo);

    var hlo_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const hlo_path = try std.fmt.bufPrint(&hlo_path_buffer, "{s}/hlo.before_optimizations.txt", .{output_dir});
    const hlo = try std.Io.Dir.cwd().readFileAlloc(io, hlo_path, allocator, .unlimited);
    defer allocator.free(hlo);

    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try parseProvenance(arena, provenance_json);
    const mapping = try buildMapping(arena, records, stablehlo, hlo);
    for (mapping.stable_ops) |stable| {
        if (stable.stablehlo_lines.len == 0 or stable.hlo_instruction_ids.len == 0) {
            std.log.err("incomplete provenance for zml.stable_op.{s}", .{stable.id});
            return error.IncompleteHloMapping;
        }
    }

    var mapping_path_buffer: [std.fs.max_path_bytes]u8 = undefined;
    const mapping_path = try std.fmt.bufPrint(&mapping_path_buffer, "{s}/mapping.json", .{output_dir});
    const mapping_file = try std.Io.Dir.createFile(.cwd(), io, mapping_path, .{});
    defer mapping_file.close(io);

    var file_buffer: [16 * 1024]u8 = undefined;
    var file_writer = mapping_file.writer(io, &file_buffer);
    try std.json.Stringify.value(mapping, .{ .whitespace = .indent_2 }, &file_writer.interface);
    try file_writer.interface.writeAll("\n");
    try file_writer.interface.flush();
}

fn findDumpFile(
    allocator: std.mem.Allocator,
    io: std.Io,
    dump_dir_path: []const u8,
    suffix: []const u8,
) ![]u8 {
    var dump_dir = try std.Io.Dir.openDir(.cwd(), io, dump_dir_path, .{ .iterate = true });
    defer dump_dir.close(io);

    var candidate: ?[]u8 = null;
    errdefer if (candidate) |name| allocator.free(name);

    var it = dump_dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file or !std.mem.endsWith(u8, entry.name, suffix)) continue;

        if (candidate) |old_name| {
            if (std.mem.order(u8, entry.name, old_name) != .gt) continue;
            allocator.free(old_name);
        }
        candidate = try allocator.dupe(u8, entry.name);
    }

    return candidate orelse error.MissingXlaDump;
}

fn copyDump(
    io: std.Io,
    source_dir_path: []const u8,
    source_name: []const u8,
    output_dir_path: []const u8,
    output_name: []const u8,
) !void {
    var source_dir = try std.Io.Dir.openDir(.cwd(), io, source_dir_path, .{});
    defer source_dir.close(io);
    var output_dir = try std.Io.Dir.openDir(.cwd(), io, output_dir_path, .{});
    defer output_dir.close(io);

    try source_dir.copyFile(source_name, output_dir, output_name, io, .{ .replace = true });
}

fn parseProvenance(allocator: std.mem.Allocator, json: []const u8) ![]ProvenanceRecord {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json, .{});
    const records_value = switch (parsed) {
        .array => parsed,
        .object => |object| object.get("stable_ops") orelse object.get("records") orelse object.get("provenance") orelse return error.InvalidProvenance,
        else => return error.InvalidProvenance,
    };
    if (records_value != .array) return error.InvalidProvenance;

    var records: std.ArrayList(ProvenanceRecord) = .empty;
    for (records_value.array.items) |value| {
        if (value != .object) return error.InvalidProvenance;
        const object = value.object;
        try records.append(allocator, .{
            .stable_op_id = try jsonUnsigned(u64, object.get("stable_op_id")),
            .file = try jsonString(object.get("file")),
            .line = try jsonUnsigned(u32, object.get("line")),
            .column = try jsonUnsigned(u32, object.get("column")),
        });
    }
    return records.toOwnedSlice(allocator);
}

fn jsonUnsigned(comptime T: type, value: ?std.json.Value) !T {
    return switch (value orelse return error.InvalidProvenance) {
        .integer => |number| std.math.cast(T, number) orelse error.InvalidProvenance,
        else => error.InvalidProvenance,
    };
}

fn jsonString(value: ?std.json.Value) ![]const u8 {
    return switch (value orelse return error.InvalidProvenance) {
        .string => |string| string,
        else => error.InvalidProvenance,
    };
}

fn buildMapping(
    allocator: std.mem.Allocator,
    records: []const ProvenanceRecord,
    stablehlo: []const u8,
    hlo: []const u8,
) !Mapping {
    var source_builders: std.ArrayList(SourceBuilder) = .empty;
    var stable_builders: std.ArrayList(StableOpBuilder) = .empty;

    for (records) |record| {
        const file = record.file;
        const source_id = try std.fmt.allocPrint(allocator, "{s}:{d}:{d}", .{ file, record.line, record.column });
        const stable_id = try std.fmt.allocPrint(allocator, "{d}", .{record.stable_op_id});

        var source_index: ?usize = null;
        for (source_builders.items, 0..) |source, i| {
            if (std.mem.eql(u8, source.id, source_id)) {
                source_index = i;
                break;
            }
        }
        if (source_index) |index| {
            try source_builders.items[index].stable_op_ids.append(allocator, stable_id);
        } else {
            var source: SourceBuilder = .{
                .id = source_id,
                .file = file,
                .line = record.line,
                .column = record.column,
            };
            try source.stable_op_ids.append(allocator, stable_id);
            try source_builders.append(allocator, source);
        }

        try stable_builders.append(allocator, .{
            .id = stable_id,
            .source_id = source_id,
        });
    }

    try mapStableHlo(allocator, stablehlo, stable_builders.items);
    const hlo_mappings = try mapHlo(allocator, hlo, stable_builders.items);

    const sources = try allocator.alloc(SourceMapping, source_builders.items.len);
    for (source_builders.items, sources) |*builder, *source| {
        source.* = .{
            .id = builder.id,
            .file = builder.file,
            .line = builder.line,
            .column = builder.column,
            .stable_op_ids = try builder.stable_op_ids.toOwnedSlice(allocator),
        };
    }

    const stable_ops = try allocator.alloc(StableOpMapping, stable_builders.items.len);
    for (stable_builders.items, stable_ops) |*builder, *stable| {
        stable.* = .{
            .id = builder.id,
            .source_id = builder.source_id,
            .operation = builder.operation,
            .stablehlo_lines = try builder.stablehlo_lines.toOwnedSlice(allocator),
            .hlo_instruction_ids = try builder.hlo_instruction_ids.toOwnedSlice(allocator),
        };
    }

    return .{
        .sources = sources,
        .stable_ops = stable_ops,
        .hlo_instructions = hlo_mappings,
    };
}

fn mapStableHlo(
    allocator: std.mem.Allocator,
    text: []const u8,
    stable_ops: []StableOpBuilder,
) !void {
    var aliases: std.ArrayList(struct { alias: []const u8, stable_id: []const u8 }) = .empty;

    var line_number: usize = 1;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| : (line_number += 1) {
        if (findStableId(line)) |stable_id| {
            const trimmed = std.mem.trimStart(u8, line, " \t");
            if (std.mem.startsWith(u8, trimmed, "#loc")) {
                const alias_end = std.mem.indexOfAny(u8, trimmed, " =") orelse continue;
                try aliases.append(allocator, .{
                    .alias = trimmed[0..alias_end],
                    .stable_id = try allocator.dupe(u8, stable_id),
                });
            } else if (findStableBuilder(stable_ops, stable_id)) |stable| {
                try addStableHloLine(allocator, stable, line, line_number);
            }
        }
    }

    line_number = 1;
    lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| : (line_number += 1) {
        for (aliases.items) |alias| {
            if (!containsLocationAlias(line, alias.alias)) continue;
            const trimmed = std.mem.trimStart(u8, line, " \t");
            if (std.mem.startsWith(u8, trimmed, alias.alias) and containsLocationAlias(trimmed[0..alias.alias.len], alias.alias)) continue;
            if (findStableBuilder(stable_ops, alias.stable_id)) |stable| {
                try addStableHloLine(allocator, stable, line, line_number);
            }
        }
    }
}

fn addStableHloLine(
    allocator: std.mem.Allocator,
    stable: *StableOpBuilder,
    line: []const u8,
    line_number: usize,
) !void {
    if (std.mem.indexOf(u8, line, "stablehlo.")) |start| {
        const operation_end = std.mem.indexOfAnyPos(u8, line, start, " \t\"(") orelse line.len;
        stable.operation = try allocator.dupe(u8, line[start..operation_end]);
        try appendUnique(usize, allocator, &stable.stablehlo_lines, line_number);

        const values = try parsePercentNames(allocator, line);
        if (values.len != 0 and stable.result_name == null) {
            stable.result_name = values[0];
            for (values[1..]) |operand| {
                try stable.operands.append(allocator, operand);
            }
        }
    }
}

fn mapHlo(
    allocator: std.mem.Allocator,
    text: []const u8,
    stable_ops: []StableOpBuilder,
) ![]HloInstructionMapping {
    var hlo_builders: std.ArrayList(HloInstructionBuilder) = .empty;
    var instructions: std.ArrayList(HloInstructionMapping) = .empty;
    var line_number: usize = 1;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| : (line_number += 1) {
        const name = try parseInstructionName(allocator, line) orelse continue;
        const opcode = try parseHloOpcode(allocator, line) orelse continue;
        const values = try parsePercentNames(allocator, line);
        const operands = if (values.len != 0 and std.mem.eql(u8, values[0], name)) values[1..] else values;
        try hlo_builders.append(allocator, .{
            .name = name,
            .opcode = opcode,
            .line = line_number,
            .operands = operands,
            .stable_op_id = if (findStableId(line)) |id| try allocator.dupe(u8, id) else null,
        });
    }

    for (hlo_builders.items) |*hlo| {
        const stable_id = hlo.stable_op_id orelse continue;
        const stable = findStableBuilder(stable_ops, stable_id) orelse continue;
        try addHloMapping(allocator, &instructions, stable, hlo, "metadata");
    }

    // XLA currently omits metadata for scalar constants and broadcasts. Trace
    // those producers through operand positions from metadata-backed consumers
    // instead of guessing from opcode uniqueness.
    var changed = true;
    while (changed) {
        changed = false;
        for (stable_ops) |*consumer_stable| {
            const consumer_hlo = findPrimaryHlo(hlo_builders.items, consumer_stable) orelse continue;
            for (consumer_stable.operands.items, 0..) |stable_operand, operand_index| {
                const producer_stable = findStableByResult(stable_ops, stable_operand) orelse continue;
                if (operand_index >= consumer_hlo.operands.len) continue;

                const producer_hlo = findHloByName(hlo_builders.items, consumer_hlo.operands[operand_index]) orelse continue;
                const expected_opcode = normalizedStableOpcode(producer_stable.operation orelse continue);
                if (!std.mem.eql(u8, producer_hlo.opcode, expected_opcode)) continue;

                if (producer_hlo.stable_op_id) |existing_id| {
                    if (!std.mem.eql(u8, existing_id, producer_stable.id)) continue;
                } else {
                    producer_hlo.stable_op_id = producer_stable.id;
                    changed = true;
                }
                try addHloMapping(allocator, &instructions, producer_stable, producer_hlo, "dataflow_operand");
            }
        }
    }

    return instructions.toOwnedSlice(allocator);
}

fn addHloMapping(
    allocator: std.mem.Allocator,
    mappings: *std.ArrayList(HloInstructionMapping),
    stable: *StableOpBuilder,
    hlo: *const HloInstructionBuilder,
    mapping: []const u8,
) !void {
    if (instructionIsMapped(mappings.items, hlo.name)) return;
    const instruction_id = try allocator.dupe(u8, hlo.name);
    try mappings.append(allocator, .{
        .id = instruction_id,
        .name = instruction_id,
        .opcode = hlo.opcode,
        .hlo_lines = try allocator.dupe(usize, &.{hlo.line}),
        .stable_op_id = stable.id,
        .mapping = mapping,
    });
    try appendUnique([]const u8, allocator, &stable.hlo_instruction_ids, instruction_id);
}

fn findPrimaryHlo(hlo_instructions: []HloInstructionBuilder, stable: *const StableOpBuilder) ?*HloInstructionBuilder {
    const expected_opcode = normalizedStableOpcode(stable.operation orelse return null);
    var match: ?*HloInstructionBuilder = null;
    for (hlo_instructions) |*instruction| {
        const stable_id = instruction.stable_op_id orelse continue;
        if (!std.mem.eql(u8, stable_id, stable.id) or !std.mem.eql(u8, instruction.opcode, expected_opcode)) continue;
        if (match != null) return null;
        match = instruction;
    }
    return match;
}

fn findStableByResult(stable_ops: []StableOpBuilder, result_name: []const u8) ?*StableOpBuilder {
    for (stable_ops) |*stable| {
        if (stable.result_name) |candidate| {
            if (std.mem.eql(u8, candidate, result_name)) return stable;
        }
    }
    return null;
}

fn findHloByName(hlo_instructions: []HloInstructionBuilder, name: []const u8) ?*HloInstructionBuilder {
    for (hlo_instructions) |*instruction| {
        if (std.mem.eql(u8, instruction.name, name)) return instruction;
    }
    return null;
}

fn normalizedStableOpcode(operation: []const u8) []const u8 {
    const prefix = "stablehlo.";
    const opcode = if (std.mem.startsWith(u8, operation, prefix)) operation[prefix.len..] else operation;
    if (std.mem.eql(u8, opcode, "broadcast_in_dim")) return "broadcast";
    return opcode;
}

fn instructionIsMapped(instructions: []const HloInstructionMapping, name: []const u8) bool {
    for (instructions) |instruction| {
        if (instruction.name) |mapped_name| {
            if (std.mem.eql(u8, mapped_name, name)) return true;
        }
    }
    return false;
}

fn findStableId(line: []const u8) ?[]const u8 {
    const prefix = "zml.stable_op.";
    const start = (std.mem.indexOf(u8, line, prefix) orelse return null) + prefix.len;
    var end = start;
    while (end < line.len and std.ascii.isDigit(line[end])) : (end += 1) {}
    if (end == start) return null;
    return line[start..end];
}

fn findStableBuilder(stable_ops: []StableOpBuilder, id: []const u8) ?*StableOpBuilder {
    for (stable_ops) |*stable| {
        if (std.mem.eql(u8, stable.id, id)) return stable;
    }
    return null;
}

fn parsePercentNames(allocator: std.mem.Allocator, line: []const u8) ![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    var cursor: usize = 0;
    while (std.mem.indexOfScalarPos(u8, line, cursor, '%')) |start| {
        var end = start + 1;
        while (end < line.len and isValueNameCharacter(line[end])) : (end += 1) {}
        if (end > start + 1) try names.append(allocator, try allocator.dupe(u8, line[start..end]));
        cursor = @max(end, start + 1);
    }
    return names.toOwnedSlice(allocator);
}

fn isValueNameCharacter(character: u8) bool {
    return std.ascii.isAlphanumeric(character) or character == '_' or character == '.' or character == '-' or character == '$';
}

fn containsLocationAlias(line: []const u8, alias: []const u8) bool {
    var cursor: usize = 0;
    while (std.mem.indexOfPos(u8, line, cursor, alias)) |start| {
        const end = start + alias.len;
        if (end == line.len or !std.ascii.isAlphanumeric(line[end])) return true;
        cursor = end;
    }
    return false;
}

fn parseInstructionName(allocator: std.mem.Allocator, line: []const u8) !?[]const u8 {
    const equals = std.mem.indexOfScalar(u8, line, '=') orelse return null;
    var before = std.mem.trim(u8, line[0..equals], " \t");
    if (std.mem.startsWith(u8, before, "ROOT ")) before = std.mem.trimStart(u8, before[5..], " \t");
    if (before.len == 0) return null;
    const token_start = std.mem.lastIndexOfAny(u8, before, " \t") orelse 0;
    const name = if (token_start == 0) before else before[token_start + 1 ..];
    return try allocator.dupe(u8, name);
}

fn parseHloOpcode(allocator: std.mem.Allocator, line: []const u8) !?[]const u8 {
    const equals = std.mem.indexOfScalar(u8, line, '=') orelse return null;
    const open_paren = std.mem.indexOfPos(u8, line, equals + 1, "(") orelse return null;
    const before = std.mem.trimEnd(u8, line[equals + 1 .. open_paren], " \t");
    const token_start = std.mem.lastIndexOfAny(u8, before, " \t") orelse 0;
    const opcode = if (token_start == 0) before else before[token_start + 1 ..];
    if (opcode.len == 0) return null;
    return try allocator.dupe(u8, opcode);
}

fn appendUnique(
    comptime T: type,
    allocator: std.mem.Allocator,
    list: *std.ArrayList(T),
    value: T,
) !void {
    for (list.items) |existing| {
        if (if (T == []const u8) std.mem.eql(u8, existing, value) else existing == value) return;
    }
    try list.append(allocator, value);
}

test "build mapping correlates MLIR location aliases and HLO metadata" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"tools/source_to_hlo_explorer/source.zig","line":4,"column":18}]}
    ;
    const stablehlo =
        \\    %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> loc(#loc1)
        \\#loc1 = loc("zml.stable_op.1"("tools/source_to_hlo_explorer/source.zig":4:18))
    ;
    const hlo =
        \\  ROOT %add.3 = f32[4]{0} add(%x.1, %y.2), metadata={op_name="zml.stable_op.1"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try parseProvenance(arena, provenance);
    const mapping = try buildMapping(arena, records, stablehlo, hlo);

    try std.testing.expectEqual(@as(usize, 1), mapping.sources.len);
    try std.testing.expectEqualStrings("tools/source_to_hlo_explorer/source.zig:4:18", mapping.sources[0].id);
    try std.testing.expectEqualSlices(usize, &.{1}, mapping.stable_ops[0].stablehlo_lines);
    try std.testing.expectEqualStrings("stablehlo.add", mapping.stable_ops[0].operation.?);
    try std.testing.expectEqual(@as(usize, 1), mapping.hlo_instructions.len);
    try std.testing.expectEqualStrings("%add.3", mapping.hlo_instructions[0].id);
    try std.testing.expectEqualStrings("add", mapping.hlo_instructions[0].opcode.?);
    try std.testing.expectEqualSlices(usize, &.{1}, mapping.hlo_instructions[0].hlo_lines);
}

test "dataflow maps metadata-free producers without opcode guessing" {
    const provenance =
        \\{"records":[
        \\  {"stable_op_id":0,"file":"source.zig","line":4,"column":28},
        \\  {"stable_op_id":1,"file":"source.zig","line":5,"column":42},
        \\  {"stable_op_id":2,"file":"source.zig","line":5,"column":42},
        \\  {"stable_op_id":3,"file":"source.zig","line":5,"column":42}
        \\]}
    ;
    const stablehlo =
        \\  %cst = stablehlo.constant dense<2.0> : tensor<f32> loc("zml.stable_op.1")
        \\  %sum = stablehlo.add %arg0, %arg1 : tensor<4xf32> loc("zml.stable_op.0")
        \\  %wide = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<f32>) -> tensor<4xf32> loc("zml.stable_op.2")
        \\  %result = stablehlo.multiply %sum, %wide : tensor<4xf32> loc("zml.stable_op.3")
    ;
    const hlo =
        \\  %sum.1 = f32[4]{0} add(%arg0.1, %arg1.1), metadata={op_name="zml.stable_op.0"}
        \\  %unrelated.1 = f32[] constant(7)
        \\  %constant.1 = f32[] constant(2)
        \\  %broadcast.1 = f32[4]{0} broadcast(%constant.1), dimensions={}
        \\  ROOT %result.1 = f32[4]{0} multiply(%sum.1, %broadcast.1), metadata={op_name="zml.stable_op.3"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try parseProvenance(arena, provenance);
    const mapping = try buildMapping(arena, records, stablehlo, hlo);

    try std.testing.expectEqual(@as(usize, 4), mapping.hlo_instructions.len);
    var found_constant = false;
    var found_broadcast = false;
    for (mapping.hlo_instructions) |instruction| {
        try std.testing.expect(!std.mem.eql(u8, instruction.id, "%unrelated.1"));
        if (std.mem.eql(u8, instruction.id, "%constant.1")) {
            found_constant = true;
            try std.testing.expectEqualStrings("1", instruction.stable_op_id);
            try std.testing.expectEqualStrings("dataflow_operand", instruction.mapping);
        }
        if (std.mem.eql(u8, instruction.id, "%broadcast.1")) {
            found_broadcast = true;
            try std.testing.expectEqualStrings("2", instruction.stable_op_id);
            try std.testing.expectEqualStrings("dataflow_operand", instruction.mapping);
        }
    }
    try std.testing.expect(found_constant);
    try std.testing.expect(found_broadcast);
}

test "location aliases are matched as complete tokens" {
    const provenance =
        \\{"records":[
        \\  {"stable_op_id":1,"file":"source.zig","line":1,"column":1},
        \\  {"stable_op_id":10,"file":"source.zig","line":2,"column":1}
        \\]}
    ;
    const stablehlo =
        \\  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> loc(#loc1)
        \\  %1 = stablehlo.multiply %0, %arg1 : tensor<4xf32> loc(#loc10)
        \\#loc1 = loc("zml.stable_op.1"("source.zig":1:1))
        \\#loc10 = loc("zml.stable_op.10"("source.zig":2:1))
    ;
    const hlo =
        \\  %add.1 = f32[4]{0} add(%arg0.1, %arg1.1), metadata={op_name="zml.stable_op.1"}
        \\  ROOT %multiply.1 = f32[4]{0} multiply(%add.1, %arg1.1), metadata={op_name="zml.stable_op.10"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try parseProvenance(arena, provenance);
    const mapping = try buildMapping(arena, records, stablehlo, hlo);
    try std.testing.expectEqualSlices(usize, &.{1}, mapping.stable_ops[0].stablehlo_lines);
    try std.testing.expectEqualSlices(usize, &.{2}, mapping.stable_ops[1].stablehlo_lines);
}
