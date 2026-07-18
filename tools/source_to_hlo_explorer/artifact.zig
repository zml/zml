const std = @import("std");

pub const ProvenanceRecord = struct {
    stable_op_id: u64,
    file: []const u8,
    line: u32,
    column: u32,
};

const LocationAlias = struct {
    name: []const u8,
    expression: []const u8,
};

const SourceLocation = struct {
    file: []const u8,
    line: u32,
    column: u32,
};

const ResolvedLocation = struct {
    stable_op_id: ?u64 = null,
    source: ?SourceLocation = null,
};

const SourceSpan = struct {
    file: []const u8,
    line: u32,
    column: u32,
    end_line: u32,
    end_column: u32,
    start_byte: ?usize,
    end_byte: ?usize,
    method: ?[]const u8,
    provenance_line: u32,
    provenance_column: u32,
};

const SourceMapping = struct {
    id: []const u8,
    file: []const u8,
    line: u32,
    column: u32,
    end_line: ?u32,
    end_column: ?u32,
    start_byte: ?usize,
    end_byte: ?usize,
    method: ?[]const u8,
    provenance_line: ?u32,
    provenance_column: ?u32,
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
    end_line: ?u32 = null,
    end_column: ?u32 = null,
    start_byte: ?usize = null,
    end_byte: ?usize = null,
    method: ?[]const u8 = null,
    provenance_line: ?u32 = null,
    provenance_column: ?u32 = null,
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

/// Builds a dependency-free explorer bundle from compiler text artifacts.
/// Provenance is recovered from StableHLO operation attributes and locations;
/// a separate `provenance.json` input is neither read nor written.
pub fn writeBundleFromTexts(
    allocator: std.mem.Allocator,
    io: std.Io,
    output_dir: []const u8,
    source: []const u8,
    source_map_json: ?[]const u8,
    stablehlo: []const u8,
    hlo: []const u8,
) !void {
    var arena_state: std.heap.ArenaAllocator = .init(allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try deriveProvenanceRecords(arena, stablehlo);
    const source_spans = if (source_map_json) |json| try parseSourceMap(arena, json) else &.{};
    const mapping = try buildMappingWithSpans(arena, records, stablehlo, hlo, source_spans);
    try validateCompleteMapping(mapping);

    try std.Io.Dir.cwd().createDirPath(io, output_dir);
    try writeFile(io, output_dir, "source.zig", source);
    try writeFile(io, output_dir, "stablehlo.mlir", stablehlo);
    try writeFile(io, output_dir, "hlo.before_optimizations.txt", hlo);
    try writeMappingFile(io, output_dir, mapping);
}

pub fn finalize(
    allocator: std.mem.Allocator,
    io: std.Io,
    output_dir: []const u8,
    xla_dump_dir: []const u8,
) !void {
    return finalizeWithSourceMap(allocator, io, output_dir, xla_dump_dir, null);
}

/// Finalizes the explorer bundle and, when present, enriches compiler
/// provenance points with exact expression spans from the AST instrumenter's
/// JSON sidecar.
pub fn finalizeWithSourceMap(
    allocator: std.mem.Allocator,
    io: std.Io,
    output_dir: []const u8,
    xla_dump_dir: []const u8,
    source_map_json: ?[]const u8,
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
    const source_spans = if (source_map_json) |json| try parseSourceMap(arena, json) else &.{};
    const mapping = try buildMappingWithSpans(arena, records, stablehlo, hlo, source_spans);
    try validateCompleteMapping(mapping);
    try writeMappingFile(io, output_dir, mapping);
}

fn validateCompleteMapping(mapping: Mapping) !void {
    if (mapping.stable_ops.len == 0) return error.MissingStableOpProvenance;
    for (mapping.stable_ops) |stable| {
        if (stable.stablehlo_lines.len == 0 or stable.hlo_instruction_ids.len == 0) {
            std.log.err("incomplete provenance for zml.stable_op.{s}", .{stable.id});
            return error.IncompleteHloMapping;
        }
    }
}

fn writeMappingFile(io: std.Io, output_dir: []const u8, mapping: Mapping) !void {
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

/// Recovers the compiler provenance table embedded in a debug-info StableHLO
/// dump. Returned strings and the returned slice live for the lifetime of
/// `allocator`; callers commonly pass an arena allocator.
pub fn deriveProvenanceRecords(
    allocator: std.mem.Allocator,
    stablehlo: []const u8,
) ![]ProvenanceRecord {
    const aliases = try collectLocationAliases(allocator, stablehlo);
    var attribute_ids: std.ArrayList(u64) = .empty;
    var records: std.ArrayList(ProvenanceRecord) = .empty;

    // Frontend attributes are the surviving operation identities. Locations
    // for region-bearing operations may be printed on a later line, so collect
    // the identities independently from the location table.
    var lines = std.mem.splitScalar(u8, stablehlo, '\n');
    while (lines.next()) |line| {
        const attribute_id = try findFrontendStableId(line);
        if (attribute_id) |id| try appendUnique(u64, allocator, &attribute_ids, id);
        if (std.mem.indexOf(u8, line, "stablehlo.") == null) continue;

        // Preserve the strict same-operation check for compact syntax.
        if (try findNamedStableLocation(line)) |named| {
            _ = try mergeStableOpIds(attribute_id, named.stable_op_id);
        }

        // Generic `loc(#locN)` syntax can resolve directly on the operation.
        if (try findLastLocationArgument(line)) |argument| {
            const resolved = try resolveLocation(allocator, argument, aliases, 0);
            const stable_op_id = (try mergeStableOpIds(attribute_id, resolved.stable_op_id)) orelse continue;
            if (resolved.source) |source| {
                try upsertProvenanceRecordSorted(allocator, &records, .{
                    .stable_op_id = stable_op_id,
                    .file = source.file,
                    .line = source.line,
                    .column = source.column,
                });
            }
        }
    }

    // Pretty MLIR prints named locations both inline and in the trailing alias
    // table. Scanning all lines joins multiline operations back to their
    // source without needing to parse StableHLO region boundaries.
    lines = std.mem.splitScalar(u8, stablehlo, '\n');
    while (lines.next()) |line| {
        const named = try findNamedStableLocation(line) orelse continue;
        var resolved = try resolveLocation(allocator, named.argument, aliases, 0);
        resolved.stable_op_id = try mergeStableOpIds(resolved.stable_op_id, named.stable_op_id);
        const source = resolved.source orelse continue;
        try upsertProvenanceRecordSorted(allocator, &records, .{
            .stable_op_id = resolved.stable_op_id.?,
            .file = source.file,
            .line = source.line,
            .column = source.column,
        });
    }

    if (attribute_ids.items.len == 0 or records.items.len == 0) return error.MissingStableOpProvenance;
    for (attribute_ids.items) |id| {
        if (!containsProvenanceRecord(records.items, id)) {
            std.log.warn("ignoring zml.stable_op.{d}: StableHLO has no source location", .{id});
        }
    }
    for (records.items) |record| {
        if (!containsStableOpId(attribute_ids.items, record.stable_op_id)) return error.MissingStableOpAttribute;
    }
    return records.toOwnedSlice(allocator);
}

const NamedStableLocation = struct {
    stable_op_id: u64,
    argument: []const u8,
};

fn collectLocationAliases(allocator: std.mem.Allocator, text: []const u8) ![]LocationAlias {
    var aliases: std.ArrayList(LocationAlias) = .empty;
    var lines = std.mem.splitScalar(u8, text, '\n');
    while (lines.next()) |line| {
        const trimmed = std.mem.trim(u8, line, " \t\r");
        if (!std.mem.startsWith(u8, trimmed, "#loc")) continue;
        const equals = std.mem.indexOfScalar(u8, trimmed, '=') orelse continue;
        const name = std.mem.trim(u8, trimmed[0..equals], " \t");
        const expression = std.mem.trim(u8, trimmed[equals + 1 ..], " \t");
        if (expression.len == 0) return error.InvalidStableOpLocation;
        try aliases.append(allocator, .{ .name = name, .expression = expression });
    }
    return aliases.toOwnedSlice(allocator);
}

fn findFrontendStableId(line: []const u8) !?u64 {
    const key = "zml.stable_op_id";
    const key_start = std.mem.indexOf(u8, line, key) orelse return null;
    var cursor = key_start + key.len;
    if (cursor < line.len and line[cursor] == '"') cursor += 1;
    while (cursor < line.len and std.ascii.isWhitespace(line[cursor])) : (cursor += 1) {}
    if (cursor == line.len or line[cursor] != '=') return error.InvalidStableOpId;
    cursor += 1;
    while (cursor < line.len and std.ascii.isWhitespace(line[cursor])) : (cursor += 1) {}

    const quoted = cursor < line.len and line[cursor] == '"';
    if (quoted) cursor += 1;
    const prefix = "zml.stable_op.";
    if (!std.mem.startsWith(u8, line[cursor..], prefix)) return error.InvalidStableOpId;
    cursor += prefix.len;
    const digits_start = cursor;
    while (cursor < line.len and std.ascii.isDigit(line[cursor])) : (cursor += 1) {}
    if (cursor == digits_start or (quoted and (cursor == line.len or line[cursor] != '"'))) {
        return error.InvalidStableOpId;
    }
    return std.fmt.parseInt(u64, line[digits_start..cursor], 10) catch error.InvalidStableOpId;
}

fn findNamedStableLocation(line: []const u8) !?NamedStableLocation {
    const marker = "\"zml.stable_op.";
    var cursor: usize = 0;
    var result: ?NamedStableLocation = null;
    while (std.mem.indexOfPos(u8, line, cursor, marker)) |start| {
        const digits_start = start + marker.len;
        var digits_end = digits_start;
        while (digits_end < line.len and std.ascii.isDigit(line[digits_end])) : (digits_end += 1) {}
        if (digits_end == digits_start or digits_end == line.len or line[digits_end] != '"') {
            cursor = start + marker.len;
            continue;
        }

        var open_paren = digits_end + 1;
        while (open_paren < line.len and std.ascii.isWhitespace(line[open_paren])) : (open_paren += 1) {}
        if (open_paren == line.len or line[open_paren] != '(') {
            cursor = digits_end + 1;
            continue;
        }

        const stable_op_id = std.fmt.parseInt(u64, line[digits_start..digits_end], 10) catch return error.InvalidStableOpId;
        const argument = try parenthesizedArgument(line, open_paren);
        if (result) |existing| {
            if (existing.stable_op_id != stable_op_id) return error.ConflictingStableOpId;
        } else {
            result = .{ .stable_op_id = stable_op_id, .argument = argument };
        }
        cursor = open_paren + argument.len + 2;
    }
    return result;
}

fn findLastLocationArgument(line: []const u8) !?[]const u8 {
    var cursor: usize = 0;
    var result: ?[]const u8 = null;
    while (std.mem.indexOfPos(u8, line, cursor, "loc(")) |start| {
        if (start != 0 and (std.ascii.isAlphanumeric(line[start - 1]) or line[start - 1] == '_')) {
            cursor = start + 4;
            continue;
        }
        result = try parenthesizedArgument(line, start + 3);
        cursor = start + 4;
    }
    return result;
}

fn parenthesizedArgument(text: []const u8, open_paren: usize) ![]const u8 {
    if (open_paren >= text.len or text[open_paren] != '(') return error.InvalidStableOpLocation;
    var depth: usize = 0;
    var in_string = false;
    var escaped = false;
    for (text[open_paren..], open_paren..) |character, index| {
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (character == '\\') {
                escaped = true;
            } else if (character == '"') {
                in_string = false;
            }
            continue;
        }
        if (character == '"') {
            in_string = true;
        } else if (character == '(') {
            depth += 1;
        } else if (character == ')') {
            if (depth == 0) return error.InvalidStableOpLocation;
            depth -= 1;
            if (depth == 0) return text[open_paren + 1 .. index];
        }
    }
    return error.InvalidStableOpLocation;
}

fn resolveLocation(
    allocator: std.mem.Allocator,
    expression: []const u8,
    aliases: []const LocationAlias,
    depth: u8,
) !ResolvedLocation {
    if (depth == 32) return error.InvalidStableOpLocation;
    const trimmed = std.mem.trim(u8, expression, " \t\r");
    if (trimmed.len == 0) return error.InvalidStableOpLocation;
    if (std.mem.eql(u8, trimmed, "[unknown]") or std.mem.eql(u8, trimmed, "unknown")) return .{};

    if (std.mem.startsWith(u8, trimmed, "loc(")) {
        return resolveLocation(allocator, try parenthesizedArgument(trimmed, 3), aliases, depth + 1);
    }

    if (try findNamedStableLocation(trimmed)) |named| {
        var resolved = try resolveLocation(allocator, named.argument, aliases, depth + 1);
        resolved.stable_op_id = try mergeStableOpIds(resolved.stable_op_id, named.stable_op_id);
        return resolved;
    }

    if (trimmed[0] == '#') {
        var end: usize = 1;
        while (end < trimmed.len and (std.ascii.isAlphanumeric(trimmed[end]) or trimmed[end] == '_' or trimmed[end] == '$')) : (end += 1) {}
        const alias_name = trimmed[0..end];
        for (aliases) |alias| {
            if (std.mem.eql(u8, alias.name, alias_name)) {
                return resolveLocation(allocator, alias.expression, aliases, depth + 1);
            }
        }
        return error.UnknownStableOpLocationAlias;
    }

    return .{ .source = try parseFileLineColumn(allocator, trimmed) };
}

fn parseFileLineColumn(allocator: std.mem.Allocator, expression: []const u8) !SourceLocation {
    const column_colon = std.mem.lastIndexOfScalar(u8, expression, ':') orelse return error.InvalidStableOpLocation;
    if (column_colon + 1 == expression.len) return error.InvalidStableOpLocation;
    const before_column = expression[0..column_colon];
    const line_colon = std.mem.lastIndexOfScalar(u8, before_column, ':') orelse return error.InvalidStableOpLocation;

    const line_text = std.mem.trim(u8, before_column[line_colon + 1 ..], " \t");
    const column_text = std.mem.trim(u8, expression[column_colon + 1 ..], " \t");
    const line = std.fmt.parseInt(u32, line_text, 10) catch return error.InvalidStableOpLocation;
    const column = std.fmt.parseInt(u32, column_text, 10) catch return error.InvalidStableOpLocation;
    if (line == 0 or column == 0) return error.InvalidStableOpLocation;

    const file_text = std.mem.trim(u8, before_column[0..line_colon], " \t");
    if (file_text.len == 0) return error.InvalidStableOpLocation;
    const file = if (file_text[0] == '"') blk: {
        if (file_text.len < 2 or file_text[file_text.len - 1] != '"') return error.InvalidStableOpLocation;
        break :blk try unescapeMlirString(allocator, file_text[1 .. file_text.len - 1]);
    } else file_text;

    return .{ .file = file, .line = line, .column = column };
}

fn unescapeMlirString(allocator: std.mem.Allocator, text: []const u8) ![]const u8 {
    if (std.mem.indexOfScalar(u8, text, '\\') == null) return text;
    var result: std.ArrayList(u8) = .empty;
    var cursor: usize = 0;
    while (cursor < text.len) {
        if (text[cursor] != '\\') {
            try result.append(allocator, text[cursor]);
            cursor += 1;
            continue;
        }
        if (cursor + 1 == text.len) return error.InvalidStableOpLocation;
        const escaped = text[cursor + 1];
        switch (escaped) {
            'n' => try result.append(allocator, '\n'),
            't' => try result.append(allocator, '\t'),
            '\\', '"' => try result.append(allocator, escaped),
            else => {
                if (cursor + 2 >= text.len) return error.InvalidStableOpLocation;
                const high = hexDigitValue(escaped) orelse return error.InvalidStableOpLocation;
                const low = hexDigitValue(text[cursor + 2]) orelse return error.InvalidStableOpLocation;
                try result.append(allocator, high * 16 + low);
                cursor += 1;
            },
        }
        cursor += 2;
    }
    return result.toOwnedSlice(allocator);
}

fn hexDigitValue(character: u8) ?u8 {
    if (character >= '0' and character <= '9') return character - '0';
    if (character >= 'a' and character <= 'f') return character - 'a' + 10;
    if (character >= 'A' and character <= 'F') return character - 'A' + 10;
    return null;
}

fn mergeStableOpIds(left: ?u64, right: ?u64) !?u64 {
    if (left != null and right != null and left.? != right.?) return error.ConflictingStableOpId;
    return left orelse right;
}

fn upsertProvenanceRecordSorted(
    allocator: std.mem.Allocator,
    records: *std.ArrayList(ProvenanceRecord),
    record: ProvenanceRecord,
) !void {
    var index: usize = 0;
    while (index < records.items.len and records.items[index].stable_op_id < record.stable_op_id) : (index += 1) {}
    if (index < records.items.len and records.items[index].stable_op_id == record.stable_op_id) {
        const existing = records.items[index];
        if (existing.line != record.line or existing.column != record.column or !std.mem.eql(u8, existing.file, record.file)) {
            return error.ConflictingStableOpLocation;
        }
        return;
    }
    try records.insert(allocator, index, record);
}

fn containsProvenanceRecord(records: []const ProvenanceRecord, stable_op_id: u64) bool {
    for (records) |record| {
        if (record.stable_op_id == stable_op_id) return true;
    }
    return false;
}

fn containsStableOpId(ids: []const u64, stable_op_id: u64) bool {
    return std.mem.indexOfScalar(u64, ids, stable_op_id) != null;
}

fn parseSourceMap(allocator: std.mem.Allocator, json: []const u8) ![]SourceSpan {
    const parsed = try std.json.parseFromSliceLeaky(std.json.Value, allocator, json, .{});
    if (parsed != .object) return error.InvalidSourceMap;

    const root = parsed.object;
    const original_file = try jsonOptionalString(root.get("original_file") orelse root.get("logical_path"));
    const expressions_value = root.get("expressions") orelse root.get("mappings") orelse root.get("source_expressions") orelse root.get("spans") orelse return error.InvalidSourceMap;
    if (expressions_value != .array) return error.InvalidSourceMap;

    var spans: std.ArrayList(SourceSpan) = .empty;
    for (expressions_value.array.items) |value| {
        if (value != .object) return error.InvalidSourceMap;
        const object = value.object;

        const file = (try jsonOptionalString(firstObjectValue(&object, &.{ "file", "original_file", "filename" }))) orelse original_file orelse return error.InvalidSourceMap;
        const line = try jsonSourceUnsigned(u32, firstObjectValue(&object, &.{ "line", "start_line", "startLine" }));
        const column = try jsonSourceUnsigned(u32, firstObjectValue(&object, &.{ "column", "start_column", "startColumn", "col" }));
        const end_line = try jsonSourceUnsigned(u32, firstObjectValue(&object, &.{ "end_line", "endLine" }));
        const end_column = try jsonSourceUnsigned(u32, firstObjectValue(&object, &.{ "end_column", "endColumn" }));
        const start_byte = try jsonSourceUnsignedOptional(usize, firstObjectValue(&object, &.{ "start_byte", "startByte" }));
        const end_byte = try jsonSourceUnsignedOptional(usize, firstObjectValue(&object, &.{ "end_byte", "endByte" }));
        const provenance_line = (try jsonSourceUnsignedOptional(u32, firstObjectValue(&object, &.{ "provenance_line", "provenanceLine" }))) orelse line;
        const provenance_column = (try jsonSourceUnsignedOptional(u32, firstObjectValue(&object, &.{ "provenance_column", "provenanceColumn" }))) orelse column;

        if (line == 0 or column == 0 or provenance_line == 0 or provenance_column == 0 or end_line < line or (end_line == line and end_column <= column)) {
            return error.InvalidSourceMap;
        }
        if (start_byte != null and end_byte != null and end_byte.? <= start_byte.?) {
            return error.InvalidSourceMap;
        }

        try spans.append(allocator, .{
            .file = file,
            .line = line,
            .column = column,
            .end_line = end_line,
            .end_column = end_column,
            .start_byte = start_byte,
            .end_byte = end_byte,
            .method = try jsonOptionalString(firstObjectValue(&object, &.{ "method", "instrumented_method", "callee", "operation" })),
            .provenance_line = provenance_line,
            .provenance_column = provenance_column,
        });
    }
    return spans.toOwnedSlice(allocator);
}

fn firstObjectValue(object: *const std.json.ObjectMap, names: []const []const u8) ?std.json.Value {
    for (names) |name| {
        if (object.get(name)) |value| return value;
    }
    return null;
}

fn jsonSourceUnsigned(comptime T: type, value: ?std.json.Value) !T {
    return (try jsonSourceUnsignedOptional(T, value)) orelse error.InvalidSourceMap;
}

fn jsonSourceUnsignedOptional(comptime T: type, value: ?std.json.Value) !?T {
    return switch (value orelse return null) {
        .null => null,
        .integer => |number| std.math.cast(T, number) orelse error.InvalidSourceMap,
        else => error.InvalidSourceMap,
    };
}

fn jsonOptionalString(value: ?std.json.Value) !?[]const u8 {
    return switch (value orelse return null) {
        .null => null,
        .string => |string| string,
        else => error.InvalidSourceMap,
    };
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
    return buildMappingWithSpans(allocator, records, stablehlo, hlo, &.{});
}

fn buildMappingWithSpans(
    allocator: std.mem.Allocator,
    records: []const ProvenanceRecord,
    stablehlo: []const u8,
    hlo: []const u8,
    source_spans: []const SourceSpan,
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

    for (source_builders.items) |*source| {
        const span = findBestSourceSpan(source.*, source_spans) orelse {
            if (source_spans.len != 0) return error.MissingSourceSpan;
            continue;
        };
        source.provenance_line = source.line;
        source.provenance_column = source.column;
        source.file = span.file;
        source.line = span.line;
        source.column = span.column;
        source.end_line = span.end_line;
        source.end_column = span.end_column;
        source.start_byte = span.start_byte;
        source.end_byte = span.end_byte;
        source.method = span.method;
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
            .end_line = builder.end_line,
            .end_column = builder.end_column,
            .start_byte = builder.start_byte,
            .end_byte = builder.end_byte,
            .method = builder.method,
            .provenance_line = builder.provenance_line,
            .provenance_column = builder.provenance_column,
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

fn findBestSourceSpan(source: SourceBuilder, spans: []const SourceSpan) ?*const SourceSpan {
    var basename_match: ?*const SourceSpan = null;

    for (spans) |*span| {
        const exact_file = std.mem.eql(u8, source.file, span.file);
        if (!exact_file and !std.mem.eql(u8, std.fs.path.basename(source.file), std.fs.path.basename(span.file))) continue;
        if (source.line != span.provenance_line or source.column != span.provenance_column) continue;
        if (exact_file) return span;
        basename_match = span;
    }
    return basename_match;
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
            try appendStableOperands(allocator, stable, values[1..]);
        }
    }
}

fn appendStableOperands(
    allocator: std.mem.Allocator,
    stable: *StableOpBuilder,
    printed_operands: []const []const u8,
) !void {
    // StableHLO prints reduce operands as `(input init: value)` pairs, while
    // XLA HLO stores all inputs first and all init values second. Normalize to
    // the physical operand order before correlating producers by position.
    if (std.mem.eql(u8, stable.operation orelse "", "stablehlo.reduce") and printed_operands.len % 2 == 0) {
        const pair_count = printed_operands.len / 2;
        for (0..pair_count) |index| {
            try stable.operands.append(allocator, printed_operands[index * 2]);
        }
        for (0..pair_count) |index| {
            try stable.operands.append(allocator, printed_operands[index * 2 + 1]);
        }
        return;
    }

    for (printed_operands) |operand| {
        try stable.operands.append(allocator, operand);
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
        if (!metadataOpcodeMatchesStable(stable.operation orelse continue, hlo.opcode)) continue;
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
                if (!hloOpcodeMatchesStable(producer_stable.operation orelse continue, producer_hlo.opcode)) continue;

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
    const stable_operation = stable.operation orelse return null;
    var match: ?*HloInstructionBuilder = null;
    for (hlo_instructions) |*instruction| {
        const stable_id = instruction.stable_op_id orelse continue;
        if (!std.mem.eql(u8, stable_id, stable.id) or !hloOpcodeMatchesStable(stable_operation, instruction.opcode)) continue;
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
    if (std.mem.eql(u8, opcode, "dot_general")) return "dot";
    return opcode;
}

fn hloOpcodeMatchesStable(stable_operation: []const u8, hlo_opcode: []const u8) bool {
    const expected = normalizedStableOpcode(stable_operation);
    if (std.mem.eql(u8, expected, hlo_opcode)) return true;
    return std.mem.eql(u8, stable_operation, "stablehlo.broadcast_in_dim") and std.mem.eql(u8, hlo_opcode, "reshape");
}

fn metadataOpcodeMatchesStable(stable_operation: []const u8, hlo_opcode: []const u8) bool {
    if (hloOpcodeMatchesStable(stable_operation, hlo_opcode)) return true;
    return std.mem.eql(u8, stable_operation, "stablehlo.reduce") and std.mem.eql(u8, hlo_opcode, "get-tuple-element");
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
    var cursor = equals + 1;
    while (std.mem.indexOfPos(u8, line, cursor, "(")) |open_paren| {
        const before = std.mem.trimEnd(u8, line[equals + 1 .. open_paren], " \t");
        const token_start = std.mem.lastIndexOfAny(u8, before, " \t") orelse 0;
        const opcode = if (token_start == 0) before else before[token_start + 1 ..];
        if (opcode.len != 0 and isHloOpcode(opcode)) return try allocator.dupe(u8, opcode);
        cursor = open_paren + 1;
    }
    return null;
}

fn isHloOpcode(value: []const u8) bool {
    for (value) |character| {
        if (!std.ascii.isAlphanumeric(character) and character != '-' and character != '_') return false;
    }
    return true;
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

test "derive provenance from pretty StableHLO locations and frontend attributes" {
    const stablehlo =
        \\  %cst = stablehlo.constant {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.1"}} dense<2.0> : tensor<f32> "zml.stable_op.1"(#loc1)
        \\  %sum = stablehlo.add %arg0, %arg1 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.0"}} : tensor<4xf32> "zml.stable_op.0"(#loc2)
        \\#loc1 = source.zig:5:42
        \\#loc2 = tools/source_to_hlo_explorer/source.zig:4:28
        \\#loc3 = "zml.stable_op.1"(#loc1)
        \\#loc4 = "zml.stable_op.0"(#loc2)
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const records = try deriveProvenanceRecords(arena_state.allocator(), stablehlo);

    try std.testing.expectEqual(@as(usize, 2), records.len);
    try std.testing.expectEqual(@as(u64, 0), records[0].stable_op_id);
    try std.testing.expectEqualStrings("tools/source_to_hlo_explorer/source.zig", records[0].file);
    try std.testing.expectEqual(@as(u32, 4), records[0].line);
    try std.testing.expectEqual(@as(u32, 28), records[0].column);
    try std.testing.expectEqual(@as(u64, 1), records[1].stable_op_id);
    try std.testing.expectEqualStrings("source.zig", records[1].file);
    try std.testing.expectEqual(@as(u32, 5), records[1].line);
    try std.testing.expectEqual(@as(u32, 42), records[1].column);
}

test "derive provenance resolves generic location alias chains" {
    const stablehlo =
        \\  %0 = stablehlo.convert %arg0 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.7"}} : (tensor<i32>) -> tensor<i8> loc(#loc2)
        \\#loc1 = loc("C:\5Csrc\5Cmodel.zig":12:34)
        \\#loc2 = loc("zml.stable_op.7"(#loc1))
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const records = try deriveProvenanceRecords(arena_state.allocator(), stablehlo);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(u64, 7), records[0].stable_op_id);
    try std.testing.expectEqualStrings("C:\\src\\model.zig", records[0].file);
    try std.testing.expectEqual(@as(u32, 12), records[0].line);
    try std.testing.expectEqual(@as(u32, 34), records[0].column);
}

test "derive provenance joins multiline region operations through the location table" {
    const stablehlo =
        \\  %0:2 = stablehlo.reduce(%arg0 init: %init) across dimensions = [0] {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.7"}} : (tensor<4xf32>, tensor<f32>) -> (tensor<f32>, tensor<i32>)
        \\   reducer(%left: tensor<f32>, %right: tensor<f32>) {
        \\    %1 = stablehlo.compare  GT, %left, %right : (tensor<f32>, tensor<f32>) -> tensor<i1>
        \\    stablehlo.return %left : tensor<f32>
        \\   } "zml.stable_op.7"(#loc1)
        \\#loc1 = model.zig:42:18
        \\#loc2 = "zml.stable_op.7"(#loc1)
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const records = try deriveProvenanceRecords(arena_state.allocator(), stablehlo);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(u64, 7), records[0].stable_op_id);
    try std.testing.expectEqualStrings("model.zig", records[0].file);
    try std.testing.expectEqual(@as(u32, 42), records[0].line);
    try std.testing.expectEqual(@as(u32, 18), records[0].column);
}

test "derive provenance ignores operations without source locations" {
    const stablehlo =
        \\  %cst = stablehlo.constant {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.4"}} dense<0.0> : tensor<f32> [unknown]
        \\  %0 = stablehlo.add %arg0, %arg1 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.5"}} : tensor<4xf32> "zml.stable_op.5"(#loc1)
        \\#loc1 = model.zig:8:12
        \\#loc2 = "zml.stable_op.5"(#loc1)
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const records = try deriveProvenanceRecords(arena_state.allocator(), stablehlo);

    try std.testing.expectEqual(@as(usize, 1), records.len);
    try std.testing.expectEqual(@as(u64, 5), records[0].stable_op_id);
}

test "derive provenance rejects conflicting embedded ids" {
    const stablehlo =
        \\  %0 = stablehlo.add %arg0, %arg1 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.1"}} : tensor<4xf32> "zml.stable_op.2"(#loc1)
        \\#loc1 = source.zig:1:1
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    try std.testing.expectError(
        error.ConflictingStableOpId,
        deriveProvenanceRecords(arena_state.allocator(), stablehlo),
    );
}

test "write bundle from texts derives provenance and writes four artifacts" {
    const source =
        \\fn forward() void {}
    ;
    const stablehlo =
        \\  %0 = stablehlo.add %arg0, %arg1 {mhlo.frontend_attributes = {zml.stable_op_id = "zml.stable_op.0"}} : tensor<4xf32> "zml.stable_op.0"(#loc1)
        \\#loc1 = source.zig:1:4
    ;
    const hlo =
        \\  ROOT %add.1 = f32[4]{0} add(%arg0.1, %arg1.1), metadata={op_name="zml.stable_op.0"}
    ;

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const output_dir = try std.fmt.allocPrint(
        std.testing.allocator,
        ".zig-cache/tmp/{s}/bundle",
        .{tmp.sub_path[0..]},
    );
    defer std.testing.allocator.free(output_dir);

    try writeBundleFromTexts(
        std.testing.allocator,
        std.testing.io,
        output_dir,
        source,
        null,
        stablehlo,
        hlo,
    );

    var bundle = try tmp.dir.openDir(std.testing.io, "bundle", .{ .iterate = true });
    defer bundle.close(std.testing.io);
    const written_source = try bundle.readFileAlloc(std.testing.io, "source.zig", std.testing.allocator, .unlimited);
    defer std.testing.allocator.free(written_source);
    const written_stablehlo = try bundle.readFileAlloc(std.testing.io, "stablehlo.mlir", std.testing.allocator, .unlimited);
    defer std.testing.allocator.free(written_stablehlo);
    const written_hlo = try bundle.readFileAlloc(std.testing.io, "hlo.before_optimizations.txt", std.testing.allocator, .unlimited);
    defer std.testing.allocator.free(written_hlo);
    const mapping_json = try bundle.readFileAlloc(std.testing.io, "mapping.json", std.testing.allocator, .unlimited);
    defer std.testing.allocator.free(mapping_json);

    try std.testing.expectEqualStrings(source, written_source);
    try std.testing.expectEqualStrings(stablehlo, written_stablehlo);
    try std.testing.expectEqualStrings(hlo, written_hlo);
    try std.testing.expect(std.mem.indexOf(u8, mapping_json, "\"stable_op_ids\"") != null);
    try std.testing.expect(std.mem.indexOf(u8, mapping_json, "\"%add.1\"") != null);

    var file_count: usize = 0;
    var iterator = bundle.iterate();
    while (try iterator.next(std.testing.io)) |entry| {
        if (entry.kind == .file) file_count += 1;
    }
    try std.testing.expectEqual(@as(usize, 4), file_count);
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

test "AST sidecar enriches compiler provenance with exact expression spans" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"source.zig","line":4,"column":17}]}
    ;
    const source_map =
        \\{
        \\  "version": 1,
        \\  "original_file": "source.zig",
        \\  "expressions": [{
        \\    "file": "source.zig",
        \\    "line": 4,
        \\    "column": 17,
        \\    "end_line": 4,
        \\    "end_column": 25,
        \\    "start_byte": 103,
        \\    "end_byte": 111,
        \\    "method": "add",
        \\    "instrumented_method": "addAt",
        \\    "function": "forward"
        \\  }]
        \\}
    ;
    const stablehlo =
        \\  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> loc("zml.stable_op.1")
    ;
    const hlo =
        \\  ROOT %add.1 = f32[4]{0} add(%arg0.1, %arg1.1), metadata={op_name="zml.stable_op.1"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try parseProvenance(arena, provenance);
    const spans = try parseSourceMap(arena, source_map);
    const mapping = try buildMappingWithSpans(arena, records, stablehlo, hlo, spans);

    try std.testing.expectEqual(@as(usize, 1), mapping.sources.len);
    try std.testing.expectEqualStrings("source.zig:4:17", mapping.sources[0].id);
    try std.testing.expectEqual(@as(u32, 4), mapping.sources[0].line);
    try std.testing.expectEqual(@as(u32, 17), mapping.sources[0].column);
    try std.testing.expectEqual(@as(u32, 4), mapping.sources[0].end_line.?);
    try std.testing.expectEqual(@as(u32, 25), mapping.sources[0].end_column.?);
    try std.testing.expectEqual(@as(usize, 103), mapping.sources[0].start_byte.?);
    try std.testing.expectEqual(@as(usize, 111), mapping.sources[0].end_byte.?);
    try std.testing.expectEqualStrings("add", mapping.sources[0].method.?);
    try std.testing.expectEqual(@as(u32, 4), mapping.sources[0].provenance_line.?);
    try std.testing.expectEqual(@as(u32, 17), mapping.sources[0].provenance_column.?);
}

test "AST provenance anchors disambiguate nested call spans" {
    const source_map =
        \\{
        \\  "version": 2,
        \\  "original_file": "mnist.zig",
        \\  "expressions": [
        \\    {
        \\      "file": "mnist.zig",
        \\      "line": 28,
        \\      "column": 16,
        \\      "end_line": 28,
        \\      "end_column": 41,
        \\      "start_byte": 500,
        \\      "end_byte": 525,
        \\      "provenance_line": 28,
        \\      "provenance_column": 28,
        \\      "method": "dot"
        \\    },
        \\    {
        \\      "file": "mnist.zig",
        \\      "line": 28,
        \\      "column": 41,
        \\      "end_line": 28,
        \\      "end_column": 56,
        \\      "start_byte": 525,
        \\      "end_byte": 540,
        \\      "provenance_line": 28,
        \\      "provenance_column": 42,
        \\      "method": "add"
        \\    }
        \\  ]
        \\}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const spans = try parseSourceMap(arena_state.allocator(), source_map);

    const source: SourceBuilder = .{
        .id = "mnist.zig:28:42",
        .file = "mnist.zig",
        .line = 28,
        .column = 42,
    };
    const span = findBestSourceSpan(source, spans).?;
    try std.testing.expectEqualStrings("add", span.method.?);
    try std.testing.expectEqual(@as(u32, 41), span.column);
}

test "mapping remains point-based without an AST sidecar" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"source.zig","line":4,"column":28}]}
    ;
    const stablehlo =
        \\  %0 = stablehlo.add %arg0, %arg1 : tensor<4xf32> loc("zml.stable_op.1")
    ;
    const hlo =
        \\  ROOT %add.1 = f32[4]{0} add(%arg0.1, %arg1.1), metadata={op_name="zml.stable_op.1"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const mapping = try buildMapping(arena_state.allocator(), try parseProvenance(arena_state.allocator(), provenance), stablehlo, hlo);

    try std.testing.expectEqual(@as(u32, 28), mapping.sources[0].column);
    try std.testing.expect(mapping.sources[0].end_line == null);
    try std.testing.expect(mapping.sources[0].start_byte == null);
    try std.testing.expect(mapping.sources[0].provenance_column == null);
}

test "AST sidecar must cover every compiler provenance point" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"source.zig","line":4,"column":17}]}
    ;
    const source_map =
        \\{
        \\  "version": 1,
        \\  "original_file": "other.zig",
        \\  "expressions": [{
        \\    "file": "other.zig",
        \\    "line": 4,
        \\    "column": 17,
        \\    "end_line": 4,
        \\    "end_column": 25,
        \\    "start_byte": 103,
        \\    "end_byte": 111,
        \\    "method": "add"
        \\  }]
        \\}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();

    const records = try parseProvenance(arena, provenance);
    const spans = try parseSourceMap(arena, source_map);
    try std.testing.expectError(error.MissingSourceSpan, buildMappingWithSpans(arena, records, "", "", spans));
}

test "AST sidecar requires an exact provenance anchor" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"source.zig","line":4,"column":18}]}
    ;
    const source_map =
        \\{
        \\  "version": 2,
        \\  "original_file": "source.zig",
        \\  "expressions": [{
        \\    "file": "source.zig",
        \\    "line": 4,
        \\    "column": 17,
        \\    "end_line": 4,
        \\    "end_column": 25,
        \\    "start_byte": 103,
        \\    "end_byte": 111,
        \\    "provenance_line": 4,
        \\    "provenance_column": 19,
        \\    "method": "add"
        \\  }]
        \\}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const records = try parseProvenance(arena, provenance);
    const spans = try parseSourceMap(arena, source_map);
    try std.testing.expectError(error.MissingSourceSpan, buildMappingWithSpans(arena, records, "", "", spans));
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

test "dataflow normalizes reduce input and init operand order" {
    const provenance =
        \\{"records":[
        \\  {"stable_op_id":13,"file":"source.zig","line":1,"column":1},
        \\  {"stable_op_id":14,"file":"source.zig","line":1,"column":1},
        \\  {"stable_op_id":24,"file":"source.zig","line":1,"column":1}
        \\]}
    ;
    const stablehlo =
        \\  %cst = stablehlo.constant dense<0xFF800000> : tensor<f32> loc("zml.stable_op.13")
        \\  %c = stablehlo.constant dense<0> : tensor<i32> loc("zml.stable_op.14")
        \\  %result:2 = stablehlo.reduce(%values init: %cst), (%indices init: %c) across dimensions = [0] : (tensor<10xf32>, tensor<10xi32>, tensor<f32>, tensor<i32>) -> (tensor<f32>, tensor<i32>) loc("zml.stable_op.24")
    ;
    const hlo =
        \\  %constant.7 = f32[] constant(-inf)
        \\  %constant.6 = s32[] constant(0)
        \\  %reduce.1 = (f32[], s32[]) reduce(%values.1, %indices.1, %constant.7, %constant.6), dimensions={0}, metadata={op_name="zml.stable_op.24"}
        \\  %value.1 = f32[] get-tuple-element(%reduce.1), index=0, metadata={op_name="zml.stable_op.24"}
        \\  ROOT %index.1 = s32[] get-tuple-element(%reduce.1), index=1, metadata={op_name="zml.stable_op.24"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mapping = try buildMapping(arena, try parseProvenance(arena, provenance), stablehlo, hlo);

    var found_float_init = false;
    var found_integer_init = false;
    for (mapping.hlo_instructions) |instruction| {
        if (std.mem.eql(u8, instruction.id, "%constant.7")) {
            found_float_init = true;
            try std.testing.expectEqualStrings("13", instruction.stable_op_id);
        }
        if (std.mem.eql(u8, instruction.id, "%constant.6")) {
            found_integer_init = true;
            try std.testing.expectEqualStrings("14", instruction.stable_op_id);
        }
    }
    try std.testing.expect(found_float_init);
    try std.testing.expect(found_integer_init);
}

test "dataflow accepts scalar broadcast lowered to reshape" {
    const provenance =
        \\{"records":[
        \\  {"stable_op_id":0,"file":"source.zig","line":1,"column":1},
        \\  {"stable_op_id":1,"file":"source.zig","line":1,"column":1},
        \\  {"stable_op_id":2,"file":"source.zig","line":1,"column":1}
        \\]}
    ;
    const stablehlo =
        \\  %cst = stablehlo.constant dense<0> : tensor<i32> loc("zml.stable_op.0")
        \\  %wide = stablehlo.broadcast_in_dim %cst, dims = [] : (tensor<i32>) -> tensor<1xi32> loc("zml.stable_op.1")
        \\  %result = stablehlo.add %wide, %wide : tensor<1xi32> loc("zml.stable_op.2")
    ;
    const hlo =
        \\  %constant.1 = s32[] constant(0)
        \\  %reshape.1 = s32[1]{0} reshape(%constant.1)
        \\  ROOT %add.1 = s32[1]{0} add(%reshape.1, %reshape.1), metadata={op_name="zml.stable_op.2"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mapping = try buildMapping(arena, try parseProvenance(arena, provenance), stablehlo, hlo);

    try std.testing.expectEqual(@as(usize, 3), mapping.hlo_instructions.len);
    for (mapping.hlo_instructions) |instruction| {
        if (std.mem.eql(u8, instruction.id, "%reshape.1")) {
            try std.testing.expectEqualStrings("1", instruction.stable_op_id);
            try std.testing.expectEqualStrings("dataflow_operand", instruction.mapping);
        }
        if (std.mem.eql(u8, instruction.id, "%constant.1")) {
            try std.testing.expectEqualStrings("0", instruction.stable_op_id);
            try std.testing.expectEqualStrings("dataflow_operand", instruction.mapping);
        }
    }
}

test "direct metadata rejects inherited compiler operations" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"source.zig","line":1,"column":1}]}
    ;
    const stablehlo =
        \\  %result = stablehlo.convert %arg0 : (tensor<i32>) -> tensor<i8> loc("zml.stable_op.1")
    ;
    const hlo =
        \\  %convert.1 = s8[1]{0} convert(%arg0.1), metadata={op_name="zml.stable_op.1"}
        \\  ROOT %sharding.1 = s8[1]{0} custom-call(%convert.1), custom_call_target="xla.sdy.FuncResultSharding", metadata={op_name="zml.stable_op.1"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mapping = try buildMapping(arena, try parseProvenance(arena, provenance), stablehlo, hlo);

    try std.testing.expectEqual(@as(usize, 1), mapping.hlo_instructions.len);
    try std.testing.expectEqualStrings("%convert.1", mapping.hlo_instructions[0].id);
}

test "direct metadata keeps reduce result projections" {
    const provenance =
        \\{"records":[{"stable_op_id":1,"file":"source.zig","line":1,"column":1}]}
    ;
    const stablehlo =
        \\  %result:2 = stablehlo.reduce(%arg0 init: %init) applies stablehlo.add across dimensions = [0] : (tensor<4xf32>, tensor<f32>) -> (tensor<f32>) loc("zml.stable_op.1")
    ;
    const hlo =
        \\  %reduce.1 = (f32[], s32[]) reduce(%arg0.1, %init.1), dimensions={0}, metadata={op_name="zml.stable_op.1"}
        \\  %value.1 = f32[] get-tuple-element(%reduce.1), index=0, metadata={op_name="zml.stable_op.1"}
        \\  ROOT %index.1 = s32[] get-tuple-element(%reduce.1), index=1, metadata={op_name="zml.stable_op.1"}
    ;

    var arena_state: std.heap.ArenaAllocator = .init(std.testing.allocator);
    defer arena_state.deinit();
    const arena = arena_state.allocator();
    const mapping = try buildMapping(arena, try parseProvenance(arena, provenance), stablehlo, hlo);
    try std.testing.expectEqual(@as(usize, 3), mapping.hlo_instructions.len);
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
