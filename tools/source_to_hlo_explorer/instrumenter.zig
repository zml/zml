const std = @import("std");

const Ast = std.zig.Ast;
const Allocator = std.mem.Allocator;

const registered_operations = [_]Operation{
    .{ .method = "add", .instrumented_method = "addAt" },
    .{ .method = "argMax", .instrumented_method = "argMaxAt", .result = .arg_max_result },
    .{ .method = "convert", .instrumented_method = "convertAt" },
    .{ .method = "dot", .instrumented_method = "dotAt" },
    .{ .method = "flatten", .instrumented_method = "flattenAt" },
    .{ .method = "mulConstant", .instrumented_method = "mulConstantAt" },
    .{ .method = "relu", .instrumented_method = "reluAt" },
};

const tensor_passthrough_methods = [_][]const u8{
    "withTags",
};

const ExpressionType = enum {
    unknown,
    tensor,
    arg_max_result,
};

const Operation = struct {
    method: []const u8,
    instrumented_method: []const u8,
    result: ExpressionType = .tensor,
};

pub const Expression = struct {
    file: []const u8,
    line: u32,
    column: u32,
    end_line: u32,
    end_column: u32,
    start_byte: u32,
    end_byte: u32,
    provenance_line: u32,
    provenance_column: u32,
    method: []const u8,
    instrumented_method: []const u8,
    function: []const u8,
};

pub const SourceMap = struct {
    version: u32 = 2,
    original_file: []const u8,
    expressions: []const Expression,
};

pub const TransformResult = struct {
    generated_source: []u8,
    source_map_json: []u8,
    expressions: []Expression,

    pub fn deinit(self: *TransformResult, allocator: Allocator) void {
        allocator.free(self.generated_source);
        allocator.free(self.source_map_json);
        allocator.free(self.expressions);
        self.* = undefined;
    }
};

const Candidate = struct {
    operation: Operation,
    function: []const u8,
    method_start: u32,
    method_end: u32,
    argument_insert: u32,
    has_arguments: bool,
    expression: Expression,
};

const Edit = struct {
    start: u32,
    end: u32,
    replacement: []const u8,
    owned: bool = false,
};

const Position = struct {
    line: u32,
    column: u32,
};

const FieldType = struct {
    container: []const u8,
    name: []const u8,
    type_name: []const u8,
};

const NamedType = struct {
    name: []const u8,
    type_name: []const u8,
};

pub fn transform(
    allocator: Allocator,
    source: [:0]const u8,
    logical_path: []const u8,
    source_map_basename: []const u8,
) !TransformResult {
    var tree = try Ast.parse(allocator, source, .zig);
    defer tree.deinit(allocator);
    if (tree.errors.len != 0) return error.InvalidZigSource;

    var candidates: std.ArrayList(Candidate) = .empty;
    defer candidates.deinit(allocator);

    var field_types: std.ArrayList(FieldType) = .empty;
    defer field_types.deinit(allocator);
    try collectFieldTypes(allocator, &tree, &field_types);

    for (0..tree.nodes.len) |node_i| {
        const function_node: Ast.Node.Index = @enumFromInt(node_i);
        if (tree.nodeTag(function_node) != .fn_decl) continue;
        try collectFunctionCandidates(allocator, &tree, function_node, logical_path, field_types.items, &candidates);
    }

    std.mem.sort(Candidate, candidates.items, {}, struct {
        fn lessThan(_: void, lhs: Candidate, rhs: Candidate) bool {
            if (lhs.expression.start_byte != rhs.expression.start_byte) {
                return lhs.expression.start_byte < rhs.expression.start_byte;
            }
            return lhs.method_start < rhs.method_start;
        }
    }.lessThan);

    const expressions = try allocator.alloc(Expression, candidates.items.len);
    errdefer allocator.free(expressions);
    for (candidates.items, expressions) |candidate, *expression| expression.* = candidate.expression;

    var edits: std.ArrayList(Edit) = .empty;
    defer {
        for (edits.items) |edit| if (edit.owned) allocator.free(edit.replacement);
        edits.deinit(allocator);
    }

    for (candidates.items) |candidate| {
        try edits.append(allocator, .{
            .start = candidate.method_start,
            .end = candidate.method_end,
            .replacement = candidate.operation.instrumented_method,
        });
        const separator = if (candidate.has_arguments) ", " else "";
        const provenance_argument = try std.fmt.allocPrint(
            allocator,
            "{s}.{{ .module = @src().module, .file = \"{f}\", .fn_name = \"{f}\", .line = {d}, .column = {d} }}",
            .{
                separator,
                std.zig.fmtString(logical_path),
                std.zig.fmtString(candidate.function),
                candidate.expression.provenance_line,
                candidate.expression.provenance_column,
            },
        );
        errdefer allocator.free(provenance_argument);
        try edits.append(allocator, .{
            .start = candidate.argument_insert,
            .end = candidate.argument_insert,
            .replacement = provenance_argument,
            .owned = true,
        });
    }

    std.mem.sort(Edit, edits.items, {}, struct {
        fn lessThan(_: void, lhs: Edit, rhs: Edit) bool {
            if (lhs.start != rhs.start) return lhs.start < rhs.start;
            return lhs.end < rhs.end;
        }
    }.lessThan);

    var generated: std.ArrayList(u8) = .empty;
    defer generated.deinit(allocator);
    var cursor: usize = 0;
    for (edits.items) |edit| {
        if (edit.start < cursor or edit.end < edit.start) return error.OverlappingEdits;
        try generated.appendSlice(allocator, source[cursor..edit.start]);
        try generated.appendSlice(allocator, edit.replacement);
        cursor = edit.end;
    }
    try generated.appendSlice(allocator, source[cursor..]);
    if (generated.items.len == 0 or generated.items[generated.items.len - 1] != '\n') {
        try generated.append(allocator, '\n');
    }
    const source_map_declaration = try std.fmt.allocPrint(
        allocator,
        "pub const zml_source_map_json = @embedFile(\"{f}\");\n",
        .{std.zig.fmtString(source_map_basename)},
    );
    defer allocator.free(source_map_declaration);
    try generated.appendSlice(allocator, source_map_declaration);

    const source_map_json = try std.json.Stringify.valueAlloc(
        allocator,
        SourceMap{ .original_file = logical_path, .expressions = expressions },
        .{ .whitespace = .indent_2 },
    );
    errdefer allocator.free(source_map_json);

    return .{
        .generated_source = try generated.toOwnedSlice(allocator),
        .source_map_json = source_map_json,
        .expressions = expressions,
    };
}

fn collectFunctionCandidates(
    allocator: Allocator,
    tree: *const Ast,
    function_node: Ast.Node.Index,
    logical_path: []const u8,
    field_types: []const FieldType,
    candidates: *std.ArrayList(Candidate),
) !void {
    const proto_node, const body_node = tree.nodeData(function_node).node_and_node;
    var proto_buffer: [1]Ast.Node.Index = undefined;
    var proto = tree.fullFnProto(&proto_buffer, proto_node) orelse return;
    const function_name_token = proto.name_token orelse return;
    const function_name = tree.tokenSlice(function_name_token);

    var tensor_names: std.ArrayList([]const u8) = .empty;
    defer tensor_names.deinit(allocator);
    var named_types: std.ArrayList(NamedType) = .empty;
    defer named_types.deinit(allocator);

    var param_it = proto.iterate(tree);
    while (param_it.next()) |param| {
        const name_token = param.name_token orelse continue;
        const type_node = param.type_expr orelse continue;
        try appendNamedType(allocator, &named_types, tree.tokenSlice(name_token), tree.getNodeSource(type_node));
        if (isTensorType(tree, type_node)) {
            try appendUniqueName(allocator, &tensor_names, tree.tokenSlice(name_token));
        }
    }

    const body_start = tree.tokenStart(tree.firstToken(body_node));
    const body_last_token = tree.lastToken(body_node);
    const body_end = tree.tokenStart(body_last_token) + @as(u32, @intCast(tree.tokenSlice(body_last_token).len));

    var declarations: std.ArrayList(Ast.Node.Index) = .empty;
    defer declarations.deinit(allocator);
    for (0..tree.nodes.len) |node_i| {
        const node: Ast.Node.Index = @enumFromInt(node_i);
        const declaration = tree.fullVarDecl(node) orelse continue;
        const node_start = tree.tokenStart(declaration.firstToken());
        const node_last = tree.lastToken(node);
        const node_end = tree.tokenStart(node_last) + @as(u32, @intCast(tree.tokenSlice(node_last).len));
        if (node_start > body_start and node_end < body_end) try declarations.append(allocator, node);
    }
    std.mem.sort(Ast.Node.Index, declarations.items, tree, struct {
        fn lessThan(context: *const Ast, lhs: Ast.Node.Index, rhs: Ast.Node.Index) bool {
            return context.tokenStart(context.firstToken(lhs)) < context.tokenStart(context.firstToken(rhs));
        }
    }.lessThan);

    for (declarations.items) |declaration_node| {
        const declaration = tree.fullVarDecl(declaration_node).?;
        const name_token = declaration.ast.mut_token + 1;
        if (tree.tokenTag(name_token) != .identifier) continue;

        var is_tensor = false;
        if (declaration.ast.type_node.unwrap()) |type_node| {
            try appendNamedType(allocator, &named_types, tree.tokenSlice(name_token), tree.getNodeSource(type_node));
            is_tensor = isTensorType(tree, type_node);
        }
        if (!is_tensor) {
            if (declaration.ast.init_node.unwrap()) |init_node| {
                is_tensor = expressionType(tree, init_node, tensor_names.items, named_types.items, field_types) == .tensor;
            }
        }
        if (is_tensor) try appendUniqueName(allocator, &tensor_names, tree.tokenSlice(name_token));
    }

    for (0..tree.nodes.len) |node_i| {
        const call_node: Ast.Node.Index = @enumFromInt(node_i);
        const call_first = tree.firstToken(call_node);
        const call_start = tree.tokenStart(call_first);
        const call_last = tree.lastToken(call_node);
        const call_end = tree.tokenStart(call_last) + @as(u32, @intCast(tree.tokenSlice(call_last).len));
        if (call_start <= body_start or call_end >= body_end) continue;

        const call_info = registeredCall(tree, call_node) orelse continue;
        if (expressionType(tree, call_info.receiver, tensor_names.items, named_types.items, field_types) != .tensor) continue;

        const expression_start = if (containsRegisteredCall(tree, call_info.receiver))
            methodSegmentStart(sourceSlice(tree), tree.tokenStart(call_info.method_token))
        else
            call_start;
        const start = positionAt(sourceSlice(tree), expression_start);
        const end = positionAt(sourceSlice(tree), call_end);
        const provenance = positionAt(sourceSlice(tree), tree.tokenStart(call_info.method_token));
        var argument_insert = tree.tokenStart(call_last);
        if (tree.nodeTag(call_node) == .call_comma or tree.nodeTag(call_node) == .call_one_comma) {
            const trailing_comma = call_last - 1;
            if (tree.tokenTag(trailing_comma) != .comma) return error.MalformedTrailingCall;
            argument_insert = tree.tokenStart(trailing_comma);
        }
        try candidates.append(allocator, .{
            .operation = call_info.operation,
            .function = function_name,
            .method_start = tree.tokenStart(call_info.method_token),
            .method_end = tree.tokenStart(call_info.method_token) + @as(u32, @intCast(tree.tokenSlice(call_info.method_token).len)),
            .argument_insert = argument_insert,
            .has_arguments = call_info.argument_count != 0,
            .expression = .{
                .file = logical_path,
                .line = start.line,
                .column = start.column,
                .end_line = end.line,
                .end_column = end.column,
                .start_byte = expression_start,
                .end_byte = call_end,
                .provenance_line = provenance.line,
                .provenance_column = provenance.column,
                .method = call_info.operation.method,
                .instrumented_method = call_info.operation.instrumented_method,
                .function = function_name,
            },
        });
    }
}

fn sourceSlice(tree: *const Ast) []const u8 {
    return tree.source[0..tree.source.len];
}

fn isTensorType(tree: *const Ast, node: Ast.Node.Index) bool {
    return std.mem.eql(u8, tree.getNodeSource(node), "zml.Tensor");
}

fn collectFieldTypes(allocator: Allocator, tree: *const Ast, field_types: *std.ArrayList(FieldType)) !void {
    for (0..tree.nodes.len) |node_i| {
        const node: Ast.Node.Index = @enumFromInt(node_i);
        const declaration = tree.fullVarDecl(node) orelse continue;
        const init_node = declaration.ast.init_node.unwrap() orelse continue;
        var container_buffer: [2]Ast.Node.Index = undefined;
        const container = tree.fullContainerDecl(&container_buffer, init_node) orelse continue;
        const name_token = declaration.ast.mut_token + 1;
        if (tree.tokenTag(name_token) != .identifier) continue;
        const container_name = tree.tokenSlice(name_token);

        for (container.ast.members) |member| {
            const field = tree.fullContainerField(member) orelse continue;
            if (field.ast.tuple_like) continue;
            const type_node = field.ast.type_expr.unwrap() orelse continue;
            try field_types.append(allocator, .{
                .container = container_name,
                .name = tree.tokenSlice(field.ast.main_token),
                .type_name = tree.getNodeSource(type_node),
            });
        }
    }
}

fn expressionType(
    tree: *const Ast,
    node: Ast.Node.Index,
    tensor_names: []const []const u8,
    named_types: []const NamedType,
    field_types: []const FieldType,
) ExpressionType {
    if (tree.nodeTag(node) == .identifier) {
        return if (containsName(tensor_names, tree.tokenSlice(tree.nodeMainToken(node)))) .tensor else .unknown;
    }

    if (tree.nodeTag(node) == .field_access) {
        const receiver, const field_token = tree.nodeData(node).node_and_token;
        const field_name = tree.tokenSlice(field_token);
        if (expressionType(tree, receiver, tensor_names, named_types, field_types) == .arg_max_result and
            (std.mem.eql(u8, field_name, "indices") or std.mem.eql(u8, field_name, "values")))
        {
            return .tensor;
        }
        if (expressionDeclaredType(tree, node, named_types, field_types)) |type_name| {
            if (std.mem.eql(u8, type_name, "zml.Tensor")) return .tensor;
        }
        return .unknown;
    }

    if (registeredCall(tree, node)) |call_info| {
        if (expressionType(tree, call_info.receiver, tensor_names, named_types, field_types) == .tensor) {
            return call_info.operation.result;
        }
        return .unknown;
    }

    const passthrough_receiver = tensorPassthroughReceiver(tree, node) orelse return .unknown;
    return if (expressionType(tree, passthrough_receiver, tensor_names, named_types, field_types) == .tensor) .tensor else .unknown;
}

fn expressionDeclaredType(
    tree: *const Ast,
    node: Ast.Node.Index,
    named_types: []const NamedType,
    field_types: []const FieldType,
) ?[]const u8 {
    if (tree.nodeTag(node) == .identifier) {
        return findNamedType(named_types, tree.tokenSlice(tree.nodeMainToken(node)));
    }
    if (tree.nodeTag(node) != .field_access) return null;

    const receiver, const field_token = tree.nodeData(node).node_and_token;
    const container = expressionDeclaredType(tree, receiver, named_types, field_types) orelse return null;
    const field_name = tree.tokenSlice(field_token);
    for (field_types) |field| {
        if (std.mem.eql(u8, field.container, container) and std.mem.eql(u8, field.name, field_name)) {
            return field.type_name;
        }
    }
    return null;
}

const CallInfo = struct {
    operation: Operation,
    receiver: Ast.Node.Index,
    method_token: Ast.TokenIndex,
    argument_count: usize,
};

fn registeredCall(tree: *const Ast, node: Ast.Node.Index) ?CallInfo {
    var call_buffer: [1]Ast.Node.Index = undefined;
    const call = tree.fullCall(&call_buffer, node) orelse return null;
    if (tree.nodeTag(call.ast.fn_expr) != .field_access) return null;
    const receiver, const method_token = tree.nodeData(call.ast.fn_expr).node_and_token;
    const method = tree.tokenSlice(method_token);
    for (registered_operations) |operation| {
        if (std.mem.eql(u8, method, operation.method)) {
            return .{
                .operation = operation,
                .receiver = receiver,
                .method_token = method_token,
                .argument_count = call.ast.params.len,
            };
        }
    }
    return null;
}

fn tensorPassthroughReceiver(tree: *const Ast, node: Ast.Node.Index) ?Ast.Node.Index {
    var call_buffer: [1]Ast.Node.Index = undefined;
    const call = tree.fullCall(&call_buffer, node) orelse return null;
    if (tree.nodeTag(call.ast.fn_expr) != .field_access) return null;
    const receiver, const method_token = tree.nodeData(call.ast.fn_expr).node_and_token;
    const method = tree.tokenSlice(method_token);
    for (tensor_passthrough_methods) |passthrough| {
        if (std.mem.eql(u8, method, passthrough)) return receiver;
    }
    return null;
}

fn containsRegisteredCall(tree: *const Ast, node: Ast.Node.Index) bool {
    if (registeredCall(tree, node) != null) return true;
    if (tree.nodeTag(node) == .field_access) {
        const receiver, _ = tree.nodeData(node).node_and_token;
        return containsRegisteredCall(tree, receiver);
    }
    if (tensorPassthroughReceiver(tree, node)) |receiver| return containsRegisteredCall(tree, receiver);
    return false;
}

fn methodSegmentStart(source: []const u8, method_start: u32) u32 {
    var cursor: usize = method_start;
    while (cursor > 0 and std.ascii.isWhitespace(source[cursor - 1])) cursor -= 1;
    return if (cursor > 0 and source[cursor - 1] == '.') @intCast(cursor - 1) else method_start;
}

fn containsName(names: []const []const u8, wanted: []const u8) bool {
    for (names) |name| if (std.mem.eql(u8, name, wanted)) return true;
    return false;
}

fn appendUniqueName(allocator: Allocator, names: *std.ArrayList([]const u8), name: []const u8) !void {
    if (!containsName(names.items, name)) try names.append(allocator, name);
}

fn appendNamedType(
    allocator: Allocator,
    named_types: *std.ArrayList(NamedType),
    name: []const u8,
    type_name: []const u8,
) !void {
    try named_types.append(allocator, .{ .name = name, .type_name = type_name });
}

fn findNamedType(named_types: []const NamedType, name: []const u8) ?[]const u8 {
    var i = named_types.len;
    while (i > 0) {
        i -= 1;
        if (std.mem.eql(u8, named_types[i].name, name)) return named_types[i].type_name;
    }
    return null;
}

fn positionAt(source: []const u8, offset: u32) Position {
    var position: Position = .{ .line = 1, .column = 1 };
    for (source[0..offset]) |byte| {
        if (byte == '\n') {
            position.line += 1;
            position.column = 1;
        } else {
            position.column += 1;
        }
    }
    return position;
}

const CliArgs = struct {
    input: []u8,
    output: []u8,
    source_map: []u8,
    logical_path: []u8,

    fn deinit(self: *CliArgs, allocator: Allocator) void {
        allocator.free(self.input);
        allocator.free(self.output);
        allocator.free(self.source_map);
        allocator.free(self.logical_path);
        self.* = undefined;
    }
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    var args = parseArgs(allocator, init.minimal.args) catch |err| {
        std.log.err("invalid arguments ({s}); expected --input PATH --output PATH --source-map PATH --logical-path PATH", .{@errorName(err)});
        return err;
    };
    defer args.deinit(allocator);

    const source_bytes = try std.Io.Dir.cwd().readFileAlloc(io, args.input, allocator, .unlimited);
    defer allocator.free(source_bytes);
    const source = try allocator.dupeZ(u8, source_bytes);
    defer allocator.free(source);

    const source_map_basename = std.Io.Dir.path.basename(args.source_map);
    var result = try transform(allocator, source, args.logical_path, source_map_basename);
    defer result.deinit(allocator);

    try writeFile(io, args.output, result.generated_source);
    try writeFile(io, args.source_map, result.source_map_json);
}

fn parseArgs(allocator: Allocator, process_args: std.process.Args) !CliArgs {
    var iterator = try std.process.Args.Iterator.initAllocator(process_args, allocator);
    defer iterator.deinit();
    _ = iterator.next();

    var input: ?[]u8 = null;
    errdefer if (input) |value| allocator.free(value);
    var output: ?[]u8 = null;
    errdefer if (output) |value| allocator.free(value);
    var source_map: ?[]u8 = null;
    errdefer if (source_map) |value| allocator.free(value);
    var logical_path: ?[]u8 = null;
    errdefer if (logical_path) |value| allocator.free(value);

    while (iterator.next()) |argument_z| {
        const argument: []const u8 = argument_z;
        if (try parseArgument(allocator, &iterator, argument, "--input", &input)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--output", &output)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--source-map", &source_map)) continue;
        if (try parseArgument(allocator, &iterator, argument, "--logical-path", &logical_path)) continue;
        return error.UnknownArgument;
    }

    return .{
        .input = input orelse return error.MissingInput,
        .output = output orelse return error.MissingOutput,
        .source_map = source_map orelse return error.MissingSourceMap,
        .logical_path = logical_path orelse return error.MissingLogicalPath,
    };
}

fn parseArgument(
    allocator: Allocator,
    iterator: *std.process.Args.Iterator,
    argument: []const u8,
    name: []const u8,
    destination: *?[]u8,
) !bool {
    if (std.mem.eql(u8, argument, name)) {
        if (destination.* != null) return error.DuplicateArgument;
        const value = iterator.next() orelse return error.MissingArgumentValue;
        destination.* = try allocator.dupe(u8, value);
        return true;
    }
    if (std.mem.startsWith(u8, argument, name) and argument.len > name.len and argument[name.len] == '=') {
        if (destination.* != null) return error.DuplicateArgument;
        destination.* = try allocator.dupe(u8, argument[name.len + 1 ..]);
        return true;
    }
    return false;
}

fn writeFile(io: std.Io, path: []const u8, contents: []const u8) !void {
    const file = try std.Io.Dir.createFile(.cwd(), io, path, .{});
    defer file.close(io);
    try file.writePositionalAll(io, contents, 0);
}

test "instruments tensor calls and carries tensor flow through const results" {
    const source: [:0]const u8 =
        \\const zml = @import("zml");
        \\
        \\pub fn forward(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
        \\    const sum = x.add(y);
        \\    const doubled = sum.mulConstant(2);
        \\    return doubled;
        \\}
    ;
    var result = try transform(std.testing.allocator, source, "models/forward.zig", "forward.source-map.json");
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 2), result.expressions.len);
    try std.testing.expectEqualStrings("x.add(y)", source[result.expressions[0].start_byte..result.expressions[0].end_byte]);
    try std.testing.expectEqualStrings("sum.mulConstant(2)", source[result.expressions[1].start_byte..result.expressions[1].end_byte]);
    try std.testing.expectEqualStrings("models/forward.zig", result.expressions[0].file);
    try std.testing.expectEqual(@as(u32, 4), result.expressions[0].line);
    try std.testing.expectEqual(@as(u32, 17), result.expressions[0].column);
    try std.testing.expectEqual(@as(u32, 25), result.expressions[0].end_column);
    try std.testing.expectEqual(@as(u32, 4), result.expressions[0].provenance_line);
    try std.testing.expectEqual(@as(u32, 19), result.expressions[0].provenance_column);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "x.addAt(y, .{ .module = @src().module") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "sum.mulConstantAt(2, .{ .module = @src().module") != null);
    try std.testing.expect(std.mem.endsWith(u8, result.generated_source, "pub const zml_source_map_json = @embedFile(\"forward.source-map.json\");\n"));

    const parsed_source_map = try std.json.parseFromSlice(SourceMap, std.testing.allocator, result.source_map_json, .{});
    defer parsed_source_map.deinit();
    try std.testing.expectEqual(@as(u32, 2), parsed_source_map.value.version);
    try std.testing.expectEqualStrings("models/forward.zig", parsed_source_map.value.original_file);
    try std.testing.expectEqual(@as(usize, 2), parsed_source_map.value.expressions.len);
    try std.testing.expectEqualStrings("models/forward.zig", parsed_source_map.value.expressions[0].file);
    try std.testing.expectEqual(@as(u32, 4), parsed_source_map.value.expressions[0].line);
    try std.testing.expectEqual(@as(u32, 17), parsed_source_map.value.expressions[0].column);
    try std.testing.expectEqual(@as(u32, 19), parsed_source_map.value.expressions[0].provenance_column);

    const generated_z = try std.testing.allocator.dupeZ(u8, result.generated_source);
    defer std.testing.allocator.free(generated_z);
    var generated_tree = try Ast.parse(std.testing.allocator, generated_z, .zig);
    defer generated_tree.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), generated_tree.errors.len);
}

test "instruments MNIST tensor fields, chains, passthroughs, and argMax projections" {
    const source: [:0]const u8 =
        \\const zml = @import("zml");
        \\
        \\const Layer = struct {
        \\    weight: zml.Tensor,
        \\    bias: zml.Tensor,
        \\
        \\    fn forward(self: Layer, input: zml.Tensor) zml.Tensor {
        \\        return self.weight.dot(input, .d).add(self.bias).relu().withTags(.{.d});
        \\    }
        \\};
        \\
        \\fn classify(input: zml.Tensor) zml.Tensor {
        \\    const x = input.flatten().convert(.f32).withTags(.{.d});
        \\    return x.argMax(0).indices.convert(.u8);
        \\}
    ;
    var result = try transform(std.testing.allocator, source, "examples/mnist/mnist.zig", "mnist.source-map.json");
    defer result.deinit(std.testing.allocator);

    const expected = [_]struct { method: []const u8, expression: []const u8 }{
        .{ .method = "dot", .expression = "self.weight.dot(input, .d)" },
        .{ .method = "add", .expression = ".add(self.bias)" },
        .{ .method = "relu", .expression = ".relu()" },
        .{ .method = "flatten", .expression = "input.flatten()" },
        .{ .method = "convert", .expression = ".convert(.f32)" },
        .{ .method = "argMax", .expression = "x.argMax(0)" },
        .{ .method = "convert", .expression = ".convert(.u8)" },
    };
    try std.testing.expectEqual(expected.len, result.expressions.len);
    for (expected, result.expressions) |wanted, expression| {
        try std.testing.expectEqualStrings(wanted.method, expression.method);
        try std.testing.expectEqualStrings(wanted.expression, source[expression.start_byte..expression.end_byte]);
    }

    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, ".dotAt(input, .d, .{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, ".addAt(self.bias, .{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, ".reluAt(.{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "input.flattenAt(.{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, ".convertAt(.f32, .{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "x.argMaxAt(0, .{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, ".indices.convertAt(.u8, .{") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "withTagsAt") == null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "At(,") == null);

    try std.testing.expect(result.expressions[0].provenance_column != result.expressions[1].provenance_column);
    try std.testing.expect(result.expressions[1].provenance_column != result.expressions[2].provenance_column);

    const generated_z = try std.testing.allocator.dupeZ(u8, result.generated_source);
    defer std.testing.allocator.free(generated_z);
    var generated_tree = try Ast.parse(std.testing.allocator, generated_z, .zig);
    defer generated_tree.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), generated_tree.errors.len);
}

test "does not instrument a same-named method on a non-tensor receiver" {
    const source: [:0]const u8 =
        \\const Fake = struct {};
        \\
        \\fn untouched(value: Fake) Fake {
        \\    return value.add(value);
        \\}
    ;
    var result = try transform(std.testing.allocator, source, "fake.zig", "fake.source-map.json");
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), result.expressions.len);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "value.add(value)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "value.addAt(value") == null);
}

test "resolves tensor fields through their owning container" {
    const source: [:0]const u8 =
        \\const zml = @import("zml");
        \\
        \\const FakeNumber = struct {
        \\    fn add(self: FakeNumber, other: FakeNumber) FakeNumber {
        \\        _ = other;
        \\        return self;
        \\    }
        \\};
        \\const TensorOwner = struct { weight: zml.Tensor };
        \\const FakeOwner = struct { weight: FakeNumber };
        \\
        \\fn untouched(value: FakeOwner) FakeNumber {
        \\    return value.weight.add(value.weight);
        \\}
    ;
    var result = try transform(std.testing.allocator, source, "owners.zig", "owners.source-map.json");
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 0), result.expressions.len);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "value.weight.add(value.weight)") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "value.weight.addAt") == null);
}

test "preserves multiline calls and trailing comments" {
    const source: [:0]const u8 =
        \\const zml = @import("zml");
        \\
        \\fn forward(x: zml.Tensor, y: zml.Tensor) zml.Tensor {
        \\    const sum = x.add(
        \\        y, // rhs
        \\    );
        \\    return sum;
        \\}
    ;
    var result = try transform(std.testing.allocator, source, "multi.zig", "multi.source-map.json");
    defer result.deinit(std.testing.allocator);

    try std.testing.expectEqual(@as(usize, 1), result.expressions.len);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, "y, .{ .module = @src().module") != null);
    try std.testing.expect(std.mem.indexOf(u8, result.generated_source, ", // rhs\n    );") != null);
    try std.testing.expectEqualStrings("x.add(\n        y, // rhs\n    )", source[result.expressions[0].start_byte..result.expressions[0].end_byte]);
}

test "source location literal has the expected builtin type" {
    const location: std.builtin.SourceLocation = .{
        .module = @src().module,
        .file = "models/forward.zig",
        .fn_name = "forward",
        .line = 4,
        .column = 17,
    };
    try std.testing.expectEqualStrings("models/forward.zig", location.file);
    try std.testing.expectEqual(@as(u32, 4), location.line);
    try std.testing.expectEqual(@as(u32, 17), location.column);
}
