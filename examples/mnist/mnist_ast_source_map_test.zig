const std = @import("std");
const mnist = @import("mnist_ast_model");

const Expression = struct {
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

const SourceMap = struct {
    version: u32,
    original_file: []const u8,
    expressions: []const Expression,
};

fn sourceOffset(source: []const u8, wanted_line: u32, wanted_column: u32) ?usize {
    var line: u32 = 1;
    var column: u32 = 1;
    for (source, 0..) |byte, offset| {
        if (line == wanted_line and column == wanted_column) return offset;
        if (byte == '\n') {
            line += 1;
            column = 1;
        } else {
            column += 1;
        }
    }
    return if (line == wanted_line and column == wanted_column) source.len else null;
}

test "MNIST source map covers every lowering expression" {
    const parsed = try std.json.parseFromSlice(
        SourceMap,
        std.testing.allocator,
        mnist.zml_source_map_json,
        .{},
    );
    defer parsed.deinit();

    try std.testing.expectEqual(@as(u32, 2), parsed.value.version);
    try std.testing.expectEqualStrings("examples/mnist/mnist.zig", parsed.value.original_file);

    const expected = [_]struct {
        method: []const u8,
        instrumented_method: []const u8,
        source: []const u8,
    }{
        .{ .method = "dot", .instrumented_method = "dotAt", .source = "self.weight.dot(input, .d)" },
        .{ .method = "add", .instrumented_method = "addAt", .source = ".add(self.bias)" },
        .{ .method = "relu", .instrumented_method = "reluAt", .source = ".relu()" },
        .{ .method = "flatten", .instrumented_method = "flattenAt", .source = "input.flatten()" },
        .{ .method = "convert", .instrumented_method = "convertAt", .source = ".convert(.f32)" },
        .{ .method = "argMax", .instrumented_method = "argMaxAt", .source = "x.argMax(0)" },
        .{ .method = "convert", .instrumented_method = "convertAt", .source = ".convert(.u8)" },
    };
    try std.testing.expectEqual(expected.len, parsed.value.expressions.len);

    const original_source = @embedFile("mnist.zig");
    for (expected, parsed.value.expressions) |wanted, expression| {
        try std.testing.expectEqualStrings(wanted.method, expression.method);
        try std.testing.expectEqualStrings(wanted.instrumented_method, expression.instrumented_method);
        try std.testing.expectEqualStrings(wanted.source, original_source[expression.start_byte..expression.end_byte]);
        const provenance_offset = sourceOffset(original_source, expression.provenance_line, expression.provenance_column) orelse return error.InvalidProvenancePosition;
        try std.testing.expectEqualStrings(wanted.method, original_source[provenance_offset .. provenance_offset + wanted.method.len]);
        try std.testing.expectEqualStrings("forward", expression.function);
        try std.testing.expect(expression.start_byte < expression.end_byte);
        try std.testing.expect(expression.provenance_line >= expression.line);
    }

    for (parsed.value.expressions, 0..) |expression, i| {
        for (parsed.value.expressions[i + 1 ..]) |other| {
            if (expression.provenance_line == other.provenance_line) {
                try std.testing.expect(expression.provenance_column != other.provenance_column);
            }
        }
    }
}
