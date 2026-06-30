const std = @import("std");
const encoding = @import("encoding.zig");

const case_1_input = @embedFile("encoding/tests/test_input_1.json");
const case_1_output = @embedFile("encoding/tests/test_output_1.txt");
const case_2_input = @embedFile("encoding/tests/test_input_2.json");
const case_2_output = @embedFile("encoding/tests/test_output_2.txt");
const case_3_input = @embedFile("encoding/tests/test_input_3.json");
const case_3_output = @embedFile("encoding/tests/test_output_3.txt");
const case_4_input = @embedFile("encoding/tests/test_input_4.json");
const case_4_output = @embedFile("encoding/tests/test_output_4.txt");

const CaseWithTools = struct {
    tools: []const encoding.Tool,
    messages: []encoding.Message,
};

test "thinking with tools encodes and parses like Python" {
    var input = try std.json.parseFromSlice(CaseWithTools, std.testing.allocator, case_1_input, .{
        .ignore_unknown_fields = true,
    });
    defer input.deinit();
    input.value.messages[0].tools = input.value.tools;

    const prompt = try encoding.encodeMessages(std.testing.allocator, input.value.messages, .{
        .thinking_mode = .thinking,
    });
    defer std.testing.allocator.free(prompt);
    try std.testing.expectEqualStrings(case_1_output, prompt);

    const marker = encoding.assistant_token ++ encoding.thinking_start_token;
    const first_marker = std.mem.indexOf(u8, prompt, marker).? + marker.len;
    const first_end = std.mem.indexOfPos(u8, prompt, first_marker, encoding.user_token).?;
    var parsed_tool_call = try encoding.parseMessageFromCompletionText(
        std.testing.allocator,
        prompt[first_marker..first_end],
        .thinking,
    );
    defer parsed_tool_call.deinit();
    try std.testing.expectEqualStrings(
        "The user wants to know the weather in Beijing. I should use the get_weather tool.",
        parsed_tool_call.reasoning_content,
    );
    try std.testing.expectEqualStrings("", parsed_tool_call.content);
    try std.testing.expectEqual(@as(usize, 1), parsed_tool_call.tool_calls.len);
    try std.testing.expectEqualStrings("get_weather", parsed_tool_call.tool_calls[0].function.name);
    try std.testing.expectEqualStrings(
        "{\"location\": \"Beijing\", \"unit\": \"celsius\"}",
        parsed_tool_call.tool_calls[0].function.arguments,
    );

    const last_marker = std.mem.lastIndexOf(u8, prompt, marker).? + marker.len;
    var parsed_final = try encoding.parseMessageFromCompletionText(
        std.testing.allocator,
        prompt[last_marker..],
        .thinking,
    );
    defer parsed_final.deinit();
    try std.testing.expectEqualStrings(
        "Got the weather data. Let me format a nice response.",
        parsed_final.reasoning_content,
    );
    try std.testing.expect(std.mem.indexOf(u8, parsed_final.content, "22°C") != null);
    try std.testing.expectEqual(@as(usize, 0), parsed_final.tool_calls.len);
}

test "thinking without tools drops earlier reasoning like Python" {
    var input = try std.json.parseFromSlice([]encoding.Message, std.testing.allocator, case_2_input, .{
        .ignore_unknown_fields = true,
    });
    defer input.deinit();

    const prompt = try encoding.encodeMessages(std.testing.allocator, input.value, .{
        .thinking_mode = .thinking,
    });
    defer std.testing.allocator.free(prompt);
    try std.testing.expectEqualStrings(case_2_output, prompt);
    try std.testing.expect(std.mem.indexOf(u8, prompt, "The user said hello") == null);

    const marker = encoding.assistant_token ++ encoding.thinking_start_token;
    const last_marker = std.mem.lastIndexOf(u8, prompt, marker).? + marker.len;
    var parsed = try encoding.parseMessageFromCompletionText(
        std.testing.allocator,
        prompt[last_marker..],
        .thinking,
    );
    defer parsed.deinit();
    try std.testing.expectEqualStrings(
        "The user asks about the capital of France. It is Paris.",
        parsed.reasoning_content,
    );
    try std.testing.expectEqualStrings("The capital of France is Paris.", parsed.content);
    try std.testing.expectEqual(@as(usize, 0), parsed.tool_calls.len);
}

test "interleaved thinking and search encodes like Python" {
    var input = try std.json.parseFromSlice([]encoding.Message, std.testing.allocator, case_3_input, .{
        .ignore_unknown_fields = true,
    });
    defer input.deinit();

    const prompt = try encoding.encodeMessages(std.testing.allocator, input.value, .{
        .thinking_mode = .thinking,
    });
    defer std.testing.allocator.free(prompt);
    try std.testing.expectEqualStrings(case_3_output, prompt);
}

test "quick instruction task encodes like Python" {
    var input = try std.json.parseFromSlice([]encoding.Message, std.testing.allocator, case_4_input, .{
        .ignore_unknown_fields = true,
    });
    defer input.deinit();

    const prompt = try encoding.encodeMessages(std.testing.allocator, input.value, .{
        .thinking_mode = .chat,
    });
    defer std.testing.allocator.free(prompt);
    try std.testing.expectEqualStrings(case_4_output, prompt);
}
