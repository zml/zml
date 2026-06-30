const std = @import("std");

pub const bos_token = "<｜begin▁of▁sentence｜>";
pub const eos_token = "<｜end▁of▁sentence｜>";
pub const thinking_start_token = "<think>";
pub const thinking_end_token = "</think>";
pub const dsml_token = "｜DSML｜";

pub const user_token = "<｜User｜>";
pub const assistant_token = "<｜Assistant｜>";
pub const latest_reminder_token = "<｜latest_reminder｜>";

pub const ThinkingMode = enum {
    chat,
    thinking,
};

pub const ReasoningEffort = enum {
    high,
    max,
};

pub const Role = enum {
    system,
    user,
    assistant,
    tool,
    latest_reminder,
    developer,
    direct_search_results,
};

pub const Task = enum {
    action,
    query,
    authority,
    domain,
    title,
    read_url,

    fn token(self: Task) []const u8 {
        return switch (self) {
            .action => "<｜action｜>",
            .query => "<｜query｜>",
            .authority => "<｜authority｜>",
            .domain => "<｜domain｜>",
            .title => "<｜title｜>",
            .read_url => "<｜read_url｜>",
        };
    }
};

pub const Tool = struct {
    type: ?[]const u8 = null,
    function: std.json.Value,
};

pub const FunctionCall = struct {
    name: []const u8,
    arguments: []const u8,
    id: ?[]const u8 = null,
};

pub const ToolCall = struct {
    id: ?[]const u8 = null,
    type: ?[]const u8 = null,
    function: FunctionCall,
};

pub const ContentBlock = struct {
    type: []const u8,
    text: ?[]const u8 = null,
    tool_use_id: ?[]const u8 = null,
    content: ?std.json.Value = null,
};

pub const Message = struct {
    role: Role,
    content: ?std.json.Value = null,
    content_blocks: ?[]const ContentBlock = null,
    tools: ?[]const Tool = null,
    response_format: ?std.json.Value = null,
    tool_calls: ?[]const ToolCall = null,
    reasoning_content: ?[]const u8 = null,
    tool_call_id: ?[]const u8 = null,
    task: ?Task = null,
    wo_eos: bool = false,
    mask: ?std.json.Value = null,
};

pub const EncodeOptions = struct {
    thinking_mode: ThinkingMode,
    context: []const Message = &.{},
    drop_thinking: bool = true,
    add_default_bos_token: bool = true,
    reasoning_effort: ?ReasoningEffort = null,
};

pub const ParsedMessage = struct {
    role: Role = .assistant,
    content: []u8,
    reasoning_content: []u8,
    tool_calls: []ParsedToolCall,
    allocator: std.mem.Allocator,

    pub const ParsedFunctionCall = struct {
        name: []u8,
        arguments: []u8,
    };

    pub const ParsedToolCall = struct {
        type: []const u8 = "function",
        function: ParsedFunctionCall,
    };

    pub fn deinit(self: *ParsedMessage) void {
        self.allocator.free(self.content);
        self.allocator.free(self.reasoning_content);
        for (self.tool_calls) |tool_call| {
            self.allocator.free(tool_call.function.name);
            self.allocator.free(tool_call.function.arguments);
        }
        self.allocator.free(self.tool_calls);
        self.* = undefined;
    }
};

const reasoning_effort_max =
    "Reasoning Effort: Absolute maximum with no shortcuts permitted.\n" ++
    "You MUST be very thorough in your thinking and comprehensively decompose the problem to resolve the root cause, rigorously stress-testing your logic against all potential paths, edge cases, and adversarial scenarios.\n" ++
    "Explicitly write out your entire deliberation process, documenting every intermediate step, considered alternative, and rejected hypothesis to ensure absolutely no assumption is left unchecked.\n\n";

const tools_prefix =
    "## Tools\n\n" ++
    "You have access to a set of tools to help answer the user's question. You can invoke tools by writing a \"<｜DSML｜tool_calls>\" block like the following:\n\n" ++
    "<｜DSML｜tool_calls>\n" ++
    "<｜DSML｜invoke name=\"$TOOL_NAME\">\n" ++
    "<｜DSML｜parameter name=\"$PARAMETER_NAME\" string=\"true|false\">$PARAMETER_VALUE</｜DSML｜parameter>\n" ++
    "...\n" ++
    "</｜DSML｜invoke>\n" ++
    "<｜DSML｜invoke name=\"$TOOL_NAME2\">\n" ++
    "...\n" ++
    "</｜DSML｜invoke>\n" ++
    "</｜DSML｜tool_calls>\n\n" ++
    "String parameters should be specified as is and set `string=\"true\"`. For all other types (numbers, booleans, arrays, objects), pass the value in JSON format and set `string=\"false\"`.\n\n" ++
    "If thinking_mode is enabled (triggered by <think>), you MUST output your complete reasoning inside <think>...</think> BEFORE any tool calls or final response.\n\n" ++
    "Otherwise, output directly after </think> with tool calls or final response.\n\n" ++
    "### Available Tool Schemas\n\n";

const tools_suffix =
    "\n\nYou MUST strictly follow the above defined tool name and parameter schemas to invoke tool calls.\n";

const NormalizedMessage = struct {
    message: Message,
    blocks: ?std.ArrayList(ContentBlock) = null,
    drop_reasoning: bool = false,
};

pub fn encodeMessages(
    allocator: std.mem.Allocator,
    messages: []const Message,
    options: EncodeOptions,
) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const scratch = arena.allocator();

    const context = try normalizeMessages(scratch, options.context);
    sortToolResults(context.items, &.{});

    const normalized = try normalizeMessages(scratch, messages);
    const preceding_calls = lastToolCalls(context.items);
    sortToolResults(normalized.items, preceding_calls);

    var full: std.ArrayList(NormalizedMessage) = .empty;
    try full.appendSlice(scratch, context.items);
    try full.appendSlice(scratch, normalized.items);

    var effective_drop_thinking = options.drop_thinking;
    for (full.items) |msg| {
        if (msg.message.tools) |tools| {
            if (tools.len != 0) {
                effective_drop_thinking = false;
                break;
            }
        }
    }

    var render_messages = full.items;
    var context_len = context.items.len;
    if (options.thinking_mode == .thinking and effective_drop_thinking) {
        render_messages = try dropThinkingMessages(scratch, full.items);
        const dropped_context = try dropThinkingMessages(scratch, context.items);
        context_len = dropped_context.len;
    }

    var out: std.Io.Writer.Allocating = .init(allocator);
    errdefer out.deinit();

    if (options.add_default_bos_token and options.context.len == 0) {
        try out.writer.writeAll(bos_token);
    }

    for (context_len..render_messages.len) |index| {
        try renderMessage(
            &out.writer,
            index,
            render_messages,
            options.thinking_mode,
            effective_drop_thinking,
            options.reasoning_effort,
            scratch,
        );
    }

    return out.toOwnedSlice();
}

fn normalizeMessages(allocator: std.mem.Allocator, messages: []const Message) !std.ArrayList(NormalizedMessage) {
    var merged: std.ArrayList(NormalizedMessage) = .empty;

    for (messages) |message| {
        switch (message.role) {
            .tool => {
                const block: ContentBlock = .{
                    .type = "tool_result",
                    .tool_use_id = message.tool_call_id orelse "",
                    .content = message.content orelse .{ .string = "" },
                };
                if (merged.items.len != 0 and
                    merged.items[merged.items.len - 1].message.role == .user and
                    merged.items[merged.items.len - 1].blocks != null)
                {
                    try merged.items[merged.items.len - 1].blocks.?.append(allocator, block);
                } else {
                    var blocks: std.ArrayList(ContentBlock) = .empty;
                    try blocks.append(allocator, block);
                    try merged.append(allocator, .{
                        .message = .{ .role = .user },
                        .blocks = blocks,
                    });
                }
            },
            .user => {
                const text = try optionalString(message.content, "");
                const block: ContentBlock = .{ .type = "text", .text = text };
                if (merged.items.len != 0 and
                    merged.items[merged.items.len - 1].message.role == .user and
                    merged.items[merged.items.len - 1].blocks != null and
                    merged.items[merged.items.len - 1].message.task == null)
                {
                    try merged.items[merged.items.len - 1].blocks.?.append(allocator, block);
                } else {
                    var blocks: std.ArrayList(ContentBlock) = .empty;
                    try blocks.append(allocator, block);
                    var copy = message;
                    copy.content_blocks = null;
                    try merged.append(allocator, .{ .message = copy, .blocks = blocks });
                }
            },
            else => try merged.append(allocator, .{ .message = message }),
        }
    }

    return merged;
}

fn lastToolCalls(messages: []const NormalizedMessage) []const ToolCall {
    var result: []const ToolCall = &.{};
    for (messages) |msg| {
        if (msg.message.role == .assistant) {
            if (msg.message.tool_calls) |calls| {
                if (calls.len != 0) result = calls;
            }
        }
    }
    return result;
}

fn sortToolResults(messages: []NormalizedMessage, initial_calls: []const ToolCall) void {
    var calls = initial_calls;
    for (messages) |*msg| {
        if (msg.message.role == .assistant) {
            if (msg.message.tool_calls) |tool_calls| {
                if (tool_calls.len != 0) calls = tool_calls;
            }
            continue;
        }
        if (msg.message.role != .user or msg.blocks == null or calls.len == 0) continue;

        const blocks = &msg.blocks.?;
        var tool_count: usize = 0;
        for (blocks.items) |block| {
            if (std.mem.eql(u8, block.type, "tool_result")) tool_count += 1;
        }
        if (tool_count < 2) continue;

        for (blocks.items, 0..) |block, index| {
            if (!std.mem.eql(u8, block.type, "tool_result")) continue;
            var best_index = index;
            var candidate_index = index + 1;
            while (candidate_index < blocks.items.len) : (candidate_index += 1) {
                if (!std.mem.eql(u8, blocks.items[candidate_index].type, "tool_result")) continue;
                const best_rank = toolCallRank(calls, blocks.items[best_index].tool_use_id orelse "");
                const candidate_rank = toolCallRank(calls, blocks.items[candidate_index].tool_use_id orelse "");
                if (candidate_rank < best_rank) best_index = candidate_index;
            }
            if (best_index != index) {
                const best = blocks.items[best_index];
                var destination = best_index;
                while (destination != index) {
                    var previous = destination;
                    while (previous > index) {
                        previous -= 1;
                        if (std.mem.eql(u8, blocks.items[previous].type, "tool_result")) break;
                    }
                    blocks.items[destination] = blocks.items[previous];
                    destination = previous;
                }
                blocks.items[index] = best;
            }
        }
    }
}

fn toolCallRank(calls: []const ToolCall, id: []const u8) usize {
    for (calls, 0..) |call, index| {
        const call_id = call.id orelse call.function.id orelse "";
        if (std.mem.eql(u8, call_id, id)) return index;
    }
    return 0;
}

fn dropThinkingMessages(allocator: std.mem.Allocator, messages: []const NormalizedMessage) ![]NormalizedMessage {
    const last_user_index = findLastUserIndex(messages);
    var result: std.ArrayList(NormalizedMessage) = .empty;

    for (messages, 0..) |message, index| {
        const keep = switch (message.message.role) {
            .user, .system, .tool, .latest_reminder, .direct_search_results => true,
            else => last_user_index == null or index >= last_user_index.?,
        };
        if (keep) {
            try result.append(allocator, message);
        } else if (message.message.role == .assistant) {
            var copy = message;
            copy.drop_reasoning = true;
            try result.append(allocator, copy);
        }
    }

    return result.toOwnedSlice(allocator);
}

fn findLastUserIndex(messages: []const NormalizedMessage) ?usize {
    var index = messages.len;
    while (index > 0) {
        index -= 1;
        switch (messages[index].message.role) {
            .user, .developer => return index,
            else => {},
        }
    }
    return null;
}

fn renderMessage(
    writer: *std.Io.Writer,
    index: usize,
    messages: []const NormalizedMessage,
    thinking_mode: ThinkingMode,
    drop_thinking: bool,
    reasoning_effort: ?ReasoningEffort,
    allocator: std.mem.Allocator,
) !void {
    const normalized = messages[index];
    const message = normalized.message;
    const last_user_index = findLastUserIndex(messages);

    if (index == 0 and thinking_mode == .thinking and reasoning_effort == .max) {
        try writer.writeAll(reasoning_effort_max);
    }

    switch (message.role) {
        .system => {
            try writer.writeAll(try optionalString(message.content, ""));
            if (message.tools) |tools| {
                if (tools.len != 0) {
                    try writer.writeAll("\n\n");
                    try renderTools(writer, tools);
                }
            }
            if (message.response_format) |response_format| {
                try writer.writeAll("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                try writePythonJson(writer, response_format);
            }
        },
        .developer => {
            const content = try requiredString(message.content);
            try writer.writeAll(user_token);
            try writer.writeAll(content);
            if (message.tools) |tools| {
                if (tools.len != 0) {
                    try writer.writeAll("\n\n");
                    try renderTools(writer, tools);
                }
            }
            if (message.response_format) |response_format| {
                try writer.writeAll("\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n");
                try writePythonJson(writer, response_format);
            }
        },
        .user => {
            try writer.writeAll(user_token);
            if (normalized.blocks) |blocks| {
                for (blocks.items, 0..) |block, block_index| {
                    if (block_index != 0) try writer.writeAll("\n\n");
                    try renderContentBlock(writer, block);
                }
            } else {
                try writer.writeAll(try optionalString(message.content, ""));
            }
        },
        .latest_reminder => {
            try writer.writeAll(latest_reminder_token);
            try writer.writeAll(try requiredString(message.content));
        },
        .tool => return error.UnmergedToolMessage,
        .assistant => {
            const previous_has_task = index > 0 and messages[index - 1].message.task != null;
            if (thinking_mode == .thinking and !previous_has_task) {
                if (!drop_thinking or last_user_index == null or index > last_user_index.?) {
                    if (!normalized.drop_reasoning) {
                        try writer.writeAll(message.reasoning_content orelse "");
                    }
                    try writer.writeAll(thinking_end_token);
                }
            }

            try writer.writeAll(try optionalString(message.content, ""));
            if (message.tool_calls) |tool_calls| {
                if (tool_calls.len != 0) {
                    try writer.writeAll("\n\n<｜DSML｜tool_calls>\n");
                    for (tool_calls, 0..) |tool_call, tool_index| {
                        if (tool_index != 0) try writer.writeByte('\n');
                        try renderToolCall(writer, tool_call, allocator);
                    }
                    try writer.writeAll("\n</｜DSML｜tool_calls>");
                }
            }
            if (!message.wo_eos) try writer.writeAll(eos_token);
        },
        .direct_search_results => return error.UnsupportedRole,
    }

    if (index + 1 < messages.len) {
        switch (messages[index + 1].message.role) {
            .assistant, .latest_reminder => {},
            else => return,
        }
    }

    if (message.task) |task| {
        if (task != .action) {
            try writer.writeAll(task.token());
        } else {
            try writer.writeAll(assistant_token);
            try writer.writeAll(if (thinking_mode == .thinking) thinking_start_token else thinking_end_token);
            try writer.writeAll(task.token());
        }
    } else switch (message.role) {
        .user, .developer => {
            try writer.writeAll(assistant_token);
            if (!drop_thinking and thinking_mode == .thinking) {
                try writer.writeAll(thinking_start_token);
            } else if (drop_thinking and thinking_mode == .thinking and (last_user_index == null or index >= last_user_index.?)) {
                try writer.writeAll(thinking_start_token);
            } else {
                try writer.writeAll(thinking_end_token);
            }
        },
        else => {},
    }
}

fn renderTools(writer: *std.Io.Writer, tools: []const Tool) !void {
    try writer.writeAll(tools_prefix);
    for (tools, 0..) |tool, index| {
        if (index != 0) try writer.writeByte('\n');
        try writePythonJson(writer, tool.function);
    }
    try writer.writeAll(tools_suffix);
}

fn renderContentBlock(writer: *std.Io.Writer, block: ContentBlock) !void {
    if (std.mem.eql(u8, block.type, "text")) {
        try writer.writeAll(block.text orelse "");
    } else if (std.mem.eql(u8, block.type, "tool_result")) {
        try writer.writeAll("<tool_result>");
        if (block.content) |content| switch (content) {
            .string => |text| try writer.writeAll(text),
            .array => |items| {
                var wrote_item = false;
                for (items.items) |item| {
                    if (wrote_item) try writer.writeAll("\n\n");
                    wrote_item = true;
                    if (item == .object) {
                        const block_type = item.object.get("type");
                        if (block_type != null and block_type.? == .string and std.mem.eql(u8, block_type.?.string, "text")) {
                            const text = item.object.get("text");
                            if (text != null and text.? == .string) try writer.writeAll(text.?.string);
                        } else {
                            try writer.writeAll("[Unsupported ");
                            if (block_type != null and block_type.? == .string) try writer.writeAll(block_type.?.string) else try writer.writeAll("None");
                            try writer.writeByte(']');
                        }
                    } else {
                        try writer.writeAll("[Unsupported None]");
                    }
                }
            },
            else => return error.InvalidContent,
        };
        try writer.writeAll("</tool_result>");
    } else {
        try writer.writeAll("[Unsupported ");
        try writer.writeAll(block.type);
        try writer.writeByte(']');
    }
}

fn renderToolCall(writer: *std.Io.Writer, tool_call: ToolCall, allocator: std.mem.Allocator) !void {
    try writer.writeAll("<｜DSML｜invoke name=\"");
    try writer.writeAll(tool_call.function.name);
    try writer.writeAll("\">\n");

    var parsed = std.json.parseFromSlice(std.json.Value, allocator, tool_call.function.arguments, .{}) catch null;
    defer if (parsed) |*value| value.deinit();

    if (parsed) |value| {
        if (value.value != .object) return error.InvalidToolArguments;
        try renderArguments(writer, value.value.object);
    } else {
        try writer.writeAll("<｜DSML｜parameter name=\"arguments\" string=\"true\">");
        try writer.writeAll(tool_call.function.arguments);
        try writer.writeAll("</｜DSML｜parameter>");
    }

    try writer.writeAll("\n</｜DSML｜invoke>");
}

fn renderArguments(writer: *std.Io.Writer, arguments: std.json.ObjectMap) !void {
    var iterator = arguments.iterator();
    var index: usize = 0;
    while (iterator.next()) |entry| : (index += 1) {
        if (index != 0) try writer.writeByte('\n');
        try writer.writeAll("<｜DSML｜parameter name=\"");
        try writer.writeAll(entry.key_ptr.*);
        try writer.writeAll("\" string=\"");
        if (entry.value_ptr.* == .string) {
            try writer.writeAll("true\">");
            try writer.writeAll(entry.value_ptr.string);
        } else {
            try writer.writeAll("false\">");
            try writePythonJson(writer, entry.value_ptr.*);
        }
        try writer.writeAll("</｜DSML｜parameter>");
    }
}

fn writePythonJson(writer: *std.Io.Writer, value: std.json.Value) !void {
    switch (value) {
        .null => try writer.writeAll("null"),
        .bool => |boolean| try writer.writeAll(if (boolean) "true" else "false"),
        .integer => |integer| try writer.print("{d}", .{integer}),
        .float => |float| try writer.print("{d}", .{float}),
        .number_string => |number| try writer.writeAll(number),
        .string => |string| try std.json.Stringify.encodeJsonString(string, .{}, writer),
        .array => |array| {
            try writer.writeByte('[');
            for (array.items, 0..) |item, index| {
                if (index != 0) try writer.writeAll(", ");
                try writePythonJson(writer, item);
            }
            try writer.writeByte(']');
        },
        .object => |object| {
            try writer.writeByte('{');
            var iterator = object.iterator();
            var index: usize = 0;
            while (iterator.next()) |entry| : (index += 1) {
                if (index != 0) try writer.writeAll(", ");
                try std.json.Stringify.encodeJsonString(entry.key_ptr.*, .{}, writer);
                try writer.writeAll(": ");
                try writePythonJson(writer, entry.value_ptr.*);
            }
            try writer.writeByte('}');
        },
    }
}

fn optionalString(value: ?std.json.Value, default: []const u8) ![]const u8 {
    const actual = value orelse return default;
    return switch (actual) {
        .null => default,
        .string => |string| string,
        else => error.InvalidContent,
    };
}

fn requiredString(value: ?std.json.Value) ![]const u8 {
    const result = try optionalString(value, "");
    if (result.len == 0) return error.InvalidContent;
    return result;
}

const ReadResult = struct {
    index: usize,
    content: []const u8,
    stop: ?[]const u8,
};

fn readUntilStop(index: usize, text: []const u8, stops: []const []const u8) ReadResult {
    var min_position = text.len;
    var matched: ?[]const u8 = null;
    for (stops) |stop| {
        if (std.mem.indexOfPos(u8, text, index, stop)) |position| {
            if (position < min_position) {
                min_position = position;
                matched = stop;
            }
        }
    }
    if (matched) |stop| {
        return .{
            .index = min_position + stop.len,
            .content = text[index..min_position],
            .stop = stop,
        };
    }
    return .{ .index = text.len, .content = text[index..], .stop = null };
}

pub fn parseMessageFromCompletionText(
    allocator: std.mem.Allocator,
    text: []const u8,
    thinking_mode: ThinkingMode,
) !ParsedMessage {
    const tool_calls_start = "\n\n<｜DSML｜tool_calls";
    var index: usize = 0;
    var reasoning: []const u8 = "";

    if (thinking_mode == .thinking) {
        const result = readUntilStop(index, text, &.{ thinking_end_token, tool_calls_start });
        if (result.stop == null or !std.mem.eql(u8, result.stop.?, thinking_end_token)) return error.InvalidThinkingFormat;
        reasoning = result.content;
        index = result.index;
    }

    const summary_result = readUntilStop(index, text, &.{ eos_token, tool_calls_start });
    const content = summary_result.content;
    index = summary_result.index;

    var parsed_calls: std.ArrayList(ParsedMessage.ParsedToolCall) = .empty;
    errdefer {
        for (parsed_calls.items) |call| {
            allocator.free(call.function.name);
            allocator.free(call.function.arguments);
        }
        parsed_calls.deinit(allocator);
    }

    if (summary_result.stop != null and std.mem.eql(u8, summary_result.stop.?, tool_calls_start)) {
        index = try parseToolCalls(allocator, index, text, &parsed_calls);
        const end_result = readUntilStop(index, text, &.{eos_token});
        if (end_result.content.len != 0) return error.UnexpectedContent;
        index = end_result.index;
    } else if (summary_result.stop == null or !std.mem.eql(u8, summary_result.stop.?, eos_token)) {
        return error.MissingEndOfSentence;
    }

    if (index != text.len) return error.UnexpectedContent;
    try rejectSpecialTokens(content);
    try rejectSpecialTokens(reasoning);

    const owned_content = try allocator.dupe(u8, content);
    errdefer allocator.free(owned_content);
    const owned_reasoning = try allocator.dupe(u8, reasoning);
    errdefer allocator.free(owned_reasoning);

    return .{
        .content = owned_content,
        .reasoning_content = owned_reasoning,
        .tool_calls = try parsed_calls.toOwnedSlice(allocator),
        .allocator = allocator,
    };
}

fn parseToolCalls(
    allocator: std.mem.Allocator,
    initial_index: usize,
    text: []const u8,
    calls: *std.ArrayList(ParsedMessage.ParsedToolCall),
) !usize {
    const invoke_start = "<｜DSML｜invoke";
    const invoke_end = "</｜DSML｜invoke";
    const parameter_start = "<｜DSML｜parameter";
    const parameter_end = "/｜DSML｜parameter";
    const tool_calls_end = "</｜DSML｜tool_calls>";

    var index = initial_index;
    while (index < text.len) {
        const next = readUntilStop(index, text, &.{ invoke_start, tool_calls_end });
        if (!std.mem.eql(u8, next.content, ">\n")) return error.InvalidToolCallFormat;
        index = next.index;
        if (next.stop == null) return error.MissingToolCallToken;
        if (std.mem.eql(u8, next.stop.?, tool_calls_end)) return index;

        const name_result = readUntilStop(index, text, &.{ parameter_start, invoke_end });
        const name = try parseToolName(name_result.content);
        index = name_result.index;
        var stop = name_result.stop orelse return error.MissingToolCallToken;

        var arguments: std.Io.Writer.Allocating = .init(allocator);
        errdefer arguments.deinit();
        var parameter_names: std.StringHashMapUnmanaged(void) = .empty;
        defer parameter_names.deinit(allocator);
        try arguments.writer.writeByte('{');
        var argument_index: usize = 0;

        while (std.mem.eql(u8, stop, parameter_start)) {
            const parameter = readUntilStop(index, text, &.{parameter_end});
            if (parameter.stop == null) return error.MissingToolCallToken;
            const parsed_parameter = try parseParameter(parameter.content);
            const name_entry = try parameter_names.getOrPut(allocator, parsed_parameter.name);
            if (name_entry.found_existing) return error.DuplicateParameter;
            if (argument_index != 0) try arguments.writer.writeAll(", ");
            try std.json.Stringify.encodeJsonString(parsed_parameter.name, .{}, &arguments.writer);
            try arguments.writer.writeAll(": ");
            if (parsed_parameter.is_string) {
                try std.json.Stringify.encodeJsonString(parsed_parameter.value, .{}, &arguments.writer);
            } else {
                try arguments.writer.writeAll(parsed_parameter.value);
            }
            argument_index += 1;

            const following = readUntilStop(parameter.index, text, &.{ parameter_start, invoke_end });
            if (!std.mem.eql(u8, following.content, ">\n")) return error.InvalidToolCallFormat;
            index = following.index;
            stop = following.stop orelse return error.MissingToolCallToken;
        }
        try arguments.writer.writeByte('}');

        const owned_arguments = try arguments.toOwnedSlice();
        errdefer allocator.free(owned_arguments);
        const owned_name = try allocator.dupe(u8, name);
        errdefer allocator.free(owned_name);
        try calls.append(allocator, .{
            .function = .{
                .name = owned_name,
                .arguments = owned_arguments,
            },
        });
    }
    return error.MissingToolCallToken;
}

fn parseToolName(content: []const u8) ![]const u8 {
    const trimmed = std.mem.trimStart(u8, content, " \t\r\n");
    const prefix = "name=\"";
    const suffix = "\">\n";
    if (!std.mem.startsWith(u8, trimmed, prefix) or !std.mem.endsWith(u8, trimmed, suffix)) {
        return error.InvalidToolName;
    }
    return trimmed[prefix.len .. trimmed.len - suffix.len];
}

const ParsedParameter = struct {
    name: []const u8,
    value: []const u8,
    is_string: bool,
};

fn parseParameter(content: []const u8) !ParsedParameter {
    const prefix = " name=\"";
    if (!std.mem.startsWith(u8, content, prefix) or !std.mem.endsWith(u8, content, "<")) {
        return error.InvalidParameter;
    }
    const name_end = std.mem.indexOfPos(u8, content, prefix.len, "\" string=\"") orelse return error.InvalidParameter;
    const string_start = name_end + "\" string=\"".len;
    const string_end = std.mem.indexOfPos(u8, content, string_start, "\">") orelse return error.InvalidParameter;
    const flag = content[string_start..string_end];
    const is_string = if (std.mem.eql(u8, flag, "true")) true else if (std.mem.eql(u8, flag, "false")) false else return error.InvalidParameter;
    return .{
        .name = content[prefix.len..name_end],
        .value = content[string_end + 2 .. content.len - 1],
        .is_string = is_string,
    };
}

fn rejectSpecialTokens(content: []const u8) !void {
    for ([_][]const u8{ bos_token, eos_token, thinking_start_token, thinking_end_token, dsml_token }) |token| {
        if (std.mem.indexOf(u8, content, token) != null) return error.UnexpectedSpecialToken;
    }
}
