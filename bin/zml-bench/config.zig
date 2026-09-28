const std = @import("std");

const Config = @This();

endpoint: []const u8 = "http://127.0.0.1:8000/v1",
model: []const u8 = "zml_model",
prompt: []const u8 = "Write a website in C",
batch: usize = 16,
maxTokens: ?u32 = null,
maxCompletionTokens: ?u32 = null,
temperature: ?f64 = null,
reasoningEffort: []const u8 = "",
systemPrompt: []const u8 = "",
stop: []const u8 = "",
tools: []const u8 = "",
documents: []const u8 = "",

pub const Reasoning = union(enum) {
    name: []const u8,
    budget: u8,

    pub fn jsonStringify(self: Reasoning, writer: *std.json.Stringify) !void {
        switch (self) {
            .name => |name| try writer.write(name),
            .budget => |budget| try writer.write(budget),
        }
    }

    pub fn parse(text: []const u8) !?Reasoning {
        if (text.len == 0 or std.mem.eql(u8, text, "auto")) return null;
        if (std.mem.eql(u8, text, "off")) return .{ .name = "none" };
        if (std.mem.eql(u8, text, "on")) return .{ .name = "high" };
        for ([_][]const u8{ "none", "minimal", "low", "medium", "high", "xhigh", "max" }) |name| {
            if (std.mem.eql(u8, text, name)) return .{ .name = name };
        }
        const budget = std.fmt.parseInt(u8, text, 10) catch return error.InvalidReasoningEffort;
        if (budget > 100) return error.InvalidReasoningEffort;
        return .{ .budget = budget };
    }
};

pub fn validate(self: Config) !void {
    const uri = std.Uri.parse(self.endpoint) catch return error.InvalidEndpoint;
    if ((!std.mem.eql(u8, uri.scheme, "http") and !std.mem.eql(u8, uri.scheme, "https")) or
        uri.host == null or uri.user != null or uri.password != null or uri.query != null or uri.fragment != null)
        return error.InvalidEndpoint;
    if (self.batch == 0 or self.batch > 256) return error.BatchMustBeBetween1And256;
    if (self.endpoint.len > 4096 or self.model.len > 4096 or self.prompt.len > 16 * 1024 or self.systemPrompt.len > 16 * 1024) return error.InputTooLong;
    if (self.model.len == 0) return error.ModelRequired;
    if (self.prompt.len == 0) return error.PromptRequired;
    if (self.maxTokens == 0 or self.maxCompletionTokens == 0) return error.MaxTokensMustBePositive;
    if (self.temperature) |t| {
        if (!std.math.isFinite(t) or t < 0 or t > std.math.floatMax(f32)) return error.TemperatureMustBeFiniteAndNonnegative;
    }
    _ = try Reasoning.parse(self.reasoningEffort);
}

pub fn url(self: Config, allocator: std.mem.Allocator) ![]u8 {
    const base = std.mem.trimEnd(u8, self.endpoint, "/");
    const suffix = if (std.mem.endsWith(u8, base, "/chat/completions")) "" else if (std.mem.endsWith(u8, base, "/v1")) "/chat/completions" else "/v1/chat/completions";
    return std.fmt.allocPrint(allocator, "{s}{s}", .{ base, suffix });
}

pub fn payload(self: Config, allocator: std.mem.Allocator) ![]u8 {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const stop: ?std.json.Value = if (self.stop.len == 0) null else if (self.stop[0] == '[')
        std.json.parseFromSliceLeaky(std.json.Value, a, self.stop, .{}) catch return error.InvalidStopSequences
    else
        .{ .string = self.stop };
    if (stop) |value| {
        if (value == .array) {
            if (value.array.items.len > 4) return error.TooManyStopSequences;
            for (value.array.items) |item| if (item != .string or item.string.len == 0) return error.InvalidStopSequences;
        }
    }
    const tools = try arrayOption(a, self.tools);
    const documents = try arrayOption(a, self.documents);
    const Message = struct { role: []const u8, content: []const u8 };
    const messages: [2]Message = .{
        .{ .role = "system", .content = self.systemPrompt },
        .{ .role = "user", .content = self.prompt },
    };
    return std.json.Stringify.valueAlloc(allocator, .{
        .model = self.model,
        .messages = messages[if (self.systemPrompt.len == 0) @as(usize, 1) else 0..],
        .stream = true,
        .stream_options = .{ .include_usage = true },
        .max_tokens = self.maxTokens,
        .max_completion_tokens = self.maxCompletionTokens,
        .temperature = self.temperature,
        .reasoning_effort = try Reasoning.parse(self.reasoningEffort),
        .stop = stop,
        .tools = tools,
        .documents = documents,
    }, .{ .emit_null_optional_fields = false });
}

fn arrayOption(allocator: std.mem.Allocator, text: []const u8) !?std.json.Value {
    if (text.len == 0) return null;
    const parsed = std.json.parseFromSliceLeaky(std.json.Value, allocator, text, .{}) catch return error.ExpectedJsonArray;
    if (parsed != .array) return error.ExpectedJsonArray;
    for (parsed.array.items) |item| if (item != .object) return error.ExpectedJsonObjects;
    return parsed;
}

test "llmd thinking budget and optional request controls keep their JSON types" {
    const allocator = std.testing.allocator;
    const config: Config = .{
        .reasoningEffort = "75",
        .systemPrompt = "Be brief",
        .maxCompletionTokens = 100,
        .temperature = 3,
        .stop = "[\"END\",\"STOP\"]",
        .tools = "[]",
        .documents = "[{\"text\":\"doc\"}]",
    };
    try config.validate();
    const body = try config.payload(allocator);
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    const root = parsed.value.object;
    try std.testing.expectEqual(@as(i64, 75), root.get("reasoning_effort").?.integer);
    try std.testing.expectEqual(@as(usize, 2), root.get("messages").?.array.items.len);
    try std.testing.expectEqual(@as(i64, 100), root.get("max_completion_tokens").?.integer);
    try std.testing.expectEqual(@as(usize, 2), root.get("stop").?.array.items.len);
    try std.testing.expectEqual(null, root.get("max_tokens"));
    try std.testing.expectError(error.InvalidReasoningEffort, Reasoning.parse("101"));
    try std.testing.expectError(error.ExpectedJsonArray, (Config{ .tools = "{}" }).payload(allocator));
}
