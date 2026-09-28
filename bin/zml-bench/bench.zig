const std = @import("std");

pub const max_batch = 256;
const max_event_size = 256 * 1024;

pub const Config = @import("config.zig");
pub const metrics = @import("metrics.zig");

pub const Status = enum { connecting, streaming, completed, failed, canceled };

pub const Result = struct {
    status: Status = .connecting,
    endMs: ?i64 = null,
    completionTokens: ?u64 = null,
    promptTokens: ?u64 = null,
    startedUs: i64 = 0,
    firstUs: ?i64 = null,
    lastUs: ?i64 = null,
    firstAnswerUs: ?i64 = null,
    endUs: ?i64 = null,
    chunks: u64 = 0,
    reasoningBytes: u64 = 0,
    itl: metrics.Histogram = .{},
    finishReason: ?enum { stop, length, tool_calls, content_filter, other } = null,
    outputBytes: u64 = 0,
    output: std.ArrayList(u8) = .empty,
    message: [192]u8 = undefined,
    messageLen: usize = 0,

    pub fn active(self: *const Result) bool {
        return self.status == .connecting or self.status == .streaming;
    }

    pub fn tokens(self: *const Result) u64 {
        return self.completionTokens orelse ((self.outputBytes + 3) / 4);
    }

    pub fn rate(self: *const Result, elapsedMs: i64) f64 {
        return perSecond(self.tokens(), (self.endMs orelse elapsedMs) - @divTrunc(self.startedUs, 1000));
    }

    pub fn ttftMs(self: *const Result) ?f64 {
        const first = self.firstUs orelse return null;
        return @as(f64, @floatFromInt(first - self.startedUs)) / 1000;
    }

    pub fn latencyMs(self: *const Result) ?f64 {
        const end = self.endUs orelse return null;
        return @as(f64, @floatFromInt(end - self.startedUs)) / 1000;
    }

    pub fn tpotMs(self: *const Result) ?f64 {
        const first = self.firstUs orelse return null;
        const last = self.lastUs orelse return null;
        const count = self.tokens();
        if (count < 2 or self.chunks < 2) return null;
        return @as(f64, @floatFromInt(last - first)) / 1000 / @as(f64, @floatFromInt(count - 1));
    }

    pub fn decodeRate(self: *const Result) ?f64 {
        const tpot = self.tpotMs() orelse return null;
        return if (tpot > 0) 1000 / tpot else null;
    }

    fn observe(self: *Result, nowUs: i64) void {
        if (self.lastUs) |last| self.itl.add(@as(f64, @floatFromInt(nowUs - last)) / 1000);
        if (self.firstUs == null) {
            self.firstUs = nowUs;
        }
        self.lastUs = nowUs;
        self.chunks += 1;
        self.status = .streaming;
    }

    pub fn deinit(self: *Result, allocator: std.mem.Allocator) void {
        self.output.deinit(allocator);
    }

    fn append(self: *Result, allocator: std.mem.Allocator, text: []const u8) !void {
        try self.output.ensureUnusedCapacity(allocator, text.len);
        self.outputBytes += text.len;
        // Retain the full response for the detail view, stripping terminal controls.
        for (text) |byte| {
            if (byte < 0x20 and byte != '\n' and byte != '\t') continue;
            if (byte == 0x7f) continue;
            self.output.appendAssumeCapacity(if (byte == '\t') ' ' else byte);
        }
    }

    fn setMessage(self: *Result, message: []const u8) void {
        self.messageLen = @min(message.len, self.message.len);
        for (message[0..self.messageLen], 0..) |byte, i| self.message[i] = if (byte >= 0x20 and byte < 0x7f) byte else ' ';
    }
};

pub const Summary = struct {
    completed: usize = 0,
    failed: usize = 0,
    stopped: usize = 0,
    active: usize = 0,
    tokens: u64 = 0,
    promptTokens: u64 = 0,
    promptUsageRequests: usize = 0,
    chunks: u64 = 0,
    estimated: bool = false,
    elapsedMs: i64 = 0,
    medianTtftMs: ?f64 = null,
    aggregateRate: f64 = 0,
    averageRate: f64 = 0,
    requestsPerSecond: f64 = 0,
    ttft: metrics.Stats = .{},
    latency: metrics.Stats = .{},
    itl: metrics.Stats = .{},
    tpot: metrics.Stats = .{},
    firstAnswer: metrics.Stats = .{},
};

pub fn summarize(results: []const Result, nowMs: i64) Summary {
    std.debug.assert(results.len <= max_batch);
    var summary: Summary = .{};
    var ttftBuffer: [max_batch]f64 = undefined;
    var ttfts: std.ArrayList(f64) = .initBuffer(&ttftBuffer);
    var latencyBuffer: [max_batch]f64 = undefined;
    var latencies: std.ArrayList(f64) = .initBuffer(&latencyBuffer);
    var tpotBuffer: [max_batch]f64 = undefined;
    var tpots: std.ArrayList(f64) = .initBuffer(&tpotBuffer);
    var answerBuffer: [max_batch]f64 = undefined;
    var answers: std.ArrayList(f64) = .initBuffer(&answerBuffer);
    var itl: metrics.Histogram = .{};
    for (results) |*result| {
        switch (result.status) {
            .completed => summary.completed += 1,
            .failed => summary.failed += 1,
            .canceled => summary.stopped += 1,
            .connecting, .streaming => summary.active += 1,
        }
        summary.tokens += result.tokens();
        summary.chunks += result.chunks;
        if (result.promptTokens) |count| {
            summary.promptTokens += count;
            summary.promptUsageRequests += 1;
        }
        summary.estimated = summary.estimated or (result.completionTokens == null and result.outputBytes > 0);
        summary.elapsedMs = @max(summary.elapsedMs, result.endMs orelse nowMs);
        summary.averageRate += result.rate(nowMs);
        if (result.ttftMs()) |ms| ttfts.appendAssumeCapacity(ms);
        // Final request latency excludes failed and canceled partial requests.
        if (result.status == .completed) if (result.latencyMs()) |ms| latencies.appendAssumeCapacity(ms);
        if (result.tpotMs()) |ms| tpots.appendAssumeCapacity(ms);
        if (result.firstAnswerUs) |us| answers.appendAssumeCapacity(@as(f64, @floatFromInt(us - result.startedUs)) / 1000);
        itl.merge(&result.itl);
    }
    summary.ttft = metrics.Stats.fromSamples(ttfts.items);
    summary.latency = metrics.Stats.fromSamples(latencies.items);
    summary.tpot = metrics.Stats.fromSamples(tpots.items);
    summary.firstAnswer = metrics.Stats.fromSamples(answers.items);
    summary.itl = itl.stats();
    summary.medianTtftMs = summary.ttft.p50Ms;
    if (results.len > 0) summary.averageRate /= @floatFromInt(results.len);
    summary.aggregateRate = perSecond(summary.tokens, summary.elapsedMs);
    summary.requestsPerSecond = perSecond(summary.completed, summary.elapsedMs);
    return summary;
}

pub const RequestReport = struct {
    index: usize,
    status: Status,
    estimated: bool,
    promptTokens: ?u64,
    completionTokens: u64,
    chunks: u64,
    reasoningBytes: u64,
    ttftMs: ?f64,
    firstAnswerMs: ?f64,
    latencyMs: ?f64,
    tpotMs: ?f64,
    decodeTokensPerSecond: ?f64,
    itl: metrics.Stats,
    finishReason: ?[]const u8,
    errorMessage: ?[]const u8,

    fn fromResult(index: usize, result: *const Result) RequestReport {
        return .{
            .index = index + 1,
            .status = result.status,
            .estimated = result.completionTokens == null,
            .promptTokens = result.promptTokens,
            .completionTokens = result.tokens(),
            .chunks = result.chunks,
            .reasoningBytes = result.reasoningBytes,
            .ttftMs = result.ttftMs(),
            .latencyMs = result.latencyMs(),
            .tpotMs = result.tpotMs(),
            .firstAnswerMs = if (result.firstAnswerUs) |us| @as(f64, @floatFromInt(us - result.startedUs)) / 1000 else null,
            .decodeTokensPerSecond = result.decodeRate(),
            .itl = result.itl.stats(),
            .finishReason = if (result.finishReason) |reason| @tagName(reason) else null,
            .errorMessage = if (result.messageLen > 0) result.message[0..result.messageLen] else null,
        };
    }
};

fn perSecond(count: u64, ms: i64) f64 {
    if (ms <= 0) return 0;
    return @as(f64, @floatFromInt(count)) * 1000 / @as(f64, @floatFromInt(ms));
}

/// Owns stable storage for all workers. Cancel/join before releasing request data.
pub const Batch = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    mutex: std.Io.Mutex = .init,
    group: std.Io.Group = .init,
    results: []Result,
    url: []u8,
    body: []u8,
    authorization: []u8,
    started: std.Io.Timestamp,

    pub fn create(allocator: std.mem.Allocator, io: std.Io, config: Config, apiKey: ?[]const u8) !*Batch {
        try config.validate();
        const self = try allocator.create(Batch);
        errdefer allocator.destroy(self);
        const url = try config.url(allocator);
        errdefer allocator.free(url);
        const body = try config.payload(allocator);
        errdefer allocator.free(body);
        const authorization = try std.fmt.allocPrint(allocator, "Bearer {s}", .{apiKey orelse ""});
        errdefer allocator.free(authorization);
        const results = try allocator.alloc(Result, config.batch);
        errdefer allocator.free(results);
        @memset(results, .{});
        errdefer for (results) |*result| result.deinit(allocator);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .results = results,
            .url = url,
            .body = body,
            .authorization = authorization,
            .started = .now(io, .awake),
        };
        errdefer self.group.cancel(io);
        for (results, 0..) |_, index| try self.group.concurrent(io, worker, .{ self, index });
        return self;
    }

    pub fn destroy(self: *Batch) void {
        self.group.cancel(self.io);
        const allocator = self.allocator;
        for (self.results) |*result| result.deinit(allocator);
        allocator.free(self.results);
        allocator.free(self.url);
        allocator.free(self.body);
        allocator.free(self.authorization);
        allocator.destroy(self);
    }

    pub fn elapsed(self: *Batch) i64 {
        return self.started.untilNow(self.io, .awake).toMilliseconds();
    }

    pub fn elapsedUs(self: *Batch) i64 {
        return self.started.untilNow(self.io, .awake).toMicroseconds();
    }

    pub fn writeReport(self: *Batch, writer: *std.Io.Writer) !void {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        var requests: [max_batch]RequestReport = undefined;
        for (self.results, 0..) |*result, i| requests[i] = .fromResult(i, result);
        const parsedRequest = try std.json.parseFromSlice(std.json.Value, self.allocator, self.body, .{});
        defer parsedRequest.deinit();
        try std.json.Stringify.value(.{
            .version = 1,
            .endpoint = self.url,
            .request = parsedRequest.value,
            .summary = summarize(self.results, self.elapsed()),
            .requests = requests[0..self.results.len],
            .timingNotes = "ITL measures nonempty SSE chunk arrival gaps; percentiles are histogram estimates. TPOT uses (last-first)/(output tokens-1), estimated until usage arrives. No server token timestamps are available.",
        }, .{ .whitespace = .indent_2 }, writer);
        try writer.writeByte('\n');
    }

    pub fn saveReport(self: *Batch, path: []const u8) !void {
        var file = try std.Io.Dir.cwd().createFile(self.io, path, .{});
        defer file.close(self.io);
        var buffer: [8192]u8 = undefined;
        var writer = file.writer(self.io, &buffer);
        try self.writeReport(&writer.interface);
        try writer.interface.flush();
    }

    pub fn summary(self: *Batch) Summary {
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        return summarize(self.results, self.elapsed());
    }

    pub fn stop(self: *Batch) void {
        // Mark first so a canceled network read cannot overwrite the requested status.
        self.mutex.lockUncancelable(self.io);
        for (self.results) |*result| if (result.active()) {
            result.status = .canceled;
            result.endUs = self.elapsedUs();
            result.endMs = @divTrunc(result.endUs.?, 1000);
        };
        self.mutex.unlock(self.io);
        self.group.cancel(self.io);
    }

    fn worker(self: *Batch, index: usize) void {
        self.mutex.lockUncancelable(self.io);
        if (!self.results[index].active()) {
            self.mutex.unlock(self.io);
            return;
        }
        self.results[index].startedUs = self.elapsedUs();
        self.mutex.unlock(self.io);
        self.request(index) catch |err| {
            self.mutex.lockUncancelable(self.io);
            defer self.mutex.unlock(self.io);
            const result = &self.results[index];
            if (result.active()) {
                result.status = .failed;
                result.endUs = self.elapsedUs();
                result.endMs = @divTrunc(result.endUs.?, 1000);
                if (result.messageLen == 0) result.setMessage(@errorName(err));
            }
            return;
        };
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const result = &self.results[index];
        if (result.active()) {
            result.status = .completed;
            result.endUs = self.elapsedUs();
            result.endMs = @divTrunc(result.endUs.?, 1000);
        }
    }

    fn request(self: *Batch, index: usize) !void {
        var client: std.http.Client = .{ .allocator = self.allocator, .io = self.io, .read_buffer_size = 64 * 1024 };
        defer client.deinit();
        const headers: []const std.http.Header = if (self.authorization.len > "Bearer ".len)
            &.{.{ .name = "Authorization", .value = self.authorization }}
        else
            &.{};
        var req = try client.request(.POST, try std.Uri.parse(self.url), .{
            .redirect_behavior = .not_allowed,
            .headers = .{
                .content_type = .{ .override = "application/json" },
                .accept_encoding = .{ .override = "identity" },
            },
            .extra_headers = headers,
        });
        defer req.deinit();
        try req.sendBodyComplete(self.body);
        var headBuffer: [8192]u8 = undefined;
        var response = try req.receiveHead(&headBuffer);
        if (response.head.status != .ok) {
            var message: [128]u8 = undefined;
            const text = try std.fmt.bufPrint(&message, "HTTP {d} {s}", .{ @intFromEnum(response.head.status), response.head.reason });
            self.mutex.lockUncancelable(self.io);
            self.results[index].setMessage(text);
            self.mutex.unlock(self.io);
            return error.HttpError;
        }
        var transferBuffer: [64 * 1024]u8 = undefined;
        const reader = response.reader(&transferBuffer);
        var sse: Sse = .{};
        defer sse.deinit(self.allocator);
        var finished = false;
        while (try reader.takeDelimiter('\n')) |line| {
            if (try sse.line(self.allocator, line)) |event| {
                if (try self.consume(index, event, &finished)) return;
            }
        }
        if (!sse.dispatched and sse.data.items.len > 0) {
            if (try self.consume(index, sse.data.items, &finished)) return;
        }
        if (!finished) return error.IncompleteStream;
    }

    fn consume(self: *Batch, index: usize, data: []const u8, finished: *bool) !bool {
        return self.consumeAt(index, data, finished, self.elapsedUs());
    }

    fn consumeAt(self: *Batch, index: usize, data: []const u8, finished: *bool, receivedUs: i64) !bool {
        const parsed = try parseChunk(self.allocator, data);
        defer if (parsed) |p| p.deinit();
        if (parsed == null) return true;
        const chunk = parsed.?.value;
        self.mutex.lockUncancelable(self.io);
        defer self.mutex.unlock(self.io);
        const result = &self.results[index];
        if (!result.active()) return error.Canceled;
        if (chunk.@"error") |apiError| {
            result.setMessage(apiError.message orelse "Endpoint returned a streaming error");
            return error.ApiError;
        }
        if (chunk.usage) |usage| {
            if (usage.completion_tokens) |count| result.completionTokens = count;
            if (usage.prompt_tokens) |count| result.promptTokens = count;
        }
        const beforeBytes = result.outputBytes;
        for (chunk.choices) |choice| {
            if (choice.index != 0) continue;
            if (choice.finish_reason) |reason| {
                finished.* = true;
                result.finishReason = std.meta.stringToEnum(@typeInfo(@FieldType(Result, "finishReason")).optional.child, reason) orelse .other;
            }
            if (choice.delta.content) |text| if (text.len > 0 and result.firstAnswerUs == null) {
                result.firstAnswerUs = receivedUs;
            };
            if (choice.delta.reasoning_content orelse choice.delta.reasoning) |reasoning| result.reasoningBytes += reasoning.len;
            const parts = .{ choice.delta.reasoning_content orelse choice.delta.reasoning, choice.delta.content };
            inline for (parts) |part| if (part) |text| {
                if (text.len > 0) {
                    try result.append(self.allocator, text);
                }
            };
            for (choice.delta.tool_calls orelse &.{}) |tool| {
                if (tool.function) |function| {
                    if (function.name) |name| try result.append(self.allocator, name);
                    if (function.arguments) |arguments| try result.append(self.allocator, arguments);
                }
            }
        }
        if (result.outputBytes > beforeBytes) result.observe(receivedUs);
        return false;
    }
};

const Chunk = struct {
    choices: []const struct {
        index: u32 = 0,
        delta: struct {
            content: ?[]const u8 = null,
            reasoning_content: ?[]const u8 = null,
            reasoning: ?[]const u8 = null,
            tool_calls: ?[]const struct {
                function: ?struct { name: ?[]const u8 = null, arguments: ?[]const u8 = null } = null,
            } = null,
        } = .{},
        finish_reason: ?[]const u8 = null,
    } = &.{},
    usage: ?struct { completion_tokens: ?u64 = null, prompt_tokens: ?u64 = null } = null,
    @"error": ?struct { message: ?[]const u8 = null } = null,
};

fn parseChunk(allocator: std.mem.Allocator, data: []const u8) !?std.json.Parsed(Chunk) {
    if (std.mem.eql(u8, std.mem.trim(u8, data, " \r\n"), "[DONE]")) return null;
    return try std.json.parseFromSlice(Chunk, allocator, data, .{ .ignore_unknown_fields = true });
}

/// SSE events can span multiple data lines; TCP and HTTP chunk boundaries are irrelevant.
const Sse = struct {
    data: std.ArrayList(u8) = .empty,
    dispatched: bool = false,

    fn deinit(self: *Sse, allocator: std.mem.Allocator) void {
        self.data.deinit(allocator);
    }

    fn line(self: *Sse, allocator: std.mem.Allocator, raw: []const u8) !?[]const u8 {
        if (self.dispatched) {
            self.data.clearRetainingCapacity();
            self.dispatched = false;
        }
        const text = std.mem.trimEnd(u8, raw, "\r");
        if (text.len == 0) {
            if (self.data.items.len == 0) return null;
            self.dispatched = true;
            return self.data.items;
        }
        if (!std.mem.startsWith(u8, text, "data:")) return null;
        var value = text[5..];
        if (std.mem.startsWith(u8, value, " ")) value = value[1..];
        if (self.data.items.len + value.len + 1 > max_event_size) return error.EventTooLarge;
        if (self.data.items.len > 0) try self.data.append(allocator, '\n');
        try self.data.appendSlice(allocator, value);
        return null;
    }
};

test "SSE handles comments, CRLF and multiline events" {
    const allocator = std.testing.allocator;
    var sse: Sse = .{};
    defer sse.deinit(allocator);
    try std.testing.expectEqual(null, try sse.line(allocator, ": heartbeat\r"));
    try std.testing.expectEqual(null, try sse.line(allocator, "data: {\r"));
    try std.testing.expectEqual(null, try sse.line(allocator, "data: \"choices\": []}\r"));
    const event = (try sse.line(allocator, "\r")).?;
    const chunk = (try parseChunk(allocator, event)).?;
    defer chunk.deinit();
    try std.testing.expectEqual(@as(usize, 0), chunk.value.choices.len);
    _ = try sse.line(allocator, "data: [DONE]");
    try std.testing.expectEqual(null, try parseChunk(allocator, (try sse.line(allocator, "")).?));
}

test "payload escapes prompts and omits automatic options" {
    const allocator = std.testing.allocator;
    const config: Config = .{ .prompt = "quote \" and\nnewline" };
    const body = try config.payload(allocator);
    defer allocator.free(body);
    const parsed = try std.json.parseFromSlice(std.json.Value, allocator, body, .{});
    defer parsed.deinit();
    try std.testing.expectEqualStrings(config.prompt, parsed.value.object.get("messages").?.array.items[0].object.get("content").?.string);
    try std.testing.expectEqual(null, parsed.value.object.get("max_tokens"));
    try std.testing.expectEqual(null, parsed.value.object.get("temperature"));
    try std.testing.expectError(error.BatchMustBeBetween1And256, (Config{ .batch = 0 }).validate());
    try std.testing.expectError(error.InvalidEndpoint, (Config{ .endpoint = "file:///tmp/x" }).validate());
    try std.testing.expectError(error.TemperatureMustBeFiniteAndNonnegative, (Config{ .temperature = std.math.nan(f64) }).validate());
}

test "summary uses usage, even median, and freezes finished duration" {
    const results: [2]Result = .{
        .{ .status = .completed, .firstUs = 100000, .endUs = 1000000, .endMs = 1000, .completionTokens = 50 },
        .{ .status = .completed, .firstUs = 200000, .endUs = 2000000, .endMs = 2000, .completionTokens = 100 },
    };
    const summary = summarize(&results, 9000);
    try std.testing.expectEqual(@as(f64, 150), summary.medianTtftMs.?);
    try std.testing.expectEqual(@as(f64, 75), summary.aggregateRate);
    try std.testing.expectEqual(@as(f64, 50), summary.averageRate);
    try std.testing.expectEqual(@as(i64, 2000), summary.elapsedMs);
    try std.testing.expect(!summary.estimated);
}

test "full output survives beyond the former preview limit and strips terminal escapes" {
    const allocator = std.testing.allocator;
    var result: Result = .{};
    defer result.deinit(allocator);
    for (0..4000) |_| try result.append(allocator, "🙂\x1b\x00");
    try std.testing.expectEqual(@as(usize, 16000), result.output.items.len);
    try std.testing.expectEqual(@as(u64, 24000), result.outputBytes);
    try std.testing.expect(std.unicode.utf8ValidateSlice(result.output.items));
    try std.testing.expectEqual(null, std.mem.indexOfScalar(u8, result.output.items, 0x1b));
}

test "timings ignore empty and usage chunks and count a multi-part SSE event once" {
    const allocator = std.testing.allocator;
    var results: [1]Result = .{.{ .startedUs = 1000 }};
    defer results[0].deinit(allocator);
    var batch: Batch = .{
        .allocator = allocator,
        .io = std.testing.io,
        .results = &results,
        .url = &.{},
        .body = &.{},
        .authorization = &.{},
        .started = .now(std.testing.io, .awake),
    };
    var finished = false;
    _ = try batch.consumeAt(0, "{\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}", &finished, 5000);
    try std.testing.expectEqual(null, results[0].firstUs);
    _ = try batch.consumeAt(0, "{\"choices\":[{\"delta\":{\"reasoning_content\":\"thought\",\"content\":\"answer\"}}]}", &finished, 11000);
    _ = try batch.consumeAt(0, "{\"choices\":[{\"delta\":{\"tool_calls\":[{\"function\":{\"arguments\":\"{}\"}}]}}]}", &finished, 31000);
    _ = try batch.consumeAt(0, "{\"choices\":[],\"usage\":{\"completion_tokens\":5,\"prompt_tokens\":10}}", &finished, 51000);
    _ = try batch.consumeAt(0, "{\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}", &finished, 61000);
    try std.testing.expect(finished);
    try std.testing.expectEqual(@as(u64, 2), results[0].chunks);
    try std.testing.expectEqual(@as(?f64, 10), results[0].ttftMs());
    try std.testing.expectEqual(@as(?f64, 20), results[0].itl.stats().meanMs);
    try std.testing.expectEqual(@as(?f64, 5), results[0].tpotMs());
    try std.testing.expectEqual(@as(?f64, 200), results[0].decodeRate());
    try std.testing.expectEqual(@as(?i64, 11000), results[0].firstAnswerUs);
    const summary = summarize(&results, 100);
    try std.testing.expectEqual(@as(u64, 0), summary.latency.count);
    try std.testing.expectEqual(@as(u64, 10), summary.promptTokens);
    try std.testing.expectEqual(@as(?f64, 20), summary.itl.meanMs);
    try std.testing.expectEqual(@as(u64, 1), summary.itl.count);
}

test "one output chunk has no ITL or TPOT even with usage" {
    var result: Result = .{ .completionTokens = 20 };
    result.observe(20000);
    try std.testing.expectEqual(null, result.tpotMs());
    try std.testing.expectEqual(null, result.itl.stats().meanMs);
    result.observe(20000);
    try std.testing.expectEqual(@as(?f64, 0), result.itl.stats().meanMs);
    try std.testing.expectEqual(null, result.decodeRate());
}
