const std = @import("std");

const c = @import("c");
const zml = @import("zml");

const models = @import("models.zig");

const log = std.log.scoped(.llm);

pub const prompt_prefix: [:0]const u8 = "> ";
pub const history_file_path: [:0]const u8 = ".llm_chat_history";

pub fn initHistory() void {
    _ = c.linenoiseHistorySetMaxLen(1_000);
    _ = c.linenoiseHistoryLoad(history_file_path.ptr);
}

pub fn rememberPrompt(line: [*:0]const u8) void {
    if (line[0] == 0) return;
    if (c.linenoiseHistoryAdd(line) == 1) {
        _ = c.linenoiseHistorySave(history_file_path.ptr);
    }
}

pub const Chat = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    session: models.Session,
    tokens: std.ArrayList(u32),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const models.CompiledModel,
        model_buffers: *models.Buffers,
    ) !Chat {
        var session = try compiled_model.newSession(
            allocator,
            io,
            platform,
            model_buffers,
            tokenizer,
        );
        errdefer session.deinit();

        return .{
            .allocator = allocator,
            .io = io,
            .session = session,
            .tokens = try .initCapacity(allocator, session.maxTokens()),
        };
    }

    pub fn deinit(self: *Chat) void {
        self.tokens.deinit(self.allocator);
        self.session.deinit();
    }

    pub fn runOnce(self: *Chat, prompt: []const u8) !void {
        const prompt_tokens = try self.session.tokenizePrompt(self.allocator, prompt);
        defer self.allocator.free(prompt_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});

        log.info("Running prefill...", .{});
        try self.tokens.appendSlice(self.allocator, prompt_tokens);
        try self.session.runPrefill(self.tokens.items);

        log.info("Running decode...", .{});
        try self.session.runDecode(&self.tokens, &stdout.interface);

        try stdout.interface.writeAll("\n\n");
        try stdout.interface.flush();
    }

    pub fn runInteractive(self: *Chat, initial_prompt: []const u8) !void {
        var turn_tokens = try self.session.tokenizePrompt(self.allocator, initial_prompt);
        defer self.allocator.free(turn_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});

        while (self.tokens.items.len <= self.session.maxTokens()) {
            try self.runPrefill(turn_tokens, &stdout.interface);
            try self.runDecodeTurn(&stdout.interface);

            turn_tokens = if (self.tokens.items.len >= self.session.maxTokens()) b: {
                self.allocator.free(turn_tokens);
                break :b try self.allocator.alloc(u32, 0);
            } else try self.readNextTurn(turn_tokens, &stdout.interface) orelse return;
            if (self.tokens.items.len + turn_tokens.len >= self.session.maxTokens()) {
                log.warn("Not enough tokens remaining ({} available, {} needed). Consider using a higher seqlen, for example: `--seqlen={d}`.", .{
                    self.session.maxTokens() -| self.tokens.items.len,
                    turn_tokens.len,
                    std.math.ceilPowerOfTwo(u32, self.session.maxTokens() + 1) catch std.math.maxInt(u32),
                });
                return;
            }
        }
    }

    fn runPrefill(self: *Chat, prompt_tokens: []const u32, stdout: *std.Io.Writer) !void {
        if (prompt_tokens.len + self.tokens.items.len > self.session.maxTokens()) {
            return error.PromptTooLong;
        }
        try stdout.writeAll("\x1b[2mprefill...\x1b[0m");
        try stdout.flush();

        try self.tokens.appendSlice(self.allocator, prompt_tokens);
        try self.session.runPrefill(self.tokens.items);

        try stdout.writeAll("\r          \r");
        try stdout.flush();
    }

    fn runDecodeTurn(self: *Chat, stdout: *std.Io.Writer) !void {
        const decode_start: std.Io.Timestamp = .now(self.io, .awake);
        const decode_pos_before = self.tokens.items.len;
        try self.session.runDecode(&self.tokens, stdout);
        const decode_duration = decode_start.untilNow(self.io, .awake);
        const decode_tokens: u64 = self.tokens.items.len - decode_pos_before;

        try stdout.writeAll("\n\n");
        try stdout.print("\x1b[36mdecode\x1b[0m \x1b[2m{f} · {:.1}tok/s\x1b[0m\n\n\n", .{
            decode_duration,
            tokensPerSecond(decode_duration, decode_tokens),
        });
        try stdout.flush();
    }

    fn readNextTurn(self: *Chat, previous_turn_tokens: []const u32, stdout: *std.Io.Writer) !?[]const u32 {
        while (true) {
            const line = c.linenoise(prompt_prefix.ptr) orelse {
                try stdout.print("\x1b[2m{}/{} tokens used\x1b[0m\n\n", .{ self.tokens.items.len, self.session.maxTokens() });
                try stdout.flush();
                return null;
            };
            defer c.linenoiseFree(line);

            rememberPrompt(line);
            const input = std.mem.sliceTo(line, 0);
            if (input.len == 0) continue;

            self.allocator.free(previous_turn_tokens);
            return try self.session.tokenizeTurn(self.allocator, input);
        }
    }
};

fn tokensPerSecond(duration: std.Io.Duration, tokens: u64) f64 {
    if (tokens == 0) return 0;
    const seconds = @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e9;
    if (seconds <= 0) return 0;
    return @as(f64, @floatFromInt(tokens)) / seconds;
}
