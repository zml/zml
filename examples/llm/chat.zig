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
    tokenizer: zml.tokenizer.Tokenizer,
    repo: *models.Repository,
    session: *models.Session,
    tokens: std.ArrayList(u32),

    pub fn init(allocator: std.mem.Allocator, io: std.Io, tokenizer: zml.tokenizer.Tokenizer, repo: *models.Repository, session: *models.Session) !Chat {
        return .{
            .allocator = allocator,
            .io = io,
            .tokenizer = tokenizer,
            .repo = repo,
            .session = session,
            .tokens = try .initCapacity(allocator, session.maxSeqLen()),
        };
    }

    pub fn deinit(self: *Chat) void {
        self.tokens.deinit(self.allocator);
    }

    pub fn runOnce(self: *Chat, prompt: []const u8) !void {
        const prompt_tokens = try self.repo.tokenizePrompt(self.allocator, self.tokenizer, prompt);
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
        var turn_tokens = try self.repo.tokenizePrompt(self.allocator, self.tokenizer, initial_prompt);
        defer self.allocator.free(turn_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});

        while (self.session.remainingTokens() > 0) {
            try self.runPrefill(turn_tokens, &stdout.interface);
            try self.runDecodeTurn(&stdout.interface);

            turn_tokens = try self.readNextTurn(turn_tokens, &stdout.interface) orelse return;
            if (turn_tokens.len > self.session.remainingTokens()) {
                log.warn("Not enough tokens remaining ({} available, {} needed). Consider using a higher seqlen, for example: `--seqlen={d}`.", .{ self.session.remainingTokens(), turn_tokens.len, std.math.ceilPowerOfTwo(u32, self.session.maxSeqLen() + 1) catch std.math.maxInt(u32) });
                return;
            }
        }
    }

    fn runPrefill(self: *Chat, prompt_tokens: []const u32, stdout: *std.Io.Writer) !void {
        if (prompt_tokens.len + self.tokens.items.len > self.session.maxSeqLen()) {
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
        const decode_pos_before = self.session.tokenPos();
        try self.session.runDecode(&self.tokens, stdout);
        const decode_duration = decode_start.untilNow(self.io, .awake);
        const decode_tokens: u64 = self.session.tokenPos() - decode_pos_before;

        try stdout.writeAll("\n\n");
        try stdout.print("\x1b[36mdecode\x1b[0m \x1b[2m{f} · {:.1}tok/s\x1b[0m\n\n\n", .{
            decode_duration,
            tokensPerSecond(decode_duration, decode_tokens),
        });
        try stdout.flush();
    }

    fn readNextTurn(self: *Chat, previous_turn_tokens: []u32, stdout: *std.Io.Writer) !?[]u32 {
        while (true) {
            const line = c.linenoise(prompt_prefix.ptr) orelse {
                try stdout.print("\x1b[2m{}/{} tokens used\x1b[0m\n\n", .{ self.session.tokenPos(), self.session.maxSeqLen() });
                try stdout.flush();
                return null;
            };
            defer c.linenoiseFree(line);

            rememberPrompt(line);
            const input = std.mem.sliceTo(line, 0);
            if (input.len == 0) continue;

            self.allocator.free(previous_turn_tokens);
            return try self.repo.tokenizeTurn(self.allocator, self.tokenizer, input);
        }
    }
};

fn tokensPerSecond(duration: std.Io.Duration, tokens: u64) f64 {
    if (tokens == 0) return 0;
    const seconds = @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e9;
    if (seconds <= 0) return 0;
    return @as(f64, @floatFromInt(tokens)) / seconds;
}
