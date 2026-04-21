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
    platform: *const zml.Platform,
    session: models.Session,
    tokens: std.ArrayList(u32),

    pub const RunOnceOptions = struct {
        profile: bool = false,
        profile_repository_path: []const u8 = zml.Platform.ProfilerOptions.defaults.repository_path,
    };

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
            .platform = platform,
            .session = session,
            .tokens = try .initCapacity(allocator, session.maxTokens()),
        };
    }

    pub fn deinit(self: *Chat) void {
        self.tokens.deinit(self.allocator);
        self.session.deinit();
    }

    pub fn runOnce(self: *Chat, prompt: []const u8) !void {
        try self.runOnceWithOptions(prompt, .{});
    }

    pub fn runOnceWithOptions(self: *Chat, prompt: []const u8, opts: RunOnceOptions) !void {
        const prompt_tokens = try self.session.tokenizePrompt(self.allocator, prompt);
        defer self.allocator.free(prompt_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});
        var profiler: ?zml.Platform.Profiler = null;
        defer if (profiler) |*p| p.deinit();

        if (opts.profile) {
            var session_id_buf: [64]u8 = undefined;
            const session_id = try std.fmt.bufPrint(
                &session_id_buf,
                "llm-runonce-{d}",
                .{@as(u64, @intCast(std.Io.Clock.now(.real, self.io).toNanoseconds()))},
            );
            profiler = try self.platform.profiler(self.allocator, self.io, .{
                .repository_path = opts.profile_repository_path,
                .session_id = session_id,
            });
            try profiler.?.start();
        }

        try self.runPrefill(prompt_tokens, &stdout.interface);
        const decode_stats = try self.runDecodeTurn(&stdout.interface);
        const profile = if (profiler) |*p| try p.stop() else null;

        try stdout.interface.writeAll("\n\n");
        try stdout.interface.print("\x1b[2m{:.1} toks/s\x1b[0m\n\n", .{
            tokensPerSecond(decode_stats.duration, decode_stats.tokens),
        });
        if (opts.profile) {
            if (profile) |trace| {
                try stdout.interface.print("\x1b[2mprofile: {s}\x1b[0m\n\n", .{trace.perfetto_path});
            } else {
                try stdout.interface.writeAll("\x1b[2mprofile: unavailable on this PJRT plugin\x1b[0m\n\n");
            }
        }
        try stdout.interface.flush();
    }

    pub fn runInteractive(self: *Chat, initial_prompt: []const u8) !void {
        var turn_tokens = try self.session.tokenizePrompt(self.allocator, initial_prompt);
        defer self.allocator.free(turn_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});

        while (self.tokens.items.len <= self.session.maxTokens()) {
            try self.runPrefill(turn_tokens, &stdout.interface);
            const decode_stats = try self.runDecodeTurn(&stdout.interface);
            try self.printDecodeStats(&stdout.interface, decode_stats);

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

    const DecodeStats = struct {
        duration: std.Io.Duration,
        tokens: u64,
    };

    fn runDecodeTurn(self: *Chat, stdout: *std.Io.Writer) !DecodeStats {
        const decode_start: std.Io.Timestamp = .now(self.io, .awake);
        const decode_pos_before = self.tokens.items.len;
        try self.session.runDecode(&self.tokens, stdout);
        const decode_duration = decode_start.untilNow(self.io, .awake);
        const decode_tokens: u64 = self.tokens.items.len - decode_pos_before;

        return .{
            .duration = decode_duration,
            .tokens = decode_tokens,
        };
    }

    fn printDecodeStats(self: *Chat, stdout: *std.Io.Writer, decode_stats: DecodeStats) !void {
        _ = self;
        try stdout.writeAll("\n\n");
        try stdout.print("\x1b[36mdecode\x1b[0m \x1b[2m{f} · {:.1}toks/s\x1b[0m\n\n\n", .{
            decode_stats.duration,
            tokensPerSecond(decode_stats.duration, decode_stats.tokens),
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
