const std = @import("std");

const c = @import("c");
const zml = @import("zml");
const stdx = zml.stdx;
const attention = zml.attention.attention;

const inference = @import("inference.zig");
const model = @import("model.zig");

const log = std.log.scoped(.lfm);

pub const DEFAULT_PROMPT: [:0]const u8 = "> ";

pub const Context = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_buffers: *zml.Bufferized(model.Model),
    exe: inference.Inference,
    tokenizer: zml.tokenizer.Tokenizer,
    config: model.Config,
    seqlen: u32,
    cache_buffers: zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
    rng_buf: zml.Bufferized(zml.Tensor.Rng),
    generated_token_slice: zml.Slice,
    token_pos: u32,
    all_tokens: std.ArrayList(u32),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        model_buffers: *zml.Bufferized(model.Model),
        cache: model.Cache,
        attention_metadata: attention.Metadata,
        exe: inference.Inference,
        tokenizer: zml.tokenizer.Tokenizer,
        config: model.Config,
        seqlen: u32,
    ) !Context {
        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .exe = exe,
            .tokenizer = tokenizer,
            .config = config,
            .seqlen = seqlen,
            .cache_buffers = try cache.initBuffers(allocator, io, platform),
            .attention_metadata_buffers = try attention_metadata.initBuffer(io, platform),
            .rng_buf = try zml.Tensor.Rng.initBuffer(platform, seed, io),
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .batch = 1, .seq = 1 }, .u32)),
            .token_pos = 0,
            .all_tokens = try .initCapacity(allocator, seqlen),
        };
    }

    pub fn deinit(self: *Context) void {
        model.Cache.unloadBuffers(&self.cache_buffers);
        attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buf);
        self.generated_token_slice.free(self.allocator);
        self.all_tokens.deinit(self.allocator);
    }

    pub fn runOnce(self: *Context, prompt: []const u8) !void {
        const prompt_tokens: []const u32 = try tokenizePrompt(
            self.allocator,
            self.tokenizer,
            self.config,
            prompt,
        );
        defer self.allocator.free(prompt_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});

        log.info("Running prefill...", .{});
        try self.runPrefill(prompt_tokens);

        log.info("Running decode...", .{});
        try self.runDecode(&stdout.interface);
    }

    pub fn start(self: *Context, prompt: []const u8) !void {
        var turn_tokens: []const u32 = try tokenizePrompt(
            self.allocator,
            self.tokenizer,
            self.config,
            prompt,
        );
        defer self.allocator.free(turn_tokens);

        var stdout = std.Io.File.stdout().writer(self.io, &.{});

        while (self.remainingTokens() > 0) {
            try stdout.interface.writeAll("\x1b[2mprefill...\x1b[0m");
            try stdout.interface.flush();

            const prefill_start: std.Io.Timestamp = .now(self.io, .awake);
            const prefill_pos_before = self.token_pos;
            try self.runPrefill(turn_tokens);
            const prefill_duration = prefill_start.untilNow(self.io, .awake);
            const prefill_tokens: u64 = self.token_pos - prefill_pos_before;

            try stdout.interface.writeAll("\r          \r");
            try stdout.interface.flush();

            const decode_start: std.Io.Timestamp = .now(self.io, .awake);
            const decode_pos_before = self.token_pos;
            try self.runDecode(&stdout.interface);
            const decode_duration = decode_start.untilNow(self.io, .awake);
            const decode_tokens: u64 = self.token_pos - decode_pos_before;

            try stdout.interface.writeAll("\n\n");

            try stdout.interface.print("\x1b[36mprefill\x1b[0m \x1b[2m{D} \xc2\xb7 {:.1}tok/s\x1b[0m \x1b[2m\xe2\x94\x82\x1b[0m \x1b[36mdecode\x1b[0m \x1b[2m{D} \xc2\xb7 {:.1}tok/s\x1b[0m\n\n\n", .{
                stdx.fmt.fmtDuration(prefill_duration),
                tokensPerSecond(prefill_duration, prefill_tokens),
                stdx.fmt.fmtDuration(decode_duration),
                tokensPerSecond(decode_duration, decode_tokens),
            });
            try stdout.interface.flush();

            while (true) {
                const line = c.linenoise(DEFAULT_PROMPT.ptr) orelse {
                    try stdout.interface.print("\x1b[2m{}/{} tokens used\x1b[0m\n\n", .{ self.token_pos, self.seqlen });
                    try stdout.interface.flush();
                    return;
                };
                defer c.linenoiseFree(line);
                const input = std.mem.sliceTo(line, 0);
                if (input.len == 0) continue;

                self.allocator.free(turn_tokens);
                turn_tokens = try tokenizeTurn(self.allocator, self.tokenizer, input);
                break;
            }

            if (turn_tokens.len > self.remainingTokens()) {
                log.warn("Not enough tokens remaining ({} available, {} needed). Consider using a higher seqlen, for example: `--seqlen={d}`.", .{ self.remainingTokens(), turn_tokens.len, std.math.ceilPowerOfTwo(u32, self.seqlen + 1) catch std.math.maxInt(u32) });
                break;
            }
        }
    }

    fn remainingTokens(self: Context) u32 {
        return self.seqlen -| self.token_pos;
    }

    fn runPrefill(self: *Context, prompt_tokens: []const u32) !void {
        try self.all_tokens.appendSlice(self.allocator, prompt_tokens);

        if (self.all_tokens.items.len > self.seqlen) {
            return error.PromptTooLong;
        }

        const tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{ .batch = 1, .seq = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);
        const tokens = tokens_slice.items(u32);
        @memset(tokens, self.config.pad_token_id);
        @memcpy(tokens[0..self.all_tokens.items.len], self.all_tokens.items);

        var tokens_buf: zml.Buffer = try .fromSlice(self.io, self.platform, tokens_slice);
        defer tokens_buf.deinit();

        // Always prefill from position 0 with all accumulated tokens.
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var tokens_pos_buf: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice);
        defer tokens_pos_buf.deinit();

        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(self.all_tokens.items.len)}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice);
        defer actual_seq_len_buf.deinit();

        try self.exe.prefill.run(.{
            .allocator = self.allocator,
            .io = self.io,
            .platform = self.platform,
            .model_buffers = self.model_buffers,
            .tokens_buf = &tokens_buf,
            .tokens_pos_buf = &tokens_pos_buf,
            .actual_seq_len_buf = &actual_seq_len_buf,
            .rng_buf = &self.rng_buf,
            .cache_buffers = &self.cache_buffers,
            .attention_metadata_buffers = self.attention_metadata_buffers,
        });

        try tokens_buf.toSlice(self.io, tokens_slice);
        self.generated_token_slice.items(u32)[0] = tokens_slice.items(u32)[self.all_tokens.items.len - 1];
        self.token_pos = @intCast(self.all_tokens.items.len);
    }

    fn runDecode(self: *Context, writer: *std.Io.Writer) !void {
        var tokenizer_decoder = try self.tokenizer.decoder();
        defer tokenizer_decoder.deinit();

        var current_token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice);
        defer current_token_buffer.deinit();

        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice);
        defer actual_seq_len_buf.deinit();

        const think_start = self.tokenizer.tokenToId("<think>") orelse unreachable;
        const think_end = self.tokenizer.tokenToId("</think>") orelse unreachable;

        generation: while (true) {
            const generated_token = self.generated_token_slice.items(u32)[0];

            if (generated_token == self.config.eos_token_id) break :generation;
            if (try tokenizer_decoder.next(generated_token)) |chunk| {
                if (generated_token == think_start) {
                    try writer.writeAll("\x1b[2m");
                }
                try writer.writeAll(chunk);
                if (generated_token == think_end) {
                    try writer.writeAll("\x1b[0m");
                }
                try writer.flush();
            }

            // Accumulate the generated token.
            try self.all_tokens.append(self.allocator, generated_token);

            if (self.remainingTokens() == 0) break :generation;
            const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{self.token_pos}));
            var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice);
            defer token_pos_buffer.deinit();

            try self.exe.decode.run(.{
                .allocator = self.allocator,
                .io = self.io,
                .platform = self.platform,
                .model_buffers = self.model_buffers,
                .tokens_buf = &current_token_buffer,
                .tokens_pos_buf = &token_pos_buffer,
                .actual_seq_len_buf = &actual_seq_len_buf,
                .rng_buf = &self.rng_buf,
                .cache_buffers = &self.cache_buffers,
                .attention_metadata_buffers = self.attention_metadata_buffers,
            });

            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
            self.token_pos += 1;
        }
    }
};

fn tokensPerSecond(duration: std.Io.Duration, tokens: u64) f64 {
    if (tokens == 0) return 0;
    const seconds = @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e9;
    if (seconds <= 0) return 0;
    return @as(f64, @floatFromInt(tokens)) / seconds;
}

fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, config: model.Config, prompt: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
    const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
    const newline = (try encoder.encode("\n"))[0];

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.appendSlice(allocator, &.{ config.bos_token_id, im_start, user, newline });
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.appendSlice(allocator, &.{ im_end, newline });
    try tokens.appendSlice(allocator, &.{ im_start, assistant, newline });

    return tokens.toOwnedSlice(allocator);
}

fn tokenizeTurn(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
    const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
    const newline = (try encoder.encode("\n"))[0];

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.appendSlice(allocator, &.{ im_end, newline, im_start, user, newline });
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.appendSlice(allocator, &.{ im_end, newline });
    try tokens.appendSlice(allocator, &.{ im_start, assistant, newline });

    return tokens.toOwnedSlice(allocator);
}
