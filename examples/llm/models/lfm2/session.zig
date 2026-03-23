const std = @import("std");

const zml = @import("zml");
const attention = zml.attention.attention;

const Shardings = @import("../common.zig").Shardings;
const inference = @import("inference.zig");
const model = @import("model.zig");
const repository = @import("repository.zig");

pub const SessionOptions = struct {
    seqlen: u32,
    backend: attention.Backend,
    single: bool,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_buffers: *model.Buffers,
    exe: inference.Inference,
    config: model.Config,
    seqlen: u32,
    cache_buffers: zml.Bufferized(model.Cache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
    rng_buf: zml.Bufferized(zml.Tensor.Rng),
    generated_token_slice: zml.Slice,
    token_pos_: u32,
    think_start: ?u32,
    think_end: ?u32,
    tokenizer: zml.tokenizer.Tokenizer,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        model_buffers: *model.Buffers,
        tokenizer: zml.tokenizer.Tokenizer,
        repo: repository.Repository,
        opts: SessionOptions,
        progress: *std.Progress.Node,
        shardings: Shardings,
    ) !Session {
        const params = inference.CompilationOptions.init(repo.inner, repo.parsed_config.value, opts.seqlen, opts.backend, opts.single);
        const exe = try inference.Inference.init(allocator, io, platform, repo.inner, params, opts.seqlen, progress, shardings);
        errdefer exe.deinit();

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .exe = exe,
            .tokenizer = tokenizer,
            .config = repo.parsed_config.value,
            .seqlen = opts.seqlen,
            .cache_buffers = try params.cache.initBuffers(allocator, io, platform, shardings.replicated),
            .attention_metadata_buffers = try params.attention_metadata.initBuffer(io, platform, shardings.model),
            .rng_buf = try zml.Tensor.Rng.initBuffer(platform, seed, io, shardings.replicated),
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .batch = 1, .seq = 1 }, .u32)),
            .token_pos_ = 0,
            .think_start = tokenizer.tokenToId("<think>") orelse unreachable,
            .think_end = tokenizer.tokenToId("</think>") orelse unreachable,
        };
    }

    pub fn deinit(self: *Session) void {
        self.exe.deinit();
        model.Cache.unloadBuffers(&self.cache_buffers);
        attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buf);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn remainingTokens(self: *const Session) u32 {
        return self.seqlen -| self.token_pos_;
    }

    pub fn tokenPos(self: *const Session) u32 {
        return self.token_pos_;
    }

    pub fn maxSeqLen(self: *const Session) u32 {
        return self.seqlen;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{ .batch = 1, .seq = self.seqlen }, .u32));
        defer tokens_slice.free(self.allocator);
        const tokens = tokens_slice.items(u32);
        @memset(tokens, self.config.pad_token_id);
        @memcpy(tokens[0..all_tokens.len], all_tokens);

        const sharding = try zml.sharding.replicatedSharding(self.platform);

        var tokens_buf: zml.Buffer = try .fromSlice(self.io, self.platform, tokens_slice, sharding);
        defer tokens_buf.deinit();

        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var tokens_pos_buf: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, sharding);
        defer tokens_pos_buf.deinit();

        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.len)}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice, sharding);
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
        self.generated_token_slice.items(u32)[0] = tokens_slice.items(u32)[all_tokens.len - 1];
        self.token_pos_ = @intCast(all_tokens.len);
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const sharding = try zml.sharding.replicatedSharding(self.platform);

        var current_token_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice, sharding);
        defer current_token_buffer.deinit();

        const actual_seq_len_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{0}));
        var actual_seq_len_buf: zml.Buffer = try .fromSlice(self.io, self.platform, actual_seq_len_slice, sharding);
        defer actual_seq_len_buf.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];

            if (token_id == self.config.eos_token_id) break :generation;

            if (try decoder.next(token_id)) |token| {
                if (self.think_start) |think_start| if (token_id == think_start) {
                    try stdout.writeAll("\x1b[2m");
                };
                try stdout.writeAll(token);
                if (self.think_end) |think_end| if (token_id == think_end) {
                    try stdout.writeAll("\x1b[0m");
                };
                try stdout.flush();
            }

            try all_tokens.append(self.allocator, token_id);
            if (self.remainingTokens() == 0) break :generation;

            const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{ .batch = 1 }, .u32), std.mem.sliceAsBytes(&[_]u32{self.token_pos_}));
            var token_pos_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, sharding);
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
            self.token_pos_ += 1;
        }
    }
};
