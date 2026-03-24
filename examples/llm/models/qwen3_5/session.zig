const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *inference.CompiledModel,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    tokenizer: zml.tokenizer.Tokenizer,
    step_token_slice: zml.Slice,
    generated_token_slice: zml.Slice,
    seqlen: u32,
    eos_token_id: u32,
    think_start: ?u32,
    think_end: ?u32,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *zml.Platform,
        model_buffers: *model.Buffers,
        tokenizer: zml.tokenizer.Tokenizer,
        mdl: model.Model,
        compiled_model: *inference.CompiledModel,
    ) !Session {
        const replicated_sharding = compiled_model.params.shardings.replicated;
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, replicated_sharding);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .kv_cache_buffers = kv_cache_buffers,
            .rng_buffers = rng_buffers,
            .tokenizer = tokenizer,
            .step_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .seqlen = compiled_model.params.seqlen,
            .eos_token_id = mdl.special_tokens.end_of_text_token_id,
            .think_start = tokenizer.tokenToId("<think>"),
            .think_end = tokenizer.tokenToId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.step_token_slice.free(self.allocator);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen }, .u32);
        const prefill_tokens_slice = try zml.Slice.alloc(self.allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding = self.compiled_model.params.shardings.replicated;

        var prefill_tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32, replicated_sharding);
        defer prefill_token_index_buffer.deinit();

        var args = try self.compiled_model.prefill_exe.args(self.allocator);
        defer args.deinit(self.allocator);

        var results = try self.compiled_model.prefill_exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(.{
            self.model_buffers,
            prefill_tokens_buffer,
            prefill_token_index_buffer,
            &self.kv_cache_buffers,
            &self.rng_buffers,
        });
        self.compiled_model.prefill_exe.call(args, &results);

        results.fill(.{ &prefill_tokens_buffer, &self.kv_cache_buffers, &self.rng_buffers });
        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);

        const generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        self.generated_token_slice.items(u32)[0] = generated_token;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (token_id == self.eos_token_id) break :generation;

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
            if (all_tokens.items.len >= self.seqlen) break :generation;

            try self.runDecodeStep(token_id, @intCast(all_tokens.items.len));
        }
    }

    fn runDecodeStep(self: *Session, token_id: u32, token_index: u32) !void {
        self.step_token_slice.items(u32)[0] = token_id;

        const replicated_sharding = self.compiled_model.params.shardings.replicated;

        var token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.step_token_slice, replicated_sharding);
        defer token_buffer.deinit();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, token_index, .u32, replicated_sharding);
        defer token_index_buffer.deinit();

        var args = try self.compiled_model.decode_exe.args(self.allocator);
        defer args.deinit(self.allocator);

        var results = try self.compiled_model.decode_exe.results(self.allocator);
        defer results.deinit(self.allocator);

        args.set(.{
            self.model_buffers,
            token_buffer,
            token_index_buffer,
            &self.kv_cache_buffers,
            &self.rng_buffers,
        });
        self.compiled_model.decode_exe.call(args, &results);

        results.fill(.{ &token_buffer, &self.kv_cache_buffers, &self.rng_buffers });
        try token_buffer.toSlice(self.io, self.generated_token_slice);
    }
};
