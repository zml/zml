const std = @import("std");

const zml = @import("zml");
const Buffer = zml.Buffer;
const attention = zml.attention.attention;

const common = @import("../common.zig");
const Shardings = common.Shardings;
const SessionOptions = common.SessionOptions;
const inference = @import("inference.zig");
const model = @import("model.zig");
const repository = @import("repository.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model_buffers: *model.Buffers,
    exe: inference.Inference,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    attention_metadata_buffers: zml.Bufferized(attention.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    generated_token_slice: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,
    config: model.Config,
    seqlen: u32,

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
        if (opts.single) return error.SingleKernelNotSupported;

        const params = inference.CompilationOptions.init(repo.inner, repo.parsed_config.value, opts.seqlen, opts.backend);
        var exe = try inference.Inference.init(allocator, io, platform, repo.inner, params, shardings, progress);
        errdefer exe.deinit();

        var kv_cache_buffers = try params.kv_cache.initBuffer(io, platform, shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        var attention_metadata_buffers = try params.attention_metadata.initBuffer(io, platform, shardings.model);
        errdefer attention.Metadata.deinitBuffer(&attention_metadata_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, shardings.replicated);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .exe = exe,
            .kv_cache_buffers = kv_cache_buffers,
            .attention_metadata_buffers = attention_metadata_buffers,
            .rng_buffers = rng_buffers,
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32)),
            .tokenizer = tokenizer,
            .config = repo.parsed_config.value,
            .seqlen = opts.seqlen,
        };
    }

    pub fn deinit(self: *Session) void {
        self.exe.deinit();
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        attention.Metadata.deinitBuffer(&self.attention_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        var prefill_args = try self.exe.prefill_exe.args(self.allocator);
        defer prefill_args.deinit(self.allocator);

        var prefill_results = try self.exe.prefill_exe.results(self.allocator);
        defer prefill_results.deinit(self.allocator);

        const prefill_tokens_slice: zml.Slice = try .alloc(self.allocator, .init(.{self.seqlen}, .u32));
        defer prefill_tokens_slice.free(self.allocator);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

        var prefill_tokens_buffer: zml.Buffer = try .fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_pos_buffer = try zml.Buffer.scalar(self.io, self.platform, 0, .u32, replicated_sharding);
        defer prefill_token_pos_buffer.deinit();

        prefill_args.set(.{
            self.model_buffers,
            prefill_tokens_buffer,
            prefill_token_pos_buffer,
            &self.kv_cache_buffers,
            &self.rng_buffers,
            &self.attention_metadata_buffers,
        });
        self.exe.prefill_exe.call(prefill_args, &prefill_results);

        prefill_results.fill(.{ &prefill_tokens_buffer, &self.kv_cache_buffers, &self.rng_buffers });
        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);

        self.generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        var decode_args = try self.exe.decode_exe.args(self.allocator);
        defer decode_args.deinit(self.allocator);

        var decode_results = try self.exe.decode_exe.results(self.allocator);
        defer decode_results.deinit(self.allocator);

        const replicated_sharding = try zml.sharding.replicatedSharding(self.platform);

        var current_token_buffer: Buffer = try .fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
        defer current_token_buffer.deinit();

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];

            if (isEosToken(self.config, token_id)) break :generation;

            if (try decoder.next(token_id)) |token| {
                try stdout.writeAll(token);
                try stdout.flush();
            }

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(all_tokens.items.len)}));
            var token_pos_buffer: Buffer = try .fromSlice(self.io, self.platform, token_pos_slice, replicated_sharding);
            defer token_pos_buffer.deinit();

            decode_args.set(.{
                self.model_buffers,
                current_token_buffer,
                token_pos_buffer,
                &self.kv_cache_buffers,
                &self.rng_buffers,
                &self.attention_metadata_buffers,
            });
            self.exe.decode_exe.call(decode_args, &decode_results);

            decode_results.fill(.{ &current_token_buffer, &self.kv_cache_buffers, &self.rng_buffers });
            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }
    }
};

fn isEosToken(config: model.Config, token_id: u32) bool {
    return switch (config.eos_token_id.value) {
        .int => |eos| token_id == eos,
        .ints => |eos_list| for (eos_list) |eos| {
            if (token_id == eos) break true;
        } else false,
    };
}
