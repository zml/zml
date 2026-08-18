const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    config: *const model.Config,
    cache_buffers: zml.Bufferized(model.Cache),
    prefill_moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    decode_moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    layer_index_buffers: []zml.Buffer,
    generated_token_slice: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,
    seqlen: u32,
    think_start: ?u32,
    think_end: ?u32,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled_model: *const inference.CompiledModel,
        model_buffers: *model.Buffers,
    ) !Session {
        const parameters = compiled_model.params;
        var cache_buffers = try parameters.cache.initBuffers(io, platform, parameters.shardings.model);
        errdefer model.Cache.deinitBuffers(&cache_buffers);

        var prefill_moe_metadata_buffers = try parameters.prefill_moe_metadata.initBuffer(io, platform);
        errdefer zml.moe.Metadata.deinitBuffer(&prefill_moe_metadata_buffers);
        var decode_moe_metadata_buffers = try parameters.decode_moe_metadata.initBuffer(io, platform);
        errdefer zml.moe.Metadata.deinitBuffer(&decode_moe_metadata_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        const layer_count = compiled_model.loaded_model.inner.layers.len;
        const layer_index_buffers = try allocator.alloc(zml.Buffer, layer_count);
        var initialized_layer_indices: usize = 0;
        errdefer {
            for (layer_index_buffers[0..initialized_layer_indices]) |*buffer| buffer.deinit();
            allocator.free(layer_index_buffers);
        }
        for (layer_index_buffers, 0..) |*buffer, layer_index| {
            buffer.* = try .scalar(io, platform, @as(u32, @intCast(layer_index)), .u32);
            initialized_layer_indices += 1;
        }

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .config = &compiled_model.loaded_model.parsed_config.value,
            .cache_buffers = cache_buffers,
            .prefill_moe_metadata_buffers = prefill_moe_metadata_buffers,
            .decode_moe_metadata_buffers = decode_moe_metadata_buffers,
            .rng_buffers = rng_buffers,
            .layer_index_buffers = layer_index_buffers,
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .tokenizer = tokenizer,
            .seqlen = parameters.seqlen,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        model.Cache.deinitBuffers(&self.cache_buffers);
        zml.moe.Metadata.deinitBuffer(&self.prefill_moe_metadata_buffers);
        zml.moe.Metadata.deinitBuffer(&self.decode_moe_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        for (self.layer_index_buffers) |*buffer| buffer.deinit();
        self.allocator.free(self.layer_index_buffers);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, false);
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const token_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen }, .u32);
        var token_slice = try zml.Slice.alloc(self.allocator, token_shape);
        defer token_slice.free(self.allocator);
        @memset(token_slice.items(u32), self.config.pad_token_id);
        @memcpy(token_slice.items(u32)[0..all_tokens.len], all_tokens);

        var token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, token_slice, .replicated);
        defer token_buffer.deinit();
        var hidden_buffer = try zml.Buffer.uninitialized(
            self.io,
            self.platform,
            hiddenShape(self.compiled_model, self.seqlen),
            .replicated,
            .{},
        );
        defer hidden_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();

        var embedding_args = try self.compiled_model.prefill.embedding.args(self.allocator);
        defer embedding_args.deinit(self.allocator);
        var embedding_results = try self.compiled_model.prefill.embedding.results(self.allocator);
        defer embedding_results.deinit(self.allocator);
        embedding_args.set(.{ self.model_buffers.embed_tokens, token_buffer });
        self.compiled_model.prefill.embedding.call(embedding_args, &embedding_results);
        embedding_results.fill(.{&hidden_buffer});

        try self.runLayers(
            &self.compiled_model.prefill,
            &hidden_buffer,
            token_index_buffer,
            self.prefill_moe_metadata_buffers,
        );

        var sampling_args = try self.compiled_model.prefill.sampling.args(self.allocator);
        defer sampling_args.deinit(self.allocator);
        var sampling_results = try self.compiled_model.prefill.sampling.results(self.allocator);
        defer sampling_results.deinit(self.allocator);
        sampling_args.set(.{ samplerBuffers(self.model_buffers), hidden_buffer, self.rng_buffers });
        self.compiled_model.prefill.sampling.call(sampling_args, &sampling_results);
        sampling_results.fill(.{ &token_buffer, &self.rng_buffers });

        try token_buffer.toSlice(self.io, token_slice);
        self.generated_token_slice.items(u32)[0] = token_slice.items(u32)[all_tokens.len - 1];
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();
        const decoded_bytes = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(decoded_bytes);

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, .replicated);
        defer current_token_buffer.deinit();
        var hidden_buffer = try zml.Buffer.uninitialized(
            self.io,
            self.platform,
            hiddenShape(self.compiled_model, 1),
            .replicated,
            .{},
        );
        defer hidden_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(
            self.io,
            self.platform,
            @as(u32, @intCast(all_tokens.items.len)),
            .u32,
        );
        defer token_index_buffer.deinit();

        var embedding_args = try self.compiled_model.decode.embedding.args(self.allocator);
        defer embedding_args.deinit(self.allocator);
        var embedding_results = try self.compiled_model.decode.embedding.results(self.allocator);
        defer embedding_results.deinit(self.allocator);
        var sampling_args = try self.compiled_model.decode.sampling.args(self.allocator);
        defer sampling_args.deinit(self.allocator);
        var sampling_results = try self.compiled_model.decode.sampling.results(self.allocator);
        defer sampling_results.deinit(self.allocator);

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (isEosToken(self.config, token_id)) break :generation;

            const decoded = try decoder.feedOne(token_id, decoded_bytes);
            if (self.think_start) |think_start| if (token_id == think_start) try stdout.writeAll("\x1b[2m");
            try stdout.writeAll(decoded);
            if (self.think_end) |think_end| if (token_id == think_end) try stdout.writeAll("\x1b[0m");
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            embedding_args.set(.{ self.model_buffers.embed_tokens, current_token_buffer });
            self.compiled_model.decode.embedding.call(embedding_args, &embedding_results);
            embedding_results.fill(.{&hidden_buffer});

            try self.runLayers(
                &self.compiled_model.decode,
                &hidden_buffer,
                token_index_buffer,
                self.decode_moe_metadata_buffers,
            );

            sampling_args.set(.{
                samplerBuffers(self.model_buffers),
                hidden_buffer,
                self.rng_buffers,
                token_index_buffer,
            });
            self.compiled_model.decode.sampling.call(sampling_args, &sampling_results);
            sampling_results.fill(.{ &current_token_buffer, &self.rng_buffers, &token_index_buffer });
            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(decoded_bytes));
        try stdout.flush();
    }

    fn runLayers(
        self: *Session,
        executables: *const inference.PhaseExecutables,
        hidden_buffer: *zml.Buffer,
        token_index_buffer: zml.Buffer,
        moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    ) !void {
        var dense_full_args = try executables.dense_full_layer.args(self.allocator);
        defer dense_full_args.deinit(self.allocator);
        var dense_full_results = try executables.dense_full_layer.results(self.allocator);
        defer dense_full_results.deinit(self.allocator);
        var sparse_full_args = try executables.sparse_full_layer.args(self.allocator);
        defer sparse_full_args.deinit(self.allocator);
        var sparse_full_results = try executables.sparse_full_layer.results(self.allocator);
        defer sparse_full_results.deinit(self.allocator);
        var sparse_shared_args = try executables.sparse_shared_layer.args(self.allocator);
        defer sparse_shared_args.deinit(self.allocator);
        var sparse_shared_results = try executables.sparse_shared_layer.results(self.allocator);
        defer sparse_shared_results.deinit(self.allocator);

        var previous_topk: ?zml.Buffer = null;
        defer if (previous_topk) |*buffer| buffer.deinit();

        for (self.model_buffers.layers, 0..) |layer_buffers, layer_index| {
            const indexer_type = self.config.indexer_types[layer_index];
            const mlp_type = self.config.mlp_layer_types[layer_index];
            switch (indexer_type) {
                .full => {
                    const exe, const args, const results = switch (mlp_type) {
                        .dense => .{ &executables.dense_full_layer, &dense_full_args, &dense_full_results },
                        .sparse => .{ &executables.sparse_full_layer, &sparse_full_args, &sparse_full_results },
                    };
                    args.set(.{
                        layer_buffers,
                        hidden_buffer.*,
                        token_index_buffer,
                        self.cache_buffers,
                        self.layer_index_buffers[layer_index],
                        moe_metadata_buffers,
                    });
                    exe.call(args.*, results);
                    var new_hidden, var new_cache, var new_topk = results.get(struct {
                        zml.Buffer,
                        zml.Bufferized(model.Cache),
                        zml.Buffer,
                    });
                    replaceBuffer(hidden_buffer, &new_hidden);
                    replaceCache(&self.cache_buffers, &new_cache);
                    if (previous_topk) |*topk| {
                        replaceBuffer(topk, &new_topk);
                    } else {
                        previous_topk = new_topk;
                    }
                },
                .shared => {
                    const topk = previous_topk orelse return error.SharedIndexerWithoutPreviousTopK;
                    sparse_shared_args.set(.{
                        layer_buffers,
                        hidden_buffer.*,
                        token_index_buffer,
                        self.cache_buffers,
                        self.layer_index_buffers[layer_index],
                        topk,
                        moe_metadata_buffers,
                    });
                    executables.sparse_shared_layer.call(sparse_shared_args, &sparse_shared_results);
                    var new_hidden, var new_cache = sparse_shared_results.get(struct {
                        zml.Buffer,
                        zml.Bufferized(model.Cache),
                    });
                    replaceBuffer(hidden_buffer, &new_hidden);
                    replaceCache(&self.cache_buffers, &new_cache);
                },
            }
        }
    }
};

fn hiddenShape(compiled_model: *const inference.CompiledModel, token_count: u32) zml.Shape {
    const mdl = compiled_model.loaded_model.inner;
    return zml.Shape.init(
        .{ .b = 1, .s = token_count, .d = mdl.config.hidden_size },
        mdl.embed_tokens.weight.dtype(),
    ).withPartitioning(.{ .b = .replicated, .s = .replicated, .d = .replicated });
}

fn samplerBuffers(buffers: *const model.Buffers) zml.Bufferized(model.Sampler) {
    return .{
        .norm = buffers.norm,
        .lm_head = buffers.lm_head,
    };
}

fn isEosToken(config: *const model.Config, token: u32) bool {
    return switch (config.eos_token_id.value) {
        .int => |value| token == value,
        .ints => |values| std.mem.indexOfScalar(u32, values, token) != null,
    };
}

fn tokenizeChatPrompt(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    first_turn: bool,
) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len + 16);
    errdefer tokens.deinit(allocator);
    try appendEncoded(&tokens, allocator, &encoder, if (first_turn) "[gMASK]<sop><|user|>" else "\n<|user|>");
    try appendEncoded(&tokens, allocator, &encoder, prompt);
    try appendEncoded(&tokens, allocator, &encoder, "\n<|assistant|><think>");
    return tokens.toOwnedSlice(allocator);
}

fn appendEncoded(
    tokens: *std.ArrayList(u32),
    allocator: std.mem.Allocator,
    encoder: *zml.tokenizer.Tokenizer.Encoder,
    text: []const u8,
) !void {
    const encoded = try encoder.encodeAlloc(allocator, text);
    defer allocator.free(encoded);
    try tokens.appendSlice(allocator, encoded);
}

fn replaceCache(destination: *zml.Bufferized(model.Cache), source: *zml.Bufferized(model.Cache)) void {
    replaceBuffer(&destination.k, &source.k);
    replaceBuffer(&destination.v, &source.v);
    replaceBuffer(&destination.indexer_k, &source.indexer_k);
}

fn replaceBuffer(destination: *zml.Buffer, source: *zml.Buffer) void {
    if (!sameBufferHandle(destination.*, source.*)) destination.deinit();
    destination.* = source.*;
}

fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
    if (a._shards.len != b._shards.len) return false;
    for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
        if (a_shard != b_shard) return false;
    }
    return true;
}
