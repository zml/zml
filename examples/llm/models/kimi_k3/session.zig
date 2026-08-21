const std = @import("std");

const zml = @import("zml");

const chat_template = @import("chat_template.zig");
const inference = @import("inference.zig");
const kda = @import("kda.zig");
const layer = @import("layer.zig");
const mla = @import("mla.zig");
const model = @import("model.zig");

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    tokenizer: zml.tokenizer.Tokenizer,
    compiled: *const inference.CompiledModel,
    buffers: *model.Buffers,
    kda_caches: []zml.Bufferized(kda.Cache),
    mla_caches: []zml.Bufferized(mla.SessionCache),
    position: usize = 0,
    last_generated_token: u32 = 0,

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        tokenizer: zml.tokenizer.Tokenizer,
        compiled: *const inference.CompiledModel,
        buffers: *model.Buffers,
    ) !Session {
        if (platform.target != .cuda) return error.NvidiaCudaRequired;
        var kda_count: usize = 0;
        var mla_count: usize = 0;
        for (compiled.loaded_model.inner.layers) |planned| switch (planned.kind()) {
            .kda_dense, .kda_moe => kda_count += 1,
            .mla_moe => mla_count += 1,
        };
        const kda_caches = try allocator.alloc(zml.Bufferized(kda.Cache), kda_count);
        errdefer allocator.free(kda_caches);
        const mla_caches = try allocator.alloc(zml.Bufferized(mla.SessionCache), mla_count);
        errdefer allocator.free(mla_caches);
        var self: Session = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .tokenizer = tokenizer,
            .compiled = compiled,
            .buffers = buffers,
            .kda_caches = kda_caches,
            .mla_caches = mla_caches,
        };
        var initialized_kda: usize = 0;
        var initialized_mla: usize = 0;
        errdefer {
            for (self.kda_caches[0..initialized_kda]) |*cache| zml.Buffer.deinitAll(kda.Cache, cache);
            for (self.mla_caches[0..initialized_mla]) |*cache| zml.Buffer.deinitAll(mla.SessionCache, cache);
        }
        for (self.kda_caches) |*cache| {
            cache.* = try buffers.loader.zeroKdaCache();
            initialized_kda += 1;
        }
        for (self.mla_caches) |*cache| {
            cache.* = try buffers.loader.zeroMlaCache(compiled.params.seqlen);
            initialized_mla += 1;
        }
        return self;
    }

    pub fn deinit(self: *Session) void {
        for (self.kda_caches) |*cache| zml.Buffer.deinitAll(kda.Cache, cache);
        for (self.mla_caches) |*cache| zml.Buffer.deinitAll(mla.SessionCache, cache);
        self.allocator.free(self.kda_caches);
        self.allocator.free(self.mla_caches);
    }

    pub fn reset(self: *Session) !void {
        for (self.kda_caches) |*cache| {
            zml.Buffer.deinitAll(kda.Cache, cache);
            cache.* = try self.buffers.loader.zeroKdaCache();
        }
        for (self.mla_caches) |*cache| {
            zml.Buffer.deinitAll(mla.SessionCache, cache);
            cache.* = try self.buffers.loader.zeroMlaCache(self.compiled.params.seqlen);
        }
        self.position = 0;
        self.last_generated_token = 0;
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return chat_template.tokenizePrompt(allocator, self.tokenizer, prompt);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return chat_template.tokenizeTurn(allocator, self.tokenizer, prompt);
    }

    fn sameBufferHandle(a: zml.Buffer, b: zml.Buffer) bool {
        if (a._shards.len != b._shards.len) return false;
        for (a._shards.constSlice(), b._shards.constSlice()) |a_shard, b_shard| {
            if (a_shard != b_shard) return false;
        }
        return true;
    }

    fn replaceBuffer(dst: *zml.Buffer, src: *zml.Buffer) void {
        if (!sameBufferHandle(dst.*, src.*)) dst.deinit();
        dst.* = src.*;
    }

    fn runToken(self: *Session, token_id: u32) !u32 {
        if (self.position >= self.compiled.params.seqlen) return error.CacheCapacityExceeded;
        var token = token_id;
        var token_buffer = try zml.Buffer.fromBytes(
            self.io,
            self.platform,
            self.compiled.params.decode_tokens.shape(),
            self.compiled.params.shardings.model,
            @ptrCast(&token),
        );
        defer token_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(
            self.io,
            self.platform,
            @as(u32, @intCast(self.position)),
            .u32,
        );
        defer token_index_buffer.deinit();
        var first_block_index_buffer = try zml.Buffer.scalar(
            self.io,
            self.platform,
            @as(u32, 0),
            .u32,
        );
        defer first_block_index_buffer.deinit();
        const active_bytes = try self.allocator.alloc(u8, self.compiled.params.source_slots);
        defer self.allocator.free(active_bytes);
        @memset(active_bytes, 0);
        active_bytes[0] = 1;
        var active_buffer = try zml.Buffer.fromBytes(
            self.io,
            self.platform,
            self.compiled.params.active_blocks.shape(),
            self.compiled.params.shardings.model,
            active_bytes,
        );
        defer active_buffer.deinit();

        var embedding_args = try self.compiled.embedding.args(self.allocator);
        defer embedding_args.deinit(self.allocator);
        var embedding_results = try self.compiled.embedding.results(self.allocator);
        defer embedding_results.deinit(self.allocator);
        embedding_args.set(.{ token_buffer, self.buffers.head.embedding });
        self.compiled.embedding.callOpts(self.io, embedding_args, &embedding_results, .{ .wait = true });
        var hidden = embedding_results.get(zml.Buffer);
        errdefer hidden.deinit();

        var layer0_args = try self.compiled.layer0.args(self.allocator);
        defer layer0_args.deinit(self.allocator);
        var layer0_results = try self.compiled.layer0.results(self.allocator);
        defer layer0_results.deinit(self.allocator);
        layer0_args.set(.{ hidden, self.buffers.layer0, self.kda_caches[0] });
        self.compiled.layer0.callOpts(self.io, layer0_args, &layer0_results, .{ .wait = true });
        var layer0_result: zml.Bufferized(layer.Layer0CompactResult) = undefined;
        layer0_results.fill(.{&layer0_result});
        replaceBuffer(&hidden, &layer0_result.output);
        replaceBuffer(&self.kda_caches[0].q_conv, &layer0_result.cache.q_conv);
        replaceBuffer(&self.kda_caches[0].k_conv, &layer0_result.cache.k_conv);
        replaceBuffer(&self.kda_caches[0].v_conv, &layer0_result.cache.v_conv);
        replaceBuffer(&self.kda_caches[0].recurrent_state, &layer0_result.cache.recurrent_state);
        var blocks = if (self.compiled.params.source_slots == 1)
            layer0_result.block_residual
        else expanded: {
            var empty = try self.buffers.loader.zeroBlocks(self.compiled.params.source_slots);
            defer empty.deinit();
            var block_args = try self.compiled.block_update.args(self.allocator);
            defer block_args.deinit(self.allocator);
            var block_results = try self.compiled.block_update.results(self.allocator);
            defer block_results.deinit(self.allocator);
            block_args.set(.{ empty, layer0_result.block_residual, first_block_index_buffer });
            self.compiled.block_update.callOpts(self.io, block_args, &block_results, .{ .wait = true });
            const expanded_blocks = block_results.get(zml.Buffer);
            layer0_result.block_residual.deinit();
            break :expanded expanded_blocks;
        };
        defer blocks.deinit();

        var kda_ordinal: usize = 1;
        var mla_ordinal: usize = 0;
        const block_size: usize = @intCast(self.compiled.loaded_model.inner.config.text_config.attn_res_block_size);
        for (self.compiled.loaded_model.inner.layers[1..]) |planned| {
            const layer_index = planned.weights().logical_index;
            switch (planned.kind()) {
                .kda_dense => return error.UnsupportedSecondDenseKimiK3Layer,
                .kda_moe => {
                    var streamed_weights: zml.Bufferized(layer.KdaMoeWeights) = undefined;
                    var streamed = false;
                    const weights = self.buffers.residentKdaMoe(layer_index) orelse temporary: {
                        streamed_weights = try self.buffers.loader.loadKdaMoe(layer_index);
                        streamed = true;
                        break :temporary &streamed_weights;
                    };
                    defer if (streamed) zml.Buffer.deinitAll(layer.KdaMoeWeights, &streamed_weights);
                    if (layer_index % block_size == 0) {
                        const exe = self.compiled.kda_moe_boundary orelse return error.MissingKdaMoeBoundaryExecutable;
                        var block_index_buffer = try zml.Buffer.scalar(
                            self.io,
                            self.platform,
                            @as(u32, @intCast(layer_index / block_size)),
                            .u32,
                        );
                        defer block_index_buffer.deinit();
                        var args = try exe.args(self.allocator);
                        defer args.deinit(self.allocator);
                        var results = try exe.results(self.allocator);
                        defer results.deinit(self.allocator);
                        args.set(.{ hidden, blocks, active_buffer, block_index_buffer, weights.*, self.kda_caches[kda_ordinal] });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.KdaMoeBoundaryCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&hidden, &actual.layer.output);
                        replaceBuffer(&blocks, &actual.block_sources);
                        replaceBuffer(&active_buffer, &actual.active_blocks);
                        replaceBuffer(&self.kda_caches[kda_ordinal].q_conv, &actual.layer.cache.q_conv);
                        replaceBuffer(&self.kda_caches[kda_ordinal].k_conv, &actual.layer.cache.k_conv);
                        replaceBuffer(&self.kda_caches[kda_ordinal].v_conv, &actual.layer.cache.v_conv);
                        replaceBuffer(&self.kda_caches[kda_ordinal].recurrent_state, &actual.layer.cache.recurrent_state);
                    } else {
                        const exe = self.compiled.kda_moe orelse return error.MissingKdaMoeExecutable;
                        var args = try exe.args(self.allocator);
                        defer args.deinit(self.allocator);
                        var results = try exe.results(self.allocator);
                        defer results.deinit(self.allocator);
                        args.set(.{ hidden, blocks, active_buffer, weights.*, self.kda_caches[kda_ordinal] });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.KdaMoeCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&hidden, &actual.output);
                        replaceBuffer(&self.kda_caches[kda_ordinal].q_conv, &actual.cache.q_conv);
                        replaceBuffer(&self.kda_caches[kda_ordinal].k_conv, &actual.cache.k_conv);
                        replaceBuffer(&self.kda_caches[kda_ordinal].v_conv, &actual.cache.v_conv);
                        replaceBuffer(&self.kda_caches[kda_ordinal].recurrent_state, &actual.cache.recurrent_state);
                    }
                    kda_ordinal += 1;
                },
                .mla_moe => {
                    var streamed_weights: zml.Bufferized(layer.MlaMoeWeights) = undefined;
                    var streamed = false;
                    const weights = self.buffers.residentMlaMoe(layer_index) orelse temporary: {
                        streamed_weights = try self.buffers.loader.loadMlaMoe(layer_index);
                        streamed = true;
                        break :temporary &streamed_weights;
                    };
                    defer if (streamed) zml.Buffer.deinitAll(layer.MlaMoeWeights, &streamed_weights);
                    if (layer_index % block_size == 0) {
                        const exe = self.compiled.mla_moe_boundary orelse return error.MissingMlaMoeBoundaryExecutable;
                        var block_index_buffer = try zml.Buffer.scalar(
                            self.io,
                            self.platform,
                            @as(u32, @intCast(layer_index / block_size)),
                            .u32,
                        );
                        defer block_index_buffer.deinit();
                        var args = try exe.args(self.allocator);
                        defer args.deinit(self.allocator);
                        var results = try exe.results(self.allocator);
                        defer results.deinit(self.allocator);
                        args.set(.{ hidden, blocks, active_buffer, block_index_buffer, weights.*, self.mla_caches[mla_ordinal], token_index_buffer });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.MlaMoeBoundaryCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&hidden, &actual.layer.output);
                        replaceBuffer(&blocks, &actual.block_sources);
                        replaceBuffer(&active_buffer, &actual.active_blocks);
                        replaceBuffer(&self.mla_caches[mla_ordinal].compressed, &actual.layer.cache.compressed);
                        replaceBuffer(&self.mla_caches[mla_ordinal].extra_key, &actual.layer.cache.extra_key);
                    } else {
                        const exe = self.compiled.mla_moe orelse return error.MissingMlaMoeExecutable;
                        var args = try exe.args(self.allocator);
                        defer args.deinit(self.allocator);
                        var results = try exe.results(self.allocator);
                        defer results.deinit(self.allocator);
                        args.set(.{ hidden, blocks, active_buffer, weights.*, self.mla_caches[mla_ordinal], token_index_buffer });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.MlaMoeCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&hidden, &actual.output);
                        replaceBuffer(&self.mla_caches[mla_ordinal].compressed, &actual.cache.compressed);
                        replaceBuffer(&self.mla_caches[mla_ordinal].extra_key, &actual.cache.extra_key);
                    }
                    mla_ordinal += 1;
                },
            }
        }

        var head_args = try self.compiled.head.args(self.allocator);
        defer head_args.deinit(self.allocator);
        var head_results = try self.compiled.head.results(self.allocator);
        defer head_results.deinit(self.allocator);
        head_args.set(.{
            hidden,
            blocks,
            active_buffer,
            self.buffers.head.output_res_norm,
            self.buffers.head.output_res_projection,
            self.buffers.head.final_norm,
            self.buffers.head.lm_head,
        });
        self.compiled.head.callOpts(self.io, head_args, &head_results, .{ .wait = true });
        var head: zml.Bufferized(layer.SessionHeadResult) = undefined;
        head_results.fill(.{&head});
        const greedy_i64 = try head.greedy_token.getValue(i64, self.io);
        if (greedy_i64 < 0 or greedy_i64 >= 163840) return error.InvalidKimiK3GreedyToken;
        const greedy: u32 = @intCast(greedy_i64);
        zml.Buffer.deinitAll(layer.SessionHeadResult, &head);
        hidden.deinit();
        self.position += 1;
        return greedy;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        if (all_tokens.len == 0) return error.EmptyKimiK3Prompt;
        if (all_tokens.len > self.compiled.params.seqlen) return error.PromptTooLong;
        try self.reset();
        for (all_tokens) |token_id| {
            self.last_generated_token = try self.runToken(token_id);
        }
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), writer: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();
        const output = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(output);
        const eos = self.tokenizer.tokenId("<|end_of_msg|>") orelse return error.KimiK3MissingEosToken;
        var token_id = self.last_generated_token;
        while (token_id != eos and all_tokens.items.len < self.compiled.params.seqlen) {
            try writer.writeAll(try decoder.feedOne(token_id, output));
            try writer.flush();
            try all_tokens.append(self.allocator, token_id);
            token_id = try self.runToken(token_id);
        }
        self.last_generated_token = token_id;
        try writer.writeAll(try decoder.finalize(output));
        try writer.flush();
    }
};
