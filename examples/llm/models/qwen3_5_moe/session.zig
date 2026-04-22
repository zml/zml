const std = @import("std");

const zml = @import("zml");

const inference = @import("inference.zig");
const model = @import("model.zig");

const LayerIndexBuffer = union(enum) {
    self_attn: zml.Buffer,
    linear_attn: zml.Buffer,
};

pub const Session = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model_buffers: *model.Buffers,
    compiled_model: *const inference.CompiledModel,
    kv_cache_buffers: zml.Bufferized(model.KvCache),
    prefill_moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    decode_moe_metadata_buffers: zml.Bufferized(zml.moe.Metadata),
    rng_buffers: zml.Bufferized(zml.Tensor.Rng),
    layer_index_buffers: []LayerIndexBuffer,
    decode_token_index_buffer: zml.Buffer,
    self_attn_layers_caches: []zml.Bufferized(model.KvCache.SelfAttnCache),
    linear_attn_layers_caches: []zml.Bufferized(model.KvCache.GatedDeltaNetCache),
    step_token_slice: zml.Slice,
    generated_token_slice: zml.Slice,
    tokenizer: zml.tokenizer.Tokenizer,
    seqlen: u32,
    eos_token_id: u32,
    special_tokens: model.Model.SpecialTokens,
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
        const shardings = compiled_model.params.shardings;
        var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, shardings.model);
        errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

        var prefill_moe_metadata_buffers = try compiled_model.params.prefill_moe_metadata.initBuffer(io, platform);
        errdefer zml.moe.Metadata.deinitBuffer(&prefill_moe_metadata_buffers);

        var decode_moe_metadata_buffers = try compiled_model.params.decode_moe_metadata.initBuffer(io, platform);
        errdefer zml.moe.Metadata.deinitBuffer(&decode_moe_metadata_buffers);

        const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
        var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, shardings.replicated);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

        const replicated_sharding = shardings.replicated;
        const layer_types = compiled_model.loaded_model.inner.config.text_config.layer_types;
        var layer_index_buffers = try allocator.alloc(LayerIndexBuffer, compiled_model.loaded_model.inner.text_model.layers.len);
        errdefer {
            for (layer_index_buffers) |*layer_index_buffer| {
                switch (layer_index_buffer.*) {
                    .self_attn => |*buffer| buffer.deinit(),
                    .linear_attn => |*buffer| buffer.deinit(),
                }
            }
            allocator.free(layer_index_buffers);
        }

        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (layer_types, 0..) |layer_type, layer_index| {
            switch (layer_type) {
                .full_attention => {
                    const layer_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(self_attn_layer_index)), .u32, replicated_sharding);
                    layer_index_buffers[layer_index] = .{ .self_attn = layer_index_buffer };
                    self_attn_layer_index += 1;
                },
                .linear_attention => {
                    const layer_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(linear_attn_layer_index)), .u32, replicated_sharding);
                    layer_index_buffers[layer_index] = .{ .linear_attn = layer_index_buffer };
                    linear_attn_layer_index += 1;
                },
            }
        }

        var decode_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32, replicated_sharding);
        errdefer decode_token_index_buffer.deinit();

        var self_attn_layers_caches = try allocator.alloc(zml.Bufferized(model.KvCache.SelfAttnCache), self_attn_layer_index);
        errdefer allocator.free(self_attn_layers_caches);

        var linear_attn_layers_caches = try allocator.alloc(zml.Bufferized(model.KvCache.GatedDeltaNetCache), linear_attn_layer_index);
        errdefer allocator.free(linear_attn_layers_caches);

        var self_attn_cache_index: usize = 0;
        var linear_attn_cache_index: usize = 0;
        for (layer_types, 0..) |layer_type, layer_index| {
            switch (layer_type) {
                .full_attention => {
                    const layer_index_buffer = switch (layer_index_buffers[layer_index]) {
                        .self_attn => |buffer| buffer,
                        .linear_attn => unreachable,
                    };
                    self_attn_layers_caches[self_attn_cache_index] = .{
                        .k = kv_cache_buffers.self_attn.k,
                        .v = kv_cache_buffers.self_attn.v,
                        .layer_index = layer_index_buffer,
                    };
                    self_attn_cache_index += 1;
                },
                .linear_attention => {
                    const layer_index_buffer = switch (layer_index_buffers[layer_index]) {
                        .linear_attn => |buffer| buffer,
                        .self_attn => unreachable,
                    };
                    linear_attn_layers_caches[linear_attn_cache_index] = .{
                        .conv_state = kv_cache_buffers.gated_delta_net.conv_state,
                        .recurrent_state = kv_cache_buffers.gated_delta_net.recurrent_state,
                        .layer_index = layer_index_buffer,
                    };
                    linear_attn_cache_index += 1;
                },
            }
        }

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .model_buffers = model_buffers,
            .compiled_model = compiled_model,
            .kv_cache_buffers = kv_cache_buffers,
            .prefill_moe_metadata_buffers = prefill_moe_metadata_buffers,
            .decode_moe_metadata_buffers = decode_moe_metadata_buffers,
            .rng_buffers = rng_buffers,
            .layer_index_buffers = layer_index_buffers,
            .decode_token_index_buffer = decode_token_index_buffer,
            .self_attn_layers_caches = self_attn_layers_caches,
            .linear_attn_layers_caches = linear_attn_layers_caches,
            .step_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32)),
            .tokenizer = tokenizer,
            .seqlen = compiled_model.params.seqlen,
            .eos_token_id = compiled_model.loaded_model.inner.special_tokens.end_of_text_token_id,
            .special_tokens = compiled_model.loaded_model.inner.special_tokens,
            .think_start = tokenizer.tokenId("<think>"),
            .think_end = tokenizer.tokenId("</think>"),
        };
    }

    pub fn deinit(self: *Session) void {
        model.KvCache.deinitBuffer(&self.kv_cache_buffers);
        zml.moe.Metadata.deinitBuffer(&self.prefill_moe_metadata_buffers);
        zml.moe.Metadata.deinitBuffer(&self.decode_moe_metadata_buffers);
        zml.Tensor.Rng.deinitBuffer(&self.rng_buffers);
        self.decode_token_index_buffer.deinit();
        for (self.layer_index_buffers) |*layer_index_buffer| {
            switch (layer_index_buffer.*) {
                .self_attn => |*buffer| buffer.deinit(),
                .linear_attn => |*buffer| buffer.deinit(),
            }
        }
        self.allocator.free(self.layer_index_buffers);
        self.allocator.free(self.self_attn_layers_caches);
        self.allocator.free(self.linear_attn_layers_caches);
        self.step_token_slice.free(self.allocator);
        self.generated_token_slice.free(self.allocator);
    }

    pub fn tokenizePrompt(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, true);
    }

    pub fn tokenizeTurn(self: *const Session, allocator: std.mem.Allocator, prompt: []const u8) ![]const u32 {
        return tokenizeChatPrompt(allocator, self.tokenizer, prompt, self.special_tokens, false);
    }

    fn storeSelfAttnLayerCache(
        self: *Session,
        dense_layer_index: usize,
        layer_index: usize,
        layer_cache: zml.Bufferized(model.KvCache.SelfAttnCache),
    ) void {
        self.kv_cache_buffers.self_attn.k = layer_cache.k;
        self.kv_cache_buffers.self_attn.v = layer_cache.v;
        self.self_attn_layers_caches[dense_layer_index] = layer_cache;
        self.layer_index_buffers[layer_index] = .{ .self_attn = layer_cache.layer_index };
    }

    fn storeLinearAttnLayerCache(
        self: *Session,
        dense_layer_index: usize,
        layer_index: usize,
        layer_cache: zml.Bufferized(model.KvCache.GatedDeltaNetCache),
    ) void {
        self.kv_cache_buffers.gated_delta_net.conv_state = layer_cache.conv_state;
        self.kv_cache_buffers.gated_delta_net.recurrent_state = layer_cache.recurrent_state;
        self.linear_attn_layers_caches[dense_layer_index] = layer_cache;
        self.layer_index_buffers[layer_index] = .{ .linear_attn = layer_cache.layer_index };
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        const hidden_size = self.compiled_model.loaded_model.inner.config.text_config.hidden_size;
        const model_dtype = self.compiled_model.loaded_model.inner.text_model.embed_tokens.weight.dtype();

        const prefill_tokens_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen }, .u32);
        const prefill_hidden_shape = zml.Shape.init(.{ .b = 1, .s = self.seqlen, .d = hidden_size }, model_dtype);

        const prefill_tokens_slice = try zml.Slice.alloc(self.allocator, prefill_tokens_shape);
        defer prefill_tokens_slice.free(self.allocator);
        @memset(prefill_tokens_slice.items(u32), 0);
        @memcpy(prefill_tokens_slice.items(u32)[0..all_tokens.len], all_tokens);

        const replicated_sharding = self.compiled_model.params.shardings.replicated;

        var prefill_tokens_buffer = try zml.Buffer.fromSlice(self.io, self.platform, prefill_tokens_slice, replicated_sharding);
        defer prefill_tokens_buffer.deinit();

        var prefill_token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32, replicated_sharding);
        defer prefill_token_index_buffer.deinit();

        var prefill_hidden_buffer = try zml.Buffer.uninitialized(self.io, self.platform, prefill_hidden_shape, replicated_sharding, .{});
        defer prefill_hidden_buffer.deinit();

        var embedding_prefill_args = try self.compiled_model.prefill_embedding_exe.args(self.allocator);
        defer embedding_prefill_args.deinit(self.allocator);
        var embedding_prefill_results = try self.compiled_model.prefill_embedding_exe.results(self.allocator);
        defer embedding_prefill_results.deinit(self.allocator);

        embedding_prefill_args.set(.{ self.model_buffers.text_model.embed_tokens, prefill_tokens_buffer });
        self.compiled_model.prefill_embedding_exe.call(embedding_prefill_args, &embedding_prefill_results);
        embedding_prefill_results.fill(.{&prefill_hidden_buffer});

        var prefill_full_layer_args: ?zml.Exe.Arguments = if (self.compiled_model.prefill_full_layer_exe) |exe| try exe.args(self.allocator) else null;
        defer if (prefill_full_layer_args) |*args| args.deinit(self.allocator);
        var prefill_full_layer_results: ?zml.Exe.Results = if (self.compiled_model.prefill_full_layer_exe) |exe| try exe.results(self.allocator) else null;
        defer if (prefill_full_layer_results) |*results| results.deinit(self.allocator);

        var prefill_linear_layer_args: ?zml.Exe.Arguments = if (self.compiled_model.prefill_linear_layer_exe) |exe| try exe.args(self.allocator) else null;
        defer if (prefill_linear_layer_args) |*args| args.deinit(self.allocator);
        var prefill_linear_layer_results: ?zml.Exe.Results = if (self.compiled_model.prefill_linear_layer_exe) |exe| try exe.results(self.allocator) else null;
        defer if (prefill_linear_layer_results) |*results| results.deinit(self.allocator);

        var self_attn_cache_index: usize = 0;
        var linear_attn_cache_index: usize = 0;

        for (self.model_buffers.text_model.layers, 0..) |layer_weights, i| {
            switch (self.layer_index_buffers[i]) {
                .self_attn => {
                    const exe = self.compiled_model.prefill_full_layer_exe orelse unreachable;
                    self.self_attn_layers_caches[self_attn_cache_index].k = self.kv_cache_buffers.self_attn.k;
                    self.self_attn_layers_caches[self_attn_cache_index].v = self.kv_cache_buffers.self_attn.v;
                    prefill_full_layer_args.?.set(.{ layer_weights, prefill_hidden_buffer, prefill_token_index_buffer, self.self_attn_layers_caches[self_attn_cache_index], self.compiled_model.loaded_model.inner.config, self.prefill_moe_metadata_buffers });
                    exe.call(prefill_full_layer_args.?, &prefill_full_layer_results.?);
                    prefill_full_layer_results.?.fill(.{ &prefill_hidden_buffer, &self.kv_cache_buffers.self_attn });
                    self_attn_cache_index += 1;
                },
                .linear_attn => {
                    const exe = self.compiled_model.prefill_linear_layer_exe orelse unreachable;
                    self.linear_attn_layers_caches[linear_attn_cache_index].conv_state = self.kv_cache_buffers.gated_delta_net.conv_state;
                    self.linear_attn_layers_caches[linear_attn_cache_index].recurrent_state = self.kv_cache_buffers.gated_delta_net.recurrent_state;
                    prefill_linear_layer_args.?.set(.{ layer_weights, prefill_hidden_buffer, prefill_token_index_buffer, self.linear_attn_layers_caches[linear_attn_cache_index], self.compiled_model.loaded_model.inner.config, self.prefill_moe_metadata_buffers });
                    exe.call(prefill_linear_layer_args.?, &prefill_linear_layer_results.?);
                    prefill_linear_layer_results.?.fill(.{ &prefill_hidden_buffer, &self.kv_cache_buffers.gated_delta_net });
                    linear_attn_cache_index += 1;
                },
            }
        }

        var sampling_prefill_args = try self.compiled_model.prefill_sampling_exe.args(self.allocator);
        defer sampling_prefill_args.deinit(self.allocator);
        var sampling_prefill_results = try self.compiled_model.prefill_sampling_exe.results(self.allocator);
        defer sampling_prefill_results.deinit(self.allocator);

        sampling_prefill_args.set(.{ .{
            .norm = self.model_buffers.text_model.norm,
            .lm_head = self.model_buffers.text_model.lm_head,
            .gen_options = self.compiled_model.loaded_model.inner.text_model.gen_options,
        }, prefill_hidden_buffer, self.rng_buffers });
        self.compiled_model.prefill_sampling_exe.call(sampling_prefill_args, &sampling_prefill_results);
        sampling_prefill_results.fill(.{ &prefill_tokens_buffer, &self.rng_buffers });

        try prefill_tokens_buffer.toSlice(self.io, prefill_tokens_slice);
        const generated_token = prefill_tokens_slice.items(u32)[all_tokens.len - 1];
        self.generated_token_slice.items(u32)[0] = generated_token;
    }

    pub fn runDecode(self: *Session, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
        var decoder = try self.tokenizer.decoder();
        defer decoder.deinit();

        const out_tokens_buffer: []u8 = try self.allocator.alloc(u8, 1024);
        defer self.allocator.free(out_tokens_buffer);

        const hidden_size = self.compiled_model.loaded_model.inner.config.text_config.hidden_size;
        const model_dtype = self.compiled_model.loaded_model.inner.text_model.embed_tokens.weight.dtype();

        const decode_hidden_shape = zml.Shape.init(.{ .b = 1, .s = 1, .d = hidden_size }, model_dtype);
        const replicated_sharding = self.compiled_model.params.shardings.replicated;

        var embedding_decode_args = try self.compiled_model.decode_embedding_exe.args(self.allocator);
        defer embedding_decode_args.deinit(self.allocator);
        var embedding_decode_results = try self.compiled_model.decode_embedding_exe.results(self.allocator);
        defer embedding_decode_results.deinit(self.allocator);

        var decode_full_layer_args: ?zml.Exe.Arguments = if (self.compiled_model.decode_full_layer_exe) |exe| try exe.args(self.allocator) else null;
        defer if (decode_full_layer_args) |*args| args.deinit(self.allocator);
        var decode_full_layer_results: ?zml.Exe.Results = if (self.compiled_model.decode_full_layer_exe) |exe| try exe.results(self.allocator) else null;
        defer if (decode_full_layer_results) |*results| results.deinit(self.allocator);

        var decode_linear_layer_args: ?zml.Exe.Arguments = if (self.compiled_model.decode_linear_layer_exe) |exe| try exe.args(self.allocator) else null;
        defer if (decode_linear_layer_args) |*args| args.deinit(self.allocator);
        var decode_linear_layer_results: ?zml.Exe.Results = if (self.compiled_model.decode_linear_layer_exe) |exe| try exe.results(self.allocator) else null;
        defer if (decode_linear_layer_results) |*results| results.deinit(self.allocator);

        var sampling_decode_args = try self.compiled_model.decode_sampling_exe.args(self.allocator);
        defer sampling_decode_args.deinit(self.allocator);
        var sampling_decode_results = try self.compiled_model.decode_sampling_exe.results(self.allocator);
        defer sampling_decode_results.deinit(self.allocator);

        var decode_hidden_buffer = try zml.Buffer.uninitialized(self.io, self.platform, decode_hidden_shape, replicated_sharding, .{});
        defer decode_hidden_buffer.deinit();

        var current_token_buffer = try zml.Buffer.fromSlice(self.io, self.platform, self.generated_token_slice, replicated_sharding);
        defer current_token_buffer.deinit();

        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, @intCast(all_tokens.items.len)), .u32, replicated_sharding);
        defer token_index_buffer.deinit();

        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;

        generation: while (true) {
            const token_id = self.generated_token_slice.items(u32)[0];
            if (token_id == self.eos_token_id) break :generation;

            const token = try decoder.feedOne(token_id, out_tokens_buffer);
            if (self.think_start) |think_start| if (token_id == think_start) {
                try stdout.writeAll("\x1b[2m");
            };
            try stdout.writeAll(token);
            if (self.think_end) |think_end| if (token_id == think_end) {
                try stdout.writeAll("\x1b[0m");
            };
            try stdout.flush();

            try all_tokens.append(self.allocator, token_id);
            if (all_tokens.items.len >= self.seqlen) break :generation;

            self_attn_layer_index = 0;
            linear_attn_layer_index = 0;

            embedding_decode_args.set(.{ self.model_buffers.text_model.embed_tokens, current_token_buffer });
            self.compiled_model.decode_embedding_exe.call(embedding_decode_args, &embedding_decode_results);
            embedding_decode_results.fill(.{&decode_hidden_buffer});

            for (self.model_buffers.text_model.layers, 0..) |layer_weights, layer_index| {
                switch (self.layer_index_buffers[layer_index]) {
                    .self_attn => {
                        const exe = self.compiled_model.decode_full_layer_exe orelse unreachable;
                        self.self_attn_layers_caches[self_attn_layer_index].k = self.kv_cache_buffers.self_attn.k;
                        self.self_attn_layers_caches[self_attn_layer_index].v = self.kv_cache_buffers.self_attn.v;

                        decode_full_layer_args.?.set(.{
                            layer_weights,
                            decode_hidden_buffer,
                            token_index_buffer,
                            self.self_attn_layers_caches[self_attn_layer_index],
                            self.compiled_model.loaded_model.inner.config,
                            self.decode_moe_metadata_buffers,
                        });

                        exe.call(decode_full_layer_args.?, &decode_full_layer_results.?);
                        decode_full_layer_results.?.fill(.{ &decode_hidden_buffer, &self.kv_cache_buffers.self_attn });

                        self_attn_layer_index += 1;
                    },
                    .linear_attn => {
                        const exe = self.compiled_model.decode_linear_layer_exe orelse unreachable;

                        self.linear_attn_layers_caches[linear_attn_layer_index].conv_state = self.kv_cache_buffers.gated_delta_net.conv_state;
                        self.linear_attn_layers_caches[linear_attn_layer_index].recurrent_state = self.kv_cache_buffers.gated_delta_net.recurrent_state;

                        decode_linear_layer_args.?.set(.{
                            layer_weights,
                            decode_hidden_buffer,
                            token_index_buffer,
                            self.linear_attn_layers_caches[linear_attn_layer_index],
                            self.compiled_model.loaded_model.inner.config,
                            self.decode_moe_metadata_buffers,
                        });

                        exe.call(decode_linear_layer_args.?, &decode_linear_layer_results.?);
                        decode_linear_layer_results.?.fill(.{ &decode_hidden_buffer, &self.kv_cache_buffers.gated_delta_net });

                        linear_attn_layer_index += 1;
                    },
                }
            }

            sampling_decode_args.set(.{ .{
                .norm = self.model_buffers.text_model.norm,
                .lm_head = self.model_buffers.text_model.lm_head,
                .gen_options = self.compiled_model.loaded_model.inner.text_model.gen_options,
            }, decode_hidden_buffer, self.rng_buffers, token_index_buffer });
            self.compiled_model.decode_sampling_exe.call(sampling_decode_args, &sampling_decode_results);
            sampling_decode_results.fill(.{ &current_token_buffer, &self.rng_buffers, &token_index_buffer });

            try current_token_buffer.toSlice(self.io, self.generated_token_slice);
        }

        try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
        try stdout.flush();
    }
};

fn tokenizeChatPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8, special_tokens: model.Model.SpecialTokens, is_first_turn: bool) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenId("<|im_start|>") orelse special_tokens.im_start_token_id;
    const im_end = tokenizer.tokenId("<|im_end|>") orelse special_tokens.im_end_token_id;

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len + 32);
    if (!is_first_turn) {
        try tokens.append(allocator, im_end);
        const newline = try encoder.encodeAlloc(allocator, "\n");
        try tokens.appendSlice(allocator, newline);
        allocator.free(newline);
    }

    try tokens.append(allocator, im_start);
    const user = try encoder.encodeAlloc(allocator, "user\n");
    try tokens.appendSlice(allocator, user);
    allocator.free(user);
    const prompt_encoded = try encoder.encodeAlloc(allocator, prompt);
    try tokens.appendSlice(allocator, prompt_encoded);
    allocator.free(prompt_encoded);
    try tokens.append(allocator, im_end);
    const newline = try encoder.encodeAlloc(allocator, "\n");
    try tokens.appendSlice(allocator, newline);
    allocator.free(newline);
    try tokens.append(allocator, im_start);
    const assistant = try encoder.encodeAlloc(allocator, "assistant\n");
    try tokens.appendSlice(allocator, assistant);
    allocator.free(assistant);
    const think = try encoder.encodeAlloc(allocator, "<think>\n");
    try tokens.appendSlice(allocator, think);
    allocator.free(think);

    return tokens.toOwnedSlice(allocator);
}

fn encodeSingleToken(encoder: *zml.tokenizer.Tokenizer.Encoder, text: []const u8) !u32 {
    const encoded = try encoder.encode(text);
    if (encoded.len != 1) return error.InvalidTokenizerEncoding;
    return encoded[0];
}
