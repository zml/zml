const std = @import("std");

const zml = @import("zml");

const chat_template = @import("chat_template.zig");
const inference = @import("inference.zig");
const kda = @import("kda.zig");
const layer = @import("layer.zig");
const mla = @import("mla.zig");
const model = @import("model.zig");

const log = std.log.scoped(.kimi_k3_session);

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
    prefill_compiled: ?inference.PrefillCompiledModel = null,

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
        for (compiled.loaded_model.inner.layers[0..compiled.params.active_layer_count]) |planned| switch (planned.kind()) {
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
        if (self.prefill_compiled) |*prefill| prefill.deinit();
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

    const CacheCursor = struct {
        kda: usize = 1,
        mla: usize = 0,
    };

    const LayerExecutables = struct {
        kda_moe: ?*const zml.Exe,
        kda_moe_boundary: ?*const zml.Exe,
        mla_moe: ?*const zml.Exe,
        mla_moe_boundary: ?*const zml.Exe,
    };

    const ExecutionState = struct {
        hidden: zml.Buffer,
        blocks: zml.Buffer,
        active_blocks: zml.Buffer,

        fn deinit(self: *ExecutionState) void {
            self.hidden.deinit();
            self.blocks.deinit();
            self.active_blocks.deinit();
        }
    };

    fn ensurePrefillCompiled(self: *Session, prompt_len: usize) !*inference.PrefillCompiledModel {
        if (self.prefill_compiled) |*prefill| {
            if (prefill.params.prompt_len == prompt_len) return prefill;
        }
        self.buffers.unloadResidentLayers(self.allocator);
        if (self.prefill_compiled) |*prefill| prefill.deinit();
        self.prefill_compiled = null;
        log.warn("KIMI_K3_PREFILL_COMPILE prompt_tokens={}", .{prompt_len});
        var progress = std.Progress.Node.none;
        self.prefill_compiled = try .init(
            self.allocator,
            self.io,
            self.platform,
            self.compiled.loaded_model,
            self.compiled.params,
            prompt_len,
            &progress,
        );
        return &self.prefill_compiled.?;
    }

    fn loadSlab(self: *Session, range: model.ResidentRange, name: []const u8) !void {
        if (self.buffers.resident_range) |resident| {
            if (std.meta.eql(resident, range)) return;
        }
        self.buffers.unloadResidentLayers(self.allocator);
        log.warn(
            "KIMI_K3_SLAB_LOAD slab={s} layers={}-{} resident_moe_layers={}",
            .{ name, range.first_layer, range.end_layer - 1, range.count() },
        );
        var progress = std.Progress.Node.none;
        try self.compiled.loaded_model.loadResidentRange(
            self.allocator,
            self.io,
            self.buffers,
            range,
            &progress,
        );
    }

    fn beginExecution(
        self: *Session,
        token_buffer: zml.Buffer,
        token_count: usize,
        embedding_exe: *const zml.Exe,
        layer0_exe: *const zml.Exe,
        block_update_exe: *const zml.Exe,
    ) !ExecutionState {
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
        errdefer active_buffer.deinit();

        var embedding_args = try embedding_exe.args(self.allocator);
        defer embedding_args.deinit(self.allocator);
        var embedding_results = try embedding_exe.results(self.allocator);
        defer embedding_results.deinit(self.allocator);
        embedding_args.set(.{ token_buffer, self.buffers.head.embedding });
        embedding_exe.callOpts(self.io, embedding_args, &embedding_results, .{ .wait = true });
        var hidden = embedding_results.get(zml.Buffer);
        errdefer hidden.deinit();

        var layer0_args = try layer0_exe.args(self.allocator);
        defer layer0_args.deinit(self.allocator);
        var layer0_results = try layer0_exe.results(self.allocator);
        defer layer0_results.deinit(self.allocator);
        layer0_args.set(.{ hidden, self.buffers.layer0, self.kda_caches[0] });
        layer0_exe.callOpts(self.io, layer0_args, &layer0_results, .{ .wait = true });
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
            var empty = try self.buffers.loader.zeroBlocksForTokens(token_count, self.compiled.params.source_slots);
            defer empty.deinit();
            var block_args = try block_update_exe.args(self.allocator);
            defer block_args.deinit(self.allocator);
            var block_results = try block_update_exe.results(self.allocator);
            defer block_results.deinit(self.allocator);
            block_args.set(.{ empty, layer0_result.block_residual, first_block_index_buffer });
            block_update_exe.callOpts(self.io, block_args, &block_results, .{ .wait = true });
            const expanded_blocks = block_results.get(zml.Buffer);
            layer0_result.block_residual.deinit();
            break :expanded expanded_blocks;
        };
        errdefer blocks.deinit();
        return .{ .hidden = hidden, .blocks = blocks, .active_blocks = active_buffer };
    }

    fn executeRange(
        self: *Session,
        state: *ExecutionState,
        cursor: *CacheCursor,
        range: model.ResidentRange,
        token_index_buffer: zml.Buffer,
        executables: LayerExecutables,
        require_resident: bool,
    ) !void {
        if (range.first_layer < 1 or range.end_layer > self.compiled.params.active_layer_count)
            return error.InvalidKimiK3ExecutionRange;
        const block_size: usize = @intCast(self.compiled.loaded_model.inner.config.text_config.attn_res_block_size);
        for (self.compiled.loaded_model.inner.layers[range.first_layer..range.end_layer]) |planned| {
            const layer_index = planned.weights().logical_index;
            switch (planned.kind()) {
                .kda_dense => return error.UnsupportedSecondDenseKimiK3Layer,
                .kda_moe => {
                    var streamed_weights: zml.Bufferized(layer.KdaMoeWeights) = undefined;
                    var streamed = false;
                    const weights = self.buffers.residentKdaMoe(layer_index) orelse temporary: {
                        if (require_resident) return error.KimiK3TwoSlabLayerNotResident;
                        streamed_weights = try self.buffers.loader.loadKdaMoe(layer_index);
                        streamed = true;
                        break :temporary &streamed_weights;
                    };
                    defer if (streamed) zml.Buffer.deinitAll(layer.KdaMoeWeights, &streamed_weights);
                    if (layer_index % block_size == 0) {
                        const exe = executables.kda_moe_boundary orelse return error.MissingKdaMoeBoundaryExecutable;
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
                        args.set(.{ state.hidden, state.blocks, state.active_blocks, block_index_buffer, weights.*, self.kda_caches[cursor.kda] });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.KdaMoeBoundaryCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&state.hidden, &actual.layer.output);
                        replaceBuffer(&state.blocks, &actual.block_sources);
                        replaceBuffer(&state.active_blocks, &actual.active_blocks);
                        replaceBuffer(&self.kda_caches[cursor.kda].q_conv, &actual.layer.cache.q_conv);
                        replaceBuffer(&self.kda_caches[cursor.kda].k_conv, &actual.layer.cache.k_conv);
                        replaceBuffer(&self.kda_caches[cursor.kda].v_conv, &actual.layer.cache.v_conv);
                        replaceBuffer(&self.kda_caches[cursor.kda].recurrent_state, &actual.layer.cache.recurrent_state);
                    } else {
                        const exe = executables.kda_moe orelse return error.MissingKdaMoeExecutable;
                        var args = try exe.args(self.allocator);
                        defer args.deinit(self.allocator);
                        var results = try exe.results(self.allocator);
                        defer results.deinit(self.allocator);
                        args.set(.{ state.hidden, state.blocks, state.active_blocks, weights.*, self.kda_caches[cursor.kda] });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.KdaMoeCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&state.hidden, &actual.output);
                        replaceBuffer(&self.kda_caches[cursor.kda].q_conv, &actual.cache.q_conv);
                        replaceBuffer(&self.kda_caches[cursor.kda].k_conv, &actual.cache.k_conv);
                        replaceBuffer(&self.kda_caches[cursor.kda].v_conv, &actual.cache.v_conv);
                        replaceBuffer(&self.kda_caches[cursor.kda].recurrent_state, &actual.cache.recurrent_state);
                    }
                    cursor.kda += 1;
                },
                .mla_moe => {
                    var streamed_weights: zml.Bufferized(layer.MlaMoeWeights) = undefined;
                    var streamed = false;
                    const weights = self.buffers.residentMlaMoe(layer_index) orelse temporary: {
                        if (require_resident) return error.KimiK3TwoSlabLayerNotResident;
                        streamed_weights = try self.buffers.loader.loadMlaMoe(layer_index);
                        streamed = true;
                        break :temporary &streamed_weights;
                    };
                    defer if (streamed) zml.Buffer.deinitAll(layer.MlaMoeWeights, &streamed_weights);
                    if (layer_index % block_size == 0) {
                        const exe = executables.mla_moe_boundary orelse return error.MissingMlaMoeBoundaryExecutable;
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
                        args.set(.{ state.hidden, state.blocks, state.active_blocks, block_index_buffer, weights.*, self.mla_caches[cursor.mla], token_index_buffer });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.MlaMoeBoundaryCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&state.hidden, &actual.layer.output);
                        replaceBuffer(&state.blocks, &actual.block_sources);
                        replaceBuffer(&state.active_blocks, &actual.active_blocks);
                        replaceBuffer(&self.mla_caches[cursor.mla].compressed, &actual.layer.cache.compressed);
                        replaceBuffer(&self.mla_caches[cursor.mla].extra_key, &actual.layer.cache.extra_key);
                    } else {
                        const exe = executables.mla_moe orelse return error.MissingMlaMoeExecutable;
                        var args = try exe.args(self.allocator);
                        defer args.deinit(self.allocator);
                        var results = try exe.results(self.allocator);
                        defer results.deinit(self.allocator);
                        args.set(.{ state.hidden, state.blocks, state.active_blocks, weights.*, self.mla_caches[cursor.mla], token_index_buffer });
                        exe.callOpts(self.io, args, &results, .{ .wait = true });
                        var actual: zml.Bufferized(layer.MlaMoeCompactResult) = undefined;
                        results.fill(.{&actual});
                        replaceBuffer(&state.hidden, &actual.output);
                        replaceBuffer(&self.mla_caches[cursor.mla].compressed, &actual.cache.compressed);
                        replaceBuffer(&self.mla_caches[cursor.mla].extra_key, &actual.cache.extra_key);
                    }
                    cursor.mla += 1;
                },
            }
        }
    }

    fn finishHead(self: *Session, state: *ExecutionState, head_exe: *const zml.Exe) !u32 {
        var head_args = try head_exe.args(self.allocator);
        defer head_args.deinit(self.allocator);
        var head_results = try head_exe.results(self.allocator);
        defer head_results.deinit(self.allocator);
        head_args.set(.{
            state.hidden,
            state.blocks,
            state.active_blocks,
            self.buffers.head.output_res_norm,
            self.buffers.head.output_res_projection,
            self.buffers.head.final_norm,
            self.buffers.head.lm_head,
        });
        head_exe.callOpts(self.io, head_args, &head_results, .{ .wait = true });
        var head: zml.Bufferized(layer.SessionHeadResult) = undefined;
        head_results.fill(.{&head});
        defer zml.Buffer.deinitAll(layer.SessionHeadResult, &head);
        const greedy_i64 = try head.greedy_token.getValue(i64, self.io);
        if (greedy_i64 < 0 or greedy_i64 >= 163840) return error.InvalidKimiK3GreedyToken;
        return @intCast(greedy_i64);
    }

    fn decodeExecutables(self: *const Session) LayerExecutables {
        return .{
            .kda_moe = if (self.compiled.kda_moe) |*exe| exe else null,
            .kda_moe_boundary = if (self.compiled.kda_moe_boundary) |*exe| exe else null,
            .mla_moe = if (self.compiled.mla_moe) |*exe| exe else null,
            .mla_moe_boundary = if (self.compiled.mla_moe_boundary) |*exe| exe else null,
        };
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
        var state = try self.beginExecution(
            token_buffer,
            1,
            &self.compiled.embedding,
            &self.compiled.layer0,
            &self.compiled.block_update,
        );
        defer state.deinit();
        var cursor: CacheCursor = .{};
        const executables = self.decodeExecutables();
        if (self.buffers.execution_mode == .two_slab) {
            if (self.compiled.params.active_layer_count != model.full_model_layer_count)
                return error.KimiK3TwoSlabRequiresFullModel;
            try self.loadSlab(model.slab_a, "A");
            try self.executeRange(&state, &cursor, model.slab_a, token_index_buffer, executables, true);
            try self.loadSlab(model.slab_b, "B");
            try self.executeRange(&state, &cursor, model.slab_b, token_index_buffer, executables, true);
        } else {
            try self.executeRange(
                &state,
                &cursor,
                .{ .first_layer = 1, .end_layer = self.compiled.params.active_layer_count },
                token_index_buffer,
                executables,
                false,
            );
        }
        const greedy = try self.finishHead(&state, &self.compiled.head);
        self.position += 1;
        return greedy;
    }

    fn runBatchedPrefill(self: *Session, all_tokens: []const u32) !void {
        const prefill = try self.ensurePrefillCompiled(all_tokens.len);
        var token_buffer = try zml.Buffer.fromBytes(
            self.io,
            self.platform,
            prefill.params.tokens.shape(),
            prefill.params.shardings.model,
            std.mem.sliceAsBytes(all_tokens),
        );
        defer token_buffer.deinit();
        var token_index_buffer = try zml.Buffer.scalar(self.io, self.platform, @as(u32, 0), .u32);
        defer token_index_buffer.deinit();
        var state = try self.beginExecution(
            token_buffer,
            all_tokens.len,
            &prefill.embedding,
            &prefill.layer0,
            &prefill.block_update,
        );
        defer state.deinit();
        var cursor: CacheCursor = .{};
        const executables: LayerExecutables = .{
            .kda_moe = &prefill.kda_moe,
            .kda_moe_boundary = &prefill.kda_moe_boundary,
            .mla_moe = &prefill.mla_moe,
            .mla_moe_boundary = &prefill.mla_moe_boundary,
        };
        try self.loadSlab(model.slab_a, "A");
        try self.executeRange(&state, &cursor, model.slab_a, token_index_buffer, executables, true);
        try self.loadSlab(model.slab_b, "B");
        try self.executeRange(&state, &cursor, model.slab_b, token_index_buffer, executables, true);
        self.last_generated_token = try self.finishHead(&state, &prefill.head);
        self.position = all_tokens.len;
    }

    pub fn runPrefill(self: *Session, all_tokens: []const u32) !void {
        if (all_tokens.len == 0) return error.EmptyKimiK3Prompt;
        if (all_tokens.len > self.compiled.params.seqlen) return error.PromptTooLong;
        try self.reset();
        if (self.buffers.execution_mode == .two_slab) {
            try self.runBatchedPrefill(all_tokens);
            return;
        }
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
