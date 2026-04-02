const std = @import("std");
const testing = std.testing;

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;

const log = std.log.scoped(.qwen3_5);

pub const Qwen35 = struct {
    pub const Config = struct {
        text_config: TextConfig,
        vision_config: VisionConfig,
    };

    pub const TextConfig = struct {
        // General
        num_hidden_layers: i64,
        layer_types: []const LayerType,
        hidden_size: i64,
        max_position_embeddings: i64,
        rms_norm_eps: f32,
        // Self attention
        head_dim: i64,
        num_attention_heads: i64,
        num_key_value_heads: i64,
        rope_parameters: struct {
            mrope_section: [3]i64,
            partial_rotary_factor: f32,
            rope_theta: f32,
        },
        // Linear attention
        linear_conv_kernel_dim: i64,
        linear_key_head_dim: i64,
        linear_num_key_heads: i64,
        linear_num_value_heads: i64,
        linear_value_head_dim: i64,
    };

    pub const VisionConfig = struct {
        hidden_size: i64,
        depth: i64,
        num_heads: i64,
        patch_size: i64,
        spatial_merge_size: i64,
        temporal_patch_size: i64,
        num_position_embeddings: i64,
        in_channels: i64,
    };

    // Each layer uses either: full attention (SelfAttn) or linear attention (GatedDeltaNet).
    pub const LayerType = enum {
        linear_attention,
        full_attention,
    };

    pub const GenOptions = struct {
        sampling_strategy: zml.nn.SamplingStrategy = .{},
        max_seq_len: i64,
    };

    // Config values which are not part of the config.json but can be useful
    pub const AutoConfig = struct {
        im_start_token_id: u32 = 248045,
        im_end_token_id: u32 = 248046,
        think_start_token_id: u32 = 248068,
        think_end_token_id: u32 = 248069,
        vision_start_token_id: u32 = 248053,
        vision_end_token_id: u32 = 248054,
        image_token_id: u32 = 248056,
        video_token_id: u32 = 248057,
        end_of_text_token_id: u32 = 248044,
        vision_patch_3d_size: i64,
    };

    text_model: TextModel,
    vision_model: VisionModel,
    lm_head: zml.nn.Linear,

    pub const ChunkMediaInsert = struct {
        grid_thw: [3]i64,
        pad_token_start: i64,
        pad_token_count: i64,
        pad_token_media_embed_offset: i64,
        temporal_offset: i64 = 0,
    };

    config: Config,
    gen_options: GenOptions,
    auto_config: AutoConfig,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Qwen35 {
        // For some Qwen3.5 versions, the output projection lm_head has a standalone weight tensor, while for others it's the same as the input embedding layer
        const lm_head_prefix = if (store.hasKey("lm_head.weight")) "lm_head" else "model.language_model.embed_tokens";
        return .{
            .text_model = try .init(allocator, store.withPrefix("model.language_model"), config),
            .vision_model = try .init(allocator, store.withPrefix("model.visual"), config),
            .lm_head = .init(store.withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, null), null, .d),
            .config = config,
            .gen_options = gen_options,
            .auto_config = .{ .vision_patch_3d_size = config.vision_config.in_channels * config.vision_config.temporal_patch_size * config.vision_config.patch_size * config.vision_config.patch_size },
        };
    }

    pub fn deinit(self: Qwen35, allocator: std.mem.Allocator) void {
        self.text_model.deinit(allocator);
        self.vision_model.deinit(allocator);
    }

    pub fn load(
        self: *const Qwen35,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Qwen35) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }
        return zml.io.load(Qwen35, self, allocator, io, platform, store, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .shardings = shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Qwen35), allocator: std.mem.Allocator) void {
        TextModel.unloadBuffers(&self.text_model, allocator);
        VisionModel.unloadBuffers(&self.vision_model, allocator);
        self.lm_head.weight.deinit();
    }

    pub fn sampleTokens(
        self: Qwen35,
        out: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, Tensor.Rng } {
        const logits = self.lm_head.forward(out.withPartialTags(.{.d})).rename(.{ .dout = .voc });
        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_options.sampling_strategy, rng);
        return .{ next_tokens, new_rng };
    }

    pub fn text_forward(
        self: Qwen35,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const token_embeds = self.text_model.embed_tokens.forward(tokens);

        const text_only_position_ids = Tensor.arange(.{ .end = token_embeds.dim(.s) }, .i64)
            .withTags(.{.s}).insertAxes(.s, .{.b}).broad(zml.Shape.init(.{ .b = token_embeds.dim(.b), .s = token_embeds.dim(.s) }, .i64))
            .add(token_index.broad(zml.Shape.init(.{ .b = token_embeds.dim(.b), .s = token_embeds.dim(.s) }, .i64)));

        const text_model_output, const updated_kv_cache = self.text_model.forward(token_embeds, token_index, kv_cache, text_only_position_ids);
        const new_tokens, const new_rng = self.sampleTokens(text_model_output, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn multimodal_prefill_forward(
        self: Qwen35,
        tokens: Tensor,
        media_pixel_values: []const Tensor,
        position_ids: Tensor,
        media_metadata: []const MultimodalPrompt.MediaMetadata,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor, Tensor.Rng } {
        const text_embeds = self.text_model.embed_tokens.forward(tokens.withPartialTags(.{.s}));

        var text_with_vision_embeds = text_embeds;
        for (media_metadata, media_pixel_values) |metadata, pixel_values| {
            const grid_thw = switch (metadata) {
                .image => |m| m.grid_thw,
                .video => |m| m.grid_thw,
            };

            const vision_embeds = self.vision_model.forward(pixel_values, grid_thw);

            switch (metadata) {
                .image => |m| {
                    const media_chunk = vision_embeds.dynamicSlice1d(vision_embeds.axis(.s), .{ .start = zml.Tensor.scalar(m.pad_token_media_embed_offset, .i64), .len = @intCast(m.pad_token_count) }).convert(text_embeds.dtype());
                    text_with_vision_embeds = text_with_vision_embeds.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(m.pad_token_start, .i64) }, media_chunk);
                },
                .video => |m| {
                    for (m.frame_pad_token_starts, m.frame_pad_token_counts, m.frame_pad_token_media_embed_offsets) |run_start, run_len, run_embed_offset| {
                        const media_chunk = vision_embeds.dynamicSlice1d(vision_embeds.axis(.s), .{ .start = zml.Tensor.scalar(run_embed_offset, .i64), .len = @intCast(run_len) }).convert(text_embeds.dtype());
                        text_with_vision_embeds = text_with_vision_embeds.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(run_start, .i64) }, media_chunk);
                    }
                },
            }
        }

        const text_model_output, const updated_kv_cache = self.text_model.forward(
            text_with_vision_embeds,
            zml.Tensor.scalar(@as(i64, 0), .i64),
            kv_cache,
            position_ids,
        );

        const last_pos = zml.Tensor.scalar(tokens.dim(.s) - 1, .i64);
        const last_hidden = text_model_output.dynamicSlice1d(text_model_output.axis(.s), .{ .start = last_pos, .len = 1 });
        const position_max = position_ids.max(.g).max(.b).max(.s).convert(.i64).asScalar();
        const mrope_position_deltas = position_max.addConstant(1 - tokens.dim(.s)).insertAxes(.last, .{ .b, .s });

        const sampled_tokens, const new_rng = self.sampleTokens(last_hidden, rng);

        return .{ sampled_tokens.convert(tokens.dtype()), updated_kv_cache, mrope_position_deltas, new_rng };
    }

    pub fn multimodal_chunk_prefill_forward(
        self: Qwen35,
        tokens: Tensor,
        media_pixel_values: Tensor,
        position_ids: Tensor,
        media_insert: ChunkMediaInsert,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor, Tensor.Rng } {
        stdx.debug.assert(media_insert.grid_thw[0] == 1, "multimodal_chunk_prefill_forward expects grid_thw.t == 1, got {d}", .{media_insert.grid_thw[0]});

        const text_embeds = self.text_model.embed_tokens.forward(tokens.withPartialTags(.{.s}));
        const vision_embeds = self.vision_model.forward(media_pixel_values, media_insert.grid_thw);
        const media_chunk = vision_embeds.dynamicSlice1d(
            vision_embeds.axis(.s),
            .{
                .start = zml.Tensor.scalar(media_insert.pad_token_media_embed_offset, .i64),
                .len = @intCast(media_insert.pad_token_count),
            },
        ).convert(text_embeds.dtype());
        const text_with_vision_embeds = text_embeds.dynamicUpdateSlice(
            .{ .s = zml.Tensor.scalar(media_insert.pad_token_start, .i64) },
            media_chunk,
        );

        const text_model_output, const updated_kv_cache = self.text_model.forward(
            text_with_vision_embeds,
            zml.Tensor.scalar(@as(i64, 0), .i64),
            kv_cache,
            position_ids,
        );

        const last_pos = zml.Tensor.scalar(tokens.dim(.s) - 1, .i64);
        const last_hidden = text_model_output.dynamicSlice1d(text_model_output.axis(.s), .{ .start = last_pos, .len = 1 });
        const position_max = position_ids.max(.g).max(.b).max(.s).convert(.i64).asScalar();
        const mrope_position_deltas = position_max.addConstant(1 - tokens.dim(.s)).insertAxes(.last, .{ .b, .s });

        const sampled_tokens, const new_rng = self.sampleTokens(last_hidden, rng);
        return .{ sampled_tokens.convert(tokens.dtype()), updated_kv_cache, mrope_position_deltas, new_rng };
    }

    pub fn multimodal_decode_forward(
        self: Qwen35,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        mrope_position_deltas: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const token_embeds = self.text_model.embed_tokens.forward(tokens);

        const position_ids_1d = token_index.add(mrope_position_deltas);
        const position_ids = zml.Tensor.stack(&.{ position_ids_1d, position_ids_1d, position_ids_1d }, 0, .g).withTags(.{ .g, .b, .s });

        const text_model_output, const updated_kv_cache = self.text_model.forward(token_embeds, token_index, kv_cache, position_ids);

        const new_tokens, const new_rng = self.sampleTokens(text_model_output, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }
};

// Multimodal prompt helper:
// Constructs the prefill prompt by including the image and video placeholders as expected by the Qwen3.5 model.
// Tracks metadata needed by the model's prefill forward function.
pub const MultimodalPrompt = struct {
    pub const MediaInput = union(enum) {
        image: struct { grid_thw: [3]i64 },
        video: struct { grid_thw: [3]i64, frame_timestamps: []const f32 },
    };

    pub const InitOptions = struct {
        max_seq_len: i64,
        enable_thinking: bool = true,
    };

    pub const MediaMetadata = union(enum) {
        image: struct {
            grid_thw: [3]i64,
            patch_count: i64,
            pad_token_start: i64,
            pad_token_count: i64,
            pad_token_media_embed_offset: i64,
        },
        video: struct {
            grid_thw: [3]i64,
            patch_count: i64,
            frame_pad_token_starts: []i64,
            frame_pad_token_counts: []i64,
            frame_pad_token_media_embed_offsets: []i64,
        },
    };

    pub const Stat = struct {
        prompt_chars: usize = 0,
        prompt_tokens: usize = 0,
        total_media_tokens: i64 = 0,
        total_tokens: usize = 0,
    };

    // Full multimodal prompt, including text and media placeholder tokens.
    token_ids: []u32 = &.{},
    // Packed [3 * seq_len] mRoPE ids on host.
    position_ids: []i64 = &.{},
    // Per-media metadata, same length/order as input media.
    media_metadata: []MediaMetadata = &.{},
    // Summary values computed during prompt layout generation.
    stat: Stat = .{},

    const TokenAccItem = struct {
        token_id: u32,
        pos_t: i64,
        pos_h: i64,
        pos_w: i64,
    };

    const Cursor = struct {
        token_cursor: i64 = 0,
        pos_cursor: i64 = 0,
        media_token_cursor: i64 = 0,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        config: Qwen35.Config,
        tokenizer: zml.tokenizer.Tokenizer,
        text_prompt: []const u8,
        media: []const MediaInput,
        options: InitOptions,
    ) !MultimodalPrompt {
        var encoder = try tokenizer.encoder();
        defer encoder.deinit();

        // Special tokens needed to build the prompt.
        const im_start_id = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
        const im_end_id = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
        const user_id = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
        const assistant_id = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
        const vision_start_id = tokenizer.tokenToId("<|vision_start|>") orelse return error.NoSuchToken;
        const vision_end_id = tokenizer.tokenToId("<|vision_end|>") orelse return error.NoSuchToken;
        const image_pad_id = tokenizer.tokenToId("<|image_pad|>") orelse return error.NoSuchToken;
        const video_pad_id = tokenizer.tokenToId("<|video_pad|>") orelse return error.NoSuchToken;
        const newline_id = tokenizer.tokenToId("\\n") orelse return error.NoSuchToken;

        // Create accumulator and cursor state
        var token_acc: std.MultiArrayList(TokenAccItem) = .{};
        defer token_acc.deinit(allocator);
        try token_acc.ensureTotalCapacity(allocator, text_prompt.len + media.len * 32 + 32);
        var media_metadata_list: std.ArrayList(MediaMetadata) = try .initCapacity(allocator, media.len);
        defer media_metadata_list.deinit(allocator);
        var cursor: Cursor = .{};

        // User text input
        try appendTextTokens(allocator, &token_acc, &cursor, &.{ im_start_id, user_id, newline_id });
        const prompt_tokens = try encoder.encode(text_prompt);
        try appendTextTokens(allocator, &token_acc, &cursor, prompt_tokens);

        // Adding each media placeholder block
        for (media) |m| {
            const dims = try MediaDims.init(m, config.vision_config.spatial_merge_size);

            switch (m) {
                .image => {
                    if (dims.t != 1) return error.InvalidImageGridThw;
                    try appendTextTokens(allocator, &token_acc, &cursor, &.{vision_start_id});
                    const pad_token_start = cursor.token_cursor;
                    const pad_token_count = dims.pad_token_count;
                    try appendVisionPadTokens(allocator, &token_acc, &cursor, image_pad_id, dims.t, dims);
                    try appendTextTokens(allocator, &token_acc, &cursor, &.{vision_end_id});
                    try media_metadata_list.append(allocator, .{
                        .image = .{ .grid_thw = dims.grid_thw, .patch_count = dims.patch_count, .pad_token_start = pad_token_start, .pad_token_count = pad_token_count, .pad_token_media_embed_offset = 0 },
                    });
                },
                .video => |vid| {
                    if (vid.frame_timestamps.len != @as(usize, @intCast(dims.t))) return error.InvalidVideoTimestamps;
                    var frame_pad_starts = try allocator.alloc(i64, vid.frame_timestamps.len);
                    errdefer allocator.free(frame_pad_starts);
                    var frame_pad_counts = try allocator.alloc(i64, vid.frame_timestamps.len);
                    errdefer allocator.free(frame_pad_counts);
                    var frame_pad_embed_offsets = try allocator.alloc(i64, vid.frame_timestamps.len);
                    errdefer allocator.free(frame_pad_embed_offsets);
                    for (vid.frame_timestamps, 0..) |timestamp_s, frame_idx| {
                        var ts_buf: [64]u8 = undefined;
                        const ts_text = try std.fmt.bufPrint(&ts_buf, "<{d:.1} seconds>", .{timestamp_s});
                        const ts_tokens = try encoder.encode(ts_text);
                        try appendTextTokens(allocator, &token_acc, &cursor, ts_tokens);
                        try appendTextTokens(allocator, &token_acc, &cursor, &.{vision_start_id});
                        frame_pad_starts[frame_idx] = cursor.token_cursor;
                        frame_pad_counts[frame_idx] = dims.frame_pad_token_count;
                        frame_pad_embed_offsets[frame_idx] = cursor.media_token_cursor;
                        try appendVisionPadTokens(allocator, &token_acc, &cursor, video_pad_id, 1, dims);
                        try appendTextTokens(allocator, &token_acc, &cursor, &.{vision_end_id});
                    }
                    try media_metadata_list.append(allocator, .{
                        .video = .{ .grid_thw = dims.grid_thw, .patch_count = dims.patch_count, .frame_pad_token_starts = frame_pad_starts, .frame_pad_token_counts = frame_pad_counts, .frame_pad_token_media_embed_offsets = frame_pad_embed_offsets },
                    });
                },
            }
        }

        // Assistant prefix and think setup
        try appendTextTokens(allocator, &token_acc, &cursor, &.{ im_end_id, newline_id, im_start_id, assistant_id, newline_id });
        const think_text = if (options.enable_thinking) "<think>" else "<think> </think>";
        try appendTextTokens(allocator, &token_acc, &cursor, try encoder.encode(think_text));

        // Extracting outputs.
        const seq_len: usize = token_acc.len;
        if (options.max_seq_len > 0 and seq_len > @as(usize, @intCast(options.max_seq_len))) return error.PromptTooLong;
        const owned_tokens = try allocator.dupe(u32, token_acc.items(.token_id));
        errdefer allocator.free(owned_tokens);
        var position_ids = try allocator.alloc(i64, 3 * seq_len);
        errdefer allocator.free(position_ids);
        @memcpy(position_ids[0..seq_len], token_acc.items(.pos_t));
        @memcpy(position_ids[seq_len .. 2 * seq_len], token_acc.items(.pos_h));
        @memcpy(position_ids[2 * seq_len .. 3 * seq_len], token_acc.items(.pos_w));
        const owned_media_metadata = try media_metadata_list.toOwnedSlice(allocator);
        errdefer allocator.free(owned_media_metadata);

        return .{
            .token_ids = owned_tokens,
            .position_ids = position_ids,
            .media_metadata = owned_media_metadata,
            .stat = .{
                .prompt_chars = text_prompt.len,
                .prompt_tokens = prompt_tokens.len,
                .total_media_tokens = cursor.media_token_cursor,
                .total_tokens = seq_len,
            },
        };
    }

    fn appendTextTokens(
        allocator: std.mem.Allocator,
        token_accumulator: *std.MultiArrayList(TokenAccItem),
        cursor: *Cursor,
        token_ids: []const u32,
    ) !void {
        for (token_ids) |token_id| {
            try token_accumulator.append(allocator, .{
                .token_id = token_id,
                .pos_t = cursor.pos_cursor,
                .pos_h = cursor.pos_cursor,
                .pos_w = cursor.pos_cursor,
            });
            cursor.token_cursor += 1;
            cursor.pos_cursor += 1;
        }
    }

    fn appendVisionPadTokens(
        allocator: std.mem.Allocator,
        token_accumulator: *std.MultiArrayList(TokenAccItem),
        cursor: *Cursor,
        pad_id: u32,
        frame_count: i64,
        dims: MediaDims,
    ) !void {
        const frame_tokens = dims.frame_pad_token_count;
        const token_count = frame_count * frame_tokens;

        for (0..@intCast(token_count)) |flat_usize| {
            const flat: i64 = @intCast(flat_usize);
            const t_idx = @divFloor(flat, frame_tokens);
            const hw = @mod(flat, frame_tokens);
            const h_idx = @divFloor(hw, dims.merged_w);
            const w_idx = @mod(hw, dims.merged_w);
            try token_accumulator.append(allocator, .{
                .token_id = pad_id,
                .pos_t = cursor.pos_cursor + t_idx,
                .pos_h = cursor.pos_cursor + h_idx,
                .pos_w = cursor.pos_cursor + w_idx,
            });
        }

        cursor.token_cursor += token_count;
        cursor.pos_cursor += @max(@max(frame_count, dims.merged_h), dims.merged_w);
        cursor.media_token_cursor += token_count;
    }

    pub fn deinit(self: *MultimodalPrompt, allocator: std.mem.Allocator) void {
        if (self.token_ids.len > 0) allocator.free(self.token_ids);
        if (self.position_ids.len > 0) allocator.free(self.position_ids);
        for (self.media_metadata) |m| {
            switch (m) {
                .image => {},
                .video => |v| {
                    if (v.frame_pad_token_starts.len > 0) allocator.free(v.frame_pad_token_starts);
                    if (v.frame_pad_token_counts.len > 0) allocator.free(v.frame_pad_token_counts);
                    if (v.frame_pad_token_media_embed_offsets.len > 0) allocator.free(v.frame_pad_token_media_embed_offsets);
                },
            }
        }
        if (self.media_metadata.len > 0) allocator.free(self.media_metadata);
        self.* = .{};
    }

    const MediaDims = struct {
        grid_thw: [3]i64,
        t: i64,
        merged_h: i64,
        merged_w: i64,
        frame_pad_token_count: i64,
        pad_token_count: i64,
        patch_count: i64,

        pub fn init(media: MultimodalPrompt.MediaInput, spatial_merge_size: i64) !MediaDims {
            const grid_thw = switch (media) {
                .image => |img| img.grid_thw,
                .video => |vid| vid.grid_thw,
            };
            const t = grid_thw[0];
            const h = grid_thw[1];
            const w = grid_thw[2];
            if (t <= 0 or h <= 0 or w <= 0) return error.InvalidMediaGridThw;
            if (@mod(h, spatial_merge_size) != 0 or @mod(w, spatial_merge_size) != 0) return error.InvalidMediaGridThw;
            const merged_h = @divExact(h, spatial_merge_size);
            const merged_w = @divExact(w, spatial_merge_size);
            const frame_pad_token_count = merged_h * merged_w;
            return .{
                .grid_thw = grid_thw,
                .t = t,
                .merged_h = merged_h,
                .merged_w = merged_w,
                .frame_pad_token_count = frame_pad_token_count,
                .pad_token_count = t * frame_pad_token_count,
                .patch_count = t * h * w,
            };
        }
    };

    pub fn format(self: MultimodalPrompt, writer: *std.Io.Writer) !void {
        try writer.print(
            "prompt_chars={d}, prompt_tokens={d},media_count={d}, total_media_tokens={d}, total_tokens={d}",
            .{
                self.stat.prompt_chars,
                self.stat.prompt_tokens,
                self.media_metadata.len,
                self.stat.total_media_tokens,
                self.stat.total_tokens,
            },
        );
        for (self.media_metadata, 0..) |metadata, i| {
            switch (metadata) {
                .image => |md| {
                    try writer.print(
                        "\n  media[{d}] image grid_thw={any} frames=1 patch_count={d} token_count={d} token_start={d}",
                        .{ i, md.grid_thw, md.patch_count, md.pad_token_count, md.pad_token_start },
                    );
                },
                .video => |md| {
                    var media_tokens: i64 = 0;
                    for (md.frame_pad_token_counts) |len| {
                        media_tokens += len;
                    }
                    try writer.print(
                        "\n  media[{d}] video grid_thw={any} frames={d} patch_count={d} token_count={d} token_start={d}",
                        .{ i, md.grid_thw, md.frame_pad_token_counts.len, md.patch_count, media_tokens, md.frame_pad_token_starts[0] },
                    );
                },
            }
        }
    }
};

//========================Vision model========================

pub const VisionModel = struct {
    vision_patch_embed: VisionPatchEmbed,
    pos_embed: zml.nn.TokenEmbedding,
    blocks: []VisionBlock,
    patch_merger: VisionPatchMerger,

    hidden_size: i64,
    num_heads: i64,
    spatial_merge_size: i64,
    num_pos_embeds: i64,
    sqrt_num_pos_embeds: f32,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Qwen35.Config) !VisionModel {
        const blocks = try allocator.alloc(VisionBlock, @intCast(config.vision_config.depth));
        errdefer allocator.free(blocks);

        for (blocks, 0..) |*block, i| {
            block.* = .init(store.withPrefix("blocks").withLayer(i), config);
        }

        return .{
            .vision_patch_embed = .init(allocator, config, store.withPrefix("patch_embed")),
            .pos_embed = .{ .weight = store.withPrefix("pos_embed").createTensor("weight", .{ .voc, .d }, null) },
            .blocks = blocks,
            .patch_merger = .init(store.withPrefix("merger"), config),
            .hidden_size = config.vision_config.hidden_size,
            .num_heads = config.vision_config.num_heads,
            .spatial_merge_size = config.vision_config.spatial_merge_size,
            .num_pos_embeds = config.vision_config.num_position_embeddings,
            .sqrt_num_pos_embeds = std.math.pow(f32, @as(f32, @floatFromInt(config.vision_config.num_position_embeddings)), 0.5),
            .rope_opts = .{
                .layout = .sequential,
                .scaling = .{ .default = .{ .rope_theta = 10000.0 } },
            },
        };
    }

    pub fn deinit(self: VisionModel, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionModel), allocator: std.mem.Allocator) void {
        VisionPatchEmbed.unloadBuffers(&self.vision_patch_embed);
        self.pos_embed.weight.deinit();
        for (self.blocks) |*block| {
            VisionBlock.unloadBuffers(block);
        }
        allocator.free(self.blocks);
        VisionPatchMerger.unloadBuffers(&self.patch_merger);
    }

    pub fn forward(self: VisionModel, pixel_values: Tensor, grid_thw: [3]i64) Tensor {
        var hidden_states = self.vision_patch_embed.forward(pixel_values.withPartialTags(.{ .p, .ps }));

        const pos_embeds = self.getPosEmbeds(&grid_thw).convert(hidden_states.dtype());
        hidden_states = hidden_states.add(pos_embeds);

        const cos, const sin = self.getVisionRopeCosAndSin(&grid_thw);

        for (self.blocks) |block| {
            hidden_states = block.forward(hidden_states, cos, sin, grid_thw[1] * grid_thw[2]);
        }

        return self.patch_merger.forward(hidden_states);
    }

    // Maps the vision patch grid to the learned positional embeddings of size num_pos_embeds
    pub fn getPosEmbeds(self: VisionModel, grid_thw: []const i64) Tensor {
        const t, const h, const w = .{ grid_thw[0], grid_thw[1], grid_thw[2] };

        // Interpolating coordinates
        const h_idxs = zml.Tensor.arange(.{ .end = h }, .f32).scale((self.sqrt_num_pos_embeds - 1) / @as(f32, @floatFromInt(h - 1)));
        const w_idxs = zml.Tensor.arange(.{ .end = w }, .f32).scale((self.sqrt_num_pos_embeds - 1) / @as(f32, @floatFromInt(w - 1)));

        const h_floor = h_idxs.floor();
        const w_floor = w_idxs.floor();
        const h_ceil = h_floor.addConstant(1).clamp(zml.Tensor.scalar(0, .f32), zml.Tensor.scalar(self.sqrt_num_pos_embeds - 1, .f32));
        const w_ceil = w_floor.addConstant(1).clamp(zml.Tensor.scalar(0, .f32), zml.Tensor.scalar(self.sqrt_num_pos_embeds - 1, .f32));

        const dh = h_idxs.sub(h_floor);
        const dw = w_idxs.sub(w_floor);

        // Expansion to 2D grid
        const dh_grid, const dw_grid = zml.Tensor.cartesianProduct(2, .{ dh, dw });
        const h_floor_grid, const w_floor_grid = zml.Tensor.cartesianProduct(2, .{ h_floor, w_floor });
        const h_ceil_grid, const w_ceil_grid = zml.Tensor.cartesianProduct(2, .{ h_ceil, w_ceil });

        // Get positional embeddings for grid corners
        const h_grid = zml.Tensor.stack(&.{ h_floor_grid, h_floor_grid, h_ceil_grid, h_ceil_grid }, 0, .corners);
        const w_grid = zml.Tensor.stack(&.{ w_floor_grid, w_ceil_grid, w_floor_grid, w_ceil_grid }, 0, .corners);
        const indices = h_grid.scale(self.sqrt_num_pos_embeds).add(w_grid).reshape(.{ 4, -1 }).convert(.i64);
        const embeds = self.pos_embed.forward(indices);

        // Compute bilinear interpolation weights
        const w11 = dh_grid.mul(dw_grid);
        const w10 = dh_grid.sub(w11);
        const w01 = dw_grid.sub(w11);
        const w00 = zml.Tensor.constant(.{ .f32 = 1 }).broad(w01.shape()).sub(dh_grid).sub(w01);
        var weights = zml.Tensor.stack(&.{ w00, w01, w10, w11 }, 0, .corners).reshape(.{ 4, -1, 1 }).convert(self.pos_embed.weight.dtype());

        // Interpolate positional embeddings
        const weights_embed = embeds.mul(weights.repeat1d(-1, @intCast(self.hidden_size))).withTags(.{ .corners, .hw, .d });
        const combined = weights_embed.sum(.corners).squeeze(.corners);
        const combined_reshape = combined.splitAxis(.hw, .{ .h = @divExact(h, self.spatial_merge_size), .m1 = self.spatial_merge_size, .w = @divExact(w, self.spatial_merge_size), .m2 = self.spatial_merge_size });
        const combined_permuted = combined_reshape.transpose(.{ .h, .w, .m1, .m2, .d });
        const output = combined_permuted.repeat1d(.h, @intCast(t)).merge(.{ .p = .{ .h, .w, .m1, .m2 } });

        return output;
    }

    // Rotary position embedding for the vision transformer
    pub fn getVisionRopeCosAndSin(self: VisionModel, grid_thw: []const i64) struct { Tensor, Tensor } {
        const t, const h, const w = .{ grid_thw[0], grid_thw[1], grid_thw[2] };
        const head_dim = @divExact(self.hidden_size, self.num_heads);
        const merge_size = self.spatial_merge_size;
        const merged_h = @divExact(h, merge_size);
        const merged_w = @divExact(w, merge_size);

        const grid_shape = zml.Shape.init(.{ .bh = merged_h, .bw = merged_w, .mh = merge_size, .mw = merge_size }, .i64);
        const row_ids = zml.Tensor.iota(grid_shape, .bh).scale(merge_size).add(zml.Tensor.iota(grid_shape, .mh)).merge(.{ .hw = .{ .bh, .bw, .mh, .mw } });
        const col_ids = zml.Tensor.iota(grid_shape, .bw).scale(merge_size).add(zml.Tensor.iota(grid_shape, .mw)).merge(.{ .hw = .{ .bh, .bw, .mh, .mw } });

        const pos_ids_hw = zml.Tensor.stack(&[2]Tensor{ row_ids, col_ids }, 1, .layers).convert(.i64);
        const pos_ids = pos_ids_hw.insertAxes(.hw, .{.t}).broad(zml.Shape.init(.{ .t = t, .hw = row_ids.dim(.hw), .layers = 2 }, .i64)).merge(.{ .p = .{ .t, .hw } });

        const max_hw = @max(h, w);
        const rotary_dim = @divExact(head_dim, 2);
        const pos_range = zml.Tensor.arange(.{ .end = max_hw }, .f32).withTags(.{.pos});
        const inv_freq = zml.nn.invFreq(rotary_dim, self.rope_opts).withTags(.{.inv});
        const freq_table = zml.Tensor.outer(pos_range, inv_freq);
        const rotary_pos_emb = freq_table.gather(.{ .pos = pos_ids }, .{}).merge(.{ .hd_half = .{ .layers, .inv } });

        const rotary_pos_emb_double = zml.Tensor.concatenate(&.{ rotary_pos_emb, rotary_pos_emb }, -1).withTags(.{ .p, .hd });
        const cos = rotary_pos_emb_double.cos();
        const sin = rotary_pos_emb_double.sin();
        return .{ cos, sin };
    }
};

pub const VisionPatchEmbed = struct {
    proj: Conv3d,
    patch_size: i64,
    temporal_patch_size: i64,
    in_channels: i64,

    pub fn init(
        allocator: std.mem.Allocator,
        config: Qwen35.Config,
        store: zml.io.TensorStore.View,
    ) VisionPatchEmbed {
        _ = allocator;
        return .{
            .proj = .init(store.withPrefix("proj"), config),
            .patch_size = config.vision_config.patch_size,
            .temporal_patch_size = config.vision_config.temporal_patch_size,
            .in_channels = config.vision_config.in_channels,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionPatchEmbed)) void {
        Conv3d.unloadBuffers(&self.proj);
    }

    pub fn forward(self: VisionPatchEmbed, pixel_values: Tensor) Tensor {
        const reshaped = pixel_values.splitAxis(.ps, .{ .c = self.in_channels, .t = self.temporal_patch_size, .h = self.patch_size, .w = self.patch_size });
        const target_type = self.proj.weight.dtype();
        const conv_output = self.proj.forward(reshaped.convert(target_type));
        const output = conv_output.rename(.{ .c = .d }).squeeze(.t).squeeze(.h).squeeze(.w);
        return output;
    }
};

pub const Conv3d = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    temporal_stride: i64,
    spatial_stride: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) Conv3d {
        return .{
            .weight = store.createTensor("weight", .{ .c, .in, .kt, .kh, .kw }, null),
            .bias = store.maybeCreateTensor("bias", .{.c}, null),
            .temporal_stride = config.vision_config.temporal_patch_size,
            .spatial_stride = config.vision_config.patch_size,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Conv3d)) void {
        self.weight.deinit();
        if (self.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Conv3d, input: Tensor) Tensor {
        const x = input;
        const input_dims = [3]i64{ x.dim(.t), x.dim(.h), x.dim(.w) };
        const kernel_dims = [3]i64{ self.weight.dim(.kt), self.weight.dim(.kh), self.weight.dim(.kw) };
        var strides: [3]i64 = .{ self.temporal_stride, self.spatial_stride, self.spatial_stride };
        const padding = getPadding(input_dims, kernel_dims, strides);
        var y = x.convolution(
            self.weight,
            .{
                .window_strides = &strides,
                .pad_value = &padding,
                .lhs_dilation = &.{ 1, 1, 1 },
                .rhs_dilation = &.{ 1, 1, 1 },
                .window_reversal = &.{ false, false, false },
                .input_batch_dimension = 0,
                .input_feature_dimension = 1,
                .input_spatial_dimensions = &.{ 2, 3, 4 },
                .kernel_input_feature_dimension = 1,
                .kernel_output_feature_dimension = 0,
                .kernel_spatial_dimensions = &.{ 2, 3, 4 },
                .output_batch_dimension = 0,
                .output_feature_dimension = 1,
                .output_spatial_dimensions = &.{ 2, 3, 4 },
                .feature_group_count = 1,
                .batch_group_count = 1,
            },
        );
        if (self.bias) |b| y = y.add(b.broadcast(y._shape, &.{1}));
        return y;
    }

    fn getPadding(input_dims: [3]i64, kernel_dims: [3]i64, strides: [3]i64) [6]i64 {
        var padding: [6]i64 = undefined;
        inline for (0..3) |i| {
            const out = @divFloor(input_dims[i], strides[i]);
            const total = (out - 1) * strides[i] + kernel_dims[i] - input_dims[i];
            const before = @divFloor(total, 2);
            padding[2 * i] = before;
            padding[2 * i + 1] = total - before;
        }
        return padding;
    }
};

pub const VisionBlock = struct {
    norm1: zml.nn.LayerNorm,
    norm2: zml.nn.LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
    num_heads: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) VisionBlock {
        return .{
            .norm1 = .{ .weight = store.createTensor("norm1.weight", .{.d}, null), .bias = store.createTensor("norm1.bias", .{.d}, null), .eps = 1e-6 },
            .norm2 = .{ .weight = store.createTensor("norm2.weight", .{.d}, null), .bias = store.createTensor("norm2.bias", .{.d}, null), .eps = 1e-6 },
            .attn = VisionAttention.init(store.withPrefix("attn"), config),
            .mlp = VisionMlp.init(store.withPrefix("mlp")),
            .num_heads = config.vision_config.num_heads,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionBlock)) void {
        self.norm1.weight.deinit();
        if (self.norm1.bias) |*bias| bias.deinit();
        self.norm2.weight.deinit();
        if (self.norm2.bias) |*bias| bias.deinit();
        VisionAttention.unloadBuffers(&self.attn);
        VisionMlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: VisionBlock, hidden_states: Tensor, cos: Tensor, sin: Tensor, frame_patch_count: i64) Tensor {
        const attn_output = self.attn.forward(self.norm1.forward(hidden_states), cos, sin, frame_patch_count);
        const x = hidden_states.add(attn_output);
        const mlp_output = self.mlp.forward(self.norm2.forward(x));
        const output = x.add(mlp_output);
        return output;
    }
};

pub const VisionMlp = struct {
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) VisionMlp {
        return .{
            .linear_fc1 = .init(store.withPrefix("linear_fc1").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("linear_fc1").maybeCreateTensor("bias", .{.dout}, null), .d),
            .linear_fc2 = .init(store.withPrefix("linear_fc2").createTensor("weight", .{ .d, .dout }, null), store.withPrefix("linear_fc2").maybeCreateTensor("bias", .{.d}, null), .dout),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionMlp)) void {
        self.linear_fc1.weight.deinit();
        if (self.linear_fc1.bias) |*bias| bias.deinit();
        self.linear_fc2.weight.deinit();
        if (self.linear_fc2.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: VisionMlp, x: Tensor) Tensor {
        const x_tagged = x.withTags(.{ .s, .d });
        const x1 = self.linear_fc1.forward(x_tagged).gelu();
        const x2 = self.linear_fc2.forward(x1);
        return x2;
    }
};

pub const VisionAttention = struct {
    qkv: zml.nn.Linear,
    proj: zml.nn.Linear,

    num_heads: i64,
    head_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) VisionAttention {
        return .{
            .qkv = .init(store.withPrefix("qkv").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("qkv").maybeCreateTensor("bias", .{.dout}, null), .d),
            .proj = .init(store.withPrefix("proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .num_heads = config.vision_config.num_heads,
            .head_dim = @divExact(config.vision_config.hidden_size, config.vision_config.num_heads),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionAttention)) void {
        self.qkv.weight.deinit();
        if (self.qkv.bias) |*bias| bias.deinit();
        self.proj.weight.deinit();
        if (self.proj.bias) |*bias| bias.deinit();
    }

    // NB: There is no cross-frame attention, so we split frames into separate batches around the sdpa call.
    pub fn forward(self: VisionAttention, hidden_states: Tensor, cos: Tensor, sin: Tensor, frame_patch_count: i64) Tensor {
        const qkv = self.qkv.forward(hidden_states);

        const qkv_split = qkv.splitAxis(.dout, .{ .qkv = 3, .h = self.num_heads, .hd = self.head_dim });
        const qkv_permuted = qkv_split.transpose(.{ .qkv, .p, .h, .hd });
        const q_, const k_, const v_ = qkv_permuted.chunkExact(.qkv, 3);
        const q = q_.rename(.{ .qkv = .bs, .p = .q });
        const k = k_.rename(.{ .qkv = .bs, .p = .k });
        const v = v_.rename(.{ .qkv = .bs, .p = .k });

        const q_embed, const k_embed = applyRope(q, k, cos, sin);
        const frame_count = @divExact(q_embed.dim(.q), frame_patch_count);
        const q_frames = q_embed.splitAxis(.q, .{ .t = frame_count, .q = frame_patch_count }).merge(.{ .bs = .{ .bs, .t } });
        const k_frames = k_embed.splitAxis(.k, .{ .t = frame_count, .k = frame_patch_count }).merge(.{ .bs = .{ .bs, .t } });
        const v_frames = v.splitAxis(.k, .{ .t = frame_count, .k = frame_patch_count }).merge(.{ .bs = .{ .bs, .t } });

        const attn_output = zml.nn.sdpa(q_frames, k_frames, v_frames, .{ .allow_cudnn = true });
        const attn = attn_output.splitAxis(.bs, .{ .bs = 1, .t = frame_count }).squeeze(.bs).merge(.{ .p = .{ .t, .q }, .d = .{ .h, .hd } });

        const output = self.proj.forward(attn).rename(.{ .dout = .d });
        return output;
    }

    fn rotate_half(x: Tensor) Tensor {
        const x1 = x.slice1d(-1, .{ .start = 0, .end = @divExact(x.dim(-1), 2) });
        const x2 = x.slice1d(-1, .{ .start = @divExact(x.dim(-1), 2), .end = x.dim(-1) });
        return Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    fn applyRope(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) struct { Tensor, Tensor } {
        const cos_q = cos.rename(.{ .p = .q }).insertAxes(.hd, .{.h}).insertAxes(.q, .{.bs}).broad(q.shape());
        const sin_q = sin.rename(.{ .p = .q }).insertAxes(.hd, .{.h}).insertAxes(.q, .{.bs}).broad(q.shape());
        const cos_k = cos.rename(.{ .p = .k }).insertAxes(.hd, .{.h}).insertAxes(.k, .{.bs}).broad(k.shape());
        const sin_k = sin.rename(.{ .p = .k }).insertAxes(.hd, .{.h}).insertAxes(.k, .{.bs}).broad(k.shape());

        const q_1 = q.convert(cos.dtype());
        const k_1 = k.convert(cos.dtype());
        const q_2 = q_1.mul(cos_q).add(rotate_half(q_1).mul(sin_q));
        const k_2 = k_1.mul(cos_k).add(rotate_half(k_1).mul(sin_k));

        return .{ q_2.convert(q.dtype()), k_2.convert(k.dtype()) };
    }
};

pub const VisionPatchMerger = struct {
    norm: zml.nn.LayerNorm,
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,
    spatial_merge_unit: i64,
    out_hidden_size: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) VisionPatchMerger {
        return .{
            .norm = .{ .weight = store.createTensor("norm.weight", .{.d}, null), .bias = store.createTensor("norm.bias", .{.d}, null), .eps = 1e-6 },
            .linear_fc1 = .init(store.withPrefix("linear_fc1").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("linear_fc1").maybeCreateTensor("bias", .{.dout}, null), .d),
            .linear_fc2 = .init(store.withPrefix("linear_fc2").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("linear_fc2").maybeCreateTensor("bias", .{.dout}, null), .d),
            .spatial_merge_unit = config.vision_config.spatial_merge_size * config.vision_config.spatial_merge_size,
            .out_hidden_size = config.vision_config.hidden_size,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionPatchMerger)) void {
        self.norm.weight.deinit();
        if (self.norm.bias) |*bias| bias.deinit();
        self.linear_fc1.weight.deinit();
        if (self.linear_fc1.bias) |*bias| bias.deinit();
        self.linear_fc2.weight.deinit();
        if (self.linear_fc2.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: VisionPatchMerger, x: Tensor) Tensor {
        const x1 = self.norm.forward(x)
            .splitAxis(.p, .{ .p = -1, .merge = self.spatial_merge_unit })
            .merge(.{ .d = .{ .merge, .d } });
        const x2 = self.linear_fc1.forward(x1).rename(.{ .dout = .d });
        const x3 = x2.gelu();
        const x4 = self.linear_fc2.forward(x3).rename(.{ .dout = .d });
        return x4.rename(.{ .p = .s });
    }
};

//========================Text model========================

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub fn init(
        allocator: std.mem.Allocator,
        store: zml.io.TensorStore.View,
        config: Qwen35.Config,
    ) !TextModel {
        const layers = try allocator.alloc(TransformerLayer, @intCast(config.text_config.num_hidden_layers));
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, i| {
            layer.* = .init(store.withPrefix("layers").withLayer(i), config, i);
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, null) },
            .layers = layers,
            .norm = RmsNorm.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn deinit(self: TextModel, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TextModel), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: TextModel,
        input_embeds: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        position_ids: Tensor,
    ) struct { Tensor, KvCache } {
        var hidden_states = input_embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, updated_kv_cache.atLayer(i), position_ids);
        }

        hidden_states = self.norm.forward(hidden_states);
        return .{ hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const TransformerLayer = struct {
    const Attn = union(enum) {
        self_attn: SelfAttn,
        linear_attn: GatedDeltaNet,
    };

    input_layernorm: RmsNorm,
    attn: Attn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config, layer_index: usize) TransformerLayer {
        const is_full_attention = config.text_config.layer_types[layer_index] == .full_attention;
        return .{
            .input_layernorm = RmsNorm.init(store.withPrefix("input_layernorm"), config.text_config.rms_norm_eps),
            .attn = if (is_full_attention)
                .{ .self_attn = .init(store.withPrefix("self_attn"), config) }
            else
                .{ .linear_attn = .init(store.withPrefix("linear_attn"), config) },
            .mlp = .init(store.withPrefix("mlp")),
            .post_attention_layernorm = RmsNorm.init(store.withPrefix("post_attention_layernorm"), config.text_config.rms_norm_eps),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        switch (self.attn) {
            .self_attn => |*self_attn| SelfAttn.unloadBuffers(self_attn),
            .linear_attn => |*linear_attn| GatedDeltaNet.unloadBuffers(linear_attn),
        }
        Mlp.unloadBuffers(&self.mlp);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache.LayerView,
        position_ids: Tensor,
    ) struct { Tensor, KvCache } {
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        var attention_output: Tensor = undefined;
        var updated_kv_cache: KvCache = kv_cache.parent;
        switch (self.attn) {
            .self_attn => |*self_attn| {
                const result = self_attn.forward(normalized_x0, token_index, kv_cache.cache.self_attn, position_ids);
                attention_output = result[0];
                updated_kv_cache.self_attn = result[1];
            },
            .linear_attn => |*linear_attn| {
                const result = linear_attn.forward(normalized_x0, kv_cache.cache.linear_attn);
                attention_output = result[0];
                updated_kv_cache.gated_delta_net = result[1];
            },
        }

        const x1 = attention_output.add(residual0);
        const residual1 = x1;
        const normalized_hidden = self.post_attention_layernorm.forward(x1);

        const mlp_output = self.mlp.forward(normalized_hidden);

        const output = mlp_output.add(residual1);
        return .{ output, updated_kv_cache };
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) Mlp {
        return .{
            .up_proj = .init(store.withPrefix("up_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("up_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .gate_proj = .init(store.withPrefix("gate_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("gate_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .down_proj = .init(store.withPrefix("down_proj").createTensor("weight", .{ .d, .dout }, null), store.withPrefix("down_proj").maybeCreateTensor("bias", .{.d}, null), .dout),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        self.up_proj.weight.deinit();
        if (self.up_proj.bias) |*bias| bias.deinit();
        self.gate_proj.weight.deinit();
        if (self.gate_proj.bias) |*bias| bias.deinit();
        self.down_proj.weight.deinit();
        if (self.down_proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const up_projed = self.up_proj.forward(x);
        const gate = self.gate_proj.forward(x);
        const hidden = gate.silu().mul(up_projed);

        const output = self.down_proj.forward(hidden);
        return output;
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    num_heads: i64,
    num_kv_heads: i64,
    head_dim: i64,
    rotary_dim: i64,
    rotary_embed: TextRotaryEmbedding,
    o_proj: zml.nn.Linear,

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensor("weight", .{ .dout, .d }, null), store.maybeCreateTensor("bias", .{.dout}, null), .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) SelfAttn {
        const rotary_dim: i64 = @intFromFloat(@as(f32, @floatFromInt(config.text_config.head_dim)) *
            config.text_config.rope_parameters.partial_rotary_factor);
        return .{
            .q_proj = initProj(store.withPrefix("q_proj")),
            .k_proj = initProj(store.withPrefix("k_proj")),
            .v_proj = initProj(store.withPrefix("v_proj")),
            .o_proj = initProj(store.withPrefix("o_proj")),
            .q_norm = RmsNorm.init(store.withPrefix("q_norm"), config.text_config.rms_norm_eps),
            .k_norm = RmsNorm.init(store.withPrefix("k_norm"), config.text_config.rms_norm_eps),
            .num_heads = config.text_config.num_attention_heads,
            .num_kv_heads = config.text_config.num_key_value_heads,
            .head_dim = config.text_config.head_dim,
            .rotary_dim = rotary_dim,
            .rotary_embed = .init(rotary_dim, config.text_config.rope_parameters.rope_theta, config.text_config.rope_parameters.mrope_section),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SelfAttn)) void {
        self.q_proj.weight.deinit();
        if (self.q_proj.bias) |*bias| bias.deinit();
        self.k_proj.weight.deinit();
        if (self.k_proj.bias) |*bias| bias.deinit();
        self.v_proj.weight.deinit();
        if (self.v_proj.bias) |*bias| bias.deinit();
        self.o_proj.weight.deinit();
        if (self.o_proj.bias) |*bias| bias.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    fn projectQAndGate(self: SelfAttn, x: Tensor) struct { Tensor, Tensor } {
        const q_proj = self.q_proj.forward(x).splitAxis(.dout, .{ .h = self.num_heads, .hd = 2 * self.head_dim });
        const q, var gate = q_proj.chunkExact(.hd, 2);
        gate = gate.merge(.{ .d_out_proj = .{ .h, .hd } });
        return .{ q, gate };
    }

    fn projectKV(self: SelfAttn, x: Tensor) struct { Tensor, Tensor } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        const k = self.k_proj.forward(x).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        const v = self.v_proj.forward(x).splitAxis(.dout, .{ .h = num_kv_heads, .hd = self.head_dim });
        return .{ k, v };
    }

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache.SelfAttnCache,
        position_ids: Tensor,
    ) struct { Tensor, KvCache.SelfAttnCache } {
        var q, const gate = self.projectQAndGate(x);
        var k, var v = self.projectKV(x);
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();

        const cos, const sin = self.rotary_embed.getMropeCosAndSin(position_ids, dtype);
        q = self.rotary_embed.applyMrope(q, cos, sin);
        k = self.rotary_embed.applyMrope(k, cos, sin);

        const new_kv_cache = kv_cache.update(k, v, token_index.convert(.u32));
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.attention.attention.attention(
            q.rename(.{ .s = .q }),
            k.rename(.{ .s = .k }),
            v.rename(.{ .s = .k }),
            token_index,
            zml.attention.attention.Metadata.init(.fromBackend(.vanilla, x.dim(.s), self.num_heads)),
            zml.attention.attention.Parameters.init(.fromBackend(.vanilla)),
        ).rename(.{ .q = .s }).merge(.{ .d_out_proj = .{ .h, .hd } });

        const gated_output = attn_output.mul(gate.sigmoid());
        const projected_output = self.o_proj.forward(gated_output.rename(.{ .d_out_proj = .d })).rename(.{ .dout = .d });

        return .{ projected_output, new_kv_cache };
    }
};

// Handles Multimodal ROPE details
pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    rotary_dim: i64,
    mrope_section: [3]i64,

    pub fn init(rotary_dim: i64, theta: f32, mrope_section: [3]i64) TextRotaryEmbedding {
        return .{
            .rope_opts = .{
                .layout = .sequential,
                .scaling = .{ .default = .{ .rope_theta = theta } },
            },
            .rotary_dim = rotary_dim,
            .mrope_section = mrope_section,
        };
    }

    pub fn getMropeCosAndSin(self: TextRotaryEmbedding, position_ids: Tensor, dtype: zml.DataType) struct { Tensor, Tensor } {
        // position_ids can have shape (b, s) or (b, s, 3) depending on whether we are in text-only or multimodal case.
        // In the multimodal case, the last dimension corresponds to t, h, w dimensions for the 3D rotary embedding.
        // We handle both cases by stacking the position ids in the text-only case to create a fake 3D rotary embedding.
        const stacked_position_ids = if (position_ids.shape().rank() == 3) position_ids else Tensor.stack(&.{ position_ids, position_ids, position_ids }, 0, .g);

        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        var freqs = stacked_position_ids.convert(.f32).outer(inv_freq);
        var freqs_t, var freqs_h, var freqs_w = freqs.chunkExact(.g, 3);
        freqs_t = freqs_t.squeeze(.g);
        freqs_h = freqs_h.squeeze(.g);
        freqs_w = freqs_w.squeeze(.g);

        const h_indices = Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[1] }, .i64), .h).scale(3).addConstant(1);
        const w_indices = Tensor.iota(zml.Shape.init(.{ .h = self.mrope_section[2] }, .i64), .h).scale(3).addConstant(2);

        const h_input = freqs_h.gather(.{ .hd = h_indices }, .{ .indices_are_sorted = true });
        const w_input = freqs_w.gather(.{ .hd = w_indices }, .{ .indices_are_sorted = true });
        freqs_t = freqs_t.scatterSlices(.{ .hd = h_indices }, h_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        freqs = freqs_t.scatterSlices(.{ .hd = w_indices }, w_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });

        const emb = Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    fn rotateHalf(x: Tensor) Tensor {
        const half_dim = @divExact(x.dim(-1), 2);
        const x1 = x.slice1d(-1, .{ .start = 0, .end = half_dim });
        const x2 = x.slice1d(-1, .{ .start = half_dim, .end = x.dim(-1) });
        return Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    pub fn applyMrope(self: TextRotaryEmbedding, x: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x_rot = x.slice1d(-1, .{ .start = 0, .end = self.rotary_dim });
        const x_pass = x.slice1d(-1, .{ .start = self.rotary_dim, .end = x.dim(-1) });

        const cos_x = cos.insertAxes(.hd, .{.h}).broad(x_rot.shape());
        const sin_x = sin.insertAxes(.hd, .{.h}).broad(x_rot.shape());

        const rotated = x_rot.mul(cos_x).add(rotateHalf(x_rot).mul(sin_x));

        return Tensor.concatenate(&.{ rotated, x_pass }, -1);
    }
};

pub const GatedDeltaNet = struct {
    in_proj_qkv: zml.nn.Linear,
    in_proj_z: zml.nn.Linear,
    in_proj_b: zml.nn.Linear,
    in_proj_a: zml.nn.Linear,
    out_proj: zml.nn.Linear,
    conv1d_weight: Tensor,
    dt_bias: Tensor,
    aLog: Tensor,
    norm: RmsNormGated,

    num_k_heads: i64,
    num_v_heads: i64,
    qk_head_repetition: i64,
    head_k_dim: i64,
    head_v_dim: i64,
    conv_kernel_size: i64,

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensor("weight", .{ .dout, .d }, null), null, .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) GatedDeltaNet {
        const qk_head_repetition =
            @divExact(config.text_config.linear_num_value_heads, config.text_config.linear_num_key_heads);
        return .{
            .in_proj_qkv = initProj(store.withPrefix("in_proj_qkv")),
            .in_proj_z = initProj(store.withPrefix("in_proj_z")),
            .in_proj_b = initProj(store.withPrefix("in_proj_b")),
            .in_proj_a = initProj(store.withPrefix("in_proj_a")),
            .out_proj = initProj(store.withPrefix("out_proj")),
            .conv1d_weight = store.withPrefix("conv1d").createTensor("weight", .{ .out, .in, .kernel_size }, null),
            .dt_bias = store.createTensor("dt_bias", .{.vh}, null),
            .aLog = store.createTensor("A_log", .{.vh}, null),
            .norm = RmsNormGated.init(store.withPrefix("norm"), config.text_config.rms_norm_eps),
            .num_k_heads = config.text_config.linear_num_key_heads,
            .num_v_heads = config.text_config.linear_num_value_heads,
            .qk_head_repetition = qk_head_repetition,
            .head_k_dim = config.text_config.linear_key_head_dim,
            .head_v_dim = config.text_config.linear_value_head_dim,
            .conv_kernel_size = config.text_config.linear_conv_kernel_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GatedDeltaNet)) void {
        self.in_proj_qkv.weight.deinit();
        self.in_proj_z.weight.deinit();
        self.in_proj_b.weight.deinit();
        self.in_proj_a.weight.deinit();
        self.out_proj.weight.deinit();
        self.conv1d_weight.deinit();
        self.dt_bias.deinit();
        self.aLog.deinit();
        RmsNormGated.unloadBuffers(&self.norm);
    }

    fn recurrent_gated_delta_rule(query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor, initial_state: Tensor) struct { Tensor, Tensor } {
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(query.dim(.khd))));
        const query_norm = zml.nn.normalizeL2(query.rename(.{ .kh = .vh }), 1e-6);
        const key_norm = zml.nn.normalizeL2(key.rename(.{ .kh = .vh }), 1e-6);

        const query_f32, const key_f32, const value_f32, const alpha_f32, const beta_f32 = .{
            query_norm.convert(.f32).scale(scale).rename(.{ .vh = .h, .khd = .k }),
            key_norm.convert(.f32).rename(.{ .vh = .h, .khd = .k }),
            value.convert(.f32).rename(.{ .vh = .h, .vhd = .v }),
            g.convert(.f32).exp().rename(.{ .vh = .h }),
            beta.convert(.f32).rename(.{ .vh = .h }),
        };

        const initial_recurrent_state = initial_state.convert(.f32).transpose(.{ .b, .vh, .vhd, .khd }).rename(.{ .vh = .h, .vhd = .v, .khd = .k });

        const result = zml.nn.GatedDeltaNet.forward(
            query_f32,
            key_f32,
            value_f32,
            alpha_f32,
            beta_f32,
            .{ .s = initial_recurrent_state },
        );

        return .{
            result.outputs.rename(.{ .h = .vh, .v = .vhd }).convert(query.dtype()),
            result.state.s.transpose(.{ .b, .h, .k, .v }).rename(.{ .h = .vh, .k = .khd, .v = .vhd }),
        };
    }

    fn buildUpdatedConvState(input: Tensor, left_pad: i64) Tensor {
        const copy_len = @min(input.dim(.s), left_pad);
        const tail = input.slice1d(.s, .{ .start = input.dim(.s) - copy_len, .end = input.dim(.s) });
        if (copy_len == left_pad) return tail;

        const padding_shape = zml.Shape.init(.{ .b = input.dim(.b), .s = left_pad - copy_len, .mix = input.dim(.mix) }, input.dtype());
        const padding = Tensor.constant(input.dtype().zero()).broad(padding_shape);
        return Tensor.concatenate(&.{ padding, tail }, .s);
    }

    pub fn forward(self: GatedDeltaNet, x: Tensor, cache: KvCache.GatedDeltaNetCache) struct { Tensor, KvCache.GatedDeltaNetCache } {
        const key_dim = self.num_k_heads * self.head_k_dim;
        const value_dim = self.num_v_heads * self.head_v_dim;
        const conv_dim = 2 * key_dim + value_dim;
        const left_pad = self.conv_kernel_size - 1;

        const projected_qkv = self.in_proj_qkv.forward(x).rename(.{ .dout = .mix });
        const conv_input = Tensor.concatenate(&.{ cache.convState(), projected_qkv }, .s);

        const kernel = self.conv1d_weight;
        var mixed_qkv = Tensor.conv1d(
            conv_input,
            kernel,
            .{
                .padding = &.{ left_pad, 0 },
                .input_batch_dimension = 0,
                .input_feature_dimension = 2,
                .input_spatial_dimensions = 1,
                .kernel_output_feature_dimension = 0,
                .kernel_input_feature_dimension = 1,
                .kernel_spatial_dimensions = 2,
                .output_batch_dimension = 0,
                .output_feature_dimension = 2,
                .output_spatial_dimensions = 1,
                .feature_group_count = conv_dim,
            },
        )
            .silu();

        mixed_qkv = mixed_qkv.slice1d(.s, .{ .start = mixed_qkv.dim(.s) - x.dim(.s), .end = mixed_qkv.dim(.s) });

        const z = self.in_proj_z.forward(x).splitAxis(.dout, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });
        const b = self.in_proj_b.forward(x).rename(.{ .dout = .vh });
        const a = self.in_proj_a.forward(x).rename(.{ .dout = .vh });

        const query = mixed_qkv
            .slice1d(.mix, .{ .start = 0, .end = key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const key = mixed_qkv
            .slice1d(.mix, .{ .start = key_dim, .end = 2 * key_dim })
            .splitAxis(.mix, .{ .kh = self.num_k_heads, .khd = self.head_k_dim });
        const value = mixed_qkv
            .slice1d(.mix, .{ .start = 2 * key_dim, .end = 2 * key_dim + value_dim })
            .splitAxis(.mix, .{ .vh = self.num_v_heads, .vhd = self.head_v_dim });

        const beta = b.sigmoid();
        const aLog_type = self.aLog.dtype();
        const g = self.aLog.broad(a.shape()).exp().mul(softplus(a.convert(aLog_type).add(self.dt_bias.convert(aLog_type).broad(a.shape())))).negate();

        const query_for_rule = if (self.qk_head_repetition == 1) query else query.stutter1d(@intCast(query.axis(.kh)), @intCast(self.qk_head_repetition));
        const key_for_rule = if (self.qk_head_repetition == 1) key else key.stutter1d(@intCast(key.axis(.kh)), @intCast(self.qk_head_repetition));

        const core_attn_out, const last_recurrent_state = recurrent_gated_delta_rule(
            query_for_rule,
            key_for_rule,
            value,
            g,
            beta,
            cache.recurrentState(),
        );

        const core_attn_out_normed = self.norm
            .forward(
                core_attn_out.rename(.{ .vhd = .d }),
                z.rename(.{ .vhd = .d }),
            )
            .rename(.{ .d = .vhd });

        const output = self.out_proj.forward(core_attn_out_normed.merge(.{ .d = .{ .vh, .vhd } })).rename(.{ .dout = .d });
        const updated_cache = cache.update(
            buildUpdatedConvState(conv_input, left_pad),
            last_recurrent_state,
        );
        return .{ output, updated_cache };
    }
};

pub const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, null), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNorm, x: Tensor) Tensor {
        const x_f32 = x.convert(.f32);
        const weight_f32 = self.weight.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        const output = normalized.mul(weight_f32.broad(x.shape())).add(normalized).convert(x.dtype());
        return output;
    }
};

pub const RmsNormGated = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNormGated {
        return .{ .weight = store.createTensor("weight", .{.d}, null), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNormGated)) void {
        self.weight.deinit();
    }

    pub fn forward(self: RmsNormGated, x: Tensor, gate: Tensor) Tensor {
        const x_f32 = x.convert(.f32);
        const gate_f32 = gate.convert(.f32);

        const normalized = zml.nn.rmsNorm(x_f32, .d, self.eps);
        const output = normalized.mul(self.weight.broad(x.shape()));

        const gated_output = output.mul(gate_f32.silu());
        const result = gated_output.convert(x.dtype());
        return result;
    }
};

pub const KvCache = struct {
    layer_types: []const Qwen35.LayerType,
    self_attn: SelfAttnCache,
    gated_delta_net: GatedDeltaNetCache,

    pub const SelfAttnCache = struct {
        k: Tensor,
        v: Tensor,
        layer_index: Tensor,

        pub fn init(config: Qwen35.Config, batch_dim: i64, max_seq_len: i64, dtype: zml.DataType) SelfAttnCache {
            const num_self_attn_layers = countLayers(config.text_config.layer_types, .full_attention);
            const kv_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_self_attn_layers,
                .s = max_seq_len,
                .h = config.text_config.num_key_value_heads,
                .hd = config.text_config.head_dim,
            }, dtype);
            return .{
                .k = .fromShape(kv_shape),
                .v = .fromShape(kv_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: SelfAttnCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(SelfAttnCache) {
            const sharding = try zml.sharding.replicatedSharding(platform);
            return .{
                .k = try KvCache.zeroInitializedBuffer(io, platform, self.k.shape(), sharding),
                .v = try KvCache.zeroInitializedBuffer(io, platform, self.v.shape(), sharding),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(SelfAttnCache)) void {
            self.k.deinit();
            self.v.deinit();
            self.layer_index.deinit();
        }

        pub fn keys(self: SelfAttnCache) Tensor {
            return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn values(self: SelfAttnCache) Tensor {
            return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn update(self: SelfAttnCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) SelfAttnCache {
            const k_shape = self.k.shape().drop(.layer);
            var layer = self.layer_index;
            layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

            return if (token_index) |idx| .{
                .k = self.k.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_k.convert(self.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.k),
                .v = self.v.scatterSlices(
                    .{ .layer = layer, .s = idx },
                    new_v.convert(self.v.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.v),
                .layer_index = self.layer_index,
            } else .{
                .k = self.k.scatterSlices(
                    .{ .layer = layer },
                    new_k.convert(self.k.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.k),
                .v = self.v.scatterSlices(
                    .{ .layer = layer },
                    new_v.convert(self.v.dtype()).transpose(k_shape),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.v),
                .layer_index = self.layer_index,
            };
        }

        pub fn atLayer(self: SelfAttnCache, layer_index: usize) SelfAttnCache {
            return .{
                .k = self.k,
                .v = self.v,
                .layer_index = Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: SelfAttnCache, other: SelfAttnCache) SelfAttnCache {
            return .{
                .k = self.k.reuseBuffer(other.k),
                .v = self.v.reuseBuffer(other.v),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub const GatedDeltaNetCache = struct {
        conv_state: Tensor,
        recurrent_state: Tensor,
        layer_index: Tensor,

        pub fn init(config: Qwen35.Config, batch_dim: i64, conv_dtype: zml.DataType, recurrent_dtype: zml.DataType) GatedDeltaNetCache {
            const num_linear_attn_layers = countLayers(config.text_config.layer_types, .linear_attention);
            const conv_dim = 2 * config.text_config.linear_num_key_heads * config.text_config.linear_key_head_dim + config.text_config.linear_num_value_heads * config.text_config.linear_value_head_dim;
            const conv_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .s = config.text_config.linear_conv_kernel_dim - 1,
                .mix = conv_dim,
            }, conv_dtype);
            const recurrent_state_shape = zml.Shape.init(.{
                .b = batch_dim,
                .layer = num_linear_attn_layers,
                .vh = config.text_config.linear_num_value_heads,
                .khd = config.text_config.linear_key_head_dim,
                .vhd = config.text_config.linear_value_head_dim,
            }, recurrent_dtype);
            return .{
                .conv_state = .fromShape(conv_state_shape),
                .recurrent_state = .fromShape(recurrent_state_shape),
                .layer_index = .init(.{}, .u32),
            };
        }

        pub fn initBuffer(self: GatedDeltaNetCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(GatedDeltaNetCache) {
            const sharding = try zml.sharding.replicatedSharding(platform);
            return .{
                .conv_state = try KvCache.zeroInitializedBuffer(io, platform, self.conv_state.shape(), sharding),
                .recurrent_state = try KvCache.zeroInitializedBuffer(io, platform, self.recurrent_state.shape(), sharding),
                .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
            };
        }

        pub fn deinitBuffer(self: *zml.Bufferized(GatedDeltaNetCache)) void {
            self.conv_state.deinit();
            self.recurrent_state.deinit();
            self.layer_index.deinit();
        }

        pub fn convState(self: GatedDeltaNetCache) Tensor {
            return self.conv_state.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn recurrentState(self: GatedDeltaNetCache) Tensor {
            return self.recurrent_state.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
        }

        pub fn update(self: GatedDeltaNetCache, new_conv_state: ?Tensor, new_recurrent_state: ?Tensor) GatedDeltaNetCache {
            const conv_state = if (new_conv_state) |state|
                self.conv_state.scatterSlices(
                    .{ .layer = self.layer_index },
                    state.convert(self.conv_state.dtype()).transpose(self.conv_state.shape().drop(.layer)),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.conv_state)
            else
                self.conv_state;

            const recurrent_state = if (new_recurrent_state) |state|
                self.recurrent_state.scatterSlices(
                    .{ .layer = self.layer_index },
                    state.convert(self.recurrent_state.dtype()).transpose(self.recurrent_state.shape().drop(.layer)),
                    .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
                ).reuseBuffer(self.recurrent_state)
            else
                self.recurrent_state;

            return .{
                .conv_state = conv_state,
                .recurrent_state = recurrent_state,
                .layer_index = self.layer_index,
            };
        }

        pub fn atLayer(self: GatedDeltaNetCache, layer_index: usize) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state,
                .recurrent_state = self.recurrent_state,
                .layer_index = Tensor.scalar(layer_index, .u32),
            };
        }

        pub fn reuseBuffer(self: GatedDeltaNetCache, other: GatedDeltaNetCache) GatedDeltaNetCache {
            return .{
                .conv_state = self.conv_state.reuseBuffer(other.conv_state),
                .recurrent_state = self.recurrent_state.reuseBuffer(other.recurrent_state),
                .layer_index = self.layer_index.reuseBuffer(other.layer_index),
            };
        }
    };

    pub fn init(
        config: Qwen35.Config,
        batch_dim: i64,
        max_seq_len: i64,
        cache_dtype: zml.DataType,
        recurrent_dtype: zml.DataType,
    ) KvCache {
        return .{
            .layer_types = config.text_config.layer_types,
            .self_attn = SelfAttnCache.init(config, batch_dim, max_seq_len, cache_dtype),
            .gated_delta_net = GatedDeltaNetCache.init(config, batch_dim, cache_dtype, recurrent_dtype),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .self_attn = try self.self_attn.initBuffer(io, platform),
            .gated_delta_net = try self.gated_delta_net.initBuffer(io, platform),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        SelfAttnCache.deinitBuffer(&self.self_attn);
        GatedDeltaNetCache.deinitBuffer(&self.gated_delta_net);
    }

    // Self-attn cache buffers are zero-initialized because the vanilla attention is not safe against garbage values in the cache
    fn zeroInitializedBuffer(
        io: std.Io,
        platform: *const zml.Platform,
        shape: zml.Shape,
        sharding: zml.sharding.Sharding,
    ) !zml.Buffer {
        const allocator = std.heap.page_allocator;
        const pool_count = platform.devices.len;

        var dma_allocators = try allocator.alloc(zml.mem.DmaAllocator, pool_count);
        defer allocator.free(dma_allocators);
        for (platform.devices, 0..) |*device, i| {
            dma_allocators[i] = .init(allocator, device);
        }

        const buffer_pools = try allocator.alloc(zml.mem.DynamicBufferPool, pool_count);
        defer allocator.free(buffer_pools);
        for (buffer_pools) |*pool| {
            pool.* = .init(2, 4096);
        }
        defer for (buffer_pools, 0..) |*pool, i| {
            pool.deinit(dma_allocators[i].allocator());
        };

        var out: zml.Buffer = undefined;
        var writer = try zml.io.MemoryWriter.init(
            allocator,
            io,
            platform,
            buffer_pools,
            dma_allocators,
            shape,
            sharding,
            &out,
        );
        defer writer.deinit(allocator);

        try writeAllZeros(writer.interface(), shape.byteSize());
        try writer.interface().flush();
        return out;
    }

    fn writeAllZeros(writer: *std.Io.Writer, len: usize) !void {
        const zero_chunk: [4096]u8 = [_]u8{0} ** 4096;
        var remaining = len;
        while (remaining > 0) {
            const chunk_len = @min(remaining, zero_chunk.len);
            try writer.writeAll(zero_chunk[0..chunk_len]);
            remaining -= chunk_len;
        }
    }

    pub const LayerView = struct {
        parent: KvCache,
        cache: union(enum) {
            self_attn: SelfAttnCache,
            linear_attn: GatedDeltaNetCache,
        },
    };

    pub fn atLayer(self: KvCache, layer_index: usize) LayerView {
        return switch (getDenseIndex(self.layer_types, layer_index)) {
            .full_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .self_attn = self.self_attn.atLayer(dense_index.layer_dense_index) },
            },
            .linear_attention => |dense_index| .{
                .parent = self,
                .cache = .{ .linear_attn = self.gated_delta_net.atLayer(dense_index.layer_dense_index) },
            },
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .layer_types = self.layer_types,
            .self_attn = self.self_attn.reuseBuffer(other.self_attn),
            .gated_delta_net = self.gated_delta_net.reuseBuffer(other.gated_delta_net),
        };
    }

    fn countLayers(layer_types: []const Qwen35.LayerType, layer_type: Qwen35.LayerType) i64 {
        var count: i64 = 0;
        for (layer_types) |registered_layer_type| {
            if (registered_layer_type == layer_type) count += 1;
        }
        return count;
    }

    fn getDenseIndex(layer_types: []const Qwen35.LayerType, layer_index: usize) union(enum) {
        full_attention: struct { layer_dense_index: usize },
        linear_attention: struct { layer_dense_index: usize },
    } {
        var self_attn_layer_index: usize = 0;
        var linear_attn_layer_index: usize = 0;
        for (layer_types[0..layer_index]) |layer_type| {
            switch (layer_type) {
                .full_attention => self_attn_layer_index += 1,
                .linear_attention => linear_attn_layer_index += 1,
            }
        }
        return switch (layer_types[layer_index]) {
            .full_attention => .{ .full_attention = .{ .layer_dense_index = self_attn_layer_index } },
            .linear_attention => .{ .linear_attention = .{ .layer_dense_index = linear_attn_layer_index } },
        };
    }
};

//========================Utils========================

fn softplus(x: Tensor) Tensor {
    return x.exp().addConstant(1).log();
}
