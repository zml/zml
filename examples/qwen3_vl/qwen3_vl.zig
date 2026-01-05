const std = @import("std");
const testing = std.testing;
const async = @import("async");
const stdx = @import("stdx");
const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const Linear = zml.nn.Linear;
const Shape = zml.Shape;
const log = std.log.scoped(.qwen3_vl);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = async.logFn(std.log.defaultLog),
};

test {
    std.testing.refAllDecls(@This());
}

pub const Qwen3VL = struct {
    qwen: Qwen,

    pub fn init(
        allocator: std.mem.Allocator,
        config: Qwen.Config,
        options: Qwen.Options,
        store: zml.aio.BufferStore,
    ) !Qwen3VL {
        return .{
            .qwen = try Qwen.init(allocator, config, options, store),
        };
    }

    // Forward pass for the prefill phase
    // image_hwc: Tensor, the image 3 dim Tensor (height, width, channels)
    // input_ids: Tensor
    // image_dim: Tensor, the image dimension (single dim vector with the the resizedshape of the image)
    // token_index: Tensor, (0 for the prefill because we consider the whole sequence)
    // prompt_shape: Tensor, the prompt shape
    // kv_cache: KvCache, key-value cache
    // h_resized: u32, height of the resized image (given at compilation time but not used in execution)
    // w_resized: u32, width of the resized image (given at compilation time but not used in execution)
    pub fn forward(
        self: Qwen3VL,
        image_hwc: Tensor,
        input_ids: Tensor,
        image_dim: Tensor,
        token_index: Tensor,
        prompt_shape: Tensor,
        kv_cache: KvCache,
        h_resized: u32,
        w_resized: u32,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor, Tensor.Rng } {
        const pixel_value, const image_grid_thw = self.processImage(image_hwc, image_dim, h_resized, w_resized);
        const next_token, const updated_cache, const mrope_position_deltas, const new_rng = zml.call(self.qwen, .forward, .{ input_ids, pixel_value, token_index, image_grid_thw, kv_cache, prompt_shape, rng });

        return .{ next_token, updated_cache, mrope_position_deltas, new_rng };
    }

    pub fn processImage(
        self: Qwen3VL,
        image_hwc: Tensor,
        image_size: Tensor,
        h_resized: u32,
        w_resized: u32,
    ) struct { Tensor, [3]u32 } {

        // Resize the image and transpose (channels, height, width)
        var image_chw = ResizeBicubic(image_hwc, .{ .h = h_resized, .w = w_resized }, .{ .original_len = image_size }).transpose(.{ .c, .h, .w });
        image_chw = image_chw.convert(.f32);

        // Rescale and normalize the image
        const rescale_factor: f32 = 1.0 / 255.0;
        const image_mean: f32 = 0.5;
        const image_std: f32 = 0.5;
        image_chw = image_chw.scale(rescale_factor); // pixel / 255.0
        var image_chw_rescaled_normalized = image_chw.sub(Tensor.scalar(image_mean, .f32)).div(Tensor.scalar(image_std, .f32));

        // Introduce the temporal dimension (1)
        image_chw_rescaled_normalized = image_chw_rescaled_normalized.reshape(.{ .c = 3, .temporal_patch_size = 1, .h = h_resized, .w = w_resized });
        const temporal_patch_size = self.qwen.config.vision_config.temporal_patch_size;

        // Repeat the image 2 times in the temporal dimension
        image_chw_rescaled_normalized = image_chw_rescaled_normalized.repeat1d(1, 2);
        const patch_size = self.qwen.config.vision_config.patch_size;

        //Hardcoded because we only have 1 temporal patch (image)
        const grid_t = 1;
        // Compute the number of grid cells based on the patch size (size of the patch embedding)
        const grid_h: u32 = @intCast(@as(u32, @divExact(h_resized, patch_size)));
        const grid_w: u32 = @intCast(@as(u32, @divExact(w_resized, patch_size)));
        const grid_thw = [3]u32{ grid_t, grid_h, grid_w };

        const merge_size = self.qwen.config.vision_config.spatial_merge_size;

        // Split height axis: h -> h_div, m1, patch1
        image_chw_rescaled_normalized = image_chw_rescaled_normalized.splitAxis(.h, .{
            .h_div = @divExact(grid_h, merge_size),
            .m1 = merge_size,
            .patch1 = patch_size,
        });

        // Split width axis: w -> w_div, m2, patch2
        image_chw_rescaled_normalized = image_chw_rescaled_normalized.splitAxis(.w, .{
            .w_div = @divExact(grid_w, merge_size),
            .m2 = merge_size,
            .patch2 = patch_size,
        });

        // After splitting axes, the shape is :
        // .{ .temporal_patch_size = temporal_patch_size, .c = 3, .h_div = @divExact(grid_h, merge_size), .m1 = merge_size, .patch1 = patch_size, .w_div = @divExact(grid_w, merge_size), .m2 = merge_size, .patch2 = patch_size }
        image_chw_rescaled_normalized = image_chw_rescaled_normalized.transpose(.{ .h_div, .w_div, .m1, .m2, .c, .temporal_patch_size, .patch1, .patch2 });
        const flatten_image = image_chw_rescaled_normalized.reshape(.{ .a = grid_h * grid_w, .b = 3 * temporal_patch_size * patch_size * patch_size });

        // Return the flattened image and the grid dimensions
        return .{ flatten_image, grid_thw };
    }

    pub fn forward_decode(
        self: Qwen3VL,
        input_ids: Tensor,
        cache_position: Tensor,
        kv_cache: KvCache,
        mrope_position_deltas: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const next_token, const updated_cache, const new_rng = zml.call(self.qwen, .forward_decode, .{ input_ids, cache_position, kv_cache, mrope_position_deltas, rng });
        const result = .{ next_token.convert(.u32), updated_cache, new_rng };

        return result;
    }
};

/// Qwen3-VL architecture, using huggingface transformers naming.
/// Vision-Language model with vision transformer and text model.
pub const Qwen = struct {
    pub const VisionConfig = struct {
        depth: u32 = 32,
        hidden_size: u32 = 1280,
        hidden_act: []const u8 = "silu",
        intermediate_size: u32 = 3420,
        num_heads: u32 = 16,
        in_channels: u32 = 3,
        patch_size: u32 = 14,
        spatial_merge_size: u32 = 2,
        temporal_patch_size: u32 = 2,
        out_hidden_size: u32 = 2048,
        initializer_range: f32 = 0.02,
        deepstack_visual_indexes: []const u32 = &[_]u32{ 5, 11, 17 },
        num_position_embeddings: u32 = 48 * 48,
    };

    pub const TextConfig = struct {
        hidden_size: u32 = 2560,
        bos_token_id: u32,
        eos_token_id: u32,
        head_dim: i64 = 128,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        tie_word_embeddings: bool = true,
        rope_scaling: RopeScaling = .{ .mrope_section = .{ 24, 20, 20 } },
        rope_theta: f32 = 5000000.0,
    };

    pub const Config = struct {
        vision_config: VisionConfig,
        text_config: TextConfig,
        tie_word_embeddings: bool = true,
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: u32,
    };

    pub const RopeScaling = struct {
        mrope_section: [3]u32 = .{ 24, 20, 20 },
    };

    vision_transformer: VisionTransformer,
    text_model: TextModel,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    // Initialize the Qwen model (Vision and Text models)
    pub fn init(allocator: std.mem.Allocator, config: Config, options: Options, store: zml.aio.BufferStore) !Qwen {
        return .{
            .config = config,
            .gen_opts = options.sampling_strategy orelse .{},
            .vision_transformer = try VisionTransformer.init(allocator, config, store),
            .text_model = try TextModel.init(allocator, config, store),
        };
    }

    // Forward pass for the qwen model
    pub fn forward(
        self: Qwen,
        input_ids: Tensor,
        pixel_values: Tensor,
        cache_position: Tensor,
        image_grid_thw: [3]u32,
        kv_cache: KvCache,
        prompt_shape: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor, Tensor.Rng } {

        // Embed the input ids
        var embedded = zml.call(self.text_model.embed_tokens, .forward, .{input_ids}).withTags(.{ .bs, .seq, .d });

        // Forward pass for the vision transformer

        const vision_embed, const deepstack_features = zml.call(self.vision_transformer, .forward, .{ pixel_values, image_grid_thw });

        // Get the number of text tokens before the image, the number of image tokens and the number of text tokens after the image
        // The number of text tokens before the image (4 tokens according to the chat template)
        const text_before_image = prompt_shape.choose1d(0, 0).convert(.i32);
        // The number of image tokens (number of image tokens in the image grid, after the spatial merge -> number of patch on height dimension * number of patch on width dimension / spatial merge size^2)
        const num_image_tokens = prompt_shape.choose1d(0, 1).convert(.i32);
        // The number of text tokens after the image (prompt tokens + 6 according to the chat template)
        const text_after_image = prompt_shape.choose1d(0, 2).convert(.i32);

        // Update the embedding with the vision embedding
        const text_with_image = embedded.dynamicUpdateSlice(.{ .seq = text_before_image }, zml.torch.unsqueeze(vision_embed.convert(embedded.dtype()), 0));
        const seq_len = text_with_image.dim(.seq);

        // Build the 3D positional ids
        const position_ids, const mrope_position_deltas = buildVisionPositionIds(
            self.config.vision_config.spatial_merge_size,
            input_ids,
            seq_len,
            prompt_shape,
            image_grid_thw,
        );

        const real_seq_len = text_before_image.add(text_after_image).add(num_image_tokens);
        const hidden, const updated_cache = zml.call(self.text_model, .forward, .{ position_ids, text_with_image, cache_position, deepstack_features, kv_cache });

        // Sample the next token using RNG
        const last_pos = real_seq_len.addConstant(-1).asScalar();
        const last_hidden = hidden.dynamicSlice1d(hidden.axis(.seq), .{ .start = last_pos, .len = 1 });
        const last_logits = projectToVocab(last_hidden, self.text_model.embed_tokens.weight);
        const next_token, const new_rng = self.sampleTokens(last_logits, rng);
        const next_token_with_shape = next_token.withTags(.{ .bs, .seq });

        const result = .{ next_token_with_shape, updated_cache, mrope_position_deltas, new_rng };

        return result;
    }

    // Forward decode pass for qwen model
    // Do not recompute the visual embeddings
    pub fn forward_decode(
        self: Qwen,
        input_ids: Tensor,
        cache_position: Tensor,
        kv_cache: KvCache,
        mrope_position_deltas: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const embedded = zml.call(self.text_model.embed_tokens, .forward, .{input_ids}).withTags(.{ .bs, .seq, .d });
        const position_ids = buildDecodePositionIds(cache_position, mrope_position_deltas);
        const hidden, const updated_cache = zml.call(self.text_model, .forward_decode, .{ position_ids, embedded, cache_position, kv_cache });
        const logits = projectToVocab(hidden, self.text_model.embed_tokens.weight);

        // Sample the next token using RNG
        const last_logits = logits.slice1d(.seq, .{ .start = logits.dim(.seq) - 1, .end = logits.dim(.seq) });
        const next_token, const new_rng = self.sampleTokens(last_logits, rng);
        const result = .{ next_token.reuseBuffer(input_ids), updated_cache, new_rng };
        return result;
    }

    fn initKvCache(k: Tensor, v: Tensor, layer_index: Tensor) KvCache {
        return .{
            .k = k.withTags(.{ .layer, .k, .h, .hd }),
            .v = v.withTags(.{ .layer, .k, .h, .hd }),
            .layer_index = layer_index,
        };
    }

    fn projectToVocab(hidden: Tensor, embedding_weight: Tensor) Tensor {
        return hidden.convert(.f32).dotGeneral(embedding_weight.convert(.f32), &.{.{ -1, -1 }}, &.{}).withTags(.{ .bs, .seq, .voc });
    }

    pub fn sampleTokens(
        self: Qwen,
        logits_: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, Tensor.Rng } {
        const logits = logits_.withPartialTags(.{ .bs, .seq, .voc });

        if (logits.shape().hasTag(.voc) == null)
            @panic("logits must have .voc tag");

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, self.gen_opts, rng);
        return .{ next_tokens, new_rng };
    }

    // Build the 3D positional ids for the vision transformer
    // Returns: (stacked_position_ids, mrope_position_deltas)
    pub fn buildVisionPositionIds(
        spatial_merge_size: u32,
        input_ids: Tensor,
        seq_len: i64,
        prompt_shape: Tensor,
        image_grid_thw: [3]u32,
    ) struct { Tensor, Tensor } {
        // Get the number of text tokens before the image, the number of image tokens and the number of text tokens after the image
        const text_before_image = prompt_shape.choose1d(0, 0).convert(.i32);
        const num_image_tokens = prompt_shape.choose1d(0, 1).convert(.i32);
        const text_after_image = prompt_shape.choose1d(0, 2).convert(.i32);

        // Build the 3D positional ids
        const before_image_positions = zml.Tensor.iota(Shape.init(.{ .bs = input_ids.dim(.bs), .seq = 4 }, .i32), .seq);

        const t = image_grid_thw[0];
        const h = @divExact(image_grid_thw[1], spatial_merge_size);
        const w = @divExact(image_grid_thw[2], spatial_merge_size);

        var position_ids = zml.Tensor.iota(
            Shape.init(.{ .bs = input_ids.dim(.bs), .seq = seq_len }, .i32),
            .seq,
        )
            .sub(num_image_tokens)
            .addConstant(@max(w, h, t));
        position_ids = position_ids.dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(0, .i32) }, before_image_positions);

        // Define the shape of the iota tensor
        const iota_shape = Shape.init(.{ .bs = input_ids.dim(.bs), .t = t, .h = h, .w = w }, .i32);
        const reshape_shape = Shape.init(.{ .bs = input_ids.dim(.bs), .seq = t * h * w }, .i32);

        // Repeat the index along the 3 dimensions based on grid size (after the text (+4 tokens according to the chat template))
        const t_index = zml.Tensor.iota(iota_shape, .t).reshape(reshape_shape).add(text_before_image);
        const h_index = zml.Tensor.iota(iota_shape, .h).reshape(reshape_shape).add(text_before_image);
        const w_index = zml.Tensor.iota(iota_shape, .w).reshape(reshape_shape).add(text_before_image);

        // Update the position ids with the 3D positional ids
        const position_ids_t = position_ids.dynamicUpdateSlice(.{ .seq = text_before_image }, t_index);
        const position_ids_h = position_ids.dynamicUpdateSlice(.{ .seq = text_before_image }, h_index);
        const position_ids_w = position_ids.dynamicUpdateSlice(.{ .seq = text_before_image }, w_index);

        // Stack the position ids
        const stacked_position_ids = zml.Tensor.stack(&.{ position_ids_t, position_ids_h, position_ids_w }, 0, .g);

        // Position max after 3d compression - real seq len
        const position_max_after_3d_compression = zml.Tensor.scalar(@max(w, h, t), .i32).add(text_after_image).add(text_before_image);
        const real_seq_len = text_before_image.add(text_after_image).add(num_image_tokens);
        const mrope_position_deltas = position_max_after_3d_compression.sub(real_seq_len).reshape(.{ .seq = 1 });

        return .{ stacked_position_ids, mrope_position_deltas };
    }

    test "buildVisionPositionIds" {
        std.debug.print("buildVisionPositionIds test started\n", .{});

        const platform = zml.testing.env();
        const allocator = std.testing.allocator;

        // Parameters
        const batch_size: u32 = 1;
        const seq_len: u32 = 79;

        // Create input buffers
        var input_ids_data = try allocator.alloc(i32, batch_size * seq_len);
        defer allocator.free(input_ids_data);
        for (0..batch_size * seq_len) |i| {
            input_ids_data[i] = @intCast(i % seq_len);
        }
        const input_ids_d = try zml.Buffer.fromSlice(platform, .{ .bs = batch_size, .seq = seq_len }, input_ids_data);
        defer input_ids_d.deinit();

        const prompt_shape_d = try zml.Buffer.fromSlice(platform, .{ .seq = 3 }, &[_]i32{ 4, 64, 11 });
        defer prompt_shape_d.deinit();

        // Compile and execute buildVisionPositionIds
        const Local = struct {
            pub fn positionIds(input_ids: zml.Tensor, prompt_shape: zml.Tensor) zml.Tensor {
                return buildVisionPositionIds(2, input_ids, seq_len, prompt_shape, .{ 1, 16, 16 })[0];
            }
        };

        const result = try zml.testing.compileAndCall(
            platform,
            Local.positionIds,
            .{ input_ids_d, prompt_shape_d },
        );
        defer result.deinit();

        const expected = [3][79]i32{
            // temporal
            .{ 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 },
            // height
            .{ 0, 1, 2, 3, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 },
            // width
            .{ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22 },
        };
        try std.testing.expectEqual(expected, try result.getValue([3][79]i32));
    }

    // Build 3d positionnal based on mrope delta (delta between the position max after 3d compression and the real sequence length)
    fn buildDecodePositionIds(cache_position: Tensor, mrope_position_deltas: Tensor) Tensor {
        const cache_pos = cache_position.reshape(.{ .bs = 1, .seq = 1 });
        const deltas = mrope_position_deltas.convert(.i64).reshape(.{ .bs = 1, .seq = 1 });
        return zml.Tensor.stack(&.{ cache_pos.add(deltas), cache_pos.add(deltas), cache_pos.add(deltas) }, 0, .g).withTags(.{ .g, .bs, .seq });
    }
};

//========================Vision model========================

pub const VisionTransformer = struct {
    vision_patch_embed: VisionPatchEmbed,
    pos_embed: zml.nn.TokenEmbedding,
    blocks: []VisionBlock,
    patch_merger: PatchMerger,
    deepstack_patch_mergers: []PatchMerger,
    num_heads: u32,
    hidden_size: u32,
    spatial_merge_size: u32,
    num_position_embeddings: u32,
    rope_opts: zml.nn.RopeOpts,

    pub fn init(allocator: std.mem.Allocator, config: Qwen.Config, store: zml.aio.BufferStore) !VisionTransformer {
        const spatial_merge_size = config.vision_config.spatial_merge_size;
        const blocks = try allocator.alloc(VisionBlock, config.vision_config.depth);
        var prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.visual.blocks");
        for (0.., blocks) |i, *block| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            var vision_attn = try zml.aio.populateModelWithPrefix(VisionAttention, allocator, store, prefix.concat("attn"));
            vision_attn.num_heads = config.vision_config.num_heads;

            var mlp = try zml.aio.populateModelWithPrefix(VisionMlp, allocator, store, prefix.concat("mlp"));
            mlp.hidden_act = zml.nn.Activation{ .gelu = {} };

            var norm1 = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm1"));
            norm1.eps = 1e-6;

            var norm2 = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm2"));
            norm2.eps = 1e-6;

            block.* = .{
                .attn = vision_attn,
                .mlp = mlp,
                .norm1 = norm1,
                .norm2 = norm2,
                .num_heads = config.vision_config.num_heads,
            };
        }

        prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.visual.deepstack_merger_list");
        const deepstack_patch_mergers = try allocator.alloc(PatchMerger, config.vision_config.deepstack_visual_indexes.len);
        for (0.., deepstack_patch_mergers) |i, *deepstack_patch_merger| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            const norm = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm"));
            const linear_fc1 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("linear_fc1"));
            const linear_fc2 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, prefix.concat("linear_fc2"));
            deepstack_patch_merger.* = .{
                .norm = norm,
                .linear_fc1 = linear_fc1,
                .linear_fc2 = linear_fc2,
                .out_hidden_size = config.vision_config.out_hidden_size,
            };
        }

        return .{
            .pos_embed = try zml.aio.populateModelWithPrefix(zml.nn.TokenEmbedding, allocator, store, "model.visual.pos_embed"),
            .blocks = blocks,
            .patch_merger = .{
                .norm = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, "model.visual.merger.norm"),
                .linear_fc1 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, "model.visual.merger.linear_fc1"),
                .linear_fc2 = try zml.aio.populateModelWithPrefix(zml.nn.Linear, allocator, store, "model.visual.merger.linear_fc2"),
                .out_hidden_size = config.vision_config.out_hidden_size,
            },
            .num_heads = config.vision_config.num_heads,
            .deepstack_patch_mergers = deepstack_patch_mergers,
            .vision_patch_embed = try VisionPatchEmbed.init(allocator, config.vision_config.patch_size, config.vision_config.temporal_patch_size, config.vision_config.in_channels, config.vision_config.out_hidden_size, store),
            .hidden_size = config.vision_config.hidden_size,
            .spatial_merge_size = spatial_merge_size,
            .num_position_embeddings = config.vision_config.num_position_embeddings,
            .rope_opts = zml.nn.RopeOpts{
                .layout = .sequential,
                .freq_base = 10000.0,
                .scaling = .{ .default = {} },
            },
        };
    }

    // Forward pass for the vision transformer
    // Outputs:
    // - hidden_states: the hidden states of the vision transformer (visual embedding)
    // - deepstack_features_list: the deepstack features (intermediate representation of the visual embedding)
    pub fn forward(self: VisionTransformer, x_input: Tensor, grid_thw: [3]u32) struct { Tensor, [3]Tensor } {
        const x = x_input;
        var pos_embeds = self.posEmbedInterpolate(&grid_thw);
        var rotary_pos_emb = rotaryPosEmbed(&grid_thw, self.spatial_merge_size, self.hidden_size, self.num_heads, self.rope_opts);
        rotary_pos_emb = zml.Tensor.concatenate(&.{ rotary_pos_emb, rotary_pos_emb }, 2);
        var hidden_states = zml.call(self.vision_patch_embed, .forward, .{x});
        hidden_states = hidden_states.add(pos_embeds[0].convert(hidden_states.dtype()));
        const cos = rotary_pos_emb.cos();
        const sin = rotary_pos_emb.sin();
        const deepstack_visual_indexes = [3]u32{ 5, 11, 17 };
        var count: usize = 0;
        var deepstack_features_list: [3]Tensor = undefined;
        for (0.., self.blocks) |layer, block| {
            hidden_states = zml.call(block, .forward, .{ hidden_states, cos, sin });
            for (deepstack_visual_indexes) |index| {
                if (layer == index) {
                    deepstack_features_list[count] = zml.call(self.deepstack_patch_mergers[count], .forward, .{ hidden_states, true });
                    count += 1;
                }
            }
        }
        hidden_states = zml.call(self.patch_merger, .forward, .{ hidden_states, false });
        return .{ hidden_states, deepstack_features_list };
    }

    // Positional embedding interpolation (representation of the image in a grid determined by the number of position embeddings 48 x 48)
    pub fn posEmbedInterpolate(self: VisionTransformer, grid: []const u32) [1]Tensor {
        // Calculate the number of grid points per side (sqrt of the number of position embeddings)
        const num_grid_per_side = std.math.pow(f32, @as(f32, @floatFromInt(self.num_position_embeddings)), 0.5);

        const m_size = self.spatial_merge_size;
        const embedding_dim = self.hidden_size;

        var outputs = [1]Tensor{undefined};

        // Retrieve the dims in the image grid
        const t = grid[0];
        const h = grid[1];
        const w = grid[2];
        const tensor_filled_1_h = zml.Tensor.constant(.{h}, zml.Data.init(.f32, 1));
        const tensor_filled_1_w = zml.Tensor.constant(.{w}, zml.Data.init(.f32, 1));

        // Build the indices for the height and width by linearly spacing the grid points
        const h_idxs = zml.Tensor.linspace(.{ .start = 0, .end = num_grid_per_side - 1, .steps = h }, .f32);
        const w_idxs = zml.Tensor.linspace(.{ .start = 0, .end = num_grid_per_side - 1, .steps = w }, .f32);
        const h_floor = h_idxs.floor();
        const w_floor = w_idxs.floor();

        // Build the ceil and floor for the height and width
        const h_ceil = h_floor.add(tensor_filled_1_h).clamp(
            zml.Tensor.scalar(0, .f32),
            zml.Tensor.scalar(num_grid_per_side - 1, .f32),
        );
        const w_ceil = w_floor.add(tensor_filled_1_w).clamp(
            zml.Tensor.scalar(0, .f32),
            zml.Tensor.scalar(num_grid_per_side - 1, .f32),
        );

        // Build the difference between the indices and the floor for the height and width -> delta with the grid points
        const dh = h_idxs.sub(h_floor);
        const dw = w_idxs.sub(w_floor);

        // Build the meshgrid for the height and width
        const tensor_filled_1_h_v = zml.Tensor.constant(.{ h, w }, zml.Data.init(.f32, 1));
        const d_tensors_meshgrid = [2]Tensor{ dh, dw };
        const floor_tensors_meshgrid = [2]Tensor{ h_floor, w_floor };
        const ceil_tensors_meshgrid = [2]Tensor{ h_ceil, w_ceil };
        const dhw_grid = zml.Tensor.cartesianProduct(2, d_tensors_meshgrid);
        const floorhw_grid = zml.Tensor.cartesianProduct(2, floor_tensors_meshgrid);
        const ceilhw_grid = zml.Tensor.cartesianProduct(2, ceil_tensors_meshgrid);

        // Compute the weights for the height and width
        const w11 = dhw_grid[0].mul(dhw_grid[1]);
        const w10 = dhw_grid[0].sub(w11);
        const w01 = dhw_grid[1].sub(w11);
        const w00 = tensor_filled_1_h_v.sub(dhw_grid[0]).sub(w01);
        const h_list = [4]Tensor{ floorhw_grid[0], floorhw_grid[0], ceilhw_grid[0], ceilhw_grid[0] };
        const w_list = [4]Tensor{ floorhw_grid[1], ceilhw_grid[1], floorhw_grid[1], ceilhw_grid[1] };

        // Stack the height and width lists
        const h_grid = zml.Tensor.stack(&h_list, 0, .layers);
        const w_grid = zml.Tensor.stack(&w_list, 0, .layers);
        const h_grid_idx = h_grid.scale(num_grid_per_side);
        const indices = h_grid_idx.add(w_grid).reshape(.{ 4, -1 }).convert(.i32);
        var weights = zml.Tensor.stack(&[4]Tensor{ w00, w01, w10, w11 }, 0, .layers).reshape(.{ 4, -1, 1 });
        const embeds = zml.call(self.pos_embed, .forward, .{indices});
        const weights_embed = embeds.convert(.f32).mul(weights.repeat1d(-1, embedding_dim));
        const combined = weights_embed.sum(0).withTags(.{ .bs, .hw, .d });

        // Split the combined tensor into the height and width dimensions
        const combined_reshape = combined.splitAxis(.hw, .{ .h = @divExact(h, m_size), .m1 = m_size, .w = @divExact(w, m_size), .m2 = m_size });
        const combined_permuted = combined_reshape.transpose(.{ .bs, .h, .w, .m1, .m2, .d });

        // Repeat the combined tensor t times along the temporal dimension
        const t_u63: u63 = @intCast(t);
        const repeated = combined_permuted.repeat1d(0, t_u63).reshape(.{ -1, embedding_dim });
        outputs[0] = repeated;

        return outputs;
    }

    // Rotary position embedding for the vision transformer
    pub fn rotaryPosEmbed(grid_thw: []const u32, m_size: u32, hidden_size: u32, num_heads: u32, rope_opts: zml.nn.RopeOpts) Tensor {
        const t = grid_thw[0];
        const h = grid_thw[1];
        const w = grid_thw[2];

        const pos_shape = zml.Shape.init(.{ .h = h, .w = w }, .f32);
        // Build the height position ids
        var hpos_ids = zml.Tensor.iota(pos_shape, 0);

        hpos_ids = hpos_ids.splitAxis(.h, .{ .h_div = @divExact(h, m_size), .m1 = m_size }).splitAxis(.w, .{ .w_div = @divExact(w, m_size), .m2 = m_size });
        hpos_ids = hpos_ids.transpose(.{ .h_div, .w_div, .m1, .m2 });
        hpos_ids = hpos_ids.reshape(.{ .seq = -1 });

        // Build the width position ids
        var wpos_ids = zml.Tensor.iota(pos_shape, 1);
        wpos_ids = wpos_ids.splitAxis(.h, .{ .h_div = @divExact(h, m_size), .m1 = m_size }).splitAxis(.w, .{ .w_div = @divExact(w, m_size), .m2 = m_size });
        wpos_ids = wpos_ids.transpose(.{ .h_div, .w_div, .m1, .m2 });
        wpos_ids = wpos_ids.reshape(.{ .seq = -1 });

        const pos_ids = zml.Tensor.stack(&[2]Tensor{ hpos_ids, wpos_ids }, 1, .layers).repeat1d(1, @as(u63, @intCast(t))).convert(.i32);

        // Compute the inverse frequency
        const inv_freq = zml.nn.invFreq(@intCast(32), rope_opts).withTags(.{.s});
        const seq = zml.Tensor.arange(.{ .end = hidden_size / num_heads / 2 }, .f32).withTags(.{.d});
        // Compute the outer product of the sequence and the inverse frequency
        const rotary_pos_emb_full = zml.Tensor.outer(seq, inv_freq);

        const output = rotary_pos_emb_full.gather(.{ .d = pos_ids }, .{}).merge(.{ .d = .{ .layers, .s } });
        // Add a bs size for the output for the moment because the image processing is not done in batch but it is needed for the text processing
        const output_with_bs = output.reshape(.{ .bs = 1, .s = output.dim(.seq), .d = output.dim(.d) });
        return output_with_bs;
    }
};

// Get the padding for the convolution (same on all dimensions)
fn getPaddingForSame(input_dims: [3]i64, kernel_dims: [3]i64, strides: [3]i64) [6]i64 {
    var res: [6]i64 = undefined;
    for (0..3) |i| {
        const output_size = @divFloor(input_dims[i], strides[i]);
        const total_padding = (output_size - 1) * strides[i] + kernel_dims[i] - input_dims[i];
        const pad_start = @divFloor(total_padding, 2);
        const pad_end = total_padding - pad_start;
        res[2 * i] = pad_start;
        res[2 * i + 1] = pad_end;
    }
    return res;
}

pub const Conv3d = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    temporal_stride: u32 = 2,
    spatial_stride: u32 = 16,

    pub fn forward(self: Conv3d, input: Tensor) Tensor {
        const x = input;
        var strides: [3]i64 = .{ self.temporal_stride, self.spatial_stride, self.spatial_stride };
        const padding = getPaddingForSame(x.dims()[2..5].*, self.weight.dims()[2..5].*, strides);
        const loc = input.getContext().location(@src(), "Conv3d.forward", .{});
        var y = x.convolution(
            self.weight.convert(x.dtype()),
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
            loc,
        );
        if (self.bias) |b| y = y.add(b.convert(y.dtype()).broadcast(y._shape, &.{1}));
        return y;
    }
};

pub const VisionBlock = struct {
    norm1: zml.nn.LayerNorm,
    norm2: zml.nn.LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
    num_heads: u32,

    pub fn forward(self: VisionBlock, hidden_states: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const x = zml.call(self.norm1, .forward, .{hidden_states});
        //Here we need to squeeze the output of the attention to remove the bs size because the image processing is not done in batch
        //To be discussed when the model will handle several images and several sequences
        const x1 = hidden_states.add(zml.call(self.attn, .forward, .{ x, cos, sin }).squeeze(0));
        const x2 = zml.call(self.norm2, .forward, .{x1});
        const x3 = x1.add(zml.call(self.mlp, .forward, .{x2}));

        return x3.reuseBuffer(hidden_states);
    }
};

// Vision patch embedding
// Project the image into a visual embedding by applying a 3D convolution along the temporal and spatial dims
pub const VisionPatchEmbed = struct {
    proj: Conv3d,
    patch_size: u32 = 14,
    temporal_patch_size: u32 = 2,
    in_channels: u32 = 3,
    embed_dim: u32 = 1152,
    pub fn init(
        allocator: std.mem.Allocator,
        patch_size: u32,
        temporal_patch_size: u32,
        in_channels: u32,
        embed_dim: u32,
        store: zml.aio.BufferStore,
    ) !VisionPatchEmbed {
        var conv3d = try zml.aio.populateModelWithPrefix(Conv3d, allocator, store, "model.visual.patch_embed.proj");
        conv3d.temporal_stride = temporal_patch_size;
        conv3d.spatial_stride = patch_size;

        return .{
            .proj = conv3d,
            .patch_size = patch_size,
            .temporal_patch_size = temporal_patch_size,
            .in_channels = in_channels,
            .embed_dim = embed_dim,
        };
    }

    pub fn forward(self: VisionPatchEmbed, hidden_states: Tensor) Tensor {
        const reshaped = hidden_states.reshape(.{ -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size });
        const conv_output = zml.call(self.proj, .forward, .{reshaped});
        const reshaped_output = conv_output.reshape(.{ conv_output.dim(0), conv_output.dim(1) });

        return reshaped_output;
    }
};

pub fn rotate_half(x: Tensor) Tensor {
    const x1 = x.slice1d(-1, .{ .start = 0, .end = @divExact(x.dim(-1), 2) });
    const x2 = x.slice1d(-1, .{ .start = @divExact(x.dim(-1), 2), .end = x.dim(-1) });
    return Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
}

pub fn applyRotaryPositionalEmbedding(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) struct { Tensor, Tensor } {
    // Broadcast cos and sin to the shape of q and k

    const cos_q = cos.broadcast(q.shape(), &.{ 0, 1, 3 });
    const sin_q = sin.broadcast(q.shape(), &.{ 0, 1, 3 });
    const cos_k = cos.broadcast(k.shape(), &.{ 0, 1, 3 });
    const sin_k = sin.broadcast(k.shape(), &.{ 0, 1, 3 });

    const q_dtype = q.convert(cos.dtype());
    const k_dtype = k.convert(cos.dtype());
    const q_embed = q_dtype.mul(cos_q).add(rotate_half(q_dtype).mul(sin_q)).withTags(.{ .bs, .q, .h, .hd });
    const k_embed = k_dtype.mul(cos_k).add(rotate_half(k_dtype).mul(sin_k)).withTags(.{ .bs, .k, .h, .hd });

    return .{ q_embed.convert(q.dtype()), k_embed.convert(k.dtype()) };
}

// Vision-specific components
pub const VisionAttention = struct {
    qkv: zml.nn.Linear,
    proj: zml.nn.Linear,

    num_heads: u32,
    //window_size: u32,
    //is_full_attention: bool,

    pub fn forward(self: VisionAttention, hidden_states: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const qkv = zml.call(self.qkv, .forward, .{hidden_states});

        const qkv_reshaped = qkv.reshape(.{
            hidden_states.dim(0),
            3,
            self.num_heads,
            -1, // head_dim
        }).withTags(.{ .s, .qkv, .h, .hd });
        const qkv_permuted = qkv_reshaped.transpose(.{ .qkv, .s, .h, .hd });
        const q, const k, var v = qkv_permuted.chunkExact(.qkv, 3);

        const q_embed, const k_embed = applyRotaryPositionalEmbedding(q, k, cos, sin);
        v = v.withTags(.{ .bs, .k, .h, .hd });
        const attn_output = zml.nn.sdpa(q_embed, k_embed, v, .{ .allow_cudnn = true });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const result = zml.call(self.proj, .forward, .{attn});
        return result;
    }
};

pub const PatchMerger = struct {
    norm: zml.nn.LayerNorm,
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,
    out_hidden_size: u32,

    pub fn forward(self: PatchMerger, x: Tensor, use_post_shuffle_norm: bool) Tensor {
        const gelu: zml.nn.Activation = .gelu;
        // Apply the post shuffle norm if needed
        var x1 = if (use_post_shuffle_norm)
            zml.call(self.norm, .forward, .{x.reshape(.{ -1, 1024 * 4 })})
        else
            zml.call(self.norm, .forward, .{x});
        x1 = x1.reshape(.{ -1, 1024 * 4 });

        const x2 = zml.call(self.linear_fc1, .forward, .{x1});
        const x3 = gelu.forward(x2);
        const x4 = zml.call(self.linear_fc2, .forward, .{x3}).withTags(.{ .seq, .d });

        return x4;
    }
};

pub const VisionMlp = struct { // MLP classique
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,
    hidden_act: zml.nn.Activation = .{ .gelu = {} },
    pub fn forward(self: VisionMlp, x: Tensor) Tensor {
        const x1 = zml.call(self.linear_fc1, .forward, .{x});
        const gelu_tanh_approximation = zml.nn.Activation{ .gelu = {} };
        const x2 = gelu_tanh_approximation.forward(x1.convert(.f32)).convert(x.dtype());
        const x3 = zml.call(self.linear_fc2, .forward, .{x2});

        return x3;
    }
};

//========================Text model========================

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
    rotary_embed: TextRotaryEmbedding,
    max_seq_len: u32 = 1000,
    num_heads: u32,
    num_kv_heads: u32,
    mrope_section: [3]u32,

    pub fn init(allocator: std.mem.Allocator, config: Qwen.Config, store: zml.aio.BufferStore) !TextModel {
        const layers = try allocator.alloc(TransformerLayer, config.text_config.num_hidden_layers);
        var prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);

        const text_rotary_embed = try TextRotaryEmbedding.init(allocator, config.text_config.hidden_size, config.text_config.rope_theta, config.text_config.rope_scaling.mrope_section);

        try prefix.push(stdx.noalloc, "model.language_model.layers");
        for (0.., layers) |i, *layer| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            var self_attn = try zml.aio.populateModelWithPrefix(SelfAttn, allocator, store, prefix.concat("self_attn"));
            self_attn.num_heads = config.text_config.num_attention_heads;
            self_attn.num_kv_heads = config.text_config.num_key_value_heads;

            const mlp = try zml.aio.populateModelWithPrefix(Mlp, allocator, store, prefix.concat("mlp"));

            var input_layernorm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, prefix.concat("input_layernorm"));
            input_layernorm.eps = 1e-6;

            var post_attention_layernorm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, prefix.concat("post_attention_layernorm"));
            post_attention_layernorm.eps = config.text_config.rms_norm_eps;

            layer.* = .{
                .self_attn = self_attn,
                .mlp = mlp,
                .input_layernorm = input_layernorm,
                .post_attention_layernorm = post_attention_layernorm,
                .num_heads = config.text_config.num_attention_heads,
            };
        }

        return .{
            .embed_tokens = try zml.aio.populateModelWithPrefix(zml.nn.TokenEmbedding, allocator, store, "model.language_model.embed_tokens"),
            .layers = layers,
            .norm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, "model.language_model.norm"),
            .num_heads = config.text_config.num_attention_heads,
            .num_kv_heads = config.text_config.num_key_value_heads,
            .rotary_embed = text_rotary_embed,
            .mrope_section = config.text_config.rope_scaling.mrope_section,
        };
    }

    // Forward prefill pass for the text model
    pub fn forward(self: TextModel, position_ids: Tensor, inputs_embeds: Tensor, cache_position: Tensor, deepstack_visual_embeds: [3]Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        var hidden_states = inputs_embeds;
        const cos, const sin = self.rotary_embed.forward(position_ids);
        var count: u32 = 0;
        // Build the indices for the deepstack visual embeddings addition
        const indices = zml.Tensor.iota(Shape.init(.{ .seq = deepstack_visual_embeds[0].dim(.seq) }, .u32), .seq).addConstant(4);

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = zml.call(layer, .forward, .{ hidden_states, cache_position, cos, sin, updated_kv_cache.atLayer(i) });
            hidden_states = hidden_states.withTags(.{ .bs, .seq, .d });

            // Add the n visual embeddings at the n first layers outputs
            if (count < deepstack_visual_embeds.len) {
                const deepstack = deepstack_visual_embeds[count];
                hidden_states = hidden_states.scatterSlices(.{ .seq = indices }, zml.torch.unsqueeze(deepstack, 0).convert(hidden_states.dtype()).withTags(.{ .bs, .seq, .d }), .{ .update_fn = zml.Tensor.ScatterOpts.increment });
                count += 1;
            }
        }
        const output = zml.call(self.norm, .forward, .{hidden_states});

        return .{ output, updated_kv_cache };
    }

    // Forward decode pass for the text model
    // Similar to the prefill pass, but without the deepstack visual embeddings addition
    pub fn forward_decode(self: TextModel, position_ids: Tensor, inputs_embeds: Tensor, cache_position: Tensor, kv_cache: KvCache) struct { Tensor, KvCache } {
        var hidden_states = inputs_embeds;
        const cos, const sin = self.rotary_embed.forward(position_ids);
        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = zml.call(layer, .forward, .{ hidden_states, cache_position, cos, sin, updated_kv_cache.atLayer(i) });
            hidden_states = hidden_states.withTags(.{ .bs, .seq, .d });
        }
        const output = zml.call(self.norm, .forward, .{hidden_states});
        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }
};

pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    dim: u32,
    mrope_section: [3]u32,

    pub fn init(allocator: std.mem.Allocator, dim: u32, theta: f32, mrope_section: [3]u32) !TextRotaryEmbedding {
        _ = allocator;
        return .{
            .rope_opts = zml.nn.RopeOpts{
                .layout = .sequential,
                .freq_base = theta,
                .scaling = .{ .default = {} },
            },
            .dim = dim,
            .mrope_section = mrope_section,
        };
    }

    pub fn forward(self: TextRotaryEmbedding, position_ids: Tensor) struct { Tensor, Tensor } {
        const mrope_section = [3]u32{ self.mrope_section[0], self.mrope_section[1], self.mrope_section[2] }; // from config 24 +20 +20 = 64 i.e. hd / 2
        const inv_freq = zml.nn.invFreq(@intCast(128), self.rope_opts).withTags(.{.dh}).convert(.f32);

        // perform the outer product between the position ids and the inverse frequencies, output shape is (3, bs, dim_head//2, seq len)
        var freqs = position_ids.convert(inv_freq.dtype()).outer(inv_freq);
        // Interleaved mrope
        // Slice the frequency tensor to get the frequency for the temporal, height and width dimensions
        var freqs_t, var freqs_h, var freqs_w = freqs.chunkExact(.g, 3);

        //Squeeze the grid dim because we process per dim independently
        freqs_t = freqs_t.squeeze(.g);
        freqs_h = freqs_h.squeeze(.g);
        freqs_w = freqs_w.squeeze(.g);

        const indices = zml.Tensor.iota(Shape.init(.{ .h = @as(u32, @intCast(mrope_section[1])) }, .i32), .h);

        // Build the indices for the height and width dimensions
        const h_indices = indices.scale(3).addConstant(1);
        const w_indices = indices.scale(3).addConstant(2);

        // Gather scatter the frequencies to build the tensor such as [t,h,w,t,h,w,...,t,h,w,t,t,t,t]
        const h_input = freqs_h.gather(.{ .dh = h_indices }, .{ .indices_are_sorted = true });
        const w_input = freqs_w.gather(.{ .dh = w_indices }, .{ .indices_are_sorted = true });
        freqs_t = freqs_t.transpose(.{ .dh, .bs, .seq });
        freqs_t = freqs_t.scatterSlices(.{ .dh = h_indices }, h_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        freqs = freqs_t.scatterSlices(.{ .dh = w_indices }, w_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        freqs = freqs.transpose(.{ .bs, .seq, .dh });
        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos();
        const sin = emb.sin();

        return .{ cos, sin };
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    mlp: Mlp,
    post_attention_layernorm: RmsNorm,
    num_heads: u32,

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});

        const delta0, const updated_kv_cache = zml.call(self.self_attn, .forward, .{
            x0_normalized,
            token_index,
            cos,
            sin,
            kv_cache,
        });

        const x1 = x0.add(delta0);
        const x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        const x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);

        const result = .{ x2.reuseBuffer(x0), updated_kv_cache };
        return result;
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_position: Tensor,
        cos: Tensor,
        sin: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {

        // Compute key query and value projections (split the dimension into head and dimension)
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = 32, .hd = .auto });
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = 8, .hd = .auto });
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = 8, .hd = .auto });

        const token_index = token_position.convert(kv_cache.layer_index.dtype());
        const seq_len = kv_cache.k.dim(.k);

        // Generate the attention mask
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.seq) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});

        q = zml.call(self.q_norm, .forward, .{q.rename(.{ .hd = .d })}).rename(.{ .d = .hd }).withTags(.{ .bs, .q, .h, .hd });
        k = zml.call(self.k_norm, .forward, .{k.rename(.{ .hd = .d })}).rename(.{ .d = .hd }).withTags(.{ .bs, .k, .h, .hd });
        v = v.withTags(.{ .bs, .k, .h, .hd });

        q, k = applyRotaryPositionalEmbedding(q, k, cos, sin);

        // Update the key-value cache
        const kv_cache_updated = kv_cache.update(k, v, token_index);
        // Retrieve the cached key and value
        const cached_k = kv_cache_updated.keys().convert(q.dtype());
        const cached_v = kv_cache_updated.values().convert(q.dtype());

        const orig_dtype = q.dtype();

        // Attention
        const attn_output = zml.nn.sdpa(q, cached_k, cached_v, .{ .attn_mask = attn_mask, .allow_cudnn = true }).convert(orig_dtype);

        // Merge head and dimension back together
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const result = .{ zml.call(self.o_proj, .forward, .{attn}), kv_cache_updated };

        return result;
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear,
    gate_proj: zml.nn.Linear,
    down_proj: zml.nn.Linear,

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = zml.call(self.up_proj, .forward, .{x});
        var output = zml.call(self.gate_proj, .forward, .{x});
        output = output.silu().mul(proj);
        const result = zml.call(self.down_proj, .forward, .{output});

        return result;
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, self.eps);
        return normalized.mul(self.weight.withTags(.{.d}).broad(x.shape()).convert(x.dtype()));
    }
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        // The KV-cache is initialized with ones to detect reads of uninitialized memory.
        return .{
            .k = Tensor.constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .v = Tensor.constant(kv_shape, kv_shape.dtype().one()).withSharding(.{.h}),
            .layer_index = Tensor.scalar(-1, .i64),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .i64),
        };
    }

    pub fn initBuffer(kv_shape: zml.Shape, platform: zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(platform, kv_shape, .{}),
            .v = try zml.Buffer.uninitialized(platform, kv_shape, .{}),
            .layer_index = try zml.Buffer.scalar(platform, 0, .i64),
        };
    }

    pub fn keys(self: KvCache) Tensor {
        return self.k.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) Tensor {
        return self.v.dynamicSlice(.{ .layer = Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx| .{
            .k = self.k.scatterSlices(
                .{ .layer = layer, .k = idx },
                new_k.convert(self.k.dtype()).transpose(k_shape),
                .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override },
            ).reuseBuffer(self.k),
            .v = self.v.scatterSlices(
                .{ .layer = layer, .k = idx },
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

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = Tensor.scalar(layer_index, .i64),
        };
    }

    pub fn reuseBuffer(self: KvCache, other: KvCache) KvCache {
        return .{
            .k = self.k.reuseBuffer(other.k),
            .v = self.v.reuseBuffer(other.v),
            .layer_index = self.layer_index.reuseBuffer(other.layer_index),
        };
    }
};

// Resize bicubic function
// Resize the image using bicubic interpolation
// image: Tensor, the image to be resized
// resized_axes: anytype, the axes to be resized
// opt: zml.nn.ResizeOpts, the options for the resize -> contain original length
// returns: Tensor, the resized image
pub fn ResizeBicubic(image: Tensor, resized_axes: anytype, opt: zml.nn.ResizeOpts) Tensor {
    const new_size, const tags_ = zml.Shape.parseStruct(u63, resized_axes);
    var out = image;
    for (new_size.constSlice(), tags_.constSlice()) |d, t| {
        const ax = image.shape().axis(t);
        const child_opt: zml.nn.ResizeOpts = .{
            .original_len = opt.original_len,
            .precision = opt.precision,
        };
        out = ResizeCubic1d(out, ax, d, child_opt);
    }
    return out;
}

/// Bicubic interpolation along a single axis
fn ResizeCubic1d(image: Tensor, axis: i8, new_len: u63, opt: zml.nn.ResizeOpts) Tensor {
    const ax = image.axis(axis);
    const res_shape = image.shape().set(ax, new_len);
    const dtype = opt.precision orelse if (image.dtype().class() == .integer) .f32 else image.dtype();

    // Extract the correct dimension from original_len if it's a vector
    const og_len = if (opt.original_len) |o| blk: {
        // If original_len is a vector ( here chw=3), extract the dimension for this axis
        if (o.rank() == 1) {

            // Get the index of the axis in the original length
            const idx_in_original = @as(i64, @intCast(ax));
            break :blk o.choose1d(0, idx_in_original).convert(dtype);
        } else {
            // It's already a scalar
            break :blk o.convert(dtype);
        }
    } else Tensor.scalar(image.dim(ax), dtype);

    // Calculate scale
    const align_corners = false;

    // Compute the scale between the original length (not the padded one) and the new length
    const scale = if (align_corners and new_len > 1)
        og_len.addConstant(-1).scale(stdx.math.divFloat(f32, 1, new_len - 1))
    else
        og_len.scale(stdx.math.divFloat(f32, 1, new_len));

    // Generate output positions
    const dst_indices = Tensor.arange(.{ .end = new_len }, dtype);
    const src_f = if (align_corners)
        dst_indices.mul(scale)
    else
        dst_indices.addConstant(0.5).mul(scale).addConstant(-0.5);

    // Calculate floor and fractional part
    const input_index_floor = src_f.floor();
    const t = src_f.sub(input_index_floor);

    // Start index for 4-pixel window (leftmost pixel is at floor - 1)
    const start_idx = input_index_floor.convert(.i32).addConstant(-1);

    // Calculate bicubic weights for all positions
    const A: f32 = -0.75; // Catmull-Rom coefficient
    const weights = computeBicubicWeights(t, A);

    // For each of the 4 neighbors, compute indices and gather values
    var accumulated = Tensor.constant(res_shape, dtype.zero());

    inline for (0..4) |i| {
        // Compute neighbor indices
        const neighbor_idx_raw = start_idx.addConstant(@as(i32, @intCast(i)));
        const neighbor_idx_clamped = neighbor_idx_raw
            .maximum(Tensor.scalar(0, .i32))
            .minimum(og_len.convert(.i32).addConstant(-1));

        // Gather neighbor values using gather_ (like resizeLinear1d)
        const neighbor_values = image
            .gather_(&.{ax}, &.{neighbor_idx_clamped}, .{ .indices_are_sorted = true })
            .convert(dtype);

        // Get weight for this neighbor
        const weight = weights[i];

        // Broadcast weight to res_shape (matching the output shape) along axis ax
        const weight_broadcasted = weight.broadcast(res_shape, &.{ax});

        // Multiply and accumulate
        const weighted = neighbor_values.mul(weight_broadcasted);
        accumulated = accumulated.add(weighted);
    }

    return accumulated.convert(image.dtype()).withTags(image.shape().tags());
}

/// Compute bicubic weights for fractional distance t
/// Returns array of 4 weight tensors (one for each neighbor at offsets -1, 0, 1, 2)
fn computeBicubicWeights(t: Tensor, A: f32) [4]Tensor {
    const x = t;
    const x2 = x.mul(x);
    const one_minus_x = Tensor.scalar(1.0, x.dtype()).sub(x);
    const one_minus_x2 = one_minus_x.mul(one_minus_x);
    const one_minus_x_plus_1 = one_minus_x.addConstant(1.0);

    // Weight for neighbor -1 (distance 1+x)
    const w0 = ((x.addConstant(1.0).scale(A).addConstant(-5.0 * A)).mul(x.addConstant(1.0)).addConstant(8.0 * A)).mul(x.addConstant(1.0)).addConstant(-4.0 * A);

    // Weight for neighbor 0 (distance x)
    const w1 = ((x.scale(A + 2.0).addConstant(-(A + 3.0))).mul(x2)).addConstant(1.0);

    // Weight for neighbor 1 (distance 1-x)
    const w2 = ((one_minus_x.scale(A + 2.0).addConstant(-(A + 3.0))).mul(one_minus_x2)).addConstant(1.0);

    // Weight for neighbor 2 (distance 2-x)
    const w3 = ((one_minus_x_plus_1.scale(A).addConstant(-5.0 * A)).mul(one_minus_x_plus_1).addConstant(8.0 * A)).mul(one_minus_x_plus_1).addConstant(-4.0 * A);

    return .{ w0, w1, w2, w3 };
}
