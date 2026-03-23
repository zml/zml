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

    pub const GenOptions = struct { sampling_strategy: zml.nn.SamplingStrategy = .{}, max_seq_len: i64 };

    pub const SpecialTokens = struct {
        im_start_token_id: u32,
        im_end_token_id: u32,
        end_of_text_token_id: u32,
    };

    text_model: TextModel,
    vision_model: VisionModel,
    lm_head: zml.nn.Linear,

    config: Config,
    gen_options: GenOptions,
    special_tokens: SpecialTokens = .{
        .im_start_token_id = 248045,
        .im_end_token_id = 248046,
        .end_of_text_token_id = 248044,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, gen_options: GenOptions) !Qwen35 {
        // For some Qwen3.5 versions, the output projection lm_head has a standalone weight tensor, while for others it's the same as the input embedding layer
        const lm_head_prefix = if (store.hasKey("lm_head.weight")) "lm_head" else "model.language_model.embed_tokens";
        return .{
            .text_model = try .init(allocator, store.withPrefix("model.language_model"), config),
            .vision_model = try .init(allocator, store.withPrefix("model.visual"), config),
            .lm_head = .init(store.withPrefix(lm_head_prefix).createTensor("weight", .{ .dout, .d }, null), null, .d),
            .config = config,
            .gen_options = gen_options,
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

    pub fn build3dPositionIds(
        self: Qwen35,
        batch_count: i64,
        seq_len: i64,
        prompt_shape: Tensor,
        image_grid_thw: [3]i64,
    ) struct { Tensor, Tensor } {
        const text_pre_image_token_count = prompt_shape.choose1d(0, 0).convert(.i64);
        const image_token_count = prompt_shape.choose1d(0, 1).convert(.i64);
        const text_post_image_token_count = prompt_shape.choose1d(0, 2).convert(.i64);

        const pre_image_positions = zml.Tensor.iota(zml.Shape.init(.{ .b = batch_count, .s = 4 }, .i64), .s).convert(.i64);

        const t = image_grid_thw[0];
        const h = @divExact(image_grid_thw[1], self.config.vision_config.spatial_merge_size);
        const w = @divExact(image_grid_thw[2], self.config.vision_config.spatial_merge_size);

        var position_ids = zml.Tensor.iota(zml.Shape.init(.{ .b = batch_count, .s = seq_len }, .i64), .s)
            .convert(.i64)
            .sub(image_token_count)
            .addConstant(@max(w, h, t));
        position_ids = position_ids.dynamicUpdateSlice(.{ .s = zml.Tensor.scalar(0, .i64) }, pre_image_positions);

        // Define the shape of the iota tensor
        const iota_shape = zml.Shape.init(.{ .b = batch_count, .t = t, .h = h, .w = w }, .i64);
        const reshape_shape = zml.Shape.init(.{ .b = batch_count, .s = t * h * w }, .i64);

        // Repeat the index along the 3 dimensions based on grid size (after the text (+4 tokens according to the chat template))
        const t_index = zml.Tensor.iota(iota_shape, .t).convert(.i64).reshape(reshape_shape).add(text_pre_image_token_count);
        const h_index = zml.Tensor.iota(iota_shape, .h).convert(.i64).reshape(reshape_shape).add(text_pre_image_token_count);
        const w_index = zml.Tensor.iota(iota_shape, .w).convert(.i64).reshape(reshape_shape).add(text_pre_image_token_count);

        // Update the position ids with the 3D positional ids
        const position_ids_t = position_ids.dynamicUpdateSlice(.{ .s = text_pre_image_token_count }, t_index);
        const position_ids_h = position_ids.dynamicUpdateSlice(.{ .s = text_pre_image_token_count }, h_index);
        const position_ids_w = position_ids.dynamicUpdateSlice(.{ .s = text_pre_image_token_count }, w_index);

        // Stack the position ids
        const stacked_position_ids = zml.Tensor.stack(&.{ position_ids_t, position_ids_h, position_ids_w }, 0, .g);

        // Position max after 3d compression - real seq len
        const position_max_after_3d_compression = zml.Tensor.scalar(@max(w, h, t), .i64).add(text_post_image_token_count).add(text_pre_image_token_count);
        const real_seq_len = text_pre_image_token_count.add(text_post_image_token_count).add(image_token_count);
        const mrope_position_deltas = position_max_after_3d_compression.sub(real_seq_len).reshape(.{ .s = 1 });

        return .{ stacked_position_ids, mrope_position_deltas };
    }

    pub fn forward(
        self: Qwen35,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const text_model_output, const updated_kv_cache = self.text_model.forward(tokens, token_index, kv_cache);
        const new_tokens, const new_rng = self.sampleTokens(text_model_output, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn vision_test_forward(
        self: Qwen35,
        tokens: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        pixel_values: Tensor,
        grid_thw: [3]i64,
        prompt_shape: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor, Tensor.Rng } {
        const token_embed = self.text_model.embed_tokens.forward(tokens.withPartialTags(.{.s}));
        const vision_embed = self.vision_model.forward(pixel_values, grid_thw);

        const text_before_image = prompt_shape.choose1d(0, 0).convert(.i64);
        const num_image_tokens = prompt_shape.choose1d(0, 1).convert(.i64);
        const text_after_image = prompt_shape.choose1d(0, 2).convert(.i64);
        const real_seq_len = text_before_image.add(text_after_image).add(num_image_tokens);

        const text_with_image_embed = token_embed.dynamicUpdateSlice(.{ .s = text_before_image }, vision_embed.convert(token_embed.dtype()));
        const seq_len = text_with_image_embed.dim(.s);

        const position_ids, const mrope_position_deltas = self.build3dPositionIds(
            1,
            seq_len,
            prompt_shape,
            grid_thw,
        );

        const text_with_image_embed_batched = text_with_image_embed.reshape(.{
            .b = 1,
            .s = seq_len,
            .d = text_with_image_embed.dim(.d),
        });
        const text_model_output, const updated_kv_cache = self.text_model.vision_test_forward(
            text_with_image_embed_batched,
            token_index,
            kv_cache,
            position_ids,
        );

        const last_pos = real_seq_len.addConstant(-1).asScalar();
        const last_hidden = text_model_output.dynamicSlice1d(text_model_output.axis(.s), .{ .start = last_pos, .len = 1 });
        const sampled_tokens, const new_rng = self.sampleTokens(last_hidden, rng);
        return .{ sampled_tokens.convert(tokens.dtype()), updated_kv_cache, mrope_position_deltas, new_rng };
    }

    fn buildVisionDecodePositionIds(cache_position: Tensor, mrope_position_deltas: Tensor) Tensor {
        const cache_pos = cache_position.convert(.i64).reshape(.{ .b = 1, .s = 1 });
        const deltas = mrope_position_deltas.convert(.i64).reshape(.{ .b = 1, .s = 1 });
        const pos = cache_pos.add(deltas);
        return zml.Tensor.stack(&.{ pos, pos, pos }, 0, .g).withTags(.{ .g, .b, .s });
    }

    pub fn vision_test_decode_forward(
        self: Qwen35,
        tokens_: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        mrope_position_deltas: Tensor,
        rng: Tensor.Rng,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const token_embed = self.text_model.embed_tokens.forward(tokens);
        const seq_len = token_embed.dim(.s);
        const token_embed_batched = token_embed.reshape(.{
            .b = 1,
            .s = seq_len,
            .d = token_embed.dim(.d),
        });
        const position_ids = buildVisionDecodePositionIds(token_index, mrope_position_deltas);
        const text_model_output, const updated_kv_cache = self.text_model.vision_test_decode_forward(
            token_embed_batched,
            token_index,
            kv_cache,
            position_ids,
        );
        const new_tokens, const new_rng = self.sampleTokens(text_model_output, rng);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
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
    num_position_embeddings: i64,
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
            .num_position_embeddings = config.vision_config.num_position_embeddings,
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
        var hidden_states = self.vision_patch_embed.forward(pixel_values);

        const pos_embeds = self.posEmbedInterpolate(&grid_thw);

        hidden_states = hidden_states.add(pos_embeds.convert(hidden_states.dtype()));

        var rotary_pos_emb = rotaryPosEmbed(&grid_thw, self.spatial_merge_size, self.hidden_size, self.num_heads, self.rope_opts);
        rotary_pos_emb = zml.Tensor.concatenate(&.{ rotary_pos_emb, rotary_pos_emb }, -1);
        const cos = rotary_pos_emb.cos();
        const sin = rotary_pos_emb.sin();

        for (self.blocks) |block| {
            hidden_states = block.forward(hidden_states, cos, sin);
        }

        return self.patch_merger.forward(hidden_states);
    }

    // Positional embedding interpolation (representation of the image in a grid determined by the number of position embeddings 48 x 48)
    pub fn posEmbedInterpolate(self: VisionModel, grid: []const i64) Tensor {
        // Calculate the number of grid points per side (sqrt of the number of position embeddings)
        const num_grid_per_side = std.math.pow(f32, @as(f32, @floatFromInt(self.num_position_embeddings)), 0.5);

        const m_size = self.spatial_merge_size;
        const embedding_dim = self.hidden_size;

        // Retrieve the dims in the image grid
        const t = grid[0];
        const h = grid[1];
        const w = grid[2];
        const tensor_filled_1_h = zml.Tensor.constant(.{ .f32 = 1 }).broad(zml.Shape.init(.{h}, .f32));
        const tensor_filled_1_w = zml.Tensor.constant(.{ .f32 = 1 }).broad(zml.Shape.init(.{w}, .f32));

        // Build the indices for the height and width with a local linspace equivalent.
        // We avoid Tensor.linspace here because this path needs Python-equivalent interpolation coordinates.
        const h_idxs = if (h == 1)
            zml.Tensor.constant(.{ .f32 = 0 }).broad(zml.Shape.init(.{1}, .f32))
        else
            zml.Tensor.arange(.{ .end = h }, .f32).scale((num_grid_per_side - 1) / @as(f32, @floatFromInt(h - 1)));
        const w_idxs = if (w == 1)
            zml.Tensor.constant(.{ .f32 = 0 }).broad(zml.Shape.init(.{1}, .f32))
        else
            zml.Tensor.arange(.{ .end = w }, .f32).scale((num_grid_per_side - 1) / @as(f32, @floatFromInt(w - 1)));
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
        const tensor_filled_1_h_v = zml.Tensor.constant(.{ .f32 = 1 }).broad(zml.Shape.init(.{ h, w }, .f32));
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
        const indices = h_grid_idx.add(w_grid).reshape(.{ 4, -1 }).convert(.i64);
        var weights = zml.Tensor.stack(&[4]Tensor{ w00, w01, w10, w11 }, 0, .layers).reshape(.{ 4, -1, 1 }).convert(self.pos_embed.weight.dtype());
        const embeds = self.pos_embed.forward(indices);
        const weights_embed = embeds.mul(weights.repeat1d(-1, @intCast(embedding_dim)));
        const combined = weights_embed.sum(0).withTags(.{ .bs, .hw, .d });

        // Split the combined tensor into the height and width dimensions
        const combined_reshape = combined.splitAxis(.hw, .{ .h = @divExact(h, m_size), .m1 = m_size, .w = @divExact(w, m_size), .m2 = m_size });
        const combined_permuted = combined_reshape.transpose(.{ .bs, .h, .w, .m1, .m2, .d });

        // Repeat the combined tensor t times along the temporal dimension
        const t_u63: u63 = @intCast(t);
        return combined_permuted.repeat1d(0, t_u63).reshape(.{ -1, embedding_dim });
    }

    // Rotary position embedding for the vision transformer
    pub fn rotaryPosEmbed(grid_thw: []const i64, m_size: i64, hidden_size: i64, num_heads: i64, rope_opts: zml.nn.RopeOpts) Tensor {
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

        // Python does `coords.repeat(num_frames, 1)`: repeat rows (tokens), not columns (coord components).
        const pos_ids = zml.Tensor.stack(&[2]Tensor{ hpos_ids, wpos_ids }, 1, .layers).repeat1d(0, @as(u63, @intCast(t))).convert(.i64);
        // Old implem:
        // const pos_ids = zml.Tensor.stack(&[2]Tensor{ hpos_ids, wpos_ids }, 1, .layers).repeat1d(1, @as(u63, @intCast(t))).convert(.i64);

        // Compute the inverse frequency
        const inv_freq = zml.nn.invFreq(@intCast(32), rope_opts).withTags(.{.s});
        const seq = zml.Tensor.arange(.{ .end = @divExact(@divExact(hidden_size, num_heads), 2) }, .f32).withTags(.{.d});
        // Compute the outer product of the sequence and the inverse frequency
        const rotary_pos_emb_full = zml.Tensor.outer(seq, inv_freq);

        const output = rotary_pos_emb_full.gather(.{ .d = pos_ids }, .{}).merge(.{ .d = .{ .layers, .s } });
        // Return without reshape to match Python
        return output;
        // Old implem: Add a bs size for the output for the moment because the image processing is not done in batch but it is needed for the text processing
        // const output_with_bs = output.reshape(.{ .bs = 1, .s = output.dim(.seq), .d = output.dim(.d) });
        // return output_with_bs;
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
        const reshaped = pixel_values.reshape(.{ -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size }); // TODO: add clean tags
        const target_type = self.proj.weight.dtype();
        const conv_output = self.proj.forward(reshaped.convert(target_type));
        return conv_output.reshape(.{ conv_output.dim(0), conv_output.dim(1) });
    }
};

pub const Conv3d = struct {
    weight: Tensor,
    bias: ?Tensor = null,
    temporal_stride: i64,
    spatial_stride: i64,

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) Conv3d {
        return .{
            .weight = store.createTensor("weight", null, null),
            .bias = store.maybeCreateTensor("bias", null, null),
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
        var strides: [3]i64 = .{ self.temporal_stride, self.spatial_stride, self.spatial_stride };
        const padding = getPadding(x.dims()[2..5].*, self.weight.dims()[2..5].*, strides);
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

    pub fn forward(self: VisionBlock, hidden_states: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const attn_output = self.attn.forward(self.norm1.forward(hidden_states), cos, sin);
        const x = hidden_states.add(attn_output);
        const mlp_output = self.mlp.forward(self.norm2.forward(x));
        return x.add(mlp_output);
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

    pub fn init(store: zml.io.TensorStore.View, config: Qwen35.Config) VisionAttention {
        return .{
            .qkv = .init(store.withPrefix("qkv").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("qkv").maybeCreateTensor("bias", .{.dout}, null), .d),
            .proj = .init(store.withPrefix("proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .num_heads = config.vision_config.num_heads,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionAttention)) void {
        self.qkv.weight.deinit();
        if (self.qkv.bias) |*bias| bias.deinit();
        self.proj.weight.deinit();
        if (self.proj.bias) |*bias| bias.deinit();
    }

    pub fn forward(self: VisionAttention, hidden_states: Tensor, cos: Tensor, sin: Tensor) Tensor {
        const qkv = self.qkv.forward(hidden_states.withTags(.{ .s, .d }));

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
        const result = self.proj.forward(attn).rename(.{ .dout = .d });
        return result.squeeze(.bs);
    }

    fn rotate_half(x: Tensor) Tensor {
        const x1 = x.slice1d(-1, .{ .start = 0, .end = @divExact(x.dim(-1), 2) });
        const x2 = x.slice1d(-1, .{ .start = @divExact(x.dim(-1), 2), .end = x.dim(-1) });
        return Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    }

    fn applyRotaryPositionalEmbedding(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) struct { Tensor, Tensor } {
        // Broadcast cos and sin to the shape of q and k

        const cos_q = cos.broadcast(q.shape(), &.{ 1, 3 });
        const sin_q = sin.broadcast(q.shape(), &.{ 1, 3 });
        const cos_k = cos.broadcast(k.shape(), &.{ 1, 3 });
        const sin_k = sin.broadcast(k.shape(), &.{ 1, 3 });

        const q_dtype = q.convert(cos.dtype());
        const k_dtype = k.convert(cos.dtype());
        const q_embed = q_dtype.mul(cos_q).add(rotate_half(q_dtype).mul(sin_q)).withTags(.{ .bs, .q, .h, .hd });
        const k_embed = k_dtype.mul(cos_k).add(rotate_half(k_dtype).mul(sin_k)).withTags(.{ .bs, .k, .h, .hd });

        return .{ q_embed.convert(q.dtype()), k_embed.convert(k.dtype()) };
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
        const merged_seq_len = @divExact(x.dim(0), self.spatial_merge_unit);
        const merged_hidden_size = x.dim(1) * self.spatial_merge_unit;
        const x1 = self.norm.forward(x).reshape(.{ merged_seq_len, merged_hidden_size }).withTags(.{ .s, .d });
        const x2 = self.linear_fc1.forward(x1).rename(.{ .dout = .d });
        const x3 = x2.gelu();
        const x4 = self.linear_fc2.forward(x3).rename(.{ .dout = .d });

        return x4;
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
        tokens: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        var hidden_states = self.embed_tokens.weight.gather(.{ .voc = tokens }, .{});

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.forward(hidden_states, token_index, updated_kv_cache.atLayer(i));
        }

        hidden_states = self.norm.forward(hidden_states);
        return .{ hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn vision_test_forward(
        self: TextModel,
        input_embeds: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        position_ids: Tensor,
    ) struct { Tensor, KvCache } {
        var hidden_states = input_embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.vision_test_forward(hidden_states, token_index, updated_kv_cache.atLayer(i), position_ids);
        }

        hidden_states = self.norm.forward(hidden_states);
        return .{ hidden_states, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn vision_test_decode_forward(
        self: TextModel,
        input_embeds: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        position_ids: Tensor,
    ) struct { Tensor, KvCache } {
        var hidden_states = input_embeds;

        var updated_kv_cache = kv_cache;
        for (self.layers, 0..) |layer, i| {
            hidden_states, updated_kv_cache = layer.vision_test_decode_forward(hidden_states, token_index, updated_kv_cache.atLayer(i), position_ids);
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
    ) struct { Tensor, KvCache } {
        const residual0 = x0;
        const normalized_x0 = self.input_layernorm.forward(x0);

        var attention_output: Tensor = undefined;
        var updated_kv_cache: KvCache = kv_cache.parent;
        switch (self.attn) {
            .self_attn => |*self_attn| {
                const result = self_attn.forward(normalized_x0, token_index, kv_cache.cache.self_attn);
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

        return .{ mlp_output.add(residual1), updated_kv_cache };
    }

    pub fn vision_test_forward(
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
                const result = self_attn.vision_test_forward(normalized_x0, token_index, kv_cache.cache.self_attn, position_ids);
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

        return .{ mlp_output.add(residual1), updated_kv_cache };
    }

    pub fn vision_test_decode_forward(
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
                const result = self_attn.vision_test_decode_forward(normalized_x0, token_index, kv_cache.cache.self_attn, position_ids);
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

        return .{ mlp_output.add(residual1), updated_kv_cache };
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
    ) struct { Tensor, KvCache.SelfAttnCache } {
        var q, const gate = self.projectQAndGate(x);
        var k, var v = self.projectKV(x);
        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });

        const dtype = q.dtype();
        const position_ids = Tensor.arange(.{ .end = x.dim(.s) }, .i64)
            .withTags(.{.s}).insertAxes(.s, .{.b}).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64))
            .add(token_index.convert(.i64).broad(zml.Shape.init(.{ .b = x.dim(.b), .s = x.dim(.s) }, .i64)));

        const cos, const sin = self.rotary_embed.getCosAndSin(position_ids, dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);

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

    pub fn vision_test_forward(
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
        const cos, const sin = if (position_ids.shape().rank() == 3)
            self.rotary_embed.getCosAndSinInterleaved(position_ids.convert(.i64), dtype)
        else
            self.rotary_embed.getCosAndSin(position_ids.convert(.i64), dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);

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

    pub fn vision_test_decode_forward(
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
        const cos, const sin = if (position_ids.shape().rank() == 3)
            self.rotary_embed.getCosAndSinInterleaved(position_ids.convert(.i64), dtype)
        else
            self.rotary_embed.getCosAndSin(position_ids.convert(.i64), dtype);
        q = self.rotary_embed.applyRope(q, cos, sin);
        k = self.rotary_embed.applyRope(k, cos, sin);

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

    pub fn getCosAndSin(self: TextRotaryEmbedding, position_ids: Tensor, dtype: zml.DataType) struct { Tensor, Tensor } {
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        const freqs_t = position_ids.convert(.f32).outer(inv_freq);

        const emb = Tensor.concatenate(&.{ freqs_t, freqs_t }, -1);
        const cos = emb.cos().convert(dtype);
        const sin = emb.sin().convert(dtype);

        return .{ cos, sin };
    }

    pub fn getCosAndSinInterleaved(self: TextRotaryEmbedding, position_ids: Tensor, dtype: zml.DataType) struct { Tensor, Tensor } {
        const stacked_position_ids = if (position_ids.shape().rank() == 3)
            position_ids.convert(.f32)
        else
            Tensor.stack(&.{ position_ids, position_ids, position_ids }, 0, .g).convert(.f32);
        const inv_freq = zml.nn.invFreq(self.rotary_dim, self.rope_opts).withTags(.{.hd});

        var freqs = stacked_position_ids.outer(inv_freq);
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

    pub fn applyRope(self: TextRotaryEmbedding, x: Tensor, cos: Tensor, sin: Tensor) Tensor {
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

    fn recurrent_gated_delta_rule(query: Tensor, key: Tensor, value: Tensor, g: Tensor, beta: Tensor, initial_state: ?Tensor) struct { Tensor, Tensor } {
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

        const initial_recurrent_state = if (initial_state) |state|
            state.convert(.f32).transpose(.{ .b, .vh, .vhd, .khd }).rename(.{ .vh = .h, .vhd = .v, .khd = .k })
        else
            Tensor.constant(zml.DataType.zero(.f32)).broad(zml.Shape.init(.{
                .b = value.dim(.b),
                .h = value.dim(.vh),
                .v = value.dim(.vhd),
                .k = query.dim(.khd),
            }, .f32));

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
        const use_cached_state = x.dim(.s) == 1 and left_pad > 0;
        const conv_input = if (use_cached_state)
            Tensor.concatenate(&.{ cache.convState(), projected_qkv }, .s)
        else
            projected_qkv;

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

        if (use_cached_state) {
            mixed_qkv = mixed_qkv.slice1d(.s, .{ .start = mixed_qkv.dim(.s) - 1, .end = mixed_qkv.dim(.s) });
        }

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
            if (use_cached_state) cache.recurrentState() else null,
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
        return normalized.mul(weight_f32.broad(x.shape())).add(normalized).convert(x.dtype());
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
        return gated_output.convert(x.dtype());
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
                .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
                .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
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
                .conv_state = try zml.Buffer.uninitialized(io, platform, self.conv_state.shape(), sharding, .{}),
                .recurrent_state = try zml.Buffer.uninitialized(io, platform, self.recurrent_state.shape(), sharding, .{}),
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
