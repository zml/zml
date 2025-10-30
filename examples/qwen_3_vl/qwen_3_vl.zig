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

/// Qwen2.5-VL architecture, using huggingface transformers naming.
/// Vision-Language model with vision transformer and text model.
pub const Qwen3VL = struct {
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
        //tokens_per_second: u32 = 2,
        //window_size: u32 = 112,
        out_hidden_size: u32 = 2048,
        //fullatt_block_indexes: []const u32 = &[_]u32{ 7, 15, 23, 31 },
        initializer_range: f32 = 0.02,
        deepstack_visual_indexes: []const u32 = &[_]u32{ 5, 11, 17 },
    };

    pub const TextConfig = struct {
        hidden_size: u32 = 2560,
        bos_token_id: u32,
        eos_token_id: u32,
        head_dim: ?u32 = null,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        tie_word_embeddings: bool = true,
    };

    pub const Config = struct {
        //Vision config (depuis vision_config)
        vision_config: VisionConfig,
        text_config: TextConfig,
        // rope_theta: f32,
        // hf_rope_impl: bool = true,
        tie_word_embeddings: bool = true,
    };

    pub const Options = struct {
        sampling_strategy: ?zml.nn.SamplingStrategy,
        max_seq_len: u32,
    };
    vision_transformer: VisionTransformer,
    text_model: TextModel,

    // Options controlling generation
    gen_opts: zml.nn.SamplingStrategy = .{},
    config: Config,

    pub fn init(allocator: std.mem.Allocator, config: Config, options: Options, store: zml.aio.BufferStore) !Qwen3VL {
        return .{
            .config = config,
            .gen_opts = options.sampling_strategy orelse .{},
            .vision_transformer = try VisionTransformer.init(allocator, config, store),
            .text_model = try TextModel.init(allocator, config, store),
        };
    }

    pub fn forward(self: Qwen3VL, input_ids: Tensor, pixel_values: Tensor, image_grid_thw: Tensor, attention_mask: Tensor, cache_position: Tensor) Tensor {
        _ = attention_mask; // autofix
        const mock_grid_thw = [3]i32{ 1, 86, 128 };
        var input_embeds = zml.call(self.text_model.embed_tokens, .forward, .{input_ids}).withTags(.{ .bs, .seq, .d });
        const image_embed, const deepstack_features_list = zml.call(self.vision_transformer, .forward, .{ pixel_values, image_grid_thw });
        const image_mask = input_ids.cmp(.EQ, zml.Tensor.scalar(151655, input_ids.dtype()));
        log.info("image_embed: {f}", .{image_embed.shape()});
        log.info("input_embeds: {f}", .{input_embeds.shape()});
        input_embeds = input_embeds.dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(4, .i32) }, zml.torch.unsqueeze(image_embed.convert(input_embeds.dtype()), 0));
        const attn_mask = zml.nn.causalAttnMask(.{ .q = input_ids.dim(1), .k = input_embeds.dim(1) }, input_embeds.dtype(), null);
        var position_ids = zml.Tensor.iota(Shape.init(.{ .bs = input_ids.dim(0), .seq = input_ids.dim(1) }, .i32), .seq);
        log.info("pixel value shape: {f}", .{pixel_values.shape()});
        log.info("input_ids shape: {f}", .{input_ids.shape()});
        log.info("input_embeds shape: {f}", .{input_embeds.shape()});
        log.info("position_ids shape: {f}", .{position_ids.shape()});
        var end_text_position = zml.Tensor.iota(Shape.init(.{ .bs = 1, .seq = 10 }, .i32), .seq);
        end_text_position = end_text_position.addConstant(68);
        log.info("end_text_position: {f}", .{end_text_position.shape()});
        log.info("position_ids: {f}", .{position_ids.shape()});
        position_ids = position_ids.dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(position_ids.dim(1) - end_text_position.dim(1), .i32) }, end_text_position);
        log.info("position_ids: {f}", .{position_ids.shape()});
        const t = mock_grid_thw[0];
        const h = @divExact(mock_grid_thw[1], 2);
        log.info("h: {d}", .{h});
        const w = @divExact(mock_grid_thw[2], 2);
        log.info("w: {d}", .{w});
        const t_index = zml.Tensor.iota(Shape.init(.{ .t = t }, .i32), .t).reshape(.{ .t = -1, .hw = 1 }).repeat1d(1, h * w).flatten();
        const h_index = zml.Tensor.iota(Shape.init(.{ .h = h }, .i32), .h).reshape(.{ .t = 1, .h = -1, .w = 1 }).repeat1d(2, w).flatten();
        const w_index = zml.Tensor.iota(Shape.init(.{ .w = w }, .i32), .w).reshape(.{ .t = 1, .h = 1, .w = -1 }).repeat1d(1, h).flatten();
        log.info("t_index: {f}", .{t_index.shape()});
        log.info("h_index: {f}", .{h_index.shape()});
        log.info("w_index: {f}", .{w_index.shape()});
        const position_ids_t = position_ids.dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(4, .i32) }, zml.torch.unsqueeze(t_index, 0));
        const position_ids_h = position_ids.dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(4, .i32) }, zml.torch.unsqueeze(h_index, 0));
        const position_ids_w = position_ids.dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(4, .i32) }, zml.torch.unsqueeze(w_index, 0));
        position_ids = zml.Tensor.stack(&.{ position_ids_t, position_ids_h, position_ids_w }, 0, .g);
        log.info("position_ids: {f}", .{position_ids.shape()});

        const output = zml.call(self.text_model, .forward, .{ position_ids, attn_mask, input_embeds, cache_position, image_mask, deepstack_features_list });
        //output = output.print();
        return output;
        //faire position ids et input embeds pour le text model
    }
};

//========================Vision model========================

pub const VisionTransformer = struct {
    vision_patch_embed: VisionPatchEmbed,
    pos_embed: zml.nn.TokenEmbedding,
    rotary_pos_emb: VisionRotaryEmbedding,
    blocks: []VisionBlock,
    patch_merger: PatchMerger,
    deepstack_patch_mergers: []PatchMerger,
    // Config values pushed down
    num_heads: u32,
    //window_size: u32,
    //fullatt_block_indexes: []u32,
    //liste de vision patch merger

    pub fn init(allocator: std.mem.Allocator, config: Qwen3VL.Config, store: zml.aio.BufferStore) !VisionTransformer {
        const blocks = try allocator.alloc(VisionBlock, config.vision_config.depth);
        var prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.visual.blocks");
        for (0.., blocks) |i, *block| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            var vision_attn = try zml.aio.populateModelWithPrefix(VisionAttention, allocator, store, prefix.concat("attn"));
            vision_attn.num_heads = config.vision_config.num_heads;
            //vision_attn.window_size = config.window_size;

            var mlp = try zml.aio.populateModelWithPrefix(VisionMlp, allocator, store, prefix.concat("mlp"));
            mlp.hidden_act = zml.nn.Activation{ .gelu = {} };

            var norm1 = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm1"));
            norm1.eps = config.text_config.rms_norm_eps;

            var norm2 = try zml.aio.populateModelWithPrefix(zml.nn.LayerNorm, allocator, store, prefix.concat("norm2"));
            norm2.eps = config.text_config.rms_norm_eps;

            block.* = .{
                .attn = vision_attn,
                .mlp = mlp,
                .norm1 = norm1,
                .norm2 = norm2,
                .num_heads = config.vision_config.num_heads,
                //.window_size = config.window_size,
                //.is_full_attention = std.mem.indexOfScalar(u32, config.fullatt_block_indexes, @intCast(i)) != null, // indexof voir dans zig std
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
        log.info("patch_size: {d}", .{config.vision_config.patch_size});
        log.info("temporal_patch_size: {d}", .{config.vision_config.temporal_patch_size});
        log.info("in_channels: {d}", .{config.vision_config.in_channels});
        log.info("out_hidden_size: {d}", .{config.vision_config.out_hidden_size});

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
            //.window_size = config.window_size,
            //.fullatt_block_indexes = config.fullatt_block_indexes,
            .vision_patch_embed = try VisionPatchEmbed.init(allocator, config.vision_config.patch_size, config.vision_config.temporal_patch_size, config.vision_config.in_channels, config.vision_config.out_hidden_size, store),
            .rotary_pos_emb = try VisionRotaryEmbedding.init(allocator, config.vision_config.out_hidden_size, 10000.0),
        };
    }
    pub fn fastPosEmbedInterpolate(self: VisionTransformer, grid_thw: []const []const i32) [1]Tensor {
        // 1. Extraire les dimensions
        const num_grid_per_side = std.math.pow(f32, 2304, 0.5); //sqrt num_positions_embeddings
        log.info("num_grid_per_side: {d}", .{num_grid_per_side});
        const m_size = 2; //spatial_merge_size
        const embedding_dim = 1024; // config vision hidden size
        // var arena = std.heap.ArenaAllocator.init(std.heap.page_allocator);
        // defer arena.deinit();
        // const allocator = arena.allocator();

        // var outputs = std.ArrayList(Tensor).init(allocator);
        var outputs = [1]Tensor{undefined};
        //defer outputs.deinit(allocator);
        for (grid_thw) |grid| {
            const t = grid[0];
            const h = grid[1];
            const w = grid[2];
            const tensor_filled_1_h = zml.Tensor.constant(.{h}, zml.Data.init(.f32, 1));
            const tensor_filled_1_w = zml.Tensor.constant(.{w}, zml.Data.init(.f32, 1));

            const h_idxs = zml.Tensor.linspace(.{ .start = 0, .end = num_grid_per_side - 1, .steps = h }, .f32);
            const w_idxs = zml.Tensor.linspace(.{ .start = 0, .end = num_grid_per_side - 1, .steps = w }, .f32);
            const h_floor = h_idxs.floor();
            const w_floor = w_idxs.floor();
            const min_clamp_h = zml.Tensor.constant(.{h}, zml.Data.init(.f32, 0));
            const max_clamp_h = zml.Tensor.constant(.{h}, zml.Data.init(.f32, num_grid_per_side - 1));
            const min_clamp_w = zml.Tensor.constant(.{w}, zml.Data.init(.f32, 0));
            const max_clamp_w = zml.Tensor.constant(.{w}, zml.Data.init(.f32, num_grid_per_side - 1));
            const h_ceil = h_floor.add(tensor_filled_1_h).clamp(min_clamp_h, max_clamp_h);
            const w_ceil = w_floor.add(tensor_filled_1_w).clamp(min_clamp_w, max_clamp_w);
            const dh = h_idxs.sub(h_floor);
            const dw = w_idxs.sub(w_floor);

            const tensor_filled_1_h_v = zml.Tensor.constant(.{ h, w }, zml.Data.init(.f32, 1));

            const d_tensors_meshgrid = [2]Tensor{ dh, dw };
            const floor_tensors_meshgrid = [2]Tensor{ h_floor, w_floor };
            const ceil_tensors_meshgrid = [2]Tensor{ h_ceil, w_ceil };
            const dhw_grid = zml.torch.meshgrid(2, d_tensors_meshgrid, .ij);
            const floorhw_grid = zml.torch.meshgrid(2, floor_tensors_meshgrid, .ij);
            const ceilhw_grid = zml.torch.meshgrid(2, ceil_tensors_meshgrid, .ij);

            log.info("tensor_filled_1: {f}", .{tensor_filled_1_h_v.shape()});
            log.info("dhw_grid[0]: {f}", .{dhw_grid[0].shape()});
            log.info("dhw_grid[1]: {f}", .{dhw_grid[1].shape()});
            log.info("h_floor: {f}", .{h_floor.shape()});
            log.info("w_floor: {f}", .{w_floor.shape()});
            log.info("h_ceil: {f}", .{h_ceil.shape()});
            log.info("w_ceil: {f}", .{w_ceil.shape()});
            log.info("dh: {f}", .{dh.shape()});
            log.info("dw: {f}", .{dw.shape()});
            const w11 = dhw_grid[0].mul(dhw_grid[1]);
            const w10 = dhw_grid[0].sub(w11);
            const w01 = dhw_grid[1].sub(w11);
            log.info("w11: {f}", .{w11.shape()});
            log.info("w10: {f}", .{w10.shape()});
            log.info("w01: {f}", .{w01.shape()});
            const w00 = tensor_filled_1_h_v.sub(dhw_grid[0]).sub(w01);
            log.info("w00: {f}", .{w00.shape()});
            const h_list = [4]Tensor{ floorhw_grid[0], floorhw_grid[0], ceilhw_grid[0], ceilhw_grid[0] };
            const w_list = [4]Tensor{ floorhw_grid[1], ceilhw_grid[1], floorhw_grid[1], ceilhw_grid[1] };
            const h_grid = zml.Tensor.stack(&h_list, 0, .layers);

            const w_grid = zml.Tensor.stack(&w_list, 0, .layers);

            log.info("h_grid: {f}", .{h_grid.shape()});
            log.info("w_grid: {f}", .{w_grid.shape()});
            const h_grid_idx = h_grid.scale(num_grid_per_side);
            var indices = h_grid_idx.add(w_grid).reshape(.{ 4, -1 }).convert(.i32);

            log.info("indices: {f}", .{indices.shape()});
            var weights = zml.Tensor.stack(&[4]Tensor{ w00, w01, w10, w11 }, 0, .layers).reshape(.{ 4, -1, 1 }); // embedding des 4coins voisins pour chaque point cible
            const embeds = zml.call(self.pos_embed, .forward, .{indices});
            const weights_embed = embeds.convert(.f32).mul(weights.repeat1d(-1, embedding_dim));
            const combined = weights_embed.sum(0);
            const combined_reshape = combined.reshape(.{ @divExact(h, m_size), m_size, @divExact(w, m_size), m_size, embedding_dim });
            const combined_permuted = combined_reshape.transpose(.{ 0, 2, 1, 3, 4 }).reshape(.{ 1, -1, embedding_dim });
            const t_u63: u63 = @intCast(t);

            const repeated = combined_permuted.repeat1d(0, t_u63).reshape(.{ -1, embedding_dim }); //meme embedding pour chaque temporalite
            outputs[0] = repeated;
        }
        log.info("outputs: {f}", .{outputs[0].shape()});
        return outputs;
    }

    pub fn rotaryPosEmbed(self: VisionTransformer, grid_thw: []const []const i32) Tensor {
        const m_size = 2; //spatial_merge_size

        const max_grid_size = std.mem.max(i32, grid_thw[0]);
        _ = max_grid_size; // autofix
        const t = grid_thw[0][0];
        const h = grid_thw[0][1];
        const w = grid_thw[0][2];
        var hpos_ids = zml.torch.unsqueeze(zml.Tensor.arange(.{ .start = 0, .end = h, .step = 1 }, .f32), 1).repeat1d(1, @as(u63, @intCast(w)));
        hpos_ids = hpos_ids.reshape(.{ @divExact(h, m_size), m_size, @divExact(w, m_size), m_size });
        hpos_ids = hpos_ids.transpose(.{ 0, 2, 1, 3 });
        hpos_ids = hpos_ids.flatten();
        log.info("hpos_ids: {f}", .{hpos_ids.shape()});

        var wpos_ids = zml.torch.unsqueeze(zml.Tensor.arange(.{ .start = 0, .end = w, .step = 1 }, .f32), 0).repeat1d(0, @as(u63, @intCast(h)));
        wpos_ids = wpos_ids.reshape(.{ @divExact(h, m_size), m_size, @divExact(w, m_size), m_size });
        wpos_ids = wpos_ids.transpose(.{ 0, 2, 1, 3 });
        wpos_ids = wpos_ids.flatten();
        log.info("wpos_ids: {f}", .{wpos_ids.shape()});
        // hpos_ids = zml.torch.unsqueeze(hpos_ids, 1);
        // wpos_ids = zml.torch.unsqueeze(wpos_ids, 1);
        const pos_ids = zml.Tensor.stack(&[2]Tensor{ hpos_ids, wpos_ids }, 1, .layers).repeat1d(1, @as(u63, @intCast(t))).convert(.i32);
        log.info("pos_ids: {f}", .{pos_ids.shape()});
        const rotary_pos_emb_full = zml.call(self.rotary_pos_emb, .forward, .{});
        log.info("rotary_pos_emb: {f}", .{rotary_pos_emb_full.shape()});
        log.info("pos_ids: {f}", .{pos_ids.shape()});
        const output = rotary_pos_emb_full.gather(.{ .d = pos_ids }, .{}).merge(.{ .d = .{ .layers, .s } });
        log.info("output: {f}", .{output.shape()});
        //const freq_table = zml.call(self.rotary_pos_emb, .forward, .{});
        return output;
    }

    pub fn forward(self: VisionTransformer, x: Tensor, grid_thw: Tensor) struct { Tensor, [3]Tensor } {
        const mock_grid_thw = [3]i32{ 1, 86, 128 };
        const mock_grid_thw_array = [_][]const i32{&mock_grid_thw};
        var pos_embeds = self.fastPosEmbedInterpolate(&mock_grid_thw_array);
        var rotary_pos_emb = self.rotaryPosEmbed(&mock_grid_thw_array);
        rotary_pos_emb = zml.Tensor.concatenate(&.{ rotary_pos_emb, rotary_pos_emb }, 1);
        var hidden_states = zml.call(self.vision_patch_embed, .forward, .{x});
        log.info("hidden_states_embed: {f}", .{hidden_states.shape()});
        log.info("pos_embeds[0]: {f}", .{pos_embeds[0].shape()});
        log.info("rotary_pos_emb: {f}", .{rotary_pos_emb.shape()});
        hidden_states = hidden_states.add(pos_embeds[0].convert(hidden_states.dtype()));
        const cos = rotary_pos_emb.cos();
        const sin = rotary_pos_emb.sin();
        log.info("cos: {f}", .{cos.shape()});
        log.info("sin: {f}", .{sin.shape()});
        const deepstack_visual_indexes = [3]u32{ 5, 11, 17 };
        var count: usize = 0;
        var deepstack_features_list: [3]Tensor = undefined;
        for (0.., self.blocks) |layer, block| {
            hidden_states = zml.call(block, .forward, .{ hidden_states, grid_thw, cos, sin });
            log.info("layer: {d}", .{layer});
            for (deepstack_visual_indexes) |index| {
                log.info("index: {d}", .{index});
                if (layer == index) {
                    log.info("deepstack features list count: {d}", .{count});
                    deepstack_features_list[count] = zml.torch.unsqueeze(zml.call(self.deepstack_patch_mergers[count], .forward, .{ hidden_states, true }), 0);
                    log.info("deepstack_features_list[{d}]: {f}", .{ count, deepstack_features_list[count].shape() });
                    count += 1;
                }
            }
        }
        hidden_states = zml.call(self.patch_merger, .forward, .{ hidden_states, false });
        return .{ hidden_states, deepstack_features_list };
    }
};

// input : kernel spatials dims : [time, height, width]
// output : padding : [front, back, top, bottom, left, right]
// fn getPaddingForSame(kernel_dims: [3]i64) [6]i64 {
//     for (0.., kernel_dims) |i, k| {
//         log.info("kernel_dims: {d} at index {d}", .{ k, i });
//     }
//     var res: [6]i64 = undefined;
//     for (kernel_dims, 0..) |k, out| {
//         log.info("kernel_dims: {d}", .{k});
//         const pad = @divFloor(k, 2);
//         res[2 * out] = pad;
//         res[2 * out + 1] = pad;
//     }
//     return res;
// }

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

    pub fn forward(self: Conv3d, input: Tensor) Tensor {
        //const x = if (input.rank() == 4) zml.torch.unsqueeze(input, 0) else input;
        const x = input;

        var strides: [3]i64 = undefined;
        for (self.weight.dims()[2..5], 0..) |k, out| strides[out] = k;

        const padding = getPaddingForSame(x.dims()[2..5].*, self.weight.dims()[2..5].*, strides);
        for (0.., padding) |i, d| {
            log.info("padding_dims: {d} at index {d}", .{ d, i });
        }
        //const weight = self.weight.convert(x.dtype()); // Convertir le poids au même type que l'input
        var y = x.conv3d(self.weight.convert(x.dtype()), .{ .padding = &.{ 0, 0, 0, 0, 0, 0 }, .window_strides = &.{ 2, 16, 16 } });
        //, .window_strides = &strides
        if (self.bias) |b| y = y.add(b.convert(y.dtype()).broadcast(y._shape, &.{1}));
        //return if (input.rank() == 3) y.squeeze(0) else y;
        return y;
    }
};

pub const VisionBlock = struct {
    norm1: zml.nn.LayerNorm,
    norm2: zml.nn.LayerNorm,
    attn: VisionAttention, // Ici window attention
    mlp: VisionMlp,

    // Config values pushed down
    num_heads: u32,
    //window_size: u32,
    //is_full_attention: bool, // based on fullatt_block_indexes
    pub fn forward(self: VisionBlock, hidden_states: Tensor, cu_seqlen: Tensor, cos: Tensor, sin: Tensor) Tensor {
        log.info("hidden_states dims: {f}", .{hidden_states.shape()});
        log.info("cu_seqlen dims: {f}", .{cu_seqlen.shape()});
        log.info("cos dims: {f}", .{cos.shape()});
        log.info("sin dims: {f}", .{sin.shape()});
        const x = zml.call(self.norm1, .forward, .{hidden_states});
        const x1 = hidden_states.add(zml.call(self.attn, .forward, .{ x, cu_seqlen, cos, sin }));
        log.info("x1 dims: {f}", .{x1.shape()});
        const x2 = zml.call(self.norm2, .forward, .{x1});
        const x3 = x1.add(zml.call(self.mlp, .forward, .{x2}));
        return x3; //REUSE BUFFER Je pense
    }
};

// Necessite une couche convolutiom=nnelle 3D

pub const VisionPatchEmbed = struct {
    // Linear layer for patch embedding (fallback temporaire)
    proj: Conv3d,

    // Config values
    patch_size: u32 = 14,
    temporal_patch_size: u32 = 2,
    in_channels: u32 = 3,
    embed_dim: u32 = 1152, // correspond à hidden_size dans la config

    pub fn init(
        allocator: std.mem.Allocator,
        patch_size: u32,
        temporal_patch_size: u32,
        in_channels: u32,
        embed_dim: u32,
        store: zml.aio.BufferStore,
    ) !VisionPatchEmbed {
        const conv3d = try zml.aio.populateModelWithPrefix(Conv3d, allocator, store, "model.visual.patch_embed.proj");

        return .{
            .proj = conv3d,
            .patch_size = patch_size,
            .temporal_patch_size = temporal_patch_size,
            .in_channels = in_channels,
            .embed_dim = embed_dim,
        };
    }

    pub fn forward(self: VisionPatchEmbed, hidden_states: Tensor) Tensor {
        // 1. Reshape pour la convolution 3D
        // hidden_states.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        const reshaped = hidden_states.reshape(.{ -1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size });

        // 2. Appliquer la convolution 3D
        const conv_output = zml.call(self.proj, .forward, .{reshaped});

        // 3. Reshape final pour avoir [batch_size, embed_dim]
        // .view(-1, self.embed_dim)

        const reshaped_output = conv_output.reshape(.{ conv_output.dim(0), conv_output.dim(1) });
        for (0.., reshaped_output.dims()) |i, d| {
            log.info("reshaped_output_dims: {d} at index {d}", .{ d, i });
        }
        return reshaped_output;
    }
};

pub const VisionRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    dim: u32,

    pub fn init(allocator: std.mem.Allocator, dim: u32, theta: f32) !VisionRotaryEmbedding {
        _ = allocator;
        return .{
            .rope_opts = zml.nn.RopeOpts{
                .layout = .sequential,
                .freq_base = theta,
                .scaling = .{ .default = {} },
            },
            .dim = dim,
        };
    }

    pub fn forward(self: VisionRotaryEmbedding) Tensor {
        const seqlen = 128;
        // Utiliser directement invFreq de ZML
        log.info("THETAself.rope_opts: {d}", .{self.rope_opts.freq_base});
        const inv_freq = zml.nn.invFreq(@intCast(32), self.rope_opts).withTags(.{.s});
        //inv_freq par rapport a une dim;
        log.info("inv_freq : {f}", .{inv_freq.shape()});
        // Créer la séquence de positions
        const seq = zml.Tensor.arange(.{ .end = seqlen }, .f32).withTags(.{.d});

        // Produit tensoriel
        return zml.Tensor.outer(seq, inv_freq);
    }
};

pub fn rotate_half(x: Tensor) Tensor {
    const x1 = x.slice1d(-1, .{ .start = 0, .end = @divExact(x.dim(-1), 2) });
    const x2 = x.slice1d(-1, .{ .start = @divExact(x.dim(-1), 2), .end = x.dim(-1) });
    return Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
}

pub fn applyRotaryPositionalEmbedding(q: Tensor, k: Tensor, cos: Tensor, sin: Tensor) struct { Tensor, Tensor } {
    //const cos_dtype = cos.convert(q.dtype());
    //const sin_dtype = sin.convert(q.dtype());
    log.info("q dims: {f}", .{q.shape()});
    log.info("k dims: {f}", .{k.shape()});
    log.info("ICI ", .{});
    const cos_q_unsqueezed = zml.torch.unsqueeze(cos, -2).broad(q.shape()); //NECESSITTE POUR K ausii
    const sin_q_unsqueezed = zml.torch.unsqueeze(sin, -2).broad(q.shape());
    const cos_k_unsqueezed = zml.torch.unsqueeze(cos, -2).broad(k.shape());
    const sin_k_unsqueezed = zml.torch.unsqueeze(sin, -2).broad(k.shape());
    log.info("cos dims: {f}", .{cos.shape()});
    log.info("sin dims: {f}", .{sin.shape()});

    const q_dtype = q.convert(cos.dtype());
    const k_dtype = k.convert(cos.dtype());
    const q_embed = q_dtype.mul(cos_q_unsqueezed).add(rotate_half(q_dtype).mul(sin_q_unsqueezed)).withTags(.{ .q, .h, .hd });
    const k_embed = k_dtype.mul(cos_k_unsqueezed).add(rotate_half(k_dtype).mul(sin_k_unsqueezed)).withTags(.{ .k, .h, .hd });
    log.info("q_embed dims: {f}", .{q_embed.shape()});
    log.info("k_embed dims: {f}", .{k_embed.shape()});
    return .{ q_embed.convert(q.dtype()), k_embed.convert(k.dtype()) };
}

// Vision-specific components
pub const VisionAttention = struct {
    qkv: zml.nn.Linear,
    proj: zml.nn.Linear,

    num_heads: u32,
    //window_size: u32,
    //is_full_attention: bool,

    pub fn forward(self: VisionAttention, hidden_states: Tensor, cu_seqlen: Tensor, cos: Tensor, sin: Tensor) Tensor {
        _ = cu_seqlen;
        log.info("qkv dims: {f}", .{self.qkv.weight.shape()});

        const qkv = zml.call(self.qkv, .forward, .{hidden_states});
        log.info("qkv dims: {f}", .{qkv.shape()});

        const qkv_reshaped = qkv.reshape(.{
            hidden_states.dim(0), // seq_length
            3,
            16, //vision_config.num_heads
            -1, // head_dim calculé automatiquement
        });
        log.info("qkv_reshaped dims: {f}", .{qkv_reshaped.shape()});
        const qkv_permuted = qkv_reshaped.transpose(.{ 1, 0, 2, 3 }).withTags(.{ .qkv, .s, .h, .hd });
        log.info("qkv_permuted dims: {f}", .{qkv_permuted.shape()});
        const q = qkv_permuted.slice1d(0, .{ .end = 1 }).squeeze(0);
        log.info("q dims: {f}", .{q.shape()});
        const k = qkv_permuted.slice1d(0, .{ .start = 1, .end = 2 }).squeeze(0);
        log.info("k dims: {f}", .{k.shape()});
        const v = qkv_permuted.slice1d(0, .{ .start = 2, .end = 3 }).squeeze(0).withTags(.{ .k, .h, .hd });
        log.info("v dims: {f}", .{v.shape()});
        //const cos = postition_embedding.slice1d(1, .{ .end = 1 }).squeeze(-1);
        //const sin = postition_embedding.slice1d(1, .{ .start = 1, .end = 2 }).squeeze(-1);
        log.info("cos dims: {f}", .{cos.shape()});
        log.info("sin dims: {f}", .{sin.shape()});
        const q_embed, const k_embed = applyRotaryPositionalEmbedding(q, k, cos, sin);
        log.info("q_embed dims: {f}", .{q_embed.shape()});
        log.info("k_embed dims: {f}", .{k_embed.shape()});
        const attn_output = zml.nn.sdpa(q_embed, k_embed, v, .{ .allow_cudnn = true });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return zml.call(self.proj, .forward, .{attn});
    }
};

pub const PatchMerger = struct {
    norm: zml.nn.LayerNorm,
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,

    // Config values
    out_hidden_size: u32,

    pub fn forward(self: PatchMerger, x: Tensor, use_post_shuffle_norm: bool) Tensor {
        const gelu = zml.nn.Activation{ .gelu = {} };
        var x1 = if (use_post_shuffle_norm) zml.call(self.norm, .forward, .{x.reshape(.{ -1, 1024 * 4 })}) else zml.call(self.norm, .forward, .{x});
        x1 = x1.reshape(.{ -1, 1024 * 4 });
        const x2 = zml.call(self.linear_fc1, .forward, .{x1});
        const x3 = gelu.forward(x2);
        const x4 = zml.call(self.linear_fc2, .forward, .{x3});
        return x4;
    }
};

pub const VisionMlp = struct { // MLP classique
    linear_fc1: zml.nn.Linear,
    linear_fc2: zml.nn.Linear,
    hidden_act: zml.nn.Activation,
    pub fn forward(self: VisionMlp, x: Tensor) Tensor {
        const x1 = zml.call(self.linear_fc1, .forward, .{x});
        log.info("x1 mlp dims: {f}", .{x1.shape()});
        const gelu_tanh_approximation = zml.nn.Activation{ .gelu = {} }; // a verifier si equivalent a gelu pytorch tanh (je crois que oui)

        const x2 = gelu_tanh_approximation.forward(x1.convert(.f32)).convert(x.dtype());
        log.info("x2 mlp dims: {f}", .{x2.shape()});
        //const x2 = x1.quickGelu();
        const x3 = zml.call(self.linear_fc2, .forward, .{x2});
        log.info("x3 mlp dims: {f}", .{x3.shape()});
        return x3;
    }
};

//========================Text model========================

pub const TextModel = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,
    rotary_embed: TextRotaryEmbedding,

    // Config values pushed down
    max_seq_len: u32 = 1000, // a definir en option
    num_heads: u32,
    num_kv_heads: u32,
    // rope_opts: zml.nn.RopeOpts = .{
    //     .layout = .interleaved,
    //     .freq_base = 10_000,
    // },

    pub fn init(allocator: std.mem.Allocator, config: Qwen3VL.Config, store: zml.aio.BufferStore) !TextModel {
        // const rope_opts: zml.nn.RopeOpts = .{
        //     .layout = if (config.hf_rope_impl) .sequential else .interleaved,
        //     .freq_base = config.rope_theta,
        //     //.scaling = config.rope_scaling,
        // };
        const layers = try allocator.alloc(TransformerLayer, config.text_config.num_hidden_layers);
        var prefix = try zml.aio.PrefixBuilder.initCapacity(allocator, 1024);
        try prefix.push(stdx.noalloc, "model.language_model.layers");
        for (0.., layers) |i, *layer| {
            try prefix.pushDigit(stdx.noalloc, i);
            defer prefix.pop();
            var self_attn = try zml.aio.populateModelWithPrefix(SelfAttn, allocator, store, prefix.concat("self_attn"));
            self_attn.num_heads = config.text_config.num_attention_heads;
            self_attn.num_kv_heads = config.text_config.num_key_value_heads;
            // self_attn.rope_opts = rope_opts;

            const mlp = try zml.aio.populateModelWithPrefix(Mlp, allocator, store, prefix.concat("mlp"));
            //mlp.hidden_act = config.hidden_act;

            var input_layernorm = try zml.aio.populateModelWithPrefix(RmsNorm, allocator, store, prefix.concat("input_layernorm"));
            input_layernorm.eps = config.text_config.rms_norm_eps;

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
            // .rope_opts = .{
            //     .layout = if (config.hf_rope_impl) .sequential else .interleaved,
            //     .freq_base = config.rope_theta,
            //     //.scaling = config.rope_scaling,
            // },
            .rotary_embed = try TextRotaryEmbedding.init(allocator, config.text_config.hidden_size, 5000000.0),
        };
    }

    /// position_ids: (3, bs, seq_len)
    /// attention_mask: (bs, seq_len)
    /// inputs_embeds: (bs, seq_len, hidden_size)
    /// cache_position: (seq_len)
    /// visual_pos_masks: (bs, seq_len)
    pub fn forward(self: TextModel, position_ids: Tensor, attention_mask: Tensor, inputs_embeds: Tensor, cache_position: Tensor, visual_pos_masks: Tensor, deepstack_visual_embeds: [3]Tensor) Tensor {
        _ = cache_position; // autofix
        _ = attention_mask; // autofix
        _ = visual_pos_masks; // autofix
        //const embeds = zml.call(self.embed_tokens, .forward, .{tokens});
        var hidden_states = inputs_embeds;
        const attn_mask = zml.nn.causalAttnMask(.{ .q = 2766, .k = 2766 }, inputs_embeds.dtype(), null);
        const cos, const sin = self.rotary_embed.forward(inputs_embeds, position_ids);
        var count: u32 = 0;

        for (self.layers, 0..) |layer, i| {
            _ = i; // autofix
            hidden_states = zml.call(layer, .forward, .{ hidden_states, attn_mask, position_ids, cos, sin });
            if (count < deepstack_visual_embeds.len) {
                hidden_states = hidden_states.convert(.f32).dynamicUpdateSlice(.{ .seq = zml.Tensor.scalar(4, .u32) }, deepstack_visual_embeds[count]).convert(hidden_states.dtype());
                count += 1;
            }
        }
        const output = zml.call(self.norm, .forward, .{hidden_states});
        return output;
    }
};

pub const TextRotaryEmbedding = struct {
    rope_opts: zml.nn.RopeOpts,
    dim: u32,

    pub fn init(allocator: std.mem.Allocator, dim: u32, theta: f32) !TextRotaryEmbedding {
        _ = allocator;
        return .{
            .rope_opts = zml.nn.RopeOpts{
                .layout = .sequential,
                .freq_base = theta,
                .scaling = .{ .default = {} },
            },
            .dim = dim,
        };
    }
    pub fn forward(self: TextRotaryEmbedding, x: Tensor, position_ids: Tensor) struct { Tensor, Tensor } {
        _ = x; // autofix
        const mrope_section = [3]i32{ 24, 20, 20 }; // from config 24 +20 +20 = 64 soit hd //2
        const inv_freq = zml.nn.invFreq(@intCast(128), self.rope_opts).withTags(.{.s}).convert(.f32);
        const inv_freq_expanded = inv_freq.reshape(.{ -1, 1 }); // repat 3 times (t, h, w) on dim 0, pos id 1dim on dim 1 (number of images I think) (3, batch size, dim_head//2, 1)
        const position_ids_expanded = position_ids.reshape(.{ 3, @as(u32, @intCast(position_ids.dim(1))), 1, -1 }).convert(.f32); // (3, bs, 1, seq len)
        log.info("inv_freq_expanded: {f}", .{inv_freq_expanded.shape()});
        log.info("position_ids_expanded: {f}", .{position_ids_expanded.shape()});
        var freqs = inv_freq_expanded.matmul(position_ids_expanded).transpose(.{ 0, 1, 3, 2 }); // (3, bs, dim_head//2, seq len)
        // interleaved mrope
        var freqs_t = freqs.slice1d(0, .{ .start = 0, .end = 1 }).squeeze(0).withTags(.{ .bs, .seq, .dh });
        const freqs_h = freqs.slice1d(0, .{ .start = 1, .end = 2 }).squeeze(0).withTags(.{ .bs, .seq, .dh });
        const freqs_w = freqs.slice1d(0, .{ .start = 2, .end = 3 }).squeeze(0).withTags(.{ .bs, .seq, .dh });
        log.info("freqs_t: {f}", .{freqs_t.shape()});
        log.info("freqs_h: {f}", .{freqs_h.shape()});
        log.info("freqs_w: {f}", .{freqs_w.shape()});
        const indices = zml.Tensor.iota(Shape.init(.{ .h = @as(u32, @intCast(mrope_section[1])) }, .i32), .h);
        const h_indices = indices.scale(3).addConstant(1);
        const w_indices = indices.scale(3).addConstant(2);

        log.info("h_indices: {f}", .{h_indices.shape()});
        log.info("w_indices: {f}", .{w_indices.shape()});
        const h_input = freqs_h.gather(.{ .dh = h_indices }, .{ .indices_are_sorted = true });
        const w_input = freqs_w.gather(.{ .dh = w_indices }, .{ .indices_are_sorted = true });
        log.info("h_input: {f}", .{h_input.shape()});
        log.info("w_input: {f}", .{w_input.shape()});
        freqs_t = freqs_t.scatterSlices(.{ .dh = h_indices }, h_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        log.info("freqs_t: {f}", .{freqs_t.shape()});
        freqs = freqs_t.scatterSlices(.{ .dh = w_indices }, w_input, .{ .update_fn = zml.Tensor.ScatterOpts.override });
        log.info("freqs: {f}", .{freqs.shape()});
        const emb = zml.Tensor.concatenate(&.{ freqs, freqs }, -1);
        const cos = emb.cos().convert(.f16);
        const sin = emb.sin().convert(.f16);
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
        x: Tensor,
        attn_mask: Tensor,
        token_index: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) Tensor {
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        //stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});
        //const x0 = x.convert(.f32);
        const x0 = x;

        var x0_normalized = zml.call(self.input_layernorm, .forward, .{x0});
        x0_normalized = x0_normalized.print();
        const delta0 = zml.call(self.self_attn, .forward, .{
            x0_normalized,
            attn_mask,
            token_index,
            cos,
            sin,
        }).convert(x0.dtype());
        log.info("delta0: {f}", .{delta0.shape()});
        var x1 = x.add(delta0);
        x1 = x1.print();
        log.info("x1: {f}", .{x1.shape()});
        // Fully Connected
        var x1_normalized = zml.call(self.post_attention_layernorm, .forward, .{x1});
        x1_normalized = x1_normalized.print();
        log.info("x1_normalized: {f}", .{x1_normalized.shape()});
        var x2 = zml.call(self.mlp, .forward, .{x1_normalized}).add(x1);
        x2 = x2.print();
        log.info("x2: {f}", .{x2.shape()});
        return x2.convert(x.dtype());
    }
};

// Reuse Llama components
pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,

    q_norm: RmsNorm,
    k_norm: RmsNorm,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    // rope_opts: zml.nn.RopeOpts = undefined,

    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        attention_mask: Tensor,
        // kv_cache: KvCache, marche pas pour test, necessite de "developper la struct"
        position_ids: Tensor,
        cos: Tensor,
        sin: Tensor,
    ) Tensor {
        _ = attention_mask; // autofix
        _ = position_ids; // autofix
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        _ = num_kv_heads; // autofix
        log.info("q_proj: {f}", .{self.q_proj.weight.shape()});
        log.info("k_proj: {f}", .{self.k_proj.weight.shape()});
        log.info("v_proj: {f}", .{self.v_proj.weight.shape()});
        var q = zml.call(self.q_proj, .forward, .{x}).splitAxis(-1, .{ .h = 32, .hd = .auto }).convert(x.dtype());
        var k = zml.call(self.k_proj, .forward, .{x}).splitAxis(-1, .{ .h = 8, .hd = .auto }).convert(x.dtype());
        var v = zml.call(self.v_proj, .forward, .{x}).splitAxis(-1, .{ .h = 8, .hd = .auto }).convert(x.dtype());
        const attn_mask = zml.nn.causalAttnMask(.{ .q = 2766, .k = 2766 }, x.dtype(), null);
        // const seq_len = token_index.dim(1); // a changer
        // log.info("seq_len: {d}", .{seq_len});
        // var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), null);
        // attn_mask = attn_mask.print();
        // attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = seq_len }, attn_mask.dtype()), token_index.reshape(.{ seq_len, 1 }), .{});
        // attn_mask = attn_mask.print();
        // log.info("attn_mask: {f}", .{attn_mask.shape()});
        // In self-attention, .s axis is used both for keys and queries.
        // const pos_index = b: {
        //     const temp = Tensor.arange(.{ .end = seq_len }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = seq_len }, token_index.dtype()));
        //     break :b temp.add(token_index.broad(temp.shape()));
        // };
        // _ = pos_index; // autofix

        log.info("q_norm: {f}", .{self.q_norm.weight.shape()});
        log.info("k_norm: {f}", .{self.k_norm.weight.shape()});
        log.info("q: {f}", .{q.shape()});
        log.info("k: {f}", .{k.shape()});
        log.info("v: {f}", .{v.shape()});
        q = zml.call(self.q_norm, .forward, .{q.rename(.{ .hd = .d })}).rename(.{ .d = .hd }).squeeze(0); //squeeze sur la dim batch pour le test
        k = zml.call(self.k_norm, .forward, .{k.rename(.{ .hd = .d })}).rename(.{ .d = .hd }).squeeze(0);
        v = v.squeeze(0);
        q = q.withTags(.{ .q, .h, .hd });
        k = k.withTags(.{ .k, .h, .hd });
        v = v.withTags(.{ .k, .h, .hd });
        log.info("q: {f}", .{q.shape()});
        log.info("k: {f}", .{k.shape()});
        log.info("v: {f}", .{v.shape()});
        q, k = applyRotaryPositionalEmbedding(q, k, cos.squeeze(0), sin.squeeze(0));
        // q = zml.nn.rope(q, pos_index, self.rope_opts);
        // k = zml.nn.rope(k, pos_index, self.rope_opts);

        log.info("q: {f}", .{q.shape()});
        log.info("k: {f}", .{k.shape()});
        log.info("v: {f}", .{v.shape()});

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .allow_cudnn = true });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        log.info("attn_output: {f}", .{attn_output.shape()});
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return zml.torch.unsqueeze(zml.call(self.o_proj, .forward, .{attn}), 0);
    }
};

pub const Mlp = struct {
    up_proj: zml.nn.Linear, // (dim -> hidden_dim)
    gate_proj: zml.nn.Linear, // (dim -> hidden_dim)
    down_proj: zml.nn.Linear, // (hidden_dim -> dim)
    //hidden_act: []const u8,

    pub fn forward(self: Mlp, x: Tensor) Tensor {
        const proj = zml.call(self.up_proj, .forward, .{x});
        var output = zml.call(self.gate_proj, .forward, .{x});
        output = output.silu().mul(proj);
        return zml.call(self.down_proj, .forward, .{output});
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const normalized = zml.nn.rmsNorm(x, .d, 1e-6);
        return normalized.mul(self.weight.convert(x.dtype()).withTags(.{.d}).broad(x.shape()));
    }
};

// KV Cache from Llama
pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,
};
