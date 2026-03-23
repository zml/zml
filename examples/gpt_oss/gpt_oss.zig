const std = @import("std");

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;
const ops = zml.ops;
const moe = zml.moe;

const log = std.log.scoped(.gpt_oss);

pub const TransferCtx = struct {
    allocator: std.mem.Allocator,
    pool: *zml.mem.DynamicBufferPool,
    transferred_bytes: *usize,
    progress: *std.Progress.Node,
};

pub const GptOss = struct {
    pub const Config = struct {
        bos_token_id: u32 = 199998,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []const u32,
        }),
        hidden_size: u32,
        head_dim: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        num_experts_per_tok: u32,
        rope_theta: f32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        sliding_window: u32,
        hf_rope_impl: bool = true,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
        num_local_experts: u32,
    };

    pub const Options = struct {
        sampling_strategy: zml.nn.SamplingStrategy,
        max_seq_len: u32,
        max_prompt_len: u32,
        tokens_per_expert_ratio: f32,
    };

    pub const Mode = union(enum) {
        /// In prefill mode we pass the actual len of the prompt
        prefill: zml.Tensor,
        /// In gen mode we pass the position of the next token
        gen: zml.Tensor,
    };

    lm_head: ?zml.nn.Linear,
    model: Model,

    config: Config,
    options: Options,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config, options: Options) !GptOss {
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensor("weight", .{ .voc, .d }, null)) |weight|
            .init(weight, null, .d)
        else
            null;

        return .{
            .lm_head = lm_head,
            .model = try Model.init(allocator, store.withPrefix("model"), config, options),
            .config = config,
            .options = options,
        };
    }

    pub fn deinit(self: GptOss, allocator: std.mem.Allocator) void {
        self.model.deinit(allocator);
    }

    pub fn load(
        self: *const GptOss,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(GptOss) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ total_bytes, took, bytes_per_sec });
        }
        const loaded = try zml.io.load(GptOss, self, allocator, io, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .store = store,
            .shardings = shardings,
            .parallelism = 16,
            .total_bytes = &total_bytes,
        });
        return loaded;
    }

    pub fn loadBuffers(
        self: *const GptOss,
        bufferize_ctx: zml.io.BufferizeContext(TransferCtx),
        group: *stdx.Io.LimitedGroup,
        store: zml.io.TensorStore.View,
        cb: zml.io.CallbackTensorBufferTransfer(TransferCtx),
    ) !zml.Bufferized(GptOss) {
        var lm_head_bufferized: ?zml.Bufferized(zml.nn.Linear) = null;
        if (self.lm_head) |lm_head| {
            lm_head_bufferized = undefined;

            const transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &lm_head, &lm_head_bufferized.?, store.withPrefix("lm_head"));
            for (transfers) |t| try group.concurrent(bufferize_ctx.io, cb, .{t});
        }

        const model_bufferized = try self.model.loadBuffers(bufferize_ctx, group, store.withPrefix("model"), cb);

        return .{
            .lm_head = lm_head_bufferized,
            .model = model_bufferized,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GptOss), allocator: std.mem.Allocator) void {
        if (self.lm_head) |*lm_head| lm_head.weight.deinit();
        Model.unloadBuffers(&self.model, allocator);
    }

    pub fn forward(
        self: GptOss,
        tokens_: zml.Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        rng: Tensor.Rng,
        token_mask: ?Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { Tensor, KvCache, Tensor.Rng } {
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache = self.model.forward(tokens, token_index, kv_cache, token_mask, moe_metadata, moe_parameters);
        var new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.options.sampling_strategy);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng };
    }

    pub fn sampleTokens(
        self: GptOss,
        lm_head_: ?zml.nn.Linear,
        out_: Tensor,
        rng: Tensor.Rng,
        opts: zml.nn.SamplingStrategy,
    ) struct { Tensor, Tensor.Rng } {
        const out = out_.withPartialTags(.{ .s, .d });

        var logits = blk: {
            if (lm_head_) |lm_head| {
                break :blk lm_head.forward(out).rename(.{ .voc = .d });
            } else {
                break :blk self.model.embed_tokens.weight.withTags(.{ .voc, .d }).dot(out, .d);
            }
        };

        if (logits.shape().hasTag(.voc) == null)
            logits = logits.rename(.{ .d = .voc });

        const next_tokens, const new_rng = zml.nn.sampleTokens(logits, opts, rng);
        return .{ next_tokens, new_rng };
    }
};

pub fn preprocessFlashinferSm90Mxfp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: *zml.Bufferized(GptOss),
) !void {
    log.info("Preprocessing SM90 MXFP4 weights (static offline transform)", .{});
    for (model.model.layers) |*layer| {
        try padMoEForSm90Mxfp4(allocator, io, platform, &layer.mlp.experts);
        try preprocessGateUpWeightSwap(allocator, io, platform, &layer.mlp.experts.gate_up_proj.blocks);
        if (layer.mlp.experts.gate_up_proj.bias) |*bias| {
            try preprocessGateUpBiasSwap(allocator, io, platform, bias);
        }
        try preprocessGateUpScaleSwapAndInterleave(allocator, io, platform, &layer.mlp.experts.gate_up_proj.scale);
        try preprocessDownScaleInterleave(allocator, io, platform, &layer.mlp.experts.down_proj.scale);
    }
}

pub fn preprocessTritonSm90Mxfp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: *zml.Bufferized(GptOss),
) !void {
    log.info("Preprocessing SM90 MXFP4 weights for Triton backend", .{});
    for (model.model.layers) |*layer| {
        try padMoEForSm90Mxfp4(allocator, io, platform, &layer.mlp.experts);
        // Triton backend keeps gate/up ordering but needs scale interleave.
        try preprocessGateUpScaleInterleave(allocator, io, platform, &layer.mlp.experts.gate_up_proj.scale);
        try preprocessDownScaleInterleave(allocator, io, platform, &layer.mlp.experts.down_proj.scale);
    }
}

fn roundUp(x: usize, alignment: usize) usize {
    return ((x + alignment - 1) / alignment) * alignment;
}

fn padMoEForSm90Mxfp4(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    experts: *zml.Bufferized(Mlp),
) !void {
    const gate_blocks_shape = experts.gate_up_proj.blocks.shape();
    const down_blocks_shape = experts.down_proj.blocks.shape();
    const hidden = @as(usize, @intCast(gate_blocks_shape.dim(2))) * @as(usize, @intCast(gate_blocks_shape.dim(3))) * 2;
    const intermediate = @as(usize, @intCast(down_blocks_shape.dim(2))) * @as(usize, @intCast(down_blocks_shape.dim(3))) * 2;

    const padded_hidden = roundUp(hidden, 128);
    const padded_intermediate = roundUp(intermediate, 128);
    if (padded_hidden == hidden and padded_intermediate == intermediate) return;

    const gate_d_padded = padded_hidden / 32;
    const down_d_padded = padded_intermediate / 32;
    const gate_out_padded = 2 * padded_intermediate;
    const down_out_padded = padded_hidden;

    try padU8Tensor4d(allocator, io, platform, &experts.gate_up_proj.blocks, gate_out_padded, gate_d_padded);
    try padU8Tensor3d(allocator, io, platform, &experts.gate_up_proj.scale, gate_out_padded, gate_d_padded);
    if (experts.gate_up_proj.bias) |*bias| {
        try padU16Tensor2d(allocator, io, platform, bias, gate_out_padded);
    }

    try padU8Tensor4d(allocator, io, platform, &experts.down_proj.blocks, down_out_padded, down_d_padded);
    try padU8Tensor3d(allocator, io, platform, &experts.down_proj.scale, down_out_padded, down_d_padded);
    if (experts.down_proj.bias) |*bias| {
        try padU16Tensor2d(allocator, io, platform, bias, down_out_padded);
    }
}

fn padU8Tensor4d(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
    out_padded: usize,
    d_padded: usize,
) !void {
    const sh = buf.shape();
    const e = @as(usize, @intCast(sh.dim(0)));
    const o = @as(usize, @intCast(sh.dim(1)));
    const d = @as(usize, @intCast(sh.dim(2)));
    const b = @as(usize, @intCast(sh.dim(3)));
    if (o == out_padded and d == d_padded) return;

    var src = try zml.Slice.alloc(allocator, sh);
    defer src.free(allocator);
    try buf.toSlice(io, src);
    const src_items = src.items(u8);

    var dst_shape = sh;
    dst_shape = dst_shape.setDim(1, @intCast(out_padded));
    dst_shape = dst_shape.setDim(2, @intCast(d_padded));
    var dst = try zml.Slice.alloc(allocator, dst_shape);
    defer dst.free(allocator);
    @memset(dst.items(u8), 0);
    const dst_items = dst.items(u8);

    const src_row = d * b;
    const dst_row = d_padded * b;
    for (0..e) |ei| {
        for (0..o) |oi| {
            const src_off = (ei * o + oi) * src_row;
            const dst_off = (ei * out_padded + oi) * dst_row;
            @memcpy(dst_items[dst_off .. dst_off + src_row], src_items[src_off .. src_off + src_row]);
        }
    }

    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, dst, sharding);
    buf.* = new_buf;
}

fn padU8Tensor3d(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
    out_padded: usize,
    d_padded: usize,
) !void {
    const sh = buf.shape();
    const e = @as(usize, @intCast(sh.dim(0)));
    const o = @as(usize, @intCast(sh.dim(1)));
    const d = @as(usize, @intCast(sh.dim(2)));
    if (o == out_padded and d == d_padded) return;

    var src = try zml.Slice.alloc(allocator, sh);
    defer src.free(allocator);
    try buf.toSlice(io, src);
    const src_items = src.items(u8);

    var dst_shape = sh;
    dst_shape = dst_shape.setDim(1, @intCast(out_padded));
    dst_shape = dst_shape.setDim(2, @intCast(d_padded));
    var dst = try zml.Slice.alloc(allocator, dst_shape);
    defer dst.free(allocator);
    @memset(dst.items(u8), 0);
    const dst_items = dst.items(u8);

    const src_row = d;
    const dst_row = d_padded;
    for (0..e) |ei| {
        for (0..o) |oi| {
            const src_off = (ei * o + oi) * src_row;
            const dst_off = (ei * out_padded + oi) * dst_row;
            @memcpy(dst_items[dst_off .. dst_off + src_row], src_items[src_off .. src_off + src_row]);
        }
    }

    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, dst, sharding);
    buf.* = new_buf;
}

fn padU16Tensor2d(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
    out_padded: usize,
) !void {
    const sh = buf.shape();
    const e = @as(usize, @intCast(sh.dim(0)));
    const o = @as(usize, @intCast(sh.dim(1)));
    if (o == out_padded) return;

    var src = try zml.Slice.alloc(allocator, sh);
    defer src.free(allocator);
    try buf.toSlice(io, src);
    const src_items = src.items(u16);

    var dst_shape = sh;
    dst_shape = dst_shape.setDim(1, @intCast(out_padded));
    var dst = try zml.Slice.alloc(allocator, dst_shape);
    defer dst.free(allocator);
    @memset(dst.items(u16), 0);
    const dst_items = dst.items(u16);

    for (0..e) |ei| {
        const src_off = ei * o;
        const dst_off = ei * out_padded;
        @memcpy(dst_items[dst_off .. dst_off + o], src_items[src_off .. src_off + o]);
    }

    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, dst, sharding);
    buf.* = new_buf;
}

fn preprocessGateUpWeightSwap(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
) !void {
    const sh = buf.shape();
    if (sh.rank() != 4) return error.InvalidGateUpWeightRank;
    const experts: usize = @intCast(sh.dim(0));
    const out: usize = @intCast(sh.dim(1));
    const d: usize = @intCast(sh.dim(2));
    const d_block: usize = @intCast(sh.dim(3));
    if ((@mod(out, 2)) != 0) return error.InvalidGateUpWeightOut;

    var host = try zml.Slice.alloc(allocator, sh);
    defer host.free(allocator);
    try buf.toSlice(io, host);

    const bytes = host.items(u8);
    const row_bytes = d * d_block;
    const half = @divExact(out, 2);
    const total = experts * out * row_bytes;
    if (bytes.len != total) return error.InvalidGateUpWeightStorageSize;
    var tmp = try allocator.alloc(u8, total);
    defer allocator.free(tmp);

    for (0..experts) |e| {
        for (0..out) |o_new| {
            const o_old = if (o_new < half) (2 * o_new + 1) else (2 * (o_new - half));
            const src_off = ((e * out) + o_old) * row_bytes;
            const dst_off = ((e * out) + o_new) * row_bytes;
            @memcpy(tmp[dst_off .. dst_off + row_bytes], bytes[src_off .. src_off + row_bytes]);
        }
    }

    @memcpy(bytes, tmp);
    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, host, sharding);
    // Keep old buffer alive to avoid ownership/alias double-free during async init.
    // TODO: reclaim after introducing explicit ownership tracking for replaced buffers.
    buf.* = new_buf;
}

fn preprocessGateUpBiasSwap(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
) !void {
    const sh = buf.shape();
    if (sh.rank() != 2) return error.InvalidGateUpBiasRank;
    const experts: usize = @intCast(sh.dim(0));
    const out: usize = @intCast(sh.dim(1));
    if ((@mod(out, 2)) != 0) return error.InvalidGateUpBiasOut;

    var host = try zml.Slice.alloc(allocator, sh);
    defer host.free(allocator);
    try buf.toSlice(io, host);

    const vals = host.items(u16);
    const half = @divExact(out, 2);
    const total = experts * out;
    if (vals.len != total) return error.InvalidGateUpBiasStorageSize;
    var tmp = try allocator.alloc(u16, total);
    defer allocator.free(tmp);

    for (0..experts) |e| {
        for (0..out) |o_new| {
            const o_old = if (o_new < half) (2 * o_new + 1) else (2 * (o_new - half));
            tmp[e * out + o_new] = vals[e * out + o_old];
        }
    }

    @memcpy(vals, tmp);
    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, host, sharding);
    // Keep old buffer alive to avoid ownership/alias double-free during async init.
    // TODO: reclaim after introducing explicit ownership tracking for replaced buffers.
    buf.* = new_buf;
}

fn preprocessGateUpScaleSwapAndInterleave(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
) !void {
    const sh = buf.shape();
    if (sh.rank() != 3) return error.InvalidGateUpScaleRank;
    const experts: usize = @intCast(sh.dim(0));
    const out: usize = @intCast(sh.dim(1));
    const k32: usize = @intCast(sh.dim(2));
    if ((@mod(out, 2)) != 0) return error.InvalidGateUpScaleOut;
    if ((@mod(k32, 4)) != 0) return error.InvalidGateUpScaleK32;

    var host = try zml.Slice.alloc(allocator, sh);
    defer host.free(allocator);
    try buf.toSlice(io, host);

    const src = host.items(u8);
    const half = @divExact(out, 2);
    const total = experts * out * k32;
    if (src.len != total) return error.InvalidGateUpScaleStorageSize;
    var swapped = try allocator.alloc(u8, total);
    defer allocator.free(swapped);
    var interleaved = try allocator.alloc(u8, total);
    defer allocator.free(interleaved);

    for (0..experts) |e| {
        for (0..out) |o_new| {
            const o_old = if (o_new < half) (2 * o_new + 1) else (2 * (o_new - half));
            const src_off = (e * out + o_old) * k32;
            const dst_off = (e * out + o_new) * k32;
            @memcpy(swapped[dst_off .. dst_off + k32], src[src_off .. src_off + k32]);
        }
    }

    const k32_div4 = @divExact(k32, 4);
    for (0..experts) |e| {
        for (0..out) |n| {
            for (0..k32) |g| {
                const src_idx = (e * out + n) * k32 + g;
                const dst_idx = (e * k32_div4 + @divTrunc(g, 4)) * (out * 4) + n * 4 + @mod(g, 4);
                interleaved[dst_idx] = swapped[src_idx];
            }
        }
    }

    @memcpy(src, interleaved);
    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, host, sharding);
    // Keep old buffer alive to avoid ownership/alias double-free during async init.
    // TODO: reclaim after introducing explicit ownership tracking for replaced buffers.
    buf.* = new_buf;
}

fn preprocessDownScaleInterleave(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
) !void {
    const sh = buf.shape();
    if (sh.rank() != 3) return error.InvalidDownScaleRank;
    const experts: usize = @intCast(sh.dim(0));
    const out: usize = @intCast(sh.dim(1));
    const k32: usize = @intCast(sh.dim(2));
    if ((@mod(k32, 4)) != 0) return error.InvalidDownScaleK32;

    var host = try zml.Slice.alloc(allocator, sh);
    defer host.free(allocator);
    try buf.toSlice(io, host);

    const src = host.items(u8);
    const total = experts * out * k32;
    if (src.len != total) return error.InvalidDownScaleStorageSize;
    var interleaved = try allocator.alloc(u8, total);
    defer allocator.free(interleaved);

    const k32_div4 = @divExact(k32, 4);
    for (0..experts) |e| {
        for (0..out) |n| {
            for (0..k32) |g| {
                const src_idx = (e * out + n) * k32 + g;
                const dst_idx = (e * k32_div4 + @divTrunc(g, 4)) * (out * 4) + n * 4 + @mod(g, 4);
                interleaved[dst_idx] = src[src_idx];
            }
        }
    }

    @memcpy(src, interleaved);
    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, host, sharding);

    buf.* = new_buf;
}

fn preprocessGateUpScaleInterleave(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    buf: *Buffer,
) !void {
    const sh = buf.shape();
    if (sh.rank() != 3) return error.InvalidGateUpScaleRank;
    const experts: usize = @intCast(sh.dim(0));
    const out: usize = @intCast(sh.dim(1));
    const k32: usize = @intCast(sh.dim(2));
    if ((@mod(k32, 4)) != 0) return error.InvalidGateUpScaleK32;

    var host = try zml.Slice.alloc(allocator, sh);
    defer host.free(allocator);
    try buf.toSlice(io, host);

    const src = host.items(u8);
    const total = experts * out * k32;
    if (src.len != total) return error.InvalidGateUpScaleStorageSize;
    var interleaved = try allocator.alloc(u8, total);
    defer allocator.free(interleaved);

    const k32_div4 = @divExact(k32, 4);
    for (0..experts) |e| {
        for (0..out) |n| {
            for (0..k32) |g| {
                const src_idx = (e * out + n) * k32 + g;
                const dst_idx = (e * k32_div4 + @divTrunc(g, 4)) * (out * 4) + n * 4 + @mod(g, 4);
                interleaved[dst_idx] = src[src_idx];
            }
        }
    }

    @memcpy(src, interleaved);
    const sharding = buf.placement().sharding;
    const new_buf = try Buffer.fromSlice(io, platform, host, sharding);
    buf.* = new_buf;
}

pub const Model = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    norm: RmsNorm,
    layers: []TransformerLayer,

    max_seq_len: u32 = 0,
    num_heads: i64 = 32,
    num_kv_heads: i64 = 32,
    rope_opts: zml.nn.RopeOpts = .{
        .layout = .interleaved,
        .scaling = .{ .default = .{} },
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options) !Model {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, layer_id| {
            layer.* = try .init(store.withPrefix("layers").withLayer(layer_id), config, options, @intCast(layer_id));
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight", .{ .voc, .d }, null) },
            .norm = .{ .weight = store.withPrefix("norm").createTensor("weight", .{.d}, null), .eps = config.rms_norm_eps },
            .layers = layers,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn loadBuffers(
        self: *const Model,
        bufferize_ctx: zml.io.BufferizeContext(TransferCtx),
        group: *stdx.Io.LimitedGroup,
        store: zml.io.TensorStore.View,
        cb: zml.io.CallbackTensorBufferTransfer(TransferCtx),
    ) !zml.Bufferized(Model) {
        const layers = try bufferize_ctx.allocator.alloc(zml.Bufferized(TransformerLayer), self.layers.len);
        errdefer bufferize_ctx.allocator.free(layers);

        // Bufferize embed_tokens and norm using the shared async transfer path.
        var embed_tokens_bufferized: zml.Bufferized(zml.nn.TokenEmbedding) = undefined;
        {
            const transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &self.embed_tokens, &embed_tokens_bufferized, store.withPrefix("embed_tokens"));
            for (transfers) |t| try group.concurrent(bufferize_ctx.io, cb, .{t});
        }

        var norm_bufferized: zml.Bufferized(RmsNorm) = undefined;
        {
            const transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &self.norm, &norm_bufferized, store.withPrefix("norm"));
            for (transfers) |t| try group.concurrent(bufferize_ctx.io, cb, .{t});
        }

        for (layers, 0..) |*layer, i| {
            const transfers = try zml.io.bufferize(TransferCtx, bufferize_ctx, &self.layers[i], layer, store.withLayer(i));
            for (transfers) |t| try group.concurrent(bufferize_ctx.io, cb, .{t});
        }

        return .{
            .embed_tokens = embed_tokens_bufferized,
            .layers = layers,
            .norm = norm_bufferized,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        self.embed_tokens.weight.deinit();
        RmsNorm.unloadBuffers(&self.norm);
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        allocator.free(self.layers);
    }

    pub fn forward(
        self: Model,
        tokens: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        tokens_mask: ?Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { Tensor, KvCache } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;

        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache = layer.forward(hidden, token_index, updated_kv_cache.atLayer(i), tokens_mask, moe_metadata, moe_parameters);
        }
        const output = self.norm.forward(hidden);

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache) };
    }

    pub fn embed(embed_tokens_: zml.nn.TokenEmbedding, tokens_: Tensor) Tensor {
        return embed_tokens_.forward(tokens_).withPartialTags(.{.d});
    }
};

pub const TransformerLayer = struct {
    input_layernorm: RmsNorm,
    self_attn: SelfAttn,
    post_attention_layernorm: RmsNorm,
    mlp: MoE,

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options, layer_id: u32) !TransformerLayer {
        return .{
            .input_layernorm = .init(store.withPrefix("input_layernorm"), config.rms_norm_eps),
            .self_attn = try .init(store.withPrefix("self_attn"), config, layer_id),
            .post_attention_layernorm = .init(store.withPrefix("post_attention_layernorm"), config.rms_norm_eps),
            .mlp = .init(store.withPrefix("mlp"), config, options),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_layernorm);
        SelfAttn.unloadBuffers(&self.self_attn);
        RmsNorm.unloadBuffers(&self.post_attention_layernorm);
        MoE.unloadBuffers(&self.mlp);
    }

    pub fn forward(
        self: TransformerLayer,
        x0: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
        tokens_mask: ?Tensor,
        moe_metadata: zml.moe.Metadata,
        moe_parameters: zml.moe.Parameters,
    ) struct { Tensor, KvCache } {

        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_layernorm.forward(x0_replicated);

        const delta0, const updated_kv_cache = self.self_attn.forward(x0_normalized, token_index, kv_cache);
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });

        // Fully Connected
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        var x2 = self.mlp.forward(x1_normalized, tokens_mask, moe_metadata, moe_parameters);
        x2 = x2.withPartitioning(.{ .d = .replicated }).add(x1).withPartitioning(.{ .d = .replicated });
        //const x2 = self.moe.forward(x1_normalized).add(x1).rename(.{ .dout = .d });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

pub const SelfAttn = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    sinks: Tensor,

    q_norm: ?RmsNorm,
    k_norm: ?RmsNorm,

    sliding_window: ?u32,

    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config, layer_id: u32) !SelfAttn {
        return .{
            .q_proj = .init(store.withPrefix("q_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("q_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .k_proj = .init(store.withPrefix("k_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("k_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .v_proj = .init(store.withPrefix("v_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("v_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .o_proj = .init(store.withPrefix("o_proj").createTensor("weight", .{ .dout, .d }, null), store.withPrefix("o_proj").maybeCreateTensor("bias", .{.dout}, null), .d),
            .sinks = store.createTensor("sinks", .{.h}, null),
            .sliding_window = if (layer_id % 2 == 0) config.sliding_window else null,
            .q_norm = null,
            .k_norm = null,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .scaling = config.rope_scaling,
            },
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
        self.sinks.deinit();

        if (self.q_norm) |*q_norm| RmsNorm.unloadBuffers(q_norm);
        if (self.k_norm) |*k_norm| RmsNorm.unloadBuffers(k_norm);
    }

    /// Self Attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: SelfAttn,
        x: Tensor,
        token_index: Tensor,
        kv_cache: KvCache,
    ) struct { Tensor, KvCache } {
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;
        // Make hidden state replicated once and reuse it across q/k/v projections.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });
        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        // Generate the attention mask.
        const seq_len = kv_cache.k.dim(.k);
        var attn_mask = zml.nn.causalAttnMask(.{ .q = seq_len, .k = seq_len }, x.dtype(), self.sliding_window);

        // Note: in Pytorch it would be very inefficient to generate the full attn_mask,
        // then slice into it, but XLA is able to optimize this correctly.
        attn_mask = attn_mask.gatherSlices(zml.Shape.init(.{ .q = x.dim(.s) }, attn_mask.dtype()), token_index.reshape(.{ .coord = 1 }), .{});

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        if (self.q_norm) |norm| q = norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        if (self.k_norm) |norm| k = norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        q = zml.nn.rope(q, pos_index, self.rope_opts);
        k = zml.nn.rope(k, pos_index, self.rope_opts);
        q = q.rename(.{ .s = .q });
        k = k.rename(.{ .s = .k });
        v = v.rename(.{ .s = .k });
        q = q.withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const dtype = q.dtype();

        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);
        k = k.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .k = .replicated, .h = .model, .hd = .replicated });

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .softmax_bias = self.sinks, .allow_cudnn = true });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensor("weight", .{.d}, null), .eps = eps };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }

    /// L2 normalization of input tensor along `.d` axis.
    pub fn forward(self: RmsNorm, input: Tensor) Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        // Note: contrary to Llama here the full layer is done in .f32, not just the variance computation.
        const normalized = zml.nn.rmsNorm(x.convert(.f32), .d, self.eps);
        return normalized.mul(self.weight.convert(.f32).withTags(.{.d}).broad(x.shape())).convert(input.dtype());
    }
};

const MoE = struct {
    experts: Mlp,
    router: zml.nn.Linear,
    moe_opts: MoeOpts,

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options) MoE {
        //log.info("MoE.init: trying to load experts tensors...", .{});
        const moe_on_disk: OnDisk = .{ .router = zml.nn.Linear.init(
            store.createTensor("router.weight", .{ .exp, .d }, null),
            store.createTensor("router.bias", .{.exp}, null),
            .d,
        ), .experts = .{
            .down_proj_bias = store.createTensor("experts.down_proj_bias", .{ .expert, .d }, null),
            .down_proj_blocks = store.createTensor("experts.down_proj_blocks", .{ .expert, .out, .d, .d_blocks }, null),
            .down_proj_scales = store.createTensor("experts.down_proj_scales", .{ .expert, .out, .d }, null),
            .gate_up_proj_bias = store.createTensor("experts.gate_up_proj_bias", .{ .expert, .d }, null),
            .gate_up_proj_blocks = store.createTensor("experts.gate_up_proj_blocks", .{ .expert, .out, .d, .d_blocks }, null),
            .gate_up_proj_scales = store.createTensor("experts.gate_up_proj_scales", .{ .expert, .out, .d }, null),
        } };
        return OnDisk.rewrite(moe_on_disk, config.num_experts_per_tok, options);
    }

    pub fn deinit(self: MoE, allocator: std.mem.Allocator) void {
        self.experts.deinit(allocator);
        //self.router.deinit(allocator);
    }

    pub fn loadBuffers(self: *const MoE, allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, platform: zml.Platform) !zml.Bufferized(MoE) {
        const experts = zml.io.loadBuffersFromId(allocator, io, self.experts, store.withPrefix("experts"), platform);
        const router = zml.io.loadBuffersFromId(allocator, io, self.router, store.withPrefix("router"), platform);
        return .{ .experts = experts, .router = router };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MoE)) void {
        self.router.weight.deinit();
        if (self.router.bias) |*b| b.deinit();
        self.experts.gate_up_proj.blocks.deinit();
        self.experts.gate_up_proj.scale.deinit();
        if (self.experts.gate_up_proj.bias) |*b| b.deinit();
        self.experts.down_proj.blocks.deinit();
        self.experts.down_proj.scale.deinit();
        if (self.experts.down_proj.bias) |*b| b.deinit();
    }

    pub fn forward(self: MoE, input: Tensor, tokens_mask: ?Tensor, moe_metadata: zml.moe.Metadata, moe_parameters: zml.moe.Parameters) Tensor {
        return moe.moe(
            input,
            tokens_mask,
            self.router,
            self.experts.gate_up_proj.blocks,
            self.experts.gate_up_proj.scale,
            self.experts.gate_up_proj.bias,
            self.experts.down_proj.blocks,
            self.experts.down_proj.scale,
            self.experts.down_proj.bias,
            self.moe_opts.num_experts_per_tok,
            moe_metadata,
            moe_parameters,
        ) catch |err| stdx.debug.panic("moe backend failed: {}", .{err});
    }

    pub const OnDisk = struct {
        router: zml.nn.Linear,
        experts: struct {
            down_proj_bias: zml.Tensor,
            down_proj_blocks: zml.Tensor,
            down_proj_scales: zml.Tensor,
            gate_up_proj_bias: zml.Tensor,
            gate_up_proj_blocks: zml.Tensor,
            gate_up_proj_scales: zml.Tensor,
        },

        pub fn rewrite(on_disk: OnDisk, num_experts_per_tok: u32, options: GptOss.Options) MoE {
            const e = on_disk.experts;
            const hidden: i64 = e.down_proj_blocks.shape().dim(1);
            const intermediate: i64 = @divExact(e.gate_up_proj_blocks.shape().dim(1), 2);
            const padded_hidden: i64 = @intCast(roundUp(@intCast(hidden), 128));
            const padded_intermediate: i64 = @intCast(roundUp(@intCast(intermediate), 128));
            const gate_out_padded: i64 = 2 * padded_intermediate;
            const down_out_padded: i64 = padded_hidden;
            const gate_d_padded: i64 = @divExact(padded_hidden, 32);
            const down_d_padded: i64 = @divExact(padded_intermediate, 32);

            var gate_blocks = e.gate_up_proj_blocks.withTags(.{ .expert, .out, .d, .d_block });
            gate_blocks._shape = gate_blocks._shape.withPartitioning(.{ .expert = .model });
            gate_blocks._shape = gate_blocks._shape.setDim(1, gate_out_padded).setDim(2, gate_d_padded);
            var gate_scale = e.gate_up_proj_scales.withTags(.{ .expert, .out, .d });
            gate_scale._shape = gate_scale._shape.withPartitioning(.{ .expert = .model });
            gate_scale._shape = gate_scale._shape.setDim(1, gate_out_padded).setDim(2, gate_d_padded);
            var gate_bias = e.gate_up_proj_bias.withTags(.{ .expert, .d });
            gate_bias._shape = gate_bias._shape.withPartitioning(.{ .expert = .model });
            gate_bias._shape = gate_bias._shape.setDim(1, gate_out_padded);

            var down_blocks = e.down_proj_blocks.withTags(.{ .expert, .out, .d, .d_block });
            down_blocks._shape = down_blocks._shape.withPartitioning(.{ .expert = .model });
            down_blocks._shape = down_blocks._shape.setDim(1, down_out_padded).setDim(2, down_d_padded);
            var down_scale = e.down_proj_scales.withTags(.{ .expert, .out, .d });
            down_scale._shape = down_scale._shape.withPartitioning(.{ .expert = .model });
            down_scale._shape = down_scale._shape.setDim(1, down_out_padded).setDim(2, down_d_padded);
            var down_bias = e.down_proj_bias.withTags(.{ .expert, .d });
            down_bias._shape = down_bias._shape.withPartitioning(.{ .expert = .model });
            down_bias._shape = down_bias._shape.setDim(1, down_out_padded);

            return .{
                .experts = .{
                    .gate_up_proj = .{
                        // We need to bitcast the scale cause safetensors doesn't encode f8 types correctly
                        .scale = gate_scale,
                        // We don't bitcast here because PJRT doesn't handle packed host buffers
                        .blocks = gate_blocks,
                        .blocks_dtype = .f4e2m1,
                        .bias = gate_bias,
                    },
                    .down_proj = .{
                        .blocks = down_blocks,
                        .blocks_dtype = .f4e2m1,
                        .scale = down_scale,
                        .bias = down_bias,
                    },
                },

                .router = zml.nn.Linear.init(
                    blk: {
                        var weight = on_disk.router.weight.withTags(.{ .expert, .d });
                        weight._shape = weight._shape.withPartitioning(.{ .expert = .model, .d = .replicated });
                        break :blk weight;
                    },
                    blk: {
                        var bias = on_disk.router.bias.?.withTags(.{.expert});
                        bias._shape = bias._shape.withPartitioning(.{ .expert = .model });
                        break :blk bias;
                    },
                    .d,
                ),

                .moe_opts = .{
                    .num_experts_per_tok = num_experts_per_tok,
                    .tokens_per_expert_ratio = options.tokens_per_expert_ratio,
                    .normalization = .softmax,
                },
            };
        }
    };
};

pub const Mlp = struct {
    gate_up_proj: BlockScaledLinear,
    down_proj: BlockScaledLinear,

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        const dt = x.dtype();
        var gate, var up = zml.nn.splitRealImg(self.gate_up_proj.forward(x), .interleaved);
        gate = .minimum(gate, .scalar(7, dt));
        up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));

        const out = gate.quickGelu().mul(up.addConstant(1));
        return self.down_proj.forward(out);
    }

    pub fn format(self: Mlp, writer: *std.Io.Writer) std.Io.Writer.Error!void {
        try writer.print("Mlp(gate_up_proj=.{f}, down_proj=.{f})", .{ self.gate_up_proj, self.down_proj });
    }
};

pub const BlockScaledLinear = struct {
    blocks: zml.Tensor,
    scale: zml.Tensor,
    bias: ?zml.Tensor = null,
    blocks_dtype: zml.DataType,

    pub fn dequantize(self: BlockScaledLinear, dtype: zml.DataType) zml.Tensor {
        const blocks_0 = self.blocks.bitCast(self.blocks_dtype);
        const blocks = blocks_0.merge(.{ .d_block = .{ .d_block, .bitcast } });

        const scale = self.scale.bitCast(.f8e8m0);

        var dequantized_weight: zml.Tensor = .mul(
            blocks.convert(dtype),
            scale.convert(dtype).appendAxes(.{.d_block}),
        );

        dequantized_weight = dequantized_weight.merge(.{ .d = .{ .d, .d_block } });

        return dequantized_weight;
    }

    pub fn forward(self: BlockScaledLinear, x: zml.Tensor) zml.Tensor {
        const res_shape = x.shape().setDim(-1, self.blocks.dim(-3));

        const blocks_0 = self.blocks.bitCast(self.blocks_dtype);
        const blocks = blocks_0.merge(.{ .d_block = .{ .d_block, .bitcast } });

        const scale = self.scale.bitCast(.f8e8m0);

        const y = y: {
            var dequantized_weight: zml.Tensor = .mul(
                blocks.convert(x.dtype()),
                scale.convert(x.dtype()).appendAxes(.{.d_block}),
            );
            // log.info("dequantized weights {f}", .{dequantized_weight.shape()});
            var y = x.dot(dequantized_weight.merge(.{ .d = .{ .d, .d_block } }), .d);
            // std.log.warn("output shape: {f}", .{y});
            std.debug.assert(y.shape().eql(res_shape));
            y._shape = res_shape;
            break :y y;
        };
        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }

    pub fn format(self: BlockScaledLinear, writer: *std.Io.Writer) !void {
        try writer.print("BlockScaledLinear(blocks={f}, scale={f}, bias={?f}, dt={t})", .{ self.blocks, self.scale, self.bias, self.blocks_dtype });
    }
};

const MoeOpts = struct {
    num_experts_per_tok: u32,
    tokens_per_expert_ratio: ?f32 = 0.0,
    normalization: Normalization,

    pub const Normalization = enum { linear, softmax };
};

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });
        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initShape(kv_shape: zml.Shape) ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = zml.Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform, sharding: zml.sharding.Sharding) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), sharding, .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), sharding, .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32, sharding),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
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
            .layer_index = Tensor.scalar(layer_index, .u32),
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
