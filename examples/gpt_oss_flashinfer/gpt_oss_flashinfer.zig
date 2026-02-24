const std = @import("std");

const zml = @import("zml");
const Buffer = zml.Buffer;
const Tensor = zml.Tensor;
const ShapeOf = zml.ShapeOf;
const stdx = zml.stdx;
const ops = zml.ops;
const flashinfer_moe = zml.flashinfer_moe;

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
        const lm_head: ?zml.nn.Linear = if (store.withPrefix("lm_head").maybeCreateTensorWithTags("weight", .{ .voc, .d })) |weight|
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
        progress: *std.Progress.Node,
    ) !zml.Bufferized(GptOss) {
        progress.increaseEstimatedTotalItems(store.view().count());
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const bytes_per_sec: u64 = @intFromFloat(@as(f64, @floatFromInt(total_bytes)) / (@as(f64, @floatFromInt(took.nanoseconds)) / std.time.ns_per_s));
            log.info("Loaded weights [{Bi:.2}, {D}, {Bi:.2}/s]", .{
                total_bytes,
                stdx.fmt.fmtDuration(took),
                bytes_per_sec,
            });
        }
        const loaded = try zml.io.load(GptOss, self, allocator, io, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .store = store,
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
        host_buffer: Tensor,
        device_buffer: Tensor,
    ) struct { Tensor, KvCache, Tensor.Rng, Tensor, Tensor } {
        const tokens = tokens_.withPartialTags(.{.s});
        const out, const updated_kv_cache, const host_buffer_out, const device_buffer_out = self.model.forward(tokens, token_index, kv_cache, token_mask, host_buffer, device_buffer);
        var new_tokens, const new_rng = self.sampleTokens(self.lm_head, out, rng, self.options.sampling_strategy);
        return .{ new_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache, new_rng, host_buffer_out.reuseBuffer(host_buffer), device_buffer_out.reuseBuffer(device_buffer) };
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
                //break :blk lm_head.forward(out).rename(.{ .dout = .d });
                break :blk lm_head.forward(out);
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

    const new_buf = try Buffer.fromSlice(io, platform, dst);
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

    const new_buf = try Buffer.fromSlice(io, platform, dst);
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

    const new_buf = try Buffer.fromSlice(io, platform, dst);
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
    const new_buf = try Buffer.fromSlice(io, platform, host);
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
    const new_buf = try Buffer.fromSlice(io, platform, host);
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
    const new_buf = try Buffer.fromSlice(io, platform, host);
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
    const new_buf = try Buffer.fromSlice(io, platform, host);
    // Keep old buffer alive to avoid ownership/alias double-free during async init.
    // TODO: reclaim after introducing explicit ownership tracking for replaced buffers.
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
        .freq_base = 10_000,
    },

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: GptOss.Config, options: GptOss.Options) !Model {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);

        for (layers, 0..) |*layer, layer_id| {
            layer.* = try .init(store.withPrefix("layers").withLayer(layer_id), config, options, @intCast(layer_id));
        }

        return .{
            .embed_tokens = .{ .weight = store.createTensor("embed_tokens.weight") },
            .norm = .{ .weight = store.withPrefix("norm").createTensorWithTags("weight", .{.d}), .eps = config.rms_norm_eps },
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

    pub fn forward(self: Model, tokens: Tensor, token_index: Tensor, kv_cache: KvCache, tokens_mask: ?Tensor, host_buffer: Tensor, device_buffer: Tensor) struct { Tensor, KvCache, Tensor, Tensor } {
        const embeds = embed(self.embed_tokens, tokens);
        var hidden = embeds;

        var updated_kv_cache = kv_cache;
        var host_buffer_out = host_buffer;
        var device_buffer_out = device_buffer;
        for (self.layers, 0..) |layer, i| {
            hidden, updated_kv_cache, host_buffer_out, device_buffer_out = layer.forward(hidden, token_index, updated_kv_cache.atLayer(i), tokens_mask, host_buffer_out, device_buffer_out);
        }
        const output = self.norm.forward(hidden);

        return .{ output, updated_kv_cache.reuseBuffer(kv_cache), host_buffer_out.reuseBuffer(host_buffer), device_buffer_out.reuseBuffer(device_buffer) };
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
        host_buffer: Tensor,
        device_buffer: Tensor,
    ) struct { Tensor, KvCache, Tensor, Tensor } {
        var host_buffer_out = host_buffer;
        var device_buffer_out = device_buffer;
        // Self Attention
        //log.debug("TransformerLayer({f}) -> {f}", .{ x0, self.input_layernorm.forward(x0) });
        stdx.debug.assert(x0.rank() >= 2 and x0.shape().hasTags(.{ .s, .d }), "TransformerLayer expected input shape: {{..., .s, .d}}, received: {f}", .{x0});

        const x0_normalized = self.input_layernorm.forward(x0);

        const delta0, const updated_kv_cache = self.self_attn.forward(x0_normalized, token_index, kv_cache);
        const x1 = x0.add(delta0);

        // Fully Connected
        const x1_normalized = self.post_attention_layernorm.forward(x1);
        var x2, host_buffer_out, device_buffer_out = self.mlp.forward(x1_normalized, tokens_mask, host_buffer_out, device_buffer_out);
        x2 = x2.add(x1);
        //const x2 = self.moe.forward(x1_normalized).add(x1).rename(.{ .dout = .d });

        return .{ x2.reuseBuffer(x0), updated_kv_cache, host_buffer_out, device_buffer_out };
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

    fn initProj(store: zml.io.TensorStore.View) zml.nn.Linear {
        return .init(store.createTensorWithTags("weight", .{ .dout, .d }), store.maybeCreateTensorWithTags("bias", .{.dout}), .d);
    }

    pub fn init(store: zml.io.TensorStore.View, config: GptOss.Config, layer_id: u32) !SelfAttn {
        return .{
            .q_proj = initProj(store.withPrefix("q_proj")),
            .k_proj = initProj(store.withPrefix("k_proj")),
            .v_proj = initProj(store.withPrefix("v_proj")),
            .o_proj = initProj(store.withPrefix("o_proj")),
            .sinks = store.createTensorWithTags("sinks", .{.h}),
            .sliding_window = if (layer_id % 2 == 0) config.sliding_window else null,
            // TODO(Corentin): fix that
            .q_norm = null,
            .k_norm = null,
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .freq_base = config.rope_theta,
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
        var q = self.q_proj.forward(x).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });

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

        const dtype = q.dtype();

        const new_kv_cache = kv_cache.update(k, v, token_index);
        k = new_kv_cache.keys().convert(dtype);
        v = new_kv_cache.values().convert(dtype);

        const attn_output = zml.nn.sdpa(q, k, v, .{ .attn_mask = attn_mask, .softmax_bias = self.sinks, .allow_cudnn = true });
        // const attn_output = zml.nn.sdpaMemEfficient(q, k, v, .{ .attn_mask = attn_mask }, .{ .q_chunk_size = 4096, .k_chunk_size = 1024 });
        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        return .{ self.o_proj.forward(attn), new_kv_cache };
    }
};

const RmsNorm = struct {
    weight: Tensor,
    eps: f32 = 1e-6,

    pub fn init(store: zml.io.TensorStore.View, eps: f32) RmsNorm {
        return .{ .weight = store.createTensorWithTags("weight", .{.d}), .eps = eps };
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
        const moe_on_disk: OnDisk = .{ .router = zml.nn.Linear.init(store.createTensorWithTags("router.weight", .{ .exp, .d }), store.createTensorWithTags("router.bias", .{.exp}), .d), .experts = .{
            .down_proj_bias = store.createTensorWithTags("experts.down_proj_bias", .{ .expert, .d }),
            .down_proj_blocks = store.createTensorWithTags("experts.down_proj_blocks", .{ .expert, .out, .d, .d_blocks }),
            .down_proj_scales = store.createTensorWithTags("experts.down_proj_scales", .{ .expert, .out, .d }),
            .gate_up_proj_bias = store.createTensorWithTags("experts.gate_up_proj_bias", .{ .expert, .d }),
            .gate_up_proj_blocks = store.createTensorWithTags("experts.gate_up_proj_blocks", .{ .expert, .out, .d, .d_blocks }),
            .gate_up_proj_scales = store.createTensorWithTags("experts.gate_up_proj_scales", .{ .expert, .out, .d }),
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
        _ = self; // autofix
        //to be filled
    }

    pub fn forward(self: MoE, input: Tensor, tokens_mask: ?Tensor, host_buffer_input: Tensor, device_buffer_input: Tensor) struct { Tensor, Tensor, Tensor } {
        var gpa = std.heap.GeneralPurposeAllocator(.{}){};
        defer _ = gpa.deinit();
        const allocator = gpa.allocator();
        _ = allocator; // autofix
        // log.info("input : {f}", .{input.shape()});
        //log.info("tokens_mask : {f}", .{tokens_mask.shape()});
        const dt = input.dtype();

        const gating = self.router.forward(input);

        const routing = gating.topK(.{ .top_expert = .expert }, self.moe_opts.num_experts_per_tok, .{});
        const expert_indices = routing.indices;
        var expert_scores = routing.values.softmax(.top_expert);
        var general_output = Tensor.zeroes(input.shape()); //intialized like that because same size
        const weights_gate_up = self.experts.gate_up_proj.blocks;
        const weights_gate_up_scales = self.experts.gate_up_proj.scale;
        const bias_gate_up = self.experts.gate_up_proj.bias.?;
        _ = bias_gate_up; // autofix
        const weights_down = self.experts.down_proj.blocks;
        const weights_down_scales = self.experts.down_proj.scale;
        const bias_down = self.experts.down_proj.bias.?;
        _ = bias_down; // autofix

        // log.info("Weights gate up {f}", .{weights_gate_up.shape()});
        // log.info("Weights gate up scales {f}", .{weights_gate_up_scales.shape()});
        // log.info("Bias gate up {f}", .{bias_gate_up.shape()});
        // log.info("Weights down {f}", .{weights_down.shape()});
        // log.info("Weights down scales {f}", .{weights_down_scales.shape()});
        // log.info("Bias down {f}", .{bias_down.shape()});

        const num_experts: u32 = @intCast(gating.dim(.expert));
        const hidden_size_orig: u32 = @intCast(input.dim(.d));
        const out_dim_gate_up: u32 = @intCast(weights_gate_up.dim(.out));
        const out_dim_down: u32 = @intCast(weights_down.dim(.out));

        const host_buffer = host_buffer_input;
        const device_buffer = device_buffer_input;

        // Apply the mask on expert scores
        if (tokens_mask) |mask| {
            // mask: [S], scores: [S, K] -> Broadcast
            const mask_b = mask.broad(expert_scores.shape()).convert(expert_scores.dtype());
            expert_scores = expert_scores.mul(mask_b);
        }

        if (tokens_mask) |mask| {
            var max_inf = Tensor.constant(expert_indices.dtype().maxValue());
            max_inf = max_inf.broad(expert_indices.shape());
            const expert_indices_with_max = Tensor.select(mask.broad(expert_indices.shape()), expert_indices, max_inf);
            const expert_indices_with_max_flattened = expert_indices_with_max.transpose(.{ .top_expert, .s }).reshape(.{ .seq = self.moe_opts.num_experts_per_tok * expert_indices.dim(0) });
            const input_repeated = input.repeat1d(.s, self.moe_opts.num_experts_per_tok).rename(.{ .s = .seq });
            const expert_indices_repeated = expert_indices.transpose(.{ .top_expert, .s }).flatten().withTags(.{.seq});

            const expert_scores_repeated = expert_scores.transpose(.{ .top_expert, .s }).flatten().withTags(.{.seq});

            const mask_repeated = mask.repeat1d(.s, self.moe_opts.num_experts_per_tok).rename(.{ .s = .seq });
            const counts0 = zml.Tensor.zeroes(.init(.{ .expert = num_experts }, .i32));
            const ones = zml.Tensor.constant(.{ .i32 = 1 }).broad(zml.Shape.init(.{ .seq = input.dim(.s) * self.moe_opts.num_experts_per_tok }, .i32));
            _ = ones; // autofix

            const counts_tokens_per_exp = counts0.scatterSlices(
                .{ .expert = expert_indices_repeated }, // indices: [seq]
                mask_repeated.convert(.i32), // updates: [seq]
                .{
                    .update_fn = zml.Tensor.ScatterOpts.increment,
                    .indices_are_unique = false, // plusieurs tokens peuvent aller au même expert
                },
            );

            const zero_prefix = zml.Tensor.zeroes(.init(.{ .expert = 1 }, .i64));

            var expert_first_tok_offset = counts_tokens_per_exp.convert(.i64).cumulativeSum(0);
            expert_first_tok_offset = Tensor.concatenate(&.{ zero_prefix, expert_first_tok_offset }, 0);

            const indices_sorted = expert_indices_with_max_flattened.argsort(.seq, .{ .descending = false });

            const idx = indices_sorted.rename(.{ .seq = .s });
            const input_sorted = input_repeated.gather(.{ .seq = idx }, .{});
            const gate_hidden_kernel: u32 = @intCast(weights_gate_up.dim(.d) * weights_gate_up.dim(.d_block) * 2);
            var input_sorted_kernel = input_sorted;
            if (input_sorted.dim(.d) < gate_hidden_kernel) {
                const pad_high: i64 = @as(i64, gate_hidden_kernel) - input_sorted.dim(.d);
                input_sorted_kernel = input_sorted.pad(0, .{ .d = Tensor.Pad{ .high = pad_high } });
            }

            var out_gate_up = flashinfer_moe.flashinferMoeForward(input_sorted_kernel, weights_gate_up, weights_gate_up_scales, expert_first_tok_offset, .{
                .num_experts = num_experts,
                .hidden_size = gate_hidden_kernel,
                .out_features = out_dim_gate_up,
                .output_shape = .init(.{ .s = input_sorted_kernel.dim(.s), .d = out_dim_gate_up }, .bf16),
            });

            const expert_indices_sorted = expert_indices_with_max_flattened.gather(.{ .seq = indices_sorted.rename(.{ .seq = .s }) }, .{});

            if (self.experts.gate_up_proj.bias) |bias| {
                const bias_per_token = bias.gather(.{ .expert = expert_indices_sorted }, .{});
                out_gate_up = out_gate_up.add(bias_per_token);
            }

            var up, var gate = zml.nn.splitRealImg(out_gate_up, .sequential);

            gate = .minimum(gate, .scalar(7, dt));
            up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));

            const out = gate.quickGelu().mul(up.addConstant(1));

            const down_hidden_kernel: u32 = @intCast(out.dim(.d));
            var moe_out = flashinfer_moe.flashinferMoeForward(out, self.experts.down_proj.blocks, weights_down_scales, expert_first_tok_offset, .{
                .num_experts = num_experts,
                .hidden_size = down_hidden_kernel,
                .out_features = out_dim_down,
                .output_shape = .init(.{ .s = input_sorted.dim(.s), .d = out_dim_down }, .bf16),
            });

            if (self.experts.down_proj.bias) |bias| {
                const bias_per_token = bias.gather(.{ .expert = expert_indices_sorted }, .{});
                moe_out = moe_out.add(bias_per_token);
            }

            if (moe_out.dim(.d) > hidden_size_orig) {
                moe_out = moe_out.slice1d(.d, .{ .end = hidden_size_orig });
            }

            //add bias

            const expert_scores_sorted = expert_scores_repeated.gather(.{ .seq = idx }, .{});

            moe_out = moe_out.mul(expert_scores_sorted.convert(moe_out.dtype()).broad(moe_out.shape()));

            const restore_indices = indices_sorted.argsort(.seq, .{ .descending = false });

            const top_output = moe_out.gather(.{ .s = restore_indices }, .{});
            general_output = top_output.reshape(.{ .expert = self.moe_opts.num_experts_per_tok, .s = input.dim(.s), .d = hidden_size_orig });
            general_output = general_output.sum(.expert).squeeze(0);
            // general_output = gate.slice1d(.seq, .{ .end = 1024 }).rename(.{ .seq = .s });
        } else { // Decode (no mask path)
            const expert_indices_flat = expert_indices.transpose(.{ .top_expert, .s }).flatten().withTags(.{.seq});
            const expert_scores_flat = expert_scores.transpose(.{ .top_expert, .s }).flatten().withTags(.{.seq});
            const input_repeated = input.repeat1d(.s, self.moe_opts.num_experts_per_tok).rename(.{ .s = .seq });

            const counts0 = zml.Tensor.zeroes(.init(.{ .expert = num_experts }, .i32));
            const ones = zml.Tensor.constant(.{ .i32 = 1 }).broad(expert_indices_flat.shape().withDtype(.i32));
            const counts_tokens_per_exp = counts0.scatterSlices(
                .{ .expert = expert_indices_flat },
                ones,
                .{
                    .update_fn = zml.Tensor.ScatterOpts.increment,
                    .indices_are_unique = false,
                },
            );

            const zero_prefix = zml.Tensor.zeroes(.init(.{ .expert = 1 }, .i64));
            var expert_first_tok_offset = counts_tokens_per_exp.convert(.i64).cumulativeSum(0);
            expert_first_tok_offset = Tensor.concatenate(&.{ zero_prefix, expert_first_tok_offset }, 0);

            const indices_sorted = expert_indices_flat.argsort(.seq, .{ .descending = false });
            const idx = indices_sorted.rename(.{ .seq = .s });
            const input_sorted = input_repeated.gather(.{ .seq = idx }, .{});

            const gate_hidden_kernel: u32 = @intCast(weights_gate_up.dim(.d) * weights_gate_up.dim(.d_block) * 2);
            var input_sorted_kernel = input_sorted;
            if (input_sorted.dim(.d) < gate_hidden_kernel) {
                const pad_high: i64 = @as(i64, gate_hidden_kernel) - input_sorted.dim(.d);
                input_sorted_kernel = input_sorted.pad(0, .{ .d = Tensor.Pad{ .high = pad_high } });
            }

            var out_gate_up = flashinfer_moe.flashinferMoeForward(input_sorted_kernel, weights_gate_up, weights_gate_up_scales, expert_first_tok_offset, .{
                .num_experts = num_experts,
                .hidden_size = gate_hidden_kernel,
                .out_features = out_dim_gate_up,
                .output_shape = .init(.{ .s = input_sorted_kernel.dim(.s), .d = out_dim_gate_up }, .bf16),
            });

            const expert_indices_sorted = expert_indices_flat.gather(.{ .seq = idx }, .{});
            if (self.experts.gate_up_proj.bias) |bias| {
                const bias_per_token = bias.gather(.{ .expert = expert_indices_sorted }, .{});
                out_gate_up = out_gate_up.add(bias_per_token);
            }

            var up, var gate = zml.nn.splitRealImg(out_gate_up, .sequential);
            gate = .minimum(gate, .scalar(7, dt));
            up = .clamp(up, .scalar(-7, dt), .scalar(7, dt));
            const out = gate.quickGelu().mul(up.addConstant(1));

            const down_hidden_kernel: u32 = @intCast(out.dim(.d));
            var moe_out = flashinfer_moe.flashinferMoeForward(out, self.experts.down_proj.blocks, weights_down_scales, expert_first_tok_offset, .{
                .num_experts = num_experts,
                .hidden_size = down_hidden_kernel,
                .out_features = out_dim_down,
                .output_shape = .init(.{ .s = input_sorted.dim(.s), .d = out_dim_down }, .bf16),
            });
            if (self.experts.down_proj.bias) |bias| {
                const bias_per_token = bias.gather(.{ .expert = expert_indices_sorted }, .{});
                moe_out = moe_out.add(bias_per_token);
            }
            if (moe_out.dim(.d) > hidden_size_orig) {
                moe_out = moe_out.slice1d(.d, .{ .end = hidden_size_orig });
            }

            const expert_scores_sorted = expert_scores_flat.gather(.{ .seq = idx }, .{});
            moe_out = moe_out.mul(expert_scores_sorted.convert(moe_out.dtype()).broad(moe_out.shape()));

            const restore_indices = indices_sorted.argsort(.seq, .{ .descending = false });
            const top_output = moe_out.gather(.{ .s = restore_indices }, .{});
            general_output = top_output.reshape(.{ .expert = self.moe_opts.num_experts_per_tok, .s = input.dim(.s), .d = hidden_size_orig });
            general_output = general_output.sum(.expert).squeeze(0);
        }
        return .{ general_output, host_buffer, device_buffer };
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
            gate_blocks._shape = gate_blocks._shape.setDim(1, gate_out_padded).setDim(2, gate_d_padded);
            var gate_scale = e.gate_up_proj_scales.withTags(.{ .expert, .out, .d });
            gate_scale._shape = gate_scale._shape.setDim(1, gate_out_padded).setDim(2, gate_d_padded);
            var gate_bias = e.gate_up_proj_bias.withTags(.{ .expert, .d });
            gate_bias._shape = gate_bias._shape.setDim(1, gate_out_padded);

            var down_blocks = e.down_proj_blocks.withTags(.{ .expert, .out, .d, .d_block });
            down_blocks._shape = down_blocks._shape.setDim(1, down_out_padded).setDim(2, down_d_padded);
            var down_scale = e.down_proj_scales.withTags(.{ .expert, .out, .d });
            down_scale._shape = down_scale._shape.setDim(1, down_out_padded).setDim(2, down_d_padded);
            var down_bias = e.down_proj_bias.withTags(.{ .expert, .d });
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

                .router = zml.nn.Linear.init(on_disk.router.weight.withTags(.{ .expert, .d }), on_disk.router.bias.?.withTags(.{.expert}), .d),

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

/// We have three algorithms,
/// * one for single-stream inference (naive),
/// * one for small batch sized with exact precision that sends all tokens to all experts.
///   this isn't too costly as long as the batch size is small and the experts are IO bound.
/// * one for big batch size that assign a fixed compute budget per expert and
/// experts chose the tokens they want to handle. This introduces noise since it's possible
/// a token doesn't get their requested expert.
///   The parameter `tokens_per_expert_ratio` control how much compute budget is granted:
///   expert_budget = ratio * (num_tokens * num_experts_per_tok / num_experts).
///   Bigger values of ratio will ensure it's rare a token doesn't get it's top 2 tokens.
///
/// The preferred algorithm is the batched one,
/// it is selected as soon there is enough tokens to guarantee that experts will be active most of the time.
///
/// - input: .{ .s, .d } per-entry vector
/// - gating: .{ .s, .expert } per-entry expert-affinity
/// - experts: .{ .expert, .d_out, .d } expert layer (need to have a .forward method).
/// -> output: .{ .s, .d_out }
pub fn mixtureOfExperts(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    log.warn("mixtureOfExperts({s}, {f}, {f}, {})", .{ @typeName(Expert), input, gating, opts });
    const num_tokens: u32 = @intCast(input.dim(.s));
    const num_experts = gating.dim(.expert);
    stdx.debug.assert(opts.num_experts_per_tok > 0, "mixtureOfExperts expects opts.num_experts_per_tok > 0, got {}", .{opts});

    if (num_tokens == 1) {
        return moePerTokenRouting(Expert, experts, input, gating, opts);
    }

    const tokens_per_expert: u32 = if (opts.tokens_per_expert_ratio) |ratio| tpe: {
        const compute_budget = ratio * @as(f32, @floatFromInt(num_tokens * opts.num_experts_per_tok));
        var tpe: u32 = @intFromFloat(stdx.math.divFloat(f32, compute_budget, num_experts));
        // Round to next multiple of 8 to avoid weird shapes.
        if (tpe % 8 != 0) tpe += 8 - (tpe % 8);
        break :tpe tpe;
    } else num_tokens;

    if (3 * tokens_per_expert <= 2 * num_tokens) {
        const routing, const tokens_ids_per_expert = dispatchTokens(gating, .{
            .tokens_per_expert = tokens_per_expert,
            .num_experts_per_tok = opts.num_experts_per_tok,
            .normalization = opts.normalization,
        });

        const scores_per_expert = routing.transpose(.{ .expert, .s }).gather(.{ .s = tokens_ids_per_expert }, .{});
        const input_per_expert = input.gather(.{ .s = tokens_ids_per_expert }, .{});
        var output_per_expert = experts.forward(input_per_expert);
        output_per_expert = output_per_expert.mul(scores_per_expert.convert(output_per_expert.dtype()).broad(output_per_expert.shape()));

        // Reverse engineer the normal output shape that one expert would have produced for all tokens.
        // If this fall short, we could use the "sliced_expert" strategy and call forward ourselves.
        const output_shape = output_per_expert.shape().drop(.expert).rename(.{ .top_token = .s }).setDim(.s, num_tokens);
        const output = zml.Tensor.scatterSlices(
            .zeroes(output_shape),
            .{ .s = tokens_ids_per_expert },
            output_per_expert,
            .{ .update_fn = zml.Tensor.ScatterOpts.increment },
        );

        log.warn("mixtureOfExperts({s}, {f}, {f}) -> fixed budget impl tpe: {d}, tokens: {d}", .{ @typeName(Expert), input, gating, tokens_per_expert, num_tokens });
        return output;
    } else {
        return mixtureOfExpertsAllToAll(Expert, experts, input, gating, opts);
    }
}

/// Few tokens: most experts are unused, experts have at most one token.
/// Select active experts and compute with that.
pub fn moePerTokenRouting(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const num_tokens: u32 = @intCast(input.dim(.s));
    stdx.debug.assert(num_tokens < 32, "Trying to unroll a lot of tokens !", .{});
    const per_token_outputs = zml.module.CompilationContext.current().allocator.alloc(Tensor, num_tokens) catch @panic("OOM");

    const routing = gating.topK(.{ .top_expert = .expert }, opts.num_experts_per_tok, .{});
    log.info("routing shape : {f}", .{routing.indices.shape()});
    const per_token_score = switch (opts.normalization) {
        .linear => routing.values.div(routing.values.sum(.top_expert)),
        .softmax => routing.values.softmax(.top_expert),
    };

    for (per_token_outputs, 0..num_tokens) |*output, tok_id| {
        for (0..opts.num_experts_per_tok) |expert_rank| {
            const expert_id = routing.indices.choose(.{ .s = tok_id, .top_expert = expert_rank }).asScalar();
            const expert_score = per_token_score.choose(.{ .s = tok_id, .top_expert = expert_rank }).asScalar();

            var sliced_expert: Expert = undefined;
            zml.meta.mapAlloc(struct {
                pub fn cb(expert_id_: zml.Tensor, expert_weight: zml.Tensor) zml.Tensor {
                    return expert_weight.gather(.{ .expert = expert_id_ }, .{});
                }
            }.cb, stdx.noalloc, expert_id, experts, &sliced_expert) catch unreachable;

            // TODO how does this work when the two experts are on different gpus?
            // does the compute overlap ?
            var expert_output = sliced_expert.forward(input.choose(.{ .s = tok_id }));
            expert_output = .mul(
                expert_output,
                expert_score.convert(input.dtype()).broad(expert_output.shape()),
            );
            output.* = if (expert_rank > 0) output.add(expert_output) else expert_output;
        }
    }

    log.warn("mixtureOfExperts({s}, {f}, {f}) -> single-stream impl", .{ @typeName(Expert), input, gating });
    return .stack(per_token_outputs, 0, .s);
}

/// Send all tokens to all experts, and apply gating.
pub fn mixtureOfExpertsAllToAll(Expert: type, experts: Expert, input: zml.Tensor, gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    log.warn("mixtureOfExperts({s}, {f}, {f}) -> all to all impl", .{ @typeName(Expert), input, gating });
    const num_experts = gating.dim(.expert);
    const hard_gating = hardGating(gating, opts);
    // TODO: `input.insertAxes(0, .{.expert}).repeat1d(.expert, num_experts)` is too verbose for just broadcasting along a new axis`
    const output_per_expert = experts.forward(input.insertAxes(0, .{.expert}).repeat1d(.expert, @intCast(num_experts)));
    return output_per_expert.dot(hard_gating.convert(input.dtype()), .expert);
}

/// Given `(token, expert) -> scores`,
/// keeps only the top-k expert per token, and normalize the scores accordingly.
/// Non selected experts will have a 0 score.
pub fn hardGating(gating: zml.Tensor, opts: MoeOpts) zml.Tensor {
    const routing = gating.topK(.{ .top_expert = .expert }, opts.num_experts_per_tok, .{});

    const per_token_score = switch (opts.normalization) {
        .linear => routing.values.div(routing.values.sum(.top_expert)),
        .softmax => routing.values.softmax(.top_expert),
    };

    return zml.Tensor.scatterSlices(
        .zeroes(gating.shape()),
        .{ .expert = routing.indices },
        per_token_score,
        .{ .indices_are_unique = true },
    );
}

/// Lot of tokens, each experts chose their tokens.
/// It means that some tokens may have only one expert assigned.
/// Each token will get assigned to at least one expert IIF the input gating is sums up to 1 (typically softmax output).
/// Returns the actual `(token, expert) -> scores` used.
pub fn dispatchTokens(
    gating: zml.Tensor,
    opts: struct {
        tokens_per_expert: u32,
        num_experts_per_tok: u32,
        normalization: MoeOpts.Normalization,
    },
) [2]zml.Tensor {
    const num_experts = gating.dim(.expert);

    const token_pref = gating.argsort(.expert, .{ .descending = true });
    var expert_rank: zml.Tensor = .scatterSlices(
        .zeroes(gating.shape().withDtype(.i32)),
        .{ .expert = token_pref },
        .addConstant(.iota(gating.shape(), .expert), 1),
        .{ .indices_are_unique = true },
    );
    // The pow(expert_rank) here means that we strongly favor top 1 over top 2 and top 2 over top 3.
    // expert_routing: (expert, top_token) -> token
    const expert_routing = gating.pow(expert_rank.convert(gating.dtype())).topK(.{ .top_token = .s }, opts.tokens_per_expert, .{});
    const scores_per_expert = gating.gather(.{ .s = expert_routing.indices }, .{});

    // Update the gating coefficient to account for the expert routing.
    // Each (token, expert) which can't be computed within the given budget is left to 0.
    const gating_v2: zml.Tensor = .scatterSlices(
        .zeroes(gating.shape()),
        .{ .s = expert_routing.indices },
        scores_per_expert,
        .{ .indices_are_unique = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
    // Now set to zero the scores (token, expert) for tokens that have been assigned more than num_experts_per_tok.
    const lowest_experts = gating_v2.topK(.{ .top_expert = .expert }, @intCast(num_experts - opts.num_experts_per_tok), .{ .descending = false });
    var gating_v3: zml.Tensor = .scatterSlices(
        gating_v2,
        .{ .expert = lowest_experts.indices },
        .zeroes(lowest_experts.values.shape()),
        .{ .indices_are_unique = true, .update_fn = zml.Tensor.ScatterOpts.override },
    );
    // Then normalize so the sum of experts scores for one token sums up to 1.
    gating_v3 = switch (opts.normalization) {
        .linear => gating_v3.div(gating_v3.sum(.expert)),
        .softmax => gating_v3.softmax(.expert),
    };
    const tokens_ids_per_expert = expert_routing.indices.transpose(.{ .expert, .top_token });

    return .{ gating_v3, tokens_ids_per_expert };
}

pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
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

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try zml.Buffer.uninitialized(io, platform, self.k.shape(), .{}),
            .v = try zml.Buffer.uninitialized(io, platform, self.v.shape(), .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
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
