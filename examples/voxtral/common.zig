const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;
const Tensor = zml.Tensor;
const Shape = zml.Shape;

const cfg = @import("config.zig");

/// Recursively deinit all Buffer fields in a Bufferized struct.
/// For models with allocated slices (e.g. layers), the caller must also free the slice.
pub fn deinitBufferized(bufferized: anytype) void {
    zml.meta.forEachVisit(bufferized, *zml.Buffer, (struct {
        fn cb(_: usize, buf: *zml.Buffer) void {
            buf.deinit();
        }
    }).cb, .{});
}

pub fn linear(store: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        store.createTensor("weight", .{ .dout, .d }, .replicated),
        store.maybeCreateTensor("bias", .{.dout}, .replicated),
        .d,
    );
}

pub fn rmsNorm(x: Tensor, weight: Tensor, eps: f32) Tensor {
    return zml.nn.rmsNorm(x, .d, eps).mul(weight.broad(x.shape()));
}

pub fn attention(q: Tensor, k: Tensor, v: Tensor, token_index: Tensor, metadata: zml.attention.Metadata, parameters: zml.attention.Parameters, window: u32) Tensor {
    // Chunked circular caches retain window + chunk - 1 entries. Mask the
    // extra history per query, including when a chunk crosses the ring boundary.
    if (k.dim(.k) > window or (parameters == .metal_fa and q.dim(.q) > 1)) {
        const positions = Tensor.arange(.{ .end = q.dim(.q) }, .i32).withTags(.{.q}).add(token_index.convert(.i32));
        const keys = Tensor.arange(.{ .end = k.dim(.k) }, .i32).withTags(.{.k});
        const mask_shape = Shape.init(.{ .q = q.dim(.q), .k = k.dim(.k) }, .i32);
        const qp = positions.broad(mask_shape);
        const kp = keys.broad(mask_shape);
        const valid = kp.cmp(.LE, qp).select(kp.cmp(.GT, qp.addConstant(-@as(i64, window))), Tensor.scalar(false, .bool).broad(mask_shape.withDtype(.bool)));
        const mask = valid.select(Tensor.scalar(0, q.dtype()).broad(mask_shape.withDtype(q.dtype())), Tensor.scalar(-std.math.inf(f32), q.dtype()).broad(mask_shape.withDtype(q.dtype())));
        // Unwritten cache entries may contain NaNs; a -inf attention mask alone
        // cannot remove those (nor can zero probabilities multiplied by NaN V).
        const populated = keys.cmp(.LE, token_index.convert(.i32).addConstant(q.dim(.q) - 1).broad(keys.shape()));
        const safe_k = populated.broad(k.shape().withDtype(.bool)).select(k, Tensor.scalar(0, k.dtype()).broad(k.shape()));
        const safe_v = populated.broad(v.shape().withDtype(.bool)).select(v, Tensor.scalar(0, v.dtype()).broad(v.shape()));
        return zml.nn.sdpa(q, safe_k, safe_v, .{ .attn_mask = mask });
    }
    var md = metadata;
    if (md == .metal_fa) md.metal_fa.num_tokens = Tensor.scalar(q.dim(.q), .u32);
    return zml.attention.attention(q, k, v, token_index, md, parameters);
}

pub fn loadModel(
    comptime T: type,
    self: *const T,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    comptime label: []const u8,
) !zml.Bufferized(T) {
    progress.increaseEstimatedTotalItems(store.view().count());

    const now: std.Io.Timestamp = .now(io, .awake);
    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .dma_chunks = 4,
        .dma_chunk_size = 16 * zml.MiB,
        .parallelism = 4,
    });
    defer loader.deinit();
    defer {
        const took = now.untilNow(io, .awake);
        log.info("Loaded " ++ label ++ " weights [{Bi:.2}, {f}, {d:.0} bytes/s]", .{
            loader.bytes_loaded.raw,
            took,
            @as(f64, @floatFromInt(loader.bytes_loaded.raw)) * std.time.ns_per_s / @as(f64, @floatFromInt(@max(1, took.nanoseconds))),
        });
    }

    var buffers = try zml.mem.bufferize(allocator, T, self);
    try loader.load(io, T, self, &buffers, store, &.{}, .{ .progress = progress });
    try loader.await(io);
    return buffers;
}

/// Unified self-attention with KV cache.
/// When `circular_buffer` is true, the KV cache uses a circular (ring) buffer for unbounded
/// sliding window support: keys/values are written at position `pos_index % cache_size`,
/// then reordered into temporal order before attention.
/// When false, the cache uses sequential indexing.
pub fn SelfAttention(comptime circular_buffer: bool) type {
    return struct {
        wq: zml.nn.Linear,
        wk: zml.nn.Linear,
        wv: zml.nn.Linear,
        wo: zml.nn.Linear,

        const Self = @This();

        pub fn init(store: zml.io.TensorStore.View) Self {
            return .{
                .wq = linear(store.withPrefix("wq")),
                .wk = linear(store.withPrefix("wk")),
                .wv = linear(store.withPrefix("wv")),
                .wo = linear(store.withPrefix("wo")),
            };
        }

        pub fn unload(self: *zml.Bufferized(Self)) void {
            deinitBufferized(self);
        }

        /// x: [s, d], token_index: scalar, kv_cache: KvCache → ([s, d], KvCache)
        pub fn forward(self: Self, x: Tensor, token_index: Tensor, kv_cache: KvCache, attn_config: cfg.AttentionConfig, attention_metadata: zml.attention.Metadata, attention_parameters: zml.attention.Parameters) struct { Tensor, KvCache } {
            const dtype = x.dtype();

            var q = self.wq.forward(x);
            var k = self.wk.forward(x);
            var v = self.wv.forward(x);

            q = q.splitAxis(.dout, .{ .h = attn_config.n_heads, .hd = attn_config.head_dim });
            k = k.splitAxis(.dout, .{ .h = attn_config.n_kv_heads, .hd = attn_config.head_dim });
            v = v.splitAxis(.dout, .{ .h = attn_config.n_kv_heads, .hd = attn_config.head_dim });

            // Position indices for RoPE: token_index + arange(s)
            const pos_index = b: {
                const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype())
                    .withTags(.{.s}).broad(Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
                break :b temp.add(token_index.broad(temp.shape()));
            };

            const rope_opts: zml.nn.RopeOpts = .{
                .layout = .interleaved,
                .scaling = .{ .default = .{ .rope_theta = attn_config.rope_theta } },
            };

            q = zml.nn.rope(q, pos_index, rope_opts);
            k = zml.nn.rope(k, pos_index, rope_opts);

            q = q.rename(.{ .s = .q });
            k = k.rename(.{ .s = .k });
            v = v.rename(.{ .s = .k });

            const cache_size = Tensor.scalar(@as(u32, @intCast(kv_cache.k.dim(.k))), token_index.dtype());

            if (circular_buffer) {
                const cache_pos = pos_index.remainder(cache_size.broad(pos_index.shape()));
                const new_kv_cache = kv_cache.update(k, v, cache_pos.rename(.{ .s = .k }));
                k = new_kv_cache.keys().convert(dtype);
                v = new_kv_cache.values().convert(dtype);

                // Reorder K/V from circular buffer to temporal order for correct causal masking.
                // When the cache wraps, physical order != temporal order, which breaks FA's
                // position-based causal mask.
                const pos_end = token_index.addConstant(x.dim(.s));
                const is_full = pos_end.cmp(.GE, cache_size);
                const rotation_start = pos_end.remainder(cache_size);
                const safe_start = is_full.select(rotation_start, Tensor.scalar(@as(u32, 0), token_index.dtype()));
                const reorder_arange = Tensor.arange(.{ .end = kv_cache.k.dim(.k) }, token_index.dtype()).withTags(.{.kk});
                const reorder_idx = reorder_arange.add(safe_start.broad(reorder_arange.shape()))
                    .remainder(cache_size.broad(reorder_arange.shape()));
                k = k.gather(.{ .k = reorder_idx }, .{}).rename(.{ .kk = .k });
                v = v.gather(.{ .k = reorder_idx }, .{}).rename(.{ .kk = .k });

                // Cap token_index so seqused_k doesn't exceed cache bounds
                const max_token_index = Tensor.scalar(@as(u32, @intCast(kv_cache.k.dim(.k) - x.dim(.s))), token_index.dtype());
                const attn_token_index = token_index.minimum(max_token_index);
                const attn_out = attention(q, k, v, attn_token_index, attention_metadata, attention_parameters, attn_config.sliding_window);

                const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });

                return .{ self.wo.forward(merged).rename(.{ .dout = .d }), new_kv_cache };
            } else {
                // Sequential: write at pos_index directly.
                const new_kv_cache = kv_cache.update(k, v, pos_index.rename(.{ .s = .k }));
                k = new_kv_cache.keys().convert(dtype);
                v = new_kv_cache.values().convert(dtype);

                const attn_token_index = token_index;
                const attn_out = attention(q, k, v, attn_token_index, attention_metadata, attention_parameters, attn_config.sliding_window);

                const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
                return .{ self.wo.forward(merged).rename(.{ .dout = .d }), new_kv_cache };
            }
        }
    };
}

pub const SwiGluFfn = struct {
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,
    w3: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) SwiGluFfn {
        return .{
            .w1 = linear(store.withPrefix("w1")),
            .w2 = linear(store.withPrefix("w2")),
            .w3 = linear(store.withPrefix("w3")),
        };
    }

    pub fn unload(self: *zml.Bufferized(SwiGluFfn)) void {
        deinitBufferized(self);
    }

    /// x: [s, d] -> [s, d]
    pub fn forward(self: SwiGluFfn, x: Tensor) Tensor {
        const gate = self.w1.forward(x).silu();
        const up = self.w3.forward(x);

        return self.w2.forward(gate.mul(up).rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

/// KV cache for all layers of a transformer.
/// Stores K/V tensors with shape {layer, k=max_seq_len, h, hd}.
pub const KvCache = struct {
    k: Tensor,
    v: Tensor,
    layer_index: Tensor,

    pub fn init(kv_shape: Shape) KvCache {
        return .{
            .k = .fromShape(kv_shape),
            .v = .fromShape(kv_shape),
            .layer_index = .init(.{}, .u32),
        };
    }

    pub fn initShape(kv_shape: Shape) zml.ShapeOf(KvCache) {
        return .{
            .k = kv_shape,
            .v = kv_shape,
            .layer_index = Shape.init(.{}, .u32),
        };
    }

    pub fn initBuffer(self: KvCache, io: std.Io, platform: *const zml.Platform) !zml.Bufferized(KvCache) {
        return .{
            .k = try .uninitialized(io, platform, self.k.shape(), .replicated, .{}),
            .v = try .uninitialized(io, platform, self.v.shape(), .replicated, .{}),
            .layer_index = try zml.Buffer.scalar(io, platform, 0, .u32),
        };
    }

    pub fn deinitBuffer(self: *zml.Bufferized(KvCache)) void {
        self.k.deinit();
        self.v.deinit();
        self.layer_index.deinit();
    }

    pub fn keys(self: KvCache) Tensor {
        return self.k.slice(.layer, .dynSingle(self.layer_index));
    }

    pub fn values(self: KvCache) Tensor {
        return self.v.slice(.layer, .dynSingle(self.layer_index));
    }

    pub fn update(self: KvCache, new_k: Tensor, new_v: Tensor, token_index: ?Tensor) KvCache {
        return .{
            .k = scatterCache(self.k, new_k, self.layer_index, token_index),
            .v = scatterCache(self.v, new_v, self.layer_index, token_index),
            .layer_index = self.layer_index,
        };
    }

    fn scatterCache(cache: Tensor, new: Tensor, layer_index: Tensor, token_index: ?Tensor) Tensor {
        const k_shape = cache.shape().drop(.layer);
        const converted = new.convert(cache.dtype()).transpose(k_shape);
        // Ring-buffer positions are not sorted when a chunk wraps around.
        const scatter_opts: Tensor.ScatterOpts = .{ .update_fn = Tensor.ScatterOpts.override };

        return if (token_index) |idx|
            cache.scatterSlices(.{ .layer = layer_index.broad(idx.shape()), .k = idx }, converted, scatter_opts).reuseBuffer(cache)
        else
            cache.scatterSlices(.{ .layer = layer_index }, converted, scatter_opts).reuseBuffer(cache);
    }

    // Unused but works for shifted kv cache
    pub fn shiftIfNeeded(self: KvCache, token_index: Tensor, seq_len: u32) struct { KvCache, Tensor } {
        const cache_k_size = self.k.dim(.k);
        const max_pos = Tensor.scalar(@as(u32, @intCast(cache_k_size - seq_len)), token_index.dtype());
        const would_overflow = token_index.cmp(.GT, max_pos);

        const shift_arange = Tensor.arange(.{ .end = cache_k_size }, token_index.dtype()).withTags(.{.kk});
        const shifted_indices = shift_arange.addConstant(seq_len);
        const max_idx = Tensor.scalar(@as(u32, @intCast(cache_k_size - 1)), token_index.dtype());
        const clamped_indices = shifted_indices.minimum(max_idx.broad(shifted_indices.shape()));

        const shifted_k = self.k.gather(.{ .k = clamped_indices }, .{}).rename(.{ .kk = .k });
        const shifted_v = self.v.gather(.{ .k = clamped_indices }, .{}).rename(.{ .kk = .k });

        const overflow_k = would_overflow.broad(shifted_k.shape());
        const new_k = overflow_k.select(shifted_k, self.k);
        const overflow_v = would_overflow.broad(shifted_v.shape());
        const new_v = overflow_v.select(shifted_v, self.v);

        return .{ .{
            .k = new_k.reuseBuffer(self.k),
            .v = new_v.reuseBuffer(self.v),
            .layer_index = self.layer_index,
        }, token_index.minimum(max_pos) };
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
