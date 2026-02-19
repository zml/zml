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
    zml.meta.visit((struct {
        fn cb(_: void, buf: *zml.Buffer) void {
            buf.deinit();
        }
    }).cb, {}, bufferized);
}

pub fn linear(store: zml.io.TensorStore.View) zml.nn.Linear {
    return .init(
        store.createTensorWithTags("weight", .{ .dout, .d }),
        store.maybeCreateTensorWithTags("bias", .{.dout}),
        .d,
    );
}

pub fn rmsNorm(x: Tensor, weight: Tensor, eps: f32) Tensor {
    return zml.nn.rmsNorm(x, .d, eps).mul(weight.broad(x.shape()));
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
    var total_bytes: usize = 0;
    defer {
        const took = now.untilNow(io, .awake);
        log.info("Loaded " ++ label ++ " weights [{Bi:.2}, {D}, {Bi:.2}/s]", .{
            total_bytes,
            stdx.fmt.fmtDuration(took),
            total_bytes / @as(usize, @intCast(took.toNanoseconds())) * std.time.ns_per_s,
        });
    }

    return zml.io.load(T, self, allocator, io, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 128 * zml.MiB,
        .progress = progress,
        .store = store,
        .parallelism = 16,
        .total_bytes = &total_bytes,
    });
}

/// Unified self-attention with KV cache.
/// When `shift_buffer` is true, the KV cache uses shift-based overflow handling for
/// unbounded sliding window support. Both paths (normal write vs shifted cache write)
/// are computed and the correct one is selected via cmp/select. The cache stays in
/// temporal order — no reordering needed.
/// When false, the cache uses sequential indexing (matching the reference implementation).
pub fn SelfAttention(comptime shift_buffer: bool) type {
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
                .freq_base = attn_config.rope_theta,
            };

            q = zml.nn.rope(q, pos_index, rope_opts);
            k = zml.nn.rope(k, pos_index, rope_opts);

            q = q.rename(.{ .s = .q });
            k = k.rename(.{ .s = .k });
            v = v.rename(.{ .s = .k });

            if (shift_buffer) {
                const shifted_cache, const clamped_token_index = kv_cache.shiftIfNeeded(token_index, @intCast(x.dim(.s)));

                const write_pos = b: {
                    const temp = Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype())
                        .withTags(.{.s}).broad(Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
                    break :b temp.add(clamped_token_index.broad(temp.shape()));
                };

                const new_kv_cache = shifted_cache.update(k, v, write_pos.rename(.{ .s = .k }));
                k = new_kv_cache.keys().convert(dtype);
                v = new_kv_cache.values().convert(dtype);

                const attn_out = zml.attention.attention(q, k, v, clamped_token_index, attention_metadata, attention_parameters, .{
                    .sliding_window = @intCast(attn_config.sliding_window),
                });

                const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
                return .{ self.wo.forward(merged).rename(.{ .dout = .d }), new_kv_cache };
            } else {
                // Sequential: write at pos_index directly (matching reference implementation).
                const new_kv_cache = kv_cache.update(k, v, pos_index.rename(.{ .s = .k }));
                k = new_kv_cache.keys().convert(dtype);
                v = new_kv_cache.values().convert(dtype);

                const attn_token_index = token_index;
                const attn_out = zml.attention.attention(q, k, v, attn_token_index, attention_metadata, attention_parameters, .{
                    .sliding_window = @intCast(attn_config.sliding_window),
                });

                const merged = attn_out.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
                return .{ self.wo.forward(merged).rename(.{ .dout = .d }), new_kv_cache };
            }
        }
    };
}

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
            .k = try .uninitialized(io, platform, self.k.shape(), .{}),
            .v = try .uninitialized(io, platform, self.v.shape(), .{}),
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
        return .{
            .k = scatterCache(self.k, new_k, self.layer_index, token_index),
            .v = scatterCache(self.v, new_v, self.layer_index, token_index),
            .layer_index = self.layer_index,
        };
    }

    fn scatterCache(cache: Tensor, new: Tensor, layer_index: Tensor, token_index: ?Tensor) Tensor {
        const k_shape = cache.shape().drop(.layer);
        const converted = new.convert(cache.dtype()).transpose(k_shape);
        const scatter_opts: Tensor.ScatterOpts = .{ .indices_are_sorted = true, .update_fn = Tensor.ScatterOpts.override };

        return if (token_index) |idx|
            cache.scatterSlices(.{ .layer = layer_index.broad(idx.shape()), .k = idx }, converted, scatter_opts).reuseBuffer(cache)
        else
            cache.scatterSlices(.{ .layer = layer_index }, converted, scatter_opts).reuseBuffer(cache);
    }

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
