const std = @import("std");

const zml = @import("zml");

const ops = @import("ops.zig");
const vision_sdpa = @import("vision_sdpa.zig");

const log = std.log.scoped(.minimax_h3_vision);

pub const VISION_START: u32 = 151652;
pub const VISION_END: u32 = 151653;
pub const IMAGE_PAD: u32 = 151655;
pub const VIDEO_PAD: u32 = 151656;

pub const Config = struct {
    depth: i64 = 27,
    hidden_size: i64 = 1152,
    intermediate_size: i64 = 4304,
    num_heads: i64 = 16,
    patch_size: i64 = 16,
    temporal_patch_size: i64 = 2,
    spatial_merge_size: i64 = 2,
    out_hidden_size: i64 = 5120,
    num_position_embeddings: i64 = 2304,
    deepstack_visual_indexes: [3]i64 = .{ 8, 16, 24 },

    pub fn headDim(self: Config) i64 {
        return @divExact(self.hidden_size, self.num_heads);
    }

    pub fn patchIn(self: Config) i64 {
        return 3 * self.temporal_patch_size * self.patch_size * self.patch_size;
    }

    pub fn mergeUnit(self: Config) i64 {
        return self.spatial_merge_size * self.spatial_merge_size;
    }
};

const FileConfig = struct {
    vision_config: ?struct {
        depth: ?i64 = null,
        hidden_size: ?i64 = null,
        intermediate_size: ?i64 = null,
        num_heads: ?i64 = null,
        patch_size: ?i64 = null,
        temporal_patch_size: ?i64 = null,
        spatial_merge_size: ?i64 = null,
        out_hidden_size: ?i64 = null,
        num_position_embeddings: ?i64 = null,
        deepstack_visual_indexes: ?[]const i64 = null,
    } = null,

    fn resolve(self: FileConfig, text_hidden: i64) Config {
        var out = Config{};
        out.out_hidden_size = text_hidden;
        if (self.vision_config) |v| {
            if (v.depth) |d| out.depth = d;
            if (v.hidden_size) |d| out.hidden_size = d;
            if (v.intermediate_size) |d| out.intermediate_size = d;
            if (v.num_heads) |d| out.num_heads = d;
            if (v.patch_size) |d| out.patch_size = d;
            if (v.temporal_patch_size) |d| out.temporal_patch_size = d;
            if (v.spatial_merge_size) |d| out.spatial_merge_size = d;
            if (v.out_hidden_size) |d| out.out_hidden_size = d;
            if (v.num_position_embeddings) |d| out.num_position_embeddings = d;
            if (v.deepstack_visual_indexes) |idx| {
                for (0..@min(idx.len, out.deepstack_visual_indexes.len)) |i| out.deepstack_visual_indexes[i] = idx[i];
            }
        }
        return out;
    }
};

fn visionView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    return store.withPrefix("model.visual");
}

fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8) zml.nn.Linear {
    return ops.linear(store, weight_name, bias_name, .replicated, .replicated);
}

const layerNorm = ops.ln;

fn conv3dLinear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8) zml.nn.Linear {
    return .init(
        store.createTensor(weight_name, .{ .dout, .d, .kt, .kh, .kw }, .replicated),
        if (bias_name) |n| store.maybeCreateTensor(n, .{.dout}, .replicated) else null,
        .d,
    );
}

fn asLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    var out = lin;
    if (out.weight.rank() == 5) {
        out.weight = out.weight.merge(.{ .d = .{ .d, .kt, .kh, .kw } });
    }
    out.weight = out.weight.withTags(.{ .dout, .d });
    return out.forward(x.convert(out.weight.dtype()));
}

fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    return zml.nn.applyRotary(x.convert(.f32), cos.convert(.f32), sin.convert(.f32)).convert(x.dtype());
}

fn visionAttn(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, head_dim: i64) zml.Tensor {
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    const target = zml.Compiler.current().platform.target;
    return if (vision_sdpa.supports(target, q.dtype(), head_dim))
        vision_sdpa.forward(q, k, v, scale)
    else
        zml.nn.sdpa(q, k, v, .{});
}

pub fn register(platform: *const zml.Platform) !void {
    try vision_sdpa.register(platform);
}

pub const VisionBlock = struct {
    norm1: zml.nn.LayerNorm,
    qkv: zml.nn.Linear,
    proj: zml.nn.Linear,
    norm2: zml.nn.LayerNorm,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    num_heads: i64,
    head_dim: i64,

    pub const Input = struct {
        layer: VisionBlock,
        hidden: zml.Tensor,
        cos: zml.Tensor,
        sin: zml.Tensor,
    };
    pub const Output = struct { hidden: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) VisionBlock {
        const attn = store.withPrefix("attn");
        const mlp = store.withPrefix("mlp");
        return .{
            .norm1 = layerNorm(store.withPrefix("norm1"), 1e-6),
            .qkv = linear(attn, "qkv.weight", "qkv.bias"),
            .proj = linear(attn, "proj.weight", "proj.bias"),
            .norm2 = layerNorm(store.withPrefix("norm2"), 1e-6),
            .fc1 = linear(mlp, "linear_fc1.weight", "linear_fc1.bias"),
            .fc2 = linear(mlp, "linear_fc2.weight", "linear_fc2.bias"),
            .num_heads = cfg.num_heads,
            .head_dim = cfg.headDim(),
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.layer;
        const residual = input.hidden.withPartialTags(.{ .b, .s, .d });
        var qkv = asLinear(self.qkv, self.norm1.forward(residual));
        const parts = qkv.chunkExact(.dout, 3);
        var q = parts[0].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
        var k = parts[1].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
        const v = parts[2].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim });
        q = applyRotary(q, input.cos, input.sin);
        k = applyRotary(k, input.cos, input.sin);
        const q_s = q.rename(.{ .s = .q });
        const k_s = k.rename(.{ .s = .k });
        const v_s = v.rename(.{ .s = .k });
        const attn = visionAttn(q_s, k_s, v_s, self.head_dim).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        const x1 = residual.add(asLinear(self.proj, attn).rename(.{ .dout = .d }));
        const h = asLinear(self.fc1, self.norm2.forward(x1));
        const ff = asLinear(self.fc2, h.gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        return .{ .hidden = x1.add(ff).reuseBuffer(input.hidden) };
    }
};

pub const EmbedModel = struct {
    proj: zml.nn.Linear,
};

pub const Merger = struct {
    norm: zml.nn.LayerNorm,
    fc1: zml.nn.Linear,
    fc2: zml.nn.Linear,
    merge: i64,
    postshuffle: bool,

    pub const Input = struct {
        model: Merger,
        hidden: zml.Tensor,
    };
    pub const Output = struct { tokens: zml.Tensor };

    pub fn init(store: zml.io.TensorStore.View, merge: i64, postshuffle: bool) Merger {
        return .{
            .norm = layerNorm(store.withPrefix("norm"), 1e-6),
            .fc1 = linear(store, "linear_fc1.weight", "linear_fc1.bias"),
            .fc2 = linear(store, "linear_fc2.weight", "linear_fc2.bias"),
            .merge = merge,
            .postshuffle = postshuffle,
        };
    }

    pub fn forward(input: Input) Output {
        const self = input.model;
        var x = input.hidden.withPartialTags(.{ .b, .s, .d });
        const grouped = @divExact(x.dim(.s), self.merge);
        if (self.postshuffle) {
            x = x.splitAxis(.s, .{ .s = grouped, .m = self.merge }).merge(.{ .d = .{ .m, .d } });
            x = self.norm.forward(x);
        } else {
            x = self.norm.forward(x);
            x = x.splitAxis(.s, .{ .s = grouped, .m = self.merge }).merge(.{ .d = .{ .m, .d } });
        }
        const h = asLinear(self.fc1, x);
        x = asLinear(self.fc2, h.geluErf().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        return .{ .tokens = x };
    }
};

pub const Model = struct {
    embed: EmbedModel,
    blocks: []VisionBlock,
    merger: Merger,
    deepstack: [3]Merger,
    pos_embed: zml.Tensor,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, store_: zml.io.TensorStore.View, cfg: Config) !Model {
        const store = visionView(store_);
        const blocks = try allocator.alloc(VisionBlock, @intCast(cfg.depth));
        errdefer allocator.free(blocks);
        const block_store = store.withPrefix("blocks");
        for (blocks, 0..) |*block, i| block.* = .init(block_store.withLayer(i), cfg);
        var deepstack: [3]Merger = undefined;
        const ds = store.withPrefix("deepstack_merger_list");
        for (&deepstack, 0..) |*m, i| m.* = .init(ds.withLayer(i), cfg.mergeUnit(), true);
        return .{
            .embed = .{ .proj = conv3dLinear(store.withPrefix("patch_embed.proj"), "weight", "bias") },
            .blocks = blocks,
            .merger = .init(store.withPrefix("merger"), cfg.mergeUnit(), false),
            .deepstack = deepstack,
            .pos_embed = store.createTensor("pos_embed.weight", .{ .s, .d }, .replicated),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
    }
};

pub const EmbedInput = struct {
    model: EmbedModel,
    patches: zml.Tensor,
    pos: zml.Tensor,
};
pub const EmbedOutput = struct { hidden: zml.Tensor };

fn embed(input: EmbedInput) EmbedOutput {
    const tokens = asLinear(input.model.proj, input.patches.withPartialTags(.{ .b, .s, .d }));
    return .{ .hidden = tokens.add(input.pos.convert(tokens.dtype())) };
}

pub const LoadedModel = struct {
    inner: Model,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !LoadedModel {
        return .{
            .inner = try .init(allocator, store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
    }

    fn loadEmbed(self: *const LoadedModel, run: *const ops.Run, store: *zml.io.TensorStore) !zml.Bufferized(EmbedModel) {
        return ops.load(run, store, EmbedModel, &self.inner.embed, null);
    }

    fn loadBlock(self: *const LoadedModel, run: *const ops.Run, store: *zml.io.TensorStore, index: usize, loader: ?*zml.io.Loader) !zml.Bufferized(VisionBlock) {
        return ops.load(run, store, VisionBlock, &self.inner.blocks[index], loader);
    }

    fn loadMerger(self: *const LoadedModel, run: *const ops.Run, store: *zml.io.TensorStore) !zml.Bufferized(Merger) {
        return ops.load(run, store, Merger, &self.inner.merger, null);
    }

    fn loadDeepstack(self: *const LoadedModel, run: *const ops.Run, store: *zml.io.TensorStore, index: usize) !zml.Bufferized(Merger) {
        return ops.load(run, store, Merger, &self.inner.deepstack[index], null);
    }

    fn loadPosEmbed(self: *const LoadedModel, run: *const ops.Run, store: *zml.io.TensorStore) !zml.Buffer {
        var part = self.inner.pos_embed;
        return ops.load(run, store, zml.Tensor, &part, null);
    }
};

pub const WeightCache = struct {
    embed: zml.Bufferized(EmbedModel),
    pos: zml.Buffer,
    blocks: []zml.Bufferized(VisionBlock),
    merger: zml.Bufferized(Merger),
    deepstack: [3]zml.Bufferized(Merger),

    pub fn deinit(self: *WeightCache, allocator: std.mem.Allocator) void {
        zml.Buffer.deinitAll(EmbedModel, &self.embed);
        self.pos.deinit();
        for (self.blocks) |*block| zml.Buffer.deinitAll(VisionBlock, block);
        allocator.free(self.blocks);
        zml.Buffer.deinitAll(Merger, &self.merger);
        for (&self.deepstack) |*m| zml.Buffer.deinitAll(Merger, m);
    }

    pub fn load(run: *const ops.Run, loaded: *const LoadedModel, store: *zml.io.TensorStore) !WeightCache {
        var embed_bufs = try loaded.loadEmbed(run, store);
        errdefer zml.Buffer.deinitAll(EmbedModel, &embed_bufs);
        var pos = try loaded.loadPosEmbed(run, store);
        errdefer pos.deinit();
        const blocks = try run.allocator.alloc(zml.Bufferized(VisionBlock), loaded.inner.blocks.len);
        errdefer run.allocator.free(blocks);
        var filled: usize = 0;
        errdefer {
            for (blocks[0..filled]) |*block| zml.Buffer.deinitAll(VisionBlock, block);
        }
        for (blocks, 0..) |*block, i| {
            block.* = try loaded.loadBlock(run, store, i, null);
            filled += 1;
        }
        var merger = try loaded.loadMerger(run, store);
        errdefer zml.Buffer.deinitAll(Merger, &merger);
        var deepstack: [3]zml.Bufferized(Merger) = undefined;
        var ds_filled: usize = 0;
        errdefer {
            for (deepstack[0..ds_filled]) |*m| zml.Buffer.deinitAll(Merger, m);
        }
        for (&deepstack, 0..) |*m, i| {
            m.* = try loaded.loadDeepstack(run, store, i);
            ds_filled += 1;
        }
        return .{
            .embed = embed_bufs,
            .pos = pos,
            .blocks = blocks,
            .merger = merger,
            .deepstack = deepstack,
        };
    }
};

pub const Compiled = struct {
    embed: zml.FnExe(embed),
    block: zml.FnExe(VisionBlock.forward),
    merger: zml.FnExe(Merger.forward),
    deepstack: zml.FnExe(Merger.forward),
    seq: u32,
    merged: u32,

    pub fn deinit(self: *Compiled) void {
        self.embed.deinit();
        self.block.deinit();
        self.merger.deinit();
        self.deepstack.deinit();
    }
};

pub const Grid = struct { h: u32, w: u32 };

/// Python 3 `round`: nearest, ties to even. Official Qwen2VL `smart_resize`.
fn pyRoundHalfEven(x: f64) f64 {
    const lo = @floor(x);
    const frac = x - lo;
    if (frac < 0.5) return lo;
    if (frac > 0.5) return lo + 1.0;
    const n: i64 = @intFromFloat(lo);
    if (@mod(n, 2) == 0) return lo;
    return lo + 1.0;
}

/// Qwen2VL `smart_resize` (f64, Python 3 even ties).
fn chooseGrid(cfg: Config, src_h: u32, src_w: u32, video: bool) struct { h: u32, w: u32 } {
    const factor: f64 = @floatFromInt(cfg.patch_size * cfg.spatial_merge_size);
    const height: f64 = @floatFromInt(src_h);
    const width: f64 = @floatFromInt(src_w);
    const min_pixels: f64 = if (video) 4096.0 else 65536.0;
    const max_pixels: f64 = if (video) 25165824.0 else 16777216.0;
    var h_bar = pyRoundHalfEven(height / factor) * factor;
    var w_bar = pyRoundHalfEven(width / factor) * factor;
    if (h_bar * w_bar > max_pixels) {
        const beta = @sqrt((height * width) / max_pixels);
        h_bar = @max(factor, @floor(height / beta / factor) * factor);
        w_bar = @max(factor, @floor(width / beta / factor) * factor);
    } else if (h_bar * w_bar < min_pixels) {
        const beta = @sqrt(min_pixels / (height * width));
        h_bar = @ceil(height * beta / factor) * factor;
        w_bar = @ceil(width * beta / factor) * factor;
    }
    return .{ .h = @intFromFloat(h_bar), .w = @intFromFloat(w_bar) };
}

fn patchifyRgb(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, cfg: Config) !struct { patches: []f32, grid: Grid, seq: u32 } {
    const size = chooseGrid(cfg, src_h, src_w, false);
    if (src_h != size.h or src_w != size.w) return error.VisionNeedsResize;
    const patch: u32 = @intCast(cfg.patch_size);
    const merge: u32 = @intCast(cfg.spatial_merge_size);
    const gh = size.h / patch;
    const gw = size.w / patch;
    const seq = gh * gw;
    const width: u32 = @intCast(cfg.patchIn());
    const out = try allocator.alloc(f32, seq * width);
    var row: usize = 0;
    var ih: u32 = 0;
    while (ih < gh) : (ih += merge) {
        var iw: u32 = 0;
        while (iw < gw) : (iw += merge) {
            var di: u32 = 0;
            while (di < merge) : (di += 1) {
                var dj: u32 = 0;
                while (dj < merge) : (dj += 1) {
                    var dst: usize = 0;
                    var c: u32 = 0;
                    while (c < 3) : (c += 1) {
                        var t: u32 = 0;
                        while (t < 2) : (t += 1) {
                            var ph: u32 = 0;
                            while (ph < patch) : (ph += 1) {
                                var pw: u32 = 0;
                                while (pw < patch) : (pw += 1) {
                                    const y = (ih + di) * patch + ph;
                                    const x = (iw + dj) * patch + pw;
                                    const v = @as(f32, @floatFromInt(rgb[(y * size.w + x) * 3 + c])) / 255.0;
                                    out[row * width + dst] = v * 2.0 - 1.0;
                                    dst += 1;
                                }
                            }
                        }
                    }
                    row += 1;
                }
            }
        }
    }
    return .{ .patches = out, .grid = .{ .h = gh, .w = gw }, .seq = seq };
}

fn patchifyVideo(
    allocator: std.mem.Allocator,
    rgb: []const u8,
    frames: u32,
    src_h: u32,
    src_w: u32,
    cfg: Config,
) !struct { patches: []f32, grid: Grid, seq: u32, temporal: u32 } {
    const size = chooseGrid(cfg, src_h, src_w, true);
    if (src_h != size.h or src_w != size.w) return error.VisionNeedsResize;
    const even = frames + (frames % 2);
    const temporal = even / 2;
    const plane = @as(usize, src_w) * src_h * 3;
    const stacked = try allocator.alloc(u8, even * plane);
    defer allocator.free(stacked);
    var f: u32 = 0;
    while (f < even) : (f += 1) {
        const src_f = if (f < frames) f else frames - 1;
        @memcpy(stacked[f * plane ..][0..plane], rgb[src_f * plane ..][0..plane]);
    }

    const patch: u32 = @intCast(cfg.patch_size);
    const merge: u32 = @intCast(cfg.spatial_merge_size);
    const gh = size.h / patch;
    const gw = size.w / patch;
    const seq = temporal * gh * gw;
    const width: u32 = @intCast(cfg.patchIn());
    const out = try allocator.alloc(f32, seq * width);
    var row: usize = 0;
    var tf: u32 = 0;
    while (tf < temporal) : (tf += 1) {
        var ih: u32 = 0;
        while (ih < gh) : (ih += merge) {
            var iw: u32 = 0;
            while (iw < gw) : (iw += merge) {
                var di: u32 = 0;
                while (di < merge) : (di += 1) {
                    var dj: u32 = 0;
                    while (dj < merge) : (dj += 1) {
                        var dst: u32 = 0;
                        var c: u32 = 0;
                        while (c < 3) : (c += 1) {
                            var t: u32 = 0;
                            while (t < 2) : (t += 1) {
                                var ph: u32 = 0;
                                while (ph < patch) : (ph += 1) {
                                    var pw: u32 = 0;
                                    while (pw < patch) : (pw += 1) {
                                        const y = (ih + di) * patch + ph;
                                        const x = (iw + dj) * patch + pw;
                                        const pix = (((tf * 2 + t) * size.h + y) * size.w + x) * 3 + c;
                                        const v = @as(f32, @floatFromInt(stacked[pix])) / 255.0;
                                        out[row * width + dst] = v * 2.0 - 1.0;
                                        dst += 1;
                                    }
                                }
                            }
                        }
                        row += 1;
                    }
                }
            }
        }
    }
    return .{ .patches = out, .grid = .{ .h = gh, .w = gw }, .seq = seq, .temporal = temporal };
}

fn interpolatePos(allocator: std.mem.Allocator, table: []const f32, table_side: u32, hidden: u32, gh: u32, gw: u32, merge: u32) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, gh) * gw * hidden);
    var row: usize = 0;
    var ih: u32 = 0;
    while (ih < gh) : (ih += merge) {
        var iw: u32 = 0;
        while (iw < gw) : (iw += merge) {
            var di: u32 = 0;
            while (di < merge) : (di += 1) {
                var dj: u32 = 0;
                while (dj < merge) : (dj += 1) {
                    const yden = @max(gh, 2) - 1;
                    const xden = @max(gw, 2) - 1;
                    const y = @as(f32, @floatFromInt(ih + di)) * @as(f32, @floatFromInt(table_side - 1)) / @as(f32, @floatFromInt(yden));
                    const x = @as(f32, @floatFromInt(iw + dj)) * @as(f32, @floatFromInt(table_side - 1)) / @as(f32, @floatFromInt(xden));
                    const y0: u32 = @intFromFloat(@floor(y));
                    const x0: u32 = @intFromFloat(@floor(x));
                    const y1 = @min(table_side - 1, y0 + 1);
                    const x1 = @min(table_side - 1, x0 + 1);
                    const fy = y - @as(f32, @floatFromInt(y0));
                    const fx = x - @as(f32, @floatFromInt(x0));
                    var d: u32 = 0;
                    while (d < hidden) : (d += 1) {
                        const a = table[(y0 * table_side + x0) * hidden + d];
                        const b = table[(y0 * table_side + x1) * hidden + d];
                        const c = table[(y1 * table_side + x0) * hidden + d];
                        const e = table[(y1 * table_side + x1) * hidden + d];
                        out[row * hidden + d] = a * (1 - fy) * (1 - fx) + b * (1 - fy) * fx + c * fy * (1 - fx) + e * fy * fx;
                    }
                    row += 1;
                }
            }
        }
    }
    return out;
}

fn visionInvFreq(i: u32, half: u32) f32 {
    return 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(half)));
}

fn visionRope(allocator: std.mem.Allocator, gh: u32, gw: u32, head_dim: u32, merge: u32) !struct { cos: []f32, sin: []f32 } {
    const seq = gh * gw;
    const half = head_dim / 2;
    const n_freq = half / 2;
    const cos = try allocator.alloc(f32, seq * head_dim);
    errdefer allocator.free(cos);
    const sin = try allocator.alloc(f32, seq * head_dim);
    var row: usize = 0;
    var ih: u32 = 0;
    while (ih < gh) : (ih += merge) {
        var iw: u32 = 0;
        while (iw < gw) : (iw += merge) {
            var di: u32 = 0;
            while (di < merge) : (di += 1) {
                var dj: u32 = 0;
                while (dj < merge) : (dj += 1) {
                    const hpos: f32 = @floatFromInt(ih + di);
                    const wpos: f32 = @floatFromInt(iw + dj);
                    var i: u32 = 0;
                    while (i < n_freq) : (i += 1) {
                        const freq = visionInvFreq(i, half);
                        const ang_h = hpos * freq;
                        const ang_w = wpos * freq;
                        const ch = @cos(ang_h);
                        const sh = @sin(ang_h);
                        const cw = @cos(ang_w);
                        const sw = @sin(ang_w);
                        cos[row * head_dim + i] = ch;
                        cos[row * head_dim + n_freq + i] = cw;
                        cos[row * head_dim + half + i] = ch;
                        cos[row * head_dim + half + n_freq + i] = cw;
                        sin[row * head_dim + i] = sh;
                        sin[row * head_dim + n_freq + i] = sw;
                        sin[row * head_dim + half + i] = sh;
                        sin[row * head_dim + half + n_freq + i] = sw;
                    }
                    row += 1;
                }
            }
        }
    }
    return .{ .cos = cos, .sin = sin };
}

pub const EncodedVisual = struct {
    merged: []f32,
    deepstack: [3][]f32,

    pub fn deinit(self: EncodedVisual, allocator: std.mem.Allocator) void {
        allocator.free(self.merged);
        for (self.deepstack) |d| allocator.free(d);
    }
};

fn runPatches(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const Compiled,
    loaded: *const LoadedModel,
    cache: *const WeightCache,
    patches: []const f32,
    grid: Grid,
    seq: u32,
    temporal: u32,
) !EncodedVisual {
    if (seq != compiled.seq) return error.VisionSeqMismatch;

    const n_blocks = cache.blocks.len;
    const vision_start: std.Io.Timestamp = .now(io, .awake);
    log.info("vision: start seq={d} grid={d}x{d} blocks={d} temporal={d}", .{
        seq,
        grid.h,
        grid.w,
        n_blocks,
        temporal,
    });

    const table_host = try toF32(allocator, io, cache.pos);
    defer allocator.free(table_host);
    const side: u32 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(loaded.cfg.num_position_embeddings))));
    const merge: u32 = @intCast(loaded.cfg.spatial_merge_size);
    const spatial_pos = try interpolatePos(allocator, table_host, side, @intCast(loaded.cfg.hidden_size), grid.h, grid.w, merge);
    defer allocator.free(spatial_pos);
    const spatial_rope = try visionRope(allocator, grid.h, grid.w, @intCast(loaded.cfg.headDim()), merge);
    defer allocator.free(spatial_rope.cos);
    defer allocator.free(spatial_rope.sin);
    const pos = try tileTemporal(allocator, spatial_pos, temporal);
    defer allocator.free(pos);
    const rope_cos = try tileTemporal(allocator, spatial_rope.cos, temporal);
    defer allocator.free(rope_cos);
    const rope_sin = try tileTemporal(allocator, spatial_rope.sin, temporal);
    defer allocator.free(rope_sin);

    var embed_runner = try zml.FnExe(embed).Runner(.{.model}).init(&compiled.embed, allocator, .{ .model = cache.embed });
    defer embed_runner.deinit(allocator);
    var patch_buf = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = seq, .d = loaded.cfg.patchIn() }, .f32), .replicated, std.mem.sliceAsBytes(patches));
    defer patch_buf.deinit();
    var pos_buf = try zml.Buffer.fromBytes(io, platform, .init(.{ .b = 1, .s = seq, .d = loaded.cfg.hidden_size }, .f32), .replicated, std.mem.sliceAsBytes(pos));
    defer pos_buf.deinit();
    var hidden: zml.Buffer = undefined;
    embed_runner.run(io, .{
        .inputs = .{ .patches = patch_buf, .pos = pos_buf },
        .outputs = .{ .hidden = &hidden },
        .opts = .{ .wait = true },
    });
    defer hidden.deinit();

    var cos_buf = try zml.Buffer.fromBytes(io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, .f32), .replicated, std.mem.sliceAsBytes(rope_cos));
    defer cos_buf.deinit();
    var sin_buf = try zml.Buffer.fromBytes(io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, .f32), .replicated, std.mem.sliceAsBytes(rope_sin));
    defer sin_buf.deinit();

    var deepstack: [3][]f32 = .{ &.{}, &.{}, &.{} };
    errdefer {
        for (deepstack) |d| if (d.len != 0) allocator.free(d);
    }
    var ds_i: usize = 0;
    const BlockRunner = zml.FnExe(VisionBlock.forward).Runner(.{.layer});
    var block_i: usize = 0;
    while (block_i < n_blocks) : (block_i += 1) {
        var block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = cache.blocks[block_i] });
        defer block_runner.deinit(allocator);
        var next: zml.Buffer = undefined;
        block_runner.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        hidden.deinit();
        hidden = next;
        if (ds_i < 3 and @as(i64, @intCast(block_i)) == loaded.cfg.deepstack_visual_indexes[ds_i]) {
            var ds_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.deepstack, allocator, .{ .model = cache.deepstack[ds_i] });
            defer ds_run.deinit(allocator);
            var tokens: zml.Buffer = undefined;
            ds_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &tokens }, .opts = .{ .wait = true } });
            defer tokens.deinit();
            deepstack[ds_i] = try toF32(allocator, io, tokens);
            ds_i += 1;
        }
    }
    var merge_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.merger, allocator, .{ .model = cache.merger });
    defer merge_run.deinit(allocator);
    var merged_buf: zml.Buffer = undefined;
    merge_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &merged_buf }, .opts = .{ .wait = true } });
    defer merged_buf.deinit();
    const merged = try toF32(allocator, io, merged_buf);
    log.info("vision: ok merged={d} [{f}]", .{ merged.len, vision_start.untilNow(io, .awake) });
    return .{
        .merged = merged,
        .deepstack = deepstack,
    };
}

fn tileTemporal(allocator: std.mem.Allocator, src: []const f32, temporal: u32) ![]f32 {
    if (temporal <= 1) return allocator.dupe(f32, src);
    const out = try allocator.alloc(f32, src.len * temporal);
    var t: u32 = 0;
    while (t < temporal) : (t += 1) {
        @memcpy(out[t * src.len ..][0..src.len], src);
    }
    return out;
}

pub fn runImage(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const Compiled,
    loaded: *const LoadedModel,
    cache: *const WeightCache,
    rgb: []const u8,
    src_h: u32,
    src_w: u32,
) !EncodedVisual {
    const patched = try patchifyRgb(allocator, rgb, src_h, src_w, loaded.cfg);
    defer allocator.free(patched.patches);
    return runPatches(allocator, io, platform, compiled, loaded, cache, patched.patches, patched.grid, patched.seq, 1);
}

pub fn runVideo(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const Compiled,
    loaded: *const LoadedModel,
    cache: *const WeightCache,
    rgb: []const u8,
    frames: u32,
    src_h: u32,
    src_w: u32,
) !EncodedVisual {
    const patched = try patchifyVideo(allocator, rgb, frames, src_h, src_w, loaded.cfg);
    defer allocator.free(patched.patches);
    return runPatches(allocator, io, platform, compiled, loaded, cache, patched.patches, patched.grid, patched.seq, patched.temporal);
}

pub fn configFromRepo(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, text_hidden: i64) !Config {
    const file = try repo.openFile(io, "text_encoder/config.json", .{});
    defer file.close(io);
    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();
    const parsed = try std.json.parseFromTokenSource(FileConfig, allocator, &reader, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    return parsed.value.resolve(text_hidden);
}

pub fn spatialTokens(cfg: Config, src_h: u32, src_w: u32, video: bool) struct { grid: Grid, seq: u32, merged: u32 } {
    const size = chooseGrid(cfg, src_h, src_w, video);
    const patch: u32 = @intCast(cfg.patch_size);
    const gh = size.h / patch;
    const gw = size.w / patch;
    const seq = gh * gw;
    return .{ .grid = .{ .h = gh, .w = gw }, .seq = seq, .merged = seq / @as(u32, @intCast(cfg.mergeUnit())) };
}

fn compileFn(
    comptime function: anytype,
    comptime name: []const u8,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
    args: std.meta.ArgsTuple(@TypeOf(function)),
) !zml.FnExe(function) {
    progress.increaseEstimatedTotalItems(1);
    return zml.FnExe(function).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = name,
    }, args);
}

fn toF32(allocator: std.mem.Allocator, io: std.Io, buffer: zml.Buffer) ![]f32 {
    const slice = try buffer.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    switch (buffer.shape().dtype()) {
        .f32 => return allocator.dupe(f32, slice.items(f32)),
        .bf16 => {
            const source = slice.items(zml.floats.BFloat16);
            const converted = try allocator.alloc(f32, source.len);
            for (converted, source) |*dst, value| dst.* = value.toF32();
            return converted;
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: Model,
    seq: u32,
    shardings: []const zml.Sharding,
    progress: *std.Progress.Node,
) !Compiled {
    const cfg = model.cfg;
    const dt = model.embed.proj.weight.dtype();
    const merged: u32 = @intCast(@divExact(@as(i64, seq), cfg.mergeUnit()));
    const embed_exe = try compileFn(embed, "minimax_h3_vision_embed", allocator, io, platform, shardings, progress, .{.{
        .model = model.embed,
        .patches = .init(.{ .b = 1, .s = seq, .d = cfg.patchIn() }, .f32),
        .pos = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, .f32),
    }});
    errdefer embed_exe.deinit();
    const block_exe = try compileFn(VisionBlock.forward, "minimax_h3_vision_block", allocator, io, platform, shardings, progress, .{.{
        .layer = model.blocks[0],
        .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
        .cos = .init(.{ .s = seq, .hd = cfg.headDim() }, .f32),
        .sin = .init(.{ .s = seq, .hd = cfg.headDim() }, .f32),
    }});
    errdefer block_exe.deinit();
    const merger_exe = try compileFn(Merger.forward, "minimax_h3_vision_merger", allocator, io, platform, shardings, progress, .{.{
        .model = model.merger,
        .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
    }});
    errdefer merger_exe.deinit();
    const ds_exe = try compileFn(Merger.forward, "minimax_h3_vision_deepstack", allocator, io, platform, shardings, progress, .{.{
        .model = model.deepstack[0],
        .hidden = .init(.{ .b = 1, .s = seq, .d = cfg.hidden_size }, dt),
    }});
    errdefer ds_exe.deinit();
    return .{
        .embed = embed_exe,
        .block = block_exe,
        .merger = merger_exe,
        .deepstack = ds_exe,
        .seq = seq,
        .merged = merged,
    };
}
