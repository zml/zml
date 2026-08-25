const std = @import("std");

const zml = @import("zml");

const config_mod = @import("../core/config.zig");
const weights = @import("../core/weights.zig");

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
    rms_norm_eps: f32 = 1e-6,

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
    if (store.hasKey("model.visual.patch_embed.proj.weight")) return store.withPrefix("model.visual");
    return store;
}

pub fn ready(store: zml.io.TensorStore.View) bool {
    return store.hasKey("model.visual.patch_embed.proj.weight");
}

fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8) zml.nn.Linear {
    const w = switch (blk: {
        var buffer: [256]u8 = undefined;
        const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", weight_name }) catch break :blk 2;
        break :blk if (store.store.getShape(key)) |s| s.rank() else 2;
    }) {
        5 => store.createTensor(weight_name, .{ .dout, .d, .kt, .kh, .kw }, .replicated),
        else => store.createTensor(weight_name, .{ .dout, .d }, .replicated),
    };
    return .init(w, if (bias_name) |n| store.maybeCreateTensor(n, .{.dout}, .replicated) else null, .d);
}

fn unloadLinear(lin: *zml.Bufferized(zml.nn.Linear)) void {
    lin.weight.deinit();
    if (lin.bias) |*b| b.deinit();
}

fn asLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    var w = lin.weight;
    if (w.rank() == 5) {
        w = w.merge(.{ .d = .{ .d, .kt, .kh, .kw } });
    } else {
        while (w.rank() > 2) w = w.squeeze(-1);
    }
    return (zml.nn.Linear.init(w.withTags(.{ .dout, .d }), lin.bias, .d)).forward(x.convert(w.dtype()));
}

const LayerNorm = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) LayerNorm {
        return .{
            .weight = store.createTensor("weight", .{.d}, .replicated),
            .bias = store.maybeCreateTensor("bias", .{.d}, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LayerNorm)) void {
        self.weight.deinit();
        if (self.bias) |*b| b.deinit();
    }

    pub fn forward(self: LayerNorm, x: zml.Tensor) zml.Tensor {
        const weight = self.weight.convert(.f32);
        const bias = if (self.bias) |b| b.convert(.f32) else null;
        return (zml.nn.LayerNorm{ .weight = weight, .bias = bias, .eps = 1e-6 }).forward(x.convert(.f32)).convert(x.dtype());
    }
};

fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const half = @divExact(x.dim(-1), 2);
    const x1 = x.slice1d(-1, .{ .start = 0, .end = half });
    const x2 = x.slice1d(-1, .{ .start = half, .end = x.dim(-1) });
    const rotated = zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    const c = cos.broad(x.shape());
    const s = sin.broad(x.shape());
    return x.mul(c).add(rotated.mul(s));
}

pub const VisionBlock = struct {
    norm1: LayerNorm,
    qkv: zml.nn.Linear,
    proj: zml.nn.Linear,
    norm2: LayerNorm,
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
            .norm1 = .init(store.withPrefix("norm1")),
            .qkv = linear(attn, "qkv.weight", "qkv.bias"),
            .proj = linear(attn, "proj.weight", "proj.bias"),
            .norm2 = .init(store.withPrefix("norm2")),
            .fc1 = linear(mlp, "linear_fc1.weight", "linear_fc1.bias"),
            .fc2 = linear(mlp, "linear_fc2.weight", "linear_fc2.bias"),
            .num_heads = cfg.num_heads,
            .head_dim = cfg.headDim(),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(VisionBlock)) void {
        LayerNorm.unloadBuffers(&self.norm1);
        unloadLinear(&self.qkv);
        unloadLinear(&self.proj);
        LayerNorm.unloadBuffers(&self.norm2);
        unloadLinear(&self.fc1);
        unloadLinear(&self.fc2);
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
        const attn = zml.nn.sdpa(q.rename(.{ .s = .q }), k.rename(.{ .s = .k }), v.rename(.{ .s = .k }), .{}).rename(.{ .q = .s }).merge(.{ .d = .{ .h, .hd } });
        const x1 = residual.add(asLinear(self.proj, attn).rename(.{ .dout = .d }));
        const ff = asLinear(self.fc2, asLinear(self.fc1, self.norm2.forward(x1)).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
        return .{ .hidden = x1.add(ff).reuseBuffer(input.hidden) };
    }
};

pub const EmbedModel = struct {
    proj: zml.nn.Linear,

    pub fn unloadBuffers(self: *zml.Bufferized(EmbedModel)) void {
        unloadLinear(&self.proj);
    }
};

pub const Merger = struct {
    norm: LayerNorm,
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
            .norm = .init(store.withPrefix("norm")),
            .fc1 = linear(store, "linear_fc1.weight", "linear_fc1.bias"),
            .fc2 = linear(store, "linear_fc2.weight", "linear_fc2.bias"),
            .merge = merge,
            .postshuffle = postshuffle,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Merger)) void {
        LayerNorm.unloadBuffers(&self.norm);
        unloadLinear(&self.fc1);
        unloadLinear(&self.fc2);
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
        x = asLinear(self.fc2, asLinear(self.fc1, x).gelu().rename(.{ .dout = .d })).rename(.{ .dout = .d });
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
            .embed = .{ .proj = linear(store.withPrefix("patch_embed.proj"), "weight", "bias") },
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

pub fn embed(input: EmbedInput) EmbedOutput {
    const tokens = asLinear(input.model.proj, input.patches.withPartialTags(.{ .b, .s, .d })).rename(.{ .dout = .d });
    return .{ .hidden = tokens.add(input.pos.convert(tokens.dtype())) };
}

pub const LoadedModel = struct {
    inner: Model,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View, text_hidden: i64) !LoadedModel {
        const cfg = try configFromRepo(allocator, io, repo, text_hidden);
        return .{
            .inner = try .init(allocator, store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
    }

    pub fn loadEmbed(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, progress: *std.Progress.Node) !zml.Bufferized(EmbedModel) {
        return weights.load(allocator, io, platform, store, shardings, EmbedModel, &self.inner.embed, progress, null);
    }

    pub fn loadBlock(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, index: usize, progress: *std.Progress.Node, loader: ?*zml.io.Loader) !zml.Bufferized(VisionBlock) {
        return weights.load(allocator, io, platform, store, shardings, VisionBlock, &self.inner.blocks[index], progress, loader);
    }

    pub fn loadMerger(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, progress: *std.Progress.Node) !zml.Bufferized(Merger) {
        return weights.load(allocator, io, platform, store, shardings, Merger, &self.inner.merger, progress, null);
    }

    pub fn loadDeepstack(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, index: usize, progress: *std.Progress.Node) !zml.Bufferized(Merger) {
        return weights.load(allocator, io, platform, store, shardings, Merger, &self.inner.deepstack[index], progress, null);
    }

    pub fn loadPosEmbed(self: *const LoadedModel, allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: *zml.io.TensorStore, shardings: []const zml.Sharding, progress: *std.Progress.Node) !zml.Buffer {
        var part = self.inner.pos_embed;
        return weights.load(allocator, io, platform, store, shardings, zml.Tensor, &part, progress, null);
    }
};

pub const Grid = struct { h: u32, w: u32 };

pub fn chooseGrid(cfg: Config, src_h: u32, src_w: u32, video: bool) struct { h: u32, w: u32 } {
    const factor: u32 = @intCast(cfg.patch_size * cfg.spatial_merge_size);
    var target_h = @max(factor, (src_h + factor / 2) / factor * factor);
    var target_w = @max(factor, (src_w + factor / 2) / factor * factor);
    const min_pixels: f32 = if (video) 4096.0 else 65536.0;
    const max_pixels: f32 = if (video) 25165824.0 else 16777216.0;
    const area = @as(f32, @floatFromInt(target_h)) * @as(f32, @floatFromInt(target_w));
    if (area > max_pixels) {
        const scale = @sqrt((@as(f32, @floatFromInt(src_h * src_w))) / max_pixels);
        target_h = @max(factor, @as(u32, @intFromFloat(@floor(@as(f32, @floatFromInt(src_h)) / scale / @as(f32, @floatFromInt(factor))))) * factor);
        target_w = @max(factor, @as(u32, @intFromFloat(@floor(@as(f32, @floatFromInt(src_w)) / scale / @as(f32, @floatFromInt(factor))))) * factor);
    } else if (area < min_pixels) {
        const scale = @sqrt(min_pixels / @as(f32, @floatFromInt(src_h * src_w)));
        target_h = @as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(src_h)) * scale / @as(f32, @floatFromInt(factor))))) * factor;
        target_w = @as(u32, @intFromFloat(@ceil(@as(f32, @floatFromInt(src_w)) * scale / @as(f32, @floatFromInt(factor))))) * factor;
    }
    return .{ .h = target_h, .w = target_w };
}

pub fn patchifyRgb(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, cfg: Config) !struct { patches: []f32, grid: Grid, seq: u32 } {
    const media = @import("../runtime/media.zig");
    const size = chooseGrid(cfg, src_h, src_w, false);
    const resized = try media.resizeRgb(allocator, rgb, src_w, src_h, size.w, size.h);
    defer allocator.free(resized);
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
                                    const v = @as(f32, @floatFromInt(resized[(y * size.w + x) * 3 + c])) / 255.0;
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

pub fn patchifyVideo(
    allocator: std.mem.Allocator,
    rgb: []const u8,
    frames: u32,
    src_h: u32,
    src_w: u32,
    cfg: Config,
) !struct { patches: []f32, grid: Grid, seq: u32, temporal: u32 } {
    const media = @import("../runtime/media.zig");
    const size = chooseGrid(cfg, src_h, src_w, true);
    const even = frames + (frames % 2);
    const temporal = even / 2;
    const plane = @as(usize, src_w) * src_h * 3;
    const resized_plane = @as(usize, size.w) * size.h * 3;
    const stacked = try allocator.alloc(u8, even * resized_plane);
    defer allocator.free(stacked);
    var f: u32 = 0;
    while (f < even) : (f += 1) {
        const src_f = if (f < frames) f else frames - 1;
        const frame = try media.resizeRgb(allocator, rgb[src_f * plane ..][0..plane], src_w, src_h, size.w, size.h);
        defer allocator.free(frame);
        @memcpy(stacked[f * resized_plane ..][0..resized_plane], frame);
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

pub fn interpolatePos(allocator: std.mem.Allocator, table: []const f32, table_side: u32, hidden: u32, gh: u32, gw: u32) ![]f32 {
    const out = try allocator.alloc(f32, @as(usize, gh) * gw * hidden);
    const merge: u32 = 2;
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

pub fn visionRope(allocator: std.mem.Allocator, gh: u32, gw: u32, head_dim: u32) !struct { cos: []f32, sin: []f32 } {
    const seq = gh * gw;
    const half = head_dim / 2;
    const cos = try allocator.alloc(f32, seq * head_dim);
    errdefer allocator.free(cos);
    const sin = try allocator.alloc(f32, seq * head_dim);
    const merge: u32 = 2;
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
                    var f: u32 = 0;
                    while (f < half) : (f += 1) {
                        const freq = 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)));
                        const ang = if (f < half / 2) hpos * freq else wpos * freq;
                        const c = @cos(ang);
                        const s = @sin(ang);
                        cos[row * head_dim + f] = c;
                        cos[row * head_dim + half + f] = c;
                        sin[row * head_dim + f] = s;
                        sin[row * head_dim + half + f] = s;
                    }
                    row += 1;
                }
            }
        }
    }
    return .{ .cos = cos, .sin = sin };
}

pub fn hostInterleavedMrope(
    pos: []const f32,
    seq: u32,
    head_dim: u32,
    theta: f32,
    section: [3]i64,
    cos: []f32,
    sin: []f32,
) void {
    const half = head_dim / 2;
    std.debug.assert(pos.len >= seq * 3);
    std.debug.assert(cos.len >= seq * head_dim);
    var i: u32 = 0;
    while (i < seq) : (i += 1) {
        const pt = pos[i * 3 + 0];
        const ph = pos[i * 3 + 1];
        const pw = pos[i * 3 + 2];
        var f: u32 = 0;
        while (f < half) : (f += 1) {
            var p = pt;
            const h_end = @as(u32, @intCast(section[1] * 3));
            const w_end = @as(u32, @intCast(section[2] * 3));
            if (f < h_end and f % 3 == 1) p = ph;
            if (f < w_end and f % 3 == 2) p = pw;
            const freq = 1.0 / std.math.pow(f32, theta, @as(f32, @floatFromInt(f)) / @as(f32, @floatFromInt(half)));
            const ang = p * freq;
            const c = @cos(ang);
            const s = @sin(ang);
            cos[i * head_dim + f] = c;
            cos[i * head_dim + half + f] = c;
            sin[i * head_dim + f] = s;
            sin[i * head_dim + half + f] = s;
        }
    }
}

pub fn fillArangePositions(out: []f32, seq: u32) void {
    var i: u32 = 0;
    while (i < seq) : (i += 1) {
        const v: f32 = @floatFromInt(i);
        out[i * 3 + 0] = v;
        out[i * 3 + 1] = v;
        out[i * 3 + 2] = v;
    }
}

pub const EncodedVisual = struct {
    merged: []f32,
    deepstack: [3][]f32,
    tokens: u32,
    grid: Grid,
    temporal: u32 = 1,

    pub fn deinit(self: EncodedVisual, allocator: std.mem.Allocator) void {
        allocator.free(self.merged);
        for (self.deepstack) |d| allocator.free(d);
    }
};

fn runPatches(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const @import("../runtime/pipeline.zig").VisionCompiled,
    loaded: *const LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    patches: []const f32,
    grid: Grid,
    seq: u32,
    temporal: u32,
    progress: *std.Progress.Node,
) !EncodedVisual {
    if (seq != compiled.seq) return error.VisionSeqMismatch;

    const n_blocks = loaded.inner.blocks.len;
    const vision_start: std.Io.Timestamp = .now(io, .awake);
    log.info("vision: start seq={d} grid={d}x{d} blocks={d} temporal={d}", .{
        seq,
        grid.h,
        grid.w,
        n_blocks,
        temporal,
    });

    var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
    defer EmbedModel.unloadBuffers(&embed_bufs);
    var pos_table = try loaded.loadPosEmbed(allocator, io, platform, store, shardings, progress);
    defer pos_table.deinit();

    var loaders = [2]zml.io.Loader{
        try weights.initLoader(allocator, platform),
        try weights.initLoader(allocator, platform),
    };
    defer loaders[0].deinit();
    defer loaders[1].deinit();
    const VisFut = @TypeOf(try io.concurrent(loadVisionBlock, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
    }));
    var current_f: ?VisFut = if (n_blocks > 0) try io.concurrent(loadVisionBlock, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 0), progress, &loaders[0],
    }) else null;
    var next_f: ?VisFut = if (n_blocks > 1) try io.concurrent(loadVisionBlock, .{
        allocator, io, platform, loaded, store, shardings, @as(usize, 1), progress, &loaders[1],
    }) else null;
    errdefer cancelVision(&current_f, io);
    errdefer cancelVision(&next_f, io);

    const table_host = try bufferToF32(allocator, io, pos_table);
    defer allocator.free(table_host);
    const side: u32 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(loaded.cfg.num_position_embeddings))));
    const spatial_pos = try interpolatePos(allocator, table_host, side, @intCast(loaded.cfg.hidden_size), grid.h, grid.w);
    defer allocator.free(spatial_pos);
    const spatial_rope = try visionRope(allocator, grid.h, grid.w, @intCast(loaded.cfg.headDim()));
    defer allocator.free(spatial_rope.cos);
    defer allocator.free(spatial_rope.sin);
    const pos = try tileTemporal(allocator, spatial_pos, temporal, @intCast(loaded.cfg.hidden_size));
    defer allocator.free(pos);
    const rope_cos = try tileTemporal(allocator, spatial_rope.cos, temporal, @intCast(loaded.cfg.headDim()));
    defer allocator.free(rope_cos);
    const rope_sin = try tileTemporal(allocator, spatial_rope.sin, temporal, @intCast(loaded.cfg.headDim()));
    defer allocator.free(rope_sin);

    var embed_runner = try zml.FnExe(embed).Runner(.{.model}).init(&compiled.embed, allocator, .{ .model = embed_bufs });
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

    var cos_buf = try bufferFromF32(allocator, io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, hidden.shape().dtype()), rope_cos);
    defer cos_buf.deinit();
    var sin_buf = try bufferFromF32(allocator, io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, hidden.shape().dtype()), rope_sin);
    defer sin_buf.deinit();

    var deepstack: [3][]f32 = .{ &.{}, &.{}, &.{} };
    errdefer {
        for (deepstack) |d| if (d.len != 0) allocator.free(d);
    }
    var ds_i: usize = 0;
    const BlockRunner = zml.FnExe(VisionBlock.forward).Runner(.{.layer});
    var block_runner: ?BlockRunner = null;
    defer if (block_runner) |*r| r.deinit(allocator);
    var block_i: usize = 0;
    while (block_i < n_blocks) : (block_i += 1) {
        var block_bufs = try current_f.?.await(io);
        current_f = null;
        defer VisionBlock.unloadBuffers(&block_bufs);
        current_f = next_f;
        next_f = if (block_i + 2 < n_blocks) try io.concurrent(loadVisionBlock, .{
            allocator, io, platform, loaded, store, shardings, block_i + 2, progress, &loaders[(block_i + 2) % 2],
        }) else null;
        if (block_runner) |*r| {
            weights.rebake(r, .{ .layer = block_bufs });
        } else {
            block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = block_bufs });
        }
        var next: zml.Buffer = undefined;
        block_runner.?.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        hidden.deinit();
        hidden = next;
        if (ds_i < 3 and @as(i64, @intCast(block_i)) == loaded.cfg.deepstack_visual_indexes[ds_i]) {
            var ds_bufs = try loaded.loadDeepstack(allocator, io, platform, store, shardings, ds_i, progress);
            defer Merger.unloadBuffers(&ds_bufs);
            var ds_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.deepstack, allocator, .{ .model = ds_bufs });
            defer ds_run.deinit(allocator);
            var tokens: zml.Buffer = undefined;
            ds_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &tokens }, .opts = .{ .wait = true } });
            defer tokens.deinit();
            const host = try bufferToF32(allocator, io, tokens);
            deepstack[ds_i] = host;
            ds_i += 1;
        }
    }
    var merge_bufs = try loaded.loadMerger(allocator, io, platform, store, shardings, progress);
    defer Merger.unloadBuffers(&merge_bufs);
    var merge_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.merger, allocator, .{ .model = merge_bufs });
    defer merge_run.deinit(allocator);
    var merged_buf: zml.Buffer = undefined;
    merge_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &merged_buf }, .opts = .{ .wait = true } });
    defer merged_buf.deinit();
    const merged = try bufferToF32(allocator, io, merged_buf);
    log.info("vision: ok merged={d} [{f}]", .{ merged.len, vision_start.untilNow(io, .awake) });
    return .{
        .merged = merged,
        .deepstack = deepstack,
        .tokens = compiled.merged,
        .grid = grid,
        .temporal = temporal,
    };
}

fn cancelVision(fut: anytype, io: std.Io) void {
    if (fut.*) |*f| {
        if (f.cancel(io)) |bufs| {
            var b = bufs;
            VisionBlock.unloadBuffers(&b);
        } else |_| {}
        fut.* = null;
    }
}

fn loadVisionBlock(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    loaded: *const LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    index: usize,
    progress: *std.Progress.Node,
    loader: *zml.io.Loader,
) !zml.Bufferized(VisionBlock) {
    return loaded.loadBlock(allocator, io, platform, store, shardings, index, progress, loader);
}

fn bufferFromF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    shape: zml.Shape,
    values: []const f32,
) !zml.Buffer {
    switch (shape.dtype()) {
        .f32 => return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(values)),
        .bf16 => {
            const tmp = try allocator.alloc(zml.floats.BFloat16, values.len);
            defer allocator.free(tmp);
            for (tmp, values) |*dst, src| dst.* = .fromF32(src);
            return zml.Buffer.fromBytes(io, platform, shape, .replicated, std.mem.sliceAsBytes(tmp));
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

fn bufferToF32(allocator: std.mem.Allocator, io: std.Io, buf: zml.Buffer) ![]f32 {
    const slice = try buf.toSliceAlloc(allocator, io);
    defer slice.free(allocator);
    switch (buf.shape().dtype()) {
        .f32 => return allocator.dupe(f32, slice.items(f32)),
        .bf16 => {
            const src = slice.items(zml.floats.BFloat16);
            const out = try allocator.alloc(f32, src.len);
            for (out, src) |*dst, v| dst.* = v.toF32();
            return out;
        },
        else => return error.UnsupportedEmbedDtype,
    }
}

fn tileTemporal(allocator: std.mem.Allocator, src: []const f32, temporal: u32, width: u32) ![]f32 {
    if (temporal <= 1) return allocator.dupe(f32, src);
    const out = try allocator.alloc(f32, src.len * temporal);
    var t: u32 = 0;
    while (t < temporal) : (t += 1) {
        @memcpy(out[t * src.len ..][0..src.len], src);
    }
    _ = width;
    return out;
}

pub fn runImage(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const @import("../runtime/pipeline.zig").VisionCompiled,
    loaded: *const LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    rgb: []const u8,
    src_h: u32,
    src_w: u32,
    progress: *std.Progress.Node,
) !EncodedVisual {
    const patched = try patchifyRgb(allocator, rgb, src_h, src_w, loaded.cfg);
    defer allocator.free(patched.patches);
    return runPatches(allocator, io, platform, compiled, loaded, store, shardings, patched.patches, patched.grid, patched.seq, 1, progress);
}

pub fn runVideo(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    compiled: *const @import("../runtime/pipeline.zig").VisionCompiled,
    loaded: *const LoadedModel,
    store: *zml.io.TensorStore,
    shardings: []const zml.Sharding,
    rgb: []const u8,
    frames: u32,
    src_h: u32,
    src_w: u32,
    progress: *std.Progress.Node,
) !EncodedVisual {
    const patched = try patchifyVideo(allocator, rgb, frames, src_h, src_w, loaded.cfg);
    defer allocator.free(patched.patches);
    return runPatches(allocator, io, platform, compiled, loaded, store, shardings, patched.patches, patched.grid, patched.seq, patched.temporal, progress);
}

pub fn applyVisionPositions(pos: []f32, start: u32, tokens: u32, grid_h: u32, grid_w: u32, temporal: u32, cursor: *f32) void {
    const rows = @max(grid_h / 2, 1);
    const cols = @max(grid_w / 2, 1);
    const time = @max(temporal, 1);
    const base = cursor.*;
    const spatial = rows * cols;
    var i: u32 = 0;
    while (i < tokens) : (i += 1) {
        const ti = if (time == 1) 0 else i / spatial;
        const rem = if (time == 1) i else i % spatial;
        const r = rem / cols;
        const c = rem % cols;
        pos[(start + i) * 3 + 0] = base + @as(f32, @floatFromInt(ti));
        pos[(start + i) * 3 + 1] = base + @as(f32, @floatFromInt(r));
        pos[(start + i) * 3 + 2] = base + @as(f32, @floatFromInt(c));
    }
    cursor.* = base + @as(f32, @floatFromInt(@max(@max(rows, cols), time)));
}

pub fn configFromRepo(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, text_hidden: i64) !Config {
    const parsed = try config_mod.parseJson(FileConfig, allocator, io, repo, "config.json");
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
