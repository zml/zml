const std = @import("std");

const zml = @import("zml");

const buffers = @import("../core/buffers.zig");
const config_mod = @import("../core/config.zig");
const geometry = @import("../conditioning/geometry.zig");
const vision_conv = @import("vision_conv.zig");
const vision_sdpa = @import("vision_sdpa.zig");
const weights = @import("../core/weights.zig");

const log = std.log.scoped(.minimax_h3_vision);

const dump_blocks = [_]u32{ 0, 7, 8, 15, 16, 23, 24, 26 };

fn dumpEnvPath() ?[]const u8 {
    const raw = std.c.getenv("H3_LAYER_DUMP") orelse return null;
    const path = std.mem.span(raw);
    return if (path.len == 0) null else path;
}

fn dumpHostF32(io: std.Io, name: []const u8, values: []const f32, dims: []const i64) !void {
    const path = dumpEnvPath() orelse return;
    try std.Io.Dir.cwd().createDirPath(io, path);
    var dir = if (std.fs.path.isAbsolute(path))
        try std.Io.Dir.openDirAbsolute(io, path, .{})
    else
        try std.Io.Dir.cwd().openDir(io, path, .{});
    defer dir.close(io);
    var file_buf: [160]u8 = undefined;
    const file_name = try std.fmt.bufPrint(&file_buf, "{s}.f32", .{name});
    const file = try dir.createFile(io, file_name, .{});
    defer file.close(io);
    var writer = file.writer(io, &.{});
    try writer.interface.writeAll(std.mem.sliceAsBytes(values));
    var shape_buf: [128]u8 = undefined;
    var used: usize = 0;
    for (dims, 0..) |d, i| {
        const part = if (i == 0)
            try std.fmt.bufPrint(shape_buf[used..], "{d}", .{d})
        else
            try std.fmt.bufPrint(shape_buf[used..], " {d}", .{d});
        used += part.len;
    }
    const shape_name = try std.fmt.bufPrint(&file_buf, "{s}.shape", .{name});
    const shape_file = try dir.createFile(io, shape_name, .{});
    defer shape_file.close(io);
    var shape_writer = shape_file.writer(io, &.{});
    try shape_writer.interface.writeAll(shape_buf[0..used]);
}

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

fn weightRank(store: zml.io.TensorStore.View, weight_name: []const u8) u8 {
    var buffer: [256]u8 = undefined;
    const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", weight_name }) catch return 2;
    return if (store.store.getShape(key)) |s| s.rank() else 2;
}

fn linear(store: zml.io.TensorStore.View, weight_name: []const u8, bias_name: ?[]const u8) zml.nn.Linear {
    if (weightRank(store, weight_name) != 5)
        return .fromStore(store, weight_name, bias_name, .replicated, .replicated, .d);
    var layer: zml.nn.Linear = .init(
        store.createTensor(weight_name, .{ .dout, .d, .kt, .kh, .kw }, .replicated),
        if (bias_name) |n| store.maybeCreateTensor(n, .{.dout}, .replicated) else null,
        .d,
    );
    layer.attachQuant(store, weight_name);
    return layer;
}

fn asLinear(lin: zml.nn.Linear, x: zml.Tensor) zml.Tensor {
    var out = lin;
    if (out.weight.rank() == 5) {
        out.weight = out.weight.merge(.{ .d = .{ .d, .kt, .kh, .kw } });
    } else {
        while (out.weight.rank() > 2) out.weight = out.weight.squeeze(-1);
    }
    out.weight = out.weight.withTags(.{ .dout, .d });
    return out.forward(x.convert(out.weight.dtype()));
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

/// Official merger `nn.GELU()` is erf, not `gelu_pytorch_tanh`.
fn geluErf(x: zml.Tensor) zml.Tensor {
    const x_f = x.convert(.f32);
    const z = x_f.scale(std.math.sqrt(0.5));
    return x_f.mul(erfApprox(z).addConstant(1)).scale(0.5).convert(x.dtype());
}

fn erfApprox(x: zml.Tensor) zml.Tensor {
    const x_f = x.convert(.f32);
    const ax = x_f.abs();
    const t = ax.scale(0.3275911).addConstant(1).powByConst(-1);
    var poly = t.scale(1.061405429).addConstant(-1.453152027);
    poly = t.mul(poly).addConstant(1.421413741);
    poly = t.mul(poly).addConstant(-0.284496736);
    poly = t.mul(poly).addConstant(0.254829592);
    poly = t.mul(poly);
    const erfc = poly.mul(ax.mul(ax).negate().exp());
    return x_f.sign().mul(erfc.negate().addConstant(1));
}

fn applyRotary(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    const x_f = x.convert(.f32);
    const half = @divExact(x_f.dim(-1), 2);
    const x1 = x_f.slice1d(-1, .{ .start = 0, .end = half });
    const x2 = x_f.slice1d(-1, .{ .start = half, .end = x_f.dim(-1) });
    const rotated = zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
    const c = cos.convert(.f32).broad(x_f.shape());
    const s = sin.convert(.f32).broad(x_f.shape());
    return x_f.mul(c).add(rotated.mul(s)).convert(x.dtype());
}

/// Official visual is Blackwell flash SDPA at native head_dim=72. FA2 pad-96 and
/// full-seq f32 softmax both seed merger row 1189 from the current embed.
fn visionAttn(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, head_dim: i64) zml.Tensor {
    const scale: f32 = 1.0 / std.math.sqrt(@as(f32, @floatFromInt(head_dim)));
    switch (q.dtype()) {
        .bf16, .f16 => return vision_sdpa.forward(q, k, v, scale),
        else => {
            const scores = q.dot(k, .hd).scale(scale).convert(.f32).softmax(.k).convert(q.dtype());
            return scores.dot(v, .k).transpose(q.shape());
        },
    }
}

pub const AttnProbeInput = struct {
    q: zml.Tensor,
    k: zml.Tensor,
    v: zml.Tensor,
};
pub const AttnProbeOutput = struct { o: zml.Tensor };

pub fn probeAttn(input: AttnProbeInput) AttnProbeOutput {
    return .{ .o = visionAttn(input.q, input.k, input.v, input.q.dim(.hd)) };
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
        zml.nn.Linear.unloadBuffers(&self.qkv);
        zml.nn.Linear.unloadBuffers(&self.proj);
        LayerNorm.unloadBuffers(&self.norm2);
        zml.nn.Linear.unloadBuffers(&self.fc1);
        zml.nn.Linear.unloadBuffers(&self.fc2);
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
        // Official `gelu_pytorch_tanh` is f32. bf16 `x³` loses bits over 27 layers.
        const ff = asLinear(self.fc2, h.convert(.f32).gelu().convert(h.dtype()).rename(.{ .dout = .d })).rename(.{ .dout = .d });
        return .{ .hidden = x1.add(ff).reuseBuffer(input.hidden) };
    }
};

pub const EmbedModel = struct {
    proj: zml.nn.Linear,

    pub fn unloadBuffers(self: *zml.Bufferized(EmbedModel)) void {
        zml.nn.Linear.unloadBuffers(&self.proj);
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
        zml.nn.Linear.unloadBuffers(&self.fc1);
        zml.nn.Linear.unloadBuffers(&self.fc2);
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
        x = asLinear(self.fc2, geluErf(asLinear(self.fc1, x)).rename(.{ .dout = .d })).rename(.{ .dout = .d });
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

pub fn register(platform: *const zml.Platform) !void {
    try vision_conv.register(platform);
    try vision_sdpa.register(platform);
}

pub fn embed(input: EmbedInput) EmbedOutput {
    const tokens = vision_conv.forward(input.model.proj, input.patches.withPartialTags(.{ .b, .s, .d }));
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

pub const WeightCache = struct {
    embed: zml.Bufferized(EmbedModel),
    pos: zml.Buffer,
    blocks: []zml.Bufferized(VisionBlock),
    merger: zml.Bufferized(Merger),
    deepstack: [3]zml.Bufferized(Merger),

    pub fn deinit(self: *WeightCache, allocator: std.mem.Allocator) void {
        EmbedModel.unloadBuffers(&self.embed);
        self.pos.deinit();
        for (self.blocks) |*block| VisionBlock.unloadBuffers(block);
        allocator.free(self.blocks);
        Merger.unloadBuffers(&self.merger);
        for (&self.deepstack) |*m| Merger.unloadBuffers(m);
    }

    pub fn load(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        loaded: *const LoadedModel,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !WeightCache {
        var embed_bufs = try loaded.loadEmbed(allocator, io, platform, store, shardings, progress);
        errdefer EmbedModel.unloadBuffers(&embed_bufs);
        var pos = try loaded.loadPosEmbed(allocator, io, platform, store, shardings, progress);
        errdefer pos.deinit();
        const blocks = try allocator.alloc(zml.Bufferized(VisionBlock), loaded.inner.blocks.len);
        errdefer allocator.free(blocks);
        var filled: usize = 0;
        errdefer {
            for (blocks[0..filled]) |*block| VisionBlock.unloadBuffers(block);
        }
        for (blocks, 0..) |*block, i| {
            block.* = try loaded.loadBlock(allocator, io, platform, store, shardings, i, progress, null);
            filled += 1;
        }
        var merger = try loaded.loadMerger(allocator, io, platform, store, shardings, progress);
        errdefer Merger.unloadBuffers(&merger);
        var deepstack: [3]zml.Bufferized(Merger) = undefined;
        var ds_filled: usize = 0;
        errdefer {
            for (deepstack[0..ds_filled]) |*m| Merger.unloadBuffers(m);
        }
        for (&deepstack, 0..) |*m, i| {
            m.* = try loaded.loadDeepstack(allocator, io, platform, store, shardings, i, progress);
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

/// Official Qwen2VL `smart_resize` (f64, Python 3 even ties).
pub fn chooseGrid(cfg: Config, src_h: u32, src_w: u32, video: bool) struct { h: u32, w: u32 } {
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

pub fn patchifyRgb(allocator: std.mem.Allocator, rgb: []const u8, src_h: u32, src_w: u32, cfg: Config) !struct { patches: []f32, grid: Grid, seq: u32 } {
    const size = chooseGrid(cfg, src_h, src_w, false);
    const resized = try geometry.resizeBicubic(allocator, rgb, src_w, src_h, size.w, size.h);
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
        const frame = try geometry.resizeBicubic(allocator, rgb[src_f * plane ..][0..plane], src_w, src_h, size.w, size.h);
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

/// Official `Qwen3VLVisionRotaryEmbedding` CPU f32 `inv_freq` for head_dim=72.
/// `std.math.pow(f32)` differs by 1 ulp and splits merger rows.
const official_inv_freq_hd72 = [_]u32{
    0x3f800000, 0x3f1977cc, 0x3eb800d6, 0x3e5c9d35, 0x3e044133, 0x3d9e91b6,
    0x3d3e1e95, 0x3ce3f280, 0x3c88a69b, 0x3c23d70a, 0x3bc47060, 0x3b6b8631,
    0x3b0d3169, 0x3aa94938, 0x3a4af7f3, 0x39f35a5c, 0x3991e2e1, 0x392ee9bf,
};

fn visionInvFreq(i: u32, half: u32) f32 {
    if (half == 36 and i < official_inv_freq_hd72.len) return @bitCast(official_inv_freq_hd72[i]);
    return 1.0 / std.math.pow(f32, 10000.0, @as(f32, @floatFromInt(i * 2)) / @as(f32, @floatFromInt(half)));
}

pub fn visionRope(allocator: std.mem.Allocator, gh: u32, gw: u32, head_dim: u32) !struct { cos: []f32, sin: []f32 } {
    const seq = gh * gw;
    const half = head_dim / 2;
    const n_freq = half / 2;
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
                    var i: u32 = 0;
                    while (i < n_freq) : (i += 1) {
                        // Official encoder keeps rotary `inv_freq` in f32. `visual.to(bf16)`
                        // (dump_vision) casts it and was the IMAGE_PAD vel gap.
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

    const table_host = try buffers.toF32(allocator, io, cache.pos);
    defer allocator.free(table_host);
    const side: u32 = @intFromFloat(@sqrt(@as(f32, @floatFromInt(loaded.cfg.num_position_embeddings))));
    const spatial_pos = try interpolatePos(allocator, table_host, side, @intCast(loaded.cfg.hidden_size), grid.h, grid.w);
    defer allocator.free(spatial_pos);
    const spatial_rope = try visionRope(allocator, grid.h, grid.w, @intCast(loaded.cfg.headDim()));
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
    if (dumpEnvPath() != null) {
        try dumpHostF32(io, "vision_patches", patches, &.{ @intCast(seq), @intCast(loaded.cfg.patchIn()) });
        try dumpHostF32(io, "vision_pos", pos, &.{ @intCast(seq), @intCast(loaded.cfg.hidden_size) });
        try dumpHostF32(io, "vision_rope_cos", rope_cos, &.{ @intCast(seq), @intCast(loaded.cfg.headDim()) });
        try dumpHostF32(io, "vision_rope_sin", rope_sin, &.{ @intCast(seq), @intCast(loaded.cfg.headDim()) });
        const embed_host = try buffers.toF32(allocator, io, hidden);
        defer allocator.free(embed_host);
        try dumpHostF32(io, "vision_embed", embed_host, &.{ @intCast(seq), @intCast(loaded.cfg.hidden_size) });
    }

    var cos_buf = try buffers.fromF32(allocator, io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, .f32), rope_cos);
    defer cos_buf.deinit();
    var sin_buf = try buffers.fromF32(allocator, io, platform, .init(.{ .s = seq, .hd = loaded.cfg.headDim() }, .f32), rope_sin);
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
        if (block_runner) |*r| {
            weights.rebake(r, .{ .layer = cache.blocks[block_i] });
        } else {
            block_runner = try BlockRunner.init(&compiled.block, allocator, .{ .layer = cache.blocks[block_i] });
        }
        var next: zml.Buffer = undefined;
        block_runner.?.run(io, .{
            .inputs = .{ .hidden = hidden, .cos = cos_buf, .sin = sin_buf },
            .outputs = .{ .hidden = &next },
            .opts = .{ .wait = true },
        });
        hidden.deinit();
        hidden = next;
        if (dumpEnvPath() != null) {
            for (dump_blocks) |want| {
                if (block_i != want) continue;
                const host = try buffers.toF32(allocator, io, hidden);
                defer allocator.free(host);
                var name_buf: [32]u8 = undefined;
                const name = try std.fmt.bufPrint(&name_buf, "vision_block_{d}", .{block_i});
                try dumpHostF32(io, name, host, &.{ @intCast(seq), @intCast(loaded.cfg.hidden_size) });
            }
        }
        if (ds_i < 3 and @as(i64, @intCast(block_i)) == loaded.cfg.deepstack_visual_indexes[ds_i]) {
            var ds_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.deepstack, allocator, .{ .model = cache.deepstack[ds_i] });
            defer ds_run.deinit(allocator);
            var tokens: zml.Buffer = undefined;
            ds_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &tokens }, .opts = .{ .wait = true } });
            defer tokens.deinit();
            deepstack[ds_i] = try buffers.toF32(allocator, io, tokens);
            ds_i += 1;
        }
    }
    var merge_run = try zml.FnExe(Merger.forward).Runner(.{.model}).init(&compiled.merger, allocator, .{ .model = cache.merger });
    defer merge_run.deinit(allocator);
    var merged_buf: zml.Buffer = undefined;
    merge_run.run(io, .{ .inputs = .{ .hidden = hidden }, .outputs = .{ .tokens = &merged_buf }, .opts = .{ .wait = true } });
    defer merged_buf.deinit();
    if (dumpEnvPath() != null) {
        const hidden_host = try buffers.toF32(allocator, io, hidden);
        defer allocator.free(hidden_host);
        try dumpHostF32(io, "vision_hidden", hidden_host, &.{ @intCast(seq), @intCast(loaded.cfg.hidden_size) });
    }
    const merged = try buffers.toF32(allocator, io, merged_buf);
    log.info("vision: ok merged={d} [{f}]", .{ merged.len, vision_start.untilNow(io, .awake) });
    return .{
        .merged = merged,
        .deepstack = deepstack,
        .tokens = compiled.merged,
        .grid = grid,
        .temporal = temporal,
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
