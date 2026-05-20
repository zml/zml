const std = @import("std");

const zml = @import("zml");
const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.deepseek_v4);

pub const Config = struct {
    num_hidden_layers: u16,
    rms_norm_eps: f32,
    hc_eps: f32,
};

pub const Buffers = zml.Bufferized(Model);

const Layer = struct {
    pub fn init(i: usize) Layer {
        _ = i; // autofix
        return .{};
    }
};

const RmsNorm = struct {
    weight: zml.Tensor,
    eps: f32,
    tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, eps: f32, tag: anytype) RmsNorm {
        return .{
            .weight = weight,
            .eps = eps,
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn forward(self: RmsNorm, x: zml.Tensor) zml.Tensor {
        const norm = zml.nn.rmsNorm(x, self.tag, self.eps);
        return norm.mul(self.weight.broad(x.shape()));
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weight.deinit();
    }
};

pub const LinearF32 = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
    tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, bias: ?zml.Tensor, tag: anytype) LinearF32 {
        return .{
            .weight = weight,
            .bias = bias,
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn forward(self: LinearF32, x: zml.Tensor) zml.Tensor {
        var y = x.dot(self.weight.convert(.f32), self.tag);
        return if (self.bias) |bias| y.add(bias.broad(y.shape())) else y;
    }
};

const LmHead = struct {
    norm: RmsNorm,
    voc_proj: LinearF32,
    hc_base: zml.Tensor,
    hc_fn: zml.nn.Linear,
    hc_scale: zml.Tensor,
    hc_eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config) LmHead {
        return .{
            .norm = .init(store.createTensor("norm.weight", .{.d}, .replicated), config.rms_norm_eps, .d,),
            .voc_proj = .init(store.createTensor("head.weight", .{.voc, .d}, .replicated), null, .d),
            .hc_base = store.createTensor("hc_head_base", .{.hc}, .replicated),
            .hc_fn = .init(store.createTensor("hc_head_fn", .{.hc, .d}, .replicated), null, .d),
            .hc_scale = store.createTensor("hc_head_scale", .{.batch}, .replicated),
            .hc_eps = config.hc_eps
        };
    }

    fn hc_head(self: LmHead, x: zml.Tensor) zml.Tensor {
        // shape(x) = [b,s,hc,d]
        const x_ = x.merge(.{ .d = .{ .hc, .d }}).convert(.f32);

        const rsqrt = blk: {
            const variance = x_.powByConst(2).mean(.d);
            break :blk zml.Tensor.rsqrt(variance.addConstant(self.norm.eps));
        };

        var mixes = self.hc_fn.forward(x_);
        mixes = mixes.mul(rsqrt.broad(mixes.shape()));

        const pre = blk: {
            // FIX: `broad` tags.
            const s = mixes.shape();
            break :blk mixes.mul(self.hc_scale.broad(s)).add(self.hc_base.broad(s)).sigmoid().addConstant(self.hc_eps);
        };

        return x.convert(.f32).mul(pre.broad(x.shape())).sum(.hc).squeeze(.hc).convert(x.dtype());
    }

    pub fn forward(self: LmHead, x: zml.Tensor) zml.Tensor {
        // TODO: fix with start = -1
        return self.voc_proj.forward(self.norm.forward(self.hc_head(x)).convert(.f32)).slice1d(.seq, .{ .start = 10 }).squeeze(.seq);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(LmHead)) void {
        RmsNorm.unloadBuffers(&self.norm);
        self.voc_proj.weight.deinit();
        self.hc_base.deinit();
        self.hc_fn.weight.deinit();
        self.hc_scale.deinit();
    }
};

pub const Model = struct {
    embeds: zml.nn.TokenEmbedding,
    layers: []Layer,
    lm_head: LmHead, 

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Model {
        const layers = try allocator.alloc(Layer, config.num_hidden_layers);
        for (layers, 0..) |*layer, i| {
            layer.* = .init(i);
        }

        return .{
            .embeds = .{
                .weight = store.createTensor("embed.weight", .{ .voc, .d }, .replicated),
            },
            .layers = layers,
            .lm_head = .init(store, config),
        };
    }

    pub fn deinit(self: *Model, allocator: std.mem.Allocator) void {
        _ = allocator; // autofix
        _ = self; // autofix
    }

    pub fn loadBuffers(
        self: *const Model,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: []const zml.Sharding,
    ) !zml.Bufferized(Model) {
        const now: std.Io.Timestamp = .now(io, .awake);
        var total_bytes: usize = 0;
        defer {
            const took = now.untilNow(io, .awake);
            const took_ns: usize = @max(1, @as(usize, @intCast(took.toNanoseconds())));
            log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{
                total_bytes,
                took,
                total_bytes * std.time.ns_per_s / took_ns,
            });
        }

        //progress.increaseEstimatedTotalItems(store.view().count());

        return zml.io.load(Model, self, allocator, io, platform, store, .{
            .dma_chunks = 8,
            .dma_chunk_size = 128 * zml.MiB,
            .progress = progress,
            .parallelism = 16,
            .total_bytes = &total_bytes,
            .shardings = shardings,
        });
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
        _ = allocator; // autofix
        self.embeds.weight.deinit();

        // for (self.layers) |*layer| {
        //     allocator.destroy(layer);
        // }

        LmHead.unloadBuffers(&self.lm_head);
    }
};

pub const LoadedModel = struct {
    inner: Model,
    parsed_config: std.json.Parsed(Config),

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        repo: std.Io.Dir,
        store: zml.io.TensorStore.View,
        generation: common.GenerationOptions,
    ) !LoadedModel {
        _ = generation; // autofix
        
        const parsed_config = try common.parseConfig(Config, allocator, io, repo);
        errdefer parsed_config.deinit();

        return .{
            .inner = try .init(allocator, store, parsed_config.value),
            .parsed_config = parsed_config,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
        self.parsed_config.deinit();
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        progress: *std.Progress.Node,
        shardings: common.Shardings,
    ) !Buffers {
        return self.inner.loadBuffers(allocator, io, platform, store, progress, &shardings.all());
    }

    pub fn unloadBuffers(self: *const LoadedModel, buffers: *Buffers, allocator: std.mem.Allocator) void {
        _ = self; // autofix
        Model.unloadBuffers(buffers, allocator);
    }

    pub fn compile(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        backend: zml.attention.attention.Backend,
        shardings: common.Shardings,
        seqlen: usize,
        progress: *std.Progress.Node,
    ) !inference.CompiledModel {
        _ = self; // autofix
        _ = allocator; // autofix
        _ = io; // autofix
        _ = platform; // autofix
        _ = backend; // autofix
        _ = shardings; // autofix
        _ = seqlen; // autofix
        _ = progress; // autofix
        return error.NotImplemented;
    }
};
