const std = @import("std");

const zml = @import("zml");
const common = @import("../common.zig");
const inference = @import("inference.zig");

const log = std.log.scoped(.deepseek_v4);

pub const Config = struct {
    num_hidden_layers: u16,
    rms_norm_eps: f32,
    hc_eps: f32,
    routed_scaling_factor: f32,
    num_hash_layers: u32,
    num_experts_per_tok: u32,
};

pub const Buffers = zml.Bufferized(Model);

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

// TODO: depending of the type of load the right tensors.
const FP8Linear = struct {
    scale: zml.Tensor,
    weight: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View, tag: anytype) FP8Linear {
        return .{
            .scale = store.createTensor("scale", null, .replicated),
            .weight = store.createTensor("weight", tag, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(FP8Linear)) void {
        self.scale.deinit();
        self.weight.deinit();
    }

    pub fn forward(self: FP8Linear, x: zml.Tensor) zml.Tensor {
        const fp8_min = zml.Tensor.constant(zml.DataType.f8e4m3fn.minValue()).convert(x.dtype());
        const fp8_max = zml.Tensor.constant(zml.DataType.f8e4m3fn.maxValue()).convert(x.dtype());
        const x_scaled = x.div(self.scale).clamp(fp8_min, fp8_max).convert(.f8e4m3fn);
        _ = x_scaled; // autofix

        // const y = zml.ops.scaled_dot(x_scaled, self.weight, self.scale, self.weight_scale_inv.?, .d).convert(x.dtype());
        // return y;
        return x;
    }
};

// TODO: Use union(enum) depending if it's either a SCA or HCA layer.
const Attention = struct {
    kv_norm: RmsNorm,
    q_norm: RmsNorm,
    wq_a: FP8Linear,

    pub fn init(store: zml.io.TensorStore.View, config: Config) Attention {
        return .{
            .kv_norm = .init(store.createTensor("kv_norm.weight", .{ .hd }, .replicated), config.rms_norm_eps, .hd),
            .q_norm = .init(store.createTensor("q_norm.weight", .{ .q }, .replicated), config.rms_norm_eps, .q),
            .wq_a = .init(store.withPrefix("wq_a"), .{ .hd, .d }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Attention)) void {
        RmsNorm.unloadBuffers(&self.kv_norm);
        RmsNorm.unloadBuffers(&self.q_norm);
        FP8Linear.unloadBuffers(&self.wq_a);
    }
};

const MoE = struct {
    gate: Gate,

    pub fn init(store: zml.io.TensorStore.View, config: Config, i: usize) MoE {
        return .{
            .gate = .init(store.withPrefix("gate"), config, i),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MoE)) void {
        Gate.unloadBuffers(&self.gate);
    }
};

const Gate = struct {
    const Kind = union(enum) {
        bias: zml.Tensor,
        tid2eid: zml.Tensor,
    };

    k: u32,
    kind: Kind,
    proj: LinearF32,
    scaling_factor: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Config, i: usize) Gate {
        // TODO: use `maybeTensor` instead of the id?
        const op = blk: {
            if (i < config.num_hash_layers) {
                break :blk Kind{ .tid2eid = store.createTensor("tid2eid", .{.tid, .eid}, .replicated) };
            } else {
                break :blk Kind { .bias = store.createTensor("bias", .{.expert}, .replicated) };
            }
        };

        return .{
            .k = config.num_experts_per_tok,
            .kind = op,
            .proj = .init(store.createTensor("weight", .{.expert, .d}, .replicated), null, .d),
            .scaling_factor = config.routed_scaling_factor,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Gate)) void {
        self.proj.weight.deinit();
        // switch(self.kind) {
        //     .bias => |t| t.deinit(),
        //     .tid2eid => |t| t.deinit(),
        // }
    }

    // TODO: add to `nn.zig`
    fn softplus(x: zml.Tensor) zml.Tensor {
        return x.exp().addConstant(1).log();
    }

    // TODO: Fix bias path
    pub fn forward(self: Gate, x: zml.Tensor, input_ids: zml.Tensor) struct { zml.Tensor, zml.Tensor } {
        // TODO: support `softmax`, `sigmoid`?
        const scores = softplus(self.proj.forward(x.convert(.f32))).sqrt();

        const indices = switch(self.kind) {
            // TODO: remove `renameTag`?
            .bias => |bias| scores.add(bias.broad(scores.shape())).topK(-1, self.k, .{}).indices.renameTag(.expert, .eid),
            .tid2eid => |tid2eid| tid2eid.gather(.{.tid = input_ids}, .{}),
        };

        // scores.print("scores");
        // indices.print("indices");

        // TODO: figure out why the heck gather returns `inf`
        var weights = scores.gather(.{ .expert = indices }, .{});
        // weights.print("weights");
        weights = weights.div(weights.sum(.eid));
        weights = weights.scale(self.scaling_factor);

        return .{weights, indices.convert(.i32)};
    }
};

const Layer = struct {
    attn_norm: RmsNorm,
    ffn_norm: RmsNorm,
    hc_attn_base: zml.Tensor,
    hc_attn_fn: zml.nn.Linear,
    hc_attn_scale: zml.Tensor,
    hc_ffn_base: zml.Tensor,
    hc_ffn_fn: zml.nn.Linear,
    hc_ffn_scale: zml.Tensor,
    attn: Attention,
    ffn: MoE,

    pub fn init(store: zml.io.TensorStore.View, config: Config, i: usize) Layer {
        return .{
            .attn_norm = .init(store.createTensor("attn_norm.weight", .{ .d }, .replicated), config.rms_norm_eps, .d),
            .ffn_norm = .init(store.createTensor("ffn_norm.weight", .{ .d }, .replicated), config.rms_norm_eps, .d),
            .hc_attn_base = store.createTensor("hc_attn_base", .{.b}, .replicated),
            .hc_attn_fn = .init(store.createTensor("hc_attn_fn", .{.b, .r}, .replicated), null, .b),
            .hc_attn_scale = store.createTensor("hc_attn_scale", .{.scale}, .replicated),
            .hc_ffn_base = store.createTensor("hc_ffn_base", .{.b}, .replicated),
            .hc_ffn_fn = .init(store.createTensor("hc_ffn_fn", .{.b, .r}, .replicated), null, .b),
            .hc_ffn_scale = store.createTensor("hc_ffn_scale", .{.b}, .replicated),
            .attn = .init(store.withPrefix("attn"), config),
            .ffn = .init(store.withPrefix("ffn"), config, i),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Layer)) void {
        self.attn_norm.weight.deinit();
        self.ffn_norm.weight.deinit();
        self.hc_attn_base.deinit();
        self.hc_attn_fn.weight.deinit();
        self.hc_attn_scale.deinit();
        self.hc_ffn_base.deinit();
        self.hc_ffn_fn.weight.deinit();
        self.hc_ffn_scale.deinit();
        Attention.unloadBuffers(&self.attn);
        MoE.unloadBuffers(&self.ffn);
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
            const layer_store = store.withPrefix("layers").withLayer(i);
            layer.* = .init(layer_store, config, i);
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
        allocator.free(self.layers);
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
        self.embeds.weight.deinit();

        for (self.layers) |*layer| {
            Layer.unloadBuffers(layer);
        }
        allocator.free(self.layers);

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
