const std = @import("std");
const log = std.log;

const zml = @import("zml");
const stdx = zml.stdx;

const Acestep = struct {
    embed_tokens: zml.nn.TokenEmbedding,
    layers: []TransformerLayer,
    norm: RmsNorm,

    pub const Config = struct {
        bos_token_id: u32,
        eos_token_id: stdx.json.Union(union(enum) {
            int: u32,
            ints: []u32,
        }),
        head_dim: ?u32 = null,
        hidden_size: u32,
        num_hidden_layers: u32,
        num_attention_heads: u32,
        num_key_value_heads: u32,
        rope_theta: f32,
        max_position_embeddings: u32,
        rms_norm_eps: f32,
        hf_rope_impl: bool = true,
        tie_word_embeddings: bool = false,
        rope_scaling: zml.nn.RopeOpts.Scaling = .{ .default = .{} },
    };

    pub const Options = struct {
        seq_len: u32 = 64,
    };

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, config: Config) !Acestep {
        const layers = try allocator.alloc(TransformerLayer, config.num_hidden_layers);
        errdefer allocator.free(layers);
        for (layers, 0..) |*layer, i| {
            layer.* = try .init(@intCast(i), store.withPrefix("layers").withLayer(i), config);
        }
        return .{
            .embed_tokens = .{ .weight = store.createTensor(
                "embed_tokens.weight",
                .{ .voc, .d },
                .{ .voc = .replicated, .d = .model },
            ) },
            .layers = layers,
            .norm = .init(store.withPrefix("norm"), config),
        };
    }

    pub fn load(
        self: *const Acestep,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
        shardings: []const zml.sharding.Sharding,
    ) !zml.Bufferized(Acestep) {
        return zml.io.load(Acestep, self, allocator, io, platform, store, .{
            .shardings = shardings,
            .parallelism = 1,
            .dma_chunks = 1,
            .dma_chunk_size = 16 * 1024 * 1024,
        });
    }

    pub fn deinit(self: *const Acestep, allocator: std.mem.Allocator) void {
        allocator.free(self.layers);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Acestep)) void {
        self.embed_tokens.weight.deinit();
        for (self.layers) |*layer| {
            TransformerLayer.unloadBuffers(layer);
        }
        RmsNorm.unloadBuffers(&self.norm);
    }

    pub fn forward(
        self: Acestep,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
        rng: zml.Tensor.Rng,
    ) struct { zml.Tensor, KvCache, zml.Tensor.Rng } {
        std.log.debug("FWD : ACESTEP", .{});
        var updated_kv_cache = kv_cache;
        var output = self.embed_tokens.forward(tokens.withPartialTags(.{.s})).withPartialTags(.{.d});
        for (self.layers, 0..) |layer, i| {
            output, updated_kv_cache = layer.forward(
                output,
                token_index,
                updated_kv_cache.atLayer(i),);
        }
        output = self.norm.forward(output);
        const logits = self.embed_tokens.weight.withTags(.{ .voc, .d }).dot(output, .d);
        const next_tokens = logits.argMax(.voc).indices.squeeze(.voc);
        return .{ next_tokens.convert(tokens.dtype()).reuseBuffer(tokens), updated_kv_cache.reuseBuffer(kv_cache), rng };
    }
};

const TransformerLayer = struct {
    id: u8,
    input_norm: RmsNorm,
    att_layer: AttLayer,
    post_att_norm: RmsNorm,
    mlp_layer: MlpLayer,

    pub fn init(id_ : u8, store: zml.io.TensorStore.View, config: Acestep.Config) !TransformerLayer {
        return .{
            .id = id_,
            .input_norm = .init(store.withPrefix("input_layernorm"), config),
            .att_layer = try .init(store.withPrefix("self_attn"), config),
            .post_att_norm = .init(store.withPrefix("post_attention_layernorm"), config),
            .mlp_layer = try .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerLayer)) void {
        RmsNorm.unloadBuffers(&self.input_norm);
        AttLayer.unloadBuffers(&self.att_layer);
        RmsNorm.unloadBuffers(&self.post_att_norm);
        MlpLayer.unloadBuffers(&self.mlp_layer);
    }

    pub fn forward(self: TransformerLayer, x0: zml.Tensor, token_index: zml.Tensor, kv_cache: KvCache) struct { zml.Tensor, KvCache } {
        std.log.debug("FWD : transformer layer {d}", .{self.id});

        // Keep the residual stream replicated to avoid repeated gathers before q/k/v.
        const x0_replicated = x0.withPartitioning(.{ .d = .replicated });
        const x0_normalized = self.input_norm.forward(x0_replicated);
        const delta0, const updated_kv_cache = self.att_layer.forward(
            x0_normalized,
            token_index,
            kv_cache,
        );

        // Fully Connected
        const x1 = x0_replicated.add(delta0).withPartitioning(.{ .d = .replicated });
        const x1_normalized = self.post_att_norm.forward(x1);
        const x2 = self.mlp_layer.forward(x1_normalized).withPartitioning(.{ .d = .replicated }).add(x1).withPartitioning(.{ .d = .replicated });

        return .{ x2.reuseBuffer(x0), updated_kv_cache };
    }
};

const AttLayer = struct {
    q_proj: zml.nn.Linear,
    k_proj: zml.nn.Linear,
    v_proj: zml.nn.Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    o_proj: zml.nn.Linear,
    num_heads: i64 = undefined,
    num_kv_heads: i64 = 0,
    rope_opts: zml.nn.RopeOpts = undefined,

    pub fn init(store: zml.io.TensorStore.View, config: Acestep.Config) !AttLayer {
        var rope_scaling = config.rope_scaling;
        rope_scaling.setRopeTheta(config.rope_theta);
        return .{
            .q_proj = .init(store.createTensor("q_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .k_proj = .init(store.createTensor("k_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .v_proj = .init(store.createTensor("v_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }), null, .d),
            .o_proj = .init(store.createTensor("o_proj.weight", .{ .d_out, .d }, .{ .d = .model }), null, .d),
            .q_norm = .init(store.withPrefix("q_norm"), config),
            .k_norm = .init(store.withPrefix("k_norm"), config),
            .num_heads = @intCast(config.num_attention_heads),
            .num_kv_heads = @intCast(config.num_key_value_heads),
            .rope_opts = .{
                .layout = if (config.hf_rope_impl) .sequential else .interleaved,
                .scaling = rope_scaling,
            },
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttLayer)) void {
        self.q_proj.weight.deinit();
        self.k_proj.weight.deinit();
        self.v_proj.weight.deinit();
        self.o_proj.weight.deinit();
        RmsNorm.unloadBuffers(&self.q_norm);
        RmsNorm.unloadBuffers(&self.k_norm);
    }

    /// Self zml.attention.attention.
    ///   - If token_index is set, x is assumed to be the representation of one new token,
    /// and kv_cache will be read for the previous tokens.
    ///   - If token_index is not set, x is assumed to be the representation of all tokens
    /// since the beginning of the sequence, and kv_cache won't be read.
    /// In both case, kv_cache will be updated with the computed key and value.
    /// x: {.b, .s, .d } -> .{.b, .s, .d}
    pub fn forward(
        self: AttLayer,
        x: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: KvCache,
    ) struct { zml.Tensor, KvCache } {
        //std.log.debug("FWD : attention layer", .{});
        const num_kv_heads = if (self.num_kv_heads > 0) self.num_kv_heads else self.num_heads;

        // Make hidden state replicated once and reuse it across q/k/v projections.
        // This avoids paying gather-style collectives independently for each projection.
        const x_qkv = x.withPartitioning(.{ .d = .replicated });

        var q = self.q_proj.forward(x_qkv).splitAxis(-1, .{ .h = self.num_heads, .hd = .auto });
        var k = self.k_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        var v = self.v_proj.forward(x_qkv).splitAxis(-1, .{ .h = num_kv_heads, .hd = .auto });
        q = q.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        k = k.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });
        v = v.withPartitioning(.{ .s = .replicated, .h = .model, .hd = .replicated });

        // In self-attention, .s axis is used both for keys and queries.
        const pos_index = b: {
            const temp = zml.Tensor.arange(.{ .end = x.dim(.s) }, token_index.dtype()).withTags(.{.s}).broad(zml.Shape.init(.{ .s = x.dim(.s) }, token_index.dtype()));
            break :b temp.add(token_index.broad(temp.shape()));
        };

        q = self.q_norm.forward(q.rename(.{ .hd = .d })).rename(.{ .d = .hd });
        k = self.k_norm.forward(k.rename(.{ .hd = .d })).rename(.{ .d = .hd });
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

        const attn_output = zml.attention.attention.attention(
            q,
            k,
            v,
            token_index,
            .vanilla,
            .vanilla,
        ).withPartitioning(.{ .q = .replicated, .h = .model, .hd = .replicated });

        const attn = attn_output.merge(.{ .d = .{ .h, .hd } }).rename(.{ .q = .s });
        const delta = self.o_proj.forward(attn).rename(.{ .d_out = .d }).withPartitioning(.{ .d = .replicated });
        return .{ delta, new_kv_cache };
    }
};

const MlpLayer = struct {
    up_proj: zml.Tensor,
    gate_proj: zml.Tensor,
    down_proj: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) !MlpLayer {
        return .{
            .up_proj = store.createTensor("up_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .gate_proj = store.createTensor("gate_proj.weight", .{ .d_out, .d }, .{ .d_out = .model }),
            .down_proj = store.createTensor("down_proj.weight", .{ .d, .d_out }, .{ .d = .model }),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(MlpLayer)) void {
        self.up_proj.deinit();
        self.gate_proj.deinit();
        self.down_proj.deinit();
    }

    pub fn forward(self: MlpLayer, input: zml.Tensor) zml.Tensor {
        //std.log.debug("FWD : mlp", .{});
        const up_projection = input.dot(self.up_proj, .d);
        const gate_projection = input.dot(self.gate_proj, .d);
        const activation = gate_projection.silu().mul(up_projection);
        const output = activation.dot(self.down_proj, .d_out);
        return output;
    }
};

const RmsNorm = struct {
    weights: zml.Tensor,
    eps: f32,

    pub fn init(store: zml.io.TensorStore.View, config: Acestep.Config) RmsNorm {
        return .{
            .weights = store.createTensor("weight", .{.d_out}, null),
            .eps = config.rms_norm_eps,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RmsNorm)) void {
        self.weights.deinit();
    }

    pub fn forward(self: RmsNorm, input: zml.Tensor) zml.Tensor {
        //std.log.debug("FWD : norm", .{});
        // normalize along d axis each one of the s input tokens
        const normalized = zml.nn.rmsNorm(input, .d, self.eps);
        // scale with learned weights by broadening weights to match s input tokens
        return normalized.mul(self.weights.withTags(.{.d}).broad(input.shape()));
    }
};

pub const KvCache = struct {
    k: zml.Tensor,
    v: zml.Tensor,
    layer_index: zml.Tensor,

    pub fn init(kv_shape: zml.Shape) KvCache {
        const sharded_shape = kv_shape.withPartitioning(.{ .h = .model });

        return .{
            .k = .fromShape(sharded_shape),
            .v = .fromShape(sharded_shape),
            .layer_index = .init(.{}, .u32),
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

    pub fn keys(self: KvCache) zml.Tensor {
        return self.k.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn values(self: KvCache) zml.Tensor {
        return self.v.dynamicSlice(.{ .layer = zml.Tensor.DynSlice{ .start = self.layer_index, .len = 1 } }).squeeze(.layer);
    }

    pub fn update(self: KvCache, new_k: zml.Tensor, new_v: zml.Tensor, token_index: ?zml.Tensor) KvCache {
        const k_shape = self.k.shape().drop(.layer);
        var layer = self.layer_index;
        layer = if (token_index) |idx| layer.broad(idx.shape()) else layer;

        return if (token_index) |idx|
            .{
                .k = self.k.scatterSlices(.{ .layer = layer, .k = idx }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer, .k = idx }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .layer_index = self.layer_index,
            }
        else
            .{
                .k = self.k.scatterSlices(.{ .layer = layer }, new_k.convert(self.k.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.k),
                .v = self.v.scatterSlices(.{ .layer = layer }, new_v.convert(self.v.dtype()).transpose(k_shape), .{ .indices_are_sorted = true, .update_fn = zml.Tensor.ScatterOpts.override }).reuseBuffer(self.v),
                .layer_index = self.layer_index,
            };
    }

    pub fn atLayer(self: KvCache, layer_index: usize) KvCache {
        return .{
            .k = self.k,
            .v = self.v,
            .layer_index = zml.Tensor.scalar(layer_index, .u32),
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

pub const Shardings = struct {
    replicated: zml.sharding.Sharding,
    model: zml.sharding.Sharding,

    pub fn init(platform: *zml.Platform) !Shardings {
        const model_mesh: zml.sharding.LogicalMesh = try .init("model", .{ .model = .high_bandwidth });
        const model_sharding_strategy: zml.sharding.Strategy = try .suggest(model_mesh, platform.physical_mesh);
        return .{
            .replicated = try zml.sharding.replicatedSharding(platform),
            .model = try .initFromStrategy(platform, model_mesh, model_sharding_strategy),
        };
    }

    pub fn all(self: *const Shardings) [2]zml.sharding.Sharding {
        return .{ self.replicated, self.model };
    }
};

const AcestepParams = struct {
    prefill_tokens: zml.Tensor,
    decode_tokens: zml.Tensor,
    token_index: zml.Tensor,
    kv_cache: KvCache,
    rng: zml.Tensor.Rng,
    shardings: Shardings,
};

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

const AudioMetadata = struct {
    bpm: []const u8,
    caption: []const u8,
    duration: []const u8,
    genres: []const u8,
    keyscale: []const u8,
    language: []const u8,
    timesignature: []const u8,
    lyric: []const u8,

    const field_names = [_][]const u8{
        "bpm:",
        "caption:",
        "duration:",
        "genres:",
        "keyscale:",
        "language:",
        "timesignature:",
        "lyric:",
    };

    fn isFieldStart(line: []const u8) bool {
        const trimmed = std.mem.trimStart(u8, line, " \t");
        for (field_names) |f| {
            if (std.mem.startsWith(u8, trimmed, f)) return true;
        }
        return false;
    }

    fn extractFieldBlock(allocator: std.mem.Allocator, input: []const u8, field: []const u8) ![]const u8 {
        var it = std.mem.splitScalar(u8, input, '\n');
        var found = false;
        var start: usize = 0;
        var end: usize = input.len;
        while (it.next()) |line| {
            const trimmed = std.mem.trimStart(u8, line, " \t");
            if (!found) {
                if (std.mem.startsWith(u8, trimmed, field)) {
                    found = true;
                    // Compute start index
                    const line_offset = @intFromPtr(line.ptr) - @intFromPtr(input.ptr);
                    const trim_offset = @intFromPtr(trimmed.ptr) - @intFromPtr(line.ptr);
                    start = line_offset + trim_offset + field.len;
                }
            } else {
                if (isFieldStart(line)) {
                    end = @intFromPtr(line.ptr) - @intFromPtr(input.ptr);
                    break;
                }
            }
        }
        if (!found) return "";
        const trimmed = std.mem.trim(u8, input[start..end], " \t\r\n");
        const output: []u8 = try allocator.alloc(u8, std.mem.replacementSize(u8, trimmed, "\n  ", " "));
        _ = std.mem.replace(u8, trimmed, "\n  ", " ", output);
        return output;
    }

    pub fn initFromString(allocator: std.mem.Allocator, input: []const u8) !AudioMetadata {
        var it = std.mem.splitSequence(u8, input, "</think>");
        const trimmed = it.first()[7..];
        return .{
            .bpm = try extractFieldBlock(allocator, trimmed, "bpm:"),
            .caption = try extractFieldBlock(allocator, trimmed, "caption:"),
            .duration = try extractFieldBlock(allocator, trimmed, "duration:"),
            .genres = try extractFieldBlock(allocator, trimmed, "genres:"),
            .keyscale = try extractFieldBlock(allocator, trimmed, "keyscale:"),
            .language = try extractFieldBlock(allocator, trimmed, "language:"),
            .timesignature = try extractFieldBlock(allocator, trimmed, "timesignature:"),
            .lyric = try extractFieldBlock(allocator, trimmed, "lyric:"),
        };
    }

    pub fn deinit(self: AudioMetadata, allocator: std.mem.Allocator) void {
        allocator.free(self.bpm);
        allocator.free(self.caption);
        allocator.free(self.duration);
        allocator.free(self.genres);
        allocator.free(self.keyscale);
        allocator.free(self.language);
        allocator.free(self.timesignature);
        allocator.free(self.lyric);
    }
};

pub fn main(init: std.process.Init) !void {
    const arena = init.arena;
    const allocator = init.gpa;
    const io = init.io;

    // Auto-select platform
    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);

    // Parse program args
    const process_args = try init.minimal.args.toSlice(arena.allocator());
    const options: std.Io.Dir.OpenOptions = .{
        .access_sub_paths = true,
        .iterate = false,
        .follow_symlinks = true,
    };
    const model_repo = try std.Io.Dir.openDirAbsolute(io, process_args[1], options);
    const model_path = process_args[2];
    const tokenizer_path = process_args[3];
    //const model2_path = process_args[4];

    // Read model shapes.
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, model_path);
    defer registry.deinit();

    // Print model shapes
    const tensors: zml.safetensors.Tensors = registry.tensors;
    const data = tensors.entries;
    for (0..data.len) |i| {
        const entry = data.get(i);
        const tensor: zml.safetensors.Tensor = tensors.get(entry.key).?;
        std.log.debug("Tensor(name={s} shape={f} size={d})", .{
            tensor.name,
            tensor.shape,
            tensor.byteSize(),
        });
    }

    // Init model
    const parsed_config = try parseConfig(allocator, io, model_repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;
    const acestep_options: Acestep.Options = .{
        .seq_len = 512,
    };
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const acestep_model: Acestep = try .init(allocator, store.view().withPrefix("model"), config);
    defer acestep_model.deinit(allocator);

    // Specify shapes of input arguments
    const dtype = acestep_model.embed_tokens.weight.dtype();
    const acestep_parameters: AcestepParams = .{
        .prefill_tokens = .init(.{ .s = acestep_options.seq_len }, .u32),
        .decode_tokens = .init(.{ .s = 1 }, .u32),
        .token_index = .init(.{}, .u32),
        .kv_cache = .init(zml.Shape.init(.{
            .layer = acestep_model.layers.len,
            .k = acestep_options.seq_len,
            .h = config.num_key_value_heads,
            .hd = config.head_dim orelse @divExact(config.hidden_size, config.num_attention_heads),
        }, dtype)),
        .rng = .init(),
        .shardings = try .init(platform),
    };

    // Compile the prefill and decode models
    std.log.debug("Compile prefill/decode", .{});
    var compiled_model = try compileModel(allocator, io, platform, acestep_model, acestep_parameters);
    defer compiled_model.prefill_exe.deinit();
    defer compiled_model.decode_exe.deinit();

    // Fill the buffers with weights
    std.log.debug("Load buffers", .{});
    var acestep_buffers = try acestep_model.load(init.arena.allocator(), io, platform, &store, &acestep_parameters.shardings.all());
    var acestep_kv_cache_buffers = try acestep_parameters.kv_cache.initBuffer(io, platform, acestep_parameters.shardings.model);
    defer Acestep.unloadBuffers(&acestep_buffers);
    defer KvCache.deinitBuffer(&acestep_kv_cache_buffers);

    // Initialize tokenizer
    std.log.debug("Initialize tokenizer", .{});
    var tokenizer: zml.tokenizer.Tokenizer = try .fromFile(allocator, io, tokenizer_path);
    defer tokenizer.deinit();

    // Test on one prompt
    const prompt = "a chill guitar melody\n\ninstrumental: true";
    std.log.debug("Start inspiration, raw prompt:\n{s}", .{prompt});
    const inspi_tokens = try tokenizeInspirationPrompt(allocator, tokenizer, prompt);
    defer allocator.free(inspi_tokens);

    if (false) {
        const inspi_result = try generateText(
            allocator,
            io,
            acestep_buffers,
            compiled_model.prefill_exe,
            compiled_model.decode_exe,
            &acestep_kv_cache_buffers,
            tokenizer,
            inspi_tokens,
            config,
            acestep_options,
            0,
            platform,
            acestep_parameters.shardings.replicated,
        );
        defer allocator.free(inspi_result);
    }

    const inspi_result =
        \\<think>
        \\bpm: 80
        \\caption: A clean, melodic electric guitar plays a catchy, looping riff with a slightly
        \\  jazzy, neo-soul flavor. The tone is bright and articulate, with a touch of light
        \\  reverb adding space to the mix. The performance is relaxed and groovy, establishing
        \\  a chill and introspective mood before ending abruptly.
        \\duration: 36
        \\genres: neo-soul
        \\keyscale: E major
        \\language: unknown
        \\timesignature: 4
        \\</think>
        \\
        \\# Lyric
        \\[Instrumental]
        \\<|im_end|>
    ;
    defer allocator.free(inspi_result);

    const metadata: AudioMetadata = try .initFromString(allocator, inspi_result);
    defer metadata.deinit(allocator);

    std.log.debug("Start generation", .{});
    const gen_tokens = try tokenizeGenerationPrompt(allocator, tokenizer, metadata);
    defer allocator.free(gen_tokens);
    
    //std.log.debug("Generation tokens ids\n{any}", .{gen_tokens});

    const gen_result = try generateText(
        allocator,
        io,
        acestep_buffers,
        compiled_model.prefill_exe,
        compiled_model.decode_exe,
        &acestep_kv_cache_buffers,
        tokenizer,
        gen_tokens,
        config,
        acestep_options,
        0,
        platform,
        acestep_parameters.shardings.replicated,
    );
    defer allocator.free(gen_result);
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(Acestep.Config) {
    log.info("Loading model config", .{});
    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(Acestep.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();
    return parsed_config;
}

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, model: Acestep, parameters: AcestepParams) !CompileModelResult {
    log.info("Compiling model", .{});
    const opts: zml.module.CompilationOptions = .{
        .shardings = &parameters.shardings.all(),
    };
    // Compile the model twice, one for prefill, one for generation.
    const prefill_exe = try platform.compile(allocator, io, model, .forward, .{ parameters.prefill_tokens, parameters.token_index, parameters.kv_cache, parameters.rng }, opts);
    const decode_exe = try platform.compile(allocator, io, model, .forward, .{ parameters.decode_tokens, parameters.token_index, parameters.kv_cache, parameters.rng }, opts);
    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
}

pub fn tokenizeInspirationPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    const im_start = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    const newline = try encodeSingleToken(&encoder, "\n");
    const system_token = try encodeSingleToken(&encoder, "system");
    const user_token = try encodeSingleToken(&encoder, "user");
    const assistant_token = try encodeSingleToken(&encoder, "assistant");

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 10);

    try tokens.append(allocator, im_start);
    try tokens.append(allocator, system_token);
    try tokens.append(allocator, newline);
    try tokens.appendSlice(allocator, try encoder.encode("# Instruction\nExpand the user's input into a more detailed and specific musical description:\n\n"));
    try tokens.append(allocator, im_end);

    try tokens.append(allocator, newline);
    try tokens.append(allocator, im_start);
    try tokens.append(allocator, user_token);
    try tokens.append(allocator, newline);
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.append(allocator, im_end);
    try tokens.append(allocator, newline);

    try tokens.append(allocator, im_start);
    try tokens.append(allocator, assistant_token);
    try tokens.append(allocator, newline);

    return tokens.toOwnedSlice(allocator);
}

pub fn tokenizeGenerationPrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, metadata: AudioMetadata) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    
    var prompt: std.ArrayList(u8) = try .initCapacity(allocator, 0);
    try prompt.appendSlice(allocator, "<|im_start|>system\n");
    try prompt.appendSlice(allocator, "# Instruction\nGenerate audio semantic tokens based on the given conditions:\n\n");
    try prompt.appendSlice(allocator, "<|im_end|>\n");
    try prompt.appendSlice(allocator, "<|im_start|>user\n");
    try prompt.appendSlice(allocator, "# Caption\n");
    try prompt.appendSlice(allocator, metadata.caption);
    try prompt.appendSlice(allocator, "\n\n");
    try prompt.appendSlice(allocator, "# Lyric\n");
    try prompt.appendSlice(allocator, "[Instrumental]\n");
    try prompt.appendSlice(allocator, "<|im_end|>\n");
    try prompt.appendSlice(allocator, "<|im_start|>assistant\n");
    try prompt.appendSlice(allocator, "<think>");
    try prompt.appendSlice(allocator, "\nbpm: ");
    try prompt.appendSlice(allocator, metadata.bpm);
    try prompt.appendSlice(allocator, "\nduration: ");
    try prompt.appendSlice(allocator, metadata.duration);
    try prompt.appendSlice(allocator, "\nkeyscale: ");
    try prompt.appendSlice(allocator, metadata.keyscale);
    try prompt.appendSlice(allocator, "\nlanguage: ");
    try prompt.appendSlice(allocator, metadata.language);
    try prompt.appendSlice(allocator, "\ntimesignature: ");
    try prompt.appendSlice(allocator, metadata.timesignature);
    try prompt.appendSlice(allocator, "\n</think>\n\n");
    try prompt.appendSlice(allocator, "<|im_end|>\n\n");
    
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, 10);
    const prompt_slice = try prompt.toOwnedSlice(allocator);
    try tokens.appendSlice(allocator, try encoder.encode(prompt_slice));
    return tokens.toOwnedSlice(allocator);
}

fn encodeSingleToken(encoder: *zml.tokenizer.Tokenizer.Encoder, text: []const u8) !u32 {
    const encoded = try encoder.encode(text);
    if (encoded.len != 1) return error.InvalidTokenizerEncoding;
    return encoded[0];
}

pub fn generateText(
    allocator: std.mem.Allocator,
    io: std.Io,
    acestep_buffers: zml.Bufferized(Acestep),
    prefill_exe: zml.exe.Exe,
    decode_exe: zml.exe.Exe,
    kv_cache_buffers: *zml.Bufferized(KvCache),
    tokenizer: zml.tokenizer.Tokenizer,
    prompt_tok: []const u32,
    config: Acestep.Config,
    options: Acestep.Options,
    seed: u128,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
) ![]u8 {
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    std.log.debug("Call 5Hz model with formatted prompt\n{s}", .{try tokenizer_decoder.decode(prompt_tok)});

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, seed, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);

    var prefill_args = try prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    // prepare device buffers for the prefill tokens and their positions
    const prefill_tokens_slice: zml.Slice = try .alloc(allocator, .init(.{options.seq_len}, .u32));
    defer prefill_tokens_slice.free(allocator);
    @memcpy(prefill_tokens_slice.items(u32)[0..prompt_tok.len], prompt_tok);

    var prefill_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prefill_tokens_slice, sharding);
    defer prefill_tokens_buffer.deinit();
    var prefill_token_pos_buffer = try zml.Buffer.scalar(io, platform, 0, .u32, sharding);
    defer prefill_token_pos_buffer.deinit();

    std.log.debug("Run prefill", .{});
    prefill_args.set(.{ acestep_buffers, prefill_tokens_buffer, prefill_token_pos_buffer, kv_cache_buffers, rng_buffers });
    prefill_exe.call(prefill_args, &prefill_results);
    prefill_results.fill(.{ &prefill_tokens_buffer, kv_cache_buffers, &rng_buffers });

    try prefill_tokens_buffer.toSlice(io, prefill_tokens_slice);
    generated_token_slice.items(u32)[0] = prefill_tokens_slice.items(u32)[prompt_tok.len - 1];

    // Prepare for token-by-token generation,
    std.log.debug("Prepare decode", .{});
    var decode_args = try decode_exe.args(allocator);
    defer decode_args.deinit(allocator);
    var decode_results = try decode_exe.results(allocator);
    defer decode_results.deinit(allocator);

    // start with the token generated based on the full prompt.
    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();
    const output_tokens_len = options.seq_len - prompt_tok.len - 1;
    // One token has already been generated by the prefill.
    var num_tokens_generated: usize = 1;
    var result: std.ArrayList(u8) = try .initCapacity(allocator, 0);

    std.log.debug("Run decode", .{});
    var stdout = std.Io.File.stdout().writer(io, &.{});
    var writer: *std.Io.Writer = &stdout.interface;
    generation: for (0..output_tokens_len + 1) |i| {
        // collect and print generated sequence
        num_tokens_generated += 1;
        const generated_token = generated_token_slice.items(u32)[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try result.appendSlice(allocator, chunk);
            try writer.writeAll(chunk);
            try writer.flush();
        }

        // check for eos
        if (i == output_tokens_len) break :generation;
        switch (config.eos_token_id.value) {
            .int => |eos| if (generated_token == @as(u32, @intCast(eos))) break :generation,
            .ints => |eos_list| {
                for (eos_list) |eos| {
                    if (generated_token == @as(u32, @intCast(eos))) break :generation;
                }
            },
        }
        // current token pos needs to go into a zml.Buffer
        const token_pos_slice: zml.Slice = .init(zml.Shape.init(.{}, .u32), std.mem.sliceAsBytes(&[_]u32{@intCast(prompt_tok.len + i)}));
        var token_pos_buffer: zml.Buffer = try .fromSlice(io, platform, token_pos_slice, sharding);
        defer token_pos_buffer.deinit();
        // call to generate the next token
        decode_args.set(.{ acestep_buffers, current_token_buffer, token_pos_buffer, kv_cache_buffers, rng_buffers });
        decode_exe.call(decode_args, &decode_results);
        decode_results.fill(.{ &current_token_buffer, kv_cache_buffers, &rng_buffers });
        // extract the generated token from the buffer
        try current_token_buffer.toSlice(io, generated_token_slice);
    }
    std.log.debug("\nDone, generated {d} tokens", .{num_tokens_generated});
    return result.toOwnedSlice(allocator);
}
