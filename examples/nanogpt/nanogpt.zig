const std = @import("std");
const log = std.log;

const zml = @import("zml");

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    const args = try init.minimal.args.toSlice(allocator);
    defer allocator.free(args);
    if (args.len != 3) {
        log.err("Usage: {s} weights.safetensors activations.safetensors", .{args[0]});
        return error.InvalidArguments;
    }
    const model_path, const activations_path = args[1..3].*;

    var activations_registry = try zml.safetensors.TensorRegistry.fromPath(
        allocator,
        io,
        activations_path,
    );
    defer activations_registry.deinit();

    var model_registry = try zml.safetensors.TensorRegistry.fromPath(
        allocator,
        io,
        model_path,
    );
    defer model_registry.deinit();

    var model_store: zml.io.TensorStore = .fromRegistry(
        allocator,
        &model_registry,
        &.{},
    );
    defer model_store.deinit();

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    var activations_store: zml.io.TensorStore = .fromRegistry(
        allocator,
        &activations_registry,
        &.{},
    );
    defer activations_store.deinit();

    // GPT hypers
    const config: Config = .{
        .block_size = 256,
        .vocab_size = 65,
        .n_layer = 6,
        .n_head = 6,
        .n_embd = 384,
        .bias = false,
    };

    // ======== MLP ========
    const mlp_view = model_store.view().withPrefix("transformer.h.0.mlp");
    const mlp: Mlp = Mlp.init(mlp_view);
    var mlp_weights = try loadWeights(
        Mlp,
        &mlp,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    defer zml.Buffer.deinitAll(Mlp, &mlp_weights);
    std.log.info("———— MLP ————", .{});
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.h.0.mlp",
        "transformer.h.0.mlp",
        mlp,
        mlp_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    // ======== LayerNorm ========
    std.log.info("———— LayerNorm ————", .{});

    const ln_1_view = model_store.view().withPrefix("transformer.h.0.ln_1");
    const ln_1: zml.nn.LayerNorm = .{
        .weight = ln_1_view.createReplicatedTensor("weight", .{.d}),
    };
    const ln_1_weights = try loadWeights(
        zml.nn.LayerNorm,
        &ln_1,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.h.0.ln_1",
        "transformer.h.0.ln_1",
        ln_1,
        ln_1_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    const ln_2_view = model_store.view().withPrefix("transformer.h.0.ln_2");
    const ln_2: zml.nn.LayerNorm = .{
        .weight = ln_2_view.createReplicatedTensor("weight", .{.d}),
    };
    const ln_2_weights = try loadWeights(
        zml.nn.LayerNorm,
        &ln_2,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.h.0.ln_2",
        "transformer.h.0.ln_2",
        ln_2,
        ln_2_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    // ======== Embeddings ========
    std.log.info("———— Embeddings ————", .{});
    // proj embed
    const wpe_view = model_store.view().withPrefix("transformer.wpe");
    const wpe: zml.nn.TokenEmbedding = .{
        .weight = wpe_view.createReplicatedTensor("weight", .{ .voc, .d }),
    };
    const wpe_weights = try loadWeights(
        zml.nn.TokenEmbedding,
        &wpe,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.wpe",
        "transformer.wpe",
        wpe,
        wpe_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    // token embed
    const wte_view = model_store.view().withPrefix("transformer.wte");
    const wte: zml.nn.TokenEmbedding = .{
        .weight = wte_view.createReplicatedTensor("weight", .{ .voc, .d }),
    };
    const wte_weights = try loadWeights(
        zml.nn.TokenEmbedding,
        &wte,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.wte",
        "transformer.wte",
        wte,
        wte_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    // ======== Attention ========
    std.log.info("———— Attention ————", .{});
    const attn_view = model_store.view().withPrefix("transformer.h.0.attn");
    const attn: Attn = Attn.init(attn_view, config);
    var attn_weights = try loadWeights(
        Attn,
        &attn,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    defer zml.Buffer.deinitAll(Attn, &attn_weights);
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.h.0.attn",
        "transformer.h.0.attn",
        attn,
        attn_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    // ======== Block ========
    std.log.info("———— Block ————", .{});
    //const block: Block = Block.init(transformer_view, config);
    const h = try init.arena.allocator().alloc(Block, config.n_layer);
    for (h, 0..) |*block, i| {
        block.* = Block.init(
            model_store.view().withPrefix("transformer.h").withLayer(i),
            config,
        );
    }
    const block = h[0];
    const block_weights = try loadWeights(
        Block,
        &block,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.h.0",
        "transformer.h.0",
        block,
        block_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );

    // ======== GPT ========
    std.log.info("———— GPT ————", .{});
    const ln_f_view = model_store.view().withPrefix("transformer.ln_f");
    const ln_f: zml.nn.LayerNorm = .{
        .weight = ln_f_view.createReplicatedTensor("weight", .{.d}),
    };
    const transformer: Transformer = .{
        .wte = wte,
        .wpe = wpe,
        .h = h,
        .ln_f = ln_f,
    };
    const gpt: Gpt = .{ .config = config, .transformer = transformer };
    var gpt_weights = try loadWeights(
        Gpt,
        &gpt,
        init.arena.allocator(),
        io,
        platform,
        &model_store,
    );
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Gpt, &gpt_weights);
    try testLayer(
        allocator,
        io,
        platform,
        activations_store.view(),
        "transformer.wte",
        "lm_head",
        gpt,
        gpt_weights,
        &.{},
        .{ .absolute_tolerance = 1e-2 },
    );
}

/// === TYPES ===
const Mlp = struct {
    c_fc: zml.nn.Linear,
    c_proj: zml.nn.Linear,

    pub fn init(view: zml.io.TensorStore.View) Mlp {
        return .{
            .c_fc = .init(
                view.createReplicatedTensor("c_fc.weight", .{ .dout, .d }),
                view.maybeCreateTensor("c_fc.bias", .{.dout}, .replicated, .{}),
                .d,
            ),
            .c_proj = .init(
                view.createReplicatedTensor("c_proj.weight", .{ .d, .dout }),
                view.maybeCreateTensor("c_proj.bias", .{.d}, .replicated, .{}),
                .dout,
            ),
        };
    }

    pub fn forward(self: Mlp, input: zml.Tensor) zml.Tensor {
        const x = if (input.shape().isFullyTagged()) input else input.withPartialTags(.{.d});
        const fc = self.c_fc.forward(x, x.dtype());
        var output = fc.gelu();
        // Omit dropout as it doesn't exist in ZML
        return self.c_proj.forward(output, output.dtype());
    }
};

const Attn = struct {
    c_attn: zml.nn.Linear,
    c_proj: zml.nn.Linear,
    n_head: i64,
    n_embd: i64,

    pub fn init(view: zml.io.TensorStore.View, config: Config) Attn {
        return .{
            .c_attn = .init(view.createReplicatedTensor("c_attn.weight", .{ .dout, .d }), null, .d),
            .c_proj = .init(view.createReplicatedTensor("c_proj.weight", .{ .d, .dout }), null, .dout),
            .n_head = config.n_head,
            .n_embd = config.n_embd,
        };
    }

    pub fn forward(self: Attn, input: zml.Tensor) zml.Tensor {
        // B, T, D
        const x = input.withTags(.{ .b, .t, .d });

        // Q, K, V
        const qkv = self.c_attn.forward(x, zml.Tensor.dtype(x));
        var q, var k, var v = qkv.chunkExact(.dout, 3);
        q = q.splitAxis(.dout, .{ .h = self.n_head, .hd = .auto }).rename(.{ .t = .q });
        k = k.splitAxis(.dout, .{ .h = self.n_head, .hd = .auto }).rename(.{ .t = .k });
        v = v.splitAxis(.dout, .{ .h = self.n_head, .hd = .auto }).rename(.{ .t = .k }); // Same axis as k

        // Self-attention
        var att = q.dot(k, .hd);
        const hd: f32 = @floatFromInt(k.dim(.hd));
        att = att.scale(1.0 / @sqrt(hd));
        const mask = zml.nn.causalAttnMask(.{ .q = att.dim(.q), .k = att.dim(.k) }, att.dtype(), null);
        att = att.add(mask.broad(att.shape()));
        att = att.softmax(.k);
        var y = att.dot(v, .k);
        // y = y.merge(.{ .dout = .{ .h, .hd } });,02
        y = y.mergeTranspose(.{ .h, .hd }, .dout); // DAMNIT

        return self.c_proj.forward(y, y.dtype()).rename(.{ .q = .t });
    }
};

const Block = struct {
    ln_1: zml.nn.LayerNorm,
    attn: Attn,
    ln_2: zml.nn.LayerNorm,
    mlp: Mlp,

    pub fn init(view: zml.io.TensorStore.View, config: Config) Block {
        return .{
            .ln_1 = .{ .weight = view.withPrefix("ln_1").createReplicatedTensor("weight", .{.d}) },
            .attn = Attn.init(view.withPrefix("attn"), config),
            .ln_2 = .{ .weight = view.withPrefix("ln_2").createReplicatedTensor("weight", .{.d}) },
            .mlp = Mlp.init(view.withPrefix("mlp")),
        };
    }

    pub fn forward(self: Block, input: zml.Tensor) zml.Tensor {
        const input_withtags = input.withTags(.{ .b, .t, .d });
        var x = self.ln_1.forward(input_withtags);
        x = input_withtags.add(self.attn.forward(x));
        return x.add(self.mlp.forward((self.ln_2.forward((x)))));
    }
};

const Transformer = struct {
    wte: zml.nn.TokenEmbedding,
    wpe: zml.nn.TokenEmbedding,
    //h: [n_layer]Block,
    h: []Block,
    ln_f: zml.nn.LayerNorm,
};

const Gpt = struct {
    config: Config,
    transformer: Transformer,

    pub fn forward(self: Gpt, idx: zml.Tensor) zml.Tensor {
        const idx_tagged = idx.withTags(.{ .b, .t });
        const pos = zml.Tensor.arange(.{ .end = idx_tagged.dim(.t) }, .i64).withTags(.{.t});
        const tok_emb = self.transformer.wte.forward(idx_tagged);
        const pos_emb = self.transformer.wpe.forward(pos);
        var x = tok_emb.add(pos_emb.broad(tok_emb.shape()));
        for (self.transformer.h) |block| {
            x = block.forward(x);
        }

        x = self.transformer.ln_f.forward(x);
        const last = x.slice(.t, .{ .start = x.dim(.t) - 1 });
        const logits = last.dot(self.transformer.wte.weight.withTags(.{ .voc, .d }), .d);

        return logits;
    }
};

const Config = struct { block_size: i64, vocab_size: i64, n_layer: i64, n_head: i64, n_embd: i64, bias: bool };

pub fn loadWeights(
    comptime T: type,
    model: *const T,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
) !zml.Bufferized(T) {
    var buffers = try zml.mem.bufferize(allocator, T, model);
    //errdefer zml.Buffer.deinitAll(T, &buffers);
    errdefer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(T, &buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .default);
    defer loader.deinit();

    try loader.load(io, T, model, &buffers, store, .{});
    try loader.await(io);

    return buffers;
}

fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    in_name: []const u8,
    out_name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    meshes: []const *const zml.Mesh,
    opts: zml.testing.CompareOpts,
) !void {
    const in_key = try std.fmt.allocPrint(allocator, "{s}.in.0", .{in_name});
    defer allocator.free(in_key);
    const in_shape = activation_store.getShape(in_key) orelse {
        log.err("key not found: {s}", .{in_key});
        return error.NotFound;
    };
    var in_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, in_key);
    defer in_buffer.deinit();
    const in_tensor = zml.Tensor.fromShape(in_shape);

    const out_key = try std.fmt.allocPrint(allocator, "{s}.out.0", .{out_name});
    defer allocator.free(out_key);
    var out_buffer_expected = try loadBufferFromStore(allocator, io, platform, activation_store, out_key);
    defer out_buffer_expected.deinit();

    // `zml.nn.Linear.forward` takes an explicit output dtype; every other layer here
    // is `fn (self, Tensor) Tensor`.
    const Layer = @TypeOf(layer);
    const Call = struct {
        fn forward(l: Layer, x: zml.Tensor) zml.Tensor {
            return if (Layer == zml.nn.Linear) l.forward(x, x.dtype()) else Layer.forward(l, x);
        }
    };
    const exe = try platform.compileFn(allocator, io, Call.forward, .{ layer, in_tensor }, .{ .meshes = meshes });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ layer_weights, in_buffer });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(io, args, &res, .{});

    var out_result = res.get(zml.Buffer);
    defer out_result.deinit();
    try zml.testing.expectClose(io, out_result, out_buffer_expected, opts);
}

fn loadBufferFromStore(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: zml.io.TensorStore.View, key: []const u8) !zml.Buffer {
    const shape = store.getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, host_bytes);
}
