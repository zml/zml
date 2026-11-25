const std = @import("std");
const log = std.log;

const zml = @import("zml");

const chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

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
    n_head: u64,
    n_embd: u64,

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
        // y = y.merge(.{ .dout = .{ .h, .hd } });
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
    h: []Block,
    ln_f: zml.nn.LayerNorm,

    pub fn init(arena: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Transformer {
        const h = try arena.alloc(Block, @intCast(config.n_layer));
        for (h, 0..) |*block, i| {
            block.* = Block.init(view.withPrefix("h").withLayer(i), config);
        }
        return .{
            .wte = .{ .weight = view.withPrefix("wte").createReplicatedTensor("weight", .{ .voc, .d }) },
            .wpe = .{ .weight = view.withPrefix("wpe").createReplicatedTensor("weight", .{ .voc, .d }) },
            .h = h,
            .ln_f = .{ .weight = view.withPrefix("ln_f").createReplicatedTensor("weight", .{.d}) },
        };
    }
};

const Gpt = struct {
    config: Config,
    transformer: Transformer,

    pub fn init(arena: std.mem.Allocator, view: zml.io.TensorStore.View, config: Config) !Gpt {
        const transformer: Transformer = try Transformer.init(arena, view.withPrefix("transformer"), config);

        return .{
            .config = config,
            .transformer = transformer,
        };
    }

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

    pub fn forwardAll(self: Gpt, idx: zml.Tensor, config: Config, active_length: zml.Tensor, rng: zml.Tensor.Rng) Output {
        const idx_tagged = idx.withTags(.{ .b, .t });
        const pos = zml.Tensor.arange(.{ .end = idx_tagged.dim(.t) }, .i64).withTags(.{.t});
        const tok_emb = self.transformer.wte.forward(idx_tagged);
        const pos_emb = self.transformer.wpe.forward(pos);

        var x = tok_emb.add(pos_emb.broad(tok_emb.shape()));
        for (self.transformer.h) |block| {
            x = block.forward(x);
        }

        x = self.transformer.ln_f.forward(x);

        const last_index = active_length.convert(.i32).addConstant(-1);
        x = x.slice(.t, .dyn(last_index, 1)); // {b, t, d} -> {b, 1, d}
        const logits = x.dot(self.transformer.wte.weight.withTags(.{ .voc, .d }), .d);
        const next_tokens, const new_rng = zml.nn.sampleTokens(
            logits,
            .{ .topk = @intCast(config.vocab_size), .temperature = 0.8 },
            rng,
        );
        return .{ .tokens = next_tokens.convert(.u32), .rng = new_rng };
    }

    pub fn generate(
        allocator: std.mem.Allocator,
        io: std.Io,
        layer: Gpt,
        platform: *const zml.Platform,
        prompt: []const u8,
        layer_weights: zml.Bufferized(Gpt),
        config: Config,
    ) !void {
        const ctx = config.block_size;
        const max_new_tokens = 512;
        const rng = zml.Tensor.Rng.init();
        var rng_buf = try rng.initBuffer(io, platform, 1337);
        defer zml.Tensor.Rng.deinitBuffer(&rng_buf);

        // Prompt + generated tokens
        //var history: [1024]i64 = undefined;
        const history = try allocator.alloc(i64, prompt.len + max_new_tokens);
        defer allocator.free(history);

        for (prompt, 0..) |c, i| {
            history[i] = @intCast(std.mem.indexOfScalar(u8, chars, c) orelse 0);
        }
        var len: usize = prompt.len;
        std.debug.print("{s}", .{prompt});

        // Returns logits for every position
        const in_tensor = zml.Tensor.init(.{ 1, ctx }, .i64);
        const active_length = zml.Tensor.init(.{}, .u32);
        const exe = try platform.compileFn(
            allocator,
            io,
            Gpt.forwardAll,
            .{ layer, in_tensor, config, active_length, rng },
            .{ .meshes = &.{} },
        );
        defer exe.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);

        var res = try exe.results(allocator);
        defer res.deinit(allocator);
        const window = try allocator.alloc(i64, ctx);
        defer allocator.free(window);

        // Autoregressive inference
        while (len < prompt.len + max_new_tokens) : (len += 1) {
            // Sliding window + padding
            const start = len -| ctx;
            @memset(window, 0);
            @memcpy(window[0 .. len - start], history[start..len]);

            var in_buffer = try zml.Buffer.fromBytes(io, platform, in_tensor.shape(), std.mem.sliceAsBytes(window));
            defer in_buffer.deinit();
            var active_length_buf = try zml.Buffer.scalar(io, platform, len - start, .u32);
            defer active_length_buf.deinit();
            args.set(.{ layer_weights, in_buffer, active_length_buf, rng_buf });

            exe.call(io, args, &res, .{});
            var out = res.get(zml.Bufferized(Output));
            defer out.tokens.deinit();

            const token = try out.tokens.getValue(u32, io);

            zml.Tensor.Rng.deinitBuffer(&rng_buf);
            rng_buf = out.rng;

            history[len] = token;
            std.debug.print("{c}", .{chars[token]});
        }
        std.debug.print("\n", .{});
    }
};

pub const Output = struct {
    tokens: zml.Tensor,
    rng: zml.Tensor.Rng,
};

const Config = struct {
    block_size: u64,
    vocab_size: u64,
    n_layer: u64,
    n_head: u64,
    n_embd: u64,
    bias: bool,
};

pub fn parseConfig(comptime T: type, allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(T) {
    const file = try dir.openFile(io, "config.json", .{});
    defer file.close(io);

    var buffer: [256]u8 = undefined;
    var file_reader = file.reader(io, &buffer);
    var reader: std.json.Reader = .init(allocator, &file_reader.interface);
    defer reader.deinit();

    return try std.json.parseFromTokenSource(T, allocator, &reader, .{ .ignore_unknown_fields = true });
}

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

/// Cross-entropy loss = (softmax(logits) - onehot(targets)) / N.
pub fn crossEntropyBackward(logits: zml.Tensor, targets: zml.Tensor) zml.Tensor {
    const l = logits.withTags(.{ .b, .t, .voc });
    const y = targets.withTags(.{ .b, .t }).convert(.i32);

    // onehot[b, t, v] = 1 if v == y[b, t], else 0
    const voc_idx = zml.Tensor.iota(l.shape(), .voc); // {b, t, voc}, value = v
    const y_full = y.broad(voc_idx.shape()); // {b, t, voc}, value = y[b, t]
    const onehot = voc_idx.cmp(.EQ, y_full).convert(l.dtype());

    const n: f32 = @floatFromInt(y.count());
    return l.softmax(.voc).sub(onehot).scale(1.0 / n);
}

fn testLoss(
    // logits <- lm_head.out.0
    // compilefn crossEntropyBackward(logits, targets)
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    opts: zml.testing.CompareOpts,
) !void {
    var logits = try loadBufferFromStore(allocator, io, platform, activation_store, "lm_head.out.0");
    defer logits.deinit();
    const logits_shape = activation_store.getShape("lm_head.out.0") orelse return;
    const logits_tensor = zml.Tensor.fromShape(logits_shape);
    const logits_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, "lm_head.out.0");

    var targets = try loadBufferFromStore(allocator, io, platform, activation_store, "y");
    defer targets.deinit();
    const targets_shape = activation_store.getShape("y") orelse return;
    const targets_tensor = zml.Tensor.fromShape(targets_shape);
    const targets_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, "y");

    const expected = try loadBufferFromStore(allocator, io, platform, activation_store, "lm_head.grad_out.0");

    const exe = try platform.compileFn(
        allocator,
        io,
        crossEntropyBackward(logits, targets),
        .{ logits_tensor, targets_tensor },
        .{ .meshes = &.{} },
    );
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ logits_buffer, targets_buffer });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(io, args, &res, .{});
    var out_result = res.get(zml.Buffer);

    defer out_result.deinit();
    try zml.testing.expectClose(io, out_result, expected, opts);
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
    const repo = try std.Io.Dir.cwd().openDir(
        io,
        "examples/nanogpt/weights/",
        .{},
    );
    defer repo.close(io);
    const parsed_config = try parseConfig(Config, allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

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

    // ======== GPT ========
    std.log.info("———— GPT ————", .{});
    const arena = init.arena.allocator(); // USE GPA
    const gpt = try Gpt.init(arena, model_store.view(), config);
    var gpt_weights = try loadWeights(
        Gpt,
        &gpt,
        arena,
        io,
        platform,
        &model_store,
    );
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Gpt, &gpt_weights); // MUST DIE

    // try testLoss(init.arena.allocator(), io, platform, activations_store.view(), .{ .absolute_tolerance = 1e-2 });
    try Gpt.generate(
        arena,
        io,
        gpt,
        platform,
        "CORIOLANUS:\nThat haz a straight stoop to his good behold! this come;\n",
        gpt_weights,
        config,
    );
}
