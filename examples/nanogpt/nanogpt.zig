const std = @import("std");
const log = std.log;

const zml = @import("zml");

const chars = "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

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

    // ======== GPT ========
    std.log.info("———— GPT ————", .{});
    const arena = init.arena.allocator();
    const gpt = try Gpt.init(arena, model_store.view(), config);
    var gpt_weights = try loadWeights(
        Gpt,
        &gpt,
        arena,
        io,
        platform,
        &model_store,
    );
    try Gpt.generate(arena, io, gpt, platform, "CORIOLANUS:\nThat haz a straight stoop to his good behold! this come;\n", gpt_weights);
    defer zml.Buffer.freeDeviceMemoryButKeepHostMetadataMemory(Gpt, &gpt_weights);
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

    pub fn forwardAll(self: Gpt, idx: zml.Tensor) zml.Tensor {
        const idx_tagged = idx.withTags(.{ .b, .t });
        const pos = zml.Tensor.arange(.{ .end = idx_tagged.dim(.t) }, .i64).withTags(.{.t});
        const tok_emb = self.transformer.wte.forward(idx_tagged);
        const pos_emb = self.transformer.wpe.forward(pos);
        var x = tok_emb.add(pos_emb.broad(tok_emb.shape()));
        for (self.transformer.h) |block| {
            x = block.forward(x);
        }

        x = self.transformer.ln_f.forward(x);
        const logits = x.dot(self.transformer.wte.weight.withTags(.{ .voc, .d }), .d);
        return logits;
    }

    pub fn nextToken(self: Gpt, idx: zml.Tensor) zml.Tensor {
        const idx_tagged = idx.withTags(.{ .b, .t });
        const pos = zml.Tensor.arange(.{ .end = idx_tagged.dim(.t) }, .i64).withTags(.{.t});
        const tok_emb = self.transformer.wte.forward(idx_tagged);
        const pos_emb = self.transformer.wpe.forward(pos);
        var x = tok_emb.add(pos_emb.broad(tok_emb.shape()));
        for (self.transformer.h) |block| {
            x = block.forward(x);
        }

        x = self.transformer.ln_f.forward(x);
        const logits = x.dot(self.transformer.wte.weight.withTags(.{ .voc, .d }), .d);

        return logits;
    }

    pub fn generate(
        allocator: std.mem.Allocator,
        io: std.Io,
        layer: Gpt,
        platform: *const zml.Platform,
        prompt: []const u8,
        layer_weights: zml.Bufferized(Gpt),
    ) !void {
        const ctx = 256;
        const max_new_tokens = 512;

        // Prompt + generated tokens
        var history: [1024]i64 = undefined;
        for (prompt, 0..) |c, i| {
            history[i] = @intCast(std.mem.indexOfScalar(u8, chars, c) orelse 0);
        }
        var len: usize = prompt.len;
        std.debug.print("{s}", .{prompt});

        // Returns logits for every position
        const in_tensor = zml.Tensor.init(.{ 1, ctx }, .i64);
        const exe = try platform.compileFn(allocator, io, Gpt.forwardAll, .{ layer, in_tensor }, .{ .meshes = &.{} });
        defer exe.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);

        var res = try exe.results(allocator);
        defer res.deinit(allocator);

        var prng = std.Random.DefaultPrng.init(1337);
        const rng = prng.random();
        // Autoregressive inference
        while (len < prompt.len + max_new_tokens) : (len += 1) {
            // Sliding window + padding
            const start = len -| ctx;
            var window: [ctx]i64 = @splat(0);
            @memcpy(window[0 .. len - start], history[start..len]);

            var in_buffer = try zml.Buffer.fromBytes(io, platform, in_tensor.shape(), std.mem.sliceAsBytes(&window));
            defer in_buffer.deinit();
            args.set(.{ layer_weights, in_buffer });

            exe.call(io, args, &res, .{});
            var out = res.get(zml.Buffer);
            defer out.deinit();

            const host = try out.toSliceAlloc(allocator, io);
            defer host.free(allocator);

            // Take the last token row
            const pos = len - start - 1;
            const scores = host.constItems(f32)[pos * chars.len ..][0..chars.len];

            // var best: usize = 0;
            // for (scores, 0..) |score, i| {
            //     if (score > scores[best]) best = i;
            // }

            // Best + temperature
            const temperature: f32 = 0.8;

            // max
            var max: f32 = scores[0];
            for (scores) |s| max = @max(max, s);

            // softmax
            var probs: [chars.len]f32 = undefined;
            var sum: f32 = 0;
            for (scores, 0..) |s, i| {
                probs[i] = @exp((s - max) / temperature);
                sum += probs[i];
            }

            // sampling
            var r = rng.float(f32) * sum;
            var best: usize = chars.len - 1;
            for (probs, 0..) |p, i| {
                r -= p;
                if (r <= 0) {
                    best = i;
                    break;
                }
            }

            history[len] = @intCast(best);
            std.debug.print("{c}", .{chars[best]});
        }
        std.debug.print("\n", .{});
    }
};

const Config = struct {
    block_size: i64,
    vocab_size: i64,
    n_layer: i64,
    n_head: i64,
    n_embd: i64,
    bias: bool,
};

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
