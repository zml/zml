const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const llama = @import("llama.zig");
const model = @import("llama/model.zig");
const inference = @import("llama/inference.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: ?[]const u8 = null,
    compare_cpu: bool = false,
    transformer_only: bool = false,
    head_only: bool = false,
    forward_only: bool = false,
    benchmark_iterations: usize = 0,
    platform: ?zml.Target = null,
    seqlen: usize = 16,
    cache_seqlen: ?usize = null,
    token_offset: u32 = 0,
    layers: usize = 1,
    first_layer: usize = 0,

    pub const help =
        \\Use llama_tests --model=<path> --activations=<path>
        \\
        \\ Validate the LLaMA implementation against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository
        \\   --activations=<path>      Path to activation safetensors
        \\   --compare-cpu             Compare actual model layers against CPU
        \\   --transformer-only        Skip individual components with --compare-cpu
        \\   --platform=<name>         Explicit test platform (default: auto)
        \\   --forward-only            Compare the single packed-weight forward with an unpacked CPU reference
        \\   --benchmark-iterations=<n> Time three full-forward decode trials before comparison (four warmups, default: 0)
        \\   --head-only               Compare greedy output-head tokens only
        \\   --seqlen=<number>         CPU comparison sequence length (default: 16)
        \\   --cache-seqlen=<number>   Transformer KV-cache length (default: seqlen)
        \\   --token-offset=<number>   Transformer query position (default: 0)
        \\   --layers=<number>         Layers per compiled block (default: 1)
        \\   --first-layer=<number>    First layer of the compared block (default: 0)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    if ((args.transformer_only or args.head_only or args.forward_only) and !args.compare_cpu) return error.RequiresCompareCpu;
    if (!args.compare_cpu and args.activations == null) return error.MissingActivations;

    if (args.seqlen == 0 or (args.transformer_only and args.head_only)) return error.InvalidComparisonOptions;
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    if (cache_seqlen < args.seqlen or args.token_offset > cache_seqlen - args.seqlen) return error.InvalidComparisonOptions;
    if (args.benchmark_iterations > 0 and (!args.forward_only or args.seqlen != 1 or cache_seqlen < 4 or args.benchmark_iterations > cache_seqlen - 4)) return error.InvalidBenchmarkOptions;

    const platform: *zml.Platform = if (args.platform) |target| try .init(allocator, io, target, .{ .cpu = .{ .device_count = 1 } }) else try .auto(allocator, io, .{ .cpu = .{ .device_count = 1 } });
    defer platform.deinit(allocator, io);
    if (args.compare_cpu and !args.forward_only and platform.target == .cpu) return error.CpuComparisonRequiresAccelerator;
    std.log.info("Testing platform: {s}", .{@tagName(platform.target)});

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var repo_model = try llama.LoadedModel.init(allocator, io, repo, store.view(), .{});
    defer repo_model.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = args.model });
    const shardings: common.Shardings = try .init(platform);

    if (args.forward_only) {
        defer progress.end();
        try compareFullForward(allocator, io, platform, &repo_model, &store, &progress, args, shardings);
        return;
    }

    var model_buffers = try repo_model.loadUnpackedBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadUnpackedBuffers(&model_buffers, allocator);
    defer progress.end();

    if (args.compare_cpu) {
        const cpu = try zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } });
        defer cpu.deinit(allocator, io);
        const cpu_shardings = try common.Shardings.init(cpu);
        var cpu_progress = progress.start("CPU reference weights", 1);
        defer cpu_progress.end();
        var cpu_buffers = try repo_model.loadUnpackedBuffers(allocator, io, cpu, &store, &cpu_progress, cpu_shardings);
        defer repo_model.unloadUnpackedBuffers(&cpu_buffers, allocator);
        try compareCpu(allocator, io, platform, cpu, repo_model.inner, &model_buffers, &cpu_buffers, args);
    } else {
        try run(allocator, io, platform, args.activations orelse return error.MissingActivations, repo_model.inner, &model_buffers, platform.replicated_sharding);
    }
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    mdl: model.Model,
    model_buffers: *model.Buffers,
    sharding: zml.Sharding,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    try testLayer(allocator, io, platform, activation_store.view(), "embed_tokens", mdl.model.embed_tokens, model_buffers.model.embed_tokens, sharding, .{ .absolute_tolerance = 1e-3 });

    if (mdl.model.layers.len == 0) return;

    const layer = mdl.model.layers[0];
    const layer_buffers = model_buffers.model.layers[0];

    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.v_proj", layer.self_attn.v_proj, layer_buffers.self_attn.v_proj, sharding, .{ .absolute_tolerance = 1e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.q_proj", layer.self_attn.q_proj, layer_buffers.self_attn.q_proj, sharding, .{ .absolute_tolerance = 2e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.k_proj", layer.self_attn.k_proj, layer_buffers.self_attn.k_proj, sharding, .{ .absolute_tolerance = 2e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.o_proj", layer.self_attn.o_proj, layer_buffers.self_attn.o_proj, sharding, .{ .absolute_tolerance = 2e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.mlp", layer.mlp, layer_buffers.mlp, sharding, .{ .absolute_tolerance = 1e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.input_layernorm", layer.input_layernorm, layer_buffers.input_layernorm, sharding, .{ .absolute_tolerance = 1e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.post_attention_layernorm", layer.post_attention_layernorm, layer_buffers.post_attention_layernorm, sharding, .{ .absolute_tolerance = 1e-2 });
}

fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    sharding: zml.Sharding,
    opts: zml.testing.CompareOpts,
) !void {
    const in_key = try std.fmt.allocPrint(allocator, "{s}.in", .{name});
    defer allocator.free(in_key);
    const in_shape = activation_store.getShape(in_key) orelse return error.NotFound;
    var in_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, in_key, sharding);
    defer in_buffer.deinit();
    const in_tensor = zml.Tensor.fromShape(in_shape);

    const out_key = try std.fmt.allocPrint(allocator, "{s}.out", .{name});
    defer allocator.free(out_key);
    var out_buffer_expected = try loadBufferFromStore(allocator, io, platform, activation_store, out_key, sharding);
    defer out_buffer_expected.deinit();

    // `zml.nn.Linear.forward` takes an explicit output dtype; every other layer here
    // is `fn (self, Tensor) Tensor`.
    const Layer = @TypeOf(layer);
    const Call = struct {
        fn forward(l: Layer, x: zml.Tensor) zml.Tensor {
            return if (Layer == zml.nn.Linear) l.forward(x, x.dtype()) else Layer.forward(l, x);
        }
    };
    const exe = try platform.compileFn(allocator, io, Call.forward, .{ layer, in_tensor }, .{ .shardings = &.{sharding} });
    defer exe.deinit();

    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    args.set(.{ layer_weights, in_buffer });

    var res = try exe.results(allocator);
    defer res.deinit(allocator);

    exe.call(args, &res);

    var out_result = res.get(zml.Buffer);
    defer out_result.deinit();
    try zml.testing.expectClose(io, out_result, out_buffer_expected, opts);
}

fn loadBufferFromStore(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, store: zml.io.TensorStore.View, key: []const u8, sharding: zml.Sharding) !zml.Buffer {
    const shape = store.getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn compareCpu(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, cpu: *zml.Platform, mdl: model.Model, actual: *model.Buffers, expected: *model.Buffers, args: Args) !void {
    const token_ids = [_]u32{ 128000, 3923, 374, 279, 6864, 315, 9822, 30, 0, 1, 127, 1024, 8192, 32768, 65536, 128255 };
    if (!args.transformer_only and !args.head_only) try compareLayer(allocator, io, platform, cpu, "embedding", mdl.model.embed_tokens, actual.model.embed_tokens, expected.model.embed_tokens, .init(.{ .s = token_ids.len }, .u32), std.mem.sliceAsBytes(&token_ids), .exact_match);
    const shape = zml.Shape.init(.{ .s = args.seqlen, .d = mdl.config.hidden_size }, .bf16);
    const data = try allocator.alloc(u16, shape.count());
    defer allocator.free(data);
    for (data, 0..) |*bits, i| {
        const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 17 + 13) % 113)) - 56)) / 64.0;
        bits.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    const bytes = std.mem.sliceAsBytes(data);
    if (args.head_only) {
        try compareLayer(allocator, io, platform, cpu, "greedy output head", model.LmHead.init(mdl), .{
            .lm_head = actual.lm_head,
            .embed_tokens = actual.model.embed_tokens,
            .norm = actual.model.norm,
        }, .{
            .lm_head = expected.lm_head,
            .embed_tokens = expected.model.embed_tokens,
            .norm = expected.model.norm,
        }, shape, bytes, .exact_match);
        return;
    }
    const layer = mdl.model.layers[0];
    const a = actual.model.layers[0];
    const e = expected.model.layers[0];
    if (!args.transformer_only and !args.head_only) {
        try compareLayer(allocator, io, platform, cpu, "input_layernorm", layer.input_layernorm, a.input_layernorm, e.input_layernorm, shape, bytes, .{ .absolute_tolerance = 0.02, .relative_tolerance = 0.02, .minimum_close_fraction = 1 });
        inline for (.{ "q_proj", "k_proj", "v_proj", "o_proj" }) |name| {
            try compareLayer(allocator, io, platform, cpu, name, @field(layer.self_attn, name), @field(a.self_attn, name), @field(e.self_attn, name), shape, bytes, .{ .absolute_tolerance = 0.05, .relative_tolerance = 0.02, .minimum_close_fraction = 1 });
        }
        try compareLayer(allocator, io, platform, cpu, "mlp", layer.mlp, a.mlp, e.mlp, shape, bytes, .{ .absolute_tolerance = 0.1, .relative_tolerance = 0.03, .minimum_close_fraction = 1 });
    }
    if (args.layers == 0 or args.first_layer >= mdl.model.layers.len or args.layers > mdl.model.layers.len - args.first_layer) return error.InvalidLayerCount;
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    std.log.info("Transformer block layers: {} at {}", .{ args.layers, args.first_layer });
    std.log.info("Comparing full transformer layer with CPU (query={d}, cache={d}, offset={d})", .{ args.seqlen, cache_seqlen, args.token_offset });
    var reference = try runTransformer(allocator, io, cpu, mdl, expected.model.layers[args.first_layer..][0..args.layers], shape, bytes, cache_seqlen, args.token_offset, args.first_layer);
    defer reference.hidden.deinit();
    defer model.KvCache.deinitBuffer(&reference.kv_cache);
    var result = try runTransformer(allocator, io, platform, mdl, actual.model.layers[args.first_layer..][0..args.layers], shape, bytes, cache_seqlen, args.token_offset, args.first_layer);
    defer result.hidden.deinit();
    defer model.KvCache.deinitBuffer(&result.kv_cache);
    std.log.info("Comparing KV values", .{});
    try zml.testing.expectClose(io, result.kv_cache.v, reference.kv_cache.v, .{ .absolute_tolerance = 0.03, .relative_tolerance = 0.02, .minimum_close_fraction = 1 });
    std.log.info("Comparing KV keys", .{});
    try zml.testing.expectClose(io, result.kv_cache.k, reference.kv_cache.k, .{ .absolute_tolerance = 0.03, .relative_tolerance = 0.02, .minimum_close_fraction = 1 });
    std.log.info("Comparing layer hidden state", .{});
    try zml.testing.expectClose(io, result.hidden, reference.hidden, .{ .absolute_tolerance = 0.1, .relative_tolerance = 0.03, .minimum_close_fraction = 1 });
    std.log.info("PASS full transformer layer", .{});
}

fn compareLayer(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, cpu: *zml.Platform, name: []const u8, layer: anytype, actual_weights: zml.Bufferized(@TypeOf(layer)), reference_weights: zml.Bufferized(@TypeOf(layer)), shape: zml.Shape, data: []const u8, opts: zml.testing.CompareOpts) !void {
    const Layer = @TypeOf(layer);
    const Call = struct {
        fn forward(l: Layer, x: zml.Tensor) zml.Tensor {
            if (Layer == model.LmHead) {
                const hidden = l.norm.forward(x);
                const logits = if (l.lm_head) |linear|
                    linear.forward(hidden).rename(.{ .dout = .voc })
                else
                    l.embed_tokens.weight.withTags(.{ .voc, .d }).dot(hidden, .d);
                return logits.argMax(.voc).indices.squeeze(.voc);
            } else {
                return if (Layer == zml.nn.Linear) l.forward(x, x.dtype()) else Layer.forward(l, x);
            }
        }
    };
    std.log.info("Comparing {s} with CPU", .{name});
    var results: [2]zml.Buffer = undefined;
    var count: usize = 0;
    defer for (results[0..count]) |*buffer| buffer.deinit();
    for ([_]*zml.Platform{ cpu, platform }, [_]zml.Bufferized(Layer){ reference_weights, actual_weights }, 0..) |target, weights, i| {
        var input = try zml.Buffer.fromBytes(io, target, shape, .replicated, data);
        defer input.deinit();
        const exe = try target.compileFn(allocator, io, Call.forward, .{ layer, zml.Tensor.fromShape(shape) }, .{ .shardings = target.shardings.values() });
        defer exe.deinit();
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        args.set(.{ weights, input });
        var output = try exe.results(allocator);
        defer output.deinit(allocator);
        exe.call(args, &output);
        results[i] = output.get(zml.Buffer);
        count += 1;
    }
    try zml.testing.expectClose(io, results[1], results[0], opts);
    std.log.info("PASS {s}", .{name});
}

fn runTransformer(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, mdl: model.Model, weights: []const zml.Bufferized(model.TransformerLayer), shape: zml.Shape, data: []const u8, cache_seqlen: usize, token_offset: u32, first_layer: usize) !zml.Bufferized(model.TransformerLayer.Output) {
    const kv_shape = zml.Shape.init(.{ .layer = mdl.model.layers.len, .k = cache_seqlen, .h = mdl.config.num_key_value_heads, .hd = mdl.config.hidden_size / mdl.config.num_attention_heads }, .bf16);
    const kv = model.KvCache.init(kv_shape);
    const exe = try zml.FnExe(model.TransformerBlock.forward).compile(allocator, io, platform, .{ .shardings = platform.shardings.values() }, .{.{
        .layers = mdl.model.layers[first_layer..][0..weights.len],
        .hidden = zml.Tensor.fromShape(shape),
        .token_index = zml.Tensor.init(.{}, .u32),
        .kv_cache = kv,
        .kv_cache_index = zml.Tensor.init(.{}, .u32),
        .attention_metadata = .vanilla,
        .attention_parameters = .vanilla,
    }});
    defer exe.deinit();
    var runner = try zml.FnExe(model.TransformerBlock.forward).Runner(.{.layers}).init(&exe, allocator, .{ .layers = weights });
    defer runner.deinit(allocator);
    var hidden = try zml.Buffer.fromBytes(io, platform, shape, .replicated, data);
    errdefer hidden.deinit();
    // Nonzero history detects lost writes to other layers and cache positions.
    const cache_data = try allocator.alloc(u16, kv_shape.count());
    defer allocator.free(cache_data);
    for (cache_data, 0..) |*bits, i| {
        const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 7) % 17)) - 8)) / 64.0;
        bits.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    const cache_bytes = std.mem.sliceAsBytes(cache_data);
    var cache: model.KvCache.Buffer = .{
        .k = try zml.Buffer.fromBytes(io, platform, kv_shape, .replicated, cache_bytes),
        .v = try zml.Buffer.fromBytes(io, platform, kv_shape, .replicated, cache_bytes),
    };
    errdefer model.KvCache.deinitBuffer(&cache);
    var layer_index = try zml.Buffer.scalar(io, platform, first_layer, .u32);
    defer layer_index.deinit();
    var position = try zml.Buffer.scalar(io, platform, token_offset, .u32);
    defer position.deinit();
    var previous_hidden = hidden;
    runner.run(io, .{
        .inputs = .{ .hidden = hidden, .token_index = position, .kv_cache = cache, .kv_cache_index = layer_index, .attention_metadata = .vanilla },
        .outputs = .{ .hidden = &hidden, .kv_cache = &cache },
    });
    if (platform.target == .furiosa) previous_hidden.deinit();
    // Check untouched cache storage exactly, independently of the floating
    // tolerance used for newly computed keys and values.
    const query_length: usize = @intCast(shape.dim(.s));
    for ([_]zml.Buffer{ cache.k, cache.v }) |buffer| {
        const host = try buffer.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        const row_size: usize = @intCast(kv_shape.dim(.h) * kv_shape.dim(.hd));
        for (cache_data, 0..) |bits, i| {
            const row = i / row_size;
            const layer = row / cache_seqlen;
            const position_in_cache = row % cache_seqlen;
            if (layer >= first_layer and layer < first_layer + weights.len and
                position_in_cache >= token_offset and position_in_cache < @as(usize, token_offset) + query_length) continue;
            try std.testing.expectEqual(bits, std.mem.readInt(u16, host.bytes[i * 2 ..][0..2], .little));
        }
    }
    return .{ .hidden = hidden, .kv_cache = cache };
}

const ReferenceForward = struct {
    const Input = struct {
        weights: model.Model,
        tokens: zml.Tensor,
        token_index: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
    };

    fn forward(input: Input) inference.Forward.Output {
        const embedded = model.EmbedTokens.forward(.{
            .embedding = .{ .embed_tokens = input.weights.model.embed_tokens },
            .tokens = input.tokens,
        });
        const transformed = model.TransformerBlock.forward(.{
            .layers = input.weights.model.layers,
            .hidden = embedded.hidden,
            .token_index = input.token_index,
            .kv_cache = input.kv_cache,
            .kv_cache_index = zml.Tensor.scalar(0, .u32),
            .attention_metadata = .vanilla,
            .attention_parameters = .vanilla,
        });
        const sampled = model.LmHead.forward(.{
            .lm_head = model.LmHead.init(input.weights),
            .hidden = transformed.hidden,
            .tokens = input.tokens,
            .rng = input.rng,
        });
        return .{ .tokens = sampled.tokens, .kv_cache = transformed.kv_cache, .rng = sampled.rng };
    }
};

fn compareFullForward(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, mdl: *model.LoadedModel, store: *zml.io.TensorStore, progress: *std.Progress.Node, args: Args, shardings: common.Shardings) !void {
    const packing = @import("llama/packed_weights.zig");
    const cpu = try zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } });
    defer cpu.deinit(allocator, io);
    const cpu_shardings = try common.Shardings.init(cpu);
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    const kv_shape = zml.Shape.init(.{ .layer = mdl.inner.model.layers.len, .k = cache_seqlen, .h = mdl.inner.config.num_key_value_heads, .hd = mdl.inner.config.hidden_size / mdl.inner.config.num_attention_heads }, .bf16);
    const kv = model.KvCache.init(kv_shape);
    const tokens = zml.Tensor.init(.{ .s = args.seqlen }, .u32);
    const position = zml.Tensor.init(.{}, .u32);
    const rng: zml.Tensor.Rng = .init();
    std.log.info("Comparing whole forward: {} layers, {} packed weight arguments, query={}, cache={}, offset={}", .{ mdl.inner.model.layers.len, mdl.packing.weights.tensors.len, args.seqlen, cache_seqlen, args.token_offset });

    const actual_exe = try zml.FnExe(inference.Forward.forward).compile(allocator, io, platform, .{ .shardings = &shardings.all(), .program_name = "llama_full_forward_comparison" }, .{.{ .weights = mdl.packing.weights, .tokens = tokens, .token_index = position, .kv_cache = kv, .rng = rng, .attention_metadata = .vanilla, .attention_parameters = .vanilla }});
    defer actual_exe.deinit();
    const reference_exe = try zml.FnExe(ReferenceForward.forward).compile(allocator, io, cpu, .{ .shardings = &cpu_shardings.all() }, .{.{ .weights = mdl.inner, .tokens = tokens, .token_index = position, .kv_cache = kv, .rng = rng }});
    defer reference_exe.deinit();

    const cache_data = try allocator.alloc(u16, kv_shape.count());
    defer allocator.free(cache_data);
    for (cache_data, 0..) |*bits, i| {
        const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 7) % 17)) - 8)) / 64.0;
        bits.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    const token_data = try allocator.alloc(u32, args.seqlen);
    defer allocator.free(token_data);
    for (token_data, 0..) |*token, i| token.* = @intCast(1000 + i);

    var outputs: [2]zml.Bufferized(inference.Forward.Output) = undefined;
    var done: usize = 0;
    defer for (outputs[0..done]) |*out| {
        out.tokens.deinit();
        model.KvCache.deinitBuffer(&out.kv_cache);
        zml.Tensor.Rng.deinitBuffer(&out.rng);
    };
    for ([_]*zml.Platform{ cpu, platform }, 0..) |target, i| {
        var token_buffer = try zml.Buffer.fromBytes(io, target, tokens.shape(), .replicated, std.mem.sliceAsBytes(token_data));
        errdefer token_buffer.deinit();
        var cache: model.KvCache.Buffer = .{
            .k = try zml.Buffer.fromBytes(io, target, kv_shape, .replicated, std.mem.sliceAsBytes(cache_data)),
            .v = try zml.Buffer.fromBytes(io, target, kv_shape, .replicated, std.mem.sliceAsBytes(cache_data)),
        };
        errdefer model.KvCache.deinitBuffer(&cache);
        var pos = try zml.Buffer.scalar(io, target, args.token_offset, .u32);
        defer pos.deinit();
        var rng_buffer = try zml.Tensor.Rng.initBuffer(io, target, .replicated, 0);
        errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffer);
        if (i == 0) {
            var reference_weights = try mdl.loadUnpackedBuffers(allocator, io, cpu, store, progress, cpu_shardings);
            defer mdl.unloadUnpackedBuffers(&reference_weights, allocator);
            var runner = try zml.FnExe(ReferenceForward.forward).Runner(.{.weights}).init(&reference_exe, allocator, .{ .weights = reference_weights });
            defer runner.deinit(allocator);
            runner.run(io, .{ .inputs = .{ .tokens = token_buffer, .token_index = pos, .kv_cache = cache, .rng = rng_buffer }, .outputs = .{ .tokens = &token_buffer, .kv_cache = &cache, .rng = &rng_buffer } });
        } else {
            var actual_weights = try mdl.packing.load(allocator, io, platform, store, progress, &shardings.all());
            defer packing.Plan.unload(&actual_weights, allocator);
            if (args.benchmark_iterations > 0) try benchmarkFullForward(allocator, io, platform, &actual_exe, actual_weights, kv, mdl.inner.config.bos_token_id, args.benchmark_iterations);
            var runner = try zml.FnExe(inference.Forward.forward).Runner(.{.weights}).init(&actual_exe, allocator, .{ .weights = actual_weights });
            defer runner.deinit(allocator);
            runner.run(io, .{ .inputs = .{ .tokens = token_buffer, .token_index = pos, .kv_cache = cache, .rng = rng_buffer, .attention_metadata = .vanilla }, .outputs = .{ .tokens = &token_buffer, .kv_cache = &cache, .rng = &rng_buffer } });
        }
        outputs[i] = .{ .tokens = token_buffer, .kv_cache = cache, .rng = rng_buffer };
        done += 1;
    }
    if (args.seqlen == 1) std.log.info("Whole-forward argmax: device={}, CPU={}", .{ try outputs[1].tokens.getValue(u32, io), try outputs[0].tokens.getValue(u32, io) });
    try zml.testing.expectClose(io, outputs[1].tokens, outputs[0].tokens, .exact_match);
    const tolerance: zml.testing.CompareOpts = .{ .absolute_tolerance = 0.03, .relative_tolerance = 0.02, .minimum_close_fraction = 1 };
    try zml.testing.expectClose(io, outputs[1].kv_cache.k, outputs[0].kv_cache.k, tolerance);
    try zml.testing.expectClose(io, outputs[1].kv_cache.v, outputs[0].kv_cache.v, tolerance);
    for ([_]zml.Buffer{ outputs[1].kv_cache.k, outputs[1].kv_cache.v }) |buffer| {
        const host = try buffer.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        const row_size: usize = @intCast(kv_shape.dim(.h) * kv_shape.dim(.hd));
        for (cache_data, 0..) |bits, i| {
            const pos = (i / row_size) % cache_seqlen;
            if (pos >= args.token_offset and pos < args.token_offset + args.seqlen) continue;
            if (host.items(u16)[i] != bits) return error.UntouchedCacheChanged;
        }
    }
    std.log.info("PASS whole forward: exact argmax, KV tolerance, untouched cache bits", .{});
}

// Measures the complete decode executable and synchronous token readback. It
// excludes compilation, weight upload, prefill, tokenization and terminal IO.
// Each trial starts from BOS and feeds every predicted token into the next
// call, including after EOS, to keep the timed workload fixed.
fn benchmarkFullForward(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, exe: *const inference.KernelExe, weights: @import("llama/packed_weights.zig").Buffers, kv: model.KvCache, bos: u32, iterations: usize) !void {
    const warmups = 4;
    const positions = try allocator.alloc(zml.Buffer, warmups + iterations);
    defer allocator.free(positions);
    var initialized: usize = 0;
    defer for (positions[0..initialized]) |*position| position.deinit();
    for (positions, 0..) |*position, i| {
        position.* = try zml.Buffer.scalar(io, platform, i, .u32);
        initialized += 1;
    }
    var runner = try inference.KernelExe.Runner(.{.weights}).init(exe, allocator, .{ .weights = weights });
    defer runner.deinit(allocator);
    const cache_shape = kv.k.shape().withPartitioning(.{});
    const zero_cache = try allocator.alloc(u8, cache_shape.byteSize());
    defer allocator.free(zero_cache);
    @memset(zero_cache, 0);
    for (0..3) |trial| {
        var tokens = try zml.Buffer.fromBytes(io, platform, .init(.{ .s = 1 }, .u32), .replicated, std.mem.asBytes(&bos));
        defer tokens.deinit();
        var cache: model.KvCache.Buffer = .{
            .k = try zml.Buffer.fromBytes(io, platform, cache_shape, .replicated, zero_cache),
            .v = try zml.Buffer.fromBytes(io, platform, cache_shape, .replicated, zero_cache),
        };
        defer model.KvCache.deinitBuffer(&cache);
        var rng = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, 0);
        defer zml.Tensor.Rng.deinitBuffer(&rng);
        var start: std.Io.Timestamp = undefined;
        var last_token: u32 = bos;
        for (positions, 0..) |position, i| {
            if (i == warmups) start = .now(io, .awake);
            runner.run(io, .{ .inputs = .{ .tokens = tokens, .token_index = position, .kv_cache = cache, .rng = rng, .attention_metadata = .vanilla }, .outputs = .{ .tokens = &tokens, .kv_cache = &cache, .rng = &rng } });
            last_token = try tokens.getValue(u32, io);
        }
        const duration = start.untilNow(io, .awake);
        const seconds = @as(f64, @floatFromInt(duration.toNanoseconds())) / 1e9;
        std.log.info("Full-forward decode trial {}: {} tokens, {f}, {d:.2} tok/s, last token={} (correctness checked separately)", .{ trial + 1, iterations, duration, @as(f64, @floatFromInt(iterations)) / seconds, last_token });
    }
}
