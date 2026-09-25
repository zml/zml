const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const llama = @import("llama.zig");
const model = @import("llama/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: ?[]const u8 = null,
    compare_cpu: bool = false,
    transformer_only: bool = false,
    head_only: bool = false,
    seqlen: usize = 16,
    cache_seqlen: ?usize = null,
    token_offset: u32 = 0,

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
        \\   --head-only               Compare greedy output-head tokens only
        \\   --seqlen=<number>         CPU comparison sequence length (default: 16)
        \\   --cache-seqlen=<number>   Transformer KV-cache length (default: seqlen)
        \\   --token-offset=<number>   Transformer query position (default: 0)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    if ((args.transformer_only or args.head_only) and !args.compare_cpu) return error.RequiresCompareCpu;
    if (!args.compare_cpu and args.activations == null) return error.MissingActivations;

    if (args.seqlen == 0 or (args.transformer_only and args.head_only)) return error.InvalidComparisonOptions;
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    if (cache_seqlen < args.seqlen or args.token_offset > cache_seqlen - args.seqlen) return error.InvalidComparisonOptions;

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);
    if (args.compare_cpu and platform.target == .cpu) return error.CpuComparisonRequiresAccelerator;
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

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);
    defer progress.end();

    if (args.compare_cpu) {
        const cpu = try zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } });
        defer cpu.deinit(allocator, io);
        const cpu_shardings = try common.Shardings.init(cpu);
        var cpu_progress = progress.start("CPU reference weights", 1);
        defer cpu_progress.end();
        var cpu_buffers = try repo_model.loadBuffers(allocator, io, cpu, &store, &cpu_progress, cpu_shardings);
        defer repo_model.unloadBuffers(&cpu_buffers, allocator);
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
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    std.log.info("Comparing full transformer layer with CPU (query={d}, cache={d}, offset={d})", .{ args.seqlen, cache_seqlen, args.token_offset });
    var reference = try runTransformer(allocator, io, cpu, mdl, e, shape, bytes, cache_seqlen, args.token_offset);
    defer reference.hidden.deinit();
    defer model.KvCache.deinitBuffer(&reference.kv_cache);
    var result = try runTransformer(allocator, io, platform, mdl, a, shape, bytes, cache_seqlen, args.token_offset);
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
                    linear.forward(hidden, hidden.dtype()).rename(.{ .dout = .voc })
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

fn runTransformer(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, mdl: model.Model, weights: zml.Bufferized(model.TransformerLayer), shape: zml.Shape, data: []const u8, cache_seqlen: usize, token_offset: u32) !zml.Bufferized(model.TransformerLayer.Output) {
    const kv_shape = zml.Shape.init(.{ .layer = mdl.model.layers.len, .k = cache_seqlen, .h = mdl.config.num_key_value_heads, .hd = mdl.config.hidden_size / mdl.config.num_attention_heads }, .bf16);
    const kv = model.KvCache.init(kv_shape);
    const exe = try zml.FnExe(model.TransformerLayer.forward).compile(allocator, io, platform, .{ .shardings = platform.shardings.values() }, .{.{
        .layer = mdl.model.layers[0],
        .hidden = zml.Tensor.fromShape(shape),
        .token_index = zml.Tensor.init(.{}, .u32),
        .kv_cache = kv,
        .kv_cache_index = zml.Tensor.init(.{}, .u32),
        .attention_metadata = .vanilla,
        .attention_parameters = .vanilla,
    }});
    defer exe.deinit();
    var runner = try zml.FnExe(model.TransformerLayer.forward).Runner(.{.layer}).init(&exe, allocator, .{ .layer = weights });
    defer runner.deinit(allocator);
    var hidden = try zml.Buffer.fromBytes(io, platform, shape, .replicated, data);
    errdefer hidden.deinit();
    const zeros = try allocator.alloc(u8, kv_shape.byteSize());
    defer allocator.free(zeros);
    @memset(zeros, 0);
    var cache: model.KvCache.Buffer = .{
        .k = try zml.Buffer.fromBytes(io, platform, kv_shape, .replicated, zeros),
        .v = try zml.Buffer.fromBytes(io, platform, kv_shape, .replicated, zeros),
    };
    errdefer model.KvCache.deinitBuffer(&cache);
    var zero = try zml.Buffer.scalar(io, platform, 0, .u32);
    defer zero.deinit();
    var position = try zml.Buffer.scalar(io, platform, token_offset, .u32);
    defer position.deinit();
    var previous_hidden = hidden;
    runner.run(io, .{
        .inputs = .{ .hidden = hidden, .token_index = position, .kv_cache = cache, .kv_cache_index = zero, .attention_metadata = .vanilla },
        .outputs = .{ .hidden = &hidden, .kv_cache = &cache },
    });
    if (platform.target == .furiosa) previous_hidden.deinit();
    return .{ .hidden = hidden, .kv_cache = cache };
}
