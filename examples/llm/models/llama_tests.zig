const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const llama = @import("llama.zig");
const model = @import("llama/model.zig");
const inference = @import("llama/inference.zig");
const llama_session = @import("llama/session.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: ?[]const u8 = null,
    compare_cpu: bool = false,
    transformer_only: bool = false,
    layerwise: bool = false,
    layerwise_stages: bool = false,
    head_only: bool = false,
    forward_only: bool = false,
    session_only: bool = false,
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
        \\   --layerwise               Diagnose per-layer and accumulated errors from token 1000
        \\   --layerwise-stages        Also expose attention/MLP stages with --layerwise
        \\   --platform=<name>         Explicit test platform (default: auto)
        \\   --session-only            Check the CPU session feedback loop against a causal full-sequence forward
        \\   --forward-only            Compare the complete forward with an independent CPU reference
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
    if (args.layerwise and (!args.compare_cpu or args.forward_only or args.head_only or args.session_only or args.first_layer != 0 or args.seqlen != 1)) return error.InvalidLayerwiseOptions;
    if (args.layerwise_stages and !args.layerwise) return error.InvalidLayerwiseOptions;
    if (!args.compare_cpu and !args.session_only and args.activations == null) return error.MissingActivations;

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

    if (args.session_only) {
        defer progress.end();
        if (platform.target != .cpu or args.compare_cpu or args.forward_only or args.seqlen < 5 or args.layers == 0 or args.layers > repo_model.inner.model.layers.len) return error.InvalidSessionComparison;
        try compareSession(allocator, io, platform, &repo_model, &store, repo, &progress, shardings, args);
        return;
    }

    if (args.forward_only) {
        defer progress.end();
        try compareFullForward(allocator, io, platform, &repo_model, &store, &progress, args, shardings);
        return;
    }

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

// Keep this focused host-loop regression small enough to run while the SDK
// compiles the full model. Both paths use actual checkpoint weights and the
// production full-forward executable; the reference recomputes the entire
// causal sequence instead of feeding tokens through Session.runDecode.
fn compareSession(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, mdl: *model.LoadedModel, store: *zml.io.TensorStore, repo: std.Io.Dir, progress: *std.Progress.Node, shardings: common.Shardings, args: Args) !void {
    var subset = mdl.*;
    subset.inner.model.layers = mdl.inner.model.layers[0..args.layers];
    var compiled = try subset.compile(allocator, io, platform, .vanilla, shardings, args.seqlen, progress);
    defer compiled.deinit();
    var weights = try subset.loadBuffers(allocator, io, platform, store, progress, shardings);
    defer subset.unloadBuffers(&weights, allocator);
    const tokenizer_file = try repo.openFile(io, "tokenizer.json", .{});
    defer tokenizer_file.close(io);
    var reader = tokenizer_file.reader(io, &.{});
    const tokenizer_bytes = try reader.interface.readAlloc(allocator, try tokenizer_file.length(io));
    defer allocator.free(tokenizer_bytes);
    var tokenizer = try zml.tokenizer.Tokenizer.fromBytes(allocator, tokenizer_bytes);
    defer tokenizer.deinit();
    var session = try llama_session.Session.init(allocator, io, platform, tokenizer, &compiled, &weights);
    defer session.deinit();
    const prompt = [_]u32{ 1000, 1001, 1002 };
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(allocator);
    try tokens.appendSlice(allocator, &prompt);
    try session.runPrefill(tokens.items);
    var text = std.Io.Writer.Allocating.init(allocator);
    defer text.deinit();
    try session.runDecode(&tokens, &text.writer);
    if (tokens.items.len != args.seqlen) return error.SessionEndedEarly;

    var predicted = try zml.Buffer.fromBytes(io, platform, compiled.params.prefill_tokens.shape(), .replicated, std.mem.sliceAsBytes(tokens.items));
    defer predicted.deinit();
    var reference_cache = try compiled.params.kv_cache.initBuffer(io, platform, shardings.model);
    defer model.KvCache.deinitBuffer(&reference_cache);
    var reference_rng = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, 0);
    defer zml.Tensor.Rng.deinitBuffer(&reference_rng);
    var metadata = try compiled.params.attention_metadata.initBuffer(io, platform, shardings.model);
    defer zml.attention.Metadata.deinitBuffer(&metadata);
    // Keep all position predictions in the independent causal reference;
    // production prefill projects only its requested final prompt position.
    const reference_exe = try inference.KernelExe.compile(allocator, io, platform, .{
        .shardings = &shardings.all(),
        .program_name = "llama_session_causal_reference",
    }, .{.{
        .weights = subset.inner,
        .tokens = compiled.params.prefill_tokens,
        .token_index = compiled.params.token_index,
        .kv_cache = compiled.params.kv_cache,
        .rng = compiled.params.rng,
        .attention_metadata = compiled.params.attention_metadata,
        .attention_parameters = compiled.params.prefill_attention_parameters,
    }});
    defer reference_exe.deinit();
    var runner = try inference.KernelRunner.init(allocator, &reference_exe, &weights);
    defer runner.deinit(allocator);
    inference.run(&runner, .{
        .io = io,
        .tokens_buf = &predicted,
        .token_index_buf = &session.token_index_buffers[0],
        .kv_cache_buffers = &reference_cache,
        .rng_buffers = &reference_rng,
        .attention_metadata_buffers = &metadata,
    });
    const expected_tokens = try predicted.toSliceAlloc(allocator, io);
    defer expected_tokens.free(allocator);
    try std.testing.expectEqualSlices(u32, expected_tokens.items(u32)[prompt.len - 1 .. tokens.items.len - 1], tokens.items[prompt.len..]);
    inline for (.{ "k", "v" }) |field| {
        const a = try @field(session.kv_cache_buffers, field).toSliceAlloc(allocator, io);
        defer a.free(allocator);
        const b = try @field(reference_cache, field).toSliceAlloc(allocator, io);
        defer b.free(allocator);
        const shape = compiled.params.kv_cache.k.shape();
        const width: usize = @intCast(shape.dim(.h) * shape.dim(.hd));
        // The last emitted token has not been fed back, so only positions
        // through length-2 have been computed by both paths.
        for (0..args.layers) |layer| {
            for (0..tokens.items.len - 1) |position| {
                const start = (layer * args.seqlen + position) * width;
                for (a.items(u16)[start..][0..width], b.items(u16)[start..][0..width], 0..) |left, right, lane| {
                    const x: f32 = @bitCast(@as(u32, left) << 16);
                    const y: f32 = @bitCast(@as(u32, right) << 16);
                    if (!std.math.isFinite(x) or !std.math.isFinite(y) or @abs(x - y) > 0.03 + 0.02 * @max(@abs(x), @abs(y))) {
                        std.log.err("Session KV {s}: layer {}, position {}, lane {}: actual={}, reference={}", .{ field, layer, position, lane, x, y });
                        return error.TestUnexpectedResult;
                    }
                }
            }
        }
    }
    for (1..tokens.items.len + 1) |prefix_len| {
        try session.runPrefill(tokens.items[0..prefix_len]);
        try std.testing.expectEqual(expected_tokens.items(u32)[prefix_len - 1], session.last_generated_token);
    }
    std.log.info("PASS session feedback: {} layers, {} prompt tokens, {} generated tokens; causal tokens and computed KV entries match; last-token prefill matches all {} prefix lengths", .{ args.layers, prompt.len, tokens.items.len - prompt.len, tokens.items.len });
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
    if (args.layerwise) return compareLayerwise(allocator, io, platform, cpu, mdl, actual, expected, args);
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

// Diagnose a layer with the exact CPU input as well as with all prior device
// outputs. This is separate from the unchanged whole-forward correctness gate.
fn compareLayerwise(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, cpu: *zml.Platform, mdl: model.Model, actual: *model.Buffers, expected: *model.Buffers, args: Args) !void {
    if (args.layers == 0 or args.layers > mdl.model.layers.len) return error.InvalidLayerCount;
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    const shape = zml.Shape.init(.{ .s = 1, .d = mdl.config.hidden_size }, .bf16);
    const embed_exe = try zml.FnExe(model.EmbedTokens.forward).compile(allocator, io, cpu, .{ .shardings = cpu.shardings.values() }, .{.{
        .embedding = .{ .embed_tokens = mdl.model.embed_tokens },
        .tokens = zml.Tensor.init(.{ .s = 1 }, .u32),
    }});
    defer embed_exe.deinit();
    var embed = try zml.FnExe(model.EmbedTokens.forward).Runner(.{.embedding}).init(&embed_exe, allocator, .{
        .embedding = .{ .embed_tokens = expected.model.embed_tokens },
    });
    defer embed.deinit(allocator);
    const token = [_]u32{1000};
    var tokens = try zml.Buffer.fromBytes(io, cpu, .init(.{ .s = 1 }, .u32), .replicated, std.mem.sliceAsBytes(&token));
    defer tokens.deinit();
    var embedded: zml.Buffer = undefined;
    embed.run(io, .{ .inputs = .{ .tokens = tokens }, .outputs = .{ .hidden = &embedded } });
    defer embedded.deinit();
    const initial = try embedded.toSliceAlloc(allocator, io);
    defer initial.free(allocator);
    const reference_bytes = try allocator.dupe(u8, initial.bytes);
    defer allocator.free(reference_bytes);
    const propagated_bytes = try allocator.dupe(u8, initial.bytes);
    defer allocator.free(propagated_bytes);
    var failed = false;
    const hidden_tolerance: zml.testing.CompareOpts = .{ .absolute_tolerance = 0.1, .relative_tolerance = 0.03, .minimum_close_fraction = 1 };
    const cache_tolerance: zml.testing.CompareOpts = .{ .absolute_tolerance = 0.03, .relative_tolerance = 0.02, .minimum_close_fraction = 1 };
    const row_size = mdl.config.num_key_value_heads * (mdl.config.hidden_size / mdl.config.num_attention_heads);
    for (0..args.layers) |layer| {
        std.log.info("Layerwise layer {}: CPU input and propagated input, position {}", .{ layer, args.token_offset });
        if (args.layerwise_stages) try compareStages(allocator, io, platform, cpu, mdl, actual.model.layers[layer], expected.model.layers[layer], shape, reference_bytes, cache_seqlen, args.token_offset, layer);
        var reference = try runTransformer(allocator, io, cpu, mdl, expected.model.layers[layer..][0..1], shape, reference_bytes, cache_seqlen, args.token_offset, layer);
        defer reference.hidden.deinit();
        defer model.KvCache.deinitBuffer(&reference.kv_cache);
        var local = try runTransformer(allocator, io, platform, mdl, actual.model.layers[layer..][0..1], shape, reference_bytes, cache_seqlen, args.token_offset, layer);
        defer local.hidden.deinit();
        defer model.KvCache.deinitBuffer(&local.kv_cache);
        var propagated = try runTransformer(allocator, io, platform, mdl, actual.model.layers[layer..][0..1], shape, propagated_bytes, cache_seqlen, args.token_offset, layer);
        defer propagated.hidden.deinit();
        defer model.KvCache.deinitBuffer(&propagated.kv_cache);
        failed = !(try reportBf16Region(allocator, io, "local hidden", layer, local.hidden, reference.hidden, 0, shape.count(), hidden_tolerance)) or failed;
        failed = !(try reportBf16Region(allocator, io, "propagated hidden", layer, propagated.hidden, reference.hidden, 0, shape.count(), hidden_tolerance)) or failed;
        inline for (.{ "k", "v" }) |field| {
            const start = (layer * cache_seqlen + args.token_offset) * row_size;
            failed = !(try reportBf16Region(allocator, io, "local " ++ field, layer, @field(local.kv_cache, field), @field(reference.kv_cache, field), start, row_size, cache_tolerance)) or failed;
            failed = !(try reportBf16Region(allocator, io, "propagated " ++ field, layer, @field(propagated.kv_cache, field), @field(reference.kv_cache, field), start, row_size, cache_tolerance)) or failed;
        }
        const next_reference = try reference.hidden.toSliceAlloc(allocator, io);
        defer next_reference.free(allocator);
        const next_propagated = try propagated.hidden.toSliceAlloc(allocator, io);
        defer next_propagated.free(allocator);
        @memcpy(reference_bytes, next_reference.bytes);
        @memcpy(propagated_bytes, next_propagated.bytes);
    }
    if (failed) return error.TestUnexpectedResult;
    std.log.info("PASS layerwise tolerances (does not replace whole-forward comparison)", .{});
}

const LayerStages = struct {
    const Output = struct { stages: [11]zml.Tensor, rms: [3]zml.Tensor, kv_cache: model.KvCache };
    fn forward(input: model.TransformerLayer.Input) Output {
        const layer = input.layer;
        const x = input.hidden.withPartitioning(.{ .d = .replicated });
        const normalized = layer.input_layernorm.forward(x);
        const attention, const cache = layer.self_attn.forward(normalized, input.token_index, input.kv_cache, input.kv_cache_index, input.attention_metadata, input.attention_parameters);
        const residual = x.add(attention).withPartitioning(.{ .d = .replicated });
        const post_norm = layer.post_attention_layernorm.forward(residual);
        const xf = residual.convert(.f32);
        const variance = xf.powByConst(2).mean(.d);
        const inverse = zml.Tensor.rsqrt(variance.addConstant(layer.post_attention_layernorm.eps));
        const scaled = xf.mul(inverse.broad(xf.shape()));
        const gate = layer.mlp.gate_proj.forward(post_norm);
        const up = layer.mlp.up_proj.forward(post_norm);
        const sigmoid = gate.sigmoid();
        const silu = gate.mul(sigmoid);
        const product = silu.mul(up);
        const mlp = layer.mlp.forward(post_norm).rename(.{ .dout = .d }).withPartitioning(.{ .d = .replicated });
        return .{ .stages = .{ normalized, attention, residual, post_norm, gate, up, sigmoid, silu, product, mlp, mlp.add(residual) }, .rms = .{ variance, inverse, scaled }, .kv_cache = cache };
    }
};

fn compareStages(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, cpu: *zml.Platform, mdl: model.Model, actual: zml.Bufferized(model.TransformerLayer), expected: zml.Bufferized(model.TransformerLayer), shape: zml.Shape, bytes: []const u8, cache_seqlen: usize, token_offset: u32, layer: usize) !void {
    const kv_shape = zml.Shape.init(.{ .layer = mdl.model.layers.len, .k = cache_seqlen, .h = mdl.config.num_key_value_heads, .hd = mdl.config.hidden_size / mdl.config.num_attention_heads }, .bf16);
    const cache_data = try allocator.alloc(u16, kv_shape.count());
    defer allocator.free(cache_data);
    for (cache_data, 0..) |*bits, i| {
        const value: f32 = @as(f32, @floatFromInt(@as(i32, @intCast((i * 13 + 7) % 17)) - 8)) / 64.0;
        bits.* = @truncate(@as(u32, @bitCast(value)) >> 16);
    }
    var results: [2]zml.Bufferized(LayerStages.Output) = undefined;
    var done: usize = 0;
    defer for (results[0..done]) |*out| {
        for (&out.stages) |*stage| stage.deinit();
        for (&out.rms) |*stage| stage.deinit();
        model.KvCache.deinitBuffer(&out.kv_cache);
    };
    for ([_]*zml.Platform{ cpu, platform }, [_]zml.Bufferized(model.TransformerLayer){ expected, actual }, 0..) |target, weights, index| {
        const exe = try zml.FnExe(LayerStages.forward).compile(allocator, io, target, .{ .shardings = target.shardings.values() }, .{.{
            .layer = mdl.model.layers[layer],
            .hidden = zml.Tensor.fromShape(shape),
            .token_index = zml.Tensor.init(.{}, .u32),
            .kv_cache = model.KvCache.init(kv_shape),
            .kv_cache_index = zml.Tensor.init(.{}, .u32),
            .attention_metadata = .vanilla,
            .attention_parameters = .vanilla,
        }});
        defer exe.deinit();
        var runner = try zml.FnExe(LayerStages.forward).Runner(.{.layer}).init(&exe, allocator, .{ .layer = weights });
        defer runner.deinit(allocator);
        var hidden = try zml.Buffer.fromBytes(io, target, shape, .replicated, bytes);
        defer hidden.deinit();
        var pos = try zml.Buffer.scalar(io, target, token_offset, .u32);
        defer pos.deinit();
        var layer_index = try zml.Buffer.scalar(io, target, layer, .u32);
        defer layer_index.deinit();
        var cache: model.KvCache.Buffer = .{
            .k = try zml.Buffer.fromBytes(io, target, kv_shape, .replicated, std.mem.sliceAsBytes(cache_data)),
            .v = try zml.Buffer.fromBytes(io, target, kv_shape, .replicated, std.mem.sliceAsBytes(cache_data)),
        };
        errdefer model.KvCache.deinitBuffer(&cache);
        var stages: [11]zml.Buffer = undefined;
        var rms: [3]zml.Buffer = undefined;
        runner.run(io, .{
            .inputs = .{ .hidden = hidden, .token_index = pos, .kv_cache = cache, .kv_cache_index = layer_index, .attention_metadata = .vanilla },
            .outputs = .{ .stages = &stages, .rms = &rms, .kv_cache = &cache },
        });
        results[index] = .{ .stages = stages, .rms = rms, .kv_cache = cache };
        done += 1;
    }
    for ([_][]const u8{ "input norm", "attention", "residual", "post norm", "gate", "up", "sigmoid", "silu", "product", "MLP", "stage hidden" }, 0..) |name, i| {
        _ = try reportBf16Region(allocator, io, name, layer, results[1].stages[i], results[0].stages[i], 0, results[0].stages[i].shape().count(), .exact_match);
    }
    try reportProjectionRoundoff(allocator, io, "gate", layer, expected.mlp.gate_proj.linear.weight, results[1].stages[3], results[0].stages[3], results[1].stages[4], results[0].stages[4]);
    try reportProjectionRoundoff(allocator, io, "up", layer, expected.mlp.up_proj.linear.weight, results[1].stages[3], results[0].stages[3], results[1].stages[5], results[0].stages[5]);
    for ([_][]const u8{ "variance", "rsqrt", "scaled" }, 0..) |name, i| {
        const a = try results[1].rms[i].toSliceAlloc(allocator, io);
        defer a.free(allocator);
        const b = try results[0].rms[i].toSliceAlloc(allocator, io);
        defer b.free(allocator);
        var max_abs: f32 = 0;
        var max_rel: f32 = 0;
        for (a.items(f32), b.items(f32)) |x, y| {
            if (!std.math.isFinite(x) or !std.math.isFinite(y)) return error.NonFiniteRms;
            max_abs = @max(max_abs, @abs(x - y));
            max_rel = @max(max_rel, @abs(x - y) / @max(1e-30, @abs(y)));
        }
        std.log.info("Layerwise RMS {s} layer {}: max_abs={}, max_rel={}, device_first={}, CPU_first={}", .{ name, layer, max_abs, max_rel, a.items(f32)[0], b.items(f32)[0] });
    }
}

// An independent FP64 dot for mismatching outputs, using the CPU input. This
// only isolates dot rounding when the preceding post-norm inputs match exactly.
fn reportProjectionRoundoff(allocator: std.mem.Allocator, io: std.Io, name: []const u8, layer: usize, weight: zml.Buffer, actual_input: zml.Buffer, input: zml.Buffer, actual: zml.Buffer, reference: zml.Buffer) !void {
    if (weight.shape().dtype() != .bf16 or input.shape().dim(.s) != 1) return;
    const a = try actual.toSliceAlloc(allocator, io);
    defer a.free(allocator);
    const b = try reference.toSliceAlloc(allocator, io);
    defer b.free(allocator);
    const x = try input.toSliceAlloc(allocator, io);
    defer x.free(allocator);
    const device_x = try actual_input.toSliceAlloc(allocator, io);
    defer device_x.free(allocator);
    if (!std.mem.eql(u8, device_x.bytes, x.bytes)) {
        std.log.info("Layerwise dot {s} layer {}: FP64 check skipped because post-norm inputs differ", .{ name, layer });
        return;
    }
    const w = try weight.toSliceAlloc(allocator, io);
    defer w.free(allocator);
    const cols: usize = @intCast(weight.shape().dim(.d));
    std.debug.assert(cols == x.items(u16).len);
    std.debug.assert(w.items(u16).len == cols * a.items(u16).len);
    var reported: usize = 0;
    for (a.items(u16), b.items(u16), 0..) |device_bits, cpu_bits, row| {
        if (device_bits == cpu_bits) continue;
        var sum: f64 = 0;
        for (x.items(u16), w.items(u16)[row * cols ..][0..cols]) |x_bits, w_bits| {
            const xf: f32 = @bitCast(@as(u32, x_bits) << 16);
            const wf: f32 = @bitCast(@as(u32, w_bits) << 16);
            sum += @as(f64, xf) * @as(f64, wf);
        }
        const device_value: f32 = @bitCast(@as(u32, device_bits) << 16);
        const cpu_value: f32 = @bitCast(@as(u32, cpu_bits) << 16);
        const device_error = @abs(@as(f64, device_value) - sum);
        const cpu_error = @abs(@as(f64, cpu_value) - sum);
        const closer: []const u8 = if (device_error < cpu_error) "device" else if (cpu_error < device_error) "CPU" else "tie";
        std.log.info("Layerwise dot {s} layer {} row {}: FP64={}, device={} (bits={x}), CPU={} (bits={x}), closer={s}", .{ name, layer, row, sum, device_value, device_bits, cpu_value, cpu_bits, closer });
        reported += 1;
        if (reported == 16) break;
    }
}

fn reportBf16Region(allocator: std.mem.Allocator, io: std.Io, name: []const u8, layer: usize, actual: zml.Buffer, reference: zml.Buffer, start: usize, count: usize, tolerance: zml.testing.CompareOpts) !bool {
    const a = try actual.toSliceAlloc(allocator, io);
    defer a.free(allocator);
    const b = try reference.toSliceAlloc(allocator, io);
    defer b.free(allocator);
    var different: usize = 0;
    var bad: usize = 0;
    var finite = true;
    var max_abs: f32 = 0;
    var sum_squared: f64 = 0;
    for (a.items(u16)[start..][0..count], b.items(u16)[start..][0..count]) |x, y| {
        const xf: f32 = @bitCast(@as(u32, x) << 16);
        const yf: f32 = @bitCast(@as(u32, y) << 16);
        different += @intFromBool(x != y);
        if (!std.math.isFinite(xf) or !std.math.isFinite(yf)) {
            finite = false;
            bad += 1;
            continue;
        }
        const err = @abs(xf - yf);
        max_abs = @max(max_abs, err);
        sum_squared += @as(f64, err) * err;
        bad += @intFromBool(err > tolerance.absolute_tolerance + tolerance.relative_tolerance * @max(@abs(xf), @abs(yf)));
    }
    const close_fraction = @as(f32, @floatFromInt(count - bad)) / @as(f32, @floatFromInt(count));
    const passed = finite and close_fraction >= tolerance.minimum_close_fraction;
    const rmse = @sqrt(sum_squared / @as(f64, @floatFromInt(count)));
    std.log.info("Layerwise {s} layer {}: max_abs={}, rmse={}, different={}/{}, close_fraction={}, finite={}, pass={}", .{ name, layer, max_abs, rmse, different, count, close_fraction, finite, passed });
    return passed;
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

    const Output = struct {
        tokens: zml.Tensor,
        kv_cache: model.KvCache,
        rng: zml.Tensor.Rng,
        logits: zml.Tensor,
    };

    fn forward(input: Input) Output {
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
        const head = model.LmHead.init(input.weights);
        const hidden = head.norm.forward(transformed.hidden.withPartialTags(.{ .s, .d }));
        const logits = if (head.lm_head) |projection|
            projection.forward(hidden).rename(.{ .dout = .voc })
        else
            head.embed_tokens.weight.withTags(.{ .voc, .d }).dot(hidden, .d);
        const next_tokens, const next_rng = zml.nn.sampleTokens(logits, head.gen_opts, input.rng);
        return .{
            .tokens = next_tokens.convert(input.tokens.dtype()).reuseBuffer(input.tokens),
            .kv_cache = transformed.kv_cache,
            .rng = next_rng,
            .logits = logits.transpose(.{ .s, .voc }),
        };
    }
};

fn compareFullForward(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, mdl: *model.LoadedModel, store: *zml.io.TensorStore, progress: *std.Progress.Node, args: Args, shardings: common.Shardings) !void {
    const cpu = try zml.Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 1 } });
    defer cpu.deinit(allocator, io);
    const cpu_shardings = try common.Shardings.init(cpu);
    const cache_seqlen = args.cache_seqlen orelse args.seqlen;
    const kv_shape = zml.Shape.init(.{ .layer = mdl.inner.model.layers.len, .k = cache_seqlen, .h = mdl.inner.config.num_key_value_heads, .hd = mdl.inner.config.hidden_size / mdl.inner.config.num_attention_heads }, .bf16);
    const kv = model.KvCache.init(kv_shape);
    const tokens = zml.Tensor.init(.{ .s = args.seqlen }, .u32);
    const position = zml.Tensor.init(.{}, .u32);
    const rng: zml.Tensor.Rng = .init();
    std.log.info("Comparing whole forward: {} layers, {} separate weight arguments, query={}, cache={}, offset={}", .{ mdl.inner.model.layers.len, zml.meta.count(zml.Tensor, &mdl.inner), args.seqlen, cache_seqlen, args.token_offset });

    const actual_exe = try zml.FnExe(inference.Forward.forward).compile(allocator, io, platform, .{ .shardings = &shardings.all(), .program_name = "llama_full_forward_comparison" }, .{.{ .weights = mdl.inner, .tokens = tokens, .token_index = position, .kv_cache = kv, .rng = rng, .attention_metadata = .vanilla, .attention_parameters = .vanilla }});
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
    var reference_logits: zml.Buffer = undefined;
    var have_reference_logits = false;
    defer if (have_reference_logits) reference_logits.deinit();
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
            var reference_weights = try mdl.loadBuffers(allocator, io, cpu, store, progress, cpu_shardings);
            defer mdl.unloadBuffers(&reference_weights, allocator);
            var runner = try zml.FnExe(ReferenceForward.forward).Runner(.{.weights}).init(&reference_exe, allocator, .{ .weights = reference_weights });
            defer runner.deinit(allocator);
            runner.run(io, .{ .inputs = .{ .tokens = token_buffer, .token_index = pos, .kv_cache = cache, .rng = rng_buffer }, .outputs = .{ .tokens = &token_buffer, .kv_cache = &cache, .rng = &rng_buffer, .logits = &reference_logits } });
            have_reference_logits = true;
        } else {
            var actual_weights = try mdl.loadBuffers(allocator, io, platform, store, progress, shardings);
            defer mdl.unloadBuffers(&actual_weights, allocator);
            if (args.benchmark_iterations > 0) try benchmarkFullForward(allocator, io, platform, &actual_exe, actual_weights, kv, mdl.inner.config.bos_token_id, args.benchmark_iterations);
            var runner = try zml.FnExe(inference.Forward.forward).Runner(.{.weights}).init(&actual_exe, allocator, .{ .weights = actual_weights });
            defer runner.deinit(allocator);
            runner.run(io, .{ .inputs = .{ .tokens = token_buffer, .token_index = pos, .kv_cache = cache, .rng = rng_buffer, .attention_metadata = .vanilla }, .outputs = .{ .tokens = &token_buffer, .kv_cache = &cache, .rng = &rng_buffer } });
        }
        outputs[i] = .{ .tokens = token_buffer, .kv_cache = cache, .rng = rng_buffer };
        done += 1;
    }
    if (args.seqlen == 1) {
        const actual_token = try outputs[1].tokens.getValue(u32, io);
        const expected_token = try outputs[0].tokens.getValue(u32, io);
        std.log.info("Whole-forward argmax: device={}, CPU={}", .{ actual_token, expected_token });
        const logits = try reference_logits.toSliceAlloc(allocator, io);
        defer logits.free(allocator);
        const values = logits.items(u16);
        if (actual_token >= values.len or expected_token >= values.len) return error.InvalidToken;
        const actual_score: f32 = @bitCast(@as(u32, values[actual_token]) << 16);
        const expected_score: f32 = @bitCast(@as(u32, values[expected_token]) << 16);
        var higher: usize = 0;
        var tied: usize = 0;
        for (values) |bits| {
            const score: f32 = @bitCast(@as(u32, bits) << 16);
            if (score > actual_score) higher += 1;
            if (score == actual_score) tied += 1;
        }
        std.log.info("CPU logits: expected token score={}, device token score={}, gap={}, scores_above_device={}, scores_tied_with_device={}", .{ expected_score, actual_score, expected_score - actual_score, higher, tied });
    }
    var comparison_failed = false;
    zml.testing.expectClose(io, outputs[1].tokens, outputs[0].tokens, .exact_match) catch |err| switch (err) {
        error.TestUnexpectedResult => comparison_failed = true,
        else => return err,
    };
    const tolerance: zml.testing.CompareOpts = .{ .absolute_tolerance = 0.03, .relative_tolerance = 0.02, .minimum_close_fraction = 1 };
    inline for (.{ "k", "v" }) |field| {
        const actual = @field(outputs[1].kv_cache, field);
        const reference = @field(outputs[0].kv_cache, field);
        zml.testing.expectClose(io, actual, reference, tolerance) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                comparison_failed = true;
                try reportCacheLayers(allocator, io, field, actual, reference, args, tolerance);
            },
            else => return err,
        };
    }
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
    if (comparison_failed) return error.TestUnexpectedResult;
    std.log.info("PASS whole forward: exact argmax, KV tolerance, untouched cache bits", .{});
}

// Preserve the failing test status while showing where errors first enter the
// updated cache. An argmax mismatch must not hide the transformer diagnostics.
fn reportCacheLayers(allocator: std.mem.Allocator, io: std.Io, field: []const u8, actual: zml.Buffer, reference: zml.Buffer, args: Args, tolerance: zml.testing.CompareOpts) !void {
    const a = try actual.toSliceAlloc(allocator, io);
    defer a.free(allocator);
    const b = try reference.toSliceAlloc(allocator, io);
    defer b.free(allocator);
    const shape = actual.shape();
    const rows: usize = @intCast(shape.dim(.k));
    const width: usize = @intCast(shape.dim(.h) * shape.dim(.hd));
    for (0..@intCast(shape.dim(.layer))) |layer| {
        var max_error: f32 = 0;
        var bad: usize = 0;
        for (args.token_offset..args.token_offset + args.seqlen) |position| {
            const start = (layer * rows + position) * width;
            for (a.items(u16)[start..][0..width], b.items(u16)[start..][0..width]) |left, right| {
                const x: f32 = @bitCast(@as(u32, left) << 16);
                const y: f32 = @bitCast(@as(u32, right) << 16);
                if (!std.math.isFinite(x) or !std.math.isFinite(y)) {
                    bad += 1;
                    continue;
                }
                const err = @abs(x - y);
                max_error = @max(max_error, err);
                if (err > tolerance.absolute_tolerance + tolerance.relative_tolerance * @max(@abs(x), @abs(y))) bad += 1;
            }
        }
        std.log.info("KV {s} layer {}: updated max_abs={}, outside_tolerance={}/{}", .{ field, layer, max_error, bad, args.seqlen * width });
    }
}

// Measures the complete decode executable and synchronous token readback. It
// excludes compilation, weight upload, prefill, tokenization and terminal IO.
// Each trial starts from BOS and feeds every predicted token into the next
// call, including after EOS, to keep the timed workload fixed.
fn benchmarkFullForward(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, exe: *const inference.KernelExe, weights: model.Buffers, kv: model.KvCache, bos: u32, iterations: usize) !void {
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
