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
    session: bool = false,
    benchmark: bool = false,
    seqlen: u32 = 1024,

    pub const help =
        \\Use llama_tests --model=<path> --activations=<path>
        \\
        \\ Validate the LLaMA implementation against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>            Path to the model repository
        \\   --activations=<path>      Path to activation safetensors
        \\   --session                Validate generated-token feedback positions
        \\   --benchmark              Time fixed-work full-model cat-story generation
        \\   --seqlen=<number>        Benchmark context capacity (default: 1024)
        \\
    ;
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;
    const args = zml.stdx.flags.parse(init.minimal.args, Args);

    const platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator, io);

    const repo = try zml.safetensors.resolveModelRepo(io, args.model);
    defer repo.close(io);
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();
    const meshes: common.Meshes = try .init(platform);
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &meshes.all());
    defer store.deinit();

    var repo_model = try llama.LoadedModel.init(allocator, io, repo, store.view(), .{});
    defer repo_model.deinit(allocator);

    var progress = std.Progress.start(io, .{ .root_name = args.model });

    if (args.session or args.benchmark) {
        defer progress.end();
        return runSession(allocator, io, platform, repo, &repo_model, &store, meshes, &progress, args);
    }

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress);
    defer repo_model.unloadBuffers(&model_buffers, allocator);
    progress.end();

    try run(allocator, io, platform, args.activations orelse return error.MissingActivations, repo_model.inner, &model_buffers, &meshes.all());
}

// Exercise the production host loop; independent causal prefix-prefills provide
// the initial prediction and KV references without duplicating position logic.
fn runSession(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, repo: std.Io.Dir, loaded: *const model.LoadedModel, store: *zml.io.TensorStore, meshes: common.Meshes, progress: *std.Progress.Node, args: Args) !void {
    var subset = loaded.*;
    subset.inner.gen_opts = .{ .topk = 1 };
    if (!args.benchmark) subset.inner.model.layers = loaded.inner.model.layers[0..@min(2, loaded.inner.model.layers.len)];
    var compiled = try subset.compile(allocator, io, platform, .{ .stablehlo = {} }, meshes, if (args.benchmark) args.seqlen else 8, progress);
    defer compiled.deinit();
    var weights = try subset.loadBuffers(allocator, io, platform, store, progress);
    defer subset.unloadBuffers(&weights, allocator);
    const file = try repo.openFile(io, "tokenizer.json", .{});
    defer file.close(io);
    var reader = file.reader(io, &.{});
    const bytes = try reader.interface.readAlloc(allocator, try file.length(io));
    defer allocator.free(bytes);
    var tokenizer = try zml.tokenizer.Tokenizer.fromBytes(allocator, bytes);
    defer tokenizer.deinit();
    var session = try llama.Session.init(allocator, io, platform, tokenizer, &compiled, &weights);
    defer session.deinit();
    if (args.benchmark) return benchmarkSession(allocator, io, &session);

    var reference = try llama.Session.init(allocator, io, platform, tokenizer, &compiled, &weights);
    defer reference.deinit();
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(allocator);
    const prompt = [_]u32{ 1000, 1001, 1002 };
    try tokens.appendSlice(allocator, &prompt);
    try session.runPrefill(tokens.items);
    var text: std.Io.Writer.Allocating = .init(allocator);
    defer text.deinit();
    try session.runDecode(&tokens, &text.writer);
    try std.testing.expectEqual(compiled.params.seqlen, tokens.items.len);

    const actual_k = try session.kv_cache_buffers.k.toSliceAlloc(allocator, io);
    defer actual_k.free(allocator);
    const actual_v = try session.kv_cache_buffers.v.toSliceAlloc(allocator, io);
    defer actual_v.free(allocator);
    // The final emitted token has not been fed back into the cache.
    for (prompt.len..tokens.items.len) |length| {
        try reference.runPrefill(tokens.items[0..length]);
        inline for (.{ "k", "v" }, .{ actual_k, actual_v }) |field, actual| {
            const expected = try @field(reference.kv_cache_buffers, field).toSliceAlloc(allocator, io);
            defer expected.free(allocator);
            errdefer std.log.err("Session cache {s} mismatch at prefix {}", .{ field, length });
            try zml.testing.expectClose(io, actual.subSlice(actual.shape.axis(.k), 0, @intCast(length)), expected.subSlice(expected.shape.axis(.k), 0, @intCast(length)), .{
                .absolute_tolerance = 0.03,
                .relative_tolerance = 0.02,
                .minimum_close_fraction = 1.0,
            });
        }
        // The initial prediction uses the same prefill shape. Later BF16
        // decode/prefill kernels can change close argmax decisions; compare
        // their caches for the actual emitted history rather than token IDs.
        if (length == prompt.len) try std.testing.expectEqual(tokens.items[length], reference.last_generated_token);
    }
    std.log.info("PASS session feedback: initial prediction and all computed KV entries match independent prefix-prefills", .{});
}

// Exercise production Session calls with fixed work, keeping generated text in
// memory. EOS is disabled only in this test tool; ordinary generation is intact.
fn benchmarkSession(allocator: std.mem.Allocator, io: std.Io, session: *llama.Session) !void {
    const original_config = session.config;
    defer session.config = original_config;
    var config = original_config.*;
    config.eos_token_id = .{ .int = std.math.maxInt(u32) };
    session.config = &config;
    const prompt = try session.tokenizePrompt(allocator, "Tell me a story about a cat");
    defer allocator.free(prompt);
    if (prompt.len >= session.seqlen) return error.BenchmarkContextTooShort;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(allocator);
    try tokens.appendSlice(allocator, prompt);
    var text: std.Io.Writer.Allocating = .init(allocator);
    defer text.deinit();

    const prefill_start: std.Io.Timestamp = .now(io, .awake);
    try session.runPrefill(tokens.items);
    const prefill_ns = prefill_start.untilNow(io, .awake).nanoseconds;
    const decode_start: std.Io.Timestamp = .now(io, .awake);
    try session.runDecode(&tokens, &text.writer);
    const decode_ns = decode_start.untilNow(io, .awake).nanoseconds;
    try std.testing.expectEqual(@as(usize, session.seqlen), tokens.items.len);
    const emitted = tokens.items.len - prompt.len;
    const steps = emitted - 1; // The first token comes from prefill.
    const seconds = @as(f64, @floatFromInt(decode_ns)) / std.time.ns_per_s;
    std.log.info("fixed-work: seqlen={}, prompt_tokens={}, emitted={}, device_decode_steps={}, prefill_ns={}, decode_ns={}, decode_steps_per_s={d:.3}, cli_tok_per_s={d:.3}", .{
        session.seqlen,                           prompt.len,                                 emitted, steps, prefill_ns, decode_ns,
        @as(f64, @floatFromInt(steps)) / seconds, @as(f64, @floatFromInt(emitted)) / seconds,
    });
}

fn run(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    activations_path: []const u8,
    mdl: model.Model,
    model_buffers: *model.Buffers,
    meshes: []const *const zml.Mesh,
) !void {
    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
    defer registry.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry, meshes);
    defer activation_store.deinit();

    try testLayer(allocator, io, platform, activation_store.view(), "embed_tokens", mdl.model.embed_tokens, model_buffers.model.embed_tokens, meshes, .{ .absolute_tolerance = 1e-3 });

    if (mdl.model.layers.len == 0) return;

    const layer = mdl.model.layers[0];
    const layer_buffers = model_buffers.model.layers[0];

    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.v_proj", layer.self_attn.v_proj, layer_buffers.self_attn.v_proj, meshes, .{ .absolute_tolerance = 1e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.q_proj", layer.self_attn.q_proj, layer_buffers.self_attn.q_proj, meshes, .{ .absolute_tolerance = 2e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.k_proj", layer.self_attn.k_proj, layer_buffers.self_attn.k_proj, meshes, .{ .absolute_tolerance = 2e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.self_attn.o_proj", layer.self_attn.o_proj, layer_buffers.self_attn.o_proj, meshes, .{ .absolute_tolerance = 2e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.mlp", layer.mlp, layer_buffers.mlp, meshes, .{ .absolute_tolerance = 1e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.input_layernorm", layer.input_layernorm, layer_buffers.input_layernorm, meshes, .{ .absolute_tolerance = 1e-2 });
    try testLayer(allocator, io, platform, activation_store.view(), "layers.0.post_attention_layernorm", layer.post_attention_layernorm, layer_buffers.post_attention_layernorm, meshes, .{ .absolute_tolerance = 1e-2 });
}

fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    activation_store: zml.io.TensorStore.View,
    name: []const u8,
    layer: anytype,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    meshes: []const *const zml.Mesh,
    opts: zml.testing.CompareOpts,
) !void {
    const in_key = try std.fmt.allocPrint(allocator, "{s}.in", .{name});
    defer allocator.free(in_key);
    const in_shape = activation_store.getShape(in_key) orelse return error.NotFound;
    var in_buffer = try loadBufferFromStore(allocator, io, platform, activation_store, in_key);
    defer in_buffer.deinit();
    const in_tensor = zml.Tensor.fromShape(in_shape);

    const out_key = try std.fmt.allocPrint(allocator, "{s}.out", .{name});
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
