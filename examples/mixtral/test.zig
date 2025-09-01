const std = @import("std");

const asynk = @import("async");
const stdx = @import("stdx");
const flags = stdx.flags;
const zml = @import("zml");

const Mixtral = @import("./Mixtral.zig");

const log = std.log.scoped(.mixtral);

pub const std_options: std.Options = .{
    .log_level = .info,
    .logFn = asynk.logFn(std.log.defaultLog),
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.smp_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ test-implementation --model-path=mixtral/ --config=config.json --reference=activation.pt
        ;
        model_path: []const u8,
        activations: []const u8,
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // Select platform
    const platform = context.autoPlatform(.{});

    // Resolve model files
    var args = std.process.args();
    const cli = flags.parse(&args, CliArgs);

    const model_tokenizer_path = try std.fs.path.join(allocator, &.{ cli.model_path, "tokenizer.json" });
    defer allocator.free(model_tokenizer_path);

    // Memory arena dedicated to model shapes and weights
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    // Parse model config
    const config: Mixtral.Config = cfg: {
        const model_config_path = try std.fs.path.join(allocator, &.{ cli.model_path, "config.json" });
        defer allocator.free(model_config_path);
        var config_json_file = try asynk.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(&config_json_buffer);
        var reader = std.json.Reader.init(model_arena, &config_reader.interface);
        defer reader.deinit();
        const cfg = try std.json.parseFromTokenSourceLeaky(Mixtral.Config, model_arena, &reader, .{ .ignore_unknown_fields = true });

        std.log.info("Parsed mixtral config: {}", .{cfg});
        break :cfg cfg;
    };

    // Read model shapes.
    var store: zml.aio.BufferStore = st: {
        const simple_path = try std.fs.path.join(allocator, &.{ cli.model_path, "model.safetensors" });
        defer allocator.free(simple_path);

        if (asynk.File.access(simple_path, .{})) {
            log.info("\tLoading mixtral weights from {s}...", .{simple_path});
            break :st try zml.aio.detectFormatAndOpen(allocator, simple_path);
        } else |_| {}

        const sharded_path = try std.fs.path.join(allocator, &.{ cli.model_path, "model.safetensors.index.json" });
        defer allocator.free(sharded_path);
        log.info("\tLoading mixtral weights from {s}...", .{sharded_path});
        break :st try zml.aio.detectFormatAndOpen(allocator, sharded_path);
    };
    defer store.deinit();

    // Create the model and configure it.
    const options: Mixtral.Options = .{
        .max_seq_len = 256,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };
    const mixtral = try Mixtral.init(model_arena, store, config, options);

    // Load the weights.
    var mixtral_weights = try mixtral.loadBuffers(model_arena, store, platform);
    defer zml.aio.unloadBuffers(&mixtral_weights);

    // Load the activations.
    var activation_store = try zml.aio.detectFormatAndOpen(allocator, cli.activations);
    defer activation_store.deinit();
    // for (activation_store.buffers.keys(), activation_store.buffers.values()) |k, v| {
    //     log.info("Loaded activations: {s} -> {f}", .{k, v});
    // }

    // Test implementation
    try testImplementation(platform, mixtral, mixtral_weights, activation_store);
}

fn testImplementation(
    platform: zml.Platform,
    mixtral: Mixtral,
    mixtral_weights: zml.Bufferized(Mixtral),
    store: zml.aio.BufferStore,
) !void {
    try zml.testing.testLayer(platform, store, "model.model.embed_tokens", mixtral.model.embed_tokens, mixtral_weights.model.embed_tokens, 1e-3);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.v_proj", mixtral.model.layers[0].self_attn.v_proj, mixtral_weights.model.layers[0].self_attn.v_proj, 1e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.q_proj", mixtral.model.layers[0].self_attn.q_proj, mixtral_weights.model.layers[0].self_attn.q_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.k_proj", mixtral.model.layers[0].self_attn.k_proj, mixtral_weights.model.layers[0].self_attn.k_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.self_attn.o_proj", mixtral.model.layers[0].self_attn.o_proj, mixtral_weights.model.layers[0].self_attn.o_proj, 2e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.mlp", mixtral.model.layers[0].mlp, mixtral_weights.model.layers[0].mlp, 1e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.input_layernorm", mixtral.model.layers[0].input_layernorm, mixtral_weights.model.layers[0].input_layernorm, 1e-2);
    try zml.testing.testLayer(platform, store, "model.model.layers.0.post_attention_layernorm", mixtral.model.layers[0].post_attention_layernorm, mixtral_weights.model.layers[0].post_attention_layernorm, 1e-2);
}
