const asynk = @import("async");
const flags = @import("tigerbeetle/flags");
const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const llama_mod = @import("./llama.zig");
const LlamaLM = llama_mod.LlamaLM;

const Tensor = zml.Tensor;

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain, .{});
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ test-implementation --weights=llama3.8B.safetensors --config=config.json --reference=activation.safetensors
        ;
        weights: []const u8,
        config: []const u8,
        reference: []const u8,
        num_heads: ?i64 = null,
        num_kv_heads: ?i64 = null,
        rope_freq_base: ?i64 = null,
    };
    var gpa = std.heap.GeneralPurposeAllocator(.{ .thread_safe = true }){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // Select platform
    const platform = context.autoPlatform(.{});

    // Parse program args
    var args = std.process.args();
    const cli_args = flags.parse(&args, CliArgs);
    const model_file = cli_args.weights;

    // Memory arena dedicated to model shapes and weights
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    std.log.info("Model file: {s}", .{model_file});

    // Read model shapes.
    var buffer_store = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer buffer_store.deinit();

    // Create the model and configure it.
    var llama = try zml.aio.populateModel(LlamaLM, model_arena, buffer_store);
    const num_heads: i64 = cli_args.num_heads orelse buffer_store.metadata("num_heads", .int64) orelse @panic("--num_heads is required for this model");
    const num_kv_heads: i64 = cli_args.num_kv_heads orelse buffer_store.metadata("num_kv_heads", .int64) orelse num_heads;

    const rope_impl = if (buffer_store.metadata("rope_impl", .string)) |val|
        std.meta.stringToEnum(zml.nn.RopeOpts.Implementation, val).?
    else
        .sequential;

    const config = blk: {
        var config_json_file = try asynk.File.open(cli_args.config, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var reader = std.json.reader(allocator, config_json_file.reader());
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };
    std.log.info("Parsed llama config: {}", .{config});

    const llama_options: llama_mod.LlamaOptions = .{
        .max_seq_len = 256,
        .num_kv_heads = num_kv_heads,
        .num_heads = num_heads,
        .gen_opts = .{},
        .rms_norm_eps = @floatCast(buffer_store.metadata("rms_norm_eps", .float64) orelse 1e-5),
        .rope_opts = .{
            .impl = rope_impl,
            .freq_base = @floatCast(buffer_store.metadata("rope_freq_base", .float64) orelse @as(f32, @floatFromInt(cli_args.rope_freq_base orelse 10_000))),
        },
    };
    std.log.info("Parsed llama config: {}", .{llama_options});
    llama.init(llama_options);

    // Load the weights.
    var llama_weights = try zml.aio.loadBuffers(LlamaLM, .{llama_options}, buffer_store, model_arena, platform);
    defer zml.aio.unloadBuffers(&llama_weights);

    // Load the activations.
    var activation_buffer_store = try zml.aio.torch.open(allocator, cli_args.reference);
    defer activation_buffer_store.deinit();

    // Test implementation
    try testImplementation(platform, llama, llama_weights, activation_buffer_store);
}

fn testImplementation(
    platform: zml.Platform,
    llama: LlamaLM,
    llama_weights: zml.Bufferized(LlamaLM),
    buffer_store: zml.aio.BufferStore,
) !void {
    try zml.testing.testLayer(platform, buffer_store, "embed_tokens", llama.model.embed_tokens, llama_weights.model.embed_tokens, 1e-3);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.self_attn.v_proj", llama.model.layers[0].self_attn.v_proj, llama_weights.model.layers[0].self_attn.v_proj, 1e-2);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.self_attn.q_proj", llama.model.layers[0].self_attn.q_proj, llama_weights.model.layers[0].self_attn.q_proj, 2e-2);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.self_attn.k_proj", llama.model.layers[0].self_attn.k_proj, llama_weights.model.layers[0].self_attn.k_proj, 2e-2);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.self_attn.o_proj", llama.model.layers[0].self_attn.o_proj, llama_weights.model.layers[0].self_attn.o_proj, 2e-2);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.mlp", llama.model.layers[0].mlp, llama_weights.model.layers[0].mlp, 1e-2);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.input_layernorm", llama.model.layers[0].input_layernorm, llama_weights.model.layers[0].input_layernorm, 1e-2);
    try zml.testing.testLayer(platform, buffer_store, "layers.0.post_attention_layernorm", llama.model.layers[0].post_attention_layernorm, llama_weights.model.layers[0].post_attention_layernorm, 1e-2);
}
