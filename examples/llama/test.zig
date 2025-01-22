const asynk = @import("async");
const flags = @import("tigerbeetle/flags");
const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const llama_mod = @import("./llama.zig");
const LlamaLM = llama_mod.LlamaLM;

const Tensor = zml.Tensor;

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const CliArgs = struct {
        pub const help =
            \\ test-implementation --model=llama3.8B.safetensors --reference=activation.safetensors
        ;
        weights: []const u8,
        config: []const u8,
        reference: []const u8,
        seq_len: ?usize = null,
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

    const config = blk: {
        var config_json_file = try asynk.File.open(cli_args.config, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var reader = std.json.reader(allocator, config_json_file.reader());
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(LlamaLM.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };
    std.log.info("Parsed llama config: {}", .{config});

    const llama_options: LlamaLM.Options = .{
        .max_seq_len = cli_args.seq_len orelse 256,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };

    llama.init(config, llama_options);

    // Load the weights.
    var llama_weights = try zml.aio.loadBuffers(LlamaLM, .{ config, llama_options }, buffer_store, model_arena, platform);
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
