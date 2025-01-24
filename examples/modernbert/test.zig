const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log;
const Tensor = zml.Tensor;
const modernbert_module = @import("modernbert.zig");
const ModernBertOptions = modernbert_module.ModernBertOptions;

// ModernBERT
const ACTIVATIONS_FILE_PATH: []const u8 = "/Users/victor/Documents/development/zml-torch-activation-example/ModernBERT-base.activations.pt";
const MODEL_WEIGHTS_FILE_PATH: []const u8 = "/Users/victor/.cache/huggingface/hub/models--answerdotai--ModernBERT-base/snapshots/5756c58a31a2478f9e62146021f48295a92c3da5/model.safetensors";

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Initialize the ZML context
    var context = try zml.Context.init();
    defer context.deinit();

    // Auto-select platform
    const compute_platform = context.autoPlatform(.{});
    log.info("Selected platform: {s}", .{@tagName(compute_platform.target)});

    // Create a dedicated memory arena for model-related allocations (dedicated to model shapes and weights)
    var arena_state = std.heap.ArenaAllocator.init(allocator);
    defer arena_state.deinit();
    const model_arena = arena_state.allocator();

    // Load the model weights file and parse its structure (shape)
    var weights_file = try zml.aio.detectFormatAndOpen(allocator, MODEL_WEIGHTS_FILE_PATH);
    defer weights_file.deinit();
    log.info("Model contains {d} layers. Loaded from: {s}", .{ weights_file.buffers.count(), MODEL_WEIGHTS_FILE_PATH });

    // Load the activation data file
    const activations = try zml.aio.torch.open(model_arena, ACTIVATIONS_FILE_PATH);
    defer activations.deinit();
    log.info("Found {} activations in {s}", .{ activations.buffers.count(), ACTIVATIONS_FILE_PATH });

    const modernbert_base_options: modernbert_module.ModernBertOptions = .{
        .num_attention_heads = 12,
        .tie_word_embeddings = true,
    };

    // model.embeddings.tok_embeddings
    log.info("\n\nTesting model.embeddings.tok_embeddings layer:", .{});

    const word_embeddings_shape = try zml.aio.populateModelWithPrefix(
        zml.nn.TokenEmbedding,
        model_arena,
        weights_file,
        "model.embeddings.tok_embeddings",
    );

    const word_embeddings_weights = try zml.aio.loadModelBuffersWithPrefix(
        zml.nn.TokenEmbedding,
        word_embeddings_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.embeddings.tok_embeddings",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.embeddings.tok_embeddings",
        word_embeddings_shape,
        word_embeddings_weights,
        1e-6,
    );

    // model.embeddings.norm
    log.info("\n\nTesting model.embeddings.norm layer:", .{});

    const norm_shape = try zml.aio.populateModelWithPrefix(
        zml.nn.LayerNorm,
        model_arena,
        weights_file,
        "model.embeddings.norm",
    );

    const norm_weights = try zml.aio.loadModelBuffersWithPrefix(
        zml.nn.LayerNorm,
        norm_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.embeddings.norm",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.embeddings.norm",
        norm_shape,
        norm_weights,
        1e-3,
    );

    // model.embeddings
    log.info("\n\nTesting model.embeddings layer:", .{});

    const embeddings_shape = try zml.aio.populateModelWithPrefix(
        modernbert_module.ModernBertEmbeddings,
        model_arena,
        weights_file,
        "model.embeddings",
    );

    const embeddings_weights = try zml.aio.loadModelBuffersWithPrefix(
        modernbert_module.ModernBertEmbeddings,
        embeddings_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.embeddings",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.embeddings",
        embeddings_shape,
        embeddings_weights,
        1e-3,
    );

    // model.final_norm
    log.info("\n\nTesting model.final_norm layer:", .{});

    const final_norm_shape = try zml.aio.populateModelWithPrefix(
        zml.nn.LayerNorm,
        model_arena,
        weights_file,
        "model.final_norm",
    );

    const final_norm_weights = try zml.aio.loadModelBuffersWithPrefix(
        zml.nn.LayerNorm,
        final_norm_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.final_norm",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.final_norm",
        final_norm_shape,
        final_norm_weights,
        1e-5,
    );

    // model.layers.2.mlp
    log.info("\n\nTesting model.layers.2.mlp layer:", .{});

    const mlp_shape = try zml.aio.populateModelWithPrefix(
        modernbert_module.ModernBertMLP,
        model_arena,
        weights_file,
        "model.layers.2.mlp",
    );

    const mlp_weights = try zml.aio.loadModelBuffersWithPrefix(
        modernbert_module.ModernBertMLP,
        mlp_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.layers.2.mlp",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2.mlp",
        mlp_shape,
        mlp_weights,
        2e-3,
    );

    // model.layers.0.mlp_norm
    log.info("\n\nTesting model.layers.2.mlp_norm layer:", .{});

    const mlp_norm_shape = try zml.aio.populateModelWithPrefix(
        zml.nn.LayerNorm,
        model_arena,
        weights_file,
        "model.layers.2.mlp_norm",
    );

    const mlp_norm_weights = try zml.aio.loadModelBuffersWithPrefix(
        zml.nn.LayerNorm,
        mlp_norm_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.layers.2.mlp_norm",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2.mlp_norm",
        mlp_norm_shape,
        mlp_norm_weights,
        1e-4,
    );

    // model.layers.2.attn
    log.info("\n\nTesting model.layers.2.attn layer:", .{});

    var attn_shape = try zml.aio.populateModelWithPrefix(
        modernbert_module.ModernBertAttention,
        model_arena,
        weights_file,
        "model.layers.2.attn",
    );
    attn_shape.num_heads = modernbert_base_options.num_attention_heads;

    const attn_weights = try zml.aio.loadModelBuffersWithPrefix(
        modernbert_module.ModernBertAttention,
        attn_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.layers.2.attn",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2.attn",
        attn_shape,
        attn_weights,
        1e-3,
    );

    // model.layers.3.attn
    log.info("\n\nTesting model.layers.3.attn layer:", .{});

    var attn_global_shape = try zml.aio.populateModelWithPrefix(
        modernbert_module.ModernBertAttention,
        model_arena,
        weights_file,
        "model.layers.3.attn",
    );
    attn_global_shape.is_global_attention = true;
    attn_global_shape.num_heads = modernbert_base_options.num_attention_heads;

    const attn_global_weights = try zml.aio.loadModelBuffersWithPrefix(
        modernbert_module.ModernBertAttention,
        attn_global_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.layers.3.attn",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.3.attn",
        attn_global_shape,
        attn_global_weights,
        1e-5,
    );

    // model.layers.2
    log.info("\n\nTesting model.layers.2 layer:", .{});

    var layer_shape = try zml.aio.populateModelWithPrefix(
        modernbert_module.ModernBertEncoderLayer,
        model_arena,
        weights_file,
        "model.layers.2",
    );
    layer_shape.attn.num_heads = modernbert_base_options.num_attention_heads;

    const layer_weights = try zml.aio.loadModelBuffersWithPrefix(
        modernbert_module.ModernBertEncoderLayer,
        layer_shape,
        weights_file,
        model_arena,
        compute_platform,
        "model.layers.2",
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2",
        layer_shape,
        layer_weights,
        2e-3,
    );

    // model
    log.info("\n\nTesting model layer:", .{});

    var model = try zml.aio.populateModel(
        modernbert_module.ModernBertForMaskedLM,
        model_arena,
        weights_file,
    );

    model.init(modernbert_base_options);

    const model_weights = try zml.aio.loadModelBuffersWithPrefix(modernbert_module.ModernBertForMaskedLM, model, weights_file, model_arena, compute_platform, "");

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model",
        model.model,
        model_weights.model,
        1e-2,
    );

    const TiedDecoder = struct {
        weight: Tensor,
        bias: Tensor,

        pub fn forward(self: @This(), head_outputs: Tensor) Tensor {
            log.warn("TiedDecoder({}, {})", .{ self.weight, head_outputs });

            const res = head_outputs.withTags(.{ .b, .s, .d }).dot(self.weight.withTags(.{ .voc, .d }), .{.d});
            return res.add(self.bias.withTags(.{.voc}).broad(res.shape()));
        }
    };

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.decoder",
        TiedDecoder{ .weight = model.decoder.weight orelse model.model.embeddings.tok_embeddings.weight, .bias = model.decoder.bias },
        .{ .weight = model_weights.model.embeddings.tok_embeddings.weight, .bias = model_weights.decoder.bias },
        1e-3,
    );
    // head
    log.info("\n\nTesting model layer:", .{});
    // log.info("ModernBertPredictionHead: {}", .{head_shape});

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.head",
        model.head,
        model_weights.head,
        0.1, // TODO: too high tolerance
    );

    // for (0..weights_file.buffers.count()) |i| {
    //     if (std.mem.indexOf(u8, weights_file.buffers.entries.get(i).key, "decoder") != null or
    //         std.mem.indexOf(u8, weights_file.buffers.entries.get(i).key, "head") != null)
    //         log.info("weights {} - {s}: {s}", .{ i, weights_file.buffers.entries.get(i).key, weights_file.buffers.entries.get(i).value.shape() });
    // }
    // for (0..activations.buffers.count()) |i| {
    //     if (std.mem.indexOf(u8, activations.buffers.entries.get(i).key, "decoder") != null or
    //         std.mem.indexOf(u8, activations.buffers.entries.get(i).key, "head") != null)
    //         log.info("activations {} - {s}: {s}", .{ i, activations.buffers.entries.get(i).key, activations.buffers.entries.get(i).value.shape() });
    // }

    // ModernBertForMaskedLM
    log.info("\n\nTesting ModernBertForMaskedLM:", .{});

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model",
        model,
        model_weights,
        0.1, // TODO: too high tolerance
    );
}
