const clap = @import("clap");
const std = @import("std");
const zml = @import("zml");
const asynk = @import("async");
const log = std.log;
const Tensor = zml.Tensor;
const modernbert_module = @import("modernbert.zig");
const ModernBertOptions = modernbert_module.ModernBertOptions;

const params = clap.parseParamsComptime(
    \\--help                    print this help
    \\--model           <PATH>  model weights path
    \\--activations     <PATH>  model activations path
);

fn printUsageAndExit(stderr: anytype) noreturn {
    stderr.print("usage: ", .{}) catch {};
    clap.usage(stderr, clap.Help, &params) catch {};
    stderr.print("\n", .{}) catch {};
    std.process.exit(0);
}
pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    // Short lived allocations
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();

    const allocator = gpa.allocator();
    const stderr = std.io.getStdErr().writer();

    // Read CLI arguments
    const parsers = comptime .{
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    var res = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        try diag.report(stderr, err);
        try printUsageAndExit(stderr);
    };
    defer res.deinit();

    if (res.args.help != 0) {
        try clap.help(stderr, clap.Help, &params, .{});
        return;
    }

    const model_file = res.args.model.?;
    const activations_file = res.args.activations.?;

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
    var weights_file = try zml.aio.detectFormatAndOpen(allocator, model_file);
    defer weights_file.deinit();
    log.info("Model contains {d} layers. Loaded from: {s}", .{ weights_file.buffers.count(), model_file });

    // Load the activation data file
    const activations = try zml.aio.torch.open(model_arena, activations_file);
    defer activations.deinit();
    log.info("Found {} activations in {s}", .{ activations.buffers.count(), activations_file });

    // Initialize model
    var model = try zml.aio.populateModel(
        modernbert_module.ModernBertForMaskedLM,
        model_arena,
        weights_file,
    );

    const modernbert_base_options: modernbert_module.ModernBertOptions = .{
        .num_attention_heads = 12,
        .tie_word_embeddings = true,
    };
    model.init(modernbert_base_options);

    // Load model weights
    const model_weights = try zml.aio.loadModelBuffers(
        modernbert_module.ModernBertForMaskedLM,
        model,
        weights_file,
        model_arena,
        compute_platform,
    );

    // Test implementation
    try testImplementation(compute_platform, model, model_weights, activations);
}

fn testImplementation(
    compute_platform: zml.Platform,
    model: modernbert_module.ModernBertForMaskedLM,
    model_weights: zml.Bufferized(modernbert_module.ModernBertForMaskedLM),
    activations: zml.aio.BufferStore,
) !void {
    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.embeddings.tok_embeddings",
        model.model.embeddings.tok_embeddings,
        model_weights.model.embeddings.tok_embeddings,
        1e-6,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.embeddings.norm",
        model.model.embeddings.norm,
        model_weights.model.embeddings.norm,
        1e-3,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.embeddings",
        model.model.embeddings,
        model_weights.model.embeddings,
        1e-3,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.final_norm",
        model.model.final_norm,
        model_weights.model.final_norm,
        1e-5,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2.mlp",
        model.model.layers[2].mlp,
        model_weights.model.layers[2].mlp,
        2e-3,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2.mlp_norm",
        model.model.layers[2].mlp_norm,
        model_weights.model.layers[2].mlp_norm,
        1e-4,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2.attn",
        model.model.layers[2].attn,
        model_weights.model.layers[2].attn,
        1e-6,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.2",
        model.model.layers[2],
        model_weights.model.layers[2],
        2e-3,
    );

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.model.layers.3.attn",
        model.model.layers[3].attn,
        model_weights.model.layers[3].attn,
        1e-5,
    );

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
            const results = head_outputs.withTags(.{ .b, .s, .d }).dot(self.weight.withTags(.{ .voc, .d }), .{.d});
            return results.add(self.bias.withTags(.{.voc}).broad(results.shape()));
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

    try zml.testing.testLayer(
        compute_platform,
        activations,
        "model.head",
        model.head,
        model_weights.head,
        0.1, // TODO: too high tolerance
    );
}
