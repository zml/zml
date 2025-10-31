const std = @import("std");
const async = @import("async");
const zml = @import("zml");
const qwen = @import("qwen_3_vl.zig");
const clap = @import("clap");
const stdx = @import("stdx");

const params = clap.parseParamsComptime(
    \\--help                 print this help
    \\--hf-model-path <STRING>   path to the directory containing model weights, config, tokenizer
    \\--activations    <STRING>   path to the activations .pt file
    \\--create-options <STRING>   platform creation options JSON, defaults to {}
);

const log = std.log.scoped(.qwen);

test {
    // enregistre tous les tests
    std.testing.refAllDecls(@This());
}

pub const std_options: std.Options = .{
    .log_level = .debug,
    .logFn = async.logFn(std.log.defaultLog),
    .log_scope_levels = &[_]std.log.ScopeLevel{
        .{ .scope = .@"zml/async", .level = .info },
    },
};

pub fn main() !void {
    try async.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn asyncMain() !void {
    const allocator = std.heap.c_allocator;

    const parsers = comptime .{
        .STRING = clap.parsers.string,
        .PATH = clap.parsers.string,
    };
    var diag: clap.Diagnostic = .{};
    var stderr_buffer: [1024]u8 = undefined;
    var stderr = std.fs.File.stderr().writer(&stderr_buffer);
    defer stderr.interface.flush() catch {};

    var cli = clap.parse(clap.Help, &params, parsers, .{
        .diagnostic = &diag,
        .allocator = allocator,
    }) catch |err| {
        diag.report(&stderr.interface, err) catch {};
        stderr.interface.writeAll("usage: ") catch {};
        clap.usage(&stderr.interface, clap.Help, &params) catch {};
        stderr.interface.writeAll("\n") catch {};
        return;
    };
    defer cli.deinit();

    if (cli.args.help != 0) {
        clap.help(&stderr.interface, clap.Help, &params, .{}) catch {};
        return;
    }

    const hf_model_path = cli.args.@"hf-model-path" orelse {
        log.err("Missing --hf-model-path", .{});
        return;
    };
    const activations_path = cli.args.activations orelse {
        log.err("Missing --activations", .{});
        return;
    };
    log.info("hf_model_path: {s}", .{hf_model_path});
    log.info("activations_path: {s}", .{activations_path});
    //const hf_model_path = "/home/louis/qwen3-vl-4b-instruct/qwen3-vl-4b-instruct";

    const model_config_path = try std.fs.path.join(allocator, &.{ hf_model_path, "config.json" });
    defer allocator.free(model_config_path);
    log.info("model_config_path: {s}", .{model_config_path});
    const model_weights_path = b: {
        const simple_path = try std.fs.path.join(allocator, &.{ hf_model_path, "model.safetensors" });
        log.info("simple_path: {s}", .{simple_path});
        if (async.File.access(simple_path, .{})) {
            break :b simple_path;
        } else |_| {
            allocator.free(simple_path);
        }

        const sharded_path = try std.fs.path.join(allocator, &.{ hf_model_path, "model.safetensors.index.json" });
        log.info("sharded_path: {s}", .{sharded_path});
        break :b sharded_path;
    };
    log.info("model_weights_path: {s}", .{model_weights_path});
    defer allocator.free(model_weights_path);

    const config = blk: {
        var config_json_file = try async.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(&config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(qwen.Qwen3VL.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };

    // Load the activations.
    //var activation_buffer_store = try zml.aio.torch.open(allocator, "/home/louis/zml/examples/qwen_3_vl/activations/qwen3-vl-4b-instruct.activations.pt");
    var activation_buffer_store = try zml.aio.torch.open(allocator, activations_path);
    log.info("activation_buffer_store: {s}", .{activations_path});
    defer activation_buffer_store.deinit();

    var iterator = activation_buffer_store.buffers.iterator();
    while (iterator.next()) |entry| {
        log.info("Buffer: {s} {f}", .{ entry.key_ptr.*, entry.value_ptr.shape() });
    }

    var context = try zml.Context.init();
    defer context.deinit();

    const compilation_options = zml.CompilationOptions{
        .xla_dump_to = "/tmp/zml/qwen",
        .sharding_enabled = false,
    };

    const create_opts_json = "{}";
    const create_opts = try std.json.parseFromSlice(zml.Platform.CreateOptions, allocator, create_opts_json, .{});
    const platform = context.autoPlatform(create_opts.value).withCompilationOptions(compilation_options);
    create_opts.deinit();
    context.printAvailablePlatforms(platform);

    var store = try zml.aio.detectFormatAndOpen(allocator, model_weights_path);
    defer store.deinit();

    // Options pour Qwen3-VL
    const qwen_options: qwen.Qwen3VL.Options = .{
        .max_seq_len = 256,
        .sampling_strategy = .{
            .topk = 1,
            .temperature = 1.0,
        },
    };

    // Arena pour la compilation
    var compiler_arena = std.heap.ArenaAllocator.init(allocator);
    defer compiler_arena.deinit();

    const qwen_tensors: qwen.Qwen3VL = try .init(compiler_arena.allocator(), config, qwen_options, store);

    log.info("Qwen3-VL loaded!", .{});

    log.info("\tLoading Llama weights from {s}...", .{model_weights_path});
    var qwen_buffers = try store.loadModelById(qwen.Qwen3VL, compiler_arena.allocator(), qwen_tensors, platform);
    defer zml.aio.unloadBuffers(&qwen_buffers);

    try testImplementation(platform, qwen_tensors, qwen_buffers, activation_buffer_store);
}

fn testImplementation(
    platform: zml.Platform,
    qwen_model: qwen.Qwen3VL,
    qwen_weights: zml.Bufferized(qwen.Qwen3VL),
    activations: zml.aio.BufferStore,
) !void {
    try zml.testing.testLayer(platform, activations, "model.visual", qwen_model.vision_transformer, qwen_weights.vision_transformer, 1e-1);
    inline for (0..24) |i| {
        const name = std.fmt.comptimePrint("model.visual.blocks.{d}", .{i});
        try zml.testing.testLayer(platform, activations, name ++ ".mlp.act_fn", qwen_model.vision_transformer.blocks[i].mlp.hidden_act, {}, 1e-2); // pas de poids ici meilleure precision on peurt en deduire peut etre problee de conversion sur le reste
        //try zml.testing.testLayer(platform, activations, name, qwen_model.vision_transformer.blocks[i], qwen_weights.vision_transformer.blocks[i], 1e-1); //pete sur la derniere comme mlp
        try zml.testing.testLayer(platform, activations, name ++ ".mlp", qwen_model.vision_transformer.blocks[i].mlp, qwen_weights.vision_transformer.blocks[i].mlp, 1e-2); //ne passe pas en 1e-2 sur le dernier block mlp
        try zml.testing.testLayer(platform, activations, name ++ ".attn", qwen_model.vision_transformer.blocks[i].attn, qwen_weights.vision_transformer.blocks[i].attn, 1e-1); //pete sur la meme que norm2 logique
        try zml.testing.testLayer(platform, activations, name ++ ".attn.qkv", qwen_model.vision_transformer.blocks[i].attn.qkv, qwen_weights.vision_transformer.blocks[i].attn.qkv, 1e-2);
        try zml.testing.testLayer(platform, activations, name ++ ".norm1", qwen_model.vision_transformer.blocks[i].norm1, qwen_weights.vision_transformer.blocks[i].norm1, 1e-2);
        // try zml.testing.testLayer(platform, activations, name ++ ".norm2", qwen_model.vision_transformer.blocks[i].norm2, qwen_weights.vision_transformer.blocks[i].norm2, 1e-1); // une norm qui plante a 1e-2
        // try zml.testing.testLayer(platform, activations, name ++ ".attn.proj", qwen_model.vision_transformer.blocks[i].attn.proj, qwen_weights.vision_transformer.blocks[i].attn.proj, 1e-2);
        // try zml.testing.testLayer(platform, activations, name ++ ".mlp.linear_fc1", qwen_model.vision_transformer.blocks[i].mlp.linear_fc1, qwen_weights.vision_transformer.blocks[i].mlp.linear_fc1, 1e-2);
        // try zml.testing.testLayer(platform, activations, name ++ ".mlp.linear_fc2", qwen_model.vision_transformer.blocks[i].mlp.linear_fc2, qwen_weights.vision_transformer.blocks[i].mlp.linear_fc2, 1e-2);

    }

    //try zml.testing.testLayer(platform, activations, "model", qwen_model, qwen_weights, 1e-3);
    //try zml.testing.testLayer(platform, activations, "model.visual.pos_embed", qwen_model.vision_transformer.pos_embed, qwen_weights.vision_transformer.pos_embed, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual.rotary_pos_emb", qwen_model.vision_transformer.rotary_pos_emb, {}, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.norm1", qwen_model.vision_transformer.blocks[0].norm1, qwen_weights.vision_transformer.blocks[0].norm1, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.norm2", qwen_model.vision_transformer.blocks[0].norm2, qwen_weights.vision_transformer.blocks[0].norm2, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.attn.proj", qwen_model.vision_transformer.blocks[0].attn.proj, qwen_weights.vision_transformer.blocks[0].attn.proj, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.mlp.linear_fc1", qwen_model.vision_transformer.blocks[0].mlp.linear_fc1, qwen_weights.vision_transformer.blocks[0].mlp.linear_fc1, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.mlp.linear_fc2", qwen_model.vision_transformer.blocks[0].mlp.linear_fc2, qwen_weights.vision_transformer.blocks[0].mlp.linear_fc2, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.merger.linear_fc1", qwen_model.vision_transformer.patch_merger.linear_fc1, qwen_weights.vision_transformer.patch_merger.linear_fc1, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.merger.linear_fc2", qwen_model.vision_transformer.patch_merger.linear_fc2, qwen_weights.vision_transformer.patch_merger.linear_fc2, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.merger.norm", qwen_model.vision_transformer.patch_merger.norm, qwen_weights.vision_transformer.patch_merger.norm, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0.linear_fc1", qwen_model.vision_transformer.deepstack_patch_mergers[0].linear_fc1, qwen_weights.vision_transformer.deepstack_patch_mergers[0].linear_fc1, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0.linear_fc2", qwen_model.vision_transformer.deepstack_patch_mergers[0].linear_fc2, qwen_weights.vision_transformer.deepstack_patch_mergers[0].linear_fc2, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0.norm", qwen_model.vision_transformer.deepstack_patch_mergers[0].norm, qwen_weights.vision_transformer.deepstack_patch_mergers[0].norm, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.attn.qkv", qwen_model.vision_transformer.blocks[0].attn.qkv, qwen_weights.vision_transformer.blocks[0].attn.qkv, 1e-2);
    //try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.attn", qwen_model.vision_transformer.blocks[0].attn, qwen_weights.vision_transformer.blocks[0].attn, 1e-2);
    //try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.mlp", qwen_model.vision_transformer.blocks[0].mlp, qwen_weights.vision_transformer.blocks[0].mlp, 1e-2); //precision pas top sur le gelu
    //try zml.testing.testLayer(platform, activations, "model.visual.blocks.0", qwen_model.vision_transformer.blocks[0], qwen_weights.vision_transformer.blocks[0], 1e-2); //idem logiquement
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.1", qwen_model.vision_transformer.blocks[1], qwen_weights.vision_transformer.blocks[1], 1e-1);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.2", qwen_model.vision_transformer.blocks[2], qwen_weights.vision_transformer.blocks[2], 1e-1);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.3", qwen_model.vision_transformer.blocks[3], qwen_weights.vision_transformer.blocks[3], 1e-1);
    //try zml.testing.testLayer(platform, activations, "model.visual.blocks.4", qwen_model.vision_transformer.blocks[4], qwen_weights.vision_transformer.blocks[4], 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.5", qwen_model.vision_transformer.blocks[5], qwen_weights.vision_transformer.blocks[5], 1e-1);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.6", qwen_model.vision_transformer.blocks[6], qwen_weights.vision_transformer.blocks[6], 1e-1);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.7", qwen_model.vision_transformer.blocks[7], qwen_weights.vision_transformer.blocks[7], 1e-1);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.8", qwen_model.vision_transformer.blocks[8], qwen_weights.vision_transformer.blocks[8], 1e-1);
    //try zml.testing.testLayer(platform, activations, "model.visual.merger", qwen_model.vision_transformer.patch_merger, qwen_weights.vision_transformer.patch_merger, 1e-2);
    //try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0", qwen_model.vision_transformer.deepstack_patch_mergers[0], qwen_weights.vision_transformer.deepstack_patch_mergers[0], 1e-2);
    //try zml.testing.testLayer(platform, activations, "model.language_model.layers.0.self_attn", qwen_model.text_model.layers[0].self_attn, qwen_weights.text_model.layers[0].self_attn, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual", qwen_model.vision_transformer, qwen_weights.vision_transformer, 1e-3);
    //try zml.testing.testLayer(platform, activations, "model.language_model.rotary_emb", qwen_model.text_model.rotary_embed, {}, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.language_model.layers.0.mlp", qwen_model.text_model.layers[0].mlp, qwen_weights.text_model.layers[0].mlp, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.language_model.layers.0.input_layernorm", qwen_model.text_model.layers[0].input_layernorm, qwen_weights.text_model.layers[0].input_layernorm, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.language_model.layers.0.post_attention_layernorm", qwen_model.text_model.layers[0].post_attention_layernorm, qwen_weights.text_model.layers[0].post_attention_layernorm, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.language_model.layers.0", qwen_model.text_model.layers[0], qwen_weights.text_model.layers[0], 1e-1);

    //try zml.testing.testLayer(platform, activations, "model.language_model.layers.0.self_attn", qwen_model.text_model.layers[0].self_attn, qwen_weights.text_model.layers[0].self_attn, 1e-2);

    //try zml.testing.testLayer(platform, activations, "model.visual", qwen_model.vision_transformer, qwen_weights.vision_transformer, 1e-1);
}
// test "Conv3D basic functionality" {
//     // Créer des tensors de test
//     const allocator = std.testing.allocator;
//     const platform = zml.testing.env();

//     const MyModule = struct {
//         pub fn forward(input: zml.Tensor, kernel: zml.Tensor) zml.Tensor {
//             return input.conv3d(kernel, .{
//                 .window_strides = &.{ 2, 14, 14 },
//                 .padding = &.{ 0, 0, 0, 0, 0, 0 },
//                 .input_spatial_dimensions = &.{ 2, 3, 4 },
//             });
//         }
//     };

//     const input_dimz = .{ 1, 3, 8, 224, 224 };
//     const input_shape = zml.Shape.init(input_dimz, .f32);
//     const input_data = try allocator.alloc(f32, input_shape.count());
//     defer allocator.free(input_data);
//     const input = try zml.Buffer.fromSlice(platform, input_shape, input_data);
//     //const buffer_awaited = try input_buf.await();

//     const kernel_dimz = .{ 64, 3, 2, 14, 14 };
//     const kernel_shape = zml.Shape.init(kernel_dimz, .f32);
//     const kernel_data = try allocator.alloc(f32, kernel_shape.count());
//     defer allocator.free(kernel_data);
//     const kernel = try zml.Buffer.fromSlice(platform, kernel_shape, kernel_data);
//     //const buffer_awaited2 = try kernel_buf.await();

//     const exe = try zml.compileFn(allocator, MyModule.forward, .{ input_shape, kernel_shape }, platform);
//     defer exe.deinit();
//     const output = exe.call(.{ input, kernel });
//     const output_cpu = try output.toHostAlloc(allocator);
//     defer output_cpu.deinit(allocator);

//     // Vérifier les dimensions
//     try std.testing.expectEqual(@as(u32, 5), output.shape().rank());
//     try std.testing.expectEqual(@as(u32, 1), output.shape().dim(0)); // batch
//     try std.testing.expectEqual(@as(u32, 64), output.shape().dim(1)); // channels
//     try std.testing.expectEqual(@as(u32, 4), output.shape().dim(2)); // time (8/2)
//     try std.testing.expectEqual(@as(u32, 16), output.shape().dim(3)); // height (224/14)
//     try std.testing.expectEqual(@as(u32, 16), output.shape().dim(4)); // width (224/14)
// }
