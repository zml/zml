const std = @import("std");
const async = @import("async");
const zml = @import("zml");
const qwen = @import("qwen_3_vl.zig");
const clap = @import("clap");
const stdx = @import("stdx");
const floats = zml.floats;

const shapesOf = zml.shapesOf;

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

    const model_tokenizer_path = try std.fs.path.join(allocator, &.{ hf_model_path, "tokenizer.json" });
    defer allocator.free(model_tokenizer_path);

    var tokenizer = blk: {
        log.info("Loading tokenizer from {s}", .{model_tokenizer_path});
        var timer = try stdx.time.Timer.start();
        defer log.info("Loaded tokenizer from {s} [{D}]", .{ model_tokenizer_path, timer.read() });

        break :blk try zml.tokenizer.Tokenizer.fromFile(allocator, model_tokenizer_path);
    };
    errdefer tokenizer.deinit();

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
    try ensureStoreFloat32(&activation_buffer_store);

    /////////////////Ajout du kv cache ///////////////////////////////
    // Récupérer les paramètres nécessaires depuis la config
    const num_layers = config.text_config.num_hidden_layers;
    const num_kv_heads = config.text_config.num_key_value_heads;
    const max_seq_len = 3000; // 256 dans votre cas
    const hidden_size = config.text_config.hidden_size;
    _ = hidden_size; // autofix
    const num_heads = config.text_config.num_attention_heads;
    _ = num_heads; // autofix
    const head_dim = config.text_config.head_dim; // dimension par tête

    const batch_size: u32 = 1; // ajustez selon vos besoins
    _ = batch_size; // autofix

    // Créer les HostBuffers pour le KV cache
    // Forme: [num_layers, batch_size, num_kv_heads, max_seq_len, head_dim]
    const kv_shape = zml.Shape.init(.{
        @as(i64, num_layers),
        @as(i64, max_seq_len),
        @as(i64, num_kv_heads),
        @as(i64, head_dim),
    }, .f32);

    // Créer des buffers vides (initialisés à zéro)
    var kv_k_buffer = try zml.HostBuffer.empty(activation_buffer_store.arena.allocator(), kv_shape);
    var kv_v_buffer = try zml.HostBuffer.empty(activation_buffer_store.arena.allocator(), kv_shape);

    // Initialiser à zéro (optionnel, empty() laisse la mémoire non initialisée)
    // Si vous voulez initialiser à zéro ou à une valeur spécifique:
    {
        const k_items = kv_k_buffer.mutItems(f32);
        @memset(k_items, 0);

        const v_items = kv_v_buffer.mutItems(f32);
        @memset(v_items, 0);
    }

    // Ajouter au BufferStore
    try activation_buffer_store.buffers.put(activation_buffer_store.arena.allocator(), "model.in.5", kv_k_buffer);

    try activation_buffer_store.buffers.put(activation_buffer_store.arena.allocator(), "model.in.6", kv_v_buffer);

    // Ajouter le layer_index (scalaire u32 initialisé à 0)
    const layer_index_shape = zml.Shape.init(.{}, .u32);
    var layer_index_data: [1]u32 = .{0};
    const layer_index_buffer = zml.HostBuffer.fromSlice(layer_index_shape, &layer_index_data);

    try activation_buffer_store.buffers.put(activation_buffer_store.arena.allocator(), "model.in.7", layer_index_buffer);
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
    try ensureStoreFloat32(&store);
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
    // Unbuffered writing of the tokens to stdout.
    // var stdout = std.fs.File.stdout().writer(&.{});
    // const output = try testFinalOutput(platform, activation_buffer_store, "model", qwen_tensors, qwen_buffers, tokenizer, &stdout.interface);
    // defer output.deinit();
    // const host_output = try output.toHostAlloc(allocator);

    // defer host_output.deinit(allocator);
    // log.info("host_output: {f}", .{host_output});
    //try testImplementation(platform, qwen_tensors, qwen_buffers, activation_buffer_store);
}

const ConvertError = error{
    UnsupportedFloatType,
};

fn ensureStoreFloat32(store: *zml.aio.BufferStore) !void {
    var it = store.buffers.iterator();
    const allocator = store.arena.allocator();
    while (it.next()) |entry| {
        const dtype = entry.value_ptr.shape().dtype();
        if (!dtype.isFloat() or dtype == .f32) continue;
        entry.value_ptr.* = try hostBufferToF32(allocator, entry.value_ptr.*);
    }
}

fn hostBufferToF32(allocator: std.mem.Allocator, src: zml.HostBuffer) !zml.HostBuffer {
    const dtype = src.shape().dtype();
    if (!dtype.isFloat() or dtype == .f32) return src;

    const dst_shape = src.shape().withDtype(.f32);
    const element_count = @as(usize, @intCast(dst_shape.count()));
    const data = try allocator.alloc(f32, element_count);
    const out = data;

    switch (dtype) {
        .bf16 => {
            const values = src.items(floats.BFloat16);
            for (values, 0..) |value, idx| {
                out[idx] = floats.BFloat16.toF32(value);
            }
        },
        .f16 => {
            const values = src.items(f16);
            for (values, 0..) |value, idx| {
                out[idx] = @as(f32, @floatCast(value));
            }
        },
        .f64 => {
            const values = src.items(f64);
            for (values, 0..) |value, idx| {
                out[idx] = @as(f32, @floatCast(value));
            }
        },
        else => return ConvertError.UnsupportedFloatType,
    }

    return zml.HostBuffer.fromBytes(dst_shape, std.mem.sliceAsBytes(out));
}

pub fn testFinalOutput(
    platform: zml.Platform,
    activations: zml.aio.BufferStore,
    comptime name: []const u8,
    model: anytype,
    model_weights: zml.Bufferized(@TypeOf(model)),
    tokenizer: zml.tokenizer.Tokenizer,
    writer: *std.Io.Writer,

    //tokenizer: zml.tokenizer.Tokenizer,
) !zml.Buffer {
    var arena = std.heap.ArenaAllocator.init(std.heap.c_allocator);
    defer arena.deinit();
    const alloc = arena.allocator();
    const fwd = @TypeOf(model).forward;
    const FwdSign = zml.ModuleSignature(fwd);

    const input_tensors = try zml.aio.populateModelWithPrefix(FwdSign.ArgsT, alloc, activations, name ++ ".in");
    const input_shapes = try shapesOf(input_tensors, alloc);

    // const n_in = zml.module.countTensors(&input_tensors);
    // const n_in_exp = activations.countLayers(name ++ ".in");
    // if (n_in != n_in_exp) {
    //     log.warn("Reference models uses {d} inputs, but implementation uses {d}", .{ n_in_exp, n_in });
    // }
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const exe = try zml.compileModel(alloc, fwd, model, input_shapes, platform);
    const mod = exe.prepare(model_weights);
    const FetchCtx = struct {
        store: zml.aio.BufferStore,
        index: u32,
        prefix: std.ArrayList(u8),
        platform: zml.Platform,

        fn fetch(ctx: *@This(), x: zml.Tensor) zml.Buffer {
            _ = x;
            defer ctx.index += 1;
            var full_prefix = ctx.*.prefix;
            _ = full_prefix.writer(undefined).print("{d}", .{ctx.index}) catch unreachable;
            const host = ctx.store.get(full_prefix.items) orelse {
                log.err("Didn't find test input: {s}", .{full_prefix.items});
                @panic("Missing test input");
            };
            return host.toDevice(ctx.platform) catch unreachable;
        }
    };

    // Note: zml.populateModelWithPrefix isn't enough,
    // because it assumes we have the same structure in the activation file
    // than in the function signature.
    // But for sake of decoupling the reference implementation
    // and ZML code that's not always the case.

    var input_buffers: zml.Bufferized(FwdSign.ArgsT) = undefined;
    var fetch_ctx: FetchCtx = .{ .store = activations, .index = 0, .prefix = .{}, .platform = platform };
    try fetch_ctx.prefix.ensureTotalCapacity(alloc, name.len + 32);
    fetch_ctx.prefix.appendSliceAssumeCapacity(name ++ ".in.");
    try zml.meta.mapAlloc(FetchCtx.fetch, alloc, &fetch_ctx, input_tensors, &input_buffers);
    defer zml.aio.unloadBuffers(&input_buffers);
    const output_tuple = mod.call(input_buffers);
    const output = output_tuple[0];
    const cache_position = output_tuple[1];
    const kv_cache = output_tuple[2];
    const mrope_position_deltas = output_tuple[3];
    const fwd_decode = @TypeOf(model).forward_decode;

    const exe_decode = try zml.compileModel(alloc, fwd_decode, model, .{ output.shape(), cache_position.shape(), qwen.KvCache.initShape(kv_cache.k.shape()), mrope_position_deltas.shape() }, platform);
    const mod_decode = exe_decode.prepare(model_weights);
    const decode_steps = 20;

    var cur_token = output;
    var cur_position = cache_position;
    var cur_cache = kv_cache;
    var cur_mrope = mrope_position_deltas;
    var generated = try std.ArrayList(u32).initCapacity(alloc, decode_steps);
    defer generated.deinit(alloc);

    var token_host = [_]u32{0};

    for (0..decode_steps) |_| {
        const new_token_buf, const new_position, const updated_cache, const updated_mrope =
            mod_decode.call(.{ cur_token, cur_position, cur_cache, cur_mrope });

        _ = try new_token_buf.toHost(std.mem.sliceAsBytes(&token_host));
        const generated_token = token_host[0];
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try writer.writeAll(chunk);
        }
        try generated.append(alloc, token_host[0]);

        cur_token = new_token_buf;
        cur_position = new_position;
        cur_cache = updated_cache;
        cur_mrope = updated_mrope;
    }
    if (generated.items.len != 0) {
        const generated_text = try tokenizer_decoder.decode(generated.items);
        std.debug.print("{s}\n", .{generated_text});
        for (generated.items) |token| std.debug.print("{d} ", .{token});
        std.debug.print("\n", .{});
    }
    return output;
}

pub fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, prompt: []const u8) ![]const u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const im_start_id = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end_id = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
    const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
    const vision_start_id = tokenizer.tokenToId("<|vision_start|>") orelse return error.NoSuchToken;
    const vision_end_id = tokenizer.tokenToId("<|vision_end|>") orelse return error.NoSuchToken;
    const image_pad_id = tokenizer.tokenToId("<|image_pad|>") orelse return error.NoSuchToken;
    const newline = (try encoder.encode("\n"))[0];

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.appendSlice(allocator, &.{ im_start_id, user, newline });
    try tokens.appendSlice(allocator, &.{ vision_start_id, image_pad_id, vision_end_id });
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.appendSlice(allocator, &.{ im_end_id, newline });
    try tokens.appendSlice(allocator, &.{ im_start_id, assistant, newline });
    return tokens.toOwnedSlice(allocator);
}

pub fn generateText(
    config: qwen.Qwen3VL.Config,
    qwen_: qwen.Qwen3VL,
    mod: zml.ModuleExe(qwen.Qwen3VL.forward),
    tokenizer: zml.tokenizer.Tokenizer,
    allocator: std.mem.Allocator,
    prompt: []const u8,
    writer: *std.Io.Writer,
) !void {
    _ = config; // autofix
    _ = qwen_; // autofix
    _ = mod; // autofix
    _ = tokenizer; // autofix
    _ = allocator; // autofix
    _ = prompt; // autofix
    _ = writer; // autofix
}

fn testImplementation(
    platform: zml.Platform,
    qwen_model: qwen.Qwen3VL,
    qwen_weights: zml.Bufferized(qwen.Qwen3VL),
    activations: zml.aio.BufferStore,
) !void {
    try zml.testing.testLayer(platform, activations, "model", qwen_model, qwen_weights, 5e-2);
    //try zml.testing.testLayer(platform, activations, "model.visual", qwen_model.vision_transformer, qwen_weights.vision_transformer, 5e-3);
    //try zml.testing.testLayer(platform, activations, "model.language_model", qwen_model.text_model, qwen_weights.text_model, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.language_model.embed_tokens", qwen_model.text_model.embed_tokens, qwen_weights.text_model.embed_tokens, 1e-4);
    //try zml.testing.testLayer(platform, activations, "model.language_model.rotary_emb", qwen_model.text_model.rotary_embed, {}, 1e-4);

    // inline for (0..36) |i| {
    //     const name = std.fmt.comptimePrint("model.language_model.layers.{d}", .{i});

    //     // try zml.testing.testLayer(platform, activations, name ++ ".input_layernorm", qwen_model.text_model.layers[i].input_layernorm, qwen_weights.text_model.layers[i].input_layernorm, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".post_attention_layernorm", qwen_model.text_model.layers[i].post_attention_layernorm, qwen_weights.text_model.layers[i].post_attention_layernorm, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".self_attn.q_proj", qwen_model.text_model.layers[i].self_attn.q_proj, qwen_weights.text_model.layers[i].self_attn.q_proj, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".self_attn.q_norm", qwen_model.text_model.layers[i].self_attn.q_norm, qwen_weights.text_model.layers[i].self_attn.q_norm, 5e-3);

    //     // try zml.testing.testLayer(platform, activations, name ++ ".self_attn.k_proj", qwen_model.text_model.layers[i].self_attn.k_proj, qwen_weights.text_model.layers[i].self_attn.k_proj, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".self_attn.k_norm", qwen_model.text_model.layers[i].self_attn.k_norm, qwen_weights.text_model.layers[i].self_attn.k_norm, 5e-3);

    //     // try zml.testing.testLayer(platform, activations, name ++ ".self_attn.v_proj", qwen_model.text_model.layers[i].self_attn.v_proj, qwen_weights.text_model.layers[i].self_attn.v_proj, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".self_attn.o_proj", qwen_model.text_model.layers[i].self_attn.o_proj, qwen_weights.text_model.layers[i].self_attn.o_proj, 1e-4);

    //     // try zml.testing.testLayer(platform, activations, name ++ ".mlp.up_proj", qwen_model.text_model.layers[i].mlp.up_proj, qwen_weights.text_model.layers[i].mlp.up_proj, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".mlp.gate_proj", qwen_model.text_model.layers[i].mlp.gate_proj, qwen_weights.text_model.layers[i].mlp.gate_proj, 1e-4);
    //     // try zml.testing.testLayer(platform, activations, name ++ ".mlp.down_proj", qwen_model.text_model.layers[i].mlp.down_proj, qwen_weights.text_model.layers[i].mlp.down_proj, 5e-3);
    //     try zml.testing.testLayer(platform, activations, name ++ ".self_attn", qwen_model.text_model.layers[i].self_attn, qwen_weights.text_model.layers[i].self_attn, 5e-3);
    //     try zml.testing.testLayer(platform, activations, name ++ ".mlp", qwen_model.text_model.layers[i].mlp, qwen_weights.text_model.layers[i].mlp, 5e-3);
    // }

    //try zml.testing.testLayer(platform, activations, "model.language_model", qwen_model.text_model, qwen_weights.text_model, 1e-1);
    //try zml.testing.testLayer(platform, activations, "model.visual.patch_embed", qwen_model.vision_transformer.vision_patch_embed, qwen_weights.vision_transformer.vision_patch_embed, 1e-4);
    //try zml.testing.testLayer(platform, activations, "model.visual.rotary_pos_embed", qwen_model.vision_transformer.rotary_pos_emb, {}, 1e-4);
    //try zml.testing.testLayer(platform, activations, "model.visual.merger", qwen_model.vision_transformer.patch_merger, qwen_weights.vision_transformer.patch_merger, 5e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0", qwen_model.vision_transformer.deepstack_patch_mergers[0], qwen_weights.vision_transformer.deepstack_patch_mergers[0], 5e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.1", qwen_model.vision_transformer.deepstack_patch_mergers[1], qwen_weights.vision_transformer.deepstack_patch_mergers[1], 5e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.2", qwen_model.vision_transformer.deepstack_patch_mergers[2], qwen_weights.vision_transformer.deepstack_patch_mergers[2], 5e-4);

    // inline for (0..24) |i| {
    //     const name = std.fmt.comptimePrint("model.visual.blocks.{d}", .{i});
    //     try zml.testing.testLayer(platform, activations, name, qwen_model.vision_transformer.blocks[i], qwen_weights.vision_transformer.blocks[i], 1e-3); // pas de poids ici meilleure precision on peurt en deduire peut etre problee de conversion sur le reste
    //     try zml.testing.testLayer(platform, activations, name ++ ".mlp", qwen_model.vision_transformer.blocks[i].mlp, qwen_weights.vision_transformer.blocks[i].mlp, 1e-4); //ne passe pas en 1e-2 sur le dernier block mlp
    //     try zml.testing.testLayer(platform, activations, name ++ ".attn", qwen_model.vision_transformer.blocks[i].attn, qwen_weights.vision_transformer.blocks[i].attn, 1e-4); //pete sur la meme que norm2 logique
    //     try zml.testing.testLayer(platform, activations, name ++ ".attn.qkv", qwen_model.vision_transformer.blocks[i].attn.qkv, qwen_weights.vision_transformer.blocks[i].attn.qkv, 1e-4);
    //     try zml.testing.testLayer(platform, activations, name ++ ".norm1", qwen_model.vision_transformer.blocks[i].norm1, qwen_weights.vision_transformer.blocks[i].norm1, 5e-4);
    //     try zml.testing.testLayer(platform, activations, name ++ ".norm2", qwen_model.vision_transformer.blocks[i].norm2, qwen_weights.vision_transformer.blocks[i].norm2, 1e-4); // une norm qui plante a 1e-2
    //     try zml.testing.testLayer(platform, activations, name ++ ".norm1", qwen_model.vision_transformer.blocks[i].norm1, qwen_weights.vision_transformer.blocks[i].norm1, 1e-4);
    //     try zml.testing.testLayer(platform, activations, name ++ ".norm2", qwen_model.vision_transformer.blocks[i].norm2, qwen_weights.vision_transformer.blocks[i].norm2, 1e-4);
    //     try zml.testing.testLayer(platform, activations, name ++ ".attn.proj", qwen_model.vision_transformer.blocks[i].attn.proj, qwen_weights.vision_transformer.blocks[i].attn.proj, 1e-4);
    //     try zml.testing.testLayer(platform, activations, name ++ ".mlp.linear_fc1", qwen_model.vision_transformer.blocks[i].mlp.linear_fc1, qwen_weights.vision_transformer.blocks[i].mlp.linear_fc1, 1e-4);
    //     try zml.testing.testLayer(platform, activations, name ++ ".mlp.linear_fc2", qwen_model.vision_transformer.blocks[i].mlp.linear_fc2, qwen_weights.vision_transformer.blocks[i].mlp.linear_fc2, 1e-4);
    // }

    //try zml.testing.testLayer(platform, activations, "model", qwen_model, qwen_weights, 1e-2);
    //try zml.testing.testLayer(platform, activations, "model.visual.pos_embed", qwen_model.vision_transformer.pos_embed, qwen_weights.vision_transformer.pos_embed, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual.rotary_pos_emb", qwen_model.vision_transformer.rotary_pos_emb, {}, 1e-3);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.norm1", qwen_model.vision_transformer.blocks[0].norm1, qwen_weights.vision_transformer.blocks[0].norm1, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.norm2", qwen_model.vision_transformer.blocks[0].norm2, qwen_weights.vision_transformer.blocks[0].norm2, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.attn.proj", qwen_model.vision_transformer.blocks[0].attn.proj, qwen_weights.vision_transformer.blocks[0].attn.proj, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.mlp.linear_fc1", qwen_model.vision_transformer.blocks[0].mlp.linear_fc1, qwen_weights.vision_transformer.blocks[0].mlp.linear_fc1, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.mlp.linear_fc2", qwen_model.vision_transformer.blocks[0].mlp.linear_fc2, qwen_weights.vision_transformer.blocks[0].mlp.linear_fc2, 1e-2);
    // try zml.testing.testLayer(platform, activations, "model.visual.merger.linear_fc1", qwen_model.vision_transformer.patch_merger.linear_fc1, qwen_weights.vision_transformer.patch_merger.linear_fc1, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.merger.linear_fc2", qwen_model.vision_transformer.patch_merger.linear_fc2, qwen_weights.vision_transformer.patch_merger.linear_fc2, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.merger.norm", qwen_model.vision_transformer.patch_merger.norm, qwen_weights.vision_transformer.patch_merger.norm, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0.linear_fc1", qwen_model.vision_transformer.deepstack_patch_mergers[0].linear_fc1, qwen_weights.vision_transformer.deepstack_patch_mergers[0].linear_fc1, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0.linear_fc2", qwen_model.vision_transformer.deepstack_patch_mergers[0].linear_fc2, qwen_weights.vision_transformer.deepstack_patch_mergers[0].linear_fc2, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.deepstack_merger_list.0.norm", qwen_model.vision_transformer.deepstack_patch_mergers[0].norm, qwen_weights.vision_transformer.deepstack_patch_mergers[0].norm, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.attn.qkv", qwen_model.vision_transformer.blocks[0].attn.qkv, qwen_weights.vision_transformer.blocks[0].attn.qkv, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.attn", qwen_model.vision_transformer.blocks[0].attn, qwen_weights.vision_transformer.blocks[0].attn, 1e-4);
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0.mlp", qwen_model.vision_transformer.blocks[0].mlp, qwen_weights.vision_transformer.blocks[0].mlp, 1e-4); //precision pas top sur le gelu
    // try zml.testing.testLayer(platform, activations, "model.visual.blocks.0", qwen_model.vision_transformer.blocks[0], qwen_weights.vision_transformer.blocks[0], 1e-4); //idem logiquement
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
    //try zml.testing.testLayer(platform, activations, "model.visual", qwen_model.vision_transformer, qwen_weights.vision_transformer, 1e-2);
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
