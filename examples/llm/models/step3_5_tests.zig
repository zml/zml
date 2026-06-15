const std = @import("std");

const zml = @import("zml");

const common = @import("common.zig");
const inference = @import("step3_5flash/inference.zig");
const step3p5flash = @import("step3_5flash.zig");
const model = @import("step3_5flash/model.zig");

pub const std_options: std.Options = .{
    .log_level = .info,
};

const Args = struct {
    model: []const u8,
    activations: []const u8,
    generate: bool = false,

    pub const help =
        \\Use step3_5_tests --model=<path> --activations=<path>
        \\
        \\ Validate the Step 3.5 Flash MoE layers against activation fixtures.
        \\
        \\ Options:
        \\   --model=<path>         Path to the model repository
        \\   --activations=<path>   Path to activation safetensors
        \\   --generate=<bool>      Testing full model
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

    // Registry stores the memory of tensors
    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    // Tensor store is a ZML representation of tensors - i.e., parent.child.leaf
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer activation_store.deinit();

    const shardings: common.Shardings = try .init(platform);

    var repo_model = try step3p5flash.LoadedModel.init(allocator, io, repo, store.view(), shardings);
    defer repo_model.deinit(allocator);

    const compilation_parameters = inference.CompilationParameters.init(repo_model.inner, repo_model.parsed_config.value, 64, zml.moe.Backend.triton, shardings);

    // Loading bar (single global Progress)
    var progress = std.Progress.start(io, .{ .root_name = args.model });
    defer progress.end();

    var model_buffers = try repo_model.loadBuffers(allocator, io, platform, &store, &progress, shardings);
    defer repo_model.unloadBuffers(&model_buffers, allocator);

    // Token history: seed with BOS (or whatever prompt tokens you have).
    // For a pure-decode smoke test, start with one token so the loop has something to embed.
    var all_tokens: std.ArrayList(u32) = .empty;
    defer all_tokens.deinit(allocator);

    const generated_token_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);
    generated_token_slice.items(u32)[0] = all_tokens.items[all_tokens.items.len - 1];

    var stdout_buf: [4096]u8 = undefined;
    const stdout: std.Io.File = .stdout();
    const writer = stdout.writer(io, &stdout_buf);

    if (args.generate == true) {
        try runDecode((allocator), io, platform, shardings, inference.CompiledModel.init(allocator, io, platform, &repo_model, repo_model.inner, compilation_parameters, &progress), &model_buffers.text_model, all_tokens, writer);
    }

    // try run(
    //     allocator,
    //     io,
    //     platform,
    //     args.activations,
    //     &store,
    //     shardings,
    //     &repo_model.inner.text_model,
    //     &model_buffers.text_model,
    // );
}

// pub fn run(
//     allocator: std.mem.Allocator,
//     io: std.Io,
//     platform: *zml.Platform,
//     activations_path: []const u8,
//     model_store: *zml.io.TensorStore,
//     shardings: common.Shardings,
//     text_model: *const model.TextModel,
//     text_buffers: *zml.Bufferized(model.TextModel),
// ) !void {
//     var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_path);
//     defer registry.deinit();

//     var activation_store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
//     defer activation_store.deinit();
// }

fn runDecode(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, shardings: common.Shardings, compiled_model: *const inference.CompiledModel, model_buffers: *model.Buffers, all_tokens: *std.ArrayList(u32), stdout: *std.Io.Writer) !void {
    var layer_index_buffers: [45]@TypeOf(.{ .self_attn = undefined }) = .init();
    const tokenizer = zml.tokenizer.Tokenizer;

    // kv cache buffers
    var kv_cache_buffers = try compiled_model.params.kv_cache.initBuffer(io, platform, shardings.replicated);
    errdefer model.KvCache.deinitBuffer(&kv_cache_buffers);

    // decode buffers
    var decode_moe_metadata_buffers = try compiled_model.params.decode_moe_metadata.initBuffer(io, platform);
    errdefer zml.moe.Metadata.deinitBuffer(&decode_moe_metadata_buffers);

    // seed
    // rng buffers
    const seed: u128 = @intCast(std.Io.Clock.now(.real, io).toNanoseconds());
    // flag shardings here
    var rng_buffers = try zml.Tensor.Rng.initBuffer(io, platform, .replicated, seed);
    errdefer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    // go through all layers and create buffer for attention
    // hardcoded for now TODO
    for (0..45) |layer_index| {
        const layer_index_buffer = try zml.Buffer.scalar(io, platform, @intCast(layer_index), .u32);
        layer_index_buffers[layer_index] = .{ .self_attn = layer_index_buffer };
    }

    // decode token index buffer
    var decode_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32);
    errdefer decode_token_index_buffer.deinit();

    // create kv cache for each buffer
    var self_attn_layers_caches = try allocator.alloc(zml.Bufferized(model.KvCache));
    errdefer allocator.free(self_attn_layers_caches);

    // create decoder
    var decoder = try tokenizer.decoder();
    defer decoder.deinit();

    // out tokens buffer
    const out_tokens_buffer: []u8 = try allocator.alloc(u8, 1024);
    defer allocator.free(out_tokens_buffer);

    const hidden_size = compiled_model.loaded_model.parsed_config.value.hidden_size;
    const model_dtype = compiled_model.loaded_model.inner.text_model.embed_tokens.weight.dtype();

    const decode_hidden_shape = zml.Shape.init(.{ .b = 1, .s = 1, .d = hidden_size }, model_dtype).withPartitioning(.{
        .b = .replicated,
        .s = .replicated,
        .d = .replicated,
    });

    const generated_token_slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));

    // create embedding args, results (not buffers idt)
    var embedding_decode_args = try compiled_model.decode_embedding_exe.args(allocator);
    defer embedding_decode_args.deinit(allocator);
    var embedding_decode_results = try compiled_model.decode_embedding_exe.results(allocator);
    defer embedding_decode_results.deinit(allocator);

    // create full layer args, results
    var sampling_decode_args = try compiled_model.decode_sampling_exe.args(allocator);
    defer sampling_decode_args.deinit(allocator);
    var sampling_decode_results = try compiled_model.decode_sampling_exe.results(allocator);
    defer sampling_decode_results.deinit(allocator);

    var decode_hidden_buffer = try zml.Buffer.uninitialized(io, platform, decode_hidden_shape, shardings.replicated, .{});
    defer decode_hidden_buffer.deinit();

    var current_token_buffer = try zml.Buffer.fromSlice(io, platform, generated_token_slice, shardings.replicated);
    defer current_token_buffer.deinit();

    var token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, @intCast(all_tokens.items.len)), .u32);
    defer token_index_buffer.deinit();

    generation: while (true) {
        const token_id = generated_token_slice.items(u32)[0];
        // if (token_id == compiled_model.loaded_model.inner.special_tokens.end_of_text_token_id) break :generation;

        const token = try decoder.feedOne(token_id, out_tokens_buffer);
        try stdout.writeAll(token);

        try all_tokens.append(allocator, token_id);
        if (all_tokens.items.len >= compiled_model.params.seqlen) break :generation;

        embedding_decode_args.set(.{ model_buffers.text_model.embed_tokens, current_token_buffer });
        compiled_model.decode_embedding_exe.call(embedding_decode_args, &embedding_decode_results);
        embedding_decode_results.fill(.{&decode_hidden_buffer});

        for (model_buffers.text_model.layers, 0..) |layer_weights, layer_index| {
            const exe = compiled_model.decode_full_layer_exe;

            self_attn_layers_caches[layer_index].k = kv_cache_buffers.self_attn.k;
            self_attn_layers_caches[layer_index].k = kv_cache_buffers.self_attn.v;

            exe.args(allocator).?.set(.{
                layer_weights,
                decode_hidden_buffer,
                token_index_buffer,
                self_attn_layers_caches[layer_index],
                compiled_model.loaded_model.inner.config,
                decode_moe_metadata_buffers,
            });

            exe.call(exe.args(allocator).?, &exe.results(allocator).?);
            &exe.results(allocator).?.fill(.{ &decode_hidden_buffer, &kv_cache_buffers.self_attn });
        }

        sampling_decode_args.set(.{ .{
            .norm = model_buffers.text_model.norm,
            .lm_head = model_buffers.text_model.lm_head,
            .gen_options = compiled_model.loaded_model.inner.text_model.gen_options,
        }, decode_hidden_buffer, rng_buffers, token_index_buffer });
        compiled_model.decode_sampling_exe.call(sampling_decode_args, &sampling_decode_results);
        sampling_decode_results.fill(.{ &current_token_buffer, &rng_buffers, &token_index_buffer });

        try current_token_buffer.toSlice(io, generated_token_slice);
    }

    // write
    try stdout.writeAll(try decoder.finalize(out_tokens_buffer));
    try stdout.flush();
}

// fn loadBufferFromStore(allocator: std.mem.Allocator, io: anytype, platform: *zml.Platform, store: *zml.io.TensorStore, key: []const u8, sharding: zml.Sharding) !zml.Buffer {
//     const shape = store.view().getShape(key) orelse return error.NotFound;

//     const host_bytes = try allocator.alloc(u8, shape.byteSize());
//     defer allocator.free(host_bytes);

//     var io_buffer: [8 * 1024]u8 = undefined;
//     var reader = try store.view().getReader(key, io, &io_buffer);
//     defer reader.deinit();

//     _ = try reader.interface.readSliceAll(host_bytes);

//     return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
// }
