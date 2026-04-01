const std = @import("std");
const log = std.log;

pub const std_options: std.Options = .{
    .log_level = .info,
};

const zml = @import("zml");
const Tensor = zml.Tensor;
const stdx = zml.stdx;

const qwen35 = @import("qwen3_5.zig");
const Qwen35 = qwen35.Qwen35;

const CliArgs = struct {
    model: []const u8,
    prompt: []const u8 = "What is in this picture?",
    pixel_values_file: []const u8 = "/home/tristan/zml/examples/qwen3_5/safetensors/vision_input.safetensors",
    image_activations_file: []const u8 = "/home/tristan/zml/examples/qwen3_5/image_test/data/image_activations.safetensors",
    len: i64 = 64,
    gen_tokens: usize = 2048,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    const options = Qwen35.GenOptions{
        .max_seq_len = args.len,
        .sampling_strategy = .{
            .topk = 20,
            .temperature = 1.0,
        },
    };

    var vfs_file: zml.io.VFS.File = .init(allocator, init.io, .{});
    defer vfs_file.deinit();

    var vfs: zml.io.VFS = try .init(allocator, init.io);
    defer vfs.deinit();
    try vfs.register("file", vfs_file.io());

    const io = vfs.io();
    const repo = try zml.safetensors.resolveModelRepo(io, args.model);

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("\n{f}", .{platform.fmtVerbose()});

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var qwen_model: Qwen35 = try .init(allocator, store.view(), config, options);
    defer qwen_model.deinit(allocator);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

    var progress = std.Progress.start(io, .{ .root_name = args.model });

    var tokenizer_future = try io.concurrent(loadTokenizer, .{ allocator, io, repo, &progress });
    errdefer blk: {
        var v = tokenizer_future.cancel(io) catch break :blk;
        v.deinit();
    }

    var qwen_buffers_future = try io.concurrent(qwen35.Qwen35.load, .{
        &qwen_model,
        allocator,
        io,
        platform,
        &store,
        &.{replicated_sharding},
        &progress,
    });
    errdefer blk: {
        var v = qwen_buffers_future.cancel(io) catch break :blk;
        Qwen35.unloadBuffers(&v, allocator);
    }

    var tokenizer = try tokenizer_future.await(io);
    defer tokenizer.deinit();

    var vision_inputs = try loadVisionInputs(
        allocator,
        io,
        platform,
        replicated_sharding,
        args.pixel_values_file,
    );
    defer vision_inputs.pixel_values.deinit();

    const merge_size = qwen_model.config.vision_config.spatial_merge_size;
    const number_image_pad_tokens: u32 = @intCast(
        @divExact(
            vision_inputs.image_grid_thw[0] * vision_inputs.image_grid_thw[1] * vision_inputs.image_grid_thw[2],
            merge_size * merge_size,
        ),
    );
    const prompt_tokens, const prompt_shape = try applyVisionChatTemplate(
        allocator,
        tokenizer,
        args.prompt,
        number_image_pad_tokens,
    );
    defer allocator.free(prompt_tokens);

    const cache_dtype = qwen_model.text_model.embed_tokens.weight.dtype();
    const required_seq_len: i64 = @intCast(prompt_tokens.len + args.gen_tokens + 1);
    const generation_cache_seq_len = @max(qwen_model.gen_options.max_seq_len, required_seq_len);
    const kv_cache = qwen35.KvCache.init(qwen_model.config, 1, generation_cache_seq_len, cache_dtype, .f32);

    var compile_result_future = try io.concurrent(compileModel, .{
        allocator,
        io,
        platform,
        qwen_model,
        kv_cache,
        &progress,
        prompt_tokens.len,
        vision_inputs.image_grid_thw,
        replicated_sharding,
    });
    errdefer if (compile_result_future.cancel(io)) |v| {
        v.prefill_exe.deinit();
        v.decode_exe.deinit();
    } else |_| {};

    var compile_result = try compile_result_future.await(io);
    var qwen_buffers = try qwen_buffers_future.await(io);
    defer Qwen35.unloadBuffers(&qwen_buffers, allocator);

    defer {
        compile_result.prefill_exe.deinit();
        compile_result.decode_exe.deinit();
    }
    progress.end();

    log.info(
        "\nRunning vision model {s} with image inputs {f} grid_thw={any} on prompt:\n{s}\n",
        .{ args.model, vision_inputs.pixel_values.shape(), vision_inputs.image_grid_thw, args.prompt },
    );

    try runGenerationLoop(
        allocator,
        io,
        platform,
        qwen_model,
        qwen_buffers,
        replicated_sharding,
        tokenizer,
        compile_result,
        prompt_tokens,
        prompt_shape,
        vision_inputs.pixel_values,
        args.gen_tokens,
        kv_cache,
    );

    // try checkLayers(
    //     allocator,
    //     io,
    //     platform,
    //     qwen_model,
    //     qwen_buffers,
    //     args.image_activations_file,
    // );
}

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(allocator: std.mem.Allocator, io: std.Io, platform: *const zml.Platform, qwen35_model: Qwen35, kv_cache: qwen35.KvCache, progress: *std.Progress.Node, prefill_len: usize, grid_thw: [3]i64, sharding: zml.sharding.Sharding) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});
    log.info("Compiling model for platform {any} with prefill length {d}...", .{ platform.target, prefill_len });

    const t = grid_thw[0];
    const h = grid_thw[1];
    const w = grid_thw[2];
    const patch_count: i64 = t * h * w;

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen35_model_: Qwen35,
            kv_cache_: qwen35.KvCache,
            prefill_len_: usize,
            grid_thw_: [3]i64,
            patch_count_: i64,
            sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node = progress_.start("Compiling prefill...", 1);
            defer node.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill [{f}]", .{now_.untilNow(io_, .awake)});

            const patch_3d_size = qwen35_model_.auto_config.vision_patch_3d_size;
            return platform_.compile(
                allocator_,
                io_,
                qwen35_model_,
                .multimodal_prefill_forward_old,
                .{
                    Tensor.init(.{ .b = 1, .s = prefill_len_ }, .u32),
                    Tensor.init(.{}, .i64),
                    kv_cache_,
                    Tensor.init(.{ .p = patch_count_, .ps = patch_3d_size }, .f32),
                    grid_thw_,
                    Tensor.init(.{3}, .i64),
                    zml.Tensor.Rng.init(),
                },
                .{ .shardings = &.{sharding_} },
            );
        }
    }.call, .{ allocator, io, platform, qwen35_model, kv_cache, prefill_len, grid_thw, patch_count, sharding, progress });
    errdefer if (prefill_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    var decode_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen35_model_: Qwen35,
            kv_cache_: qwen35.KvCache,
            sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node = progress_.start("Compiling decode...", 1);
            defer node.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled decode [{f}]", .{now_.untilNow(io_, .awake)});

            return platform_.compile(
                allocator_,
                io_,
                qwen35_model_,
                .multimodal_decode_forward,
                .{
                    Tensor.init(.{ .b = 1, .s = 1 }, .u32),
                    Tensor.init(.{}, .i64),
                    kv_cache_,
                    Tensor.init(.{ .b = 1, .s = 1 }, .i64),
                    zml.Tensor.Rng.init(),
                },
                .{ .shardings = &.{sharding_} },
            );
        }
    }.call, .{ allocator, io, platform, qwen35_model, kv_cache, sharding, progress });
    errdefer if (decode_future.cancel(io)) |v| {
        v.deinit();
    } else |_| {};

    const vision_prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{
        .prefill_exe = vision_prefill_exe,
        .decode_exe = decode_exe,
    };
}

fn runGenerationLoop(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model: Qwen35,
    model_buffers: zml.Bufferized(Qwen35),
    sharding: zml.sharding.Sharding,
    tokenizer: zml.tokenizer.Tokenizer,
    compile_result: CompileModelResult,
    input_token_ids: []const u32,
    prompt_shape_values: [3]i64,
    pixel_values_buffer: zml.Buffer,
    gen_tokens: usize,
    kv_cache: qwen35.KvCache,
) !void {
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&kv_cache_buffers);

    const total_seq_len: i64 = prompt_shape_values[0] + prompt_shape_values[1] + prompt_shape_values[2];

    const prompt_tokens_shape = zml.Shape.init(.{ .b = 1, .s = input_token_ids.len }, .u32);
    var prompt_tokens_slice: zml.Slice = try .alloc(allocator, prompt_tokens_shape);
    defer prompt_tokens_slice.free(allocator);
    @memcpy(prompt_tokens_slice.items(u32), input_token_ids);
    var prompt_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_tokens_slice, sharding);
    defer prompt_tokens_buffer.deinit();

    var prefill_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(i64, 0), .i64, sharding);
    defer prefill_token_index_buffer.deinit();

    const prompt_shape_shape = zml.Shape.init(.{3}, .i64);
    var prompt_shape_slice: zml.Slice = try .alloc(allocator, prompt_shape_shape);
    defer prompt_shape_slice.free(allocator);
    @memcpy(prompt_shape_slice.items(i64), &prompt_shape_values);
    var prompt_shape_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_shape_slice, sharding);
    defer prompt_shape_buffer.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var vision_prefill_args = try compile_result.prefill_exe.args(allocator);
    defer vision_prefill_args.deinit(allocator);
    var vision_prefill_results = try compile_result.prefill_exe.results(allocator);
    defer vision_prefill_results.deinit(allocator);

    const prefill_now: std.Io.Timestamp = .now(io, .awake);
    vision_prefill_args.set(.{ model_buffers, prompt_tokens_buffer, prefill_token_index_buffer, kv_cache_buffers, pixel_values_buffer, prompt_shape_buffer, rng_buffers });
    compile_result.prefill_exe.call(vision_prefill_args, &vision_prefill_results);
    const prefill_duration = prefill_now.untilNow(io, .awake);
    var prefill_generated_token_buffer: zml.Buffer = undefined;
    var mrope_position_deltas_buffer: zml.Buffer = undefined;
    vision_prefill_results.fill(.{ &prefill_generated_token_buffer, &kv_cache_buffers, &mrope_position_deltas_buffer, &rng_buffers });
    defer prefill_generated_token_buffer.deinit();
    defer mrope_position_deltas_buffer.deinit();

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);
    try prefill_generated_token_buffer.toSlice(io, generated_token_slice);

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();

    var stdout = std.Io.File.stdout().writer(io, &.{});
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();
    const now: std.Io.Timestamp = .now(io, .awake);
    var num_tokens_generated: usize = 1;

    for (0..gen_tokens) |i| {
        try current_token_buffer.toSlice(io, generated_token_slice);
        const generated_token = generated_token_slice.items(u32)[0];
        num_tokens_generated += 1;
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try stdout.interface.writeAll(chunk);
            try stdout.interface.flush();
        }

        if (generated_token == model.auto_config.end_of_text_token_id) break;

        var token_index_buffer = try zml.Buffer.scalar(
            io,
            platform,
            total_seq_len - 1 + @as(i64, @intCast(i)),
            .i64,
            sharding,
        );
        defer token_index_buffer.deinit();

        var decode_args = try compile_result.decode_exe.args(allocator);
        defer decode_args.deinit(allocator);
        var decode_results = try compile_result.decode_exe.results(allocator);
        defer decode_results.deinit(allocator);

        decode_args.set(.{ model_buffers, current_token_buffer, token_index_buffer, kv_cache_buffers, mrope_position_deltas_buffer, rng_buffers });
        compile_result.decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, &kv_cache_buffers, &rng_buffers });
    }
    try stdout.interface.writeAll("\n");
    try stdout.interface.flush();
    const duration = now.untilNow(io, .awake);
    log.info("Generated {} tokens in {f}: {:.3} tok/s", .{
        num_tokens_generated,
        duration,
        stdx.Io.Duration.hzFloat(stdx.Io.Duration.div(duration, num_tokens_generated)),
    });
    log.info("Prefill duration: {:.3}s", .{
        @as(f64, @floatFromInt(prefill_duration.nanoseconds)) / std.time.ns_per_s,
    });
}

fn applyVisionChatTemplate(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    number_image_pad_tokens: u32,
) !struct { []u32, [3]i64 } {
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

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len + @as(usize, @intCast(number_image_pad_tokens)) + 16);
    try tokens.appendSlice(allocator, &.{ im_start_id, user, newline });
    try tokens.append(allocator, vision_start_id);
    for (0..number_image_pad_tokens) |_| {
        try tokens.append(allocator, image_pad_id);
    }
    try tokens.append(allocator, vision_end_id);
    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.appendSlice(allocator, &.{ im_end_id, newline });
    try tokens.appendSlice(allocator, &.{ im_start_id, assistant, newline });

    const prompt_tokens_only = try encoder.encode(prompt);
    const prompt_shape: [3]i64 = .{
        4,
        @intCast(number_image_pad_tokens),
        @as(i64, @intCast(prompt_tokens_only.len)) + 6,
    };

    return .{ try tokens.toOwnedSlice(allocator), prompt_shape };
}

fn loadTokenizer(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir, progress: *std.Progress.Node) !zml.tokenizer.Tokenizer {
    progress.increaseEstimatedTotalItems(1);
    var node = progress.start("Loading tokenizer...", 1);
    defer node.end();

    const bytes = b: {
        const file = try dir.openFile(io, "tokenizer.json", .{});
        defer file.close(io);
        var reader = file.reader(io, &.{});
        break :b try reader.interface.readAlloc(allocator, try file.length(io));
    };
    defer allocator.free(bytes);

    return try .fromBytes(allocator, io, bytes);
}

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(qwen35.Qwen35.Config) {
    const parsed_config = blk: {
        const config_json_file = try dir.openFile(io, "config.json", .{});
        defer config_json_file.close(io);
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(io, &config_json_buffer);
        var reader: std.json.Reader = .init(allocator, &config_reader.interface);
        defer reader.deinit();
        break :blk try std.json.parseFromTokenSource(qwen35.Qwen35.Config, allocator, &reader, .{ .ignore_unknown_fields = true });
    };
    errdefer parsed_config.deinit();

    return parsed_config;
}

fn checkLayers(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    qwen_model: Qwen35,
    qwen35_buffers: zml.Bufferized(Qwen35),
    activations_file: []const u8,
) !void {
    var activations_registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, activations_file);
    defer activations_registry.deinit();

    var activations_store: zml.io.TensorStore = .fromRegistry(allocator, &activations_registry);
    defer activations_store.deinit();

    const comp_opts: zml.testing.CompareOpts = .{
        .absolute_tolerance = 1e-2,
        .relative_tolerance = 1e-2,
        .minimum_close_fraction = 0.99,
    };
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    const shardings = &.{replicated_sharding};
    const vision_grid_thw = try loadGridThwFromActivations(allocator, io, activations_store.view());

    const VisionPatchEmbedHarness = struct {
        patch_embed: qwen35.VisionPatchEmbed,
        pub fn forward(self: @This(), pixel_values: Tensor) Tensor {
            return self.patch_embed.forward(pixel_values.withPartialTags(.{ .p, .ps }));
        }
    };
    const patchEmbedHarness: VisionPatchEmbedHarness = .{
        .patch_embed = qwen_model.vision_model.vision_patch_embed,
    };
    const patchEmbedHarnessBuffers: zml.Bufferized(VisionPatchEmbedHarness) = .{
        .patch_embed = qwen35_buffers.vision_model.vision_patch_embed,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        patchEmbedHarness,
        .forward,
        activations_store.view(),
        "model.visual.patch_embed",
        patchEmbedHarnessBuffers,
        shardings,
        comp_opts,
    );

    const VisionPreBlockInputHarness = struct {
        model: qwen35.VisionModel,
        grid_thw: [3]i64,
        pub fn forward(self: @This(), pixel_values: Tensor, grid_thw: Tensor) Tensor {
            _ = grid_thw;
            const patch_embeds = self.model.vision_patch_embed.forward(pixel_values.withPartialTags(.{ .p, .ps }));
            const pos_embeds = self.model.getPosEmbeds(&self.grid_thw).convert(patch_embeds.dtype());
            return patch_embeds.add(pos_embeds);
        }
    };
    const preBlockHarness: VisionPreBlockInputHarness = .{
        .model = qwen_model.vision_model,
        .grid_thw = vision_grid_thw,
    };
    const preBlockHarnessBuffers: zml.Bufferized(VisionPreBlockInputHarness) = .{
        .model = qwen35_buffers.vision_model,
    };
    try testHarnessAgainstActivationKeys(
        allocator,
        io,
        platform,
        preBlockHarness,
        activations_store.view(),
        "model.visual.in",
        &.{"model.visual.blocks.0.in.0"},
        preBlockHarnessBuffers,
        shardings,
        comp_opts,
        "model.visual pre-block input (patch_embed + pos_embeds)",
    );

    const VisionRopeHarness = struct {
        model: qwen35.VisionModel,
        grid_thw: [3]i64,
        pub fn forward(self: @This(), pixel_values: Tensor, grid_thw: Tensor) struct { Tensor, Tensor } {
            _ = pixel_values;
            _ = grid_thw;
            const cos, const sin = self.model.getVisionRopeCosAndSin(&self.grid_thw);
            return .{ cos, sin };
        }
    };
    const ropeHarness: VisionRopeHarness = .{
        .model = qwen_model.vision_model,
        .grid_thw = vision_grid_thw,
    };
    const ropeHarnessBuffers: zml.Bufferized(VisionRopeHarness) = .{
        .model = qwen35_buffers.vision_model,
    };
    try testHarnessAgainstActivationKeys(
        allocator,
        io,
        platform,
        ropeHarness,
        activations_store.view(),
        "model.visual.in",
        &.{ "model.visual.blocks.0.in.2", "model.visual.blocks.0.in.3" },
        ropeHarnessBuffers,
        shardings,
        comp_opts,
        "model.visual rope (cos, sin)",
    );

    const VisionBlockHarness = struct {
        block: qwen35.VisionBlock,
        pub fn forward(self: @This(), hidden_states: Tensor, cu_seqlens: Tensor, cos: Tensor, sin: Tensor) Tensor {
            _ = cu_seqlens;
            return self.block.forward(
                hidden_states.withTags(.{ .p, .d }),
                cos.withTags(.{ .p, .hd }),
                sin.withTags(.{ .p, .hd }),
            );
        }
    };
    _ = VisionBlockHarness; // autofix

    const VisionAttentionHarness = struct {
        attn: qwen35.VisionAttention,
        pub fn forward(self: @This(), hidden_states: Tensor, cu_seqlens: Tensor, cos: Tensor, sin: Tensor) Tensor {
            _ = cu_seqlens;
            return self.attn.forward(
                hidden_states.withTags(.{ .p, .d }),
                cos.withTags(.{ .p, .hd }),
                sin.withTags(.{ .p, .hd }),
            );
        }
    };
    _ = VisionAttentionHarness; // autofix

    const VisionMlpHarness = struct {
        mlp: qwen35.VisionMlp,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.mlp.forward(x.withTags(.{ .p, .d }));
        }
    };
    _ = VisionMlpHarness; // autofix

    const VisionNormHarness = struct {
        norm: zml.nn.LayerNorm,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.norm.forward(x.withTags(.{ .p, .d }));
        }
    };
    _ = VisionNormHarness; // autofix

    // for (qwen_model.vision_model.blocks, qwen35_buffers.vision_model.blocks, 0..) |block, block_buffers, block_index| {
    //     var name_buf: [128]u8 = undefined;

    //     const block_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}", .{block_index});
    //     const blockHarness: VisionBlockHarness = .{ .block = block };
    //     const blockHarnessBuffers: zml.Bufferized(VisionBlockHarness) = .{ .block = block_buffers };
    //     try zml.testing.testLayer(
    //         allocator,
    //         io,
    //         platform,
    //         blockHarness,
    //         .forward,
    //         activations_store.view(),
    //         block_name,
    //         blockHarnessBuffers,
    //         shardings,
    //         comp_opts,
    //     );

    //     const norm1_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.norm1", .{block_index});
    //     const norm1Harness: VisionNormHarness = .{ .norm = block.norm1 };
    //     const norm1HarnessBuffers: zml.Bufferized(VisionNormHarness) = .{ .norm = block_buffers.norm1 };
    //     try zml.testing.testLayer(
    //         allocator,
    //         io,
    //         platform,
    //         norm1Harness,
    //         .forward,
    //         activations_store.view(),
    //         norm1_name,
    //         norm1HarnessBuffers,
    //         shardings,
    //         comp_opts,
    //     );

    //     const attn_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.attn", .{block_index});
    //     const attnHarness: VisionAttentionHarness = .{ .attn = block.attn };
    //     const attnHarnessBuffers: zml.Bufferized(VisionAttentionHarness) = .{ .attn = block_buffers.attn };
    //     try zml.testing.testLayer(
    //         allocator,
    //         io,
    //         platform,
    //         attnHarness,
    //         .forward,
    //         activations_store.view(),
    //         attn_name,
    //         attnHarnessBuffers,
    //         shardings,
    //         comp_opts,
    //     );

    //     const norm2_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.norm2", .{block_index});
    //     const norm2Harness: VisionNormHarness = .{ .norm = block.norm2 };
    //     const norm2HarnessBuffers: zml.Bufferized(VisionNormHarness) = .{ .norm = block_buffers.norm2 };
    //     try zml.testing.testLayer(
    //         allocator,
    //         io,
    //         platform,
    //         norm2Harness,
    //         .forward,
    //         activations_store.view(),
    //         norm2_name,
    //         norm2HarnessBuffers,
    //         shardings,
    //         comp_opts,
    //     );

    //     const mlp_name = try std.fmt.bufPrint(&name_buf, "model.visual.blocks.{d}.mlp", .{block_index});
    //     const mlpHarness: VisionMlpHarness = .{ .mlp = block.mlp };
    //     const mlpHarnessBuffers: zml.Bufferized(VisionMlpHarness) = .{ .mlp = block_buffers.mlp };
    //     try zml.testing.testLayer(
    //         allocator,
    //         io,
    //         platform,
    //         mlpHarness,
    //         .forward,
    //         activations_store.view(),
    //         mlp_name,
    //         mlpHarnessBuffers,
    //         shardings,
    //         comp_opts,
    //     );
    // }

    const VisionPatchMergerHarness = struct {
        merger: qwen35.VisionPatchMerger,
        pub fn forward(self: @This(), x: Tensor) Tensor {
            return self.merger.forward(x.withTags(.{ .p, .d }));
        }
    };
    const patchMergerHarness: VisionPatchMergerHarness = .{
        .merger = qwen_model.vision_model.patch_merger,
    };
    const patchMergerHarnessBuffers: zml.Bufferized(VisionPatchMergerHarness) = .{
        .merger = qwen35_buffers.vision_model.patch_merger,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        patchMergerHarness,
        .forward,
        activations_store.view(),
        "model.visual.merger",
        patchMergerHarnessBuffers,
        shardings,
        comp_opts,
    );

    const VisionModelHarness = struct {
        model: qwen35.VisionModel,
        grid_thw: [3]i64,
        pub fn forward(self: @This(), pixel_values: Tensor, grid_thw: Tensor) struct { Tensor, Tensor } {
            _ = grid_thw;
            const merged_hidden_states = self.model.forward(pixel_values.withPartialTags(.{ .p, .ps }), self.grid_thw);
            const patch_count = self.grid_thw[0] * self.grid_thw[1] * self.grid_thw[2];
            const fake_hidden_states = zml.Tensor.constant(.{ .f32 = 0 })
                .broad(zml.Shape.init(.{ .p = patch_count, .d = self.model.hidden_size }, .f32))
                .convert(merged_hidden_states.dtype());
            return .{ fake_hidden_states, merged_hidden_states };
        }
    };
    const visionModelHarness: VisionModelHarness = .{
        .model = qwen_model.vision_model,
        .grid_thw = vision_grid_thw,
    };
    const visionModelHarnessBuffers: zml.Bufferized(VisionModelHarness) = .{
        .model = qwen35_buffers.vision_model,
    };
    try zml.testing.testLayer(
        allocator,
        io,
        platform,
        visionModelHarness,
        .forward,
        activations_store.view(),
        "model.visual",
        visionModelHarnessBuffers,
        shardings,
        comp_opts,
    );

    try checkTextModelInputAgainstReference(
        allocator,
        io,
        platform,
        qwen_model,
        qwen35_buffers,
        activations_store.view(),
        vision_grid_thw,
        shardings,
        comp_opts,
    );
}

fn loadGridThwFromActivations(
    allocator: std.mem.Allocator,
    io: std.Io,
    activation_store: zml.io.TensorStore.View,
) ![3]i64 {
    var reader_buffer: [4096]u8 = undefined;
    var reader = try activation_store.getReader("model.visual.in.1", io, &reader_buffer);
    defer reader.deinit();

    const shape = activation_store.getShape("model.visual.in.1") orelse return error.MissingVisionGrid;
    var grid_slice: zml.Slice = try .alloc(allocator, shape);
    defer grid_slice.free(allocator);
    try reader.interface.readSliceAll(grid_slice.data());

    const grid = grid_slice.items(i64);
    if (grid.len < 3) return error.InvalidImageGrid;
    return .{ grid[0], grid[1], grid[2] };
}

fn testHarnessAgainstActivationKeys(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    harness: anytype,
    activation_store: zml.io.TensorStore.View,
    input_prefix: []const u8,
    expected_keys: []const []const u8,
    harness_buffers: zml.Bufferized(@TypeOf(harness)),
    shardings: []const zml.sharding.Sharding,
    opts: zml.testing.CompareOpts,
    label: []const u8,
) !void {
    const forward = @TypeOf(harness).forward;
    const ArgsT = stdx.meta.Tail(std.meta.ArgsTuple(@TypeOf(forward)));
    var args: ArgsT = undefined;

    const LocalContext = struct {
        activation_store: zml.io.TensorStore.View,
        index: usize = 0,
    };

    var ctx: LocalContext = .{
        .activation_store = activation_store.withPrefix(input_prefix),
    };
    try zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *zml.Tensor) !void {
            var buffer: [256]u8 = undefined;
            const subkey = std.fmt.bufPrint(&buffer, "{d}", .{ctx_.index}) catch unreachable;
            tensor.* = ctx_.activation_store.createTensor(subkey, null, null);
            ctx_.index += 1;
        }
    }.cb, &ctx, &args);

    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    const exe = try platform.compile(
        allocator,
        io,
        harness,
        .forward,
        args,
        .{ .shardings = if (shardings.len != 0) shardings else &.{replicated_sharding} },
    );
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    var args_buffers = try zml.io.load(ArgsT, &args, allocator, io, platform, activation_store.store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &args_buffers);

    exe_args.set(.{ harness_buffers, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, exe.output_shapes.len);
    defer allocator.free(results);
    exe_results.fill(.{results});
    defer for (results) |*b| b.deinit();

    var failed: bool = false;
    var reader_buffer: [4096]u8 = undefined;
    for (expected_keys, 0..) |expected_key, i| {
        if (i >= results.len) break;

        const expected_slice = b: {
            var reader = try activation_store.getReader(expected_key, io, &reader_buffer);
            defer reader.deinit();
            const shape = activation_store.getShape(expected_key) orelse return error.MissingActivationTensor;
            const s: zml.Slice = try .alloc(allocator, shape);
            errdefer s.free(allocator);
            try reader.interface.readSliceAll(s.data());
            break :b s;
        };
        defer expected_slice.free(allocator);

        const output_slice = try results[i].toSliceAlloc(allocator, io);
        defer output_slice.free(allocator);

        zml.testing.expectClose(io, expected_slice, output_slice, opts) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                const stderr = std.debug.lockStderr(&.{});
                defer std.debug.unlockStderr();
                const w = &stderr.file_writer.interface;
                try w.print("{s} output {d} doesn't match {s} !\n", .{ label, i, expected_key });
                failed = true;
                continue;
            },
            else => return err,
        };
    }

    if (failed) {
        log.info("❌ check failed for {s} ! (absolute tolerance: {e} - relative tolerance: {e} - minimum_close_fraction: {d:0>3})", .{ label, opts.absolute_tolerance, opts.relative_tolerance, opts.minimum_close_fraction });
    } else {
        log.info("✅ all good for {s} ! (absolute tolerance: {e} - relative tolerance: {e} - minimum_close_fraction: {d:0>3})", .{ label, opts.absolute_tolerance, opts.relative_tolerance, opts.minimum_close_fraction });
    }
}

fn checkTextModelInputAgainstReference(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen_model: Qwen35,
    qwen35_buffers: zml.Bufferized(Qwen35),
    activation_store: zml.io.TensorStore.View,
    vision_grid_thw: [3]i64,
    shardings: []const zml.sharding.Sharding,
    opts: zml.testing.CompareOpts,
) !void {
    const keys = try findTextInputCompareKeys(allocator, activation_store, qwen_model.config.text_config.hidden_size, qwen_model.auto_config.vision_patch_3d_size);
    defer keys.deinit(allocator);

    const pre_image_tokens = try firstImageTokenIndexFromMmTokenType(allocator, io, activation_store, keys.mm_token_type_ids_key);

    const TextModelInputHarness = struct {
        model: qwen35.Qwen35,
        grid_thw: [3]i64,
        pre_tokens: i64,

        pub fn forward(self: @This(), input_ids: Tensor, pixel_values: Tensor) Tensor {
            const token_embed = self.model.text_model.embed_tokens.forward(input_ids.withPartialTags(.{.s}));
            const vision_embeds = self.model.vision_model.forward(pixel_values.withPartialTags(.{ .p, .ps }), self.grid_thw);
            const vision_embeds_batched = vision_embeds.insertAxes(.s, .{.b});
            return token_embed.dynamicUpdateSlice(
                .{ .s = zml.Tensor.scalar(self.pre_tokens, .i64) },
                vision_embeds_batched.convert(token_embed.dtype()),
            );
        }
    };

    const harness: TextModelInputHarness = .{
        .model = qwen_model,
        .grid_thw = vision_grid_thw,
        .pre_tokens = pre_image_tokens,
    };
    const harness_buffers: zml.Bufferized(TextModelInputHarness) = .{
        .model = qwen35_buffers,
    };

    const args = .{
        activation_store.createTensor(keys.input_ids_key, null, null),
        activation_store.createTensor(keys.pixel_values_key, null, null),
    };
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    const exe = try platform.compile(
        allocator,
        io,
        harness,
        .forward,
        args,
        .{ .shardings = if (shardings.len != 0) shardings else &.{replicated_sharding} },
    );
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    var args_buffers = try zml.io.load(@TypeOf(args), &args, allocator, io, platform, activation_store.store, .auto);
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &args_buffers);

    exe_args.set(.{ harness_buffers, args_buffers });
    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var out: zml.Buffer = undefined;
    exe_results.fill(.{&out});
    defer out.deinit();

    const expected_slice = b: {
        var reader_buffer: [4096]u8 = undefined;
        var reader = try activation_store.getReader(keys.text_inputs_embeds_key, io, &reader_buffer);
        defer reader.deinit();
        const shape = activation_store.getShape(keys.text_inputs_embeds_key) orelse return error.MissingActivationTensor;
        const s: zml.Slice = try .alloc(allocator, shape);
        errdefer s.free(allocator);
        try reader.interface.readSliceAll(s.data());
        break :b s;
    };
    defer expected_slice.free(allocator);

    const actual_slice = try out.toSliceAlloc(allocator, io);
    defer actual_slice.free(allocator);

    zml.testing.expectClose(io, expected_slice, actual_slice, opts) catch |err| switch (err) {
        error.TestUnexpectedResult => {
            log.info("❌ text-model input mismatch against python reference key '{s}'", .{keys.text_inputs_embeds_key});
            return;
        },
        else => return err,
    };
    log.info("✅ text-model input matches python reference key '{s}'", .{keys.text_inputs_embeds_key});
}

const TextInputCompareKeys = struct {
    input_ids_key: []const u8,
    pixel_values_key: []const u8,
    mm_token_type_ids_key: []const u8,
    text_inputs_embeds_key: []const u8,

    fn deinit(self: @This(), allocator: std.mem.Allocator) void {
        allocator.free(self.input_ids_key);
        allocator.free(self.pixel_values_key);
        allocator.free(self.mm_token_type_ids_key);
        allocator.free(self.text_inputs_embeds_key);
    }
};

fn findTextInputCompareKeys(
    allocator: std.mem.Allocator,
    activation_store: zml.io.TensorStore.View,
    text_hidden_size: i64,
    vision_patch_3d_size: i64,
) !TextInputCompareKeys {
    var input_ids_key: ?[]const u8 = null;
    var pixel_values_key: ?[]const u8 = null;
    var mm_token_type_ids_key: ?[]const u8 = null;
    var text_inputs_embeds_key: ?[]const u8 = null;

    var it = activation_store.store.registry.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const shape = entry.value_ptr.*.shape;
        const dims = shape.dims();
        const dtype = shape.dtype();

        if (std.mem.startsWith(u8, key, "model.in.")) {
            if (dtype == .i64 and dims.len == 2 and input_ids_key == null) {
                input_ids_key = try allocator.dupe(u8, key);
            } else if (dtype == .f32 and dims.len == 2 and dims[1] == vision_patch_3d_size and pixel_values_key == null) {
                pixel_values_key = try allocator.dupe(u8, key);
            } else if (dtype == .i32 and dims.len == 2 and mm_token_type_ids_key == null) {
                mm_token_type_ids_key = try allocator.dupe(u8, key);
            }
        } else if (std.mem.startsWith(u8, key, "model.language_model.in.")) {
            if (dims.len == 3 and dims[dims.len - 1] == text_hidden_size and text_inputs_embeds_key == null) {
                text_inputs_embeds_key = try allocator.dupe(u8, key);
            }
        }
    }

    if (input_ids_key == null or pixel_values_key == null or mm_token_type_ids_key == null or text_inputs_embeds_key == null) {
        if (input_ids_key) |k| allocator.free(k);
        if (pixel_values_key) |k| allocator.free(k);
        if (mm_token_type_ids_key) |k| allocator.free(k);
        if (text_inputs_embeds_key) |k| allocator.free(k);
        return error.CouldNotInferTextInputComparisonKeys;
    }

    return .{
        .input_ids_key = input_ids_key.?,
        .pixel_values_key = pixel_values_key.?,
        .mm_token_type_ids_key = mm_token_type_ids_key.?,
        .text_inputs_embeds_key = text_inputs_embeds_key.?,
    };
}

fn firstImageTokenIndexFromMmTokenType(
    allocator: std.mem.Allocator,
    io: std.Io,
    activation_store: zml.io.TensorStore.View,
    mm_token_type_ids_key: []const u8,
) !i64 {
    var reader_buffer: [4096]u8 = undefined;
    var reader = try activation_store.getReader(mm_token_type_ids_key, io, &reader_buffer);
    defer reader.deinit();

    const shape = activation_store.getShape(mm_token_type_ids_key) orelse return error.MissingActivationTensor;
    var mm_slice: zml.Slice = try .alloc(allocator, shape);
    defer mm_slice.free(allocator);
    try reader.interface.readSliceAll(mm_slice.data());

    const vals = mm_slice.items(i32);
    for (vals, 0..) |v, i| {
        if (v == 1) return @intCast(i);
    }
    return error.NoImageTokensInMmTokenTypeIds;
}

const VisionInputs = struct {
    pixel_values: zml.Buffer,
    image_grid_thw: [3]i64,
};

fn loadVisionInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    path: []const u8,
) !VisionInputs {
    const PixelValuesOnly = struct {
        pixel_values: Tensor,
        image_grid_thw: Tensor,
    };

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: PixelValuesOnly = .{
        .pixel_values = store.view().createTensor("pixel_values", null, null),
        .image_grid_thw = store.view().createTensor("image_grid_thw", null, null),
    };
    var loaded = try zml.io.load(PixelValuesOnly, &model, allocator, io, platform, &store, .{
        .parallelism = 1,
        .shardings = &.{sharding},
        .progress = null,
        .dma_chunks = 1,
        .dma_chunk_size = 4 * zml.MiB,
    });
    var grid_slice: zml.Slice = try .alloc(allocator, loaded.image_grid_thw.shape());
    defer grid_slice.free(allocator);
    try loaded.image_grid_thw.toSlice(io, grid_slice);
    loaded.image_grid_thw.deinit();

    const grid = grid_slice.items(i64);
    if (grid.len < 3) return error.InvalidImageGrid;

    return .{
        .pixel_values = loaded.pixel_values,
        .image_grid_thw = .{ grid[0], grid[1], grid[2] },
    };
}
