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
                .multimodal_prefill_forward,
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
