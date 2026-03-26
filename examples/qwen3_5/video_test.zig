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
    prompt: []const u8 = "What is happening in this video?",
    pixel_values_file: []const u8 = "/home/tristan/zml/examples/qwen3_5/safetensors/video_input.safetensors",
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

    var vision_inputs = try loadVideoInputs(
        allocator,
        io,
        platform,
        replicated_sharding,
        args.pixel_values_file,
    );
    defer vision_inputs.pixel_values.deinit();
    defer allocator.free(vision_inputs.timestamps);

    const video_prompt = try applyVideoChatTemplate(
        allocator,
        tokenizer,
        args.prompt,
        vision_inputs.video_grid_thw,
        vision_inputs.timestamps,
        qwen_model.config.vision_config.spatial_merge_size,
    );
    defer allocator.free(video_prompt.tokens);
    defer allocator.free(video_prompt.video_chunk_starts);
    defer allocator.free(video_prompt.position_ids);

    const cache_dtype = qwen_model.text_model.embed_tokens.weight.dtype();
    const required_seq_len: i64 = @intCast(video_prompt.tokens.len + args.gen_tokens + 1);
    const generation_cache_seq_len = @max(qwen_model.gen_options.max_seq_len, required_seq_len);
    const kv_cache = qwen35.KvCache.init(qwen_model.config, 1, generation_cache_seq_len, cache_dtype, .f32);

    var compile_result_future = try io.concurrent(compileModel, .{
        allocator,
        io,
        platform,
        qwen_model,
        kv_cache,
        &progress,
        video_prompt.tokens.len,
        vision_inputs.video_grid_thw,
        video_prompt.frame_token_count,
        video_prompt.video_chunk_starts.len,
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
        "\nRunning vision model {s} with video inputs {f} grid_thw={any} on prompt:\n{s}\n",
        .{ args.model, vision_inputs.pixel_values.shape(), vision_inputs.video_grid_thw, args.prompt },
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
        video_prompt.tokens,
        video_prompt.video_chunk_starts,
        video_prompt.position_ids,
        vision_inputs.pixel_values,
        args.gen_tokens,
        kv_cache,
    );
}

const CompileModelResult = struct {
    prefill_exe: zml.Exe,
    decode_exe: zml.Exe,
};

fn compileModel(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    qwen35_model: Qwen35,
    kv_cache: qwen35.KvCache,
    progress: *std.Progress.Node,
    prefill_len: usize,
    grid_thw: [3]i64,
    frame_token_count: i64,
    video_chunk_count: usize,
    sharding: zml.sharding.Sharding,
) !CompileModelResult {
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
            frame_token_count_: i64,
            video_chunk_count_: usize,
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
                .multimodal_video_prefill_forward,
                .{
                    Tensor.init(.{ .b = 1, .s = prefill_len_ }, .u32),
                    Tensor.init(.{}, .i64),
                    kv_cache_,
                    Tensor.init(.{ .p = patch_count_, .ps = patch_3d_size }, .f32),
                    grid_thw_,
                    Tensor.init(.{ .n = video_chunk_count_ }, .i64),
                    frame_token_count_,
                    Tensor.init(.{ .g = 3, .b = 1, .s = prefill_len_ }, .i64),
                    zml.Tensor.Rng.init(),
                },
                .{ .shardings = &.{sharding_} },
            );
        }
    }.call, .{ allocator, io, platform, qwen35_model, kv_cache, prefill_len, grid_thw, patch_count, frame_token_count, video_chunk_count, sharding, progress });
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
    video_chunk_starts: []const i64,
    position_ids_values: []const i64,
    pixel_values_buffer: zml.Buffer,
    gen_tokens: usize,
    kv_cache: qwen35.KvCache,
) !void {
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&kv_cache_buffers);

    const total_seq_len: i64 = @intCast(input_token_ids.len);

    const prompt_tokens_shape = zml.Shape.init(.{ .b = 1, .s = input_token_ids.len }, .u32);
    var prompt_tokens_slice: zml.Slice = try .alloc(allocator, prompt_tokens_shape);
    defer prompt_tokens_slice.free(allocator);
    @memcpy(prompt_tokens_slice.items(u32), input_token_ids);
    var prompt_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_tokens_slice, sharding);
    defer prompt_tokens_buffer.deinit();

    var prefill_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(i64, 0), .i64, sharding);
    defer prefill_token_index_buffer.deinit();

    const video_chunk_starts_shape = zml.Shape.init(.{ .n = video_chunk_starts.len }, .i64);
    var video_chunk_starts_slice: zml.Slice = try .alloc(allocator, video_chunk_starts_shape);
    defer video_chunk_starts_slice.free(allocator);
    @memcpy(video_chunk_starts_slice.items(i64), video_chunk_starts);
    var video_chunk_starts_buffer: zml.Buffer = try .fromSlice(io, platform, video_chunk_starts_slice, sharding);
    defer video_chunk_starts_buffer.deinit();

    const position_ids_shape = zml.Shape.init(.{ .g = 3, .b = 1, .s = input_token_ids.len }, .i64);
    var position_ids_slice: zml.Slice = try .alloc(allocator, position_ids_shape);
    defer position_ids_slice.free(allocator);
    @memcpy(position_ids_slice.items(i64), position_ids_values);
    var position_ids_buffer: zml.Buffer = try .fromSlice(io, platform, position_ids_slice, sharding);
    defer position_ids_buffer.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var vision_prefill_args = try compile_result.prefill_exe.args(allocator);
    defer vision_prefill_args.deinit(allocator);
    var vision_prefill_results = try compile_result.prefill_exe.results(allocator);
    defer vision_prefill_results.deinit(allocator);

    const prefill_now: std.Io.Timestamp = .now(io, .awake);
    vision_prefill_args.set(.{
        model_buffers,
        prompt_tokens_buffer,
        prefill_token_index_buffer,
        kv_cache_buffers,
        pixel_values_buffer,
        video_chunk_starts_buffer,
        position_ids_buffer,
        rng_buffers,
    });
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

const VideoPrompt = struct {
    tokens: []u32,
    video_chunk_starts: []i64,
    position_ids: []i64,
    frame_token_count: i64,
};

fn applyVideoChatTemplate(
    allocator: std.mem.Allocator,
    tokenizer: zml.tokenizer.Tokenizer,
    prompt: []const u8,
    video_grid_thw: [3]i64,
    timestamps: []const f32,
    spatial_merge_size: i64,
) !VideoPrompt {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();
    const im_start_id = tokenizer.tokenToId("<|im_start|>") orelse return error.NoSuchToken;
    const im_end_id = tokenizer.tokenToId("<|im_end|>") orelse return error.NoSuchToken;
    const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
    const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
    const vision_start_id = tokenizer.tokenToId("<|vision_start|>") orelse return error.NoSuchToken;
    const vision_end_id = tokenizer.tokenToId("<|vision_end|>") orelse return error.NoSuchToken;
    const video_pad_id = tokenizer.tokenToId("<|video_pad|>") orelse return error.NoSuchToken;
    const newline = (try encoder.encode("\n"))[0];
    if (timestamps.len != @as(usize, @intCast(video_grid_thw[0]))) return error.InvalidVideoTimestamps;

    const frame_token_count = @divExact(video_grid_thw[1] * video_grid_thw[2], spatial_merge_size * spatial_merge_size);
    const estimated_capacity = prompt.len + timestamps.len * @as(usize, @intCast(frame_token_count + 8)) + 16;
    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, estimated_capacity);
    var video_chunk_starts: std.ArrayList(i64) = try .initCapacity(allocator, timestamps.len);
    var is_video_token: std.ArrayList(bool) = try .initCapacity(allocator, estimated_capacity);
    defer is_video_token.deinit(allocator);

    try appendTokenSlice(allocator, &tokens, &is_video_token, &.{ im_start_id, user, newline }, false);
    for (timestamps) |timestamp| {
        var timestamp_buffer: [64]u8 = undefined;
        const timestamp_text = try std.fmt.bufPrint(&timestamp_buffer, "<{d:.1} seconds>", .{timestamp});
        try appendTokenSlice(allocator, &tokens, &is_video_token, try encoder.encode(timestamp_text), false);
        try appendToken(allocator, &tokens, &is_video_token, vision_start_id, false);
        try video_chunk_starts.append(allocator, @intCast(tokens.items.len));
        for (0..@intCast(frame_token_count)) |_| {
            try appendToken(allocator, &tokens, &is_video_token, video_pad_id, true);
        }
        try appendToken(allocator, &tokens, &is_video_token, vision_end_id, false);
    }
    try appendTokenSlice(allocator, &tokens, &is_video_token, try encoder.encode(prompt), false);
    try appendTokenSlice(allocator, &tokens, &is_video_token, &.{ im_end_id, newline, im_start_id, assistant, newline }, false);

    const position_ids = try buildVideoPositionIds(
        allocator,
        is_video_token.items,
        video_grid_thw,
        spatial_merge_size,
    );

    return .{
        .tokens = try tokens.toOwnedSlice(allocator),
        .video_chunk_starts = try video_chunk_starts.toOwnedSlice(allocator),
        .position_ids = position_ids,
        .frame_token_count = frame_token_count,
    };
}

fn appendToken(
    allocator: std.mem.Allocator,
    tokens: *std.ArrayList(u32),
    is_video_token: *std.ArrayList(bool),
    token: u32,
    is_video: bool,
) !void {
    try tokens.append(allocator, token);
    try is_video_token.append(allocator, is_video);
}

fn appendTokenSlice(
    allocator: std.mem.Allocator,
    tokens: *std.ArrayList(u32),
    is_video_token: *std.ArrayList(bool),
    token_slice: []const u32,
    is_video: bool,
) !void {
    for (token_slice) |token| {
        try appendToken(allocator, tokens, is_video_token, token, is_video);
    }
}

fn buildVideoPositionIds(
    allocator: std.mem.Allocator,
    is_video_token: []const bool,
    video_grid_thw: [3]i64,
    spatial_merge_size: i64,
) ![]i64 {
    const seq_len = is_video_token.len;
    const llm_grid_h: i64 = @divExact(video_grid_thw[1], spatial_merge_size);
    const llm_grid_w: i64 = @divExact(video_grid_thw[2], spatial_merge_size);
    const frame_token_count: i64 = llm_grid_h * llm_grid_w;
    const video_pos_step: i64 = @max(llm_grid_h, llm_grid_w);

    var temporal = try allocator.alloc(i64, seq_len);
    errdefer allocator.free(temporal);
    var height = try allocator.alloc(i64, seq_len);
    errdefer allocator.free(height);
    var width = try allocator.alloc(i64, seq_len);
    errdefer allocator.free(width);

    var cursor: usize = 0;
    var current_pos: i64 = 0;
    while (cursor < seq_len) {
        const run_is_video = is_video_token[cursor];
        var run_end = cursor;
        while (run_end < seq_len and is_video_token[run_end] == run_is_video) : (run_end += 1) {}
        const run_len: i64 = @intCast(run_end - cursor);

        if (!run_is_video) {
            for (cursor..run_end, 0..) |idx, offset| {
                const pos = current_pos + @as(i64, @intCast(offset));
                temporal[idx] = pos;
                height[idx] = pos;
                width[idx] = pos;
            }
            current_pos += run_len;
        } else {
            if (run_len != frame_token_count) return error.InvalidVideoTokenLayout;
            for (cursor..run_end, 0..) |idx, offset| {
                const row: i64 = @divFloor(@as(i64, @intCast(offset)), llm_grid_w);
                const col: i64 = @mod(@as(i64, @intCast(offset)), llm_grid_w);
                temporal[idx] = current_pos;
                height[idx] = current_pos + row;
                width[idx] = current_pos + col;
            }
            current_pos += video_pos_step;
        }

        cursor = run_end;
    }

    var position_ids = try allocator.alloc(i64, 3 * seq_len);
    @memcpy(position_ids[0..seq_len], temporal);
    @memcpy(position_ids[seq_len .. 2 * seq_len], height);
    @memcpy(position_ids[2 * seq_len .. 3 * seq_len], width);
    allocator.free(temporal);
    allocator.free(height);
    allocator.free(width);
    return position_ids;
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
    video_grid_thw: [3]i64,
    timestamps: []f32,
};

fn loadVideoInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    path: []const u8,
) !VisionInputs {
    const PixelValuesOnly = struct {
        pixel_values: Tensor,
        video_grid_thw: Tensor,
        timestamps: Tensor,
    };

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const model: PixelValuesOnly = .{
        .pixel_values = store.view().createTensor("pixel_values", null, null),
        .video_grid_thw = store.view().createTensor("video_grid_thw", null, null),
        .timestamps = store.view().createTensor("timestamps", null, null),
    };
    var loaded = try zml.io.load(PixelValuesOnly, &model, allocator, io, platform, &store, .{
        .parallelism = 1,
        .shardings = &.{sharding},
        .progress = null,
        .dma_chunks = 1,
        .dma_chunk_size = 4 * zml.MiB,
    });
    var grid_slice: zml.Slice = try .alloc(allocator, loaded.video_grid_thw.shape());
    defer grid_slice.free(allocator);
    try loaded.video_grid_thw.toSlice(io, grid_slice);
    loaded.video_grid_thw.deinit();
    var timestamps_slice: zml.Slice = try .alloc(allocator, loaded.timestamps.shape());
    defer timestamps_slice.free(allocator);
    try loaded.timestamps.toSlice(io, timestamps_slice);
    loaded.timestamps.deinit();

    const grid = grid_slice.items(i64);
    if (grid.len < 3) return error.InvalidVideoGrid;

    return .{
        .pixel_values = loaded.pixel_values,
        .video_grid_thw = .{ grid[0], grid[1], grid[2] },
        .timestamps = try allocator.dupe(f32, timestamps_slice.items(f32)),
    };
}
