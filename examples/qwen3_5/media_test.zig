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
    prompt: []const u8,
    // Path to the media input file in safetensors format. The file should contain the pixel values and media info for each media input, organized as described in the loadMediaInputs function.
    media_input_file: []const u8,
    max_seq_len: i64 = 4096,
    gen_len: usize = 2048,
    enable_thinking: bool = false,
};

// This is a demo of Qwen 3.5 with any number of images and videos as input.
// The media pixel values are loaded from a safetensors file, obtained from Python preprocessing for simplicity: launch run.sh to trigger the whole pipeline.

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    const options = Qwen35.GenOptions{
        .max_seq_len = args.max_seq_len,
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
    defer blk: {
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
    defer blk: {
        var v = qwen_buffers_future.cancel(io) catch break :blk;
        Qwen35.unloadBuffers(&v, allocator);
    }

    const tokenizer = try tokenizer_future.await(io);

    var media_inputs = try loadMediaInputs(
        allocator,
        io,
        platform,
        replicated_sharding,
        args.media_input_file,
    );
    defer media_inputs.deinit(allocator);

    var multimodal_prompt = try qwen35.MultimodalPrompt.init(
        allocator,
        qwen_model.config,
        tokenizer,
        args.prompt,
        media_inputs.media_input_list,
        .{
            .max_seq_len = options.max_seq_len,
            .enable_thinking = args.enable_thinking,
        },
    );
    defer multimodal_prompt.deinit(allocator);

    log.info("Prompt layout: {f}", .{multimodal_prompt});

    const cache_dtype = qwen_model.text_model.embed_tokens.weight.dtype();
    const kv_cache = qwen35.KvCache.init(qwen_model.config, 1, options.max_seq_len, cache_dtype, .f32);

    var compile_result_future = try io.concurrent(compileModel, .{
        allocator,
        io,
        platform,
        qwen_model,
        kv_cache,
        &progress,
        multimodal_prompt,
        replicated_sharding,
    });
    errdefer if (compile_result_future.cancel(io)) |v| {
        v.prefill_exe.deinit();
        v.decode_exe.deinit();
    } else |_| {};

    var compile_result = try compile_result_future.await(io);
    const qwen_buffers = try qwen_buffers_future.await(io);

    defer {
        compile_result.prefill_exe.deinit();
        compile_result.decode_exe.deinit();
    }
    progress.end();

    log.info(
        "\nRunning media model {s} with {d} media input(s) on prompt:\n{s}\n",
        .{ args.model, media_inputs.media_input_list.len, args.prompt },
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
        multimodal_prompt,
        media_inputs.pixel_values,
        args.gen_len,
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
    prefill_layout: qwen35.MultimodalPrompt,
    sharding: zml.sharding.Sharding,
) !CompileModelResult {
    const now: std.Io.Timestamp = .now(io, .awake);
    defer log.info("Compiled model [{f}]", .{now.untilNow(io, .awake)});
    log.info("Compiling model for platform {any} with prefill length {d}...", .{ platform.target, prefill_layout.token_ids.len });

    var prefill_future = try io.concurrent(struct {
        fn call(
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const zml.Platform,
            qwen35_model_: Qwen35,
            kv_cache_: qwen35.KvCache,
            prefill_layout_: qwen35.MultimodalPrompt,
            sharding_: zml.sharding.Sharding,
            progress_: *std.Progress.Node,
        ) !zml.Exe {
            progress_.increaseEstimatedTotalItems(1);
            var node = progress_.start("Compiling prefill...", 1);
            defer node.end();
            const now_: std.Io.Timestamp = .now(io_, .awake);
            defer log.info("Compiled prefill [{f}]", .{now_.untilNow(io_, .awake)});

            const patch_3d_size = qwen35_model_.auto_config.vision_patch_3d_size;
            var media_pixel_values = try allocator_.alloc(Tensor, prefill_layout_.media_metadata.len);
            defer allocator_.free(media_pixel_values);

            for (prefill_layout_.media_metadata, 0..) |metadata, i| {
                const patch_count = switch (metadata) {
                    .image => |m| m.patch_count,
                    .video => |m| m.patch_count,
                };
                media_pixel_values[i] = Tensor.init(.{ .p = patch_count, .ps = patch_3d_size }, .f32);
            }

            return platform_.compile(
                allocator_,
                io_,
                qwen35_model_,
                .generic_multimodal_prefill_forward,
                .{
                    Tensor.init(.{ .b = 1, .s = prefill_layout_.token_ids.len }, .u32),
                    media_pixel_values,
                    Tensor.init(.{ .g = 3, .b = 1, .s = prefill_layout_.token_ids.len }, .i64),
                    prefill_layout_.media_metadata,
                    kv_cache_,
                    zml.Tensor.Rng.init(),
                },
                .{ .shardings = &.{sharding_} },
            );
        }
    }.call, .{ allocator, io, platform, qwen35_model, kv_cache, prefill_layout, sharding, progress });
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

    const prefill_exe = try prefill_future.await(io);
    const decode_exe = try decode_future.await(io);

    return .{ .prefill_exe = prefill_exe, .decode_exe = decode_exe };
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
    prefill_layout: qwen35.MultimodalPrompt,
    media_pixel_values: []zml.Buffer,
    gen_len: usize,
    kv_cache: qwen35.KvCache,
) !void {
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&kv_cache_buffers);

    // Prefill

    const prefill_seq_len: i64 = @intCast(prefill_layout.stat.total_tokens);
    const prompt_tokens_shape = zml.Shape.init(.{ .b = 1, .s = prefill_layout.token_ids.len }, .u32);
    var prompt_tokens_slice: zml.Slice = try .alloc(allocator, prompt_tokens_shape);
    defer prompt_tokens_slice.free(allocator);
    @memcpy(prompt_tokens_slice.items(u32), prefill_layout.token_ids);
    var prompt_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_tokens_slice, sharding);
    defer prompt_tokens_buffer.deinit();

    const position_ids_shape = zml.Shape.init(.{ .g = 3, .b = 1, .s = prefill_layout.token_ids.len }, .i64);
    var position_ids_slice: zml.Slice = try .alloc(allocator, position_ids_shape);
    defer position_ids_slice.free(allocator);
    @memcpy(position_ids_slice.items(i64), prefill_layout.position_ids);
    var position_ids_buffer: zml.Buffer = try .fromSlice(io, platform, position_ids_slice, sharding);
    defer position_ids_buffer.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var prefill_args = try compile_result.prefill_exe.args(allocator);
    defer prefill_args.deinit(allocator);
    var prefill_results = try compile_result.prefill_exe.results(allocator);
    defer prefill_results.deinit(allocator);

    const prefill_now: std.Io.Timestamp = .now(io, .awake);
    prefill_args.set(.{ model_buffers, prompt_tokens_buffer, media_pixel_values, position_ids_buffer, kv_cache_buffers, rng_buffers });
    compile_result.prefill_exe.call(prefill_args, &prefill_results);
    const prefill_duration = prefill_now.untilNow(io, .awake);

    var prefill_generated_token_buffer: zml.Buffer = undefined;
    var mrope_position_deltas_buffer: zml.Buffer = undefined;
    prefill_results.fill(.{ &prefill_generated_token_buffer, &kv_cache_buffers, &mrope_position_deltas_buffer, &rng_buffers });
    defer prefill_generated_token_buffer.deinit();
    defer mrope_position_deltas_buffer.deinit();

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);
    try prefill_generated_token_buffer.toSlice(io, generated_token_slice);

    // Decode

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();

    var stdout = std.Io.File.stdout().writer(io, &.{});
    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();
    const now: std.Io.Timestamp = .now(io, .awake);
    var num_tokens_generated: usize = 1;

    for (0..gen_len) |i| {
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
            prefill_seq_len - 1 + @as(i64, @intCast(i)),
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

const LoadedMediaInputs = struct {
    pixel_values: []zml.Buffer,
    media_input_list: []qwen35.MultimodalPrompt.MediaInput,

    fn deinit(self: *LoadedMediaInputs, allocator: std.mem.Allocator) void {
        for (self.pixel_values) |*buffer| {
            buffer.deinit();
        }
        allocator.free(self.pixel_values);

        for (self.media_input_list) |media| {
            switch (media) {
                .image => {},
                .video => |video| {
                    if (video.frame_timestamps.len > 0) allocator.free(video.frame_timestamps);
                },
            }
        }
        allocator.free(self.media_input_list);

        self.* = .{
            .pixel_values = &.{},
            .media_input_list = &.{},
        };
    }
};

// Prepares the pixel values buffers and media info for each media entry in the input file.
fn loadMediaInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    path: []const u8,
) !LoadedMediaInputs {
    const MediaPixelValuesFile = struct {
        pixel_values: Tensor,
    };

    var registry: zml.safetensors.TensorRegistry = try .fromPath(allocator, io, path);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    var media_count_slice = try loadHostTensorSlice(allocator, io, store.view(), "media_count");
    defer media_count_slice.free(allocator);
    const media_count = media_count_slice.items(i64)[0];

    var pixel_values = try allocator.alloc(zml.Buffer, @intCast(media_count));
    errdefer allocator.free(pixel_values);
    var pixel_values_initialized: usize = 0;
    errdefer {
        for (pixel_values[0..pixel_values_initialized]) |*buffer| {
            buffer.deinit();
        }
    }

    var media_input_list = try allocator.alloc(qwen35.MultimodalPrompt.MediaInput, @intCast(media_count));
    errdefer allocator.free(media_input_list);
    var media_list_initialized: usize = 0;
    errdefer {
        for (media_input_list[0..media_list_initialized]) |media| {
            switch (media) {
                .image => {},
                .video => |video| {
                    if (video.frame_timestamps.len > 0) allocator.free(video.frame_timestamps);
                },
            }
        }
    }

    for (0..@intCast(media_count)) |i| {
        const media_view = store.view().withPrefix("media").withLayer(i);
        const media_pixel_values_model: MediaPixelValuesFile = .{
            .pixel_values = media_view.createTensor("pixel_values", null, null),
        };
        var loaded_media = try zml.io.load(MediaPixelValuesFile, &media_pixel_values_model, allocator, io, platform, &store, .{
            .parallelism = 1,
            .shardings = &.{sharding},
            .progress = null,
            .dma_chunks = 1,
            .dma_chunk_size = 4 * zml.MiB,
        });
        errdefer loaded_media.pixel_values.deinit();

        var media_type_slice = try loadHostTensorSlice(allocator, io, media_view, "type");
        defer media_type_slice.free(allocator);
        var grid_slice = try loadHostTensorSlice(allocator, io, media_view, "grid_thw");
        defer grid_slice.free(allocator);

        const media_type = media_type_slice.items(i64)[0];
        const grid_items = grid_slice.items(i64);
        const grid: [3]i64 = .{ grid_items[0], grid_items[1], grid_items[2] };

        pixel_values[i] = loaded_media.pixel_values;
        pixel_values_initialized += 1;

        switch (media_type) {
            // image type
            0 => {
                media_input_list[i] = .{ .image = .{ .grid_thw = grid } };
            },
            // video type
            1 => {
                var timestamps_slice = try loadHostTensorSlice(allocator, io, media_view, "timestamps");
                defer timestamps_slice.free(allocator);
                if (timestamps_slice.dtype() != .f32 or timestamps_slice.shape.rank() != 1) return error.InvalidTimestampsDtype;
                if (timestamps_slice.shape.dim(0) != grid[0]) return error.InvalidMediaTimestamps;
                const owned_ts = try allocator.dupe(f32, timestamps_slice.items(f32));
                media_input_list[i] = .{ .video = .{ .grid_thw = grid, .frame_timestamps = owned_ts } };
            },
            else => return error.InvalidMediaType,
        }

        media_list_initialized += 1;
    }

    return .{
        .pixel_values = pixel_values,
        .media_input_list = media_input_list,
    };
}

fn loadHostTensorSlice(allocator: std.mem.Allocator, io: std.Io, store: zml.io.TensorStore.View, subkey: []const u8) !zml.Slice {
    const shape = store.getShape(subkey) orelse return error.NotFound;
    var slice = try zml.Slice.alloc(allocator, shape);
    errdefer slice.free(allocator);

    var reader_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.getReader(subkey, io, &reader_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(slice.data());

    return slice;
}
