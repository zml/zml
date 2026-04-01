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
    video_width: i64 = 256,
    video_height: i64 = 256,
    video_fps: f32 = 1.0,
    max_seq_len: i64 = 4096,
    gen_len: usize = 2048,
    enable_thinking: bool = false,
};

// This demo expects a single video as raw rgb24 bytes from stdin.

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
        qwen_model.config,
        args.video_width,
        args.video_height,
        args.video_fps,
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
                .multimodal_prefill_forward,
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

// Reads raw rgb24 video frames from stdin and prepares a single media video input.
fn loadMediaInputs(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    sharding: zml.sharding.Sharding,
    config: qwen35.Qwen35.Config,
    video_width: i64,
    video_height: i64,
    video_fps: f32,
) !LoadedMediaInputs {
    const in_channels = config.vision_config.in_channels;
    const patch_size = config.vision_config.patch_size;
    const temporal_patch_size = config.vision_config.temporal_patch_size;
    const spatial_merge_size = config.vision_config.spatial_merge_size;
    const grid_h = @divExact(video_height, patch_size);
    const grid_w = @divExact(video_width, patch_size);

    if (@mod(video_width, patch_size) != 0 or @mod(video_height, patch_size) != 0) return error.InvalidVideoDimensions;
    if (@mod(grid_h, spatial_merge_size) != 0 or @mod(grid_w, spatial_merge_size) != 0) return error.InvalidVideoDimensions;

    const frame_byte_count_i64 = video_width * video_height;
    const frame_byte_count_i64_rgb = frame_byte_count_i64 * in_channels;
    const frame_byte_count: usize = @intCast(frame_byte_count_i64_rgb);

    var stdin_read_buffer: [8 * 1024]u8 = undefined;
    var stdin_reader = std.Io.File.stdin().reader(io, &stdin_read_buffer);

    const frame_buffer = try allocator.alloc(u8, frame_byte_count);
    defer allocator.free(frame_buffer);

    var packed_frames: std.ArrayList(u8) = .empty;
    defer packed_frames.deinit(allocator);
    var frame_count: usize = 0;

    while (true) {
        const eof = try readFrameFromReader(&stdin_reader, frame_buffer);
        if (eof) break;
        try packed_frames.appendSlice(allocator, frame_buffer);
        frame_count += 1;
    }

    if (frame_count == 0) return error.EmptyVideoInput;

    const temporal_patch_size_usize: usize = @intCast(temporal_patch_size);
    const grid_t_usize = try std.math.divCeil(usize, frame_count, temporal_patch_size_usize);
    const grid_t: i64 = @intCast(grid_t_usize);
    const patch_count = grid_t * grid_h * grid_w;
    const patch_3d_size = in_channels * temporal_patch_size * patch_size * patch_size;

    var pixel_values_slice = try zml.Slice.alloc(allocator, zml.Shape.init(.{ .p = patch_count, .ps = patch_3d_size }, .f32));
    defer pixel_values_slice.free(allocator);
    const pixel_values = pixel_values_slice.items(f32);

    const video_width_usize: usize = @intCast(video_width);
    const patch_size_usize: usize = @intCast(patch_size);
    const grid_h_usize: usize = @intCast(grid_h);
    const grid_w_usize: usize = @intCast(grid_w);
    const spatial_merge_size_usize: usize = @intCast(spatial_merge_size);
    const merged_h_usize = @divExact(grid_h_usize, spatial_merge_size_usize);
    const merged_w_usize = @divExact(grid_w_usize, spatial_merge_size_usize);
    const ps_usize: usize = @intCast(patch_3d_size);

    const image_mean = [3]f32{ 0.48145466, 0.4578275, 0.40821073 };
    const image_std = [3]f32{ 0.26862954, 0.2613026, 0.2757771 };

    var patch_index: usize = 0;
    for (0..grid_t_usize) |t_idx| {
        for (0..merged_h_usize) |gh_block| {
            for (0..merged_w_usize) |gw_block| {
                for (0..spatial_merge_size_usize) |gh_local| {
                    for (0..spatial_merge_size_usize) |gw_local| {
                        const gh = gh_block * spatial_merge_size_usize + gh_local;
                        const gw = gw_block * spatial_merge_size_usize + gw_local;

                        const patch_offset = patch_index * ps_usize;
                        var patch_elem: usize = 0;

                        for (0..3) |c| {
                            for (0..temporal_patch_size_usize) |tt| {
                                const source_frame = @min(t_idx * temporal_patch_size_usize + tt, frame_count - 1);
                                const frame_offset = source_frame * frame_byte_count;
                                const y_base = gh * patch_size_usize;
                                const x_base = gw * patch_size_usize;

                                for (0..patch_size_usize) |py| {
                                    for (0..patch_size_usize) |px| {
                                        const y = y_base + py;
                                        const x = x_base + px;
                                        const byte_index = frame_offset + ((y * video_width_usize + x) * 3 + c);
                                        const raw = @as(f32, @floatFromInt(packed_frames.items[byte_index])) / 255.0;
                                        pixel_values[patch_offset + patch_elem] = (raw - image_mean[c]) / image_std[c];
                                        patch_elem += 1;
                                    }
                                }
                            }
                        }

                        patch_index += 1;
                    }
                }
            }
        }
    }
    stdx.debug.assert(
        patch_index == @as(usize, @intCast(patch_count)),
        "patch_index mismatch: got={d} expected={d}",
        .{ patch_index, patch_count },
    );

    var pixel_values_buffer = try zml.Buffer.fromSlice(io, platform, pixel_values_slice, sharding);
    errdefer pixel_values_buffer.deinit();

    const timestamps = try allocator.alloc(f32, grid_t_usize);
    errdefer allocator.free(timestamps);
    for (timestamps, 0..) |*ts, i| {
        ts.* = @as(f32, @floatFromInt(i * temporal_patch_size_usize)) / video_fps;
    }

    var pixel_value_buffers = try allocator.alloc(zml.Buffer, 1);
    errdefer allocator.free(pixel_value_buffers);
    pixel_value_buffers[0] = pixel_values_buffer;

    var media_input_list = try allocator.alloc(qwen35.MultimodalPrompt.MediaInput, 1);
    errdefer allocator.free(media_input_list);
    media_input_list[0] = .{
        .video = .{
            .grid_thw = .{ grid_t, grid_h, grid_w },
            .frame_timestamps = timestamps,
        },
    };

    log.info(
        "Loaded stdin video: frames={d}, grid_thw={any}, patch_count={d}, width={d}, height={d}, fps={d:.3}",
        .{ frame_count, [3]i64{ grid_t, grid_h, grid_w }, patch_count, video_width, video_height, video_fps },
    );

    return .{
        .pixel_values = pixel_value_buffers,
        .media_input_list = media_input_list,
    };
}

fn readFrameFromReader(reader: anytype, frame_buffer: []u8) !bool {
    var offset: usize = 0;
    while (offset < frame_buffer.len) {
        const read_n = try reader.interface.readSliceShort(frame_buffer[offset..]);
        if (read_n == 0) {
            if (offset == 0) return true;
            return error.TruncatedVideoFrame;
        }
        offset += read_n;
    }
    return false;
}
