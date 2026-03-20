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
    gen_tokens: usize = 128,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const args = stdx.flags.parse(init.minimal.args, CliArgs);

    const options = Qwen35.GenOptions{
        .max_seq_len = args.len,
        .sampling_strategy = .{
            .topk = 1,
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

    var tokenizer = try loadTokenizer(allocator, io, repo, &progress);
    defer tokenizer.deinit();

    var qwen_buffers = try qwen35.Qwen35.load(
        &qwen_model,
        allocator,
        io,
        platform,
        &store,
        &.{replicated_sharding},
        &progress,
    );
    defer Qwen35.unloadBuffers(&qwen_buffers, allocator);

    progress.end();

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

    // ----------------------------------------------------------------
    // First pass: Qwen35.vision_test_forward, then decode with Qwen35.forward.
    // ----------------------------------------------------------------
    try runGenerationLoop(
        allocator,
        io,
        platform,
        qwen_model,
        qwen_buffers,
        replicated_sharding,
        tokenizer,
        prompt_tokens,
        prompt_shape,
        vision_inputs.image_grid_thw,
        vision_inputs.pixel_values,
        args.gen_tokens,
    );
}

fn runGenerationLoop(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    model: Qwen35,
    model_buffers: zml.Bufferized(Qwen35),
    sharding: zml.sharding.Sharding,
    tokenizer: zml.tokenizer.Tokenizer,
    input_token_ids: []const u32,
    prompt_shape_values: [3]i64,
    image_grid_thw: [3]i64,
    pixel_values_buffer: zml.Buffer,
    gen_tokens: usize,
) !void {
    if (input_token_ids.len == 0) return;

    const cache_dtype = model.text_model.embed_tokens.weight.dtype();
    const required_seq_len: i64 = @intCast(input_token_ids.len + gen_tokens + 1);
    const generation_cache_seq_len = @max(model.gen_options.max_seq_len, required_seq_len);
    const kv_cache = qwen35.KvCache.init(model.config, 1, generation_cache_seq_len, cache_dtype, .f32);
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&kv_cache_buffers);

    const prefill_len = input_token_ids.len;
    const total_seq_len: i64 = prompt_shape_values[0] + prompt_shape_values[1] + prompt_shape_values[2];
    const decode_exe = try platform.compile(
        allocator,
        io,
        model,
        .vision_test_decode_forward,
        .{
            Tensor.init(.{ .b = 1, .s = 1 }, .u32),
            Tensor.init(.{}, .u32),
            kv_cache,
            Tensor.init(.{ .s = 1 }, .i64),
            zml.Tensor.Rng.init(),
        },
        .{ .shardings = &.{sharding} },
    );
    defer decode_exe.deinit();

    const grid_thw = image_grid_thw;
    const t = image_grid_thw[0];
    const h = image_grid_thw[1];
    const w = image_grid_thw[2];
    const patch_flat_dim: i64 = model.config.vision_config.in_channels *
        model.config.vision_config.temporal_patch_size *
        model.config.vision_config.patch_size *
        model.config.vision_config.patch_size;
    const pixel_rows: i64 = t * h * w;

    const pixel_shape = pixel_values_buffer.shape();
    if (pixel_shape.rank() != 2 or pixel_shape.dim(0) != pixel_rows or pixel_shape.dim(1) != patch_flat_dim) {
        log.err(
            "pixel_values shape mismatch: got {f}, expected {{n={d}, f={d}}}",
            .{ pixel_shape, pixel_rows, patch_flat_dim },
        );
        return error.InvalidPixelValuesShape;
    }

    const vision_prefill_exe = try platform.compile(
        allocator,
        io,
        model,
        .vision_test_forward,
        .{
            Tensor.init(.{ .b = 1, .s = prefill_len }, .u32),
            Tensor.init(.{}, .u32),
            kv_cache,
            Tensor.init(.{ .n = pixel_rows, .f = patch_flat_dim }, .f32),
            grid_thw,
            Tensor.init(.{3}, .i64),
            zml.Tensor.Rng.init(),
        },
        .{ .shardings = &.{sharding} },
    );
    defer vision_prefill_exe.deinit();

    const prompt_tokens_shape = zml.Shape.init(.{ .b = 1, .s = prefill_len }, .u32);
    var prompt_tokens_slice: zml.Slice = try .alloc(allocator, prompt_tokens_shape);
    defer prompt_tokens_slice.free(allocator);
    @memcpy(prompt_tokens_slice.items(u32), input_token_ids);
    var prompt_tokens_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_tokens_slice, sharding);
    defer prompt_tokens_buffer.deinit();

    var prefill_token_index_buffer = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32, sharding);
    defer prefill_token_index_buffer.deinit();

    const prompt_shape_shape = zml.Shape.init(.{3}, .i64);
    var prompt_shape_slice: zml.Slice = try .alloc(allocator, prompt_shape_shape);
    defer prompt_shape_slice.free(allocator);
    @memcpy(prompt_shape_slice.items(i64), &prompt_shape_values);
    var prompt_shape_buffer: zml.Buffer = try .fromSlice(io, platform, prompt_shape_slice, sharding);
    defer prompt_shape_buffer.deinit();

    var rng_buffers = try zml.Tensor.Rng.initBuffer(platform, 0, io, sharding);
    defer zml.Tensor.Rng.deinitBuffer(&rng_buffers);

    var vision_prefill_args = try vision_prefill_exe.args(allocator);
    defer vision_prefill_args.deinit(allocator);
    var vision_prefill_results = try vision_prefill_exe.results(allocator);
    defer vision_prefill_results.deinit(allocator);

    vision_prefill_args.set(.{
        model_buffers,
        prompt_tokens_buffer,
        prefill_token_index_buffer,
        kv_cache_buffers,
        pixel_values_buffer,
        prompt_shape_buffer,
        rng_buffers,
    });
    vision_prefill_exe.call(vision_prefill_args, &vision_prefill_results);
    var prefill_generated_token_buffer: zml.Buffer = undefined;
    var mrope_position_deltas_buffer: zml.Buffer = undefined;
    vision_prefill_results.fill(.{ &prefill_generated_token_buffer, &kv_cache_buffers, &mrope_position_deltas_buffer, &rng_buffers });
    defer prefill_generated_token_buffer.deinit();
    defer mrope_position_deltas_buffer.deinit();
    log.info("vision_test_forward output shape: {any}", .{prefill_generated_token_buffer.shape()});

    var generated_token_slice: zml.Slice = try .alloc(allocator, zml.Shape.init(.{ .b = 1, .s = 1 }, .u32));
    defer generated_token_slice.free(allocator);
    try prefill_generated_token_buffer.toSlice(io, generated_token_slice);

    var current_token_buffer: zml.Buffer = try .fromSlice(io, platform, generated_token_slice, sharding);
    defer current_token_buffer.deinit();

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();
    var generated_ids: std.ArrayList(u32) = try .initCapacity(allocator, gen_tokens);
    defer generated_ids.deinit(allocator);

    var stdout = std.Io.File.stdout().writer(io, &.{});
    try stdout.interface.writeAll("generated token ids: ");

    for (0..gen_tokens) |i| {
        try current_token_buffer.toSlice(io, generated_token_slice);
        const generated_token = generated_token_slice.items(u32)[0];
        try generated_ids.append(allocator, generated_token);

        var id_buf: [32]u8 = undefined;
        const token_text = try std.fmt.bufPrint(&id_buf, "{d}", .{generated_token});
        try stdout.interface.writeAll(token_text);
        try stdout.interface.writeAll(if (i + 1 == gen_tokens) "\n" else " ");

        if (generated_token == model.special_tokens.end_of_text_token_id) break;

        var token_index_buffer = try zml.Buffer.scalar(
            io,
            platform,
            @as(u32, @intCast(total_seq_len - 1 + @as(i64, @intCast(i)))),
            .u32,
            sharding,
        );
        defer token_index_buffer.deinit();

        var decode_args = try decode_exe.args(allocator);
        defer decode_args.deinit(allocator);
        var decode_results = try decode_exe.results(allocator);
        defer decode_results.deinit(allocator);

        decode_args.set(.{ model_buffers, current_token_buffer, token_index_buffer, kv_cache_buffers, mrope_position_deltas_buffer, rng_buffers });
        decode_exe.call(decode_args, &decode_results);

        decode_results.fill(.{ &current_token_buffer, &kv_cache_buffers, &rng_buffers });
    }

    try stdout.interface.writeAll("generated text: ");
    for (generated_ids.items) |generated_token| {
        if (generated_token == model.special_tokens.end_of_text_token_id) break;
        if (try tokenizer_decoder.next(generated_token)) |chunk| {
            try stdout.interface.writeAll(chunk);
        }
    }
    try stdout.interface.writeAll("\n");
    try stdout.interface.flush();
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
    var loaded = try zml.io.load(PixelValuesOnly, &model, allocator, io, platform, .{
        .parallelism = 1,
        .store = &store,
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
