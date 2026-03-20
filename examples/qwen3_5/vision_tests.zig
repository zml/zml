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

const model = "/var/models/Qwen/Qwen3.5-0.8B";
const safetensors_file = "/home/tristan/zml/examples/qwen3_5/safetensors/vision_tests.safetensors";
const seq_len = 128;

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const options = Qwen35.GenOptions{
        .max_seq_len = seq_len,
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

    const repo = try zml.safetensors.resolveModelRepo(io, model);

    var registry: zml.safetensors.TensorRegistry = try .fromRepo(allocator, io, repo);
    defer registry.deinit();

    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const parsed_config = try parseConfig(allocator, io, repo);
    defer parsed_config.deinit();
    const config = parsed_config.value;

    var qwen_model: Qwen35 = try .init(allocator, store.view(), config, options);
    defer qwen_model.deinit(allocator);
    const model_dtype = qwen_model.text_model.embed_tokens.weight.dtype();
    const kv_cache = qwen35.KvCache.init(
        config,
        1,
        options.max_seq_len,
        model_dtype,
        .f32,
    );

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("\n{f}", .{platform.fmtVerbose()});
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);
    var kv_cache_buffers = try kv_cache.initBuffer(io, platform);
    defer qwen35.KvCache.deinitBuffer(&kv_cache_buffers);
    try zeroKvCacheBuffers(allocator, io, platform, replicated_sharding, &kv_cache_buffers);

    var progress = std.Progress.start(io, .{ .root_name = "vision_tests" });
    defer progress.end();
    var qwen35_buffers = try qwen_model.load(allocator, io, platform, &store, &.{replicated_sharding}, &progress);
    defer Qwen35.unloadBuffers(&qwen35_buffers, allocator);

    var vision_registry = try openVisionTestRegistry(allocator, io);
    defer vision_registry.deinit();
    var vision_store: zml.io.TensorStore = .fromRegistry(allocator, &vision_registry);
    defer vision_store.deinit();

    var pixel_values = try loadBufferFromStore(allocator, io, platform, replicated_sharding, &vision_store, "pixel_values");
    defer pixel_values.deinit();
    var input_ids = try loadInputIdsFromStore(allocator, io, platform, replicated_sharding, &vision_store, "input_ids");
    defer input_ids.deinit();
    var expected_output = try loadExpectedOutputFromStore(allocator, io, platform, replicated_sharding, &vision_store, "output");
    defer expected_output.deinit();
    const grid_thw = try loadGridThwFromStore(allocator, io, &vision_store, "grid_thw");

    const input_ids_tensor = Tensor.fromShape(input_ids.shape());
    const pixel_values_tensor = Tensor.fromShape(pixel_values.shape());
    var prompt_shape = try buildPromptShapeBufferFromMmTokenTypeIds(allocator, io, platform, replicated_sharding, &vision_store, "mm_token_type_ids");
    defer prompt_shape.deinit();
    const prompt_shape_tensor = Tensor.fromShape(prompt_shape.shape());
    var token_index = try zml.Buffer.scalar(io, platform, @as(u32, 0), .u32, replicated_sharding);
    defer token_index.deinit();
    const token_index_tensor = Tensor.fromShape(token_index.shape());

    const exe = try platform.compile(allocator, io, qwen_model, .vision_test_forward, .{ input_ids_tensor, token_index_tensor, kv_cache, pixel_values_tensor, grid_thw, prompt_shape_tensor }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    exe_args.set(.{ qwen35_buffers, input_ids, token_index, kv_cache_buffers, pixel_values, prompt_shape });

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);
    exe.call(exe_args, &exe_results);

    var output = exe_results.get(zml.Buffer);
    defer output.deinit();

    const output_slice = try output.toSliceAlloc(allocator, io);
    defer output_slice.free(allocator);
    const expected_slice = try expected_output.toSliceAlloc(allocator, io);
    defer expected_slice.free(allocator);

    logSliceChunk("output", output_slice, 8, 8);
    logSliceChunk("expected", expected_slice, 8, 8);

    try zml.testing.expectClose(io, output, expected_output, .{ .absolute_tolerance = 5 * 1e-2, .minimum_close_fraction = 0.9 });
    log.info("vision_test_forward output matches expected tensor", .{});
}

fn zeroBuffer(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, sharding: zml.sharding.Sharding, buffer: *zml.Buffer) !void {
    const shape = buffer.shape();
    const zero_slice: zml.Slice = try .alloc(allocator, shape);
    defer zero_slice.free(allocator);
    @memset(zero_slice.data(), 0);

    var zeroed = try zml.Buffer.fromSlice(io, platform, zero_slice, sharding);
    errdefer zeroed.deinit();
    buffer.deinit();
    buffer.* = zeroed;
}

fn zeroKvCacheBuffers(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    cache: *zml.Bufferized(qwen35.KvCache),
) !void {
    try zeroBuffer(allocator, io, platform, sharding, &cache.self_attn.k);
    try zeroBuffer(allocator, io, platform, sharding, &cache.self_attn.v);
    try zeroBuffer(allocator, io, platform, sharding, &cache.gated_delta_net.conv_state);
    try zeroBuffer(allocator, io, platform, sharding, &cache.gated_delta_net.recurrent_state);
}

//======================= Load and parse setup ========================

fn parseConfig(allocator: std.mem.Allocator, io: std.Io, dir: std.Io.Dir) !std.json.Parsed(qwen35.Qwen35.Config) {
    const now: std.Io.Timestamp = .now(io, .awake);
    log.info("Loading model config", .{});
    defer log.info("Loaded model config [{f}]", .{now.untilNow(io, .awake)});

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

fn openVisionTestRegistry(allocator: std.mem.Allocator, io: std.Io) !zml.safetensors.TensorRegistry {
    return zml.safetensors.TensorRegistry.fromPath(allocator, io, safetensors_file);
}

fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

fn loadInputIdsFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const flat_shape = if (shape.rank() == 2 and shape.dim(0) == 1)
        zml.Shape.init(.{shape.dim(1)}, shape.dtype())
    else if (shape.rank() == 1)
        shape
    else
        return error.InvalidInputIds;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, flat_shape, sharding, host_bytes);
}

fn loadExpectedOutputFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    const normalized_shape = if (shape.rank() == 3 and shape.dim(0) == 1)
        zml.Shape.init(.{ shape.dim(1), shape.dim(2) }, shape.dtype())
    else if (shape.rank() == 2)
        shape
    else
        shape;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    return zml.Buffer.fromBytes(io, platform, normalized_shape, sharding, host_bytes);
}

fn buildPromptShapeBufferFromMmTokenTypeIds(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    store: *zml.io.TensorStore,
    key: []const u8,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    if (shape.rank() != 2 or shape.dim(0) != 1) return error.InvalidMmTokenTypeIds;
    const seq_len_local: usize = @intCast(shape.dim(1));

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    const slice = zml.Slice.init(shape, host_bytes);

    const Local = struct {
        fn tokenType(shape_: zml.Shape, slice_: zml.Slice, i: usize) !i64 {
            return switch (shape_.dtype()) {
                .i32 => @intCast(slice_.constItems(i32)[i]),
                .i64 => slice_.constItems(i64)[i],
                .u32 => @intCast(slice_.constItems(u32)[i]),
                .u64 => std.math.cast(i64, slice_.constItems(u64)[i]) orelse error.InvalidMmTokenTypeIds,
                else => error.InvalidMmTokenTypeIds,
            };
        }
    };

    var first_image: ?usize = null;
    var last_image: ?usize = null;
    for (0..seq_len_local) |i| {
        if (try Local.tokenType(shape, slice, i) == 1) {
            if (first_image == null) first_image = i;
            last_image = i;
        }
    }

    const text_before_image: i64, const image_tokens: i64, const text_after_image: i64 = if (first_image) |start| blk: {
        const end = last_image.?;
        for (start..end + 1) |i| {
            if (try Local.tokenType(shape, slice, i) != 1) return error.InvalidMmTokenTypeIds;
        }
        break :blk .{
            @intCast(start),
            @intCast(end - start + 1),
            @intCast(seq_len_local - end - 1),
        };
    } else .{ @intCast(seq_len_local), 0, 0 };

    const prompt_shape_data = [3]i64{
        text_before_image,
        image_tokens,
        text_after_image,
    };
    const prompt_shape = zml.Shape.init(.{3}, .i64);
    return zml.Buffer.fromBytes(io, platform, prompt_shape, sharding, std.mem.sliceAsBytes(&prompt_shape_data));
}

fn loadGridThwFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    store: *zml.io.TensorStore,
    key: []const u8,
) ![3]i64 {
    const shape = store.view().getShape(key) orelse return error.NotFound;
    if (shape.count() != 3) return error.InvalidGridThw;

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();
    _ = try reader.interface.readSliceAll(host_bytes);

    const slice = zml.Slice.init(shape, host_bytes);
    return switch (shape.dtype()) {
        .u32 => blk: {
            const items = slice.constItems(u32);
            break :blk .{
                @intCast(items[0]),
                @intCast(items[1]),
                @intCast(items[2]),
            };
        },
        .u64 => blk: {
            const items = slice.constItems(u64);
            break :blk .{
                std.math.cast(i64, items[0]) orelse return error.InvalidGridThw,
                std.math.cast(i64, items[1]) orelse return error.InvalidGridThw,
                std.math.cast(i64, items[2]) orelse return error.InvalidGridThw,
            };
        },
        .i32 => blk: {
            const items = slice.constItems(i32);
            break :blk .{
                @intCast(items[0]),
                @intCast(items[1]),
                @intCast(items[2]),
            };
        },
        .i64 => blk: {
            const items = slice.constItems(i64);
            break :blk .{ items[0], items[1], items[2] };
        },
        else => return error.InvalidGridThw,
    };
}

fn logSliceChunk(name: []const u8, slice: zml.Slice, max_rows: usize, max_cols: usize) void {
    const shape = slice.shape;
    if (shape.rank() != 2) {
        log.info("{s} shape {f} (non-2D tensor, skipping detailed dump)", .{ name, shape });
        return;
    }

    const rows_total: usize = @intCast(shape.dim(0));
    const cols_total: usize = @intCast(shape.dim(1));
    const rows: usize = @min(max_rows, rows_total);
    const cols: usize = @min(max_cols, cols_total);

    log.info("{s} shape {f}, showing first {d}x{d} values:", .{ name, shape, rows, cols });
    switch (shape.dtype()) {
        .f32 => dumpFloatChunk(f32, name, slice.constItems(f32), cols_total, rows, cols),
        .f16 => dumpFloatChunk(f16, name, slice.constItems(f16), cols_total, rows, cols),
        .bf16 => dumpFloatChunk(zml.floats.BFloat16, name, slice.constItems(zml.floats.BFloat16), cols_total, rows, cols),
        else => log.info("{s}: chunk dump supports only f32/f16/bf16, got {}", .{ name, shape.dtype() }),
    }
}

fn dumpFloatChunk(comptime T: type, name: []const u8, data: []const T, cols_total: usize, rows: usize, cols: usize) void {
    for (0..rows) |r| {
        var line_buf: [512]u8 = undefined;
        var used: usize = 0;

        used += (std.fmt.bufPrint(line_buf[used..], "{s}[{d},0:{d}] ", .{ name, r, cols }) catch break).len;
        used += (std.fmt.bufPrint(line_buf[used..], "[", .{}) catch break).len;
        for (0..cols) |c| {
            const idx = r * cols_total + c;
            const v = zml.floats.floatCast(f32, data[idx]);
            if (c + 1 < cols) {
                used += (std.fmt.bufPrint(line_buf[used..], "{d:.6} ", .{v}) catch break).len;
            } else {
                used += (std.fmt.bufPrint(line_buf[used..], "{d:.6}", .{v}) catch break).len;
            }
        }
        used += (std.fmt.bufPrint(line_buf[used..], "]", .{}) catch break).len;
        log.info("{s}", .{line_buf[0..used]});
    }
}
