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

    var platform: *zml.Platform = try .auto(allocator, io, .{});
    defer platform.deinit(allocator);
    log.info("\n{f}", .{platform.fmtVerbose()});
    const replicated_sharding = try zml.sharding.replicatedSharding(platform);

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
    var expected_output = try loadBufferFromStore(allocator, io, platform, replicated_sharding, &vision_store, "output");
    defer expected_output.deinit();
    const grid_thw = try loadGridThwFromStore(allocator, io, &vision_store, "grid_thw");

    const pixel_values_tensor = Tensor.fromShape(pixel_values.shape());

    const exe = try platform.compile(allocator, io, qwen_model, .vision_test_forward, .{ pixel_values_tensor, grid_thw }, .{ .shardings = &.{replicated_sharding} });
    defer exe.deinit();

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);
    exe_args.set(.{ qwen35_buffers, pixel_values });

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

    try zml.testing.expectClose(io, output, expected_output, .{ .absolute_tolerance = 5 * 1e-2 });
    log.info("vision_test_forward output matches expected tensor", .{});
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
