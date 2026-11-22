//! y = x * 2 + 1 on a tensor sharded over every device of a platform, the
//! smallest program the Tenstorrent backend compiles today, and the weight
//! loader that feeds such tensors.

const std = @import("std");

const zml = @import("zml");

pub const cols = 96;
/// Rows per device: two tile rows.
pub const rows_per_device = 64;

fn scaleShift(x: zml.Tensor) zml.Tensor {
    // Keeps y sharded like x: a replicated result would need an all-gather.
    return x.scale(2).addConstant(1).withPartitioning(.rows, .{ .b = .rows, .d = .replicated });
}

/// Compiles and runs y = x * 2 + 1 on `platform`, with x sharded along its
/// rows over all its devices. Returns y, owned by the caller.
pub fn run(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, x: []const f32) ![]f32 {
    const mesh = try rowsMesh(platform);
    const shape = zml.Shape.init(.{ .b = @as(i64, @intCast(x.len / cols)), .d = cols }, .f32)
        .withPartitioning(mesh, .{ .b = .rows });

    var exe = try platform.compileFn(allocator, io, scaleShift, .{zml.Tensor.fromShape(shape)}, .{
        .meshes = &.{mesh},
        .program_name = "scale_shift",
    });
    defer exe.deinit();

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, shape, std.mem.sliceAsBytes(x));
    defer x_buffer.deinit();
    try std.testing.expectEqual(platform.devices.len, x_buffer.numShards());

    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var y_buffer: zml.Buffer = undefined;
    runner.run(io, .{x_buffer}, .{&y_buffer}, .{ .wait = true });
    defer y_buffer.deinit();

    const y = try y_buffer.toSliceAlloc(allocator, io);
    defer y.free(allocator);
    return allocator.dupe(f32, y.items(f32));
}

fn rowsMesh(platform: *zml.Platform) !*const zml.Mesh {
    return platform.meshes.get("rows") orelse platform.registerMesh("rows", .mesh(.{ .rows = .high_bandwidth }));
}

/// Loads weights sharded over every device through `zml.io.Loader`, the path
/// of model weights, and checks that each reads back unchanged. Several weights
/// of one size load at once, as llmd's loader does (16 in parallel): every
/// shard must land in its own buffer whatever order the uploads interleave in.
pub fn expectLoaderRoundTrip(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !void {
    return expectLoaderRoundTripIn(allocator, io, platform, .default);
}

/// `expectLoaderRoundTrip` with weights that target `memory`, as
/// `TensorStore.View.createHostPinnedTensor` makes them for `.host_pinned`.
pub fn expectLoaderRoundTripIn(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, comptime memory: zml.Memory.Kind) !void {
    const num_weights = 16;
    const per_weight = platform.devices.len * rows_per_device * cols;
    const x = try allocator.alloc(f32, num_weights * per_weight);
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @floatFromInt(i % 65521);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "weights", .data = std.mem.sliceAsBytes(x) });
    const path = try tmp.dir.realPathFileAlloc(io, "weights", allocator);
    defer allocator.free(path);

    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    var name_bufs: [num_weights][8]u8 = undefined;
    var names: [num_weights][]const u8 = undefined;
    for (&names, &name_bufs, 0..) |*name, *buf, i| name.* = try std.fmt.bufPrint(buf, "w{d}", .{i});
    for (names, 0..) |name, i| {
        try registry.registerTensor(.{
            .name = name,
            .file_uri = path,
            .shape = .init(.{ @as(i64, @intCast(per_weight / cols)), cols }, .f32),
            .offset = i * per_weight * @sizeOf(f32),
        });
    }
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{try rowsMesh(platform)});
    defer store.deinit();

    const Weights = struct { w: []zml.Tensor };
    var tensors: [num_weights]zml.Tensor = undefined;
    for (&tensors, names) |*w, name| w.* = switch (memory) {
        .default => store.view().createTensor(name, .{ .b, .d }, .rows, .{ .b = .rows }),
        .host_pinned => store.view().createHostPinnedTensor(name, .{ .b, .d }, .rows, .{ .b = .rows }),
        else => @compileError("weights target the default or the pinned host memory"),
    };
    const weights: Weights = .{ .w = &tensors };
    var buffers = try zml.mem.bufferize(allocator, Weights, &weights);
    defer allocator.free(buffers.w);

    var loader: zml.io.Loader = try .init(allocator, platform, .{ .parallelism = num_weights, .dma_chunks = 4, .dma_chunk_size = 1 << 20 });
    defer loader.deinit();
    try loader.load(io, Weights, &weights, &buffers, &store, .{});
    try loader.await(io);
    defer for (buffers.w) |*b| b.deinit();

    const expected_kind = platform.memoryKind(if (memory == .default) .device else memory);
    for (buffers.w, 0..) |*b, i| {
        try std.testing.expectEqual(platform.devices.len, b.numShards());
        var shards = b.shards();
        while (shards.next()) |shard| {
            try std.testing.expectEqualStrings(expected_kind, shard._pjrt_buffer.memory(platform.pjrt_api).kind_(platform.pjrt_api));
        }
        const back = try b.toSliceAlloc(allocator, io);
        defer back.free(allocator);
        try std.testing.expectEqualSlices(f32, x[i * per_weight ..][0..per_weight], back.items(f32));
    }
}

const LinearWeights = struct { w: zml.Tensor };

fn linear(weights: LinearWeights, x: zml.Tensor) zml.Tensor {
    // ZML's linear layer: x [s, d] . w [dout, d] contracted on d, as Llama's.
    return x.dot(weights.w, .d).withPartitioning(.rows, .{ .s = .replicated, .dout = .rows });
}

/// A linear layer's weight, [dout, d] split over every device along dout:
/// the executable reports the layout it stores it in (on Tenstorrent, tiles
/// with the two minor dims swapped: the matmul reads it without a transpose),
/// the loader uploads it in that layout, it reads back unchanged, and the
/// layer's result matches the host's. Returns the weight's layout, printed.
pub fn expectLinearLayerWeights(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform, layout_buf: []u8) ![]const u8 {
    const mesh = try rowsMesh(platform);
    const dout: i64 = @intCast(platform.devices.len * rows_per_device);
    const d: i64 = 64;
    // Small integers: every product and sum is exact in f32.
    const w = try allocator.alloc(f32, @intCast(dout * d));
    defer allocator.free(w);
    for (w, 0..) |*v, i| v.* = @floatFromInt(@as(i64, @intCast((i * 13) % 9)) - 4);
    const x = try allocator.alloc(f32, @intCast(32 * d));
    defer allocator.free(x);
    for (x, 0..) |*v, i| v.* = @floatFromInt(@as(i64, @intCast((i * 7) % 11)) - 5);

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "linear", .data = std.mem.sliceAsBytes(w) });
    const path = try tmp.dir.realPathFileAlloc(io, "linear", allocator);
    defer allocator.free(path);
    var registry: zml.safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .name = "w", .file_uri = path, .shape = .init(.{ dout, d }, .f32), .offset = 0 });
    var store: zml.io.TensorStore = .fromRegistry(allocator, &registry, &.{mesh});
    defer store.deinit();
    const weights: LinearWeights = .{ .w = store.view().createTensor("w", .{ .dout, .d }, .rows, .{ .dout = .rows }) };

    const x_shape = zml.Shape.init(.{ .s = 32, .d = d }, .f32).withPartitioning(mesh, .{ .s = .replicated, .d = .replicated });
    var exe = try platform.compileFn(allocator, io, linear, .{ weights, zml.Tensor.fromShape(x_shape) }, .{
        .meshes = &.{mesh},
        .program_name = "linear",
    });
    defer exe.deinit();

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();
    const layouts = (try exe.modelLayouts(arena.allocator(), &weights)).?;
    try std.testing.expectEqual(1, layouts.len);
    const printed = try std.fmt.bufPrint(layout_buf, "{f}", .{layouts[0]});

    var buffers = try zml.mem.bufferize(allocator, LinearWeights, &weights);
    var loader: zml.io.Loader = try .init(allocator, platform, .{ .parallelism = 1, .dma_chunks = 2, .dma_chunk_size = 1 << 20 });
    defer loader.deinit();
    try loader.load(io, LinearWeights, &weights, &buffers, &store, .{ .layouts = layouts });
    try loader.await(io);
    defer buffers.w.deinit();

    const back = try buffers.w.toSliceAlloc(allocator, io);
    defer back.free(allocator);
    try std.testing.expectEqualSlices(f32, w, back.items(f32));

    var x_buffer: zml.Buffer = try .fromBytes(io, platform, x_shape, std.mem.sliceAsBytes(x));
    defer x_buffer.deinit();
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    var y_buffer: zml.Buffer = undefined;
    runner.run(io, .{ buffers, x_buffer }, .{&y_buffer}, .{ .wait = true });
    defer y_buffer.deinit();
    const y = try y_buffer.toSliceAlloc(allocator, io);
    defer y.free(allocator);
    const n: usize = @intCast(dout);
    const k: usize = @intCast(d);
    for (0..32) |i| {
        for (0..n) |j| {
            var sum: f32 = 0;
            for (0..k) |c| sum += x[i * k + c] * w[j * k + c];
            try std.testing.expectEqual(sum, y.items(f32)[i * n + j]);
        }
    }
    return printed;
}

/// Inputs whose doubling and increment are exact in f32 on any backend.
pub fn input(allocator: std.mem.Allocator, num_devices: usize) ![]f32 {
    const x = try allocator.alloc(f32, num_devices * rows_per_device * cols);
    for (x, 0..) |*v, i| v.* = @as(f32, @floatFromInt(@as(i64, @intCast(i % 4099)) - 2048)) * 0.25;
    return x;
}

/// Runs the program on `platform` and on a CPU platform, and checks that both
/// give the same bits.
pub fn expectMatchesCpu(allocator: std.mem.Allocator, io: std.Io, platform: *zml.Platform) !void {
    const n = platform.devices.len;
    const x = try input(allocator, n);
    defer allocator.free(x);

    const y = try run(allocator, io, platform, x);
    defer allocator.free(y);

    // The CPU platform keeps ZML's power-of-two device rule; shards do not change y.
    const cpu_devices: u32 = if (std.math.isPowerOfTwo(n)) @intCast(n) else 1;
    const cpu: *zml.Platform = try .init(allocator, io, .cpu, .{ .cpu = .{ .device_count = cpu_devices } });
    defer cpu.deinit(allocator, io);
    const expected = try run(allocator, io, cpu, x);
    defer allocator.free(expected);

    for (x, expected) |xi, e| try std.testing.expectEqual(xi * 2 + 1, e);
    try std.testing.expectEqualSlices(f32, expected, y);
}
