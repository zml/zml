const std = @import("std");
const zml = @import("zml");
const model = @import("llama/model.zig");

test "head-major Llama cache preserves donated prefill and decode updates" {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = 0 } });
    defer platform.deinit(allocator, io);
    const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));
    const bf16 = zml.DataType.bf16.toZigType();
    const cache = model.KvCache.init(.init(.{ .layer = 2, .h = 8, .k = 64, .hd = 32 }, .bf16), mesh);
    var expected_k: [2 * 8 * 64 * 32]bf16 = undefined;
    var expected_v: [2 * 8 * 64 * 32]bf16 = undefined;
    for (&expected_k, &expected_v, 0..) |*k, *v, i| {
        k.* = .fromF32(@as(f32, @floatFromInt(i % 19)) / 16);
        v.* = .fromF32(-@as(f32, @floatFromInt(i % 23)) / 16);
    }
    var buffers: model.KvCache.Buffer = .{
        .k = try zml.Buffer.fromBytes(io, platform, cache.k.shape(), std.mem.asBytes(&expected_k)),
        .v = try zml.Buffer.fromBytes(io, platform, cache.v.shape(), std.mem.asBytes(&expected_v)),
    };
    defer model.KvCache.deinitBuffer(&buffers);
    const Update = struct {
        fn run(kv: model.KvCache, k: zml.Tensor, v: zml.Tensor, position: zml.Tensor) model.KvCache {
            var result = kv;
            for (0..2) |layer| {
                result = result.updateAt(k.addConstant(@as(f32, @floatFromInt(layer))), v, position, .scalar(@as(u32, @intCast(layer)), .u32));
            }
            return result.reuseBuffer(kv);
        }
    };
    inline for (.{ 32, 1 }) |seq| {
        const update: zml.Tensor = .init(.{ .k = seq, .h = 8, .hd = 32 }, .bf16);
        const value_update: zml.Tensor = .init(update.shape(), .bf16);
        const position: zml.Tensor = .init(.{}, .u32);
        const exe = try platform.compileFn(allocator, io, Update.run, .{ cache, update, value_update, position }, .{ .meshes = &.{mesh} });
        defer exe.deinit();
        try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, exe.donated_input_indices);
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        var new_k: [seq * 8 * 32]bf16 = undefined;
        var new_v: [seq * 8 * 32]bf16 = undefined;
        for (&new_k, &new_v, 0..) |*k, *v, i| {
            k.* = .fromF32(@as(f32, @floatFromInt(i % 29)) / 8);
            v.* = .fromF32(-@as(f32, @floatFromInt(i % 31)) / 8);
        }
        var kb = try zml.Buffer.fromBytes(io, platform, update.shape(), std.mem.asBytes(&new_k));
        defer kb.deinit();
        var vb = try zml.Buffer.fromBytes(io, platform, update.shape(), std.mem.asBytes(&new_v));
        defer vb.deinit();
        const positions = if (seq == 32) &[_]u32{ 0, 32 } else &[_]u32{ 0, 31, 32, 63 };
        for (positions) |pos| {
            var pb = try zml.Buffer.scalar(io, platform, pos, .u32);
            defer pb.deinit();
            runner.run(io, .{ buffers, kb, vb, pb }, .{&buffers}, .{ .wait = true });
            for (0..2) |layer| {
                for (0..8) |head| {
                    for (0..seq) |token| {
                        for (0..32) |d| {
                            const dst = ((layer * 8 + head) * 64 + pos + token) * 32 + d;
                            const src = (token * 8 + head) * 32 + d;
                            expected_k[dst] = .fromF32(new_k[src].toF32() + @as(f32, @floatFromInt(layer)));
                            expected_v[dst] = new_v[src];
                        }
                    }
                }
            }
            try zml.testing.expectClose(io, zml.Slice.init(cache.k.shape(), std.mem.asBytes(&expected_k)), buffers.k, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
            try zml.testing.expectClose(io, zml.Slice.init(cache.v.shape(), std.mem.asBytes(&expected_v)), buffers.v, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
        }
        try zml.testing.expectClose(io, zml.Slice.init(update.shape(), std.mem.asBytes(&new_k)), kb, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
        try zml.testing.expectClose(io, zml.Slice.init(update.shape(), std.mem.asBytes(&new_v)), vb, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
    }
}

test "Llama first generated tokens fill the next cache rows after prefill" {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const session = @import("llama/session.zig");
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = 0 } });
    defer platform.deinit(allocator, io);
    const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));
    const bf16 = zml.DataType.bf16.toZigType();
    const cache = model.KvCache.init(.init(.{ .layer = 2, .h = 8, .k = 64, .hd = 32 }, .bf16), mesh);
    var expected_k: [2 * 8 * 64 * 32]bf16 = undefined;
    var expected_v: [2 * 8 * 64 * 32]bf16 = undefined;
    for (&expected_k, &expected_v, 0..) |*k, *v, i| {
        k.* = .fromF32(@as(f32, @floatFromInt(i % 19)) / 16);
        v.* = .fromF32(-@as(f32, @floatFromInt(i % 23)) / 16);
    }
    var buffers: model.KvCache.Buffer = .{
        .k = try zml.Buffer.fromBytes(io, platform, cache.k.shape(), std.mem.asBytes(&expected_k)),
        .v = try zml.Buffer.fromBytes(io, platform, cache.v.shape(), std.mem.asBytes(&expected_v)),
    };
    defer model.KvCache.deinitBuffer(&buffers);
    const Update = struct {
        fn run(kv: model.KvCache, k: zml.Tensor, v: zml.Tensor, position: zml.Tensor) model.KvCache {
            var result = kv;
            for (0..2) |layer| {
                result = result.updateAt(k.addConstant(@as(f32, @floatFromInt(layer))), v, position, .scalar(@as(u32, @intCast(layer)), .u32));
            }
            return result.reuseBuffer(kv);
        }
    };
    const prefix_len = 32;
    var tokens: std.ArrayList(u32) = .empty;
    defer tokens.deinit(allocator);
    for (0..prefix_len) |_| try tokens.append(allocator, 1);
    inline for (.{ prefix_len, 1 }) |rows| {
        const row: zml.Tensor = .init(.{ .k = rows, .h = 8, .hd = 32 }, .bf16);
        const values: zml.Tensor = .init(row.shape(), .bf16);
        const exe = try platform.compileFn(allocator, io, Update.run, .{ cache, row, values, zml.Tensor.init(.{}, .u32) }, .{ .meshes = &.{mesh} });
        defer exe.deinit();
        try std.testing.expectEqualSlices(u32, &.{ 0, 1 }, exe.donated_input_indices);
        var runner = try exe.runner(allocator);
        defer runner.deinit(allocator);
        const steps = if (rows == prefix_len) 1 else 2;
        for (0..steps) |step| {
            const value: f32 = if (rows == prefix_len) 0.5 else @as(f32, @floatFromInt(step + 2));
            var new_k: [rows * 8 * 32]bf16 = undefined;
            var new_v: [rows * 8 * 32]bf16 = undefined;
            for (&new_k, &new_v, 0..) |*k, *v, i| {
                k.* = .fromF32(value + @as(f32, @floatFromInt(i % 29)) / 8);
                v.* = .fromF32(-value - @as(f32, @floatFromInt(i % 31)) / 8);
            }
            var kb = try zml.Buffer.fromBytes(io, platform, row.shape(), std.mem.asBytes(&new_k));
            defer kb.deinit();
            var vb = try zml.Buffer.fromBytes(io, platform, values.shape(), std.mem.asBytes(&new_v));
            defer vb.deinit();
            // This follows runDecode: publish the current token, then consume it.
            // Its position comes from the production helper, while the expected
            // cache frontier below comes independently from the prefilled prefix.
            const position = if (rows == prefix_len) 0 else blk: {
                try tokens.append(allocator, @intCast(step + 2));
                break :blk session.decodePosition(tokens.items.len);
            };
            var position_buffer = try zml.Buffer.scalar(io, platform, position, .u32);
            defer position_buffer.deinit();
            runner.run(io, .{ buffers, kb, vb, position_buffer }, .{&buffers}, .{ .wait = true });
            const expected_start = if (rows == prefix_len) 0 else prefix_len + step;
            for (0..2) |layer| {
                for (0..8) |head| {
                    for (0..rows) |offset| {
                        for (0..32) |d| {
                            const index = ((layer * 8 + head) * 64 + expected_start + offset) * 32 + d;
                            const source = (offset * 8 + head) * 32 + d;
                            expected_k[index] = .fromF32(new_k[source].toF32() + @as(f32, @floatFromInt(layer)));
                            expected_v[index] = new_v[source];
                        }
                    }
                }
            }
            // Entire-cache equality catches both a skipped first slot and any
            // overwrite of the prefilled prefix or still-unused suffix.
            errdefer std.log.err("Boundary cache update failed: rows={d}, step={d}, position={d}", .{ rows, step, position });
            try zml.testing.expectClose(io, zml.Slice.init(cache.k.shape(), std.mem.asBytes(&expected_k)), buffers.k, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
            try zml.testing.expectClose(io, zml.Slice.init(cache.v.shape(), std.mem.asBytes(&expected_v)), buffers.v, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
            try zml.testing.expectClose(io, zml.Slice.init(row.shape(), std.mem.asBytes(&new_k)), kb, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
            try zml.testing.expectClose(io, zml.Slice.init(values.shape(), std.mem.asBytes(&new_v)), vb, .{ .absolute_tolerance = 0, .relative_tolerance = 0 });
        }
    }
}
