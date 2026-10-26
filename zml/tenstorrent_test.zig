const std = @import("std");
const zml = @import("zml");

test "TTIR gated DeltaNet matches scalar recurrence and masks inactive positions" {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const platform = try zml.Platform.init(std.testing.allocator, std.testing.io, .tenstorrent, .{});
    defer platform.deinit(std.testing.allocator, std.testing.io);
    const heads = 2;
    const key_dim = 32;
    const value_dim = 32;
    inline for (.{ false, true }) |batched| {
        const channels = heads * (if (batched) 2 else 1);
        inline for (.{ 1, 3 }) |seq| {
            const q: zml.Tensor = if (batched) .init(.{ .b = 2, .s = seq, .h = heads, .k = key_dim }, .f32) else .init(.{ .s = seq, .h = heads, .k = key_dim }, .f32);
            const k: zml.Tensor = .init(q.shape(), .f32);
            const v: zml.Tensor = if (batched) .init(.{ .b = 2, .s = seq, .h = heads, .v = value_dim }, .f32) else .init(.{ .s = seq, .h = heads, .v = value_dim }, .f32);
            const gate: zml.Tensor = if (batched) .init(.{ .b = 2, .s = seq, .h = heads }, .f32) else .init(.{ .s = seq, .h = heads }, .f32);
            const beta: zml.Tensor = .init(gate.shape(), .f32);
            const state: zml.Tensor = if (batched) .init(.{ .b = 2, .h = heads, .v = value_dim, .k = key_dim }, .f32) else .init(.{ .h = heads, .v = value_dim, .k = key_dim }, .f32);
            const active: zml.Tensor = .init(.{}, .i32);
            var exe = try platform.compileFn(std.testing.allocator, std.testing.io, zml.nn.GatedDeltaNet.forward, .{ q, k, v, gate, beta, .{ .s = state }, active }, .{});
            defer exe.deinit();
            var runner = try exe.runner(std.testing.allocator);
            defer runner.deinit(std.testing.allocator);
            var q_data: [seq * channels * key_dim]f32 = undefined;
            var k_data: [seq * channels * key_dim]f32 = undefined;
            var v_data: [seq * channels * value_dim]f32 = undefined;
            var initial: [channels * value_dim * key_dim]f32 = undefined;
            for (&q_data, &k_data, 0..) |*qi, *ki, i| {
                qi.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 7)) - 3)) / 16;
                ki.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 5)) - 2)) / 16;
            }
            for (&v_data, 0..) |*vi, i| vi.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 9)) - 4)) / 8;
            for (&initial, 0..) |*si, i| si.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 11)) - 5)) / 64;
            const alpha_data: [seq * channels]f32 = @splat(0.75);
            const beta_data: [seq * channels]f32 = @splat(0.5);
            var qb = try zml.Buffer.fromBytes(std.testing.io, platform, q.shape(), std.mem.sliceAsBytes(&q_data));
            defer qb.deinit();
            var kb = try zml.Buffer.fromBytes(std.testing.io, platform, k.shape(), std.mem.sliceAsBytes(&k_data));
            defer kb.deinit();
            var vb = try zml.Buffer.fromBytes(std.testing.io, platform, v.shape(), std.mem.sliceAsBytes(&v_data));
            defer vb.deinit();
            var ab = try zml.Buffer.fromBytes(std.testing.io, platform, gate.shape(), std.mem.sliceAsBytes(&alpha_data));
            defer ab.deinit();
            var bb = try zml.Buffer.fromBytes(std.testing.io, platform, gate.shape(), std.mem.sliceAsBytes(&beta_data));
            defer bb.deinit();
            var sb = try zml.Buffer.fromBytes(std.testing.io, platform, state.shape(), std.mem.sliceAsBytes(&initial));
            defer sb.deinit();
            for ([_]i32{ -1, 0, 1, seq, seq + 2 }) |active_len| {
                var expected_state = initial;
                var expected_output: [seq * channels * value_dim]f32 = @splat(0);
                for (0..@intCast(@max(0, @min(active_len, seq)))) |t| {
                    for (0..channels) |h| {
                        const token_head = if (batched) ((h / heads) * seq + t) * heads + h % heads else t * heads + h;
                        for (0..value_dim) |row| {
                            const state_offset = (h * value_dim + row) * key_dim;
                            const vector_offset = token_head * key_dim;
                            var prediction: f32 = 0;
                            for (0..key_dim) |col| prediction += expected_state[state_offset + col] * k_data[vector_offset + col];
                            prediction *= alpha_data[token_head];
                            const delta = (v_data[token_head * value_dim + row] - prediction) * beta_data[token_head];
                            var y: f32 = 0;
                            for (0..key_dim) |col| {
                                const next = expected_state[state_offset + col] * alpha_data[token_head] + delta * k_data[vector_offset + col];
                                expected_state[state_offset + col] = next;
                                y += next * q_data[vector_offset + col];
                            }
                            expected_output[token_head * value_dim + row] = y;
                        }
                    }
                }
                var length = try zml.Buffer.scalar(std.testing.io, platform, active_len, .i32);
                defer length.deinit();
                var result: zml.Bufferized(zml.nn.GatedDeltaNet.Output) = undefined;
                runner.run(std.testing.io, .{ qb, kb, vb, ab, bb, .{ .s = sb }, length }, .{&result}, .{ .wait = true });
                defer result.state.s.deinit();
                defer result.outputs.deinit();
                try zml.testing.expectClose(std.testing.io, zml.Slice.init(state.shape(), std.mem.sliceAsBytes(&expected_state)), result.state.s, .{ .absolute_tolerance = 1e-4, .relative_tolerance = 1e-4 });
                try zml.testing.expectClose(std.testing.io, zml.Slice.init(v.shape(), std.mem.sliceAsBytes(&expected_output)), result.outputs, .{ .absolute_tolerance = 1e-4, .relative_tolerance = 1e-4 });
            }
        }
    }
}

test "two-card tensor-parallel matmul and donated shards" {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{
        .tenstorrent = .{ .device_index = null },
    });
    defer platform.deinit(allocator, io);
    try std.testing.expectEqual(@as(usize, 2), platform.devices.len);
    const mesh = try platform.registerMesh("model", .mesh(.{ .model = .high_bandwidth }));
    const Functions = struct {
        fn matmul(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
            return a.dot(b, .k).replicate();
        }
        fn update(value: zml.Tensor) zml.Tensor {
            return value.addConstant(1).reuseBuffer(value);
        }
        fn argmax(value: zml.Tensor) zml.Tensor {
            return value.argMax(.k).indices.replicate();
        }
    };
    const a = zml.Tensor.init(.{ .m = 32, .k = 64 }, .f32).withPartitioning(mesh, .{ .k = .model });
    const b = zml.Tensor.init(.{ .k = 64, .n = 32 }, .f32).withPartitioning(mesh, .{ .k = .model });
    const exe = try platform.compileFn(allocator, io, Functions.matmul, .{ a, b }, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    var a_data: [32][64]f32 = undefined;
    var b_data: [64][32]f32 = undefined;
    for (&a_data) |*row| {
        for (row, 0..) |*value, k| value.* = if (k < 32) 1 else 3;
    }
    for (&b_data, 0..) |*row, k| row.* = @splat(if (k < 32) 2 else 4);
    var ab = try zml.Buffer.fromBytes(io, platform, a.shape(), std.mem.asBytes(&a_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, b.shape(), std.mem.asBytes(&b_data));
    defer bb.deinit();
    for (0..3) |_| {
        var output = try exe.eval(allocator, io, .{ ab, bb });
        defer output.deinit();
        const actual = try output.getValue([32][32]f32, io);
        for (actual) |row| for (row) |value| {
            try std.testing.expectEqual(@as(f32, 448), value);
        };
    }
    const update = try platform.compileFn(allocator, io, Functions.update, .{a}, .{ .meshes = &.{mesh} });
    defer update.deinit();
    var runner = try update.runner(allocator);
    defer runner.deinit(allocator);
    for (0..3) |iteration| {
        runner.run(io, .{ab}, .{&ab}, .{ .wait = true });
        const actual = try ab.toSliceAlloc(allocator, io);
        defer actual.free(allocator);
        for (actual.items(f32), 0..) |value, i| {
            const k = i % 64;
            const initial: f32 = if (k < 32) 1 else 3;
            try std.testing.expectEqual(initial + @as(f32, @floatFromInt(iteration + 1)), value);
        }
    }
    try std.testing.expectEqual(b_data, try bb.getValue([64][32]f32, io));
    const argmax = try platform.compileFn(allocator, io, Functions.argmax, .{a}, .{ .meshes = &.{mesh} });
    defer argmax.deinit();
    var indices = try argmax.eval(allocator, io, .{ab});
    defer indices.deinit();
    for (try indices.getValue([32]i32, io)) |index| try std.testing.expectEqual(@as(i32, 32), index);
}
