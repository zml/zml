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

fn checkDecodeAttention(comptime batch: usize, comptime heads: usize, comptime kv_heads: usize, comptime dim: usize, comptime seq: usize, comptime mesh_execution: bool) !void {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = if (mesh_execution) null else 0 } });
    defer platform.deinit(allocator, io);
    const bf16 = zml.DataType.bf16.toZigType();
    var q: zml.Tensor = if (batch == 1) .init(.{ .q = 1, .h = heads, .hd = dim }, .bf16) else .init(.{ .b = batch, .q = 1, .h = heads, .hd = dim }, .bf16);
    var k: zml.Tensor = if (batch == 1) .init(.{ .k = seq, .h = kv_heads, .hd = dim }, .bf16) else .init(.{ .b = batch, .k = seq, .h = kv_heads, .hd = dim }, .bf16);
    var v: zml.Tensor = .init(k.shape(), .bf16);
    if (mesh_execution) {
        const mesh = try platform.registerMesh("attention_heads", .mesh(.{ .model = .high_bandwidth }));
        q = q.withPartitioning(mesh, .{ .h = .model });
        k = k.withPartitioning(mesh, .{ .h = .model });
        v = v.withPartitioning(mesh, .{ .h = .model });
    }
    const pos: zml.Tensor = if (batch == 1) .init(.{}, .u32) else .init(.{ .b = batch }, .u32);
    const metadata: zml.attention.Metadata = .{ .flash_attention = .{ .tenstorrent = {} } };
    const parameters: zml.attention.Parameters = .{ .flash_attention = .{ .tenstorrent = {} } };
    const Attention = struct {
        fn run(query: zml.Tensor, key: zml.Tensor, value: zml.Tensor, position: zml.Tensor, meta: zml.attention.Metadata, params: zml.attention.Parameters) zml.Tensor {
            // Elementwise results leave shard dimensions open, as do Llama's
            // cache slices. The round trip preserves these BF16 values exactly.
            const computed_key = if (batch > 1) key.convert(.f32).convert(.bf16) else key;
            return zml.attention.attention(query, computed_key, value, position, meta, params);
        }
    };
    const exe = try platform.compileFn(allocator, io, Attention.run, .{ q, k, v, pos, metadata, parameters }, .{ .meshes = platform.meshes.values() });
    defer exe.deinit();
    var q_data: [batch * heads * dim]bf16 = undefined;
    var k_data: [batch * seq * kv_heads * dim]bf16 = undefined;
    var v_data: [batch * seq * kv_heads * dim]bf16 = undefined;
    for (&q_data, 0..) |*x, i| x.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16);
    for (&k_data, &v_data, 0..) |*ki, *vi, i| {
        ki.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16);
        vi.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 8);
    }
    var qb = try zml.Buffer.fromBytes(io, platform, q.shape(), std.mem.asBytes(&q_data));
    defer qb.deinit();
    var kb = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.asBytes(&k_data));
    defer kb.deinit();
    var vb = try zml.Buffer.fromBytes(io, platform, v.shape(), std.mem.asBytes(&v_data));
    defer vb.deinit();
    for ([_]u32{ 0, 1, 31, 32, seq / 2 - 1, seq / 2, seq - 1, seq + 17, std.math.maxInt(u32) }) |last_position| {
        var positions: [batch]u32 = undefined;
        for (&positions, 0..) |*position, b| position.* = last_position -| @as(u32, @intCast(3 * b));
        var pb = try zml.Buffer.fromBytes(io, platform, pos.shape(), std.mem.asBytes(&positions));
        defer pb.deinit();
        var actual = try exe.eval(allocator, io, .{ qb, kb, vb, pb, metadata });
        defer actual.deinit();
        const host = try actual.toSliceAlloc(allocator, io);
        defer host.free(allocator);
        var max_error: f64 = 0;
        var max_error_ratio: f64 = 0;
        for (positions, 0..) |raw_position, b| {
            const position: usize = @min(raw_position, seq - 1);
            for (0..heads) |h| {
                var logits: [seq]f64 = undefined;
                var denom: f64 = 0;
                for (0..position + 1) |s| {
                    var dot: f64 = 0;
                    for (0..dim) |d| dot += @as(f64, q_data[(b * heads + h) * dim + d].toF32()) * k_data[((b * seq + s) * kv_heads + h / (heads / kv_heads)) * dim + d].toF32();
                    logits[s] = @exp(dot / @sqrt(@as(f64, dim)));
                    denom += logits[s];
                }
                for (0..dim) |d| {
                    var expected: f64 = 0;
                    for (0..position + 1) |s| expected += logits[s] / denom * v_data[((b * seq + s) * kv_heads + h / (heads / kv_heads)) * dim + d].toF32();
                    const actual_value: f64 = host.items(bf16)[(b * heads + h) * dim + d].toF32();
                    try std.testing.expect(std.math.isFinite(actual_value));
                    const err = @abs(expected - actual_value);
                    // Match the established BF16 attention tolerances, while
                    // requiring every element to satisfy the scalar oracle.
                    const tolerance = 5e-3 + 1e-2 * @abs(expected);
                    max_error = @max(max_error, err);
                    max_error_ratio = @max(max_error_ratio, err / tolerance);
                }
            }
        }
        errdefer std.debug.print("attention batch {} sequence {} position {} max error {d}, tolerance ratio {d}\n", .{ batch, seq, last_position, max_error, max_error_ratio });
        try std.testing.expect(max_error_ratio <= 1);
    }
}

test "TTIR decode attention matches scalar causal GQA" {
    try checkDecodeAttention(1, 32, 8, 128, 128, false);
    try checkDecodeAttention(1, 32, 8, 128, 1024, false);
    try checkDecodeAttention(2, 16, 4, 64, 128, false);
    try checkDecodeAttention(1, 24, 4, 256, 128, false);
}

test "two-card TTIR attention preserves GQA head partitions" {
    try checkDecodeAttention(1, 32, 8, 128, 128, true);
    try checkDecodeAttention(2, 16, 4, 64, 1024, true);
    try checkDecodeAttention(1, 24, 4, 256, 128, true);
}

test "Tenstorrent rotary fusion preserves head values and BF16 rotation" {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = 0 } });
    defer platform.deinit(allocator, io);
    const bf16 = zml.DataType.bf16.toZigType();
    inline for (.{ 8, 32 }) |heads| {
        const x: zml.Tensor = .init(.{ .s = 1, .h = heads, .hd = 128 }, .bf16);
        const pos: zml.Tensor = .init(.{ .s = 1 }, .u32);
        const Rotation = struct {
            fn run(input: zml.Tensor, position: zml.Tensor) zml.Tensor {
                return zml.nn.rope(input, position, .{ .layout = .real_im_pass });
            }
        };
        const exe = try platform.compileFn(allocator, io, Rotation.run, .{ x, pos }, .{});
        defer exe.deinit();
        var values: [heads * 128]bf16 = undefined;
        for (&values, 0..) |*value, i| value.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 29)) - 14)) / 8);
        var xb = try zml.Buffer.fromBytes(io, platform, x.shape(), std.mem.asBytes(&values));
        defer xb.deinit();
        for ([_]u32{ 0, 1, 31, 32, 127, 1023 }) |position| {
            var positions = [_]u32{position};
            var pb = try zml.Buffer.fromBytes(io, platform, pos.shape(), std.mem.asBytes(&positions));
            defer pb.deinit();
            var result = try exe.eval(allocator, io, .{ xb, pb });
            defer result.deinit();
            const actual = try result.toSliceAlloc(allocator, io);
            defer actual.free(allocator);
            for (actual.items(bf16), 0..) |value, i| {
                const d = i % 128;
                const frequency = @exp(-@log(@as(f32, 10000)) * @as(f32, @floatFromInt(d % 64)) / 64);
                const angle = @as(f32, @floatFromInt(position)) * frequency;
                const c = bf16.fromF32(@cos(angle)).toF32();
                const s = bf16.fromF32(@sin(angle)).toF32();
                const real = values[(i / 128) * 128 + d % 64].toF32();
                const imag = values[(i / 128) * 128 + d % 64 + 64].toF32();
                const expected = if (d < 64)
                    bf16.fromF32(bf16.fromF32(real * c).toF32() - bf16.fromF32(imag * s).toF32()).toF32()
                else
                    bf16.fromF32(bf16.fromF32(real * s).toF32() + bf16.fromF32(imag * c).toF32()).toF32();
                const observed = value.toF32();
                try std.testing.expect(std.math.isFinite(observed));
                const tolerance = 0.016 + 0.008 * @abs(expected);
                if (@abs(observed - expected) > tolerance) {
                    std.debug.print("rotary heads={d} position={d} element={d}: expected {d}, got {d}\n", .{ heads, position, i, expected, observed });
                    return error.TestExpectedApproxEqAbs;
                }
            }
        }
    }
}
fn checkPrefillAttention(comptime batch: usize, comptime heads: usize, comptime kv_heads: usize, comptime dim: usize, comptime queries: usize, comptime keys: usize, comptime options: struct { scalar_position: bool = true, transposed: bool = false, unbatched: bool = false }) !void {
    const scalar_position = options.scalar_position;
    const transposed = options.transposed;
    const unbatched = options.unbatched;
    comptime std.debug.assert(!unbatched or (batch == 1 and scalar_position));
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{ .tenstorrent = .{ .device_index = 0 } });
    defer platform.deinit(allocator, io);
    const bf16 = zml.DataType.bf16.toZigType();
    const q: zml.Tensor = .init(.{ .b = batch, .q = queries, .h = heads, .hd = dim }, .bf16);
    const k: zml.Tensor = .init(.{ .b = batch, .k = keys, .h = kv_heads, .hd = dim }, .bf16);
    const v: zml.Tensor = .init(k.shape(), .bf16);
    const pos: zml.Tensor = if (scalar_position) .init(.{}, .u32) else .init(.{ .b = batch }, .u32);
    const Attention = struct {
        fn run(query: zml.Tensor, key: zml.Tensor, value: zml.Tensor, position: zml.Tensor) zml.Tensor {
            const q_local = if (unbatched) query.reshape(.{ .q = queries, .h = heads, .hd = dim }) else query;
            const k_local = if (unbatched) key.reshape(.{ .k = keys, .h = kv_heads, .hd = dim }) else key;
            const v_local = if (unbatched) value.reshape(.{ .k = keys, .h = kv_heads, .hd = dim }) else value;
            const result = zml.attention.attention(
                if (!transposed) q_local else if (unbatched) q_local.transpose(.{ .h, .hd, .q }) else q_local.transpose(.{ .h, .b, .hd, .q }),
                if (!transposed) k_local else if (unbatched) k_local.transpose(.{ .hd, .k, .h }) else k_local.transpose(.{ .hd, .k, .b, .h }),
                if (!transposed) v_local else if (unbatched) v_local.transpose(.{ .hd, .k, .h }) else v_local.transpose(.{ .hd, .k, .b, .h }),
                position,
                .{ .flash_attention = .{ .tenstorrent = {} } },
                .{ .flash_attention = .{ .tenstorrent = {} } },
            );
            return result.transpose(q_local.shape()).reshape(query.shape());
        }
    };
    const exe = try platform.compileFn(allocator, io, Attention.run, .{ q, k, v, pos }, .{});
    defer exe.deinit();
    const q_data = try allocator.alloc(bf16, batch * queries * heads * dim);
    defer allocator.free(q_data);
    const k_data = try allocator.alloc(bf16, batch * keys * kv_heads * dim);
    defer allocator.free(k_data);
    const v_data = try allocator.alloc(bf16, k_data.len);
    defer allocator.free(v_data);
    for (q_data, 0..) |*x, i| x.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 16);
    for (k_data, v_data, 0..) |*ki, *vi, i| {
        ki.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 16);
        vi.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 8);
    }
    var qb = try zml.Buffer.fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(q_data));
    defer qb.deinit();
    var kb = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(k_data));
    defer kb.deinit();
    var vb = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(v_data));
    defer vb.deinit();
    const offsets: []const u32 = if (queries == keys) &.{0} else &.{ 0, @min(31, keys - queries), keys - queries };
    for (offsets) |offset| {
        var positions: [if (scalar_position) 1 else batch]u32 = undefined;
        for (&positions, 0..) |*position, b| position.* = offset -| @as(u32, @intCast(3 * b));
        var pb = try zml.Buffer.fromBytes(io, platform, pos.shape(), std.mem.asBytes(&positions));
        defer pb.deinit();
        // Repeat identical inputs to exercise native program/trace replay.
        for (0..2) |_| {
            var actual = try exe.eval(allocator, io, .{ qb, kb, vb, pb });
            defer actual.deinit();
            const host = try actual.toSliceAlloc(allocator, io);
            defer host.free(allocator);
            for (0..batch) |b| {
                for (0..queries) |row| {
                    const last = positions[if (scalar_position) 0 else b] + row;
                    for (0..heads) |h| {
                        var scores: [keys]f64 = undefined;
                        var maximum: f64 = -std.math.inf(f64);
                        for (0..last + 1) |s| {
                            var dot: f64 = 0;
                            for (0..dim) |d| dot += @as(f64, q_data[((b * queries + row) * heads + h) * dim + d].toF32()) * k_data[((b * keys + s) * kv_heads + h / (heads / kv_heads)) * dim + d].toF32();
                            scores[s] = dot / @sqrt(@as(f64, dim));
                            maximum = @max(maximum, scores[s]);
                        }
                        var denominator: f64 = 0;
                        for (scores[0 .. last + 1]) |*score| {
                            score.* = @exp(score.* - maximum);
                            denominator += score.*;
                        }
                        for (0..dim) |d| {
                            var expected: f64 = 0;
                            for (0..last + 1) |s| expected += scores[s] / denominator * v_data[((b * keys + s) * kv_heads + h / (heads / kv_heads)) * dim + d].toF32();
                            const observed: f64 = host.items(bf16)[((b * queries + row) * heads + h) * dim + d].toF32();
                            if (!std.math.isFinite(observed) or @abs(expected - observed) > 5e-3 + 1e-2 * @abs(expected)) {
                                std.debug.print("prefill B={} H={}/{} D={} Q={} K={} t={} b={} row={} h={} d={}: expected {d}, got {d}\n", .{ batch, heads, kv_heads, dim, queries, keys, offset, b, row, h, d, expected, observed });
                                return error.TestExpectedApproxEqAbs;
                            }
                        }
                    }
                }
            }
        }
    }
    inline for (.{ .{ qb, q_data }, .{ kb, k_data }, .{ vb, v_data } }) |input| {
        const host = try input[0].toSliceAlloc(allocator, io);
        defer host.free(allocator);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(input[1]), std.mem.sliceAsBytes(host.items(bf16)));
    }
}

test "TTIR flash prefill matches scalar causal attention" {
    try checkPrefillAttention(1, 8, 2, 128, 128, 128, .{});
    try checkPrefillAttention(1, 4, 1, 64, 1024, 1024, .{});
    try checkPrefillAttention(2, 8, 2, 64, 8, 64, .{ .scalar_position = false });
    try checkPrefillAttention(2, 4, 4, 256, 3, 65, .{ .transposed = true });
    try checkPrefillAttention(1, 4, 1, 512, 33, 97, .{});
    try checkPrefillAttention(1, 4, 1, 80, 7, 35, .{ .transposed = true });
    try checkPrefillAttention(1, 4, 1, 80, 17, 17, .{});
    try checkPrefillAttention(1, 4, 1, 80, 1, 35, .{});
    try checkPrefillAttention(32, 8, 2, 64, 8, 64, .{ .scalar_position = false });
}

test "TTIR flash prefill preserves unbatched tagged layouts" {
    try checkPrefillAttention(1, 4, 1, 80, 17, 17, .{ .unbatched = true });
    try checkPrefillAttention(1, 8, 2, 80, 7, 35, .{ .transposed = true, .unbatched = true });
    try checkPrefillAttention(1, 4, 1, 80, 1, 35, .{ .transposed = true, .unbatched = true });
}

fn checkQwenDeltaNet(comptime seq: usize, comptime mesh_execution: bool) !void {
    if (comptime !zml.Target.tenstorrent.isEnabled()) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = try zml.Platform.init(allocator, io, .tenstorrent, .{
        .tenstorrent = .{ .device_index = if (mesh_execution) null else 0 },
    });
    defer platform.deinit(allocator, io);
    if (mesh_execution) try std.testing.expectEqual(@as(usize, 2), platform.devices.len);
    const bf16 = zml.DataType.bf16.toZigType();
    const heads = 48;
    const dim = 128;
    var q = zml.Tensor.init(.{ .s = seq, .h = heads, .k = dim }, .bf16);
    var k = zml.Tensor.init(q.shape(), .bf16);
    var v = zml.Tensor.init(.{ .s = seq, .h = heads, .v = dim }, .bf16);
    var alpha = zml.Tensor.init(.{ .s = seq, .h = heads }, .f32);
    var beta = zml.Tensor.init(alpha.shape(), .f32);
    var state = zml.Tensor.init(.{ .h = heads, .v = dim, .k = dim }, .f32);
    if (mesh_execution) {
        const mesh = try platform.registerMesh("delta_heads", .mesh(.{ .model = .high_bandwidth }));
        q = q.withPartitioning(mesh, .{ .h = .model });
        k = k.withPartitioning(mesh, .{ .h = .model });
        v = v.withPartitioning(mesh, .{ .h = .model });
        alpha = alpha.withPartitioning(mesh, .{ .h = .model });
        beta = beta.withPartitioning(mesh, .{ .h = .model });
        state = state.withPartitioning(mesh, .{ .h = .model });
    }
    const Functions = struct {
        fn forward(query: zml.Tensor, key: zml.Tensor, value: zml.Tensor, decay: zml.Tensor, update: zml.Tensor, initial: zml.Tensor, length: zml.Tensor) zml.nn.GatedDeltaNet.Output {
            // Model projections and cache slices leave partitioning open.
            // Exercise inference of the common head partition from the mesh.
            var result = zml.nn.GatedDeltaNet.forward(query.addConstant(0), key.addConstant(0), value.addConstant(0), decay, update, .{ .s = initial.addConstant(0) }, length);
            result.state.s = result.state.s.reuseBuffer(initial);
            if (value.shape()._sharding.mesh) |mesh| {
                result.outputs = result.outputs.withPartitioning(mesh, value.shape()._sharding.partition);
            }
            return result;
        }
    };
    const exe = try platform.compileFn(allocator, io, Functions.forward, .{ q, k, v, alpha, beta, state, zml.Tensor.init(.{}, .i32) }, .{ .meshes = platform.meshes.values() });
    defer exe.deinit();
    try std.testing.expectEqualSlices(u32, &.{5}, exe.donated_input_indices);
    var runner = try exe.runner(allocator);
    defer runner.deinit(allocator);
    const q_data = try allocator.alloc(bf16, seq * heads * dim);
    defer allocator.free(q_data);
    const k_data = try allocator.alloc(bf16, q_data.len);
    defer allocator.free(k_data);
    const v_data = try allocator.alloc(bf16, q_data.len);
    defer allocator.free(v_data);
    const initial = try allocator.alloc(f32, heads * dim * dim);
    defer allocator.free(initial);
    const expected_state = try allocator.alloc(f64, initial.len);
    defer allocator.free(expected_state);
    const state_values = try allocator.alloc(f32, initial.len);
    defer allocator.free(state_values);
    const expected_output = try allocator.alloc(bf16, v_data.len);
    defer allocator.free(expected_output);
    var alpha_data: [seq * heads]f32 = undefined;
    var beta_data: [seq * heads]f32 = undefined;
    for (q_data, k_data, v_data, 0..) |*qi, *ki, *vi, i| {
        qi.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 17)) - 8)) / 64);
        ki.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 13)) - 6)) / 128);
        vi.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 19)) - 9)) / 32);
    }
    for (initial, expected_state, 0..) |*si, *ei, i| {
        si.* = @as(f32, @floatFromInt(@as(i32, @intCast(i % 23)) - 11)) / 512;
        ei.* = si.*;
    }
    for (&alpha_data, &beta_data, 0..) |*ai, *bi, i| {
        ai.* = 0.5 + @as(f32, @floatFromInt(i % 4)) / 8;
        bi.* = 0.25 + @as(f32, @floatFromInt(i % 3)) / 8;
    }
    var qb = try zml.Buffer.fromBytes(io, platform, q.shape(), std.mem.sliceAsBytes(q_data));
    defer qb.deinit();
    var kb = try zml.Buffer.fromBytes(io, platform, k.shape(), std.mem.sliceAsBytes(k_data));
    defer kb.deinit();
    var vb = try zml.Buffer.fromBytes(io, platform, v.shape(), std.mem.sliceAsBytes(v_data));
    defer vb.deinit();
    var ab = try zml.Buffer.fromBytes(io, platform, alpha.shape(), std.mem.asBytes(&alpha_data));
    defer ab.deinit();
    var bb = try zml.Buffer.fromBytes(io, platform, beta.shape(), std.mem.asBytes(&beta_data));
    defer bb.deinit();
    var sb = try zml.Buffer.fromBytes(io, platform, state.shape(), std.mem.sliceAsBytes(initial));
    defer sb.deinit();
    var retained: ?zml.Buffer = null;
    defer if (retained) |*buffer| buffer.deinit();
    var retained_values: ?[]bf16 = null;
    defer if (retained_values) |values| allocator.free(values);
    for ([_]i32{ 2, 0, 3, -1, 1, seq + 2 }) |active| {
        @memset(expected_output, bf16.fromF32(0));
        for (0..@intCast(@max(0, @min(active, seq)))) |t| {
            for (0..heads) |h| {
                const token_head = t * heads + h;
                for (0..dim) |row| {
                    const base = (h * dim + row) * dim;
                    var prediction: f64 = 0;
                    for (0..dim) |col| prediction += expected_state[base + col] * k_data[token_head * dim + col].toF32();
                    const delta = (v_data[token_head * dim + row].toF32() - prediction * alpha_data[token_head]) * beta_data[token_head];
                    var output: f64 = 0;
                    for (0..dim) |col| {
                        expected_state[base + col] = expected_state[base + col] * alpha_data[token_head] + delta * k_data[token_head * dim + col].toF32();
                        output += expected_state[base + col] * q_data[token_head * dim + col].toF32();
                    }
                    expected_output[token_head * dim + row] = .fromF32(@floatCast(output));
                }
            }
        }
        for (state_values, expected_state) |*value, expected| value.* = @floatCast(expected);
        var length = try zml.Buffer.scalar(io, platform, active, .i32);
        defer length.deinit();
        var result: zml.Bufferized(zml.nn.GatedDeltaNet.Output) = undefined;
        runner.run(io, .{ qb, kb, vb, ab, bb, sb, length }, .{&result}, .{ .wait = true });
        sb = result.state.s;
        try zml.testing.expectClose(io, zml.Slice.init(state.shape(), std.mem.sliceAsBytes(state_values)), sb, .{ .absolute_tolerance = 1e-4, .relative_tolerance = 1e-4 });
        try zml.testing.expectClose(io, zml.Slice.init(v.shape(), std.mem.sliceAsBytes(expected_output)), result.outputs, .{ .absolute_tolerance = 1e-3, .relative_tolerance = 1e-2 });
        if (retained) |buffer| {
            defer result.outputs.deinit();
            try zml.testing.expectClose(io, zml.Slice.init(v.shape(), std.mem.sliceAsBytes(retained_values.?)), buffer, .{ .absolute_tolerance = 1e-3, .relative_tolerance = 1e-2 });
        } else {
            retained = result.outputs;
            retained_values = try allocator.dupe(bf16, expected_output);
        }
    }
    inline for (.{ .{ qb, q_data }, .{ kb, k_data }, .{ vb, v_data } }) |input| {
        const host = try input[0].toSliceAlloc(allocator, io);
        defer host.free(allocator);
        try std.testing.expectEqualSlices(u8, std.mem.sliceAsBytes(input[1]), std.mem.sliceAsBytes(host.items(bf16)));
    }
    inline for (.{ .{ ab, &alpha_data }, .{ bb, &beta_data } }) |input| {
        const host = try input[0].toSliceAlloc(allocator, io);
        defer host.free(allocator);
        try std.testing.expectEqualSlices(f32, input[1], host.items(f32));
    }
}

test "Qwen-sized BF16 DeltaNet preserves donated state" {
    try checkQwenDeltaNet(1, false);
    try checkQwenDeltaNet(3, false);
}

test "two-card Qwen-sized BF16 DeltaNet preserves head partitions and donated state" {
    try checkQwenDeltaNet(1, true);
    try checkQwenDeltaNet(3, true);
}
