const std = @import("std");

const zml = @import("zml");

// LmHead's greedy vocabulary reduction returns the same U32 token on every rank.
pub fn readGreedyToken(buffer: zml.Buffer, io: std.Io) !u32 {
    std.debug.assert(buffer._platform.target == .furiosa);
    std.debug.assert(buffer.shape().dtype() == .u32 and buffer.shape().count() == 1);
    // Donation copies and tuple writes can fail after the shared group finishes.
    try buffer.await(io);
    var token: u32 = undefined;
    const api = buffer._platform.pjrt_api;
    if (try buffer._shards.get(0).toHostBuffer(api, std.mem.asBytes(&token))) |event| {
        defer event.deinit(api);
        try event.await(api, io);
    }
    return token;
}

pub fn nextPosition(position: zml.Tensor) zml.Tensor {
    return position.addConstant(1).reuseBuffer(position);
}

test "nextPosition: donated decode positions advance from the consumed token" {
    const io = std.testing.io;
    const platform = zml.testing.env();
    var exe = try platform.compileFn(std.testing.allocator, io, nextPosition, .{zml.Tensor.init(.{}, .u32)}, .{});
    defer exe.deinit();
    for ([_]u32{ 0, 3, 511, 1022 }) |start| {
        var position = try zml.Buffer.scalar(io, platform, start, .u32);
        defer position.deinit();
        for (1..4) |step| {
            position = try exe.eval(std.testing.allocator, io, .{position});
            try std.testing.expectEqual(start + @as(u32, @intCast(step)), try position.getValue(u32, io));
        }
    }
}

/// Furiosa Llama uses compile-time rotation tables for valid integer positions
/// in [0, capacity), where capacity is the model's KV-cache length.
pub fn rope(x: zml.Tensor, positions: zml.Tensor, opts: zml.nn.RopeOpts, capacity: u32) zml.Tensor {
    std.debug.assert(zml.Compiler.current().platform.target == .furiosa);
    std.debug.assert(capacity > 0 and positions.dtype().isInteger());
    const head_dim = x.dim(.hd);
    const rotary_dim = opts.scaling.partialRotaryDim(head_dim);
    std.debug.assert(@mod(head_dim, 2) == 0 and rotary_dim > 0 and @mod(rotary_dim, 2) == 0);
    const allocator = zml.Compiler.current().allocator;
    const frequencies = allocator.alloc(f32, rotary_dim / 2) catch @panic("OOM");
    defer allocator.free(frequencies);
    zml.nn.fillInvFreq(opts, frequencies, @intCast(@divExact(head_dim, 2)));
    const elements = @as(usize, capacity) * frequencies.len;
    const cos_values = allocator.alloc(f32, elements) catch @panic("OOM");
    defer allocator.free(cos_values);
    const sin_values = allocator.alloc(f32, elements) catch @panic("OOM");
    defer allocator.free(sin_values);
    const scaling = opts.scaling.attentionScaling();
    for (0..capacity) |position| {
        for (frequencies, 0..) |frequency, frequency_index| {
            const angle: f32 = @as(f32, @floatFromInt(position)) * frequency;
            const i = position * frequencies.len + frequency_index;
            cos_values[i] = @cos(angle) * scaling;
            sin_values[i] = @sin(angle) * scaling;
        }
    }
    const shape = zml.Shape.init(.{ .rope_position = capacity, .hd = rotary_dim / 2 }, .f32);
    const cos_table = zml.Tensor.constantTensor(shape, @ptrCast(cos_values)).convert(x.dtype());
    const sin_table = zml.Tensor.constantTensor(shape, @ptrCast(sin_values)).convert(x.dtype());
    const real, const imag, const pass = zml.nn.splitRealImgPass(x, opts.layout, rotary_dim);
    const cos = cos_table.gather(.{ .rope_position = positions }, .{}).broad(real.shape());
    const sin = sin_table.gather(.{ .rope_position = positions }, .{}).broad(real.shape());
    return zml.nn.mergeRealImgPass(real.mul(cos).sub(imag.mul(sin)), real.mul(sin).add(imag.mul(cos)), pass, opts.layout);
}

test "rope: cached BF16 rotations match dynamic scaling, layouts and boundaries" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    const Bf16 = zml.floats.BFloat16;
    const Local = struct {
        fn forward(x: zml.Tensor, positions: zml.Tensor, opts: zml.nn.RopeOpts) zml.Tensor {
            return zml.nn.rope(x, positions, opts);
        }

        fn cached(x: zml.Tensor, positions: zml.Tensor, opts: zml.nn.RopeOpts) zml.Tensor {
            return rope(x, positions, opts, 1024);
        }
    };
    const llama3: zml.nn.RopeOpts.Scaling = .{ .llama3 = .{ .factor = 8, .low_freq_factor = 1, .high_freq_factor = 4, .original_max_position_embeddings = 8192, .rope_theta = 500_000 } };
    const Case = struct { opts: zml.nn.RopeOpts, queries: usize };
    for ([_]Case{
        .{ .opts = .{ .scaling = llama3 }, .queries = 1 },
        .{ .opts = .{ .scaling = llama3 }, .queries = 8 },
        .{ .opts = .{ .layout = .interleaved }, .queries = 1 },
        .{ .opts = .{ .layout = .real_pass_im_pass, .scaling = .{ .proportional = .{ .partial_rotary_factor = 0.5 } } }, .queries = 8 },
        .{ .opts = .{ .layout = .interleaved, .scaling = .{ .yarn = .{ .factor = 4, .original_max_position_embeddings = 8192, .partial_rotary_factor = 0.5 } } }, .queries = 1 },
        .{ .opts = .{ .scaling = .{ .linear = .{ .factor = 4 } } }, .queries = 8 },
    }) |case| {
        const x: zml.Tensor = .init(.{ .b = 1, .s = case.queries, .hd = 128 }, .bf16);
        const positions: zml.Tensor = .init(.{ .s = case.queries }, .u32);
        var direct_exe = try platform.compileFn(allocator, io, Local.forward, .{ x, positions, case.opts }, .{});
        defer direct_exe.deinit();
        var cached_exe = try platform.compileFn(allocator, io, Local.cached, .{ x, positions, case.opts }, .{});
        defer cached_exe.deinit();

        const input = try zml.Slice.alloc(allocator, x.shape());
        defer input.free(allocator);
        for (input.items(Bf16), 0..) |*value, i| {
            value.* = .fromF32(@as(f32, @floatFromInt(@as(i32, @intCast(i % 37)) - 18)) / 16);
        }
        var x_buffer = try zml.Buffer.fromSlice(io, platform, input);
        defer x_buffer.deinit();
        const indices = try zml.Slice.alloc(allocator, positions.shape());
        defer indices.free(allocator);
        // Changing runtime indices include each benchmark size's final row.
        for ([_]usize{ 0, 63, 256 - case.queries, 512 - case.queries, 1024 - case.queries }) |start| {
            for (indices.items(u32), 0..) |*position, i| position.* = @intCast(start + i);
            var positions_buffer = try zml.Buffer.fromSlice(io, platform, indices);
            defer positions_buffer.deinit();
            var direct = try direct_exe.eval(allocator, io, .{ x_buffer, positions_buffer });
            defer direct.deinit();
            var cached = try cached_exe.eval(allocator, io, .{ x_buffer, positions_buffer });
            defer cached.deinit();
            errdefer std.log.err("cached RoPE mismatch: layout={t}, scaling={t}, queries={}, start={}", .{ case.opts.layout, std.meta.activeTag(case.opts.scaling), case.queries, start });
            try zml.testing.expectClose(io, direct, cached, .{
                .absolute_tolerance = 0.008,
                .relative_tolerance = 0.01,
                .minimum_close_fraction = 1.0,
            });
        }
    }
}

test "readGreedyToken: global vocabulary winners agree on every rank" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .furiosa) return error.SkipZigTest;
    const Local = struct {
        fn greedy(logits: zml.Tensor) zml.Tensor {
            const result = zml.nn.sampleTokens(logits, .{ .topk = 1 }, .init());
            return result[0].convert(.u32);
        }
    };
    const mesh = zml.testing.tpMesh();
    const logits = zml.Tensor.init(.{ .s = 1, .voc = 8 }, .bf16)
        .withPartitioning(mesh, .{ .s = .replicated, .voc = .model });
    var exe = try platform.compileFn(allocator, io, Local.greedy, .{logits}, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    const Case = struct { values: [8]f32, expected: u32 };
    for ([_]Case{
        .{ .values = .{ 9, 0, 0, 0, 1, 0, 0, 0 }, .expected = 0 },
        .{ .values = .{ 0, 0, 0, 0, 0, 9, 0, 0 }, .expected = 5 },
        .{ .values = .{ 0, 9, 0, 0, 0, 9, 0, 0 }, .expected = 1 },
    }) |case| {
        var values: [8]zml.floats.BFloat16 = undefined;
        for (&values, case.values) |*dst, value| dst.* = .fromF32(value);
        var input = try zml.Buffer.fromBytes(io, platform, logits.shape(), std.mem.sliceAsBytes(&values));
        defer input.deinit();
        var token = try exe.eval(allocator, io, .{input});
        defer token.deinit();
        try std.testing.expectEqual(case.expected, try readGreedyToken(token, io));
        const api = platform.pjrt_api;
        for (token._shards.constSlice()) |shard| {
            var actual: u32 = undefined;
            if (try shard.toHostBuffer(api, std.mem.asBytes(&actual))) |event| {
                defer event.deinit(api);
                try event.await(api, io);
            }
            try std.testing.expectEqual(case.expected, actual);
        }
    }
}

test "readGreedyToken: a later rank error precedes every download" {
    const io = std.testing.io;
    const platform = zml.testing.env();
    if (platform.target != .furiosa or platform.devices.len < 2) return error.SkipZigTest;
    const real_api = platform.pjrt_api;
    var token = try zml.Buffer.scalar(io, platform, 7, .u32);
    defer token.deinit();
    const rank1 = platform.physical_mesh.devices_in_canonical_order[1];
    const manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(real_api, .{
        .shape_specs = &.{zml.pjrt.ShapeSpec.init(&.{}, .u32)},
        .memory = platform.devices[rank1.id].memory(.default).?.pjrt_memory,
    });
    defer manager.deinit(real_api);
    try manager.setBufferError(real_api, 0, @backingInt(zml.pjrt.ErrorCode.internal), "rank-1 readiness failure");
    const failed = try manager.retrieveBuffer(real_api, 0);
    token._shards.slice()[1].deinit(real_api);
    token._shards.slice()[1] = failed;

    // Count only this token's download requests; retain the real API for all work.
    const Copy = @FieldType(@FieldType(zml.pjrt.Api, "inner"), "PJRT_Buffer_ToHostBuffer");
    const Signature = @typeInfo(@typeInfo(@typeInfo(Copy).optional.child).pointer.child).@"fn";
    const Spy = struct {
        var calls: usize = 0;
        var original: Copy = null;

        fn copy(args: Signature.param_types[0].?) callconv(.c) Signature.return_type.? {
            calls += 1;
            return original.?(args);
        }
    };
    Spy.calls = 0;
    Spy.original = real_api.inner.PJRT_Buffer_ToHostBuffer;
    var api = real_api.*;
    api.inner.PJRT_Buffer_ToHostBuffer = Spy.copy;
    var local_platform = platform.*;
    local_platform.pjrt_api = &api;
    token._platform = &local_platform;
    defer token._platform = platform;

    try std.testing.expectError(error.Internal, readGreedyToken(token, io));
    try std.testing.expectEqual(@as(usize, 0), Spy.calls);
}
