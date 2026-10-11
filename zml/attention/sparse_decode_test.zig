const std = @import("std");

const zml = @import("zml");
const paged = zml.attention.triton.paged;

fn sparseDecode(p: paged.Parameters, q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, indices: zml.Tensor) zml.Tensor {
    return zml.attention.paged_attention.pagedAttention(
        .{ .triton = p },
        q.convert(.bf16),
        k.convert(.bf16),
        v.convert(.bf16),
        .{ .split = .{ .k = k.convert(.bf16), .v = v.convert(.bf16) } },
        .{ .mask = .{ .indices = indices } },
    ).convert(.f32);
}

test "sparse GQA decode matches host softmax with holes and empty segments" {
    const platform = zml.testing.env();
    if (platform.target != .cuda) return error.SkipZigTest;
    const cc = zml.platform.cuda.computeCapability(platform) orelse return error.SkipZigTest;
    if (!cc.eql(.{ .major = 10, .minor = 3 })) return error.SkipZigTest;
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    for ([_]usize{ 8, 24, 32 }) |heads| {
        const groups = 2;
        const hd = 256;
        const pages = 96;
        const page_size = 64;
        const batch = 2;
        const q_shape = zml.Shape.init(.{ .b = batch, .hkv = groups, .hg = heads / groups, .hd = hd }, .f32);
        const kv_shape = zml.Shape.init(.{ .page = pages, .k_chunk = page_size, .hkv = groups, .hd = hd }, .f32);
        const q_data = try allocator.alloc(f32, q_shape.count());
        defer allocator.free(q_data);
        const k_data = try allocator.alloc(f32, kv_shape.count());
        defer allocator.free(k_data);
        const v_data = try allocator.alloc(f32, kv_shape.count());
        defer allocator.free(v_data);
        var rng = std.Random.DefaultPrng.init(1729);
        for ([_][]f32{ q_data, k_data, v_data }) |data| {
            for (data) |*value| value.* = @as(f32, @floatFromInt(rng.random().intRangeAtMost(i32, -16, 16))) / 16.0;
        }
        // Very negative logits must survive empty leading splits without a
        // spurious zero maximum swallowing the softmax mass.
        for (0..batch) |b| @memset(q_data[b * heads * hd ..][0..hd], -4);
        for (0..pages * page_size) |slot| @memset(k_data[slot * groups * hd ..][0..hd], 4);
        var q_d = try zml.Buffer.fromBytes(io, platform, q_shape, std.mem.sliceAsBytes(q_data));
        defer q_d.deinit();
        var k_d = try zml.Buffer.fromBytes(io, platform, kv_shape, std.mem.sliceAsBytes(k_data));
        defer k_d.deinit();
        var v_d = try zml.Buffer.fromBytes(io, platform, kv_shape, std.mem.sliceAsBytes(v_data));
        defer v_d.deinit();
        const parameters = paged.Parameters.init(.{ .batch_size = batch, .max_num_pages = pages, .max_seqlen_q = 256, .is_prefill = false });
        var table: [batch][pages]i32 = undefined;
        for (&table, 0..) |*row, b| {
            for (row, 0..) |*page, i| page.* = @intCast((pages - 1 - i + b * 13) % pages);
            row[7] = -1;
        }
        const lengths = [_]i32{ 4099, 43 };
        const starts = [_]i32{ 0, 1, 2 };
        var p_d: zml.Bufferized(paged.Parameters) = .{
            .block_table = try .fromBytes(io, platform, parameters.block_table.shape(), std.mem.sliceAsBytes(&table)),
            .seq_lens = try .fromBytes(io, platform, parameters.seq_lens.shape(), std.mem.sliceAsBytes(&lengths)),
            .query_start_len = try .fromBytes(io, platform, parameters.query_start_len.shape(), std.mem.sliceAsBytes(&starts)),
        };
        defer zml.Buffer.deinitAll(paged.Parameters, &p_d);
        for ([_]usize{ 33, 2051, 2052 }) |count| {
            const index_shape = zml.Shape.init(.{ .b = batch, .topk = count }, .i32);
            const indices = try allocator.alloc(i32, batch * count);
            defer allocator.free(indices);
            for (indices, 0..) |*position, i| {
                const j = i % count;
                position.* = if (j % 5 == 0 or j >= count - @min(count / 2, 32)) -1 else @intCast((j * 7) % (pages * page_size));
            }
            // Keep a valid tail after padding and duplicates in distinct splits.
            indices[count - 1] = 4098;
            indices[count - 2] = 0;
            indices[1] = 0;
            var exe = try platform.compileFn(allocator, io, sparseDecode, .{ parameters, zml.Tensor.init(q_shape, .f32), zml.Tensor.init(kv_shape, .f32), zml.Tensor.init(kv_shape, .f32), zml.Tensor.init(index_shape, .i32) }, .{});
            defer exe.deinit();
            // Cover a zero normalizer and an inactive sequence before the active one.
            for (0..3) |empty_row| {
                if (empty_row == 1) @memset(indices[count..], -1);
                var indices_d = try zml.Buffer.fromBytes(io, platform, index_shape, std.mem.sliceAsBytes(indices));
                defer indices_d.deinit();
                const case_starts: [3]i32 = if (empty_row == 2) .{ 0, 0, 1 } else .{ 0, 1, 2 };
                var starts_d = try zml.Buffer.fromBytes(io, platform, parameters.query_start_len.shape(), std.mem.sliceAsBytes(&case_starts));
                defer starts_d.deinit();
                var case_p_d = p_d;
                case_p_d.query_start_len = starts_d;
                var output_d = try exe.eval(allocator, io, .{ case_p_d, q_d, k_d, v_d, indices_d });
                defer output_d.deinit();
                const expected = try allocator.alloc(f32, q_shape.count());
                defer allocator.free(expected);
                @memset(expected, 0);
                const scores = try allocator.alloc(f64, count);
                defer allocator.free(scores);
                for (0..batch) |b| {
                    if (case_starts[b] == case_starts[b + 1]) continue;
                    const query: usize = @intCast(case_starts[b]);
                    for (0..heads) |h| {
                        var denominator: f64 = 0;
                        for (indices[query * count ..][0..count], 0..) |position, j| {
                            scores[j] = 0;
                            if (position < 0 or position >= lengths[b]) continue;
                            const pos: usize = @intCast(position);
                            if (table[b][pos / page_size] < 0) continue;
                            const physical: usize = @intCast(table[b][pos / page_size]);
                            const kv_start = ((physical * page_size + pos % page_size) * groups + h / (heads / groups)) * hd;
                            var dot: f64 = 0;
                            for (0..hd) |d| dot += @as(f64, q_data[(query * heads + h) * hd + d]) * k_data[kv_start + d];
                            scores[j] = @exp(dot / @sqrt(@as(f64, hd)));
                            denominator += scores[j];
                        }
                        if (denominator == 0) continue;
                        for (indices[query * count ..][0..count], 0..) |position, j| {
                            if (scores[j] == 0) continue;
                            const pos: usize = @intCast(position);
                            const physical: usize = @intCast(table[b][pos / page_size]);
                            const kv_start = ((physical * page_size + pos % page_size) * groups + h / (heads / groups)) * hd;
                            for (0..hd) |d| expected[(query * heads + h) * hd + d] += @floatCast(scores[j] / denominator * v_data[kv_start + d]);
                        }
                    }
                }
                const active: usize = @intCast(case_starts[batch]);
                var output = try output_d.toSliceAlloc(allocator, io);
                defer output.free(allocator);
                try zml.testing.expectClose(io, zml.Slice.init(q_shape.setDim(.b, @intCast(active)), std.mem.sliceAsBytes(expected[0 .. active * heads * hd])), output.subSlice(0, 0, @intCast(active)), .{ .absolute_tolerance = 0.004, .relative_tolerance = 0.02 });
            }
        }
    }
}
