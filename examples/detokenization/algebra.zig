const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const graph = @import("graph.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const inference = @import("inference.zig");
const save_load = @import("saveload.zig");
const tokens = @import("tokens.zig");

const Zml_handler = main.Zml_handler;
const Model_handler = model_.Model_handler;


pub const SimilarityMatrix = struct {
    data: []zml.floats.BFloat16,
    data_i8: []i8,
    nearest_neighbors: []const i32,
    row_offsets: []usize,
    n: usize,
    k: usize,

    pub inline fn dist(self: *SimilarityMatrix, i: usize, j: usize) f32 {
        const row = @min(i, j);
        const col = @max(i, j);
        const index = self.row_offsets[row] + col - row;
        return self.data[index].toF32();
        //return @as(f32, @floatFromInt(self.data_i8[index])) / 127.0;
    }

    pub fn initOffsets(self: *SimilarityMatrix) void {
        var offset: usize = 0;
        for (0..self.n) |row| {
            self.row_offsets[row] = offset;
            offset += self.n - row;
        }
    }

    pub fn nearestNeighbor(self: *SimilarityMatrix, row: usize, pos: usize) usize {
        const index = row * self.k + pos;
        return @intCast(self.nearest_neighbors[index]);
    }

    pub fn showDistribution(self: *SimilarityMatrix, k: usize) void {
        std.debug.assert(k < @bitSizeOf(usize));
        const bucket_count = @as(usize, 1) << @intCast(k);
        const allocator = std.heap.page_allocator;
        const buckets = allocator.alloc(u64, bucket_count) catch @panic("OOM");
        defer allocator.free(buckets);
        @memset(buckets, 0);

        var below: u64 = 0;
        var above: u64 = 0;
        var min_value: f32 = std.math.inf(f32);
        var max_value: f32 = -std.math.inf(f32);

        for (self.data) |value_bf16| {
            const value = value_bf16.toF32();
            min_value = @min(min_value, value);
            max_value = @max(max_value, value);

            if (value < -1.0) {
                below += 1;
                continue;
            }
            if (value > 1.0) {
                above += 1;
                continue;
            }

            const scaled = (value + 1.0) * 0.5 * @as(f32, @floatFromInt(bucket_count));
            var bucket: usize = @intFromFloat(scaled);
            if (bucket == bucket_count) bucket = bucket_count - 1;
            buckets[bucket] += 1;
        }

        var max_bucket_count: u64 = 0;
        var in_range_count: u64 = 0;
        for (buckets) |count| {
            max_bucket_count = @max(max_bucket_count, count);
            in_range_count += count;
        }

        const total_count: u64 = @intCast(self.data.len);
        std.log.info("SimilarityMatrix distribution over [-1, 1]: buckets=2^{d}={d}, values={d}", .{ k, bucket_count, total_count });
        std.log.info("min={d:.6}, max={d:.6}, below=-1: {d}, above=1: {d}", .{ min_value, max_value, below, above });

        const bucket_width = 2.0 / @as(f32, @floatFromInt(bucket_count));
        const bar_width: u64 = 48;
        for (buckets, 0..) |count, bucket_i| {
            const lower = -1.0 + @as(f32, @floatFromInt(bucket_i)) * bucket_width;
            const upper = lower + bucket_width;
            const percent = if (in_range_count == 0)
                0.0
            else
                100.0 * @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(in_range_count));
            const bar_len: usize = if (max_bucket_count == 0)
                0
            else
                @intCast((count * bar_width) / max_bucket_count);

            var bar: [48]u8 = undefined;
            @memset(&bar, ' ');
            @memset(bar[0..bar_len], '#');
            std.log.info("[{d:>8.5}, {d:>8.5}) {d:>12} {d:>7.3}% |{s}|", .{ lower, upper, count, percent, bar[0..] });
        }
    }

    pub fn integerCrossover(self: *SimilarityMatrix) void {
        std.log.warn("SimilarityMatrix.integerCrossover", .{});
        std.debug.assert(self.data_i8.len == self.data.len);

        for (self.data, 0..) |value_bf16, i| {
            const value = std.math.clamp(value_bf16.toF32(), -1.0, 1.0);
            const rounded = @round(value * 127.0);
            const quantized: i32 = @intFromFloat(rounded);
            self.data_i8[i] = @intCast(quantized);
        }
    }

    pub fn deinit(self: *SimilarityMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.nearest_neighbors);
        allocator.free(self.row_offsets);
    }
};

pub const LmHeadMatrix = struct {
    n: usize,
    d: usize,
    data: []const f32,
    data_t: []const f32,
    is_junk: []const bool,
    row_norms: []const f32,

    pub fn deinit(self: *LmHeadMatrix, allocator: std.mem.Allocator) void {
        allocator.free(self.data);
        allocator.free(self.data_t);
        allocator.free(self.is_junk);
        allocator.free(self.row_norms);
    }
};


pub fn computeSimilarityMatrix(zml_handler: *Zml_handler, model_handler: *Model_handler, normalize: bool) !SimilarityMatrix {
    std.log.info("Compute similarity matrix ({s})", .{if (normalize) "normalized" else "raw"});
    const allocator = zml_handler.allocator;

    const lm_head_shape = model_handler.model.shape();
    const n: usize = @intCast(lm_head_shape.dim(.voc));
    const d: usize = @intCast(lm_head_shape.dim(.d));
    std.log.info("lm_head shape: {d} x {d}", .{ n, d });
    const triangular_len: usize = @divExact(n * (n + 1), 2);
    const batch_size: usize = @intCast(model_.Model.row_batch_size);
    const batch_count = std.math.divCeil(usize, n, batch_size) catch unreachable;
    const k: usize = @intCast(model_.Model.row_k_neighbors);

    const similarity_matrix_slice: zml.Slice = try .alloc(allocator, .init(.{ .data = triangular_len }, .bf16));
    errdefer similarity_matrix_slice.free(allocator);

    const nearest_neighbors_slice: zml.Slice = try .alloc(allocator, .init(.{ .data = n * k }, .u64));
    errdefer nearest_neighbors_slice.free(allocator);

    const similarity_batch_slice: zml.Slice = try .alloc(allocator, .init(.{ .row = batch_size, .col = n }, .bf16));
    defer similarity_batch_slice.free(allocator);

    const nearest_batch_slice: zml.Slice = try .alloc(allocator, .init(.{ .row = batch_size, .nearest = k }, .u64));
    defer nearest_batch_slice.free(allocator);

    var write_offset: usize = 0;
    var row_start: usize = 0;
    while (row_start < n) : (row_start += batch_size) {
        std.log.info("batch {d}/{d}", .{ @divExact(row_start, batch_size) + 1, batch_count });
        const rows_to_copy = @min(batch_size, n - row_start);
        const kernel_row_start = if (row_start + batch_size <= n) row_start else n - batch_size;

        var row_start_buffer = try zml.Buffer.scalar(zml_handler.io, zml_handler.platform, @as(u32, @intCast(kernel_row_start)), .u32);
        defer row_start_buffer.deinit();

        if (normalize) {
            model_handler.exes.similarity_matrix_normalized_args.set(.{ model_handler.model_buffers, row_start_buffer });
            model_handler.exes.similarity_matrix_normalized_exe.call(model_handler.exes.similarity_matrix_normalized_args, &model_handler.exes.similarity_matrix_normalized_results);
        } else {
            model_handler.exes.similarity_matrix_args.set(.{ model_handler.model_buffers, row_start_buffer });
            model_handler.exes.similarity_matrix_exe.call(model_handler.exes.similarity_matrix_args, &model_handler.exes.similarity_matrix_results);
        }
        var similarity_batch_buffer: zml.Buffer = undefined;
        var nearest_batch_buffer: zml.Buffer = undefined;
        if (normalize) {
            model_handler.exes.similarity_matrix_normalized_results.fill(.{ &similarity_batch_buffer, &nearest_batch_buffer });
        } else {
            model_handler.exes.similarity_matrix_results.fill(.{ &similarity_batch_buffer, &nearest_batch_buffer });
        }
        defer similarity_batch_buffer.deinit();
        defer nearest_batch_buffer.deinit();

        try similarity_batch_buffer.toSlice(zml_handler.io, similarity_batch_slice);
        try nearest_batch_buffer.toSlice(zml_handler.io, nearest_batch_slice);

        const batch_items = similarity_batch_slice.items(zml.floats.BFloat16);
        const output_items = similarity_matrix_slice.items(zml.floats.BFloat16);
        const nearest_batch_items = nearest_batch_slice.items(u64);
        const nearest_items = nearest_neighbors_slice.items(u64);
        for (0..rows_to_copy) |local_row| {
            const global_row = row_start + local_row;
            const kernel_local_row = global_row - kernel_row_start;
            // Copy the diagonal-included upper triangular row: cols global_row..n.
            const len = n - global_row;
            const src_start = kernel_local_row * n + global_row;
            @memcpy(output_items[write_offset..][0..len], batch_items[src_start..][0..len]);
            write_offset += len;
            @memcpy(nearest_items[global_row * k ..][0..k], nearest_batch_items[kernel_local_row * k ..][0..k]);
        }
    }
    std.debug.assert(write_offset == triangular_len);

    const matrix_data = similarity_matrix_slice.items(zml.floats.BFloat16);
    var matrix: SimilarityMatrix = .{
        .data = matrix_data,
        .data_i8 = @as([*]i8, @ptrCast(matrix_data.ptr))[0..matrix_data.len],
        .nearest_neighbors = nearest_neighbors_slice.constItems(i32),
        .row_offsets = try allocator.alloc(usize, n),
        .n = n,
        .d = d,
        .k = k,
    };
    matrix.initOffsets();
    return matrix;
}

pub fn loadSimilarityMatrix(zml_handler: *Zml_handler, normalize: bool) !SimilarityMatrix {
    const allocator = zml_handler.allocator;
    const suffix = if (normalize) "norm" else "raw";
    const matrix_filename = try std.fmt.allocPrint(allocator, "qwen_dist_mat_{s}.safetensors", .{suffix});
    defer allocator.free(matrix_filename);
    const nearest_filename = try std.fmt.allocPrint(allocator, "qwen_256_nn_{s}.safetensors", .{suffix});
    defer allocator.free(nearest_filename);

    std.log.info("Load similarity matrix from checkpoint: {s}", .{matrix_filename});
    const data = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, matrix_filename, "data");
    errdefer data.free(allocator);

    std.log.info("Load nearest neighbors from checkpoint: {s}", .{nearest_filename});
    const nearest_neighbors = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, nearest_filename, "indices");
    errdefer nearest_neighbors.free(allocator);

    std.debug.assert(nearest_neighbors.shape.rank() == 2);
    const n: usize = @intCast(nearest_neighbors.shape.dims()[0]);
    const k: usize = @intCast(nearest_neighbors.shape.dims()[1]);

    const matrix_data = data.items(zml.floats.BFloat16);
    var matrix: SimilarityMatrix = .{
        .data = matrix_data,
        .data_i8 = @as([*]i8, @ptrCast(matrix_data.ptr))[0..matrix_data.len],
        .nearest_neighbors = nearest_neighbors.constItems(i32),
        .row_offsets = try allocator.alloc(usize, n),
        .n = n,
        .k = k,
    };
    matrix.initOffsets();
    return matrix;
}


pub fn getLmHead(zml_handler: *Zml_handler) !LmHeadMatrix {
    const allocator = zml_handler.allocator;

    std.log.info("Load lm_head matrix from checkpoint: {s}", .{zml_handler.uris.qwen});
    const data = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.qwen, "model-00005-of-00005.safetensors", "lm_head.weight");
    defer data.free(allocator);

    const lm_head_shape = data.shape;
    const n: usize = @intCast(lm_head_shape.dim(0));
    const d: usize = @intCast(lm_head_shape.dim(1));
    std.debug.assert(data.shape.rank() == 2);
    const data_items_bf16 = data.constItems(zml.floats.BFloat16);

    const data_items = try allocator.alloc(f32, n * d);
    for (0..n*d) |i| {
        data_items[i] = data_items_bf16[i].toF32();
    }

    const row_norms = try allocator.alloc(f32, n);
    errdefer allocator.free(row_norms);
    for (0..n) |row_i| {
        const row = data_items[row_i * d ..][0..d];
        var norm2: f32 = 0;
        for (row) |value| {
            norm2 += value * value;
        }
        row_norms[row_i] = @sqrt(norm2);
    }

    const data_t = try allocator.alloc(f32, n * d);
    errdefer allocator.free(data_t);
    for (0..n) |row_i| {
        const row = data_items[row_i * d ..][0..d];
        for (row, 0..) |value, col_i| {
            data_t[col_i * n + row_i] = value;
        }
    }

    const row_order = try allocator.alloc(usize, n);
    defer allocator.free(row_order);
    for (row_order, 0..) |*row_i, i| row_i.* = i;
    const NormOrder = struct {
        row_norms: []const f32,

        fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
            const lhs_norm = ctx.row_norms[lhs];
            const rhs_norm = ctx.row_norms[rhs];
            return lhs_norm < rhs_norm or (lhs_norm == rhs_norm and lhs < rhs);
        }
    };
    std.mem.sort(usize, row_order, NormOrder{ .row_norms = row_norms }, NormOrder.lessThan);

    const junk_dir = try allocator.alloc(f32, d);
    defer allocator.free(junk_dir);
    @memset(junk_dir, 0);
    const junk_sample_count = 100;
    for (row_order[0..junk_sample_count]) |row_i| {
        const row = data_items[row_i * d ..][0..d];
        for (junk_dir, row) |*dst, value| {
            dst.* += value;
        }
    }
    var junk_dir_norm2: f32 = 0;
    for (junk_dir) |*value| {
        value.* /= @floatFromInt(junk_sample_count);
        junk_dir_norm2 += value.* * value.*;
    }
    const junk_dir_norm = @sqrt(junk_dir_norm2);
    for (junk_dir) |*value| {
        value.* /= junk_dir_norm;
    }

    const is_junk = try allocator.alloc(bool, n);
    errdefer allocator.free(is_junk);
    for (0..n) |row_i| {
        const row = data_items[row_i * d ..][0..d];
        var dot: f32 = 0;
        for (row, junk_dir) |row_value, dir_value| {
            dot += row_value * dir_value;
        }
        is_junk[row_i] = dot / row_norms[row_i] > 0.75;
    }

    return .{
        .n = n,
        .d = d,
        .data = data_items,
        .data_t = data_t,
        .is_junk = is_junk,
        .row_norms = row_norms,
    };
}


pub fn getLmHeadRotated(zml_handler: *Zml_handler, model_handler: *Model_handler, rotation: zml.Slice) !struct { zml.Slice, zml.Slice } {
    var rot_buffer = try zml.Buffer.fromSlice(zml_handler.io, zml_handler.platform, rotation, .replicated);
    defer rot_buffer.deinit();
    model_handler.exes.rotated_lm_head_args.set(.{ model_handler.model_buffers, rot_buffer });
    model_handler.exes.rotated_lm_head_exe.call(model_handler.exes.rotated_lm_head_args, &model_handler.exes.rotated_lm_head_results);
    var buff1: zml.Buffer = undefined;
    var buff2: zml.Buffer = undefined;
    defer buff1.deinit();
    defer buff2.deinit();
    model_handler.exes.rotated_lm_head_results.fill(.{ &buff1, &buff2 });
    return .{ try buff1.toSliceAlloc(zml_handler.allocator, zml_handler.io), try buff2.toSliceAlloc(zml_handler.allocator, zml_handler.io) };
}


pub fn getMrtOrder(zml_handler: *Zml_handler, model_handler: *Model_handler) ![]usize {
    model_handler.exes.sort_by_first_row_args.set(.{model_handler.model_buffers});
    model_handler.exes.sort_by_first_row_exe.call(model_handler.exes.sort_by_first_row_args, &model_handler.exes.sort_by_first_row_results);
    var order_buffer = model_handler.exes.sort_by_first_row_results.get(zml.Buffer);
    defer order_buffer.deinit();

    const order_slice = try order_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer order_slice.free(zml_handler.allocator);
    const order_items = order_slice.constItems(u64);
    const order = try zml_handler.allocator.alloc(usize, order_items.len);
    errdefer zml_handler.allocator.free(order);
    for (order_items, order) |src, *dst| {
        dst.* = @intCast(src);
    }
    return order;
}


pub fn loadSvd(zml_handler: *Zml_handler, model_handler: *Model_handler) !struct { zml.Slice, zml.Slice } {
    const allocator = zml_handler.allocator;
    const filename = "qwen_svd.safetensors";
    std.log.info("Load SVD from checkpoint: {s}", .{filename});

    const u = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, filename, "U");
    defer u.free(allocator);

    const diag = try save_load.loadSafetensorSlice(zml_handler, zml_handler.uris.checkpoint, filename, "diag");
    defer diag.free(allocator);

    const d: usize = @intCast(model_handler.model.shape().dim(.d));
    std.debug.assert(u.dtype() == .f32);
    std.debug.assert(diag.dtype() == .f64);
    std.debug.assert(u.shape.rank() == 1 or u.shape.rank() == 2);
    std.debug.assert(diag.shape.rank() == 1);
    std.debug.assert(u.shape.count() == d * d);
    std.debug.assert(diag.shape.count() == d);

    const reversed_u = try zml.Slice.alloc(allocator, u.shape);
    errdefer reversed_u.free(allocator);
    const reversed_diag = try zml.Slice.alloc(allocator, diag.shape);
    errdefer reversed_diag.free(allocator);

    const u_items = u.constItems(f32);
    const reversed_u_items = reversed_u.items(f32);
    for (0..d) |row| {
        for (0..d) |col| {
            reversed_u_items[row * d + col] = u_items[row * d + (d - 1 - col)];
        }
    }

    const diag_items = diag.constItems(f64);
    const reversed_diag_items = reversed_diag.items(f64);
    for (0..d) |i| {
        reversed_diag_items[i] = diag_items[d - 1 - i];
    }

    //testRotationMatrix(reversed_u);

    return .{ reversed_u, reversed_diag };
}

pub fn testRotationMatrix(matrix: zml.Slice) void {
    std.debug.assert(matrix.dtype() == .f32);
    std.debug.assert(matrix.shape.rank() == 1 or matrix.shape.rank() == 2);

    const n_float = @sqrt(@as(f64, @floatFromInt(matrix.shape.count())));
    const n: usize = @intFromFloat(n_float);
    std.debug.assert(n * n == matrix.shape.count());

    std.log.info("Testing rotation matrix, flops needed: {d}", .{n * n * n});

    const items = matrix.constItems(f32);
    var max_diag_error: f32 = 0;
    var max_off_diag_error: f32 = 0;
    for (0..n) |row| {
        for (0..n) |col| {
            var dot: f32 = 0;
            for (0..n) |k| {
                dot += items[row * n + k] * items[col * n + k];
            }
            if (row == col) {
                max_diag_error = @max(max_diag_error, @abs(dot - 1));
            } else {
                max_off_diag_error = @max(max_off_diag_error, @abs(dot));
            }
        }
    }

    std.log.info("Rotation matrix test: max_diag_error={d}, max_off_diag_error={d}", .{ max_diag_error, max_off_diag_error });
}

pub fn printTopRowsSection(tokenizer: zml.tokenizer.Tokenizer, title: []const u8, value_label: []const u8, indices: []const u64, values: []const f32, row_norms: []const f32) !void {
    std.debug.assert(indices.len == values.len);
    std.log.info("{s}", .{title});
    if (std.mem.eql(u8, value_label, "row_norm")) {
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s}", .{ "rank", "token_id", value_label, "token" });
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s}", .{ "------", "----------", "--------------", "-----" });
    } else {
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s}", .{ "rank", "token_id", "row_norm", value_label, "token" });
        std.log.info("{s:>6}  {s:>10}  {s:>14}  {s:>14}  {s}", .{ "------", "----------", "--------------", "--------------", "-----" });
    }
    for (indices, values, 0..) |token_id, value, i| {
        const token_index: usize = @intCast(token_id);
        std.debug.assert(token_index < row_norms.len);
        var decoded_buf: [512]u8 = undefined;
        const decoded = try tokens.decodeToken(tokenizer, @intCast(token_id), &decoded_buf);
        var escaped_buf: [512]u8 = undefined;
        const escaped = tokens.escapeTokenText(decoded, &escaped_buf);
        if (std.mem.eql(u8, value_label, "row_norm")) {
            std.log.info("{d:>6}  {d:>10}  {d:>14.6}  {s}", .{ i + 1, token_id, value, escaped });
        } else {
            std.log.info("{d:>6}  {d:>10}  {d:>14.6}  {d:>14.6}  {s}", .{ i + 1, token_id, row_norms[token_index], value, escaped });
        }
    }
}
