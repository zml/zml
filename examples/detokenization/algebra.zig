const std = @import("std");
const zml = @import("zml");

const main = @import("main.zig");
const graph = @import("graph.zig");
const model_ = @import("model.zig");
const llm_ = @import("llm.zig");
const svd_ = @import("svd.zig");
const inference = @import("inference.zig");
const save_load = @import("saveload.zig");
const tokens = @import("tokens.zig");

const Zml_handler = main.Zml_handler;
const Model_handler = model_.Model_handler;

pub const SimilarityMatrix = struct {
    data: zml.Slice,
    data_f32: []const f32,
    nearest_neighbors: zml.Slice,
    row_offsets: []usize,
    n: usize,
    d: usize,
    k: usize,

    pub inline fn dist(self: *SimilarityMatrix, i: usize, j: usize) f32 {
        const row = @min(i, j);
        const col = @max(i, j);
        const index = self.row_offsets[row] + col - row;
        return self.data_f32[index];
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
        return @intCast(self.nearest_neighbors.constItems(i32)[index]);
    }

    pub fn deinit(self: *SimilarityMatrix, allocator: std.mem.Allocator) void {
        self.data.free(allocator);
        self.nearest_neighbors.free(allocator);
        allocator.free(self.row_offsets);
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

    const similarity_matrix_slice: zml.Slice = try .alloc(allocator, .init(.{ .data = triangular_len }, .f32));
    errdefer similarity_matrix_slice.free(allocator);

    const nearest_neighbors_slice: zml.Slice = try .alloc(allocator, .init(.{ .data = n * k }, .u64));
    errdefer nearest_neighbors_slice.free(allocator);

    const similarity_batch_slice: zml.Slice = try .alloc(allocator, .init(.{ .row = batch_size, .col = n }, .f32));
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

        const batch_items = similarity_batch_slice.items(f32);
        const output_items = similarity_matrix_slice.items(f32);
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

    var matrix: SimilarityMatrix = .{
        .data = similarity_matrix_slice,
        .data_f32 = similarity_matrix_slice.constItems(f32),
        .nearest_neighbors = nearest_neighbors_slice,
        .row_offsets = try allocator.alloc(usize, n),
        .n = n,
        .d = d,
        .k = k,
    };
    matrix.initOffsets();
    return matrix;
}

pub fn loadSimilarityMatrix(zml_handler: *Zml_handler, model_handler: *Model_handler, normalize: bool) !SimilarityMatrix {
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

    const lm_head_shape = model_handler.model.shape();
    const n: usize = @intCast(lm_head_shape.dim(.voc));
    const d: usize = @intCast(lm_head_shape.dim(.d));
    std.debug.assert(nearest_neighbors.shape.rank() == 2);
    std.debug.assert(@as(usize, @intCast(nearest_neighbors.shape.dims()[0])) == n);
    const k: usize = @intCast(nearest_neighbors.shape.dims()[1]);

    const expected_data_len = @divExact(n * (n + 1), 2);
    std.debug.assert(data.shape.count() == expected_data_len);

    var matrix: SimilarityMatrix = .{
        .data = data,
        .data_f32 = data.constItems(f32),
        .nearest_neighbors = nearest_neighbors,
        .row_offsets = try allocator.alloc(usize, n),
        .n = n,
        .d = d,
        .k = k,
    };
    matrix.initOffsets();
    return matrix;
}

const MatrixCandidate = struct {
    node: usize,
    similarity: f32,

    fn beforeThan(_: void, lhs: MatrixCandidate, rhs: MatrixCandidate) bool {
        return lhs.similarity > rhs.similarity or (lhs.similarity == rhs.similarity and lhs.node < rhs.node);
    }
};

pub fn testSimilarityMatrix(zml_handler: *Zml_handler, model_handler: *Model_handler, similarity_matrix: *SimilarityMatrix, normalized_rows: bool) !void {
    std.log.info("Test similarity matrix", .{});

    const lm_head, const lm_head_translation = try getLmHead(zml_handler, model_handler);
    defer lm_head.free(zml_handler.allocator);
    defer lm_head_translation.free(zml_handler.allocator);

    const n: usize = @intCast(lm_head.shape.dim(.voc));
    const d: usize = @intCast(lm_head.shape.dim(.d));
    std.debug.assert(n == similarity_matrix.n);
    std.debug.assert(d == similarity_matrix.d);

    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();
    const lm_head_items = lm_head.items(f32);

    var max_abs_diff: f32 = 0;
    var max_rel_diff: f32 = 0;
    for (0..1000) |_| {
        const i = random.uintLessThan(usize, n);
        const j = random.uintLessThan(usize, n);
        const u = lm_head_items[i * d ..][0..d];
        const v = lm_head_items[j * d ..][0..d];
        var dot: f32 = 0;
        var u_norm2: f32 = 0;
        var v_norm2: f32 = 0;
        for (u, v) |u_value, v_value| {
            dot += u_value * v_value;
            u_norm2 += u_value * u_value;
            v_norm2 += v_value * v_value;
        }
        if (normalized_rows) {
            dot /= (@sqrt(u_norm2) * @sqrt(v_norm2));
        }
        const expected = dot;
        const actual = similarity_matrix.dist(i, j);
        const abs_diff = @abs(expected - actual);
        const rel_diff = abs_diff / @max(@abs(expected), 1e-6);
        max_abs_diff = @max(max_abs_diff, abs_diff);
        if (rel_diff > max_rel_diff) {
            max_rel_diff = rel_diff;
            std.log.info("New max relative diff rows=({d},{d}) expected={d} actual={d} abs_diff={d} rel_diff={d}", .{
                i,
                j,
                expected,
                actual,
                abs_diff,
                rel_diff,
            });
        }
    }

    std.log.info("1000 random pairs, max_abs_diff={d}, max_rel_diff={d}", .{ max_abs_diff, max_rel_diff });
    std.log.info("Test kNN", .{});

    const candidates = try zml_handler.allocator.alloc(MatrixCandidate, n - 1);
    defer zml_handler.allocator.free(candidates);
    var knn_mismatches: usize = 0;
    for (0..100) |_| {
        const row = random.uintLessThan(usize, n);
        var nb_candidates: usize = 0;
        for (0..n) |col| {
            if (col == row) continue;
            candidates[nb_candidates] = .{ .node = col, .similarity = similarity_matrix.dist(row, col) };
            nb_candidates += 1;
        }
        std.mem.sort(MatrixCandidate, candidates[0..nb_candidates], {}, MatrixCandidate.beforeThan);
        for (0..similarity_matrix.k) |pos| {
            const expected = candidates[pos].node;
            const actual = similarity_matrix.nearestNeighbor(row, pos);
            if (actual != expected) {
                const err = @abs(candidates[pos].similarity - similarity_matrix.dist(row, actual));
                std.log.err("kNN mismatch row={d} pos={d} expected={d} actual={d} expected_sim={d} sim_err={d}", .{
                    row,
                    pos,
                    expected,
                    actual,
                    candidates[pos].similarity,
                    err,
                });
                if (err > 0) knn_mismatches += 1;
            }
        }
    }
    if (knn_mismatches != 0) return error.KnnMismatch;
    std.log.info("Similarity matrix kNN test passed: 100 random rows, k={d}", .{similarity_matrix.k});
}

pub fn getLmHead(zml_handler: *Zml_handler, model_handler: *Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_exe.call(model_handler.exes.get_lm_head_args, &model_handler.exes.get_lm_head_results);
    var lm_head_buffer: zml.Buffer = undefined;
    model_handler.exes.get_lm_head_results.fill(.{ &lm_head_buffer });
    defer lm_head_buffer.deinit();
    const lm_head = try lm_head_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    errdefer lm_head.free(zml_handler.allocator);
    return lm_head;
}

pub fn getLmHeadTransposed(zml_handler: *Zml_handler, model_handler: *Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_transposed_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_transposed_exe.call(model_handler.exes.get_lm_head_transposed_args, &model_handler.exes.get_lm_head_transposed_results);
    var lm_head_transposed_buffer: zml.Buffer = undefined;
    model_handler.exes.get_lm_head_transposed_results.fill(.{ &lm_head_transposed_buffer });
    defer lm_head_transposed_buffer.deinit();
    const lm_head_transposed = try lm_head_transposed_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    errdefer lm_head_transposed.free(zml_handler.allocator);
    return lm_head_transposed;
}

pub fn getLmHeadNormalized(zml_handler: *Zml_handler, model_handler: *Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_normalized_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_normalized_exe.call(model_handler.exes.get_lm_head_normalized_args, &model_handler.exes.get_lm_head_normalized_results);
    var lm_head_normalized_buffer: zml.Buffer = undefined;
    model_handler.exes.get_lm_head_normalized_results.fill(.{ &lm_head_normalized_buffer });
    defer lm_head_normalized_buffer.deinit();
    const lm_head_normalized = try lm_head_normalized_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    errdefer lm_head_normalized.free(zml_handler.allocator);
    return lm_head_normalized;
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

pub fn getLmHeadRowNorms(zml_handler: *Zml_handler, model_handler: *Model_handler) !zml.Slice {
    model_handler.exes.get_lm_head_row_norms_args.set(.{model_handler.model_buffers});
    model_handler.exes.get_lm_head_row_norms_exe.call(model_handler.exes.get_lm_head_row_norms_args, &model_handler.exes.get_lm_head_row_norms_results);
    var lm_head_row_norms_buffer = model_handler.exes.get_lm_head_row_norms_results.get(zml.Buffer);
    defer lm_head_row_norms_buffer.deinit();
    return lm_head_row_norms_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
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

pub fn getMedoid(zml_handler: *Zml_handler, model_handler: *Model_handler, junk_rows: []const usize) !usize {
    const n: usize = @intCast(model_handler.model.shape().dim(.voc));
    const sentinel: u64 = @intCast(n);
    const junk_rows_slice = try zml.Slice.alloc(zml_handler.allocator, .init(.{ .junk = n }, .u64));
    defer junk_rows_slice.free(zml_handler.allocator);
    @memset(junk_rows_slice.items(u64), sentinel);
    for (junk_rows, 0..) |row, i| {
        junk_rows_slice.items(u64)[i] = @intCast(row);
    }

    var junk_rows_buffer = try zml.Buffer.fromSlice(zml_handler.io, zml_handler.platform, junk_rows_slice, .replicated);
    defer junk_rows_buffer.deinit();

    model_handler.exes.get_medoid_args.set(.{ model_handler.model_buffers, junk_rows_buffer });
    model_handler.exes.get_medoid_exe.call(model_handler.exes.get_medoid_args, &model_handler.exes.get_medoid_results);
    var medoid_buffer = model_handler.exes.get_medoid_results.get(zml.Buffer);
    defer medoid_buffer.deinit();

    const medoid_slice = try medoid_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer medoid_slice.free(zml_handler.allocator);
    return @intCast(medoid_slice.constItems(u64)[0]);
}

pub fn getJunkRows(zml_handler: *Zml_handler, model_handler: *Model_handler) ![]usize {
    model_handler.exes.find_junk_rows_args.set(.{model_handler.model_buffers});
    model_handler.exes.find_junk_rows_exe.call(model_handler.exes.find_junk_rows_args, &model_handler.exes.find_junk_rows_results);
    var junk_rows_buffer = model_handler.exes.find_junk_rows_results.get(zml.Buffer);
    defer junk_rows_buffer.deinit();

    var junk_rows_slice = try junk_rows_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer junk_rows_slice.free(zml_handler.allocator);

    const sentinel: u64 = @intCast(model_handler.model.shape().dim(.voc));
    var junk_rows: std.ArrayList(usize) = try .initCapacity(zml_handler.allocator, 0);
    errdefer junk_rows.deinit(zml_handler.allocator);
    for (junk_rows_slice.constItems(u64)) |row| {
        if (row == sentinel) continue;
        try junk_rows.append(zml_handler.allocator, @intCast(row));
    }
    return junk_rows.toOwnedSlice(zml_handler.allocator);
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

pub fn analyzeTopRows(zml_handler: *Zml_handler, model_handler: *Model_handler) !void {
    std.log.info("Analyze top lm_head rows", .{});

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    model_handler.exes.analyze_top_rows_args.set(.{model_handler.model_buffers});
    model_handler.exes.analyze_top_rows_exe.call(model_handler.exes.analyze_top_rows_args, &model_handler.exes.analyze_top_rows_results);

    var top_norm_values_buffer: zml.Buffer = undefined;
    var top_norm_indices_buffer: zml.Buffer = undefined;
    var top_dot_values_buffer: zml.Buffer = undefined;
    var top_dot_indices_buffer: zml.Buffer = undefined;
    var top_avg_dot_values_buffer: zml.Buffer = undefined;
    var top_avg_dot_indices_buffer: zml.Buffer = undefined;
    var anti_junk_values_buffer: zml.Buffer = undefined;
    var anti_junk_indices_buffer: zml.Buffer = undefined;
    model_handler.exes.analyze_top_rows_results.fill(.{
        &top_norm_values_buffer,
        &top_norm_indices_buffer,
        &top_dot_values_buffer,
        &top_dot_indices_buffer,
        &top_avg_dot_values_buffer,
        &top_avg_dot_indices_buffer,
        &anti_junk_values_buffer,
        &anti_junk_indices_buffer,
    });
    defer top_norm_values_buffer.deinit();
    defer top_norm_indices_buffer.deinit();
    defer top_dot_values_buffer.deinit();
    defer top_dot_indices_buffer.deinit();
    defer top_avg_dot_values_buffer.deinit();
    defer top_avg_dot_indices_buffer.deinit();
    defer anti_junk_values_buffer.deinit();
    defer anti_junk_indices_buffer.deinit();

    const top_norm_values_slice = try top_norm_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_norm_values_slice.free(zml_handler.allocator);
    const top_norm_indices_slice = try top_norm_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_norm_indices_slice.free(zml_handler.allocator);
    const top_dot_values_slice = try top_dot_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_dot_values_slice.free(zml_handler.allocator);
    const top_dot_indices_slice = try top_dot_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_dot_indices_slice.free(zml_handler.allocator);
    const top_avg_dot_values_slice = try top_avg_dot_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_avg_dot_values_slice.free(zml_handler.allocator);
    const top_avg_dot_indices_slice = try top_avg_dot_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer top_avg_dot_indices_slice.free(zml_handler.allocator);
    const anti_junk_values_slice = try anti_junk_values_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer anti_junk_values_slice.free(zml_handler.allocator);
    const anti_junk_indices_slice = try anti_junk_indices_buffer.toSliceAlloc(zml_handler.allocator, zml_handler.io);
    defer anti_junk_indices_slice.free(zml_handler.allocator);

    const lm_head_row_norms_slice = try getLmHeadRowNorms(zml_handler, model_handler);
    defer lm_head_row_norms_slice.free(zml_handler.allocator);
    const lm_head_row_norms = lm_head_row_norms_slice.constItems(f32);

    try printTopRowsSection(
        tokenizer,
        "Top 100 highest-norm lm_head rows",
        "row_norm",
        top_norm_indices_slice.constItems(u64),
        top_norm_values_slice.constItems(f32),
        lm_head_row_norms,
    );
    try printTopRowsSection(
        tokenizer,
        "Top 100 dot products with the highest-norm row",
        "dot",
        top_dot_indices_slice.constItems(u64),
        top_dot_values_slice.constItems(f32),
        lm_head_row_norms,
    );
    try printTopRowsSection(
        tokenizer,
        "Top 100 dot products with the average of top 100 highest-norm rows",
        "dot",
        top_avg_dot_indices_slice.constItems(u64),
        top_avg_dot_values_slice.constItems(f32),
        lm_head_row_norms,
    );
    try printTopRowsSection(
        tokenizer,
        "Top 100 smallest dot products with the junk direction",
        "junk_dot",
        anti_junk_indices_slice.constItems(u64),
        anti_junk_values_slice.constItems(f32),
        lm_head_row_norms,
    );
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
