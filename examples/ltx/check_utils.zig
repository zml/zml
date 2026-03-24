const std = @import("std");
const zml = @import("zml");

/// Diagnostic stage for stage-by-stage parity validation.
pub const DiagnosticStage = enum {
    q_norm,
    q_head_split,
    k_head_split,
    v_head_split,
    q_rotated,
    k_rotated,
    gate_logits,
    pre_to_out,
    attention_output,
};

/// Parity metrics for a single diagnostic stage.
pub const StageMetrics = struct {
    stage: DiagnosticStage,
    max_abs_error: f64,
    mean_abs_error: f64,
    close_fraction: f64,
    passed: bool,
};

pub const CompareMetrics = struct {
    max_abs_error: f64,
    mean_abs_error: f64,
    close_fraction: f64,
};

pub const ExtendedCompareMetrics = struct {
    max_abs_error: f64,
    mean_abs_error: f64,
    rmse_error: f64,
    rel_l2_error: f64,
    cosine_similarity: f64,
    close_fraction: f64,
    abs_err_p50: f64,
    abs_err_p90: f64,
    abs_err_p99: f64,
    abs_err_p999: f64,
    positive_diff_fraction: f64,
    negative_diff_fraction: f64,
    zero_diff_fraction: f64,
    frac_abs_err_gt_1e3: f64,
    frac_abs_err_gt_1e2: f64,
    frac_abs_err_gt_1e1: f64,
};

/// Compares two buffers element-wise and produces metrics.
pub fn compareBuffers(
    io: std.Io,
    computed: zml.Buffer,
    expected: zml.Buffer,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) !CompareMetrics {
    var computed_slice = try computed.toSliceAlloc(std.heap.smp_allocator, io);
    defer computed_slice.free(std.heap.smp_allocator);

    var expected_slice = try expected.toSliceAlloc(std.heap.smp_allocator, io);
    defer expected_slice.free(std.heap.smp_allocator);

    const computed_bytes = computed_slice.data();
    const expected_bytes = expected_slice.data();

    if (computed.shape().byteSize() != expected.shape().byteSize()) {
        return error.ShapeMismatch;
    }

    const dtype = computed.shape().dtype();
    if (dtype != expected.shape().dtype()) {
        return error.DtypeMismatch;
    }

    const elem_size = dtype.sizeOf();
    const num_elems = computed.shape().byteSize() / elem_size;

    var max_abs_err: f64 = 0.0;
    var sum_abs_err: f64 = 0.0;
    var close_count: u64 = 0;

    var i: usize = 0;
    while (i < num_elems) : (i += 1) {
        const comp_ptr = computed_bytes.ptr + i * elem_size;
        const exp_ptr = expected_bytes.ptr + i * elem_size;

        const comp_val = switch (dtype) {
            .f32 => @as(f64, @floatCast(@as(*align(1) const f32, @ptrCast(comp_ptr)).*)),
            .f64 => @as(f64, @as(*align(1) const f64, @ptrCast(comp_ptr)).*),
            .bf16 => bf16_to_f32(@as(*align(1) const u16, @ptrCast(comp_ptr)).*),
            else => return error.UnsupportedDtype,
        };

        const exp_val = switch (dtype) {
            .f32 => @as(f64, @floatCast(@as(*align(1) const f32, @ptrCast(exp_ptr)).*)),
            .f64 => @as(f64, @as(*align(1) const f64, @ptrCast(exp_ptr)).*),
            .bf16 => bf16_to_f32(@as(*align(1) const u16, @ptrCast(exp_ptr)).*),
            else => return error.UnsupportedDtype,
        };

        const abs_err = @abs(comp_val - exp_val);
        max_abs_err = @max(max_abs_err, abs_err);
        sum_abs_err += abs_err;

        const rel_err = if (exp_val != 0.0) abs_err / @abs(exp_val) else abs_err;
        if (abs_err <= absolute_tolerance or rel_err <= relative_tolerance) {
            close_count += 1;
        }
    }

    const mean_abs_err = sum_abs_err / @as(f64, @floatFromInt(num_elems));
    const close_fraction = @as(f64, @floatFromInt(close_count)) / @as(f64, @floatFromInt(num_elems));

    return .{
        .max_abs_error = max_abs_err,
        .mean_abs_error = mean_abs_err,
        .close_fraction = close_fraction,
    };
}

fn quantileFromSorted(data: []const f64, q: f64) f64 {
    if (data.len == 0) return 0.0;
    const qq = std.math.clamp(q, 0.0, 1.0);
    const last_idx: f64 = @floatFromInt(data.len - 1);
    const idx_f = qq * last_idx;
    const idx_low: usize = @intFromFloat(@floor(idx_f));
    const idx_high: usize = @intFromFloat(@ceil(idx_f));
    if (idx_low == idx_high) return data[idx_low];

    const w = idx_f - @as(f64, @floatFromInt(idx_low));
    return data[idx_low] * (1.0 - w) + data[idx_high] * w;
}

/// Compares two buffers with distribution-aware diagnostics to disambiguate
/// global rounding drift from sparse outliers.
pub fn compareBuffersExtended(
    io: std.Io,
    computed: zml.Buffer,
    expected: zml.Buffer,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) !ExtendedCompareMetrics {
    var computed_slice = try computed.toSliceAlloc(std.heap.smp_allocator, io);
    defer computed_slice.free(std.heap.smp_allocator);

    var expected_slice = try expected.toSliceAlloc(std.heap.smp_allocator, io);
    defer expected_slice.free(std.heap.smp_allocator);

    const computed_bytes = computed_slice.data();
    const expected_bytes = expected_slice.data();

    if (computed.shape().byteSize() != expected.shape().byteSize()) {
        return error.ShapeMismatch;
    }

    const dtype = computed.shape().dtype();
    if (dtype != expected.shape().dtype()) {
        return error.DtypeMismatch;
    }

    const elem_size = dtype.sizeOf();
    const num_elems = computed.shape().byteSize() / elem_size;
    if (num_elems == 0) return error.EmptyBuffer;

    var abs_errors = try std.heap.smp_allocator.alloc(f64, num_elems);
    defer std.heap.smp_allocator.free(abs_errors);

    var max_abs_err: f64 = 0.0;
    var sum_abs_err: f64 = 0.0;
    var sum_sq_err: f64 = 0.0;
    var sum_sq_exp: f64 = 0.0;
    var sum_sq_comp: f64 = 0.0;
    var dot_comp_exp: f64 = 0.0;

    var close_count: u64 = 0;
    var pos_count: u64 = 0;
    var neg_count: u64 = 0;
    var zero_count: u64 = 0;
    var gt_1e3_count: u64 = 0;
    var gt_1e2_count: u64 = 0;
    var gt_1e1_count: u64 = 0;

    var i: usize = 0;
    while (i < num_elems) : (i += 1) {
        const comp_ptr = computed_bytes.ptr + i * elem_size;
        const exp_ptr = expected_bytes.ptr + i * elem_size;

        const comp_val = switch (dtype) {
            .f32 => @as(f64, @floatCast(@as(*align(1) const f32, @ptrCast(comp_ptr)).*)),
            .f64 => @as(f64, @as(*align(1) const f64, @ptrCast(comp_ptr)).*),
            .bf16 => bf16_to_f32(@as(*align(1) const u16, @ptrCast(comp_ptr)).*),
            else => return error.UnsupportedDtype,
        };

        const exp_val = switch (dtype) {
            .f32 => @as(f64, @floatCast(@as(*align(1) const f32, @ptrCast(exp_ptr)).*)),
            .f64 => @as(f64, @as(*align(1) const f64, @ptrCast(exp_ptr)).*),
            .bf16 => bf16_to_f32(@as(*align(1) const u16, @ptrCast(exp_ptr)).*),
            else => return error.UnsupportedDtype,
        };

        const diff = comp_val - exp_val;
        const abs_err = @abs(diff);
        abs_errors[i] = abs_err;

        max_abs_err = @max(max_abs_err, abs_err);
        sum_abs_err += abs_err;
        sum_sq_err += diff * diff;
        sum_sq_exp += exp_val * exp_val;
        sum_sq_comp += comp_val * comp_val;
        dot_comp_exp += comp_val * exp_val;

        if (diff > 0.0) {
            pos_count += 1;
        } else if (diff < 0.0) {
            neg_count += 1;
        } else {
            zero_count += 1;
        }

        if (abs_err > 1e-3) gt_1e3_count += 1;
        if (abs_err > 1e-2) gt_1e2_count += 1;
        if (abs_err > 1e-1) gt_1e1_count += 1;

        const rel_err = if (exp_val != 0.0) abs_err / @abs(exp_val) else abs_err;
        if (abs_err <= absolute_tolerance or rel_err <= relative_tolerance) {
            close_count += 1;
        }
    }

    std.mem.sort(f64, abs_errors, {}, std.sort.asc(f64));

    const n: f64 = @floatFromInt(num_elems);
    const mean_abs_err = sum_abs_err / n;
    const close_fraction = @as(f64, @floatFromInt(close_count)) / n;
    const rmse_error = @sqrt(sum_sq_err / n);
    const rel_l2_error = if (sum_sq_exp > 0.0) @sqrt(sum_sq_err) / @sqrt(sum_sq_exp) else 0.0;

    const denom = @sqrt(sum_sq_comp) * @sqrt(sum_sq_exp);
    const cosine_similarity = if (denom > 0.0) dot_comp_exp / denom else 1.0;

    return .{
        .max_abs_error = max_abs_err,
        .mean_abs_error = mean_abs_err,
        .rmse_error = rmse_error,
        .rel_l2_error = rel_l2_error,
        .cosine_similarity = cosine_similarity,
        .close_fraction = close_fraction,
        .abs_err_p50 = quantileFromSorted(abs_errors, 0.50),
        .abs_err_p90 = quantileFromSorted(abs_errors, 0.90),
        .abs_err_p99 = quantileFromSorted(abs_errors, 0.99),
        .abs_err_p999 = quantileFromSorted(abs_errors, 0.999),
        .positive_diff_fraction = @as(f64, @floatFromInt(pos_count)) / n,
        .negative_diff_fraction = @as(f64, @floatFromInt(neg_count)) / n,
        .zero_diff_fraction = @as(f64, @floatFromInt(zero_count)) / n,
        .frac_abs_err_gt_1e3 = @as(f64, @floatFromInt(gt_1e3_count)) / n,
        .frac_abs_err_gt_1e2 = @as(f64, @floatFromInt(gt_1e2_count)) / n,
        .frac_abs_err_gt_1e1 = @as(f64, @floatFromInt(gt_1e1_count)) / n,
    };
}

pub fn copyBuffer(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);
    return zml.Buffer.fromSlice(io, platform, src_slice, sharding);
}

/// Converts bfloat16 to f32.
fn bf16_to_f32(bf16_val: u16) f64 {
    const f32_bits: u32 = @as(u32, bf16_val) << 16;
    const f32_val = @as(f32, @bitCast(f32_bits));
    return @as(f64, @floatCast(f32_val));
}

pub fn sliceTokenPrefix(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    token_limit: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 3) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const t: usize = @intCast(dims[1]);
    const d: usize = @intCast(dims[2]);
    const out_t = @min(token_limit, t);
    const elem_size: usize = src.shape().dtype().sizeOf();

    const out_shape = zml.Shape.init(.{ dims[0], @as(i64, @intCast(out_t)), dims[2] }, src.shape().dtype());
    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();
    const src_batch_stride = t * d * elem_size;
    const dst_batch_stride = out_t * d * elem_size;
    const per_batch_copy = out_t * d * elem_size;

    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        const src_batch_off = bi * src_batch_stride;
        const dst_batch_off = bi * dst_batch_stride;
        std.mem.copyForwards(u8, out_bytes[dst_batch_off .. dst_batch_off + per_batch_copy], src_bytes[src_batch_off .. src_batch_off + per_batch_copy]);
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}

pub fn sliceTokenPrefixBHTD(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    token_limit: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 4) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const h: usize = @intCast(dims[1]);
    const t: usize = @intCast(dims[2]);
    const d: usize = @intCast(dims[3]);
    const out_t = @min(token_limit, t);
    const elem_size: usize = src.shape().dtype().sizeOf();

    const out_shape = zml.Shape.init(.{ dims[0], dims[1], @as(i64, @intCast(out_t)), dims[3] }, src.shape().dtype());
    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();

    const src_b_stride = h * t * d * elem_size;
    const src_h_stride = t * d * elem_size;
    const dst_b_stride = h * out_t * d * elem_size;
    const dst_h_stride = out_t * d * elem_size;
    const per_head_copy = out_t * d * elem_size;

    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        var hi: usize = 0;
        while (hi < h) : (hi += 1) {
            const src_off = bi * src_b_stride + hi * src_h_stride;
            const dst_off = bi * dst_b_stride + hi * dst_h_stride;
            std.mem.copyForwards(u8, out_bytes[dst_off .. dst_off + per_head_copy], src_bytes[src_off .. src_off + per_head_copy]);
        }
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}

/// Slices a [B, T, H, HD] buffer along dimension 1 (token dim) to at most token_limit tokens.
/// Use this for diagnostic reference tensors saved in ZML-native [B, T, H, HD] layout.
pub fn sliceTokenPrefixBTHD(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    token_limit: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 4) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const t: usize = @intCast(dims[1]);
    const h: usize = @intCast(dims[2]);
    const hd: usize = @intCast(dims[3]);
    const out_t = @min(token_limit, t);
    const elem_size: usize = src.shape().dtype().sizeOf();

    const out_shape = zml.Shape.init(.{ dims[0], @as(i64, @intCast(out_t)), dims[2], dims[3] }, src.shape().dtype());

    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();

    // Each batch row is [T, H, HD]; copy first out_t * H * HD elements per batch
    const src_b_stride = t * h * hd * elem_size;
    const row_size = out_t * h * hd * elem_size;

    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        std.mem.copyForwards(u8, out_bytes[bi * row_size .. bi * row_size + row_size], src_bytes[bi * src_b_stride .. bi * src_b_stride + row_size]);
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}

/// Slices a 4D tensor along its token axis, accepting either [B, T, H, HD] or [B, H, T, HD].
/// Heuristic: token axis is usually the larger of dims[1] and dims[2].
pub fn sliceTokenPrefixBTHDorBHTD(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    token_limit: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 4) return error.UnexpectedRank;

    const dims = shape.dims();
    const dim1: usize = @intCast(dims[1]);
    const dim2: usize = @intCast(dims[2]);

    if (dim1 >= dim2) {
        return sliceTokenPrefixBTHD(io, platform, src, sharding, token_limit);
    }
    return sliceTokenPrefixBHTD(io, platform, src, sharding, token_limit);
}

/// Transpose [B, H, T, HD] -> [B, T, H, HD].
pub fn transposeBHTDToBTHD(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 4) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const h: usize = @intCast(dims[1]);
    const t: usize = @intCast(dims[2]);
    const hd: usize = @intCast(dims[3]);
    const elem_size: usize = src.shape().dtype().sizeOf();

    const out_shape = zml.Shape.init(.{ dims[0], dims[2], dims[1], dims[3] }, src.shape().dtype());

    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();

    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        var hi: usize = 0;
        while (hi < h) : (hi += 1) {
            var ti: usize = 0;
            while (ti < t) : (ti += 1) {
                const src_off = (((bi * h + hi) * t + ti) * hd) * elem_size;
                const dst_off = (((bi * t + ti) * h + hi) * hd) * elem_size;
                std.mem.copyForwards(
                    u8,
                    out_bytes[dst_off .. dst_off + hd * elem_size],
                    src_bytes[src_off .. src_off + hd * elem_size],
                );
            }
        }
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}

/// Compare buffers, accepting expected in either [B,T,H,HD] or [B,H,T,HD] layout.
pub fn compareBuffersBTHDCompatible(
    io: std.Io,
    platform: *zml.Platform,
    computed: zml.Buffer,
    expected: zml.Buffer,
    sharding: zml.sharding.Sharding,
    absolute_tolerance: f32,
    relative_tolerance: f32,
) !CompareMetrics {
    const cshape = computed.shape();
    const eshape = expected.shape();

    if (cshape.rank() == 4 and eshape.rank() == 4) {
        const cd = cshape.dims();
        const ed = eshape.dims();

        // If expected is [B,H,T,HD] while computed is [B,T,H,HD], transpose expected.
        if (cd[0] == ed[0] and cd[1] == ed[2] and cd[2] == ed[1] and cd[3] == ed[3]) {
            var expected_bthd = try transposeBHTDToBTHD(io, platform, expected, sharding);
            defer expected_bthd.deinit();
            return compareBuffers(io, computed, expected_bthd, absolute_tolerance, relative_tolerance);
        }
    }

    return compareBuffers(io, computed, expected, absolute_tolerance, relative_tolerance);
}

pub fn loadBufferFromStore(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    key: []const u8,
    sharding: zml.sharding.Sharding,
) !zml.Buffer {
    const shape = store.view().getShape(key) orelse {
        std.log.err("Tensor not found in fixture: {s}", .{key});
        return error.NotFound;
    };

    const host_bytes = try allocator.alloc(u8, shape.byteSize());
    defer allocator.free(host_bytes);

    var io_buffer: [8 * 1024]u8 = undefined;
    var reader = try store.view().getReader(key, io, &io_buffer);
    defer reader.deinit();

    _ = try reader.interface.readSliceAll(host_bytes);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, host_bytes);
}

/// Applies head-split transformation: [B, T, D] -> [B, H, T, HD]
/// This is used to match the head-split format for diagnostic comparison.
pub fn applyHeadSplit(
    io: std.Io,
    platform: *zml.Platform,
    src: zml.Buffer,
    sharding: zml.sharding.Sharding,
    num_heads: usize,
) !zml.Buffer {
    const shape = src.shape();
    if (shape.rank() != 3) return error.UnexpectedRank;

    const dims = shape.dims();
    const b: usize = @intCast(dims[0]);
    const t: usize = @intCast(dims[1]);
    const d: usize = @intCast(dims[2]);

    if (d % num_heads != 0) return error.InvalidHeadDimension;
    const hd = d / num_heads;

    const elem_size = shape.dtype().sizeOf();

    // Output shape: [B, H, T, HD]
    const out_shape = zml.Shape.init(.{ @as(i64, @intCast(b)), @as(i64, @intCast(num_heads)), @as(i64, @intCast(t)), @as(i64, @intCast(hd)) }, shape.dtype());
    
    var src_slice = try src.toSliceAlloc(std.heap.smp_allocator, io);
    defer src_slice.free(std.heap.smp_allocator);

    var out_slice = try zml.Slice.alloc(std.heap.smp_allocator, out_shape);
    defer out_slice.free(std.heap.smp_allocator);

    const src_bytes = src_slice.constData();
    const out_bytes = out_slice.data();

    // [B, T, D] -> [B, H, T, HD]
    // For each batch and token, split D into H chunks of size HD
    var bi: usize = 0;
    while (bi < b) : (bi += 1) {
        var ti: usize = 0;
        while (ti < t) : (ti += 1) {
            var hi: usize = 0;
            while (hi < num_heads) : (hi += 1) {
                const src_off = (bi * t + ti) * d * elem_size + hi * hd * elem_size;
                const dst_off = ((bi * num_heads + hi) * t + ti) * hd * elem_size;
                std.mem.copyForwards(u8, out_bytes[dst_off .. dst_off + hd * elem_size], src_bytes[src_off .. src_off + hd * elem_size]);
            }
        }
    }

    return zml.Buffer.fromSlice(io, platform, out_slice, sharding);
}

/// Diagnostic reference: stores q, k, v split tensors and rotated versions from ground truth.
pub const DiagnosticReference = struct {
    q_head_split: ?zml.Buffer = null,
    k_head_split: ?zml.Buffer = null,
    v_head_split: ?zml.Buffer = null,
    q_rotated: ?zml.Buffer = null,
    k_rotated: ?zml.Buffer = null,

    pub fn deinit(self: *DiagnosticReference) void {
        if (self.q_head_split) |*b| b.deinit();
        if (self.k_head_split) |*b| b.deinit();
        if (self.v_head_split) |*b| b.deinit();
        if (self.q_rotated) |*b| b.deinit();
        if (self.k_rotated) |*b| b.deinit();
    }
};

/// Loads diagnostic reference tensors from the reference file if available.
/// Returns empty struct if reference file is not available or tensors are missing.
pub fn loadDiagnosticReference(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    store: *zml.io.TensorStore,
    mode_name: []const u8,
    replicated_sharding: zml.sharding.Sharding,
) !DiagnosticReference {
    var ref: DiagnosticReference = .{};

    // Try to load each diagnostic tensor; silently skip if not found
    const keys = .{
        "q_head_split",
        "k_head_split",
        "v_head_split",
        "q_rotated",
        "k_rotated",
    };

    inline for (keys) |key_suffix| {
        const full_key = try std.fmt.allocPrint(allocator, "{s}.{s}", .{ mode_name, key_suffix });
        defer allocator.free(full_key);

        if (store.view().hasKey(full_key)) {
            const buf = try loadBufferFromStore(allocator, io, platform, store, full_key, replicated_sharding);
            if (comptime std.mem.eql(u8, key_suffix, "q_head_split")) {
                ref.q_head_split = buf;
            } else if (comptime std.mem.eql(u8, key_suffix, "k_head_split")) {
                ref.k_head_split = buf;
            } else if (comptime std.mem.eql(u8, key_suffix, "v_head_split")) {
                ref.v_head_split = buf;
            } else if (comptime std.mem.eql(u8, key_suffix, "q_rotated")) {
                ref.q_rotated = buf;
            } else if (comptime std.mem.eql(u8, key_suffix, "k_rotated")) {
                ref.k_rotated = buf;
            }
        }
    }

    return ref;
}

/// Logs stage metrics and returns whether it passed tolerance threshold.
pub fn reportStageMetrics(
    stage: DiagnosticStage,
    metrics: anytype,
    minimum_close_fraction: f32,
) bool {
    const passed = metrics.close_fraction >= minimum_close_fraction;
    const status = if (passed) "PASS" else "FAIL";
    
    std.log.info(
        "{s} stage: max_abs_error={d:.4}, mean_abs_error={d:.4}, close_fraction={d:.4} [{s}]",
        .{ @tagName(stage), metrics.max_abs_error, metrics.mean_abs_error, metrics.close_fraction, status },
    );
    
    return passed;
}
