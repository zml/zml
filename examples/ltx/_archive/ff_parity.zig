const std = @import("std");
const zml = @import("zml");

const Metrics = struct {
    max_abs_error: f64,
    mean_abs_error: f64,
    rel_l2_error: f64,
    cosine_similarity: f64,
};

const Thresholds = struct {
    max_abs_error: f64 = 0.20,
    mean_abs_error: f64 = 0.005,
    rel_l2_error: f64 = 0.01,
    cosine_similarity_min: f64 = 0.999,
};

pub fn main(init: std.process.Init) !void {
    const allocator = init.gpa;
    const io = init.io;

    var it = init.minimal.args.iterate();
    _ = it.next(); // executable name

    const fixture_path = it.next() orelse {
        std.log.err("Usage: bazel run //examples/ltx:ff_parity -- <fixture.safetensors> [candidate.safetensors]", .{});
        return error.InvalidArgs;
    };
    const candidate_path = it.next() orelse fixture_path;

    var fixture = try zml.safetensors.TensorRegistry.fromPath(allocator, io, fixture_path);
    defer fixture.deinit();

    var candidate = if (std.mem.eql(u8, fixture_path, candidate_path))
        fixture
    else
        try zml.safetensors.TensorRegistry.fromPath(allocator, io, candidate_path);
    defer if (!std.mem.eql(u8, fixture_path, candidate_path)) candidate.deinit();

    const expected = try loadTensorF32(allocator, io, &fixture, "ff.output0");
    defer allocator.free(expected);

    const predicted = try loadTensorF32(allocator, io, &candidate, "ff.output0");
    defer allocator.free(predicted);

    if (expected.len != predicted.len) {
        std.log.err("Length mismatch expected={d} predicted={d}", .{ expected.len, predicted.len });
        return error.ShapeMismatch;
    }

    const metrics = computeMetrics(expected, predicted);
    printMetrics(metrics);

    const th: Thresholds = .{};
    const ok = metrics.max_abs_error <= th.max_abs_error and
        metrics.mean_abs_error <= th.mean_abs_error and
        metrics.rel_l2_error <= th.rel_l2_error and
        metrics.cosine_similarity >= th.cosine_similarity_min;

    if (!ok) {
        std.log.err("Parity FAILED against thresholds: max_abs<= {d:.6}, mean_abs<= {d:.6}, rel_l2<= {d:.6}, cosine>= {d:.6}", .{ th.max_abs_error, th.mean_abs_error, th.rel_l2_error, th.cosine_similarity_min });
        return error.ParityFailed;
    }

    std.log.info("Parity PASSED", .{});
}

fn loadTensorF32(
    allocator: std.mem.Allocator,
    io: std.Io,
    registry: *zml.safetensors.TensorRegistry,
    key: []const u8,
) ![]f32 {
    const floats = zml.floats;
    const t = registry.tensors.get(key) orelse {
        std.log.err("Tensor key not found: {s}", .{key});
        return error.TensorNotFound;
    };

    const is_bf16 = t.shape.dtype() == .bf16;
    const is_f32 = t.shape.dtype() == .f32;
    if (!is_bf16 and !is_f32) {
        std.log.err("Tensor {s} must be bf16 or f32, got {s}", .{ key, t.shape.dtype().str() });
        return error.InvalidDType;
    }

    const count = t.shape.count();
    var values = try allocator.alloc(f32, count);
    errdefer allocator.free(values);

    const read_buf = try allocator.alloc(u8, 64 * 1024);
    defer allocator.free(read_buf);

    var reader = try registry.reader(io, key, read_buf);
    defer reader.deinit();

    if (is_f32) {
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(values));
    } else {
        const bf16_buf = try allocator.alloc(floats.BFloat16, count);
        defer allocator.free(bf16_buf);
        try reader.interface.readSliceAll(std.mem.sliceAsBytes(bf16_buf));
        for (bf16_buf, values[0..]) |bf, *f| {
            f.* = bf.toF32();
        }
    }

    return values;
}

fn computeMetrics(expected: []const f32, predicted: []const f32) Metrics {
    var max_abs: f64 = 0.0;
    var sum_abs: f64 = 0.0;

    var sum_sq_ref: f64 = 0.0;
    var sum_sq_err: f64 = 0.0;
    var dot: f64 = 0.0;
    var sum_sq_pred: f64 = 0.0;

    for (expected, predicted) |a_f32, b_f32| {
        const a = @as(f64, a_f32);
        const b = @as(f64, b_f32);

        const err = b - a;
        const abs_err = @abs(err);
        if (abs_err > max_abs) max_abs = abs_err;
        sum_abs += abs_err;

        sum_sq_ref += a * a;
        sum_sq_pred += b * b;
        sum_sq_err += err * err;
        dot += a * b;
    }

    const n = @as(f64, @floatFromInt(expected.len));
    const mean_abs = if (n > 0.0) sum_abs / n else 0.0;
    const rel_l2 = if (sum_sq_ref > 0.0) @sqrt(sum_sq_err) / @sqrt(sum_sq_ref) else 0.0;

    const denom = @sqrt(sum_sq_ref) * @sqrt(sum_sq_pred);
    const cosine = if (denom > 0.0) dot / denom else 1.0;

    return .{
        .max_abs_error = max_abs,
        .mean_abs_error = mean_abs,
        .rel_l2_error = rel_l2,
        .cosine_similarity = cosine,
    };
}

fn printMetrics(m: Metrics) void {
    std.log.info("max_abs_error={d:.8}", .{m.max_abs_error});
    std.log.info("mean_abs_error={d:.8}", .{m.mean_abs_error});
    std.log.info("rel_l2_error={d:.8}", .{m.rel_l2_error});
    std.log.info("cosine_similarity={d:.8}", .{m.cosine_similarity});
}
