//! Run provenance and independently readable case results.
const std = @import("std");
const builtin = @import("builtin");
const stamp = @import("bazel/stamp").stamp;
const zml = @import("zml");
const cases = @import("cases.zig");

pub const Source = struct {
    revision: []const u8,
    dirty: bool,
    source_hash: []const u8,
    zig_version: []const u8,
    optimization: std.builtin.OptimizeMode,
};

pub fn source() Source {
    return .{
        .revision = stamp("STABLE_MOE_BENCH_REVISION"),
        .dirty = std.mem.eql(u8, stamp("STABLE_MOE_BENCH_DIRTY"), "true"),
        .source_hash = stamp("STABLE_MOE_BENCH_SOURCE_HASH"),
        .zig_version = builtin.zig_version_string,
        .optimization = builtin.mode,
    };
}

pub const Status = enum {
    listed,
    skipped,
    measured,
    passed,
    compilation_failed,
    execution_failed,

    pub fn failed(status: Status) bool {
        return switch (status) {
            .listed, .skipped, .measured, .passed => false,
            else => true,
        };
    }
};

pub const Difference = struct {
    /// Flat output index, including all previously compared reference rows.
    index: usize,
    // Null represents a nonfinite operand, keeping JSON valid.
    actual: ?f64,
    expected: ?f64,
    allowed_error: ?f64,
};

pub const Comparison = struct {
    exact: usize = 0,
    elements: usize = 0,
    outside_tolerance: usize = 0,
    nonfinite: usize = 0,
    relative_l2_error: ?f64 = null,
    max_absolute_error: f64 = 0,
    max_absolute_error_index: ?usize = null,
    first_failure: ?Difference = null,
    sum_squared_error: f64 = 0,
    sum_squared_actual: f64 = 0,
    sum_squared_reference: f64 = 0,
};

pub const Timing = struct {
    samples_us: []const f64,
    iterations_per_sample: u32,
    median_us: f64,
    minimum_us: f64,
    maximum_us: f64,
    mad_us: f64,

    pub fn summarize(allocator: std.mem.Allocator, samples: []const f64, iterations: u32) !Timing {
        std.debug.assert(samples.len != 0);
        const sorted = try allocator.dupe(f64, samples);
        defer allocator.free(sorted);
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
        const median = middle(sorted);
        const minimum = sorted[0];
        const maximum = sorted[sorted.len - 1];
        for (sorted) |*s| s.* = @abs(s.* - median);
        std.mem.sort(f64, sorted, {}, std.sort.asc(f64));
        return .{
            .samples_us = samples,
            .iterations_per_sample = iterations,
            .median_us = median,
            .minimum_us = minimum,
            .maximum_us = maximum,
            .mad_us = middle(sorted),
        };
    }

    fn middle(sorted: []const f64) f64 {
        const i = sorted.len / 2;
        return if (sorted.len % 2 == 0) (sorted[i - 1] + sorted[i]) / 2 else sorted[i];
    }
};

pub const Statistics = struct {
    timing: Timing,
    local_routes: i64,
    touched_experts: i64,
    maximum_routes: i64,
    padded_rows: [5]i64,
    ranks: []const @import("workload.zig").RankStatistics,
    implied_tflops: f64,
    implied_weight_gbs: f64,
    output_mean_absolute: f64,
    nonfinite: usize,
};

pub const DeviceDetails = struct {
    name: []const u8,
    id: u32,
    local_hardware_id: i32,
    core_count: ?i64,
    compute_capability: ?[]const u8,
};

pub fn deviceDetails(allocator: std.mem.Allocator, platform: *const zml.Platform) ![]DeviceDetails {
    const result = try allocator.alloc(DeviceDetails, platform.devices.len);
    for (platform.devices, result) |device, *details| {
        const cores = device.pjrt_desc.attribute(platform.pjrt_api, "core_count");
        const capability = device.pjrt_desc.attribute(platform.pjrt_api, "compute_capability");
        details.* = .{
            .name = device.kind(),
            .id = device.id(),
            .local_hardware_id = device.localHardwareId(),
            .core_count = if (cores) |value| if (value == .int64) value.int64 else null else null,
            .compute_capability = if (capability) |value| if (value == .string) value.string else null else null,
        };
    }
    return result;
}

const EnvironmentEntry = struct { name: []const u8, value: []const u8 };

/// Only record settings relevant to reproducibility, never the whole environment.
pub fn environment(allocator: std.mem.Allocator, environ: *const std.process.Environ.Map) ![]EnvironmentEntry {
    var result: std.ArrayList(EnvironmentEntry) = .empty;
    var it = environ.iterator();
    while (it.next()) |entry| {
        const name = entry.key_ptr.*;
        if (std.mem.startsWith(u8, name, "ZML_MOE_") or
            std.mem.eql(u8, name, "CUDA_VISIBLE_DEVICES") or
            std.mem.eql(u8, name, "XLA_FLAGS") or
            std.mem.eql(u8, name, "SKIP_PJRT_PROFILER") or
            std.mem.eql(u8, name, "ZML_AUTOTUNE_CACHE_DIR"))
        {
            try result.append(allocator, .{ .name = name, .value = entry.value_ptr.* });
        }
    }
    return result.toOwnedSlice(allocator);
}

pub const Record = struct {
    kind: enum { case } = .case,
    id: []const u8,
    graph_id: []const u8,
    workload: cases.Workload,
    status: Status,
    reason: ?[]const u8 = null,
    // Recorded once for the graph; other runtime variants share graph_id.
    compile_ns: ?i64 = null,
    reference_compile_ns: ?i64 = null,
    cute_launch: ?zml.moe.cute.kernels.LaunchPlan = null,
    statistics: ?Statistics = null,
    reference_check: ?Comparison = null,
    replay_comparison: ?Comparison = null,
    dump_comparison: ?Comparison = null,
    dump_source: ?Source = null,
};

pub fn writeJsonLine(writer: *std.Io.Writer, value: anytype) !void {
    var stringify: std.json.Stringify = .{ .writer = writer };
    try stringify.write(value);
    try writer.writeByte('\n');
    try writer.flush();
}

pub fn report(stdout: *std.Io.Writer, json: ?*std.Io.Writer, record: Record) !void {
    try stdout.print("{s} [{s}]  {s}\n", .{ @tagName(record.status), record.workload.preset, record.id });
    if (record.statistics) |stats| {
        const t = stats.timing;
        try stdout.print("  median {d:.3} us/call; MAD {d:.3}; range {d:.3}..{d:.3}; {d} samples x {d} calls\n", .{ t.median_us, t.mad_us, t.minimum_us, t.maximum_us, t.samples_us.len, t.iterations_per_sample });
        try stdout.print("  routes {d}, touched {d}/{d}, max {d}; implied {d:.2} TFLOP/s, {d:.2} GB/s; mean |output| {d:.6}\n", .{ stats.local_routes, stats.touched_experts, record.workload.case.experts, stats.maximum_routes, stats.implied_tflops, stats.implied_weight_gbs, stats.output_mean_absolute });
    }
    if (record.reason) |reason| try stdout.print("  {s}\n", .{reason});
    if (record.reference_check) |c| {
        try stdout.print("  reference: exact {d}/{d}, max |diff| {d:.6}, outside tolerance {d}\n", .{ c.exact, c.elements, c.max_absolute_error, c.outside_tolerance });
        try reportDifference(stdout, record.workload.case.width, c);
    }
    if (record.dump_comparison) |c| {
        try stdout.print("  baseline: exact {d}/{d}, max |diff| {d:.6}, outside tolerance {d}\n", .{ c.exact, c.elements, c.max_absolute_error, c.outside_tolerance });
        try reportDifference(stdout, record.workload.case.width, c);
    }
    if (record.replay_comparison) |c| if (record.status.failed()) {
        try stdout.print("  replay: exact {d}/{d}, max |diff| {d:.6}, outside tolerance {d}\n", .{ c.exact, c.elements, c.max_absolute_error, c.outside_tolerance });
        try reportDifference(stdout, record.workload.case.width, c);
    };
    try stdout.flush();
    if (json) |writer| try writeJsonLine(writer, record);
}

fn reportDifference(writer: *std.Io.Writer, width: i64, comparison: Comparison) !void {
    if (comparison.first_failure) |d| {
        const columns: usize = @intCast(width);
        try writer.print("  first failure: token {d}, column {d}, actual {?d}, expected {?d}, allowed |diff| {?d}\n", .{
            d.index / columns, d.index % columns, d.actual, d.expected, d.allowed_error,
        });
    }
}

test "timing summary preserves samples and reports median absolute deviation" {
    const samples = [_]f64{ 100, 4, 2, 3, 1 };
    const t = try Timing.summarize(std.testing.allocator, &samples, 20);
    try std.testing.expectEqual(@as(f64, 3), t.median_us);
    try std.testing.expectEqual(@as(f64, 1), t.mad_us);
    try std.testing.expectEqual(@as(f64, 1), t.minimum_us);
    try std.testing.expectEqual(@as(f64, 100), t.maximum_us);
    try std.testing.expectEqualSlices(f64, &.{ 100, 4, 2, 3, 1 }, t.samples_us);
}
