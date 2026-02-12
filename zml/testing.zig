const std = @import("std");
const builtin = @import("builtin");

const stdx = @import("stdx");

const Slice = @import("slice.zig").Slice;
const Platform = @import("platform.zig").Platform;
const zml = @import("zml.zig");

const log = std.log.scoped(.@"zml/testing");

var _platform: ?*const Platform = null;

pub fn env() *const Platform {
    if (!builtin.is_test) @compileError("Cannot use zml.testing.env outside of a test block");
    if (_platform == null) {
        _platform = Platform.auto(std.heap.c_allocator, std.testing.io, .{
            .cuda = .{ .allocator = .{ .bfc = .{ .preallocate = true, .memory_fraction = 0.85 } } },
        }) catch unreachable;
    }

    return _platform.?;
}

/// In neural network we generally care about the relative precision,
/// but on a given dimension, if the output is close to 0, then the precision
/// don't matter as much.
pub fn approxEq(comptime Float: type, l: Float, r: Float, tolerance: Float) bool {
    const closeRel = std.math.approxEqRel(Float, l, r, @floatCast(tolerance));
    const closeAbs = std.math.approxEqAbs(Float, l, r, @floatCast(tolerance / 2));
    return closeRel or closeAbs;
}

fn toSlice(allocator: std.mem.Allocator, io: std.Io, slice_or_buffer: anytype) !struct { Slice, bool } {
    return if (@TypeOf(slice_or_buffer) == zml.Buffer) b: {
        const slice = try slice_or_buffer.toSliceAlloc(allocator, io);
        break :b .{ slice, true };
    } else .{ slice_or_buffer, false };
}

/// Testing utility. Accepts both zml.Buffer and zml.HostBuffer but zml.Buffer will be copied to the
/// host for comparison !
pub fn expectClose(io: std.Io, left_: anytype, right_: anytype, opts: CompareOpts) !void {
    const allocator = if (builtin.is_test) std.testing.allocator else std.heap.smp_allocator;
    var left: Slice, const should_free_left = try toSlice(allocator, io, left_);
    defer if (should_free_left) left.free(allocator);

    var right: Slice, const should_free_right = try toSlice(allocator, io, right_);
    defer if (should_free_right) right.free(allocator);

    const stderr = std.debug.lockStderr(&.{});
    defer std.debug.unlockStderr();
    const w = &stderr.file_writer.interface;

    //errdefer log.err("\n--> Left: {0d:24.3}\n--> Right: {1d:24.3}", .{ left, right });
    if (!std.mem.eql(i64, left.shape.dims(), right.shape.dims())) {
        try w.print("left.shape() {f} != right.shape() {f}\n", .{ left.shape, right.shape });
        return error.TestUnexpectedResult;
    }

    if (left.dtype() != right.dtype()) {
        try w.print("left.dtype ({}) != right.dtype ({})\n", .{ left.dtype(), right.dtype() });
        return error.TestUnexpectedResult;
    }

    switch (left.dtype()) {
        inline .bf16,
        .f16,
        .f32,
        .f64,
        .f4e2m1,
        .f8e3m4,
        .f8e4m3,
        .f8e4m3b11fnuz,
        .f8e4m3fn,
        .f8e4m3fnuz,
        .f8e5m2,
        .f8e5m2fnuz,
        .f8e8m0,
        => |t| {
            const L = t.toZigType();
            const left_data = left.constItems(L);
            switch (right.dtype()) {
                inline .bf16,
                .f16,
                .f32,
                .f64,
                .f8e4m3b11fnuz,
                .f8e4m3fn,
                .f8e4m3fnuz,
                .f8e5m2,
                .f8e5m2fnuz,
                => |rt| {
                    const R = rt.toZigType();
                    const right_data = right.constItems(R);
                    const compare_report = try compareSlices(allocator, L, R, left_data, right_data, opts);
                    const ok = !compare_report.nan_or_inf and compare_report.close_fraction >= opts.minimum_close_fraction;
                    if (ok) {
                        return;
                    } else {
                        try w.print("{f}\n", .{compare_report});
                        return error.TestUnexpectedResult;
                    }
                },
                else => unreachable,
            }
        },
        inline .bool, .u2, .u4, .u8, .u16, .u32, .u64, .i2, .i4, .i8, .i16, .i32, .i64 => |t| {
            const T = t.toZigType();
            return std.testing.expectEqualSlices(T, left.constItems(T), right.constItems(T));
        },
        .c64, .c128 => @panic("TODO: support comparison of complex"),
    }
}

pub const CompareOpts = struct {
    absolute_tolerance: f32 = 1e-3,
    relative_tolerance: f32 = 1e-2,
    epsilon_relative: f32 = 1e-6,
    minimum_close_fraction: f32 = 0.999,
};

pub const CompareReport = struct {
    nan_or_inf: bool,
    max_absolute_error: f32,
    mean_absolute_error: f32,
    rmse: f32,

    close_fraction: f32,

    p50_absolute_error: f32,
    p90_absolute_error: f32,
    p99_absolute_error: f32,
    p999_absolute_error: f32,

    pub fn format(
        self: @This(),
        writer: *std.Io.Writer,
    ) std.Io.Writer.Error!void {
        try writer.print(
            \\    nan_or_inf: {}
            \\    max_absolute_error: {d}
            \\    mean_absolute_error: {d}
            \\    rmse: {d}
            \\    close_fraction: {d}
            \\    p50_absolute_error: {d}
            \\    p90_absolute_error: {d}
            \\    p99_absolute_error: {d}
            \\    p999_absolute_error: {d}
        , .{
            self.nan_or_inf,
            self.max_absolute_error,
            self.mean_absolute_error,
            self.rmse,
            self.close_fraction,
            self.p50_absolute_error,
            self.p90_absolute_error,
            self.p99_absolute_error,
            self.p999_absolute_error,
        });
    }
};

pub fn compareSlices(allocator: std.mem.Allocator, comptime L: type, comptime R: type, left: []const L, right: []const R, opts: CompareOpts) !CompareReport {
    var nan_or_inf: bool = false;
    var max_absolute_error: f32 = 0;
    var sum_absolute_error: f64 = 0;
    var sum_squared_error: f64 = 0;
    var count_close: usize = 0;
    var absolute_errors = try allocator.alloc(f32, left.len);
    defer allocator.free(absolute_errors);
    var relative_errors = try allocator.alloc(f32, left.len);
    defer allocator.free(relative_errors);

    for (left, right, 0..) |l, r, i| {
        const l_f32 = zml.floats.floatCast(f32, l);
        const r_f32 = zml.floats.floatCast(f32, r);
        if (!std.math.isFinite(l_f32) or !std.math.isFinite(r_f32)) {
            nan_or_inf = true;
            continue;
        }
        const absolute_error = @abs(l_f32 - r_f32);
        max_absolute_error = @max(max_absolute_error, absolute_error);
        sum_absolute_error += @as(f64, @floatCast(absolute_error));
        sum_squared_error += @as(f64, @floatCast(absolute_error)) * @as(f64, @floatCast(absolute_error));

        const scale = @max(@abs(l_f32), @abs(r_f32));
        const tolerance = opts.absolute_tolerance + opts.relative_tolerance * scale;
        if (absolute_error <= tolerance) {
            count_close += 1;
        }

        absolute_errors[i] = absolute_error;
        const denom = @max(opts.epsilon_relative, scale);
        relative_errors[i] = absolute_error / denom;
    }

    std.sort.heap(f32, absolute_errors, {}, std.sort.asc(f32));
    std.sort.heap(f32, relative_errors, {}, std.sort.asc(f32));

    const q = struct {
        pub fn q(values: []const f32, frac: f32) f32 {
            if (values.len == 0) return 0;
            const idx: usize = @intFromFloat(std.math.round(@as(f32, @floatFromInt(values.len - 1)) * frac));
            return values[idx];
        }
    }.q;

    const mean_absolute_error: f32 = @floatCast(sum_absolute_error / @as(f64, @floatFromInt(left.len)));
    const rmse: f32 = @floatCast(std.math.sqrt(sum_squared_error / @as(f64, @floatFromInt(left.len))));
    const close_fraction = @as(f32, @floatFromInt(count_close)) / @as(f32, @floatFromInt(left.len));

    const report: CompareReport = .{
        .nan_or_inf = nan_or_inf,
        .max_absolute_error = max_absolute_error,
        .mean_absolute_error = mean_absolute_error,
        .rmse = rmse,
        .close_fraction = close_fraction,
        .p50_absolute_error = q(absolute_errors, 0.5),
        .p90_absolute_error = q(absolute_errors, 0.9),
        .p99_absolute_error = q(absolute_errors, 0.99),
        .p999_absolute_error = q(absolute_errors, 0.999),
    };

    // TODO: Move
    //const ok = !report.nan_or_inf and report.close_fraction >= opts.minimum_close_fraction;
    return report;
}

pub fn expectEqualShapes(expected: zml.Shape, actual: zml.Shape) error{TestExpectedEqual}!void {
    if (expected.eqlWithTags(actual)) return;

    std.debug.print("Expected {f}, got {f}", .{ expected, actual });
    return error.TestExpectedEqual;
}

fn BufferizedWithArgs(comptime T: type) type {
    return zml.meta.MapType(zml.Tensor, zml.Buffer).map(T);
}

/// Automatically calls the executable with the given arguments, taking care of arguments and results allocation.
/// This helper can be used in tests to make the code a bit less verbose.
/// It doesn't handle pointers in inputs/outputs structs.
pub fn autoCall(allocator: std.mem.Allocator, io: std.Io, exe: *const zml.exe.Exe, func: anytype, inputs: zml.Bufferized(stdx.meta.FnArgs(func))) !zml.Bufferized(stdx.meta.FnResult(func)) {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);

    var results = try exe.results(allocator);
    defer results.deinit(allocator);

    args.set(inputs);
    exe.callOpts(io, args, &results, .{ .wait = true });

    var output: zml.Bufferized(stdx.meta.FnResult(func)) = undefined;
    results.fill(.{&output});

    return output;
}

fn countWithPrefix(store: zml.io.TensorStore.View, prefix: []const u8) usize {
    var count: usize = 0;
    var it = store.store.registry.iterator();
    while (it.next()) |entry| {
        if (std.mem.startsWith(u8, entry.key_ptr.*, prefix)) {
            count += 1;
        }
    }

    return count;
}

/// Test a layer implementation against reference activations stored in a TensorStore.
/// The layer is expected to have a method with the name given by `func`.
/// The reference activations are expected to be stored under the keys:
/// - `{name}.in.0`, `{name}.in.1`, ... for inputs
/// - `{name}.out.0`, `{name}.out.1`, ... for outputs
///
/// The layer weights are expected to be provided in `layer_weights`.
/// The comparison tolerance can be adjusted with `tolerance`.
///
/// The function arguments need to be valid when initialized to undefined, meaning no slices,
/// no unions, no pointers.
pub fn testLayer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    layer: anytype,
    comptime func: std.meta.DeclEnum(@TypeOf(layer)),
    activation_store: zml.io.TensorStore.View,
    comptime name: []const u8,
    layer_weights: zml.Bufferized(@TypeOf(layer)),
    opts: CompareOpts,
) !void {
    const forward = @field(@TypeOf(layer), @tagName(func));
    const ArgsT = stdx.meta.Tail(stdx.meta.FnArgs(forward));

    var args: ArgsT = undefined;

    const LocalContext = struct {
        activation_store: zml.io.TensorStore.View,
        index: usize = 0,
    };

    const input_count = zml.meta.count(zml.Tensor, &args);
    const expected_input_count = countWithPrefix(activation_store, name ++ ".in");
    if (input_count != expected_input_count) {
        log.warn("Reference models uses {d} inputs, but implementation uses {d}", .{ expected_input_count, input_count });
    }

    const store_input = activation_store.withPrefix(name ++ ".in");
    var ctx = LocalContext{ .activation_store = store_input };
    try zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *zml.Tensor) !void {
            var buffer: [256]u8 = undefined;
            const subkey = std.fmt.bufPrint(&buffer, "{d}", .{ctx_.index}) catch unreachable;

            tensor.* = ctx_.activation_store.createTensor(subkey);
            ctx_.index += 1;
        }
    }.cb, &ctx, &args);

    const exe = try platform.compile(allocator, io, layer, func, args);
    defer exe.deinit();

    const output_count = exe.output_shapes.len;
    const store_output = activation_store.withPrefix(name ++ ".out");
    const expected_output_count = countWithPrefix(store_output, name ++ ".out");
    if (output_count != expected_output_count) {
        log.warn("Reference models produces {d} outputs, but implementation produces {d}", .{ expected_output_count, output_count });
    }

    var exe_args = try exe.args(allocator);
    defer exe_args.deinit(allocator);

    var exe_results = try exe.results(allocator);
    defer exe_results.deinit(allocator);

    var args_buffers = try zml.io.load(ArgsT, &args, allocator, io, platform, .{ .dma_chunks = 1, .dma_chunk_size = 4096, .store = activation_store.store, .parallelism = 1 });
    defer zml.meta.visit(struct {
        fn cb(_: void, b: *zml.Buffer) void {
            b.deinit();
        }
    }.cb, {}, &args_buffers);

    exe_args.set(.{ layer_weights, args_buffers });

    exe.callOpts(io, exe_args, &exe_results, .{ .wait = true });

    var results = try allocator.alloc(zml.Buffer, output_count);
    defer allocator.free(results);

    exe_results.fill(.{results});

    var failed: bool = false;
    var reader_buffer: [4096]u8 = undefined;
    for (0..output_count) |i| {
        const expected_slice = b: {
            var buffer: [256]u8 = undefined;
            const subkey = try std.fmt.bufPrint(&buffer, "{d}", .{i});
            var reader = try store_output.getReader(subkey, io, &reader_buffer);
            defer reader.deinit();
            const shape = store_output.getShape(subkey).?;
            const expected_slice: zml.Slice = try .alloc(allocator, shape);
            errdefer expected_slice.free(allocator);
            try reader.interface.readSliceAll(expected_slice.data());
            break :b expected_slice;
        };
        defer allocator.free(expected_slice.data());

        const output_slice = try results[i].toSliceAlloc(allocator, io);
        defer output_slice.free(allocator);

        zml.testing.expectClose(io, expected_slice, output_slice, opts) catch |err| switch (err) {
            error.TestUnexpectedResult => {
                const stderr = std.debug.lockStderr(&.{});
                defer std.debug.unlockStderr();
                const w = &stderr.file_writer.interface;
                try w.print("{s}.{d} doesn't match !", .{ name ++ ".out", i });
                failed = true;
                continue;
            },
            else => return err,
        };
    }

    if (failed) {
        log.info("❌ check failed for {s} ! (absolute tolerance: {e} - relative tolerance: {e} - minimum_close_fraction: {d:0>3})", .{ name, opts.absolute_tolerance, opts.relative_tolerance, opts.minimum_close_fraction });
    } else {
        log.info("✅ all good for {s} ! (absolute tolerance: {e} - relative tolerance: {e} - minimum_close_fraction: {d:0>3})", .{ name, opts.absolute_tolerance, opts.relative_tolerance, opts.minimum_close_fraction });
    }
}

fn center(slice: anytype, i: usize) @TypeOf(slice) {
    const c = 20;
    const start = if (i < c) 0 else i - c;
    const end = @min(start + 2 * c, slice.len);
    return slice[start..end];
}

pub inline fn expectEqual(expected: anytype, actual: @TypeOf(expected)) !void {
    return std.testing.expectEqual(expected, actual);
}
