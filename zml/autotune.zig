const std = @import("std");

const tracer = @import("profiling/tracer.zig");

const log = std.log.scoped(.@"zml/autotune");

pub const AutotuneError = error{
    NoConfigurations,
    NoViableConfiguration,
    InvalidOptions,
    OutOfMemory,
};

pub const Options = struct {
    warmup_rounds: usize = 3,
    initial_samples: usize = 9,
    max_samples: usize = 25,
    target_sample_duration: std.Io.Duration = .fromMilliseconds(5),
    max_sample_duration: std.Io.Duration = .fromMilliseconds(100),
    max_repetitions: usize = 1 << 20,
    tie_threshold: f64 = 0.02,
    shuffle_seed: u64 = 0,
};

pub const Source = enum {
    tuned,
    cache,
    disabled,
    nested,
};

pub fn Result(comptime Config: type) type {
    return struct {
        config: Config,
        source: Source,
        candidate_index: usize,
        median: std.Io.Duration,
        mad: std.Io.Duration,
        repetitions: usize,
        sample_count: usize,
        compiled_count: usize,
        rejected_count: usize,
    };
}

/// Compiles and benchmarks all `configs`, returning a copy of the selected
/// configuration. No benchmark program escapes this function: every
/// successfully compiled program is passed to `deinitFn` exactly once before
/// return, and the caller should compile the selected production program
/// separately.
///
/// `measureFn` must execute exactly `repetitions` representative runs, wait for
/// their measured device work, and return the total measured duration. Any
/// correctness checks and mutable-input restoration must happen outside the
/// duration it returns.
pub fn autotune(
    allocator: std.mem.Allocator,
    ctx: anytype,
    configs: anytype,
    comptime compileFn: anytype,
    comptime measureFn: anytype,
    comptime deinitFn: anytype,
    options: Options,
) AutotuneError!Result(ConfigType(@TypeOf(configs))) {
    const Config = ConfigType(@TypeOf(configs));
    const Program = programType(compileFn);
    const Candidate = CandidateState(Program);
    const config_slice: []const Config = configs;

    if (config_slice.len == 0) return error.NoConfigurations;
    if (!validOptions(options)) return error.InvalidOptions;

    const candidates = try allocator.alloc(Candidate, config_slice.len);
    defer allocator.free(candidates);
    for (candidates, 0..) |*candidate, index| {
        candidate.* = .{ .index = index };
    }
    defer {
        for (candidates) |*candidate| {
            if (candidate.program) |*program| {
                @call(.auto, deinitFn, .{ ctx, program });
                candidate.program = null;
            }
            candidate.samples.deinit(allocator);
        }
    }

    const order = try allocator.alloc(usize, config_slice.len);
    defer allocator.free(order);
    for (order, 0..) |*index, i| index.* = i;

    var compiled_count: usize = 0;
    var rejected_count: usize = 0;

    {
        var span = tracer.span("zml.autotune.compile", .{});
        defer span.end();

        for (config_slice, 0..) |config, index| {
            const program = @call(.auto, compileFn, .{ ctx, config }) catch |err| {
                log.warn("rejecting candidate {d}: compilation failed: {s}", .{ index, @errorName(err) });
                rejected_count += 1;
                traceRejection(index, .compile);
                continue;
            };
            candidates[index].program = program;
            candidates[index].viable = true;
            compiled_count += 1;
        }
    }

    if (viableCount(candidates) == 0) return error.NoViableConfiguration;

    var prng = std.Random.DefaultPrng.init(options.shuffle_seed);
    const random = prng.random();

    {
        var span = tracer.span("zml.autotune.warmup", .{});
        defer span.end();

        for (0..options.warmup_rounds) |_| {
            random.shuffleWithIndex(usize, order, u64);
            for (order) |index| {
                const candidate = &candidates[index];
                if (!candidate.viable) continue;
                const program = if (candidate.program) |*program| program else unreachable;
                const duration = @call(.auto, measureFn, .{ ctx, program, @as(usize, 1) }) catch |err| {
                    log.warn("rejecting candidate {d}: warm-up failed: {s}", .{ index, @errorName(err) });
                    rejectCandidate(ctx, candidate, deinitFn, &rejected_count, .warmup);
                    continue;
                };
                if (duration.nanoseconds < 0) {
                    log.warn("rejecting candidate {d}: warm-up returned a negative duration", .{index});
                    rejectCandidate(ctx, candidate, deinitFn, &rejected_count, .warmup);
                }
            }
        }
    }

    if (viableCount(candidates) == 0) return error.NoViableConfiguration;

    {
        var span = tracer.span("zml.autotune.calibration", .{});
        defer span.end();

        while (hasUncalibratedCandidate(candidates)) {
            random.shuffleWithIndex(usize, order, u64);
            for (order) |index| {
                const candidate = &candidates[index];
                if (!candidate.viable or candidate.calibrated) continue;

                const program = if (candidate.program) |*program| program else unreachable;
                const duration = @call(.auto, measureFn, .{ ctx, program, candidate.repetitions }) catch |err| {
                    log.warn("rejecting candidate {d}: calibration failed: {s}", .{ index, @errorName(err) });
                    rejectCandidate(ctx, candidate, deinitFn, &rejected_count, .calibration);
                    continue;
                };
                if (duration.nanoseconds < 0) {
                    log.warn("rejecting candidate {d}: calibration returned a negative duration", .{index});
                    rejectCandidate(ctx, candidate, deinitFn, &rejected_count, .calibration);
                    continue;
                }

                if (duration.nanoseconds == 0) {
                    if (candidate.repetitions == options.max_repetitions) {
                        log.warn("rejecting candidate {d}: timer resolution remained zero at the repetition limit", .{index});
                        rejectCandidate(ctx, candidate, deinitFn, &rejected_count, .calibration);
                    } else {
                        candidate.repetitions = nextRepetitionCount(candidate.repetitions, options.max_repetitions);
                    }
                    continue;
                }

                if (duration.nanoseconds >= options.target_sample_duration.nanoseconds or
                    candidate.repetitions == options.max_repetitions)
                {
                    candidate.calibrated = true;
                    continue;
                }

                const next_repetitions = nextRepetitionCount(candidate.repetitions, options.max_repetitions);
                const projected_duration = @as(f64, @floatFromInt(duration.nanoseconds)) *
                    @as(f64, @floatFromInt(next_repetitions)) /
                    @as(f64, @floatFromInt(candidate.repetitions));
                if (projected_duration > @as(f64, @floatFromInt(options.max_sample_duration.nanoseconds))) {
                    candidate.calibrated = true;
                } else {
                    candidate.repetitions = next_repetitions;
                }
            }
        }
    }

    if (viableCount(candidates) == 0) return error.NoViableConfiguration;

    const scratch = try allocator.alloc(i96, options.max_samples);
    defer allocator.free(scratch);

    {
        var span = tracer.span("zml.autotune.sampling", .{});
        defer span.end();

        for (0..options.initial_samples) |_| {
            try sampleRound(
                allocator,
                ctx,
                candidates,
                order,
                random,
                measureFn,
                deinitFn,
                options.max_repetitions,
                &rejected_count,
            );
            if (viableCount(candidates) == 0) return error.NoViableConfiguration;
        }

        updateAllStatistics(candidates, scratch);

        while (true) {
            const leaders = findLeaders(candidates);
            if (leaders.second == null or !candidatesAreTied(candidates, leaders.first.?, leaders.second.?, options.tie_threshold)) break;
            if (candidates[leaders.first.?].samples.items.len >= options.max_samples) break;

            try sampleRound(
                allocator,
                ctx,
                candidates,
                order,
                random,
                measureFn,
                deinitFn,
                options.max_repetitions,
                &rejected_count,
            );
            if (viableCount(candidates) == 0) return error.NoViableConfiguration;
            updateAllStatistics(candidates, scratch);
        }
    }

    var selection_span = tracer.span("zml.autotune.select", .{});
    defer selection_span.end();

    const leaders = findLeaders(candidates);
    const fastest_index = leaders.first orelse return error.NoViableConfiguration;
    var selected_index = fastest_index;
    if (leaders.second) |second_index| {
        if (candidatesAreTied(candidates, fastest_index, second_index, options.tie_threshold) and
            candidates[fastest_index].samples.items.len >= options.max_samples)
        {
            for (candidates, 0..) |candidate, index| {
                if (!candidate.viable) continue;
                if (candidatesAreTied(candidates, fastest_index, index, options.tie_threshold)) {
                    selected_index = @min(selected_index, index);
                }
            }
        }
    }

    const selected = &candidates[selected_index];
    log.info("selected autotune candidate {d} with median {f} and MAD {f}", .{
        selected_index,
        std.Io.Duration.fromNanoseconds(selected.median_ns),
        std.Io.Duration.fromNanoseconds(selected.mad_ns),
    });
    return .{
        .config = config_slice[selected_index],
        .source = .tuned,
        .candidate_index = selected_index,
        .median = .fromNanoseconds(selected.median_ns),
        .mad = .fromNanoseconds(selected.mad_ns),
        .repetitions = selected.repetitions,
        .sample_count = selected.samples.items.len,
        .compiled_count = compiled_count,
        .rejected_count = rejected_count,
    };
}

/// Measures a synchronized host fallback using one outer interval around all
/// repetitions. `runFn` must wait for completion (for an `Exe`, use
/// `.wait = true`). Inputs, result storage, and device buffers must already
/// exist before calling this helper.
pub fn measureHost(
    io: std.Io,
    ctx: anytype,
    program: anytype,
    repetitions: usize,
    comptime runFn: anytype,
) !std.Io.Duration {
    std.debug.assert(repetitions != 0);
    const start: std.Io.Timestamp = .now(io, .awake);
    for (0..repetitions) |_| try @call(.auto, runFn, .{ ctx, program });
    return start.untilNow(io, .awake);
}

const RejectionStage = enum {
    compile,
    warmup,
    calibration,
    sampling,
};

fn traceRejection(index: usize, stage: RejectionStage) void {
    var span = tracer.span("zml.autotune.reject", .{ .candidate_index = index, .stage = stage });
    span.end();
}

fn rejectCandidate(
    ctx: anytype,
    candidate: anytype,
    comptime deinitFn: anytype,
    rejected_count: *usize,
    stage: RejectionStage,
) void {
    std.debug.assert(candidate.viable);
    if (candidate.program) |*program| {
        @call(.auto, deinitFn, .{ ctx, program });
        candidate.program = null;
    }
    candidate.viable = false;
    rejected_count.* += 1;
    traceRejection(candidate.index, stage);
}

fn sampleRound(
    allocator: std.mem.Allocator,
    ctx: anytype,
    candidates: anytype,
    order: []usize,
    random: std.Random,
    comptime measureFn: anytype,
    comptime deinitFn: anytype,
    max_repetitions: usize,
    rejected_count: *usize,
) AutotuneError!void {
    random.shuffleWithIndex(usize, order, u64);
    for (order) |index| {
        const candidate = &candidates[index];
        if (!candidate.viable) continue;

        while (true) {
            const program = if (candidate.program) |*program| program else unreachable;
            const duration = @call(.auto, measureFn, .{ ctx, program, candidate.repetitions }) catch |err| {
                log.warn("rejecting candidate {d}: measurement failed: {s}", .{ index, @errorName(err) });
                rejectCandidate(ctx, candidate, deinitFn, rejected_count, .sampling);
                break;
            };
            if (duration.nanoseconds < 0) {
                log.warn("rejecting candidate {d}: measurement returned a negative duration", .{index});
                rejectCandidate(ctx, candidate, deinitFn, rejected_count, .sampling);
                break;
            }
            if (duration.nanoseconds == 0) {
                if (candidate.repetitions == max_repetitions) {
                    log.warn("rejecting candidate {d}: timer resolution remained zero at the repetition limit", .{index});
                    rejectCandidate(ctx, candidate, deinitFn, rejected_count, .sampling);
                    break;
                }
                candidate.repetitions = nextRepetitionCount(candidate.repetitions, max_repetitions);
                continue;
            }

            const repetitions_i96: i96 = @intCast(candidate.repetitions);
            const per_run_ns = @divTrunc(duration.nanoseconds - 1, repetitions_i96) + 1;
            candidate.samples.append(allocator, per_run_ns) catch return error.OutOfMemory;
            break;
        }
    }
}

fn validOptions(options: Options) bool {
    return options.initial_samples != 0 and
        options.max_samples >= options.initial_samples and
        options.target_sample_duration.nanoseconds > 0 and
        options.max_sample_duration.nanoseconds >= options.target_sample_duration.nanoseconds and
        options.max_repetitions != 0 and
        std.math.isFinite(options.tie_threshold) and
        options.tie_threshold >= 0;
}

fn nextRepetitionCount(current: usize, maximum: usize) usize {
    std.debug.assert(current < maximum);
    return if (current > maximum - current) maximum else current * 2;
}

fn viableCount(candidates: anytype) usize {
    var count: usize = 0;
    for (candidates) |candidate| count += @intFromBool(candidate.viable);
    return count;
}

fn hasUncalibratedCandidate(candidates: anytype) bool {
    for (candidates) |candidate| {
        if (candidate.viable and !candidate.calibrated) return true;
    }
    return false;
}

fn updateAllStatistics(candidates: anytype, scratch: []i96) void {
    for (candidates) |*candidate| {
        if (!candidate.viable) continue;
        updateStatistics(candidate, scratch);
    }
}

fn updateStatistics(candidate: anytype, scratch: []i96) void {
    const samples = candidate.samples.items;
    std.debug.assert(samples.len != 0 and samples.len <= scratch.len);
    @memcpy(scratch[0..samples.len], samples);
    std.mem.sort(i96, scratch[0..samples.len], {}, comptime std.sort.asc(i96));
    candidate.median_ns = medianOfSorted(scratch[0..samples.len]);

    for (samples, 0..) |sample, i| {
        scratch[i] = @intCast(@abs(sample - candidate.median_ns));
    }
    std.mem.sort(i96, scratch[0..samples.len], {}, comptime std.sort.asc(i96));
    candidate.mad_ns = medianOfSorted(scratch[0..samples.len]);
}

fn medianOfSorted(values: []const i96) i96 {
    std.debug.assert(values.len != 0);
    const middle = values.len / 2;
    if (values.len % 2 != 0) return values[middle];
    const lower = values[middle - 1];
    const upper = values[middle];
    return lower + @divTrunc(upper - lower, 2);
}

const Leaders = struct {
    first: ?usize = null,
    second: ?usize = null,
};

fn findLeaders(candidates: anytype) Leaders {
    var result: Leaders = .{};
    for (candidates, 0..) |candidate, index| {
        if (!candidate.viable) continue;
        if (result.first == null or candidateLessThan(candidate, candidates[result.first.?])) {
            result.second = result.first;
            result.first = index;
        } else if (result.second == null or candidateLessThan(candidate, candidates[result.second.?])) {
            result.second = index;
        }
    }
    return result;
}

fn candidateLessThan(lhs: anytype, rhs: @TypeOf(lhs)) bool {
    return lhs.median_ns < rhs.median_ns or
        (lhs.median_ns == rhs.median_ns and lhs.index < rhs.index);
}

fn candidatesAreTied(candidates: anytype, lhs_index: usize, rhs_index: usize, base_threshold: f64) bool {
    if (lhs_index == rhs_index) return true;
    const lhs = candidates[lhs_index];
    const rhs = candidates[rhs_index];
    const faster_ns = @min(lhs.median_ns, rhs.median_ns);
    const slower_ns = @max(lhs.median_ns, rhs.median_ns);
    if (faster_ns <= 0) return faster_ns == slower_ns;

    const lhs_relative_mad = @as(f64, @floatFromInt(lhs.mad_ns)) /
        @as(f64, @floatFromInt(lhs.median_ns));
    const rhs_relative_mad = @as(f64, @floatFromInt(rhs.mad_ns)) /
        @as(f64, @floatFromInt(rhs.median_ns));
    const threshold = @max(base_threshold, 3 * @max(lhs_relative_mad, rhs_relative_mad));
    const relative_difference = @as(f64, @floatFromInt(slower_ns - faster_ns)) /
        @as(f64, @floatFromInt(faster_ns));
    return relative_difference < threshold;
}

fn CandidateState(comptime Program: type) type {
    return struct {
        index: usize,
        program: ?Program = null,
        viable: bool = false,
        calibrated: bool = false,
        repetitions: usize = 1,
        samples: std.ArrayList(i96) = .empty,
        median_ns: i96 = 0,
        mad_ns: i96 = 0,
    };
}

pub fn ConfigType(comptime T: type) type {
    const message = "configs must be a slice, array, or pointer to an array";
    return switch (@typeInfo(T)) {
        .array => |array| array.child,
        .pointer => |pointer| switch (pointer.size) {
            .slice => pointer.child,
            .one => switch (@typeInfo(pointer.child)) {
                .array => |array| array.child,
                else => @compileError(message),
            },
            else => @compileError(message),
        },
        else => @compileError(message),
    };
}

fn programType(comptime compileFn: anytype) type {
    const function = switch (@typeInfo(@TypeOf(compileFn))) {
        .@"fn" => |function| function,
        else => @compileError("compileFn must be a function"),
    };
    const Return = function.return_type orelse @compileError("compileFn must have an explicit return type");
    return switch (@typeInfo(Return)) {
        .error_union => |error_union| error_union.payload,
        else => @compileError("compileFn must return an error union"),
    };
}

const FakeProgram = struct {
    config: u8,
    measure_calls: usize = 0,
};

const FakeContext = struct {
    const max_configs = 8;
    const max_records = 2048;

    const Record = struct {
        config: u8,
        repetitions: usize,
    };

    per_run_ns: [max_configs]i96 = [_]i96{100} ** max_configs,
    compile_fails: [max_configs]bool = [_]bool{false} ** max_configs,
    measure_fail_at: [max_configs]?usize = [_]?usize{null} ** max_configs,
    always_zero: [max_configs]bool = [_]bool{false} ** max_configs,
    sample_values: [max_configs][8]i96 = [_][8]i96{[_]i96{0} ** 8} ** max_configs,
    sample_value_count: [max_configs]usize = [_]usize{0} ** max_configs,
    deinit_counts: [max_configs]usize = [_]usize{0} ** max_configs,
    records: [max_records]Record = undefined,
    record_count: usize = 0,

    fn compile(self: *FakeContext, config: u8) error{CompileFailed}!FakeProgram {
        if (self.compile_fails[config]) return error.CompileFailed;
        return .{ .config = config };
    }

    fn measure(self: *FakeContext, program: *FakeProgram, repetitions: usize) error{MeasureFailed}!std.Io.Duration {
        const config = program.config;
        std.debug.assert(self.record_count < self.records.len);
        self.records[self.record_count] = .{ .config = config, .repetitions = repetitions };
        self.record_count += 1;

        const call_index = program.measure_calls;
        program.measure_calls += 1;
        if (self.measure_fail_at[config]) |failure_index| {
            if (call_index >= failure_index) return error.MeasureFailed;
        }
        if (self.always_zero[config]) return .zero;

        var per_run_ns = self.per_run_ns[config];
        if (call_index > 0 and self.sample_value_count[config] != 0) {
            const sample_index = (call_index - 1) % self.sample_value_count[config];
            per_run_ns = self.sample_values[config][sample_index];
        }
        return .fromNanoseconds(per_run_ns * @as(i96, @intCast(repetitions)));
    }

    fn deinit(self: *FakeContext, program: *FakeProgram) void {
        self.deinit_counts[program.config] += 1;
    }
};

fn testOptions() Options {
    return .{
        .warmup_rounds = 0,
        .initial_samples = 3,
        .max_samples = 3,
        .target_sample_duration = .fromNanoseconds(1),
        .max_sample_duration = .fromNanoseconds(1_000_000),
        .max_repetitions = 16,
        .tie_threshold = 0.02,
        .shuffle_seed = 0,
    };
}

test "autotune infers arbitrary configuration and program types" {
    const Config = struct {
        id: u8,
        tile_size: u16,
    };
    const Program = struct {
        duration_ns: i96,
    };
    const Context = struct {
        deinit_count: usize = 0,

        fn compile(_: *@This(), config: Config) error{}!Program {
            return .{ .duration_ns = if (config.id == 0) 20 else 10 };
        }

        fn measure(_: *@This(), program: *Program, repetitions: usize) error{}!std.Io.Duration {
            return .fromNanoseconds(program.duration_ns * @as(i96, @intCast(repetitions)));
        }

        fn deinit(self: *@This(), _: *Program) void {
            self.deinit_count += 1;
        }
    };

    var ctx: Context = .{};
    const configs = [_]Config{
        .{ .id = 0, .tile_size = 16 },
        .{ .id = 1, .tile_size = 32 },
    };
    const result = try autotune(
        std.testing.allocator,
        &ctx,
        &configs,
        Context.compile,
        Context.measure,
        Context.deinit,
        testOptions(),
    );

    try std.testing.expectEqual(Config, @TypeOf(result.config));
    try std.testing.expectEqual(Source.tuned, result.source);
    try std.testing.expectEqual(@as(u8, 1), result.config.id);
    try std.testing.expectEqual(@as(usize, 2), ctx.deinit_count);
}

test "autotune rejects empty input and invalid options" {
    var ctx: FakeContext = .{};
    const empty: [0]u8 = .{};
    try std.testing.expectError(error.NoConfigurations, autotune(
        std.testing.allocator,
        &ctx,
        empty[0..],
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        .{},
    ));

    var invalid = testOptions();
    invalid.initial_samples = 0;
    try std.testing.expectError(error.InvalidOptions, autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{0},
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        invalid,
    ));
}

test "autotune reports all failed compilations" {
    var ctx: FakeContext = .{};
    ctx.compile_fails[0] = true;
    ctx.compile_fails[1] = true;
    try std.testing.expectError(error.NoViableConfiguration, autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{ 0, 1 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        testOptions(),
    ));
    try std.testing.expectEqual(@as(usize, 0), ctx.deinit_counts[0]);
    try std.testing.expectEqual(@as(usize, 0), ctx.deinit_counts[1]);
}

test "autotune isolates partial compile and measurement failures" {
    var ctx: FakeContext = .{};
    ctx.compile_fails[1] = true;
    ctx.measure_fail_at[2] = 0;
    const result = try autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{ 0, 1, 2 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        testOptions(),
    );

    try std.testing.expectEqual(@as(usize, 0), result.candidate_index);
    try std.testing.expectEqual(@as(usize, 2), result.compiled_count);
    try std.testing.expectEqual(@as(usize, 2), result.rejected_count);
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[0]);
    try std.testing.expectEqual(@as(usize, 0), ctx.deinit_counts[1]);
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[2]);
}

test "autotune deinitializes every measured failure before returning" {
    var ctx: FakeContext = .{};
    ctx.measure_fail_at[0] = 0;
    ctx.measure_fail_at[1] = 0;
    try std.testing.expectError(error.NoViableConfiguration, autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{ 0, 1 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        testOptions(),
    ));
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[0]);
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[1]);
}

test "autotune deinitializes every compiled program on allocator failures" {
    const Program = struct { duration_ns: i96 };
    const Context = struct {
        compiled: usize = 0,
        deinitialized: usize = 0,

        fn compile(self: *@This(), config: u8) error{}!Program {
            self.compiled += 1;
            return .{ .duration_ns = 100 - config };
        }

        fn measure(_: *@This(), program: *Program, repetitions: usize) error{}!std.Io.Duration {
            return .fromNanoseconds(program.duration_ns * @as(i96, @intCast(repetitions)));
        }

        fn deinit(self: *@This(), _: *Program) void {
            self.deinitialized += 1;
        }
    };
    const testImpl = struct {
        fn run(allocator: std.mem.Allocator) !void {
            var ctx: Context = .{};
            const result = autotune(
                allocator,
                &ctx,
                &[_]u8{ 0, 1 },
                Context.compile,
                Context.measure,
                Context.deinit,
                testOptions(),
            );
            try std.testing.expectEqual(ctx.compiled, ctx.deinitialized);
            _ = try result;
        }
    }.run;

    try std.testing.checkAllAllocationFailures(std.testing.allocator, testImpl, .{});
}

test "autotune doubles repetitions during independent calibration" {
    var ctx: FakeContext = .{};
    ctx.per_run_ns[0] = 1;
    var options = testOptions();
    options.initial_samples = 1;
    options.max_samples = 1;
    options.target_sample_duration = .fromNanoseconds(8);
    const result = try autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{0},
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        options,
    );

    try std.testing.expectEqual(@as(usize, 8), result.repetitions);
    try std.testing.expectEqual(@as(usize, 5), ctx.record_count);
    try std.testing.expectEqual(@as(usize, 1), ctx.records[0].repetitions);
    try std.testing.expectEqual(@as(usize, 2), ctx.records[1].repetitions);
    try std.testing.expectEqual(@as(usize, 4), ctx.records[2].repetitions);
    try std.testing.expectEqual(@as(usize, 8), ctx.records[3].repetitions);
    try std.testing.expectEqual(@as(usize, 8), ctx.records[4].repetitions);
}

test "autotune rejects zero-resolution timing only at repetition limit" {
    var ctx: FakeContext = .{};
    ctx.always_zero[0] = true;
    var options = testOptions();
    options.max_repetitions = 4;
    try std.testing.expectError(error.NoViableConfiguration, autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{0},
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        options,
    ));

    try std.testing.expectEqual(@as(usize, 3), ctx.record_count);
    try std.testing.expectEqual(@as(usize, 1), ctx.records[0].repetitions);
    try std.testing.expectEqual(@as(usize, 2), ctx.records[1].repetitions);
    try std.testing.expectEqual(@as(usize, 4), ctx.records[2].repetitions);
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[0]);
}

test "autotune computes median and median absolute deviation" {
    var ctx: FakeContext = .{};
    ctx.sample_values[0][0..3].* = .{ 90, 100, 110 };
    ctx.sample_value_count[0] = 3;
    const result = try autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{0},
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        testOptions(),
    );

    try std.testing.expectEqual(@as(i96, 100), result.median.nanoseconds);
    try std.testing.expectEqual(@as(i96, 10), result.mad.nanoseconds);
}

test "autotune adaptively samples close candidates and keeps stable order ties" {
    var ctx: FakeContext = .{};
    ctx.per_run_ns[0] = 100;
    ctx.per_run_ns[1] = 101;
    var options = testOptions();
    options.initial_samples = 1;
    options.max_samples = 3;
    const result = try autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{ 0, 1 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        options,
    );

    try std.testing.expectEqual(@as(usize, 0), result.candidate_index);
    try std.testing.expectEqual(@as(usize, 3), result.sample_count);
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[0]);
    try std.testing.expectEqual(@as(usize, 1), ctx.deinit_counts[1]);
}

test "autotune stops initial sampling when leaders are distinct" {
    var ctx: FakeContext = .{};
    ctx.per_run_ns[0] = 100;
    ctx.per_run_ns[1] = 80;
    var options = testOptions();
    options.initial_samples = 2;
    options.max_samples = 5;
    const result = try autotune(
        std.testing.allocator,
        &ctx,
        &[_]u8{ 0, 1 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        options,
    );

    try std.testing.expectEqual(@as(usize, 1), result.candidate_index);
    try std.testing.expectEqual(@as(usize, 2), result.sample_count);
}

test "autotune shuffling is deterministic for a fixed seed" {
    var first: FakeContext = .{};
    var second: FakeContext = .{};
    var options = testOptions();
    options.warmup_rounds = 2;
    options.initial_samples = 2;
    options.max_samples = 2;
    options.shuffle_seed = 42;
    _ = try autotune(
        std.testing.allocator,
        &first,
        &[_]u8{ 0, 1, 2, 3 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        options,
    );
    _ = try autotune(
        std.testing.allocator,
        &second,
        &[_]u8{ 0, 1, 2, 3 },
        FakeContext.compile,
        FakeContext.measure,
        FakeContext.deinit,
        options,
    );

    try std.testing.expectEqual(first.record_count, second.record_count);
    try std.testing.expectEqualSlices(
        FakeContext.Record,
        first.records[0..first.record_count],
        second.records[0..second.record_count],
    );
}

test "measureHost surrounds all synchronized repetitions with one interval" {
    const Program = struct {
        delay: std.Io.Duration,
    };
    const Context = struct {
        io: std.Io,
        run_count: usize = 0,

        fn run(self: *@This(), program: *Program) !void {
            self.run_count += 1;
            if (program.delay.nanoseconds != 0) try self.io.sleep(program.delay, .awake);
        }
    };

    var ctx: Context = .{ .io = std.testing.io };
    var fast: Program = .{ .delay = .zero };
    var slow: Program = .{ .delay = .fromMilliseconds(2) };
    const fast_duration = try measureHost(std.testing.io, &ctx, &fast, 2, Context.run);
    const slow_duration = try measureHost(std.testing.io, &ctx, &slow, 2, Context.run);

    try std.testing.expectEqual(@as(usize, 4), ctx.run_count);
    try std.testing.expect(slow_duration.nanoseconds > fast_duration.nanoseconds);
}
