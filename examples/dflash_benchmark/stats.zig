const std = @import("std");

pub const Duration = struct {
    nanoseconds: i128 = 0,
};

pub const QualityComparison = struct {
    exact_match: bool = false,
    compared_tokens: usize = 0,
    first_mismatch_index: ?usize = null,
};

pub const MethodResult = struct {
    /// Generated token ids are optional, but allow exact baseline-vs-DFlash checks.
    token_ids: []const u32 = &.{},
    /// Number of generated output tokens. If zero and token_ids is populated,
    /// metric helpers use token_ids.len.
    decoded_tokens: usize = 0,
    generated_text: []const u8 = "",
    elapsed: Duration = .{ .nanoseconds = 0 },
    stopped_on_eos: bool = false,

    /// One entry per DFlash speculative step: number of accepted draft tokens.
    acceptance_lengths: []const u32 = &.{},
    /// One entry per DFlash speculative step: total committed output tokens.
    committed_lengths: []const u32 = &.{},
    /// Optional precomputed counts indexed by accepted draft-token count.
    acceptance_length_histogram: []const u64 = &.{},
    /// Optional per-position counts. Position 0 represents first draft token.
    per_position_accepts: []const u64 = &.{},
    per_position_trials: []const u64 = &.{},

    pub fn tokenCount(self: MethodResult) usize {
        return if (self.decoded_tokens != 0) self.decoded_tokens else self.token_ids.len;
    }

    pub fn tpotMs(self: MethodResult) f64 {
        return calcTpotMs(self.elapsed, self.tokenCount());
    }

    pub fn tokensPerSecond(self: MethodResult) f64 {
        return calcTokensPerSecond(self.elapsed, self.tokenCount());
    }
};

pub const SampleResult = struct {
    id: []const u8,
    dataset: []const u8,
    prompt_tokens: usize,
    baseline: MethodResult,
    dflash: MethodResult,
    quality: QualityComparison,
};

pub const Summary = struct {
    sample_count: usize,
    dataset: []const u8,
    baseline_tokens: usize,
    dflash_tokens: usize,
    baseline_elapsed: Duration,
    dflash_elapsed: Duration,
    baseline_tpot_ms: f64,
    baseline_tps: f64,
    dflash_tpot_ms: f64,
    dflash_tps: f64,
    speedup: f64,
    tau: f64,
    per_position_acceptance_rates: []f64,
    per_position_accepts: []u64,
    per_position_trials: []u64,
    acceptance_length_histogram: []u64,
    acceptance_length_histogram_rates: []f64,
    exact_match_count: usize,

    pub fn deinit(self: Summary, allocator: std.mem.Allocator) void {
        allocator.free(self.per_position_acceptance_rates);
        allocator.free(self.per_position_accepts);
        allocator.free(self.per_position_trials);
        allocator.free(self.acceptance_length_histogram);
        allocator.free(self.acceptance_length_histogram_rates);
    }
};

pub const JsonConfig = struct {
    model: []const u8 = "",
    draft_model: []const u8 = "",
    dataset: []const u8 = "",
    split: []const u8 = "",
    sample_count: usize = 0,
    seed: u64 = 0,
    max_tokens: usize = 0,
    temperature: f64 = 0,
};

pub fn compareTokenIds(baseline: []const u32, dflash: []const u32) QualityComparison {
    const compared = @min(baseline.len, dflash.len);
    for (baseline[0..compared], dflash[0..compared], 0..) |a, b, i| {
        if (a != b) return .{
            .exact_match = false,
            .compared_tokens = compared,
            .first_mismatch_index = i,
        };
    }
    return .{
        .exact_match = baseline.len == dflash.len,
        .compared_tokens = compared,
        .first_mismatch_index = if (baseline.len == dflash.len) null else compared,
    };
}

pub fn computeSummary(allocator: std.mem.Allocator, samples: []const SampleResult) !Summary {
    var baseline_tokens: usize = 0;
    var dflash_tokens: usize = 0;
    var baseline_ns: i128 = 0;
    var dflash_ns: i128 = 0;
    var exact_match_count: usize = 0;
    var max_acceptance_len: usize = 0;
    var max_position_count: usize = 0;
    var acceptance_step_count: u64 = 0;
    var acceptance_total: u64 = 0;

    for (samples) |sample| {
        baseline_tokens += sample.baseline.tokenCount();
        dflash_tokens += sample.dflash.tokenCount();
        baseline_ns += sample.baseline.elapsed.nanoseconds;
        dflash_ns += sample.dflash.elapsed.nanoseconds;
        if (sample.quality.exact_match) exact_match_count += 1;

        max_position_count = @max(maxPositionCount(sample.dflash), max_position_count);
        if (sample.dflash.acceptance_length_histogram.len != 0) {
            max_acceptance_len = @max(max_acceptance_len, sample.dflash.acceptance_length_histogram.len - 1);
        }
        if (sample.dflash.acceptance_lengths.len != 0) {
            for (sample.dflash.acceptance_lengths) |len| {
                max_acceptance_len = @max(max_acceptance_len, len);
            }
        }
        if (sample.dflash.committed_lengths.len != 0) {
            acceptance_step_count += sample.dflash.committed_lengths.len;
            for (sample.dflash.committed_lengths) |len| acceptance_total += len;
        } else if (sample.dflash.acceptance_lengths.len != 0) {
            acceptance_step_count += sample.dflash.acceptance_lengths.len;
            for (sample.dflash.acceptance_lengths) |len| acceptance_total += len + 1;
        } else {
            for (sample.dflash.acceptance_length_histogram, 0..) |count, len| {
                if (count == 0) continue;
                max_acceptance_len = @max(max_acceptance_len, len);
                acceptance_step_count += count;
                acceptance_total += count * (len + 1);
            }
        }
    }

    const histogram = try allocator.alloc(u64, max_acceptance_len + 1);
    errdefer allocator.free(histogram);
    @memset(histogram, 0);
    const position_accepts = try allocator.alloc(u64, max_position_count);
    errdefer allocator.free(position_accepts);
    @memset(position_accepts, 0);
    const position_trials = try allocator.alloc(u64, max_position_count);
    errdefer allocator.free(position_trials);
    @memset(position_trials, 0);

    for (samples) |sample| {
        aggregateHistogram(histogram, sample.dflash);
        aggregatePositions(position_accepts, position_trials, sample.dflash);
    }
    var histogram_total: u64 = 0;
    for (histogram) |count| histogram_total += count;

    const position_rates = try allocator.alloc(f64, max_position_count);
    errdefer allocator.free(position_rates);
    for (position_rates, 0..) |*rate, i| {
        rate.* = if (position_trials[i] == 0)
            0
        else
            @as(f64, @floatFromInt(position_accepts[i])) / @as(f64, @floatFromInt(position_trials[i]));
    }

    const histogram_rates = try allocator.alloc(f64, histogram.len);
    errdefer allocator.free(histogram_rates);
    for (histogram_rates, histogram) |*rate, count| {
        rate.* = if (histogram_total == 0)
            0
        else
            @as(f64, @floatFromInt(count)) / @as(f64, @floatFromInt(histogram_total));
    }

    const baseline_elapsed: Duration = .{ .nanoseconds = baseline_ns };
    const dflash_elapsed: Duration = .{ .nanoseconds = dflash_ns };
    const baseline_tpot = calcTpotMs(baseline_elapsed, baseline_tokens);
    const dflash_tpot = calcTpotMs(dflash_elapsed, dflash_tokens);

    return .{
        .sample_count = samples.len,
        .dataset = if (samples.len == 0) "" else samples[0].dataset,
        .baseline_tokens = baseline_tokens,
        .dflash_tokens = dflash_tokens,
        .baseline_elapsed = baseline_elapsed,
        .dflash_elapsed = dflash_elapsed,
        .baseline_tpot_ms = baseline_tpot,
        .baseline_tps = calcTokensPerSecond(baseline_elapsed, baseline_tokens),
        .dflash_tpot_ms = dflash_tpot,
        .dflash_tps = calcTokensPerSecond(dflash_elapsed, dflash_tokens),
        .speedup = if (dflash_tpot == 0) 0 else baseline_tpot / dflash_tpot,
        .tau = if (acceptance_step_count == 0)
            0
        else
            @as(f64, @floatFromInt(acceptance_total)) / @as(f64, @floatFromInt(acceptance_step_count)),
        .per_position_acceptance_rates = position_rates,
        .per_position_accepts = position_accepts,
        .per_position_trials = position_trials,
        .acceptance_length_histogram = histogram,
        .acceptance_length_histogram_rates = histogram_rates,
        .exact_match_count = exact_match_count,
    };
}

pub fn printSample(writer: *std.Io.Writer, index: usize, total: usize, sample: SampleResult) !void {
    try writer.print("[{d}/{d}] dataset={s} id={s} prompt_tokens={d}\n", .{ index, total, sample.dataset, sample.id, sample.prompt_tokens });
    try writer.print("  Baseline: {d} tokens, TPOT={d:.2}ms, TPS={d:.1}\n", .{
        sample.baseline.tokenCount(),
        sample.baseline.tpotMs(),
        sample.baseline.tokensPerSecond(),
    });
    try writer.print("  DFlash:   {d} tokens, TPOT={d:.2}ms, TPS={d:.1}, tau={d:.2}\n", .{
        sample.dflash.tokenCount(),
        sample.dflash.tpotMs(),
        sample.dflash.tokensPerSecond(),
        methodTau(sample.dflash),
    });
    if (sample.quality.exact_match) {
        try writer.print("  Quality:  MATCH first {d} output tokens\n", .{sample.quality.compared_tokens});
    } else if (sample.quality.first_mismatch_index) |mismatch| {
        try writer.print("  Quality:  MISMATCH at output token {d} ({d} compared)\n", .{ mismatch, sample.quality.compared_tokens });
    } else {
        try writer.print("  Quality:  MISMATCH ({d} compared)\n", .{sample.quality.compared_tokens});
    }
}

pub fn printReport(writer: *std.Io.Writer, summary: Summary) !void {
    try writer.writeAll(
        \\============================================================
        \\RESULTS
        \\============================================================
        \\
    );
    try writer.print("Dataset:        {s}\n", .{summary.dataset});
    try writer.print("Samples:        {d}\n", .{summary.sample_count});
    try writer.print("Baseline TPOT:  {d:.2} ms ({d:.1} TPS)\n", .{ summary.baseline_tpot_ms, summary.baseline_tps });
    try writer.print("DFlash TPOT:    {d:.2} ms ({d:.1} TPS)\n", .{ summary.dflash_tpot_ms, summary.dflash_tps });
    try writer.print("Speedup:        {d:.2}x\n", .{summary.speedup});
    try writer.print("Tau:            {d:.2}\n\n", .{summary.tau});

    try writer.writeAll("Per-position acceptance rate:\n");
    for (summary.per_position_acceptance_rates, 0..) |rate, i| {
        try writer.print("  pos {d:>2}: {d:.3} ", .{ i + 1, rate });
        try writer.splatByteAll('x', barLen(rate, 50));
        try writer.writeByte('\n');
    }
    if (summary.per_position_acceptance_rates.len == 0) try writer.writeAll("  (no DFlash acceptance data)\n");

    try writer.writeAll("\nAcceptance length histogram: [");
    for (summary.acceptance_length_histogram_rates, 0..) |rate, i| {
        if (i != 0) try writer.writeAll(", ");
        try writer.print("{d:.3}", .{rate});
    }
    try writer.writeAll("]\n\n");
    try writer.print("Output quality: {d}/{d} samples match baseline exactly\n", .{
        summary.exact_match_count,
        summary.sample_count,
    });
}

pub fn writeJson(writer: *std.Io.Writer, config: ?JsonConfig, samples: []const SampleResult, summary: Summary) !void {
    try writer.writeAll("{\"config\":");
    if (config) |cfg| {
        try writeConfigJson(writer, cfg);
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll(",\"samples\":[");
    for (samples, 0..) |sample, i| {
        if (i != 0) try writer.writeByte(',');
        try writeSampleJson(writer, sample);
    }
    try writer.writeAll("],\"summary\":");
    try writeSummaryJson(writer, summary);
    try writer.writeAll("}\n");
}

fn writeConfigJson(writer: *std.Io.Writer, cfg: JsonConfig) !void {
    try writer.writeAll("{\"model\":");
    try writeJsonString(writer, cfg.model);
    try writer.writeAll(",\"draft_model\":");
    try writeJsonString(writer, cfg.draft_model);
    try writer.writeAll(",\"dataset\":");
    try writeJsonString(writer, cfg.dataset);
    try writer.writeAll(",\"split\":");
    try writeJsonString(writer, cfg.split);
    try writer.print(",\"sample_count\":{d},\"seed\":{d},\"max_tokens\":{d},\"temperature\":{d}", .{
        cfg.sample_count,
        cfg.seed,
        cfg.max_tokens,
        cfg.temperature,
    });
    try writer.writeByte('}');
}

fn writeSampleJson(writer: *std.Io.Writer, sample: SampleResult) !void {
    try writer.writeAll("{\"id\":");
    try writeJsonString(writer, sample.id);
    try writer.writeAll(",\"dataset\":");
    try writeJsonString(writer, sample.dataset);
    try writer.print(",\"prompt_tokens\":{d},\"baseline\":", .{sample.prompt_tokens});
    try writeMethodJson(writer, sample.baseline, false);
    try writer.writeAll(",\"dflash\":");
    try writeMethodJson(writer, sample.dflash, true);
    try writer.writeAll(",\"quality\":");
    try writer.print("{{\"exact_match\":{},\"compared_tokens\":{d},\"first_mismatch_index\":", .{
        sample.quality.exact_match,
        sample.quality.compared_tokens,
    });
    if (sample.quality.first_mismatch_index) |idx| {
        try writer.print("{d}", .{idx});
    } else {
        try writer.writeAll("null");
    }
    try writer.writeAll("}}");
    try writer.writeByte('}');
}

fn writeMethodJson(writer: *std.Io.Writer, method: MethodResult, include_acceptance: bool) !void {
    try writer.print("{{\"decoded_tokens\":{d},\"elapsed_ms\":{d:.3},\"tpot_ms\":{d:.3},\"tps\":{d:.3},\"stopped_on_eos\":{},\"token_ids\":", .{
        method.tokenCount(),
        durationMs(method.elapsed),
        method.tpotMs(),
        method.tokensPerSecond(),
        method.stopped_on_eos,
    });
    try writeU32Array(writer, method.token_ids);
    try writer.writeAll(",\"generated_text\":");
    try writeJsonString(writer, method.generated_text);
    if (include_acceptance) {
        try writer.writeAll(",\"tau\":");
        try writer.print("{d:.6}", .{methodTau(method)});
        try writer.writeAll(",\"acceptance_lengths\":");
        try writeU32Array(writer, method.acceptance_lengths);
        try writer.writeAll(",\"committed_lengths\":");
        try writeU32Array(writer, method.committed_lengths);
        try writer.writeAll(",\"acceptance_length_histogram\":");
        try writeU64Array(writer, method.acceptance_length_histogram);
        try writer.writeAll(",\"per_position_accepts\":");
        try writeU64Array(writer, method.per_position_accepts);
        try writer.writeAll(",\"per_position_trials\":");
        try writeU64Array(writer, method.per_position_trials);
    }
    try writer.writeByte('}');
}

fn writeSummaryJson(writer: *std.Io.Writer, summary: Summary) !void {
    try writer.print(
        "{{\"sample_count\":{d},\"dataset\":",
        .{summary.sample_count},
    );
    try writeJsonString(writer, summary.dataset);
    try writer.print(",\"baseline_tpot_ms\":{d:.6},\"baseline_tps\":{d:.6},\"dflash_tpot_ms\":{d:.6},\"dflash_tps\":{d:.6},\"speedup\":{d:.6},\"tau\":{d:.6},\"exact_match_count\":{d}", .{
        summary.baseline_tpot_ms,
        summary.baseline_tps,
        summary.dflash_tpot_ms,
        summary.dflash_tps,
        summary.speedup,
        summary.tau,
        summary.exact_match_count,
    });
    try writer.writeAll(",\"per_position_acceptance_rates\":");
    try writeF64Array(writer, summary.per_position_acceptance_rates);
    try writer.writeAll(",\"per_position_accepts\":");
    try writeU64Array(writer, summary.per_position_accepts);
    try writer.writeAll(",\"per_position_trials\":");
    try writeU64Array(writer, summary.per_position_trials);
    try writer.writeAll(",\"acceptance_length_histogram\":");
    try writeU64Array(writer, summary.acceptance_length_histogram);
    try writer.writeAll(",\"acceptance_length_histogram_rates\":");
    try writeF64Array(writer, summary.acceptance_length_histogram_rates);
    try writer.writeByte('}');
}

fn calcTpotMs(elapsed: Duration, tokens: usize) f64 {
    if (tokens == 0) return 0;
    return durationMs(elapsed) / @as(f64, @floatFromInt(tokens));
}

fn calcTokensPerSecond(elapsed: Duration, tokens: usize) f64 {
    if (tokens == 0 or elapsed.nanoseconds == 0) return 0;
    return @as(f64, @floatFromInt(tokens)) / (@as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s);
}

fn durationMs(elapsed: Duration) f64 {
    return @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_ms;
}

fn methodTau(method: MethodResult) f64 {
    var steps: u64 = 0;
    var total: u64 = 0;
    if (method.committed_lengths.len != 0) {
        steps = method.committed_lengths.len;
        for (method.committed_lengths) |len| total += len;
    } else if (method.acceptance_lengths.len != 0) {
        steps = method.acceptance_lengths.len;
        for (method.acceptance_lengths) |len| total += len + 1;
    } else {
        for (method.acceptance_length_histogram, 0..) |count, len| {
            steps += count;
            total += count * (len + 1);
        }
    }
    return if (steps == 0) 0 else @as(f64, @floatFromInt(total)) / @as(f64, @floatFromInt(steps));
}

fn maxPositionCount(method: MethodResult) usize {
    var max_count = @max(method.per_position_accepts.len, method.per_position_trials.len);
    if (method.acceptance_length_histogram.len != 0) {
        // Histograms are indexed by the number of accepted draft tokens. Their
        // highest index is also the number of draft positions proposed per step.
        max_count = @max(max_count, method.acceptance_length_histogram.len - 1);
    }
    if (method.acceptance_lengths.len != 0) {
        for (method.acceptance_lengths) |len| max_count = @max(max_count, len);
    } else {
        for (method.acceptance_length_histogram, 0..) |count, len| {
            if (count != 0) max_count = @max(max_count, len);
        }
    }
    return max_count;
}

fn aggregateHistogram(histogram: []u64, method: MethodResult) void {
    if (method.acceptance_lengths.len != 0) {
        for (method.acceptance_lengths) |len| histogram[len] += 1;
    } else {
        for (method.acceptance_length_histogram, 0..) |count, len| histogram[len] += count;
    }
}

fn aggregatePositions(accepts: []u64, trials: []u64, method: MethodResult) void {
    if (method.per_position_accepts.len != 0 or method.per_position_trials.len != 0) {
        for (method.per_position_accepts, 0..) |count, i| accepts[i] += count;
        for (method.per_position_trials, 0..) |count, i| trials[i] += count;
        return;
    }

    if (method.acceptance_lengths.len != 0) {
        for (method.acceptance_lengths) |len| {
            for (trials, 0..) |*trial, i| {
                trial.* += 1;
                if (i < len) accepts[i] += 1;
            }
        }
        return;
    }

    for (method.acceptance_length_histogram, 0..) |count, len| {
        if (count == 0) continue;
        for (trials, 0..) |*trial, i| {
            trial.* += count;
            if (i < len) accepts[i] += count;
        }
    }
}

fn barLen(rate: f64, width: usize) usize {
    if (rate <= 0) return 0;
    const scaled = @as(usize, @intFromFloat(@round(rate * @as(f64, @floatFromInt(width)))));
    return @min(width, @max(@as(usize, 1), scaled));
}

fn writeJsonString(writer: *std.Io.Writer, s: []const u8) !void {
    try writer.writeByte('"');
    for (s) |c| switch (c) {
        '"' => try writer.writeAll("\\\""),
        '\\' => try writer.writeAll("\\\\"),
        '\n' => try writer.writeAll("\\n"),
        '\r' => try writer.writeAll("\\r"),
        '\t' => try writer.writeAll("\\t"),
        0...0x1f => try writer.print("\\u{x:0>4}", .{c}),
        else => try writer.writeByte(c),
    };
    try writer.writeByte('"');
}

fn writeU32Array(writer: *std.Io.Writer, items: []const u32) !void {
    try writer.writeByte('[');
    for (items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writer.print("{d}", .{item});
    }
    try writer.writeByte(']');
}

fn writeU64Array(writer: *std.Io.Writer, items: []const u64) !void {
    try writer.writeByte('[');
    for (items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writer.print("{d}", .{item});
    }
    try writer.writeByte(']');
}

fn writeF64Array(writer: *std.Io.Writer, items: []const f64) !void {
    try writer.writeByte('[');
    for (items, 0..) |item, i| {
        if (i != 0) try writer.writeByte(',');
        try writer.print("{d:.6}", .{item});
    }
    try writer.writeByte(']');
}

test "compare token ids finds exact and first mismatch" {
    try std.testing.expect(compareTokenIds(&.{ 1, 2, 3 }, &.{ 1, 2, 3 }).exact_match);

    const mismatch = compareTokenIds(&.{ 1, 2, 3 }, &.{ 1, 9, 3 });
    try std.testing.expect(!mismatch.exact_match);
    try std.testing.expectEqual(@as(?usize, 1), mismatch.first_mismatch_index);

    const length_mismatch = compareTokenIds(&.{ 1, 2 }, &.{1});
    try std.testing.expect(!length_mismatch.exact_match);
    try std.testing.expectEqual(@as(?usize, 1), length_mismatch.first_mismatch_index);
}

test "summary aggregates timing and acceptance math" {
    const samples = [_]SampleResult{
        .{
            .id = "a",
            .dataset = "fixture/test",
            .prompt_tokens = 4,
            .baseline = .{
                .decoded_tokens = 4,
                .elapsed = .{ .nanoseconds = 40 * std.time.ns_per_ms },
            },
            .dflash = .{
                .decoded_tokens = 4,
                .elapsed = .{ .nanoseconds = 20 * std.time.ns_per_ms },
                .acceptance_lengths = &.{ 0, 2 },
            },
            .quality = .{ .exact_match = true, .compared_tokens = 4 },
        },
        .{
            .id = "b",
            .dataset = "fixture/test",
            .prompt_tokens = 5,
            .baseline = .{
                .decoded_tokens = 6,
                .elapsed = .{ .nanoseconds = 60 * std.time.ns_per_ms },
            },
            .dflash = .{
                .decoded_tokens = 6,
                .elapsed = .{ .nanoseconds = 30 * std.time.ns_per_ms },
                .acceptance_lengths = &.{ 1, 2 },
            },
            .quality = .{ .exact_match = false, .compared_tokens = 6, .first_mismatch_index = 3 },
        },
    };

    const allocator = std.testing.allocator;
    const summary = try computeSummary(allocator, &samples);
    defer summary.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 2), summary.sample_count);
    try std.testing.expectApproxEqAbs(@as(f64, 10.0), summary.baseline_tpot_ms, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 5.0), summary.dflash_tpot_ms, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.0), summary.speedup, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 2.25), summary.tau, 0.0001);
    try std.testing.expectEqual(@as(usize, 1), summary.exact_match_count);
    try std.testing.expectEqualSlices(u64, &.{ 1, 1, 2 }, summary.acceptance_length_histogram);
    try std.testing.expectApproxEqAbs(@as(f64, 0.75), summary.per_position_acceptance_rates[0], 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.50), summary.per_position_acceptance_rates[1], 0.0001);
}

test "summary preserves fixed draft width for zero acceptance histogram" {
    const samples = [_]SampleResult{
        .{
            .id = "zero",
            .dataset = "fixture/test",
            .prompt_tokens = 4,
            .baseline = .{
                .decoded_tokens = 1,
                .elapsed = .{ .nanoseconds = std.time.ns_per_ms },
            },
            .dflash = .{
                .decoded_tokens = 1,
                .elapsed = .{ .nanoseconds = std.time.ns_per_ms },
                .acceptance_lengths = &.{0},
                .acceptance_length_histogram = &.{ 1, 0, 0, 0, 0, 0 },
            },
            .quality = .{ .exact_match = true, .compared_tokens = 1 },
        },
    };

    const allocator = std.testing.allocator;
    const summary = try computeSummary(allocator, &samples);
    defer summary.deinit(allocator);

    try std.testing.expectEqual(@as(usize, 6), summary.acceptance_length_histogram.len);
    try std.testing.expectEqualSlices(u64, &.{ 1, 0, 0, 0, 0, 0 }, summary.acceptance_length_histogram);
    try std.testing.expectEqual(@as(usize, 5), summary.per_position_acceptance_rates.len);
    try std.testing.expectApproxEqAbs(@as(f64, 1), summary.tau, 0.0001);
    for (summary.per_position_acceptance_rates) |rate| {
        try std.testing.expectApproxEqAbs(@as(f64, 0), rate, 0.0001);
    }
    try std.testing.expectEqualSlices(u64, &.{ 1, 1, 1, 1, 1 }, summary.per_position_trials);
}

test "summary separates raw acceptance from committed tau" {
    const samples = [_]SampleResult{
        .{
            .id = "capped",
            .dataset = "fixture/test",
            .prompt_tokens = 4,
            .baseline = .{
                .decoded_tokens = 1,
                .elapsed = .{ .nanoseconds = std.time.ns_per_ms },
            },
            .dflash = .{
                .decoded_tokens = 1,
                .elapsed = .{ .nanoseconds = std.time.ns_per_ms },
                .acceptance_lengths = &.{5},
                .committed_lengths = &.{1},
                .acceptance_length_histogram = &.{ 0, 0, 0, 0, 0, 1 },
            },
            .quality = .{ .exact_match = true, .compared_tokens = 1 },
        },
    };

    const allocator = std.testing.allocator;
    const summary = try computeSummary(allocator, &samples);
    defer summary.deinit(allocator);

    try std.testing.expectApproxEqAbs(@as(f64, 1), summary.tau, 0.0001);
    try std.testing.expectEqualSlices(u64, &.{ 0, 0, 0, 0, 0, 1 }, summary.acceptance_length_histogram);
    for (summary.per_position_acceptance_rates) |rate| {
        try std.testing.expectApproxEqAbs(@as(f64, 1), rate, 0.0001);
    }
}
