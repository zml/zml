pub fn testEmbedQuantizedSearch(zml_handler: *Zml_handler, quantizer: *Quantized) !void {
    std.log.info("Test embed Quantized search", .{});

    const PartialDotStats = struct {
        label: []const u8,
        found_top1: usize = 0,
        missing_top16: usize = 0,
        total_dense_scored: usize = 0,
        total_partial_scored: usize = 0,
        total_pruned: usize = 0,
        total_pruned_by_phase: [quantized.partial_dot_block_count]usize = [_]usize{0} ** quantized.partial_dot_block_count,
        min_dense_scored: usize = std.math.maxInt(usize),
        max_dense_scored: usize = 0,
        min_partial_scored: usize = std.math.maxInt(usize),
        max_partial_scored: usize = 0,
        min_pruned: usize = std.math.maxInt(usize),
        max_pruned: usize = 0,

        fn record(self: *@This(), sample: quantized.TwoPhaseSample, top1_token: usize) void {
            self.total_dense_scored += sample.nb_scored;
            self.total_partial_scored += sample.nb_partial_dot_scored;
            self.total_pruned += sample.nb_partial_dot_pruned;
            for (sample.partial_dot_pruned_by_phase, 0..) |pruned, phase_i| {
                self.total_pruned_by_phase[phase_i] += pruned;
            }
            self.min_dense_scored = @min(self.min_dense_scored, sample.nb_scored);
            self.max_dense_scored = @max(self.max_dense_scored, sample.nb_scored);
            self.min_partial_scored = @min(self.min_partial_scored, sample.nb_partial_dot_scored);
            self.max_partial_scored = @max(self.max_partial_scored, sample.nb_partial_dot_scored);
            self.min_pruned = @min(self.min_pruned, sample.nb_partial_dot_pruned);
            self.max_pruned = @max(self.max_pruned, sample.nb_partial_dot_pruned);

            for (sample.rows) |tok| {
                if (tok == top1_token) {
                    self.found_top1 += 1;
                    return;
                }
            }
            self.missing_top16 += 1;
        }

        fn log(self: @This(), total_count: usize) void {
            const inv_total = 1.0 / @as(f64, @floatFromInt(total_count));
            const percent_found = 100.0 * @as(f64, @floatFromInt(self.found_top1)) * inv_total;
            const avg_dense_scored = @as(f64, @floatFromInt(self.total_dense_scored)) * inv_total;
            const avg_partial_scored = @as(f64, @floatFromInt(self.total_partial_scored)) * inv_total;
            const avg_pruned = @as(f64, @floatFromInt(self.total_pruned)) * inv_total;

            std.log.info("{s} found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ self.label, self.found_top1, percent_found, self.missing_top16 });
            std.log.info("{s} row-block scored: min={d} max={d} avg={d:.2}", .{ self.label, self.min_partial_scored, self.max_partial_scored, avg_partial_scored });
            std.log.info("{s} rows pruned: min={d} max={d} avg={d:.2}", .{ self.label, self.min_pruned, self.max_pruned, avg_pruned });
            std.log.info("{s} dense exact scored rows: min={d} max={d} avg={d:.2}", .{ self.label, self.min_dense_scored, self.max_dense_scored, avg_dense_scored });
            for (self.total_pruned_by_phase, 0..) |phase_pruned, phase_i| {
                std.log.info("{s} phase {d:>2} pruned: total={d} avg={d:.2}", .{ self.label, phase_i + 1, phase_pruned, @as(f64, @floatFromInt(phase_pruned)) * inv_total });
            }
        }
    };

    const repo = try zml.safetensors.resolveModelRepo(zml_handler.io, zml_handler.uris.qwen);
    var tokenizer = try llm_.Llm_handler.loadTokenizer(zml_handler, repo);
    defer tokenizer.deinit();

    var total_count: usize = 0;
    var found_top1_1bit_count: usize = 0;
    var found_top1_2bits_count: usize = 0;
    var found_top1_3phases_count: usize = 0;
    var found_top1_4phases_count: usize = 0;
    var found_top1_5phases_count: usize = 0;
    var missing_top16_1bit_count: usize = 0;
    var missing_top16_2bits_count: usize = 0;
    var missing_top16_3phases_count: usize = 0;
    var missing_top16_4phases_count: usize = 0;
    var missing_top16_5phases_count: usize = 0;
    var total_dense_scored_1bit: usize = 0;
    var total_dense_scored_2bits: usize = 0;
    var total_dense_scored_3phases: usize = 0;
    var total_dense_scored_4phases: usize = 0;
    var total_dense_scored_5phases: usize = 0;
    var total_1bit_scored_5phases: usize = 0;
    var total_half_1bit_scored_5phases: usize = 0;
    var total_2bit_scored_3phases: usize = 0;
    var total_2bit_scored_4phases: usize = 0;
    var total_2bit_scored_5phases: usize = 0;
    var min_dense_scored_1bit: usize = std.math.maxInt(usize);
    var min_dense_scored_2bits: usize = std.math.maxInt(usize);
    var min_dense_scored_3phases: usize = std.math.maxInt(usize);
    var min_dense_scored_4phases: usize = std.math.maxInt(usize);
    var min_dense_scored_5phases: usize = std.math.maxInt(usize);
    var min_1bit_scored_5phases: usize = std.math.maxInt(usize);
    var min_half_1bit_scored_5phases: usize = std.math.maxInt(usize);
    var min_2bit_scored_3phases: usize = std.math.maxInt(usize);
    var min_2bit_scored_4phases: usize = std.math.maxInt(usize);
    var min_2bit_scored_5phases: usize = std.math.maxInt(usize);
    var max_dense_scored_1bit: usize = 0;
    var max_dense_scored_2bits: usize = 0;
    var max_dense_scored_3phases: usize = 0;
    var max_dense_scored_4phases: usize = 0;
    var max_dense_scored_5phases: usize = 0;
    var max_1bit_scored_5phases: usize = 0;
    var max_half_1bit_scored_5phases: usize = 0;
    var max_2bit_scored_3phases: usize = 0;
    var max_2bit_scored_4phases: usize = 0;
    var max_2bit_scored_5phases: usize = 0;
    var partial_dot_estimator_stats: PartialDotStats = .{ .label = "partial-dot estimator" };

    const tasks_id = [5]u8{ 0, 1, 2, 3, 4 };

    for (tasks_id) |task_id| {
        const task = switch (task_id) {
            0 => "coding",
            1 => "history",
            2 => "math",
            3 => "story",
            4 => "translate",
            else => return error.InvalidTask,
        };
        const top1 = switch (task_id) {
            0 => codingTop1(),
            1 => historyTop1(),
            2 => mathTop1(),
            3 => storyTop1(),
            4 => translateTop1(),
            else => return error.InvalidTask,
        };

        const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds.safetensors", task, .f32, true);
        defer embed_slice.free(zml_handler.allocator);

        const n: usize = @intCast(embed_slice.shape.dims()[0]);
        const d: usize = @intCast(embed_slice.shape.dims()[1]);
        std.debug.assert(d == quantizer.d);

        std.log.info("Test quantized graph search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];
            const top1_token = top1[embed_index];

            const sample_1bit = try quantizer.sample2Phase1Bit(embed);
            const sample_2bits = try quantizer.sample2Phase2Bits(embed);
            const sample_3phases = try quantizer.sample3Phases(embed);
            const sample_4phases = try quantizer.sample4Phases(embed);
            const sample_5phases = try quantizer.sample5PhasesOptimized(embed);

            zml_handler.tic(&zml_handler.timers.quant_search);
            const sample_partial_dot_estimator = try quantizer.sampleMultiPhasePartialDot(embed);
            zml_handler.toc(&zml_handler.timers.quant_search);

            if (sample_1bit.nb_scored > 93340) {
                try quantizer.sample2Phase1BitLog(embed, &tokenizer);
                try quantizer.sample2Phase2BitsLog(embed, &tokenizer);
                //return;
            }
            if (sample_2bits.nb_scored > 63) {
                std.log.info("Needed {d} re scores", .{sample_2bits.nb_scored});
                //try quantizer.sample2Phase1BitLog(embed, &tokenizer);
                //try quantizer.sample2Phase2BitsLog(embed, &tokenizer);
                //return;
            }

            zml_handler.timers.nb_detokenize += 1;
            total_count += 1;

            total_dense_scored_1bit += sample_1bit.nb_scored;
            total_dense_scored_2bits += sample_2bits.nb_scored;
            total_dense_scored_3phases += sample_3phases.nb_scored;
            total_dense_scored_4phases += sample_4phases.nb_scored;
            total_dense_scored_5phases += sample_5phases.nb_scored;
            total_1bit_scored_5phases += sample_5phases.nb_1bit_scored;
            total_half_1bit_scored_5phases += sample_5phases.nb_half_1bit_scored;
            total_2bit_scored_3phases += sample_3phases.nb_2bit_scored;
            total_2bit_scored_4phases += sample_4phases.nb_2bit_scored;
            total_2bit_scored_5phases += sample_5phases.nb_2bit_scored;
            partial_dot_estimator_stats.record(sample_partial_dot_estimator, top1_token);
            min_dense_scored_1bit = @min(min_dense_scored_1bit, sample_1bit.nb_scored);
            min_dense_scored_2bits = @min(min_dense_scored_2bits, sample_2bits.nb_scored);
            min_dense_scored_3phases = @min(min_dense_scored_3phases, sample_3phases.nb_scored);
            min_dense_scored_4phases = @min(min_dense_scored_4phases, sample_4phases.nb_scored);
            min_dense_scored_5phases = @min(min_dense_scored_5phases, sample_5phases.nb_scored);
            min_1bit_scored_5phases = @min(min_1bit_scored_5phases, sample_5phases.nb_1bit_scored);
            min_half_1bit_scored_5phases = @min(min_half_1bit_scored_5phases, sample_5phases.nb_half_1bit_scored);
            min_2bit_scored_3phases = @min(min_2bit_scored_3phases, sample_3phases.nb_2bit_scored);
            min_2bit_scored_4phases = @min(min_2bit_scored_4phases, sample_4phases.nb_2bit_scored);
            min_2bit_scored_5phases = @min(min_2bit_scored_5phases, sample_5phases.nb_2bit_scored);
            max_dense_scored_1bit = @max(max_dense_scored_1bit, sample_1bit.nb_scored);
            max_dense_scored_2bits = @max(max_dense_scored_2bits, sample_2bits.nb_scored);
            max_dense_scored_3phases = @max(max_dense_scored_3phases, sample_3phases.nb_scored);
            max_dense_scored_4phases = @max(max_dense_scored_4phases, sample_4phases.nb_scored);
            max_dense_scored_5phases = @max(max_dense_scored_5phases, sample_5phases.nb_scored);
            max_1bit_scored_5phases = @max(max_1bit_scored_5phases, sample_5phases.nb_1bit_scored);
            max_half_1bit_scored_5phases = @max(max_half_1bit_scored_5phases, sample_5phases.nb_half_1bit_scored);
            max_2bit_scored_3phases = @max(max_2bit_scored_3phases, sample_3phases.nb_2bit_scored);
            max_2bit_scored_4phases = @max(max_2bit_scored_4phases, sample_4phases.nb_2bit_scored);
            max_2bit_scored_5phases = @max(max_2bit_scored_5phases, sample_5phases.nb_2bit_scored);

            var found_top1_1bit = false;
            for (sample_1bit.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_1bit = true;
                    break;
                }
            }
            if (found_top1_1bit) {
                found_top1_1bit_count += 1;
            } else {
                missing_top16_1bit_count += 1;
            }

            var found_top1_2bits = false;
            for (sample_2bits.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_2bits = true;
                    break;
                }
            }
            if (found_top1_2bits) {
                found_top1_2bits_count += 1;
            } else {
                //try quantizer.sample2Phase2BitsLog(embed, &tokenizer);
                missing_top16_2bits_count += 1;
            }

            var found_top1_3phases = false;
            for (sample_3phases.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_3phases = true;
                    break;
                }
            }
            if (found_top1_3phases) {
                found_top1_3phases_count += 1;
            } else {
                missing_top16_3phases_count += 1;
            }

            var found_top1_4phases = false;
            for (sample_4phases.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_4phases = true;
                    break;
                }
            }
            if (found_top1_4phases) {
                found_top1_4phases_count += 1;
            } else {
                missing_top16_4phases_count += 1;
            }

            var found_top1_5phases = false;
            for (sample_5phases.rows) |tok| {
                if (tok == top1_token) {
                    found_top1_5phases = true;
                    break;
                }
            }
            if (found_top1_5phases) {
                found_top1_5phases_count += 1;
            } else {
                missing_top16_5phases_count += 1;
            }
        }
    }

    const inv_total = 1.0 / @as(f64, @floatFromInt(total_count));
    const percent_found_1bit = 100.0 * @as(f64, @floatFromInt(found_top1_1bit_count)) * inv_total;
    const percent_found_2bits = 100.0 * @as(f64, @floatFromInt(found_top1_2bits_count)) * inv_total;
    const percent_found_3phases = 100.0 * @as(f64, @floatFromInt(found_top1_3phases_count)) * inv_total;
    const percent_found_4phases = 100.0 * @as(f64, @floatFromInt(found_top1_4phases_count)) * inv_total;
    const percent_found_5phases = 100.0 * @as(f64, @floatFromInt(found_top1_5phases_count)) * inv_total;
    const avg_dense_scored_1bit = @as(f64, @floatFromInt(total_dense_scored_1bit)) * inv_total;
    const avg_dense_scored_2bits = @as(f64, @floatFromInt(total_dense_scored_2bits)) * inv_total;
    const avg_dense_scored_3phases = @as(f64, @floatFromInt(total_dense_scored_3phases)) * inv_total;
    const avg_dense_scored_4phases = @as(f64, @floatFromInt(total_dense_scored_4phases)) * inv_total;
    const avg_dense_scored_5phases = @as(f64, @floatFromInt(total_dense_scored_5phases)) * inv_total;
    const avg_1bit_scored_5phases = @as(f64, @floatFromInt(total_1bit_scored_5phases)) * inv_total;
    const avg_half_1bit_scored_5phases = @as(f64, @floatFromInt(total_half_1bit_scored_5phases)) * inv_total;
    const avg_2bit_scored_3phases = @as(f64, @floatFromInt(total_2bit_scored_3phases)) * inv_total;
    const avg_2bit_scored_4phases = @as(f64, @floatFromInt(total_2bit_scored_4phases)) * inv_total;
    const avg_2bit_scored_5phases = @as(f64, @floatFromInt(total_2bit_scored_5phases)) * inv_total;
    std.log.info("Embed quantized search: total={d}", .{total_count});
    std.log.info("1-bit found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_1bit_count, percent_found_1bit, missing_top16_1bit_count });
    std.log.info("2-bit found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_2bits_count, percent_found_2bits, missing_top16_2bits_count });
    std.log.info("3-phase found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_3phases_count, percent_found_3phases, missing_top16_3phases_count });
    std.log.info("4-phase found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_4phases_count, percent_found_4phases, missing_top16_4phases_count });
    std.log.info("5-phase found_top1_in_top16={d} ({d:.4}%) missing_top16={d}", .{ found_top1_5phases_count, percent_found_5phases, missing_top16_5phases_count });
    partial_dot_estimator_stats.log(total_count);
    std.log.info("1-bit dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_1bit, max_dense_scored_1bit, avg_dense_scored_1bit });
    std.log.info("2-bit dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_2bits, max_dense_scored_2bits, avg_dense_scored_2bits });
    std.log.info("3-phase 2-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_2bit_scored_3phases, max_2bit_scored_3phases, avg_2bit_scored_3phases });
    std.log.info("3-phase dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_3phases, max_dense_scored_3phases, avg_dense_scored_3phases });
    std.log.info("4-phase 2-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_2bit_scored_4phases, max_2bit_scored_4phases, avg_2bit_scored_4phases });
    std.log.info("4-phase dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_4phases, max_dense_scored_4phases, avg_dense_scored_4phases });
    std.log.info("5-phase half 1-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_half_1bit_scored_5phases, max_half_1bit_scored_5phases, avg_half_1bit_scored_5phases });
    std.log.info("5-phase full 1-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_1bit_scored_5phases, max_1bit_scored_5phases, avg_1bit_scored_5phases });
    std.log.info("5-phase 2-bit approx scored rows: min={d} max={d} avg={d:.2}", .{ min_2bit_scored_5phases, max_2bit_scored_5phases, avg_2bit_scored_5phases });
    std.log.info("5-phase dense exact scored rows: min={d} max={d} avg={d:.2}", .{ min_dense_scored_5phases, max_dense_scored_5phases, avg_dense_scored_5phases });
}

pub fn testEmbedGraphQuantizedSearch(zml_handler: *Zml_handler, g: *Graph, quantizer: *Quantized) !void {
    std.log.info("Test embed GraphQuantized search", .{});

    var total_count: usize = 0;
    var found_top1: [4]usize = [_]usize{0} ** 4;
    var found_top1_acc: [4]usize = [_]usize{0} ** 4;
    var total_visited: usize = 0;
    var min_visited: usize = std.math.maxInt(usize);
    var max_visited: usize = 0;

    const tasks_id = [5]u8{ 0, 1, 2, 3, 4 };

    for (tasks_id) |task_id| {
        const task = switch (task_id) {
            0 => "coding",
            1 => "history",
            2 => "math",
            3 => "story",
            4 => "translate",
            else => return error.InvalidTask,
        };
        const top1 = switch (task_id) {
            0 => codingTop1(),
            1 => historyTop1(),
            2 => mathTop1(),
            3 => storyTop1(),
            4 => translateTop1(),
            else => return error.InvalidTask,
        };

        const embed_slice = try save_load.getSlice(zml_handler, "qwen_embeds.safetensors", task, .f32, true);
        defer embed_slice.free(zml_handler.allocator);

        const n: usize = @intCast(embed_slice.shape.dims()[0]);
        const d: usize = @intCast(embed_slice.shape.dims()[1]);
        std.debug.assert(d == quantizer.d);

        std.log.info("Test quantized graph search task={s} embeddings={d} shape={f}", .{ task, n, embed_slice.shape });
        zml_handler.tic(&zml_handler.timers.graph_search_tot);
        for (0..n) |embed_index| {
            const embed = embed_slice.constItems(f32)[embed_index * d .. (embed_index + 1) * d];

            zml_handler.timers.nb_detokenize += 1;
            const top1_token = top1[embed_index];
            var scored: usize = 0;
            var found_acc = false;
            for (0..4) |start| {
                const entry_point: u32 = @as(u32, @intCast(start)) * (g.n / 4);
                g.greedySearchPrefetch(embed, entry_point);
                scored += g.nb_scored;
                var found = false;
                for (0..g.L) |i| {
                    if (g.visited[i].node == top1_token) {
                        found = true;
                        break;
                    }
                }
                found_acc |= found;
                if (found) found_top1[start] += 1;
                if (found_acc) found_top1_acc[start] += 1;
            }

            const nb_visited = scored;
            total_visited += nb_visited;
            min_visited = @min(min_visited, nb_visited);
            max_visited = @max(max_visited, nb_visited);
            total_count += 1;

            //g.params.search_budget = 2048;
            //try g.greedySearchQuantized2x2(embed, quantizer);

            //g.params.search_budget = 1024;
            //g.quantizedCrossover(embed);
        }
        zml_handler.toc(&zml_handler.timers.graph_search_tot);
    }

    const average_visit = @as(f64, @floatFromInt(total_visited)) / @as(f64, @floatFromInt(total_count));
    std.log.info("Embed quantized search: total={d}", .{total_count});
    for (0..4) |start| {
        const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1[start])) / @as(f64, @floatFromInt(total_count));
        std.log.info("Pass {d} found_top1={d} ({d:.4}%)", .{ start, found_top1[start], percent_found });
    }
    for (0..4) |start| {
        const percent_found = 100.0 * @as(f64, @floatFromInt(found_top1_acc[start])) / @as(f64, @floatFromInt(total_count));
        std.log.info("Pass up to {d} found_top1_acc={d} ({d:.4}%)", .{ start, found_top1_acc[start], percent_found });
    }
    std.log.info("Embed graph search nb_visited: min={d} max={d} avg={d:.2}", .{ min_visited, max_visited, average_visit });
}

const std = @import("std");
const zml = @import("zml");
const main = @import("main.zig");
const tokens = @import("tokens.zig");
const algebra = @import("algebra.zig");

const LmHeadMatrix = algebra.LmHeadMatrix;
const Tokenizer = zml.tokenizer.Tokenizer;

pub const Logit = struct {
    index: usize,
    logit_f32: f32,
    logit_i8: f32,
    logit_sliced: f32,
    logit_qjl: f32,
    lower_bound: f32,
    upper_bound: f32,
    proba: f32,
    quantized_rank: usize,
    sliced_rank: usize,
    qjl_rank: usize,

    pub fn decreasingOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_f32 > rhs.logit_f32) or (lhs.logit_f32 == rhs.logit_f32 and lhs.index < rhs.index);
    }

    pub fn decreasingQuantizedOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_i8 > rhs.logit_i8) or (lhs.logit_i8 == rhs.logit_i8 and lhs.index < rhs.index);
    }

    pub fn decreasingSlicedOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_sliced > rhs.logit_sliced) or (lhs.logit_sliced == rhs.logit_sliced and lhs.index < rhs.index);
    }

    pub fn decreasingQjlOrder(_: void, lhs: Logit, rhs: Logit) bool {
        return (lhs.logit_qjl > rhs.logit_qjl) or (lhs.logit_qjl == rhs.logit_qjl and lhs.index < rhs.index);
    }

    fn boundViolation(logit: Logit) f32 {
        if (logit.logit_f32 < logit.lower_bound) return logit.lower_bound - logit.logit_f32;
        if (logit.logit_f32 > logit.upper_bound) return logit.logit_f32 - logit.upper_bound;
        return 0.0;
    }

    fn boundMargin(logit: Logit) f32 {
        return @min(@abs(logit.logit_f32 - logit.lower_bound), @abs(logit.upper_bound - logit.logit_f32));
    }

    pub fn boundPressureOrder(_: void, lhs: Logit, rhs: Logit) bool {
        const lhs_violation = boundViolation(lhs);
        const rhs_violation = boundViolation(rhs);
        if (lhs_violation != rhs_violation) return lhs_violation > rhs_violation;

        const lhs_margin = boundMargin(lhs);
        const rhs_margin = boundMargin(rhs);
        return lhs_margin < rhs_margin or (lhs_margin == rhs_margin and lhs.index < rhs.index);
    }
};

pub const top_k_sliced: usize = 16;
pub const default_syntax_token_id_threshold: usize = 2048;
pub const qjl_micro_coord_count: usize = 1024;
pub const qjl_micro_u128_word_count: usize = qjl_micro_coord_count / 128;
pub const qjl_half_coord_count: usize = 2 * qjl_micro_coord_count;
pub const qjl_half_to_full_coord_count: usize = 4096 - qjl_half_coord_count;
pub const qjl_half_to_full_u128_word_count: usize = qjl_half_to_full_coord_count / 128;
pub const partial_dot_block_size: usize = 128;
pub const partial_dot_block_count: usize = 4096 / partial_dot_block_size;
pub const partial_dot_phase_exact_top_k: usize = 16;
pub const partial_dot_prefix_coord_count: usize = 128;

pub const Quantized = struct {
    zml_handler: *main.Zml_handler,
    allocator: std.mem.Allocator,
    lm_head: *LmHeadMatrix,

    vocab_size: usize,
    d: usize,
    d_inv_norm: f32,

    lm_head_quantized: []i8,
    row_scale: []f32,
    row_scale_sliced: []f32,
    f32_buffer: []f32,
    dense_logits: []f32,
    partial_logits: []f32,
    partial_dot_residual_logits: []f32,
    active_rows: []u32,
    i8_buffer: []i8,

    lm_head_sliced: []vector_U4096,
    row_offset: []i32,

    lm_head_qjl: []U4096,
    qjl_micro_rows: [][qjl_micro_u128_word_count]u128,
    qjl_micro_to_half_rows: [][qjl_micro_u128_word_count]u128,
    qjl_half_to_full_rows: [][qjl_half_to_full_u128_word_count]u128,
    qjl_micro_mismatches: []u16,
    qjl_row_scale: []f32,
    qjl_query_lut: []f32,
    qjl_inv_dim: f32 = 0.0,

    partial_dot_blocks: []f32,

    lm_head_qjl_2bits: [][64]u128,
    qjl_row_scale_2bits: []f32,

    pub fn init(zml_handler: *main.Zml_handler, lm_head: *LmHeadMatrix) !Quantized {
        const allocator = zml_handler.allocator;

        const vocab_size: usize = lm_head.n;
        const d: usize = lm_head.d;
        std.debug.assert(d == 4096);

        const d_inv_norm: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(d)));

        const f32_buffer = try allocator.alloc(f32, d);
        const dense_logits = try allocator.alloc(f32, vocab_size);
        const partial_logits = try allocator.alloc(f32, vocab_size);
        const partial_dot_residual_logits = try allocator.alloc(f32, vocab_size);
        const active_rows = try allocator.alloc(u32, vocab_size);

        const lm_head_quantized = try allocator.alloc(i8, d * vocab_size);
        const lm_head_sliced = try allocator.alloc(vector_U4096, vocab_size);

        const row_offset = try allocator.alloc(i32, vocab_size);
        const row_scale = try allocator.alloc(f32, vocab_size);
        const row_scale_sliced = try allocator.alloc(f32, vocab_size);
        const i8_buffer = try allocator.alloc(i8, d);

        const lm_head_qjl = try allocator.alloc(U4096, vocab_size);
        const qjl_micro_rows = try allocator.alloc([qjl_micro_u128_word_count]u128, vocab_size);
        const qjl_micro_to_half_rows = try allocator.alloc([qjl_micro_u128_word_count]u128, vocab_size);
        const qjl_half_to_full_rows = try allocator.alloc([qjl_half_to_full_u128_word_count]u128, vocab_size);
        const qjl_micro_mismatches = try allocator.alloc(u16, vocab_size);
        const qjl_row_scale = try allocator.alloc(f32, vocab_size);
        const qjl_query_lut = try allocator.alloc(f32, 512 * 256);
        const partial_dot_blocks = try allocator.alloc(f32, vocab_size * d);
        const lm_head_qjl_2bits = try allocator.alloc([64]u128, vocab_size);
        const qjl_row_scale_2bits = try allocator.alloc(f32, vocab_size);

        return .{
            .zml_handler = zml_handler,
            .allocator = allocator,
            .lm_head = lm_head,
            .vocab_size = vocab_size,
            .d = d,
            .d_inv_norm = d_inv_norm,
            .row_scale = row_scale,
            .row_scale_sliced = row_scale_sliced,
            .f32_buffer = f32_buffer,
            .dense_logits = dense_logits,
            .partial_logits = partial_logits,
            .partial_dot_residual_logits = partial_dot_residual_logits,
            .active_rows = active_rows,
            .lm_head_quantized = lm_head_quantized,
            .lm_head_sliced = lm_head_sliced,
            .row_offset = row_offset,
            .i8_buffer = i8_buffer,
            .lm_head_qjl = lm_head_qjl,
            .qjl_micro_rows = qjl_micro_rows,
            .qjl_micro_to_half_rows = qjl_micro_to_half_rows,
            .qjl_half_to_full_rows = qjl_half_to_full_rows,
            .qjl_micro_mismatches = qjl_micro_mismatches,
            .qjl_row_scale = qjl_row_scale,
            .qjl_query_lut = qjl_query_lut,
            .partial_dot_blocks = partial_dot_blocks,
            .lm_head_qjl_2bits = lm_head_qjl_2bits,
            .qjl_row_scale_2bits = qjl_row_scale_2bits,
        };
    }

    pub fn deinit(self: *Quantized) void {
        self.allocator.free(self.row_scale);
        self.allocator.free(self.row_scale_sliced);
        self.allocator.free(self.f32_buffer);
        self.allocator.free(self.dense_logits);
        self.allocator.free(self.partial_logits);
        self.allocator.free(self.partial_dot_residual_logits);
        self.allocator.free(self.active_rows);
        self.allocator.free(self.lm_head_quantized);
        self.allocator.free(self.lm_head_sliced);
        self.allocator.free(self.row_offset);
        self.allocator.free(self.i8_buffer);
        self.allocator.free(self.lm_head_qjl);
        self.allocator.free(self.qjl_micro_rows);
        self.allocator.free(self.qjl_micro_to_half_rows);
        self.allocator.free(self.qjl_half_to_full_rows);
        self.allocator.free(self.qjl_micro_mismatches);
        self.allocator.free(self.qjl_row_scale);
        self.allocator.free(self.qjl_query_lut);
        self.allocator.free(self.partial_dot_blocks);
        self.allocator.free(self.lm_head_qjl_2bits);
        self.allocator.free(self.qjl_row_scale_2bits);
    }

    
    inline fn insertTopK(comptime k: usize, row: usize, score: f32, rows: *[k]usize, scores: *[k]f32) void {
        if (score <= scores[k - 1]) return;

        var insert_pos = k - 1;
        while (insert_pos > 0 and score > scores[insert_pos - 1]) {
            scores[insert_pos] = scores[insert_pos - 1];
            rows[insert_pos] = rows[insert_pos - 1];
            insert_pos -= 1;
        }
        scores[insert_pos] = score;
        rows[insert_pos] = row;
    }

    inline fn insertTop16(row: usize, score: f32, rows: *[top_k_sliced]usize, scores: *[top_k_sliced]f32) void {
        insertTopK(top_k_sliced, row, score, rows, scores);
    }

    inline fn insertTop8(row: usize, score: f32, rows: *[8]usize, scores: *[8]f32) void {
        insertTopK(8, row, score, rows, scores);
    }

    inline fn inTop16(row: usize, rows: *const [top_k_sliced]usize) bool {
        for (rows) |top_row| {
            if (top_row == row) return true;
        }
        return false;
    }

    inline fn resetTopK(comptime k: usize, rows: *[k]usize, scores: *[k]f32) void {
        rows.* = [_]usize{0} ** k;
        scores.* = [_]f32{-std.math.inf(f32)} ** k;
    }

    inline fn exactScorePhaseTopK(
        comptime k: usize,
        self: *Quantized,
        query: []const f32,
        phase_rows: *const [k]usize,
        phase_scores: *const [k]f32,
        top_rows: *[top_k_sliced]usize,
        top_scores: *[top_k_sliced]f32,
        best_lower_bound: *f32,
        nb_scored: *usize,
    ) void {
        self.zml_handler.tic(&self.zml_handler.timers.quant_5p_dense_top8);
        defer self.zml_handler.toc(&self.zml_handler.timers.quant_5p_dense_top8);
        for (0..k) |top_i| {
            if (phase_scores[top_i] == -std.math.inf(f32)) break;
            const row_i = phase_rows[top_i];
            if (self.lm_head.is_junk[row_i]) continue;
            if (self.dense_logits[row_i] == -std.math.inf(f32)) continue;
            const row = self.lm_head.data[row_i * self.d ..][0..self.d];
            const real = self.realLogit(query, row);
            nb_scored.* += 1;
            best_lower_bound.* = @max(best_lower_bound.*, real);
            insertTop16(row_i, real, top_rows, top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }
    }
    
    
    pub fn sample2Phase1Bit(self: *Quantized, query: []const f32) !TwoPhaseSample {
        var approx_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var approx_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = approx_rows, .nb_scored = 0 };

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(query);
        const z_score: f32 = 5.0;
        const err_const: f32 = 0.92;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score every row with the 1-bit QJL approximation and store an upper bound.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            const lower_bound = approx - err;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, lower_bound);
            insertTop16(i, approx, &approx_rows, &approx_scores);
        }

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        var nb_scored: usize = 0;
        var prune_threshold = best_lower_bound;

        // Phase 2a: exact-score the approximate top16 to tighten the pruning threshold.
        i = 0;
        while (i < top_k_sliced) : (i += 1) {
            const row_i = approx_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 2b: exact-score only rows whose 1-bit upper bound can still beat the threshold.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < prune_threshold) continue;
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(i, real, &top_rows, &top_scores);
        }
        return .{ .rows = top_rows, .nb_scored = nb_scored };
    }

    pub fn sample2Phase2Bits(self: *Quantized, query: []const f32) !TwoPhaseSample {
        var approx_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var approx_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = approx_rows, .nb_scored = 0 };

        try self.walshHadamardSimd(query, 12);
        var query_l2_norm: f32 = 0.0;
        for (self.f32_buffer) |v| {
            query_l2_norm += v * v;
        }
        query_l2_norm = @sqrt(query_l2_norm);

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        const z_score: f32 = 5.0;
        const err_const: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score every row with the 2-bit QJL approximation and store an upper bound.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(i, &query_msb, &query_lsb));
            const approx = query_scale * self.qjl_row_scale_2bits[i] * dot;
            const err = z_score * err_const * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            const lower_bound = approx - err;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, lower_bound);
            insertTop16(i, approx, &approx_rows, &approx_scores);
        }

        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        var nb_scored: usize = 0;
        var prune_threshold = best_lower_bound;

        // Phase 2a: exact-score the approximate top16 to tighten the pruning threshold.
        i = 0;
        while (i < top_k_sliced) : (i += 1) {
            const row_i = approx_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 2b: exact-score only rows whose 2-bit upper bound can still beat the threshold.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < prune_threshold) continue;
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            prune_threshold = @max(prune_threshold, real);
            insertTop16(i, real, &top_rows, &top_scores);
        }
        return .{ .rows = top_rows, .nb_scored = nb_scored };
    }

    pub fn sample3Phases(self: *Quantized, query: []const f32) !TwoPhaseSample {
        var phase1_rows: [8]usize = [_]usize{0} ** 8;
        var phase1_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = top_rows, .nb_scored = 0, .nb_2bit_scored = 0 };

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(query);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score all rows with full 1-bit QJL and keep only loose upper bounds.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const_1bit * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase1_rows, &phase1_scores);
        }

        var nb_scored: usize = 0;
        // Phase 1b: exact-score the 1-bit top8 to raise the global lower bound.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase1_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase1_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        var phase2_rows: [8]usize = [_]usize{0} ** 8;
        var phase2_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var nb_2bit_scored: usize = 0;

        // Phase 2: rescore rows surviving the 1-bit bound with the tighter 2-bit QJL approximation.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) {
                self.dense_logits[i] = -std.math.inf(f32);
                continue;
            }

            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(i, &query_msb, &query_lsb));
            const approx = query_scale * self.qjl_row_scale_2bits[i] * dot;
            const err = z_score * err_const_2bits * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase2_rows, &phase2_scores);
        }

        // Phase 2b: exact-score the 2-bit top8 to tighten the final dense threshold.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase2_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase2_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 3: exact-score the remaining rows whose 2-bit upper bound still survives.
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) continue;
            const row = self.lm_head.data[i * self.d .. (i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(i, real, &top_rows, &top_scores);
        }

        return .{ .rows = top_rows, .nb_scored = nb_scored, .nb_2bit_scored = nb_2bit_scored };
    }

    pub fn sample4Phases(self: *Quantized, query: []const f32) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);
        const prefix_coord_count = 4;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;
        var phase1_rows: [8]usize = [_]usize{0} ** 8;
        var phase1_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{ .rows = top_rows, .nb_scored = 0, .nb_2bit_scored = 0 };

        var coord: usize = 0;
        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        while (coord < self.d) : (coord += 1) {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }

        @memset(self.partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);

        var prefix_i: usize = 0;
        // Phase 0b: compute the exact lm_head dot contribution from the raw coordinate prefix.
        while (prefix_i < prefix_coord_count) : (prefix_i += 1) {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = self.lm_head.data_t[query_coord * self.vocab_size ..][0..self.vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= self.vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = self.partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                self.partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < self.vocab_size) : (row_i += 1) {
                self.partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }

        // Phase 0c: zero the exact prefix coordinates before quantizing the residual query.
        @memcpy(self.f32_buffer[0..self.d], query);
        prefix_i = 0;
        while (prefix_i < prefix_coord_count) : (prefix_i += 1) {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(self.f32_buffer[0..self.d]);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 1: score all rows as exact prefix plus full 1-bit QJL residual bounds.
        var i: usize = 0;
        while (i < self.vocab_size) : (i += 1) {
            const approx = self.partial_logits[i] + self.dotQjlSymmetric(i, quantized_query, query_l2_norm);
            const err = z_score * err_const_1bit * query_l2_norm * self.lm_head.row_norms[i] * self.d_inv_norm;
            self.dense_logits[i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(i, approx, &phase1_rows, &phase1_scores);
        }

        var nb_scored: usize = 0;
        // Phase 1b: exact-score the 1-bit top8 to tighten the lower bound.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase1_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase1_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 1c: compact rows whose 1-bit upper bound survives into the 2-bit phase.
        var active_count: usize = 0;
        i = 0;
        while (i < self.vocab_size) : (i += 1) {
            if (self.dense_logits[i] < best_lower_bound) {
                self.dense_logits[i] = -std.math.inf(f32);
                continue;
            }
            self.active_rows[active_count] = @intCast(i);
            active_count += 1;
        }

        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        var phase2_rows: [8]usize = [_]usize{0} ** 8;
        var phase2_scores: [8]f32 = [_]f32{-std.math.inf(f32)} ** 8;
        var nb_2bit_scored: usize = 0;

        // Phase 2: rescore surviving rows as exact prefix plus 2-bit QJL residual bounds.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(row_i, &query_msb, &query_lsb));
            const approx = self.partial_logits[row_i] + query_scale * self.qjl_row_scale_2bits[row_i] * dot;
            const err = z_score * err_const_2bits * query_l2_norm * self.lm_head.row_norms[row_i] * self.d_inv_norm;
            self.dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTop8(row_i, approx, &phase2_rows, &phase2_scores);
        }

        // Phase 2b: exact-score the 2-bit top8 to tighten the final dense threshold.
        i = 0;
        while (i < 8) : (i += 1) {
            if (phase2_scores[i] == -std.math.inf(f32)) break;
            const row_i = phase2_rows[i];
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
            self.dense_logits[row_i] = -std.math.inf(f32);
        }

        // Phase 3: exact-score remaining rows whose 2-bit upper bound still survives.
        i = 0;
        while (i < active_count) : (i += 1) {
            const row_i: usize = @intCast(self.active_rows[i]);
            if (self.dense_logits[row_i] < best_lower_bound) continue;
            const row = self.lm_head.data[row_i * self.d .. (row_i + 1) * self.d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }

        return .{ .rows = top_rows, .nb_scored = nb_scored, .nb_2bit_scored = nb_2bit_scored };
    }

    pub fn sample5Phases(self: *Quantized, query: []const f32) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);
        const prefix_coord_count = 12;
        const phase_exact_top_k = 16;
        const vocab_size = self.vocab_size;
        const d = self.d;
        const lm_head = self.lm_head;
        const lm_data_t = lm_head.data_t;
        const row_norms = lm_head.row_norms;
        const partial_logits = self.partial_logits;
        const dense_logits = self.dense_logits;
        const active_rows = self.active_rows;
        const qjl_micro_rows = self.qjl_micro_rows;
        const qjl_micro_to_half_rows = self.qjl_micro_to_half_rows;
        const qjl_micro_mismatches = self.qjl_micro_mismatches;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;
        var phase_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;

        if (true) return .{
            .rows = top_rows,
            .nb_scored = 0,
            .nb_1bit_scored = 0,
            .nb_half_1bit_scored = 0,
            .nb_2bit_scored = 0,
        };
        
        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prefix_top);
        for (0..d) |coord| {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prefix_top);

        // Phase 0b: compute the exact lm_head dot contribution from the raw coordinate prefix.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prefix_dot);
        @memset(partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = lm_data_t[query_coord * vocab_size ..][0..vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < vocab_size) : (row_i += 1) {
                partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prefix_dot);

        // Phase 0c: zero the exact prefix coordinates before quantizing the residual query.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_quantize);
        @memcpy(self.f32_buffer[0..d], query);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        const quantized_query, _, const query_l2_norm = try self.quantizeQjlVector(self.f32_buffer[0..d]);
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_quantize);
        const z_score: f32 = 5.0;
        const err_const_1bit: f32 = 0.92;
        const err_const_2bits: f32 = 0.49;
        const qjl_micro_err_scale: f32 = @sqrt(@as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(qjl_micro_coord_count)));
        const qjl_half_err_scale: f32 = @sqrt(@as(f32, @floatFromInt(d)) / @as(f32, @floatFromInt(qjl_half_coord_count)));
        const base_err_scale_1bit = z_score * err_const_1bit * query_l2_norm * self.d_inv_norm;
        const err_scale_micro = base_err_scale_1bit * qjl_micro_err_scale;
        const err_scale_half = base_err_scale_1bit * qjl_half_err_scale;
        const err_scale_2bits = z_score * err_const_2bits * query_l2_norm * self.d_inv_norm;
        var best_lower_bound: f32 = -std.math.inf(f32);

        // Phase 0.5: score the first 1024 rotated sign bits with widened 1-bit bounds.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_micro);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        const query_micro_words = qjlMicroWords(&quantized_query);
        const query_micro_vec: @Vector(qjl_micro_u128_word_count, u128) = query_micro_words.*;
        for (0..vocab_size) |row_i| {
            if (lm_head.is_junk[row_i]) {
                dense_logits[row_i] = -std.math.inf(f32);
                continue;
            }
            const row_norm = row_norms[row_i];

            const row_words = &qjl_micro_rows[row_i];
            var mismatch_count_u16: u16 = 0;
            mismatch_count_u16 += qjlMismatchU128Loaded(qjl_micro_u128_word_count, row_words, query_micro_vec);
            qjl_micro_mismatches[row_i] = mismatch_count_u16;

            const approx = partial_logits[row_i] + qjlPartialSignDot(query_l2_norm, row_norm, mismatch_count_u16, qjl_micro_coord_count);
            const err = err_scale_micro * row_norm;
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_micro);

        var nb_scored: usize = 0;
        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 0.5 prune: compact rows whose micro upper bound survives.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prune);
        var active_count: usize = 0;
        for (0..vocab_size) |row_i| {
            active_rows[active_count] = @intCast(row_i);
            active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prune);

        // Phase 0.75: score the next 1024 QJL bits, reusing the micro mismatch counts.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_half);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        const query_micro_to_half_words = qjlMicroToHalfWords(&quantized_query);
        const query_micro_to_half_vec: @Vector(qjl_micro_u128_word_count, u128) = query_micro_to_half_words.*;
        var nb_half_1bit_scored: usize = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            nb_half_1bit_scored += 1;

            const row_words = &qjl_micro_to_half_rows[row_i];
            var mismatch_count_u16: u16 = qjl_micro_mismatches[row_i];
            mismatch_count_u16 += qjlMismatchU128Loaded(qjl_micro_u128_word_count, row_words, query_micro_to_half_vec);
            qjl_micro_mismatches[row_i] = mismatch_count_u16;

            const row_norm = row_norms[row_i];
            const approx = partial_logits[row_i] + qjlPartialSignDot(query_l2_norm, row_norm, mismatch_count_u16, qjl_half_coord_count);
            const err = err_scale_half * row_norm;
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_half);

        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 0.75 prune: compact rows whose half-dot upper bound survives.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prune);
        var next_active_count: usize = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            active_rows[next_active_count] = row_u32;
            next_active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
        }
        active_count = next_active_count;
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prune);

        // Phase 1: rescore survivors with full 1-bit QJL, reusing the first 2048-bit popcount.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_full_1bit);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        var nb_1bit_scored: usize = 0;
        const query_half_to_full_words = qjlHalfToFullWords(&quantized_query);
        const query_half_to_full_vec: @Vector(qjl_half_to_full_u128_word_count, u128) = query_half_to_full_words.*;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            nb_1bit_scored += 1;
            var mismatch_count_u16 = qjl_micro_mismatches[row_i];
            mismatch_count_u16 += qjlMismatchU128Loaded(
                qjl_half_to_full_u128_word_count,
                &self.qjl_half_to_full_rows[row_i],
                query_half_to_full_vec,
            );
            const approx = partial_logits[row_i] + self.dotQjlSymmetricFromMismatchCount(row_i, query_l2_norm, mismatch_count_u16);
            const err = base_err_scale_1bit * row_norms[row_i];
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_full_1bit);

        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 1 prune: compact rows whose full 1-bit upper bound survives into 2-bit scoring.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_prune);
        next_active_count = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            active_rows[next_active_count] = row_u32;
            next_active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
        }
        active_count = next_active_count;
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_prune);

        // Phase 2: score survivors with the tighter 2-bit QJL residual approximation.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_2bit);
        const query_msb, const query_lsb = self.quantize2BitsSymmetric(query_l2_norm);
        const query_scale = scaleFactor2BitsSymmetric(self.f32_buffer[0..4096], &query_msb, &query_lsb);
        resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
        var nb_2bit_scored: usize = 0;
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            nb_2bit_scored += 1;
            const dot: f32 = @floatFromInt(self.dotQjlSymmetric2Bits(row_i, &query_msb, &query_lsb));
            const approx = partial_logits[row_i] + query_scale * self.qjl_row_scale_2bits[row_i] * dot;
            const err = err_scale_2bits * row_norms[row_i];
            dense_logits[row_i] = approx + err;
            best_lower_bound = @max(best_lower_bound, approx - err);
            insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_2bit);

        exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

        // Phase 3: exact-score remaining rows whose 2-bit upper bound still survives.
        //self.zml_handler.tic(&self.zml_handler.timers.quant_5p_dense_final);
        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            if (dense_logits[row_i] < best_lower_bound) continue;
            const row = lm_head.data[row_i * d ..][0..d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }
        //self.zml_handler.toc(&self.zml_handler.timers.quant_5p_dense_final);

        return .{
            .rows = top_rows,
            .nb_scored = nb_scored,
            .nb_1bit_scored = nb_1bit_scored,
            .nb_half_1bit_scored = nb_half_1bit_scored,
            .nb_2bit_scored = nb_2bit_scored,
        };
    }

    pub fn samplePerBlock(self: *Quantized, query: []const f32) !TwoPhaseSample {
        std.debug.assert(query.len == self.d);

        const phase_exact_top_k = partial_dot_phase_exact_top_k;
        const vocab_size = self.vocab_size;
        const d = self.d;
        const lm_head = self.lm_head;
        const lm_data = lm_head.data;
        const lm_data_t = lm_head.data_t;
        const is_junk = lm_head.is_junk;
        const active_rows = self.active_rows;
        const partial_logits = self.partial_logits;
        const residual_logits = self.partial_dot_residual_logits;
        const dense_logits = self.dense_logits;
        const prefix_coord_count = partial_dot_prefix_coord_count;

        var prefix_coords: [prefix_coord_count]usize = [_]usize{0} ** prefix_coord_count;
        var prefix_scores: [prefix_coord_count]f32 = [_]f32{-std.math.inf(f32)} ** prefix_coord_count;

        // Phase 0a: find the largest raw query coordinates for exact prefix scoring.
        for (0..d) |coord| {
            insertTopK(prefix_coord_count, coord, @abs(query[coord]), &prefix_coords, &prefix_scores);
        }

        // Phase 0b: compute the exact lm_head contribution of the raw coordinate prefix.
        @memset(partial_logits, 0.0);
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            const query_coord = prefix_coords[prefix_i];
            const query_value = query[query_coord];
            const coord_values = lm_data_t[query_coord * vocab_size ..][0..vocab_size];
            const query_vec: Vec = @splat(query_value);

            var row_i: usize = 0;
            while (row_i + simd_len <= vocab_size) : (row_i += simd_len) {
                const coord_vec: Vec = coord_values[row_i..][0..simd_len].*;
                var logits_vec: Vec = partial_logits[row_i..][0..simd_len].*;
                logits_vec = @mulAdd(Vec, coord_vec, query_vec, logits_vec);
                partial_logits[row_i..][0..simd_len].* = logits_vec;
            }
            while (row_i < vocab_size) : (row_i += 1) {
                partial_logits[row_i] += coord_values[row_i] * query_value;
            }
        }

        // Phase 0c: zero the exact prefix coordinates before residual rotation.
        @memcpy(self.f32_buffer[0..d], query);
        for (0..prefix_coord_count) |prefix_i| {
            if (prefix_scores[prefix_i] == -std.math.inf(f32)) break;
            self.f32_buffer[prefix_coords[prefix_i]] = 0.0;
        }

        // Phase 1 setup: rotate the residual query globally. Exact partial blocks are sampled from this rotated space.
        try self.walshHadamardSimd(self.f32_buffer[0..d], 12);
        var residual_query_norm2: f32 = 0.0;
        for (self.f32_buffer[0..d]) |value| {
            residual_query_norm2 += value * value;
        }
        const residual_query_norm = @sqrt(residual_query_norm2);

        var active_count: usize = 0;
        for (0..vocab_size) |row_i| {
            if (is_junk[row_i]) {
                dense_logits[row_i] = -std.math.inf(f32);
                continue;
            }
            active_rows[active_count] = @intCast(row_i);
            active_count += 1;
            residual_logits[row_i] = 0.0;
            dense_logits[row_i] = std.math.inf(f32);
        }

        var phase_rows: [phase_exact_top_k]usize = [_]usize{0} ** phase_exact_top_k;
        var phase_scores: [phase_exact_top_k]f32 = [_]f32{-std.math.inf(f32)} ** phase_exact_top_k;
        var top_rows: [top_k_sliced]usize = [_]usize{0} ** top_k_sliced;
        var top_scores: [top_k_sliced]f32 = [_]f32{-std.math.inf(f32)} ** top_k_sliced;
        var pruned_by_phase: [partial_dot_block_count]usize = [_]usize{0} ** partial_dot_block_count;

        const z_score: f32 = 5.0;
        const d_f32: f32 = @floatFromInt(d);

        var best_lower_bound: f32 = -std.math.inf(f32);
        var nb_scored: usize = 0;
        var nb_partial_dot_scored: usize = 0;
        var nb_partial_dot_pruned: usize = 0;

        for (0..partial_dot_block_count) |phase_i| {
            if (active_count == 0) break;

            const block_i = phase_i;
            const block_start = block_i * partial_dot_block_size;
            const block_offset = block_i * partial_dot_block_size * vocab_size;
            const query_block = self.f32_buffer[block_start..][0..partial_dot_block_size];
            const scored_dim = (phase_i + 1) * partial_dot_block_size;
            const scored_dim_f32: f32 = @floatFromInt(scored_dim);
            const remaining_dim = d - scored_dim;
            const remaining_dim_f32: f32 = @floatFromInt(remaining_dim);
            const estimator_scale = d_f32 / scored_dim_f32;
            const err_scale = if (remaining_dim == 0)
                0.0
            else
                z_score * @sqrt(remaining_dim_f32 / d_f32) / @sqrt(scored_dim_f32);

            resetTopK(phase_exact_top_k, &phase_rows, &phase_scores);
            for (active_rows[0..active_count]) |row_u32| {
                const row_i: usize = @intCast(row_u32);
                const row_block = self.partial_dot_blocks[block_offset + row_i * partial_dot_block_size ..][0..partial_dot_block_size];
                residual_logits[row_i] += dotPartialDotBlock(query_block, row_block);
                nb_partial_dot_scored += 1;
                const approx = partial_logits[row_i] + estimator_scale * residual_logits[row_i];
                const err = err_scale * residual_query_norm * lm_head.row_norms[row_i];
                dense_logits[row_i] = approx + err;
                insertTopK(phase_exact_top_k, row_i, approx, &phase_rows, &phase_scores);
            }

            exactScorePhaseTopK(phase_exact_top_k, self, query, &phase_rows, &phase_scores, &top_rows, &top_scores, &best_lower_bound, &nb_scored);

            var next_active_count: usize = 0;
            for (active_rows[0..active_count]) |row_u32| {
                const row_i: usize = @intCast(row_u32);
                active_rows[next_active_count] = row_u32;
                next_active_count += @intFromBool(dense_logits[row_i] >= best_lower_bound);
            }
            const phase_pruned = active_count - next_active_count;
            pruned_by_phase[phase_i] = phase_pruned;
            nb_partial_dot_pruned += phase_pruned;
            active_count = next_active_count;
        }

        for (active_rows[0..active_count]) |row_u32| {
            const row_i: usize = @intCast(row_u32);
            if (dense_logits[row_i] < best_lower_bound) continue;
            const row = lm_data[row_i * d ..][0..d];
            const real = self.realLogit(query, row);
            nb_scored += 1;
            best_lower_bound = @max(best_lower_bound, real);
            insertTop16(row_i, real, &top_rows, &top_scores);
        }

        return .{
            .rows = top_rows,
            .nb_scored = nb_scored,
            .nb_partial_dot_scored = nb_partial_dot_scored,
            .nb_partial_dot_pruned = nb_partial_dot_pruned,
            .partial_dot_pruned_by_phase = pruned_by_phase,
        };
    }

    inline fn dotPartialDotBlock( query_block: *const [partial_dot_block_size]f32, row_block: *const [partial_dot_block_size]f32) f32 {
        const simd_len = 16;
        const Vec = @Vector(simd_len, f32);
        var acc: Vec = @splat(0);
        var i: usize = 0;
        while (i + simd_len <= partial_dot_block_size) : (i += simd_len) {
            const query_v: Vec = query_block[i..][0..simd_len].*;
            const row_v: Vec = row_block[i..][0..simd_len].*;
            acc = @mulAdd(Vec, query_v, row_v, acc);
        }
        return @reduce(.Add, acc);
    }

};
