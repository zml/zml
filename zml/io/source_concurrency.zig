//! Source concurrency policy. The runtime supplies completed-read evidence
//! and backpressure; each decision returns a width and a measurement generation.

const std = @import("std");

pub const Parallelism = union(enum) {
    adaptive: Adaptive,
    fixed: usize,

    pub const Adaptive = struct {
        initial: usize,
        maximum: usize,
    };

    pub fn initial(self: Parallelism) usize {
        return switch (self) {
            .adaptive => |adaptive| adaptive.initial,
            .fixed => |fixed| fixed,
        };
    }

    pub fn maximum(self: Parallelism) usize {
        return switch (self) {
            .adaptive => |adaptive| adaptive.maximum,
            .fixed => |fixed| fixed,
        };
    }

    pub fn isAdaptive(self: Parallelism) bool {
        return switch (self) {
            .adaptive => true,
            .fixed => false,
        };
    }
};

pub const widths = [_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };

/// Source-only adaptive state. DMA width and request size never enter its
/// evidence or decisions.
/// Climb-and-hold source width policy. Every scored window is attributed by
/// the admission fence to the rung in effect when its reads were admitted,
/// so a rung change never drains the read gate. The controller climbs the
/// ladder one rung per window while each rung beats the best rate seen by
/// 3%, tolerating one rung that does not, then holds at the lowest rung
/// within 3% of the best. One downward probe below the start rung bounds the
/// number of windows a load spends away from its final width; the probe is
/// adopted only when it beats the best rate, never on retention. The climb
/// stops at the widest rung the pre-grown pinned capacity already covers, so
/// no scored window pays for a slab the load has yet to map.
pub const Controller = struct {
    const State = enum { climbing, holding };

    /// A rung must beat the best rate by this factor to keep the climb going.
    const improvement_ratio = 1.03;
    /// Tolerate one noisy rung before ending the climb. The hold rule then
    /// selects the lowest measured rung within the best rate's noise band.
    const stall_tolerance = 2;
    /// The hold rung is the lowest rung retaining this fraction of the best.
    const hold_ratio = 0.97;

    pub const Evidence = struct {
        completed_requests: usize,
        elapsed_ns: u64,
        bytes: u64,
        exercised_width: usize,

        pub fn scoreable(self: Evidence, expected_width: usize) bool {
            return self.exercised_width >= expected_width and
                self.completed_requests >= @max(@as(usize, 8), expected_width) and
                self.elapsed_ns >= 100 * std.time.ns_per_ms and self.bytes != 0;
        }

        pub fn bytesPerSecond(self: Evidence) f64 {
            if (self.elapsed_ns == 0) return 0;
            return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
                @as(f64, @floatFromInt(self.elapsed_ns));
        }
    };

    /// A width and the generation that measures it. Every decision opens a
    /// new generation, so the runtime re-fences its window on each one.
    pub const Decision = struct {
        width: usize,
        generation: u64,
    };

    fixed_width: ?usize = null,
    index: usize,
    /// Where measurement began: the configured initial rung, or the blind
    /// bootstrap's last rung for a high-latency source.
    start_index: usize,
    /// Pinned-feasibility clip, lowered by backpressure.
    max_index: usize,
    best_index: usize,
    /// Per rung: the mean of its scored windows.
    rates: [widths.len]?f64 = @splat(null),
    samples: [widths.len]u8 = @splat(0),
    state: State,
    /// Consecutive climb samples that failed `improvement_ratio`.
    stalls: u8 = 0,
    probed_down: bool = false,
    generation: u64 = 0,
    last_backoff_generation: u64 = std.math.maxInt(u64),

    /// `growth_free_width` is the widest read width the pool already holds
    /// mapped beyond the DMA stage (`dma.BlockPool.growthFreeRequestWidth`):
    /// above it a scored window maps a new pinned slab, which on a GB300 cost
    /// a whole window (20.8 GiB/s at 48 against 48.8 sustained at 32). It is
    /// a ceiling on the climb, never a reason to start below the configured
    /// rung: a pool with no growth-free headroom is better served by starting
    /// where the caller asked and accepting some growth than by reading one
    /// request at a time. A fixed width is not clipped by it, only by
    /// feasibility.
    pub fn init(
        configured: Parallelism,
        pinned_feasible_width: usize,
        growth_free_width: usize,
    ) Controller {
        const configured_max = @min(configured.maximum(), pinned_feasible_width, @max(configured.initial(), growth_free_width));
        const max_index = widthIndexAtMost(configured_max);
        if (!configured.isAdaptive()) {
            const fixed = @min(configured.initial(), pinned_feasible_width);
            const fixed_index = widthIndexAtMost(fixed);
            return .{
                .fixed_width = @max(@as(usize, 1), fixed),
                .index = fixed_index,
                .start_index = fixed_index,
                .max_index = fixed_index,
                .best_index = fixed_index,
                .state = .holding,
            };
        }
        const initial_index = @min(widthIndexAtMost(configured.initial()), max_index);
        return .{
            .index = initial_index,
            .start_index = initial_index,
            .max_index = max_index,
            .best_index = initial_index,
            .state = .climbing,
        };
    }

    fn widthIndexAtMost(maximum: usize) usize {
        var result: usize = 0;
        for (widths, 0..) |candidate_width, index| {
            if (candidate_width > maximum) break;
            result = index;
        }
        return result;
    }

    pub fn width(self: *const Controller) usize {
        return self.fixed_width orelse widths[self.index];
    }

    pub fn isAdaptive(self: *const Controller) bool {
        return self.fixed_width == null;
    }

    pub fn currentDecision(self: *const Controller) Decision {
        return .{ .width = self.width(), .generation = self.generation };
    }

    /// Opens a new generation at the current width: the first measured
    /// window after a blind bootstrap, whose admissions overlap generations.
    pub fn newGeneration(self: *Controller) Decision {
        self.generation +|= 1;
        return self.currentDecision();
    }

    /// Pre-response growth for a high-latency source: 24 then 32, before any
    /// window is scored. Measurement then starts from the reached rung.
    pub fn blindGrow(self: *Controller) ?Decision {
        if (!self.isAdaptive() or self.state != .climbing or
            self.rates[self.best_index] != null or self.index >= self.max_index)
            return null;
        const ceiling: usize = if (self.width() < 24) 24 else if (self.width() < 32) 32 else return null;
        const target = @min(widthIndexAtMost(ceiling), self.max_index);
        if (target <= self.index) return null;
        self.start_index = target;
        self.best_index = target;
        return self.moveTo(target);
    }

    pub fn observe(self: *Controller, evidence: Evidence) Decision {
        if (!self.isAdaptive() or self.state == .holding) return self.currentDecision();
        std.debug.assert(evidence.scoreable(self.width()));
        // While climbing the scored rung is the best rung or the one above
        // it; a re-measure or the downward probe scores a rung below it.
        const climb_sample = self.index >= self.best_index;
        const best_rate = self.rates[self.best_index];
        const rate = self.addSample(self.index, evidence.bytesPerSecond());
        const improved = if (best_rate) |best| rate > improvement_ratio * best else true;
        if (improved) self.best_index = self.index;
        if (climb_sample) {
            self.stalls = if (improved) 0 else self.stalls +| 1;
            if (self.index < self.max_index and (improved or self.stalls < stall_tolerance))
                return self.moveTo(self.index + 1);
            if (improved) return self.hold(self.index);
        }

        // A rung below the start rung is the downward probe. Its one window
        // reads high when it inherits the wider rung's queued transfers: on
        // a GB300 a probe at 8 measured 41 GiB/s right after 16 against 36.9
        // sustained, and retention then held the whole load at the narrowest
        // rung it ever tried (3.70 s against 3.05 s at 32). Only an
        // improvement adopts it.
        if (self.index < self.start_index and !improved) return self.hold(self.start_index);

        const hold_index = self.holdIndex();
        if (hold_index == self.start_index and self.start_index > 0 and !self.probed_down) {
            self.probed_down = true;
            return self.moveTo(self.start_index - 1);
        }
        return self.hold(hold_index);
    }

    fn addSample(self: *Controller, index: usize, rate: f64) f64 {
        const count: f64 = @floatFromInt(self.samples[index]);
        const mean = if (self.rates[index]) |previous|
            (previous * count + rate) / (count + 1)
        else
            rate;
        self.rates[index] = mean;
        self.samples[index] +|= 1;
        return mean;
    }

    /// The lowest measured rung at or below the best one that retains
    /// `hold_ratio` of the best rate.
    fn holdIndex(self: *const Controller) usize {
        const best_rate = self.rates[self.best_index].?;
        for (self.rates[0 .. self.best_index + 1], 0..) |maybe_rate, index| {
            const rate = maybe_rate orelse continue;
            if (rate >= hold_ratio * best_rate) return index;
        }
        return self.best_index;
    }

    fn moveTo(self: *Controller, index: usize) Decision {
        self.index = index;
        return self.newGeneration();
    }

    fn hold(self: *Controller, index: usize) Decision {
        self.index = index;
        self.state = .holding;
        return self.newGeneration();
    }

    /// Source backpressure: one rung down, clipped there, and holding. At
    /// most once per generation of fresh admissions: a further sample in the
    /// generation a backoff opened is delayed feedback from the old width
    /// unless a read admitted under the new generation has begun
    /// (`fresh_admissions`), so it cannot ratchet through several rungs.
    pub fn backoff(self: *Controller, fresh_admissions: bool) ?Decision {
        if (!self.isAdaptive()) return null;
        if (self.last_backoff_generation == self.generation and !fresh_admissions) return null;
        self.index -|= 1;
        self.max_index = self.index;
        self.state = .holding;
        return self.openBackoffGeneration();
    }

    /// Transient backpressure (retries, connection failures, 5xx without a
    /// throttle): one rung down, ceiling and state unchanged. A climbing
    /// controller restarts its climb at the lower rung: it becomes the best
    /// rung and its mean is forgotten, so the next window there is a fresh
    /// climb sample that can lead back above the step. A holding one keeps
    /// holding. Same once-per-generation rule as `backoff`.
    pub fn stepDownTransient(self: *Controller, fresh_admissions: bool) ?Decision {
        if (!self.isAdaptive()) return null;
        if (self.last_backoff_generation == self.generation and !fresh_admissions) return null;
        self.index -|= 1;
        if (self.state == .climbing) {
            self.best_index = self.index;
            self.rates[self.index] = null;
            self.samples[self.index] = 0;
            self.stalls = 0;
        }
        return self.openBackoffGeneration();
    }

    fn openBackoffGeneration(self: *Controller) Decision {
        const decision = self.newGeneration();
        self.last_backoff_generation = self.generation;
        return decision;
    }
};

fn testEvidence(
    controller: *const Controller,
    rate: f64,
) Controller.Evidence {
    return .{
        .completed_requests = @max(@as(usize, 8), controller.width()),
        .elapsed_ns = std.time.ns_per_s,
        .bytes = @intFromFloat(rate * 1024 * 1024),
        .exercised_width = controller.width(),
    };
}

const CurvePoint = struct { width: usize, rate: f64 };

/// Replays a curve of per-width rates: each window scores the controller's
/// current width and returns the number of windows until it holds.
fn replayCurve(
    controller: *Controller,
    curve: []const CurvePoint,
) !usize {
    var windows: usize = 0;
    while (controller.state == .climbing) : (windows += 1) {
        const rate = for (curve) |point| {
            if (point.width == controller.width()) break point.rate;
        } else return error.UnmeasuredWidth;
        const generation = controller.generation;
        _ = controller.observe(testEvidence(controller, rate));
        try std.testing.expect(controller.generation == generation + 1);
    }
    return windows;
}

test "source read controller bounds blind growth at 32" {
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expectEqual(@as(usize, 24), controller.blindGrow().?.width);
    try std.testing.expectEqual(@as(usize, 32), controller.blindGrow().?.width);
    try std.testing.expect(controller.blindGrow() == null);
    try std.testing.expectEqual(@as(usize, 32), widths[controller.start_index]);
    // Measurement starts at the reached rung; a scored window ends growth.
    _ = controller.observe(testEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 48), controller.width());
    try std.testing.expect(controller.blindGrow() == null);
}

test "source read controller clips infeasible adaptive and fixed widths" {
    var adaptive = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        10,
        10,
    );
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());
    try std.testing.expect(adaptive.blindGrow() == null);
    // At the clip the first window holds the only rung it can use.
    _ = adaptive.observe(testEvidence(&adaptive, 100));
    try std.testing.expectEqual(Controller.State.holding, adaptive.state);
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());

    const fixed = Controller.init(.{ .fixed = 20 }, 7, 7);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expectEqual(Controller.State.holding, fixed.state);

    const configured_initial = Controller.init(
        .{ .adaptive = .{ .initial = 48, .maximum = 128 } },
        128,
        128,
    );
    try std.testing.expectEqual(@as(usize, 48), configured_initial.width());
}

test "source read evidence requires enough concurrency and duration" {
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    var short = testEvidence(&controller, 100);
    short.elapsed_ns = 99 * std.time.ns_per_ms;
    try std.testing.expect(!short.scoreable(controller.width()));
    var unexercised = testEvidence(&controller, 100);
    unexercised.exercised_width -= 1;
    try std.testing.expect(!unexercised.scoreable(controller.width()));
    var few = testEvidence(&controller, 100);
    few.completed_requests = 11;
    try std.testing.expect(!few.scoreable(controller.width()));
    var empty = testEvidence(&controller, 100);
    empty.bytes = 0;
    try std.testing.expect(!empty.scoreable(controller.width()));
    try std.testing.expectEqual(
        @as(usize, 16),
        controller.observe(testEvidence(&controller, 100)).width,
    );
}

test "source read controller replays the B70 32 MiB curve and holds 12" {
    // Recorded on one B70 at 32 MiB requests (CTX "Source request size is
    // backend-dependent"), GiB/s. Eight was not screened there; the probe
    // below the start rung gets a value below the best.
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replayCurve(&controller, &.{
        .{ .width = 8, .rate = 19.90 },
        .{ .width = 12, .rate = 21.33 },
        .{ .width = 16, .rate = 20.69 },
        .{ .width = 24, .rate = 18.90 },
        .{ .width = 32, .rate = 17.33 },
    });
    // 12, 16 (0.970 of 12: the first stall), 24 (0.886: the second stops the
    // climb), the downward probe of 8. A declining curve costs one window
    // more than it did on a single-strike rule and reaches the same rung.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expect(controller.probed_down);
    try std.testing.expectEqual(@as(u8, 1), controller.samples[Controller.widthIndexAtMost(24)]);
    // Holding: further evidence and blind growth change nothing.
    const held = controller.observe(testEvidence(&controller, 30));
    try std.testing.expectEqual(@as(usize, 12), held.width);
    try std.testing.expectEqual(controller.generation, held.generation);
    try std.testing.expect(controller.blindGrow() == null);
}

test "source read controller climbs past one rung inside the noise band" {
    // gb300-2 loading DeepSeek-V4 at 16 MiB requests: the rungs from 12 to 48
    // sustain 44.8 to 48.8 GiB/s over a whole load while one 120 ms window at
    // a single rung spreads 37.8 to 53.1, so a rung is regularly measured
    // below its neighbour by more than the 3% band. Run 6 of the baseline set
    // read 40.50 at 12 and 41.61 at 16 (1.027 of it) and ended its climb
    // there.
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    _ = controller.observe(testEvidence(&controller, 40.50));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    // The stalled rung carries the climb on instead of ending it.
    _ = controller.observe(testEvidence(&controller, 41.61));
    try std.testing.expectEqual(@as(usize, 24), controller.width());
    try std.testing.expectEqual(@as(u8, 1), controller.stalls);
    // The rung above it is the sustained plateau, and clears the stall.
    _ = controller.observe(testEvidence(&controller, 48.20));
    try std.testing.expectEqual(@as(usize, 32), controller.width());
    try std.testing.expectEqual(@as(u8, 0), controller.stalls);
    // Two rungs in a row inside the band stop the climb; the hold rule then
    // picks the lowest rung within 3% of the best, whatever the climb passed
    // through.
    _ = controller.observe(testEvidence(&controller, 48.80));
    try std.testing.expectEqual(@as(usize, 48), controller.width());
    _ = controller.observe(testEvidence(&controller, 47.30));
    try std.testing.expectEqual(Controller.State.holding, controller.state);
    try std.testing.expectEqual(@as(usize, 24), controller.width());
}

test "source read controller keeps the start rung when the probe only matches it" {
    // Same host, run 6: the climb stopped at 12 and the probe at 8 read
    // 41.07 GiB/s, above 12's 40.50 window but well under the 36.9 that 8
    // sustains -- a rung stepped down to inherits the wider rung's queued
    // transfers. Retention used to adopt it and hold the load at 8 (3.70 s
    // against 3.05 s at 32).
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replayCurve(&controller, &.{
        .{ .width = 8, .rate = 41.07 },
        .{ .width = 12, .rate = 40.50 },
        .{ .width = 16, .rate = 39.00 },
        .{ .width = 24, .rate = 39.50 },
    });
    // 12, 16, 24, then the probe of 8: 1.014 of the best is not the 3% an
    // adoption needs, so the load holds the rung it started from.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expect(controller.probed_down);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
}

test "source read controller holds at the lowest rung within 3% on a flat curve" {
    // Real AWS shape: 16 MiB requests plateau near 950 MiB/s from 16 up.
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replayCurve(&controller, &.{
        .{ .width = 12, .rate = 900 },
        .{ .width = 16, .rate = 940 },
        .{ .width = 24, .rate = 948 },
        .{ .width = 32, .rate = 950 },
        .{ .width = 48, .rate = 950 },
    });
    // 12, 16 (better by 4.4%), 24 and 32 (not better by 3%): hold at 16, the
    // lowest rung within 3% of it; 12 at 0.957 is below the band.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    try std.testing.expectEqual(@as(usize, 16), widths[controller.best_index]);
    try std.testing.expect(!controller.probed_down);
}

test "source read controller probes below the start rung once" {
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        128,
    );
    const windows = try replayCurve(&controller, &.{
        .{ .width = 8, .rate = 105 },
        .{ .width = 12, .rate = 100 },
        .{ .width = 16, .rate = 100 },
        .{ .width = 24, .rate = 100 },
    });
    // 12, 16 and 24 (flat), 8 (better by 5%): hold 8 without climbing further
    // down.
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    try std.testing.expectEqual(@as(usize, 8), widths[controller.best_index]);

    var from_one = Controller.init(
        .{ .adaptive = .{ .initial = 1, .maximum = 128 } },
        128,
        128,
    );
    // Nothing below the lowest rung to probe.
    _ = try replayCurve(&from_one, &.{
        .{ .width = 1, .rate = 100 },
        .{ .width = 2, .rate = 100 },
        .{ .width = 4, .rate = 100 },
    });
    try std.testing.expectEqual(@as(usize, 1), from_one.width());
    try std.testing.expect(!from_one.probed_down);
}

test "source read controller backs off once per generation of fresh admissions" {
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    _ = controller.observe(testEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    const first = controller.backoff(false).?;
    try std.testing.expectEqual(@as(usize, 12), first.width);
    try std.testing.expectEqual(controller.generation, first.generation);
    try std.testing.expectEqual(Controller.State.holding, controller.state);
    try std.testing.expectEqual(@as(usize, 12), widths[controller.max_index]);
    // Delayed feedback from the old width in the same generation is ignored.
    try std.testing.expect(controller.backoff(false) == null);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    // Feedback after a fresh admission under the new generation counts.
    try std.testing.expectEqual(@as(usize, 8), controller.backoff(true).?.width);
    try std.testing.expectEqual(@as(usize, 8), widths[controller.max_index]);
    // Holding: evidence no longer moves the width.
    _ = controller.observe(testEvidence(&controller, 1000));
    try std.testing.expectEqual(@as(usize, 8), controller.width());

    var floor = Controller.init(
        .{ .adaptive = .{ .initial = 1, .maximum = 64 } },
        64,
        64,
    );
    try std.testing.expectEqual(@as(usize, 1), floor.backoff(false).?.width);
}

test "source read controller steps down on transient backpressure and climbs again" {
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    _ = controller.observe(testEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    const step = controller.stepDownTransient(false).?;
    try std.testing.expectEqual(@as(usize, 12), step.width);
    try std.testing.expectEqual(controller.generation, step.generation);
    // Still climbing, ceiling untouched, the climb restarts at 12.
    try std.testing.expectEqual(Controller.State.climbing, controller.state);
    try std.testing.expectEqual(@as(usize, 64), widths[controller.max_index]);
    try std.testing.expectEqual(@as(usize, 12), widths[controller.best_index]);
    try std.testing.expect(controller.rates[controller.index] == null);
    // Once per generation of fresh admissions, shared with the throttle rule.
    try std.testing.expect(controller.stepDownTransient(false) == null);
    try std.testing.expect(controller.backoff(false) == null);
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    // A fresh window at 12 is a climb sample: back to 16, then above the step.
    _ = controller.observe(testEvidence(&controller, 100));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    _ = controller.observe(testEvidence(&controller, 110));
    try std.testing.expectEqual(@as(usize, 24), controller.width());
    try std.testing.expectEqual(Controller.State.climbing, controller.state);
    // A fresh admission under the step's generation admits another step.
    try std.testing.expectEqual(@as(usize, 16), controller.stepDownTransient(false).?.width);
    try std.testing.expectEqual(@as(usize, 12), controller.stepDownTransient(true).?.width);
    try std.testing.expectEqual(@as(usize, 64), widths[controller.max_index]);

    // Holding: one rung down, still holding, evidence changes nothing.
    var holding = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
        64,
    );
    // 12 -> 16 and 24 (not better) -> the downward probe of 8 (10% below the
    // best) -> hold 12.
    for ([_]f64{ 100, 100, 90, 90 }) |rate| _ = holding.observe(testEvidence(&holding, rate));
    try std.testing.expectEqual(@as(usize, 12), holding.width());
    try std.testing.expectEqual(Controller.State.holding, holding.state);
    try std.testing.expectEqual(@as(usize, 8), holding.stepDownTransient(false).?.width);
    try std.testing.expectEqual(Controller.State.holding, holding.state);
    try std.testing.expectEqual(@as(usize, 64), widths[holding.max_index]);
    _ = holding.observe(testEvidence(&holding, 1000));
    try std.testing.expectEqual(@as(usize, 8), holding.width());
}

test "source read controller stops climbing at the growth-free width" {
    // gb300-2: a pool holding 49 blocks against a 16-block DMA stage
    // leaves 33 growth-free requests, so the climb stops at 32. Above it the
    // lifecycle limit maps a new pinned slab inside a scored window: one such
    // window measured 20.8 GiB/s at 48 against 48.8 sustained at 32.
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        33,
    );
    try std.testing.expectEqual(@as(usize, 32), widths[controller.max_index]);
    // Each rung beats the last: the climb runs up to the ceiling and holds.
    const windows = try replayCurve(&controller, &.{
        .{ .width = 12, .rate = 100 },
        .{ .width = 16, .rate = 110 },
        .{ .width = 24, .rate = 120 },
        .{ .width = 32, .rate = 130 },
    });
    try std.testing.expectEqual(@as(usize, 4), windows);
    try std.testing.expectEqual(@as(usize, 32), controller.width());
}

test "source read controller starts at the configured rung without headroom" {
    // A pool fully covered by the DMA stage reports a growth-free width of
    // 0. Clipping the ceiling to 1 there made every
    // adaptive load on eight MI300X read one request at a time (5.34 s
    // against 1.42 s); starting where the caller asked and accepting some
    // mid-load growth is strictly better.
    var controller = Controller.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
        0,
    );
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expectEqual(@as(usize, 12), widths[controller.max_index]);
}

test "source read controller keeps a fixed width" {
    // A fixed width the caller asked for is not clipped by the growth-free
    // width, only by feasibility.
    var fixed = Controller.init(.{ .fixed = 7 }, 64, 4);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.backoff(true) == null);
    try std.testing.expect(fixed.stepDownTransient(true) == null);
    try std.testing.expect(fixed.blindGrow() == null);
    const observed = fixed.observe(testEvidence(&fixed, 100));
    try std.testing.expectEqual(@as(usize, 7), observed.width);
    try std.testing.expectEqual(@as(u64, 0), fixed.generation);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
}
