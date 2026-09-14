//! One governed HTTP request loop for every backend: wait out the backend's
//! hold, send, classify. A throttle (429 everywhere, 503 on the object
//! stores) arms a backend-wide hold instead of charging the request's retry
//! budget, so every reader of that backend backs off together and the traffic
//! actually falls; every other retryable failure keeps the per-request
//! backoff. The hold is a deadline, not a permit: nothing is narrowed
//! permanently, recovery is the deadline passing, and the loader's source
//! width stays the ceiling it chose, because this loop runs inside it.

const std = @import("std");

const base = @import("base.zig");
const AtomicReadStats = base.AtomicReadStats;
const ReadFailure = base.ReadFailure;

const log = std.log.scoped(.@"zml/vfs/request");

/// Backend-wide rate limiting exceeded `throttle_budget` without a clean
/// window: the caller's request never ran.
pub const RateLimited = error.RateLimited;

pub const RetryConfig = struct {
    /// Retries per request for failures that are not rate limiting.
    max_retries: usize,
    initial_delay: std.Io.Duration,
    max_delay: std.Io.Duration,
    /// Longest hold a throttle may arm, server-named delays included.
    max_hold: std.Io.Duration,
    /// A throttling episode with no clean window longer than this fails the
    /// backend's requests with `error.RateLimited`.
    throttle_budget: std.Io.Duration,

    /// Reads the `max_retries`, `retry_initial_delay`, `retry_max_delay`,
    /// `max_hold` and `throttle_budget` fields every backend's `InitOpts`
    /// declares.
    pub fn fromOptions(opts: anytype) RetryConfig {
        const self: RetryConfig = .{
            .max_retries = opts.max_retries,
            .initial_delay = opts.retry_initial_delay,
            .max_delay = opts.retry_max_delay,
            .max_hold = opts.max_hold,
            .throttle_budget = opts.throttle_budget,
        };
        std.debug.assert(self.initial_delay.nanoseconds >= 0);
        std.debug.assert(self.max_delay.nanoseconds >= self.initial_delay.nanoseconds);
        std.debug.assert(self.max_hold.nanoseconds >= self.initial_delay.nanoseconds);
        return self;
    }
};

/// What a hold covers. Today one governor holds one scope, its backend
/// instance; the authority is carried so that a per-authority map is a local
/// change in `Governor.holdFor`.
pub const Key = []const u8;

/// The rate-limit scope of a request: its URI authority. Only the host is
/// taken, since `Governor.holdFor` ignores the key today; a per-authority
/// scope will want the port beside it.
pub fn authorityOf(uri: std.Uri) Key {
    const host = uri.host orelse return "";
    return switch (host) {
        .raw => |raw| raw,
        .percent_encoded => |encoded| encoded,
    };
}

/// One scope's rate-limit state, owned by the governor's mutex.
pub const Hold = struct {
    /// When requests may go again; null when nothing is held.
    until: ?std.Io.Timestamp = null,
    /// The length of the current hold: a clean window that long ends the
    /// episode.
    length: std.Io.Duration = .fromNanoseconds(0),
    /// Holds armed since the episode began; the backoff base.
    consecutive: usize = 0,
    episode_started: ?std.Io.Timestamp = null,
    /// The episode outlived `throttle_budget`; requests fail until a clean
    /// window clears it.
    exhausted: bool = false,
};

/// One per backend instance, beside its read statistics.
pub const Governor = struct {
    retry: RetryConfig,
    mutex: std.Io.Mutex = .init,
    hold: Hold = .{},

    pub fn init(retry: RetryConfig) Governor {
        return .{ .retry = retry };
    }

    /// The scope `key` belongs to. One generic `HTTP` instance serves every
    /// host of its scheme and one `S3` spans buckets, so per-instance
    /// over-reaches there; a per-authority scope replaces this body with a
    /// map without touching the loop.
    fn holdFor(self: *Governor, key: Key) *Hold {
        _ = key;
        return &self.hold;
    }
};

/// What one request needs from its backend.
pub const Context = struct {
    io: std.Io,
    client: *std.http.Client,
    governor: *Governor,
    stats: *AtomicReadStats,
};

/// One attempt as a backend's request hook sees it.
pub const Attempt = struct {
    /// Zero-based; charged retries are attempt 1 and above. A throttle does
    /// not advance it.
    ordinal: usize,
    /// The status the previous attempt returned, for a hook that reacts to
    /// one (an expired signed URL answering 403 after a long hold).
    previous_status: ?std.http.Status = null,
};

/// Identity of one request, for the logs and for the classification.
pub const RequestSpec = struct {
    /// Backend name for log lines.
    backend: []const u8,
    /// Object identity for log lines.
    target: []const u8,
    /// What a 503 means for this backend: rate limiting on the object stores
    /// (AWS `SlowDown`, GCS), a server failure elsewhere.
    unavailable: ReadFailure,
    /// The rate-limit scope: the request URI's authority.
    key: Key = "",
    /// Statuses this request retries although they are not retryable in
    /// general, because its hook can do something about them: a signed
    /// download URL that expired during a hold answers 401 or 403, and the
    /// hook re-resolves it. Charged to the retry budget.
    retry_once: []const std.http.Status = &.{},
};

pub const Failure = struct {
    failure: ReadFailure,
    /// The delay the server named, if any.
    delay: ?std.Io.Duration = null,
    status: ?std.http.Status = null,
};

/// What one attempt produced: a value, or a reason to try again.
pub fn Outcome(comptime T: type) type {
    return union(enum) {
        done: T,
        retry: Failure,
    };
}

/// The one request loop. `attemptFn` performs a single HTTP exchange (see
/// `exchange`); this drives the hold, the classification and the two budgets
/// around it. A throttle re-runs the attempt without charging a retry, after
/// the hold it armed; any other retryable failure sleeps the per-request
/// backoff and charges one of `max_retries`. Cancellation propagates.
pub fn perform(
    comptime T: type,
    ctx: Context,
    spec: RequestSpec,
    context: anytype,
    comptime attemptFn: fn (@TypeOf(context), Attempt) anyerror!Outcome(T),
) anyerror!T {
    var charged: usize = 0;
    var previous_status: ?std.http.Status = null;
    while (true) {
        try admit(ctx, spec);
        ctx.stats.recordAttempt();
        switch (try attemptFn(context, .{ .ordinal = charged, .previous_status = previous_status })) {
            .done => |value| return value,
            .retry => |failure| {
                ctx.stats.recordFailure(failure.failure);
                previous_status = failure.status;
                if (failure.failure == .throttle) {
                    // Not charged to the retry budget: the hold is what
                    // reduces the traffic, and the episode budget bounds it.
                    try reportThrottle(ctx, spec, failure.delay);
                    continue;
                }
                if (charged >= ctx.governor.retry.max_retries) return error.RetriesExhausted;
                charged += 1;
                ctx.stats.recordRetry();
                const delay = failure.delay orelse
                    fullJitterDelay(ctx.io, ctx.governor.retry.initial_delay, ctx.governor.retry.max_delay, charged);
                ctx.stats.recordRetryDelay(delay);
                try ctx.io.sleep(delay, .awake);
            },
        }
    }
}

/// Waits out the scope's hold. Returns at once when nothing is held; fails
/// with `error.RateLimited` when the episode outlived its budget. Waiters
/// wake at the deadline plus their own jitter, so the readers of one backend
/// do not all fire in the same instant, and re-check, since the hold may have
/// been extended while they slept.
pub fn admit(ctx: Context, spec: RequestSpec) error{ Canceled, RateLimited }!void {
    const governor = ctx.governor;
    while (true) {
        const deadline = deadline: {
            governor.mutex.lockUncancelable(ctx.io);
            defer governor.mutex.unlock(ctx.io);
            const hold = governor.holdFor(spec.key);
            const now: std.Io.Timestamp = .now(ctx.io, .awake);
            refresh(hold, now);
            if (hold.exhausted) return error.RateLimited;
            const until = hold.until orelse return;
            if (until.nanoseconds <= now.nanoseconds) return;
            break :deadline until;
        };
        const jitter = fullJitterDelay(ctx.io, governor.retry.initial_delay, governor.retry.initial_delay, 0);
        const started: std.Io.Timestamp = .now(ctx.io, .awake);
        try std.Io.Clock.Timestamp.wait(deadline.addDuration(jitter).withClock(.awake), ctx.io);
        ctx.stats.recordHoldWait(started.durationTo(.now(ctx.io, .awake)));
    }
}

/// Arms or extends the scope's hold after a throttle. `named` is the delay
/// the server asked for, if any; it is honoured up to `max_hold` and never
/// below `initial_delay`, because `Retry-After: 0` and Hugging Face's
/// `RateLimit; t=0` parse to zero and a zero hold is a hot loop now that a
/// throttle does not charge a retry.
pub fn reportThrottle(ctx: Context, spec: RequestSpec, named: ?std.Io.Duration) error{RateLimited}!void {
    const governor = ctx.governor;
    const retry = governor.retry;
    governor.mutex.lockUncancelable(ctx.io);
    defer governor.mutex.unlock(ctx.io);
    const hold = governor.holdFor(spec.key);
    const now: std.Io.Timestamp = .now(ctx.io, .awake);
    refresh(hold, now);

    if (hold.episode_started) |started| {
        if (started.durationTo(now).nanoseconds > retry.throttle_budget.nanoseconds) {
            hold.exhausted = true;
            // The caller reports the failure; this is the one line that
            // says why its request never ran.
            log.warn("{s}: rate limited for longer than the throttle budget while reading {s}; giving up", .{ spec.backend, spec.target });
            return error.RateLimited;
        }
    } else {
        hold.episode_started = now;
    }

    const holding = if (hold.until) |until| now.nanoseconds < until.nanoseconds else false;
    if (!holding) {
        hold.consecutive += 1;
        const backoff = fullJitterDelay(ctx.io, retry.initial_delay, retry.max_delay, hold.consecutive - 1);
        var delay = backoff;
        if (named) |value| {
            if (value.nanoseconds > delay.nanoseconds) delay = value;
        }
        hold.length = clampDelay(delay, retry.initial_delay, retry.max_hold);
        hold.until = now.addDuration(hold.length);
        log.warn("{s}: {s} is rate limiting; holding its requests for {d:.3}s", .{
            spec.backend,
            spec.target,
            @as(f64, @floatFromInt(hold.length.toNanoseconds())) / std.time.ns_per_s,
        });
        ctx.stats.recordHold();
        return;
    }
    // A request already in flight when the hold armed got its own answer,
    // with a later reset: extend, never shorten.
    const asked = named orelse return;
    const capped = clampDelay(asked, retry.initial_delay, retry.max_hold);
    const extended = now.addDuration(capped);
    if (extended.nanoseconds > hold.until.?.nanoseconds) {
        hold.until = extended;
        hold.length = capped;
    }
}

/// A clean window as long as the hold itself, with no new throttle, ends the
/// episode: nothing else needs counting, since a throttle inside the window
/// would have re-armed `until`.
fn refresh(hold: *Hold, now: std.Io.Timestamp) void {
    const until = hold.until orelse return;
    if (now.nanoseconds < until.addDuration(hold.length).nanoseconds) return;
    hold.* = .{};
}

fn clampDelay(value: std.Io.Duration, low: std.Io.Duration, high: std.Io.Duration) std.Io.Duration {
    return .fromNanoseconds(std.math.clamp(value.nanoseconds, low.nanoseconds, high.nanoseconds));
}

/// Statuses an attempt accepts beyond the 2xx class; anything else is
/// classified as a retry or fails the request.
pub const Accept = struct {
    /// Individual statuses the caller maps itself (404 to `FileNotFound`, a
    /// 401 it re-authenticates).
    statuses: []const std.http.Status = &.{},

    fn allows(self: Accept, status: std.http.Status, redirects: Redirects) bool {
        if (status.class() == .success) return true;
        if (redirects == .surface and status.class() == .redirect) return true;
        for (self.statuses) |accepted| {
            if (accepted == status) return true;
        }
        return false;
    }
};

/// What happens to a redirect: the client chases it (the std default of
/// three hops), `consume` receives it because the caller walks the chain
/// itself, or it fails the request.
pub const Redirects = enum {
    follow,
    surface,
    forbid,

    fn behavior(self: Redirects) std.http.Client.Request.RedirectBehavior {
        return switch (self) {
            .follow => @enumFromInt(3),
            .surface => .unhandled,
            .forbid => .not_allowed,
        };
    }
};

pub const ExchangeOptions = struct {
    method: std.http.Method = .GET,
    headers: std.http.Client.Request.Headers = .{ .accept_encoding = .{ .override = "identity" } },
    extra_headers: []const std.http.Header = &.{},
    /// Sent as the whole request body; the method must allow one. The
    /// writer uses it as its own buffer, so it must be mutable.
    payload: ?[]u8 = null,
    accept: Accept = .{},
    redirects: Redirects = .forbid,
    /// The buffer `receiveHead` parses into; the head bytes, and any
    /// `location`, live in it while `consume` runs.
    head_buffer: []u8,
};

/// One HTTP exchange: send, receive the head, classify it, and hand an
/// accepted response to `consume` while the request is alive. Connection,
/// send and receive failures and every classified status become `.retry`;
/// an unclassified status fails with `error.RequestFailed`.
pub fn exchange(
    comptime T: type,
    ctx: Context,
    uri: std.Uri,
    options: ExchangeOptions,
    spec: RequestSpec,
    context: anytype,
    comptime consume: fn (@TypeOf(context), *std.http.Client.Response) anyerror!T,
) anyerror!Outcome(T) {
    var req = ctx.client.request(options.method, uri, .{
        .redirect_behavior = options.redirects.behavior(),
        .headers = options.headers,
        .extra_headers = options.extra_headers,
    }) catch |err| switch (err) {
        error.Timeout => return retryable(T, spec, "connect", err, .timeout),
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.HostUnreachable,
        error.NetworkUnreachable,
        error.NetworkDown,
        error.NameServerFailure,
        => return retryable(T, spec, "connect", err, .transient),
        else => return fatal(spec, "connect", err),
    };
    defer req.deinit();

    if (options.payload) |payload| {
        req.sendBodyComplete(payload) catch |err| switch (err) {
            error.WriteFailed => return retryable(T, spec, "send body", err, .transient),
        };
    } else {
        req.sendBodiless() catch |err| switch (err) {
            error.WriteFailed => return retryable(T, spec, "send headers", err, .transient),
        };
    }

    var res = req.receiveHead(options.head_buffer) catch |err| switch (err) {
        error.Timeout => return retryable(T, spec, "receive headers", err, .timeout),
        error.HttpConnectionClosing,
        error.HttpRequestTruncated,
        error.ReadFailed,
        error.WriteFailed,
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.HostUnreachable,
        error.NetworkUnreachable,
        error.NetworkDown,
        error.NameServerFailure,
        => return retryable(T, spec, "receive headers", err, .transient),
        else => return fatal(spec, "receive headers", err),
    };

    if (!options.accept.allows(res.head.status, options.redirects)) {
        for (spec.retry_once) |status| {
            if (status != res.head.status) continue;
            log.warn("{s}: {s} answered {d}; retrying once with a fresh request", .{
                spec.backend,
                spec.target,
                @intFromEnum(res.head.status),
            });
            return .{ .retry = .{ .failure = .transient, .status = res.head.status } };
        }
        const failure = classifyStatus(res.head.status, spec.unavailable) orelse {
            log.err("{s}: {s} failed: {s}", .{ spec.backend, spec.target, res.head.bytes });
            return error.RequestFailed;
        };
        log.warn("{s}: {s} failed: {s}", .{ spec.backend, spec.target, res.head.bytes });
        return .{ .retry = .{
            .failure = failure,
            .delay = serverRetryDelay(res.head),
            .status = res.head.status,
        } };
    }

    return .{ .done = consume(context, &res) catch |err| switch (err) {
        error.EndOfStream, error.ReadFailed, error.WriteFailed => return retryable(T, spec, "read body", err, .transient),
        else => return err,
    } };
}

fn retryable(
    comptime T: type,
    spec: RequestSpec,
    stage: []const u8,
    err: anyerror,
    failure: ReadFailure,
) Outcome(T) {
    log.warn("{s}: {s} for {s} failed: {}", .{ spec.backend, stage, spec.target, err });
    return .{ .retry = .{ .failure = failure } };
}

fn fatal(spec: RequestSpec, stage: []const u8, err: anyerror) anyerror {
    log.err("{s}: {s} for {s} failed: {}", .{ spec.backend, stage, spec.target, err });
    return err;
}

/// Retry classification of a non-2xx status; null when the status is not
/// retried. `unavailable` is what a 503 means for the backend: rate limiting
/// on the object stores (AWS `SlowDown`, GCS), a server failure elsewhere.
pub fn classifyStatus(status: std.http.Status, unavailable: ReadFailure) ?ReadFailure {
    return switch (status) {
        .request_timeout => .timeout,
        .too_many_requests => .throttle,
        .service_unavailable => unavailable,
        else => if (status.class() == .server_error) .server_failure else null,
    };
}

/// The retry delay the server names, if any: `Retry-After` delta-seconds
/// (the HTTP-date form is not parsed and falls back to the jittered delay)
/// or the `t=` reset of the `RateLimit` header Hugging Face sends.
pub fn serverRetryDelay(head: std.http.Client.Response.Head) ?std.Io.Duration {
    var it = head.iterateHeaders();
    while (it.next()) |header| {
        if (std.ascii.eqlIgnoreCase(header.name, "Retry-After")) {
            return delaySeconds(header.value) orelse continue;
        }
        if (std.ascii.eqlIgnoreCase(header.name, "RateLimit")) {
            var parts = std.mem.splitScalar(u8, header.value, ';');
            while (parts.next()) |part| {
                const trimmed = std.mem.trim(u8, part, " \t");
                if (std.mem.startsWith(u8, trimmed, "t=")) return delaySeconds(trimmed[2..]) orelse continue;
            }
        }
    }
    return null;
}

fn delaySeconds(value: []const u8) ?std.Io.Duration {
    const seconds = std.fmt.parseInt(u32, std.mem.trim(u8, value, " \t"), 10) catch return null;
    return .fromSeconds(seconds);
}

pub fn fullJitterDelay(
    io: std.Io,
    initial: std.Io.Duration,
    maximum: std.Io.Duration,
    attempt: usize,
) std.Io.Duration {
    const max_delay_ns: i96 = @min(
        maximum.toNanoseconds(),
        initial.toNanoseconds() *| (@as(i96, 1) << @as(u7, @intCast(@min(attempt, std.math.maxInt(u7))))),
    );
    if (max_delay_ns <= 0) return .fromNanoseconds(0);

    var seed: u64 = undefined;
    io.random(@ptrCast(&seed));
    var prng: std.Random.DefaultPrng = .init(seed);
    return .fromNanoseconds(prng.random().intRangeAtMost(i96, 0, max_delay_ns));
}

const test_spec: RequestSpec = .{
    .backend = "test",
    .target = "object",
    .unavailable = .throttle,
};

fn testGovernor(initial_ms: i64, max_delay_ms: i64, max_hold_ms: i64, budget_ms: i64) Governor {
    return .init(.{
        .max_retries = 3,
        .initial_delay = .fromMilliseconds(initial_ms),
        .max_delay = .fromMilliseconds(max_delay_ms),
        .max_hold = .fromMilliseconds(max_hold_ms),
        .throttle_budget = .fromMilliseconds(budget_ms),
    });
}

fn testContext(governor: *Governor, stats: *AtomicReadStats) Context {
    // No attempt in these tests reaches the network.
    return .{ .io = std.testing.io, .client = undefined, .governor = governor, .stats = stats };
}

test "a throttle arms a hold of at least the initial delay, even when the server names zero" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(20, 20, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    // `Retry-After: 0` and Hugging Face's `RateLimit; t=0` parse to zero.
    try reportThrottle(ctx, test_spec, .fromSeconds(0));
    const hold = governor.hold;
    try std.testing.expect(hold.until != null);
    try std.testing.expectEqual(@as(i96, 20 * std.time.ns_per_ms), hold.length.nanoseconds);
    try std.testing.expectEqual(@as(usize, 1), hold.consecutive);
    try std.testing.expect(hold.episode_started != null);
    try std.testing.expectEqual(@as(u64, 1), stats.snapshot().holds);
}

test "a longer named delay extends a running hold and a shorter one does not" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(10, 10, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    try reportThrottle(ctx, test_spec, null);
    const armed = governor.hold.until.?;

    // A request that was already in flight got its own answer, with a later
    // reset: the hold extends.
    try reportThrottle(ctx, test_spec, .fromMilliseconds(500));
    const extended = governor.hold.until.?;
    try std.testing.expect(extended.nanoseconds > armed.nanoseconds);

    // A shorter one never shortens it, and neither arms a second hold.
    try reportThrottle(ctx, test_spec, .fromMilliseconds(1));
    try std.testing.expectEqual(extended.nanoseconds, governor.hold.until.?.nanoseconds);
    try std.testing.expectEqual(@as(usize, 1), governor.hold.consecutive);
    try std.testing.expectEqual(@as(u64, 1), stats.snapshot().holds);
}

test "a named delay is capped by max_hold" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(10, 10, 50, 300_000);
    const ctx = testContext(&governor, &stats);

    try reportThrottle(ctx, test_spec, .fromSeconds(3600));
    try std.testing.expectEqual(@as(i96, 50 * std.time.ns_per_ms), governor.hold.length.nanoseconds);
}

test "a second throttle inside the clean window doubles the backoff base" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(10, 1000, 10_000, 300_000);
    const ctx = testContext(&governor, &stats);

    try reportThrottle(ctx, test_spec, null);
    try std.testing.expectEqual(@as(i96, 10 * std.time.ns_per_ms), governor.hold.length.nanoseconds);

    // The hold has passed but the clean window has not: rewind it by hand
    // rather than sleeping through a race.
    const now: std.Io.Timestamp = .now(std.testing.io, .awake);
    governor.hold.until = now.subDuration(.fromMilliseconds(1));
    try reportThrottle(ctx, test_spec, null);
    try std.testing.expectEqual(@as(usize, 2), governor.hold.consecutive);
    // Full jitter over twice the base, floored at the initial delay.
    try std.testing.expect(governor.hold.length.nanoseconds >= 10 * std.time.ns_per_ms);
    try std.testing.expect(governor.hold.length.nanoseconds <= 20 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(u64, 2), stats.snapshot().holds);
}

test "a clean window as long as the hold ends the episode" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(10, 1000, 10_000, 300_000);
    const ctx = testContext(&governor, &stats);

    try reportThrottle(ctx, test_spec, null);
    const length = governor.hold.length;
    const now: std.Io.Timestamp = .now(std.testing.io, .awake);
    governor.hold.until = now.subDuration(length).subDuration(.fromMilliseconds(1));

    try reportThrottle(ctx, test_spec, null);
    try std.testing.expectEqual(@as(usize, 1), governor.hold.consecutive);
    try std.testing.expectEqual(@as(i96, 10 * std.time.ns_per_ms), governor.hold.length.nanoseconds);
}

test "an episode longer than the throttle budget fails requests until a clean window" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(10, 10, 10_000, 50);
    const ctx = testContext(&governor, &stats);

    try reportThrottle(ctx, test_spec, .fromSeconds(30));
    const now: std.Io.Timestamp = .now(std.testing.io, .awake);
    governor.hold.episode_started = now.subDuration(.fromMilliseconds(51));

    try std.testing.expectError(error.RateLimited, reportThrottle(ctx, test_spec, null));
    try std.testing.expect(governor.hold.exhausted);
    // Every other request of this backend fails at once, without waiting the
    // hold out.
    try std.testing.expectError(error.RateLimited, admit(ctx, test_spec));

    // The next clean window clears the episode.
    governor.hold.until = now.subDuration(governor.hold.length).subDuration(.fromMilliseconds(1));
    try admit(ctx, test_spec);
    try std.testing.expect(governor.hold.until == null);
    try std.testing.expect(!governor.hold.exhausted);
}

test "admit returns at once when nothing is held" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(10, 10, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    const started: std.Io.Timestamp = .now(std.testing.io, .awake);
    try admit(ctx, test_spec);
    try std.testing.expect(started.durationTo(.now(std.testing.io, .awake)).nanoseconds < 5 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(u64, 0), stats.snapshot().hold_wait_ns);
}

test "admit waits a hold out and counts the wait" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(1, 1, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    try reportThrottle(ctx, test_spec, .fromMilliseconds(15));
    const started: std.Io.Timestamp = .now(std.testing.io, .awake);
    try admit(ctx, test_spec);
    try std.testing.expect(started.durationTo(.now(std.testing.io, .awake)).nanoseconds >= 10 * std.time.ns_per_ms);
    try std.testing.expect(stats.snapshot().hold_wait_ns > 0);
}

test "a waiter blocked on a hold is cancelable" {
    const io = std.testing.io;
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(1, 1, 100_000, 300_000);
    const ctx = testContext(&governor, &stats);
    try reportThrottle(ctx, test_spec, .fromSeconds(60));

    var result: std.atomic.Value(u16) = .init(0);
    var started: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(ctx_: Context, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            started_.set(ctx_.io);
            admit(ctx_, test_spec) catch |err| {
                result_.store(@intFromError(err), .release);
                return;
            };
        }
    }.run, .{ ctx, &started, &result });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    group.cancel(io);
    try std.testing.expectEqual(@intFromError(error.Canceled), result.load(.acquire));
}

test "only rate limiting holds: a timeout and a server failure keep the per-request budget" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(0, 0, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    const Attempts = struct {
        seen: usize = 0,
        fn run(self: *@This(), _: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            return switch (self.seen) {
                1 => .{ .retry = .{ .failure = .timeout } },
                2 => .{ .retry = .{ .failure = .server_failure } },
                else => .{ .done = self.seen },
            };
        }
    };
    var attempts: Attempts = .{};
    try std.testing.expectEqual(@as(usize, 3), try perform(usize, ctx, test_spec, &attempts, Attempts.run));
    try std.testing.expect(governor.hold.until == null);
    const snapshot = stats.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snapshot.holds);
    try std.testing.expectEqual(@as(u64, 2), snapshot.retries);
    try std.testing.expectEqual(@as(u64, 3), snapshot.physical_requests);
}

test "a throttle retries without charging the retry budget" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(0, 0, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    const Attempts = struct {
        seen: usize = 0,
        fn run(self: *@This(), current: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            // The retry budget is three; ten throttles do not exhaust it.
            if (self.seen <= 10) {
                try std.testing.expectEqual(@as(usize, 0), current.ordinal);
                return .{ .retry = .{ .failure = .throttle, .status = .too_many_requests } };
            }
            try std.testing.expectEqual(@as(?std.http.Status, .too_many_requests), current.previous_status);
            return .{ .done = self.seen };
        }
    };
    var attempts: Attempts = .{};
    try std.testing.expectEqual(@as(usize, 11), try perform(usize, ctx, test_spec, &attempts, Attempts.run));
    const snapshot = stats.snapshot();
    try std.testing.expectEqual(@as(u64, 0), snapshot.retries);
    try std.testing.expectEqual(@as(u64, 10), snapshot.throttles);
    try std.testing.expectEqual(@as(u64, 11), snapshot.physical_requests);
    try std.testing.expect(snapshot.holds >= 1);
}

test "the retry budget still bounds failures that are not rate limiting" {
    var stats: AtomicReadStats = .{};
    var governor = testGovernor(0, 0, 1000, 300_000);
    const ctx = testContext(&governor, &stats);

    const Attempts = struct {
        seen: usize = 0,
        fn run(self: *@This(), _: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            return .{ .retry = .{ .failure = .transient } };
        }
    };
    var attempts: Attempts = .{};
    try std.testing.expectError(error.RetriesExhausted, perform(usize, ctx, test_spec, &attempts, Attempts.run));
    // The first attempt plus `max_retries` of them.
    try std.testing.expectEqual(@as(usize, 4), attempts.seen);
}

test "retry status classification is typed and 503 depends on the backend" {
    try std.testing.expectEqual(ReadFailure.timeout, classifyStatus(.request_timeout, .server_failure).?);
    try std.testing.expectEqual(ReadFailure.throttle, classifyStatus(.too_many_requests, .server_failure).?);
    try std.testing.expectEqual(ReadFailure.server_failure, classifyStatus(.bad_gateway, .throttle).?);
    try std.testing.expectEqual(ReadFailure.server_failure, classifyStatus(.service_unavailable, .server_failure).?);
    try std.testing.expectEqual(ReadFailure.throttle, classifyStatus(.service_unavailable, .throttle).?);
    try std.testing.expect(classifyStatus(.not_found, .throttle) == null);
}

test "server retry delay comes from Retry-After seconds or a RateLimit reset" {
    const Head = std.http.Client.Response.Head;
    const retry_after = try Head.parse("HTTP/1.1 503 Service Unavailable\r\nretry-after: 3\r\n\r\n");
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(3), serverRetryDelay(retry_after).?);

    const rate_limit = try Head.parse("HTTP/1.1 429 Too Many Requests\r\nRateLimit: \"default\"; r=0; t=7\r\n\r\n");
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(7), serverRetryDelay(rate_limit).?);

    const http_date = try Head.parse("HTTP/1.1 503 Service Unavailable\r\nRetry-After: Wed, 21 Oct 2015 07:28:00 GMT\r\n\r\n");
    try std.testing.expect(serverRetryDelay(http_date) == null);
    const none = try Head.parse("HTTP/1.1 500 Internal Server Error\r\n\r\n");
    try std.testing.expect(serverRetryDelay(none) == null);
}

test "retry configuration comes from the backend init options" {
    const retry: RetryConfig = .fromOptions(.{
        .max_retries = @as(usize, 2),
        .retry_initial_delay = std.Io.Duration.fromMilliseconds(5),
        .retry_max_delay = std.Io.Duration.fromSeconds(1),
        .max_hold = std.Io.Duration.fromSeconds(120),
        .throttle_budget = std.Io.Duration.fromSeconds(300),
    });
    try std.testing.expectEqual(@as(usize, 2), retry.max_retries);
    try std.testing.expectEqual(std.Io.Duration.fromMilliseconds(5), retry.initial_delay);
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(1), retry.max_delay);
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(120), retry.max_hold);
    try std.testing.expectEqual(std.Io.Duration.fromSeconds(300), retry.throttle_budget);
}

test "the request authority is the rate-limit scope" {
    const uri: std.Uri = try .parse("https://huggingface.co/api/models/x");
    try std.testing.expectEqualStrings("huggingface.co", authorityOf(uri));
}
