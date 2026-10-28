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

/// The retry knobs of one remote backend: its `Options.retry`.
pub const Retry = struct {
    /// Retries per request for failures that are not rate limiting.
    max_retries: usize = 5,
    initial_delay: std.Io.Duration = .fromMilliseconds(500),
    /// Longest backoff before a request's next attempt after a failure that
    /// is not a throttle, a server-named delay included. A throttle waits
    /// out the backend's hold instead (`max_hold`).
    max_delay: std.Io.Duration = .fromSeconds(30),
    /// Longest hold one throttle may arm over every request of the backend,
    /// a server-named delay included.
    max_hold: std.Io.Duration = .fromSeconds(120),
    /// A throttling episode with no clean window longer than this fails the
    /// backend's requests.
    throttle_budget: std.Io.Duration = .fromSeconds(300),
    /// An exchange that receives nothing for this long is canceled and
    /// retried as a timeout: a peer that stops sending without closing the
    /// connection would otherwise hold its reader, and the load waiting for
    /// it, forever. Body readers count what they receive
    /// (`Attempt.progress`); an exchange that counts nothing (a HEAD, a token
    /// exchange) must finish within it.
    stall_timeout: std.Io.Duration = .fromSeconds(60),
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

/// One per backend instance: what each of its requests needs, its
/// rate-limit state and its read statistics.
pub const Governor = struct {
    io: std.Io,
    client: *std.http.Client,
    /// The backend's name, for log lines.
    name: []const u8,
    /// What a 503 means for this backend: rate limiting on the object stores
    /// (AWS `SlowDown`, GCS), a server failure elsewhere.
    unavailable: ReadFailure,
    retry: Retry,
    stats: AtomicReadStats = .{},
    mutex: std.Io.Mutex = .init,
    hold: Hold = .{},

    pub fn init(io: std.Io, client: *std.http.Client, name: []const u8, unavailable: ReadFailure, retry: Retry) Governor {
        std.debug.assert(retry.initial_delay.nanoseconds >= 0);
        std.debug.assert(retry.max_delay.nanoseconds >= retry.initial_delay.nanoseconds);
        std.debug.assert(retry.max_hold.nanoseconds >= retry.initial_delay.nanoseconds);
        return .{ .io = io, .client = client, .name = name, .unavailable = unavailable, .retry = retry };
    }

    /// Logs the read totals at debug, once, from the backend's `deinit`;
    /// nothing when the backend sent no request. Holds, retries and failures
    /// are logged at warn as they happen.
    pub fn logTotals(self: *const Governor) void {
        const stats = &self.stats;
        const requests = stats.physical_requests.load(.acquire);
        if (requests == 0) return;
        // Under the VFS's scope, which examples/io shows at debug.
        std.log.scoped(.@"zml/vfs").debug("{s} read totals: requests={d}, bytes={Bi:.2}, retries={d}, transient_retries={d}, timeouts={d}, server_failures={d}, throttles={d}, holds={d}, hold_wait_ms={d:.3}, retry_delay_ms={d:.3}", .{
            self.name,
            requests,
            stats.physical_bytes.load(.acquire),
            stats.retries.load(.acquire),
            stats.transient_retries.load(.acquire),
            stats.timeouts.load(.acquire),
            stats.server_failures.load(.acquire),
            stats.throttles.load(.acquire),
            stats.holds.load(.acquire),
            @as(f64, @floatFromInt(stats.hold_wait_ns.load(.acquire))) / std.time.ns_per_ms,
            @as(f64, @floatFromInt(stats.retry_delay_ns.load(.acquire))) / std.time.ns_per_ms,
        });
    }

    /// A failed open, stat or access of `path` as `OpenError` reports it.
    pub fn openFailure(self: *const Governor, path: []const u8, err: anyerror) OpenError {
        return switch (err) {
            error.FileNotFound => error.FileNotFound,
            error.PermissionDenied => error.PermissionDenied,
            error.Canceled => error.Canceled,
            // Logged by `exchange`, with the status head.
            error.RequestFailed => error.Unexpected,
            error.OutOfMemory => error.SystemResources,
            else => {
                log.err("{s}: looking up {s} failed: {t}", .{ self.name, path, err });
                return error.Unexpected;
            },
        };
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

/// One attempt as a backend's request hook sees it.
pub const Attempt = struct {
    /// The backend and the request as `perform` was given them, which
    /// `exchange` sends and logs with.
    governor: *Governor,
    spec: RequestSpec,
    /// The status the previous attempt returned, for a hook that reacts to
    /// one (an expired signed URL answering 403 after a long hold).
    previous_status: ?std.http.Status = null,
    /// Bytes received so far, for a hook whose body is long: `exchange`
    /// cancels an exchange this counter does not move for
    /// `Retry.stall_timeout`.
    progress: *std.atomic.Value(u64),
};

/// How often a watched exchange's progress is checked, per stall timeout.
const stall_checks = 12;

/// Identity of one request, for the logs and for the classification.
pub const RequestSpec = struct {
    /// Object identity for log lines.
    target: []const u8,
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
/// the hold it armed; any other retryable failure, a stall
/// (`Retry.stall_timeout`) included, sleeps the per-request backoff and charges
/// one of `max_retries`. A cancel of the caller propagates.
pub fn perform(
    comptime T: type,
    governor: *Governor,
    spec: RequestSpec,
    context: anytype,
    comptime attemptFn: fn (@TypeOf(context), Attempt) anyerror!Outcome(T),
) anyerror!T {
    var charged: usize = 0;
    var previous_status: ?std.http.Status = null;
    while (true) {
        try admit(governor, spec);
        governor.stats.recordAttempt();
        var progress: std.atomic.Value(u64) = .init(0);
        switch (try attemptFn(context, .{ .governor = governor, .spec = spec, .previous_status = previous_status, .progress = &progress })) {
            .done => |value| return value,
            .retry => |failure| {
                governor.stats.recordFailure(failure.failure);
                previous_status = failure.status;
                if (failure.failure == .throttle) {
                    // Not charged to the retry budget: the hold is what
                    // reduces the traffic, and the episode budget bounds it.
                    try reportThrottle(governor, spec, failure.delay);
                    continue;
                }
                if (charged >= governor.retry.max_retries) return error.RetriesExhausted;
                charged += 1;
                governor.stats.recordRetry();
                const retry = governor.retry;
                const delay = if (failure.delay) |named|
                    clampDelay(named, .fromNanoseconds(0), retry.max_delay)
                else
                    fullJitterDelay(governor.io, retry.initial_delay, retry.max_delay, charged);
                governor.stats.recordRetryDelay(delay);
                try governor.io.sleep(delay, .awake);
            },
        }
    }
}

/// Runs one exchange on its own unit of concurrency while this task watches
/// the attempt's progress, and cancels it after `Retry.stall_timeout` without
/// progress. The exchange returns a canceled read as `Canceled`, unlogged,
/// so the outcome of a stalled exchange is decided here, as a timeout.
/// Without a unit of concurrency the exchange runs unwatched.
fn watched(
    comptime T: type,
    context: anytype,
    comptime exchangeFn: fn (@TypeOf(context), Attempt) anyerror!Outcome(T),
    attempt: Attempt,
) anyerror!Outcome(T) {
    const Run = struct {
        result: anyerror!Outcome(T) = undefined,
        done: std.Io.Event = .unset,

        fn run(self: *@This(), io: std.Io, context_: @TypeOf(context), attempt_: Attempt) void {
            self.result = exchangeFn(context_, attempt_);
            self.done.set(io);
        }
    };
    const governor = attempt.governor;
    var run: Run = .{};
    var future = governor.io.concurrent(Run.run, .{ &run, governor.io, context, attempt }) catch |err| switch (err) {
        error.ConcurrencyUnavailable => return exchangeFn(context, attempt),
    };
    const stall_timeout = governor.retry.stall_timeout;
    const check: std.Io.Timeout = .{ .duration = .{
        .raw = .fromNanoseconds(@divTrunc(stall_timeout.nanoseconds, stall_checks)),
        .clock = .awake,
    } };
    var seen = attempt.progress.load(.monotonic);
    var quiet: usize = 0;
    while (quiet < stall_checks) {
        run.done.waitTimeout(governor.io, check) catch |err| switch (err) {
            error.Timeout => {
                const now = attempt.progress.load(.monotonic);
                quiet = if (now == seen) quiet + 1 else 0;
                seen = now;
                continue;
            },
            error.Canceled => {
                future.cancel(governor.io);
                // An exchange already past its last cancelation point
                // finished: its value is the caller's, and the cancel
                // reaches the caller at its next cancelation point.
                if (run.result) |outcome| switch (outcome) {
                    .done => {
                        governor.io.recancel();
                        return outcome;
                    },
                    .retry => {},
                } else |_| {}
                return error.Canceled;
            },
        };
        future.await(governor.io);
        return run.result;
    }
    future.cancel(governor.io);
    // It may have finished while being canceled.
    if (run.result) |outcome| switch (outcome) {
        .done => return outcome,
        .retry => {},
    } else |_| {}
    log.warn("{s}: {s} received nothing for {d:.1}s; retrying", .{ governor.name, attempt.spec.target, @as(f64, @floatFromInt(stall_timeout.nanoseconds)) / std.time.ns_per_s });
    return .{ .retry = .{ .failure = .timeout } };
}

/// Waits out the scope's hold. Returns at once when nothing is held; fails
/// with `error.RateLimited` when the episode outlived its budget. Waiters
/// wake at the deadline plus their own jitter, so the readers of one backend
/// do not all fire in the same instant, and re-check, since the hold may have
/// been extended while they slept.
fn admit(governor: *Governor, spec: RequestSpec) error{ Canceled, RateLimited }!void {
    while (true) {
        const deadline = deadline: {
            governor.mutex.lockUncancelable(governor.io);
            defer governor.mutex.unlock(governor.io);
            const hold = governor.holdFor(spec.key);
            const now: std.Io.Timestamp = .now(governor.io, .awake);
            refresh(hold, now);
            if (hold.exhausted) return error.RateLimited;
            const until = hold.until orelse return;
            if (until.nanoseconds <= now.nanoseconds) return;
            break :deadline until;
        };
        const jitter = fullJitterDelay(governor.io, governor.retry.initial_delay, governor.retry.initial_delay, 0);
        const started: std.Io.Timestamp = .now(governor.io, .awake);
        try std.Io.Clock.Timestamp.wait(deadline.addDuration(jitter).withClock(.awake), governor.io);
        governor.stats.recordHoldWait(started.durationTo(.now(governor.io, .awake)));
    }
}

/// Arms or extends the scope's hold after a throttle. `named` is the delay
/// the server asked for, if any; it is honoured up to `max_hold` and never
/// below `initial_delay`, because `Retry-After: 0` and Hugging Face's
/// `RateLimit; t=0` parse to zero and a zero hold is a hot loop now that a
/// throttle does not charge a retry.
fn reportThrottle(governor: *Governor, spec: RequestSpec, named: ?std.Io.Duration) error{RateLimited}!void {
    const retry = governor.retry;
    governor.mutex.lockUncancelable(governor.io);
    defer governor.mutex.unlock(governor.io);
    const hold = governor.holdFor(spec.key);
    const now: std.Io.Timestamp = .now(governor.io, .awake);
    refresh(hold, now);

    if (hold.episode_started) |started| {
        if (started.durationTo(now).nanoseconds > retry.throttle_budget.nanoseconds) {
            hold.exhausted = true;
            // The caller reports the failure; this is the one line that
            // says why its request never ran.
            log.warn("{s}: rate limited for longer than the throttle budget while reading {s}; giving up", .{ governor.name, spec.target });
            return error.RateLimited;
        }
    } else {
        hold.episode_started = now;
    }

    const holding = if (hold.until) |until| now.nanoseconds < until.nanoseconds else false;
    if (!holding) {
        hold.consecutive += 1;
        const backoff = fullJitterDelay(governor.io, retry.initial_delay, retry.max_delay, hold.consecutive - 1);
        var delay = backoff;
        if (named) |value| {
            if (value.nanoseconds > delay.nanoseconds) delay = value;
        }
        hold.length = clampDelay(delay, retry.initial_delay, retry.max_hold);
        hold.until = now.addDuration(hold.length);
        log.warn("{s}: {s} is rate limiting; holding its requests for {d:.3}s", .{
            governor.name,
            spec.target,
            @as(f64, @floatFromInt(hold.length.toNanoseconds())) / std.time.ns_per_s,
        });
        governor.stats.recordHold();
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

/// The hops `Redirects.follow` takes: std's default.
pub const max_redirects = 3;

/// What happens to a redirect: the client chases it (`max_redirects` hops),
/// `consume` receives it because the caller walks the chain itself, or it
/// fails the request.
pub const Redirects = enum {
    follow,
    surface,
    forbid,

    fn behavior(self: Redirects) std.http.Client.Request.RedirectBehavior {
        return switch (self) {
            .follow => .init(max_redirects),
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
    /// An open, stat or access: an absent or denied object fails the
    /// request with its `OpenError` before `consume`, unlogged.
    open: bool = false,
    redirects: Redirects = .forbid,
    /// Where `receiveHead` keeps a followed redirect's Locations and merged
    /// paths, hop after hop (`Redirects.follow`); the head itself is in the
    /// connection's buffer while `consume` runs.
    head_buffer: []u8,
};

/// One HTTP exchange of `attempt`'s request: send, receive the head,
/// classify it, and hand a 2xx (or a redirect `Redirects.surface` hands
/// back) to `consume` while the request is alive. Connection, send and
/// receive failures and every classified status become `.retry`; an
/// unclassified status fails with `error.RequestFailed`, except that an open
/// request (`ExchangeOptions.open`) fails an absent or denied object with
/// `FileNotFound` or `PermissionDenied` first, unlogged. The exchange alone
/// is watched (`Retry.stall_timeout`); a governed request its hook runs
/// first (a re-resolve, a token refresh) is watched by its own exchange.
pub fn exchange(
    comptime T: type,
    attempt: Attempt,
    uri: std.Uri,
    options: ExchangeOptions,
    context: anytype,
    comptime consume: fn (@TypeOf(context), *std.http.Client.Response) anyerror!T,
) anyerror!Outcome(T) {
    const Exchange = struct {
        uri: std.Uri,
        options: ExchangeOptions,
        context: @TypeOf(context),

        fn run(self: @This(), current: Attempt) anyerror!Outcome(T) {
            return roundTrip(T, current, self.uri, self.options, self.context, consume);
        }
    };
    return watched(T, Exchange{ .uri = uri, .options = options, .context = context }, Exchange.run, attempt);
}

fn roundTrip(
    comptime T: type,
    attempt: Attempt,
    uri: std.Uri,
    options: ExchangeOptions,
    context: anytype,
    comptime consume: fn (@TypeOf(context), *std.http.Client.Response) anyerror!T,
) anyerror!Outcome(T) {
    const spec = attempt.spec;
    var req = attempt.governor.client.request(options.method, uri, .{
        .redirect_behavior = options.redirects.behavior(),
        .headers = options.headers,
        .extra_headers = options.extra_headers,
    }) catch |err| switch (err) {
        error.Timeout => return retryable(T, attempt, "connect", err, .timeout),
        error.ConnectionRefused,
        error.ConnectionResetByPeer,
        error.HostUnreachable,
        error.NetworkUnreachable,
        error.NetworkDown,
        error.NameServerFailure,
        => return retryable(T, attempt, "connect", err, .transient),
        // A stalled exchange that `watched` canceled, or the caller's cancel.
        error.Canceled => return err,
        error.OutOfMemory => return err,
        else => return fatal(attempt, "connect", err),
    };
    defer req.deinit();

    // A request that was not sent whole leaves its connection unusable;
    // `deinit` would otherwise pool it, its reader still `ready`.
    if (options.payload) |payload| {
        req.sendBodyComplete(payload) catch |err| switch (err) {
            error.WriteFailed => {
                req.connection.?.closing = true;
                if (canceled(&req)) return error.Canceled;
                return retryable(T, attempt, "send body", err, .transient);
            },
        };
    } else {
        req.sendBodiless() catch |err| switch (err) {
            error.WriteFailed => {
                req.connection.?.closing = true;
                if (canceled(&req)) return error.Canceled;
                return retryable(T, attempt, "send headers", err, .transient);
            },
        };
    }

    var res = req.receiveHead(options.head_buffer) catch |err| {
        // A failed read already closed it; a redirect re-sends on a new
        // connection that a failure leaves unusable too (or null).
        if (req.connection) |connection| connection.closing = true;
        if (canceled(&req)) return error.Canceled;
        switch (err) {
            error.Timeout => return retryable(T, attempt, "receive headers", err, .timeout),
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
            => return retryable(T, attempt, "receive headers", err, .transient),
            // A stalled exchange that `watched` canceled, or the caller's cancel.
            error.Canceled => return err,
            error.OutOfMemory => return err,
            else => return fatal(attempt, "receive headers", err),
        }
    };

    if (options.open) {
        if (openStatus(res.head.status)) |err| return err;
    }
    const class = res.head.status.class();
    if (class != .success and !(class == .redirect and options.redirects == .surface)) {
        for (spec.retry_once) |status| {
            if (status != res.head.status) continue;
            log.warn("{s}: {s} answered {d}; retrying once with a fresh request", .{
                attempt.governor.name,
                spec.target,
                @backingInt(res.head.status),
            });
            return .{ .retry = .{ .failure = .transient, .status = res.head.status } };
        }
        const failure = classifyStatus(res.head.status, attempt.governor.unavailable) orelse {
            log.err("{s}: {s} failed: {s}", .{ attempt.governor.name, spec.target, res.head.bytes });
            return error.RequestFailed;
        };
        log.warn("{s}: {s} failed: {s}", .{ attempt.governor.name, spec.target, res.head.bytes });
        return .{ .retry = .{
            .failure = failure,
            .delay = serverRetryDelay(res.head),
            .status = res.head.status,
        } };
    }

    return .{ .done = consume(context, &res) catch |err| switch (err) {
        error.EndOfStream, error.ReadFailed => {
            if (canceled(&req)) return error.Canceled;
            return retryable(T, attempt, "read body", err, .transient);
        },
        else => return err,
    } };
}

/// Whether the exchange's read or write failed because it was canceled:
/// std's stream keeps the cancel and reports a failed read or write.
fn canceled(req: *const std.http.Client.Request) bool {
    const connection = req.connection orelse return false;
    if (connection.stream_reader.err) |err| return err == error.Canceled;
    if (connection.stream_writer.err) |err| return err == error.Canceled;
    return false;
}

fn retryable(
    comptime T: type,
    attempt: Attempt,
    stage: []const u8,
    err: anyerror,
    failure: ReadFailure,
) Outcome(T) {
    log.warn("{s}: {s} for {s} failed: {}", .{ attempt.governor.name, stage, attempt.spec.target, err });
    return .{ .retry = .{ .failure = failure } };
}

fn fatal(attempt: Attempt, stage: []const u8, err: anyerror) anyerror {
    log.err("{s}: {s} for {s} failed: {}", .{ attempt.governor.name, stage, attempt.spec.target, err });
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

/// How opening, stat-ing or probing (`access`) a remote path fails, on every
/// backend: an absent object is `FileNotFound` (404, 410), a forbidden one
/// `PermissionDenied` (401, 403), a canceled task `Canceled`, and a failed
/// allocation `SystemResources`. Any other failure is logged once, by
/// `exchange` or by `Governor.openFailure`, and surfaces as `Unexpected`.
pub const OpenError = error{ FileNotFound, PermissionDenied, Canceled, SystemResources, Unexpected };

/// The error a status means for the path an open request
/// (`ExchangeOptions.open`) asked about; null when the status is not one.
fn openStatus(status: std.http.Status) ?error{ FileNotFound, PermissionDenied } {
    return switch (status) {
        .not_found, .gone => error.FileNotFound,
        .unauthorized, .forbidden => error.PermissionDenied,
        else => null,
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

const test_spec: RequestSpec = .{ .target = "object" };

fn testGovernor(initial_ms: i64, max_delay_ms: i64, max_hold_ms: i64, budget_ms: i64) Governor {
    // No attempt in these tests reaches the network: no client.
    return .init(std.testing.io, undefined, "test", .throttle, .{
        .max_retries = 3,
        .initial_delay = .fromMilliseconds(initial_ms),
        .max_delay = .fromMilliseconds(max_delay_ms),
        .max_hold = .fromMilliseconds(max_hold_ms),
        .throttle_budget = .fromMilliseconds(budget_ms),
    });
}

test "a throttle arms a hold of at least the initial delay, even when the server names zero" {
    var governor = testGovernor(20, 20, 1000, 300_000);

    // `Retry-After: 0` and Hugging Face's `RateLimit; t=0` parse to zero.
    try reportThrottle(&governor, test_spec, .fromSeconds(0));
    const hold = governor.hold;
    try std.testing.expect(hold.until != null);
    try std.testing.expectEqual(@as(i96, 20 * std.time.ns_per_ms), hold.length.nanoseconds);
    try std.testing.expectEqual(@as(usize, 1), hold.consecutive);
    try std.testing.expect(hold.episode_started != null);
    try std.testing.expectEqual(@as(u64, 1), governor.stats.holds.load(.acquire));
}

test "a longer named delay extends a running hold and a shorter one does not" {
    var governor = testGovernor(10, 10, 1000, 300_000);

    try reportThrottle(&governor, test_spec, null);
    const armed = governor.hold.until.?;

    // A request that was already in flight got its own answer, with a later
    // reset: the hold extends.
    try reportThrottle(&governor, test_spec, .fromMilliseconds(500));
    const extended = governor.hold.until.?;
    try std.testing.expect(extended.nanoseconds > armed.nanoseconds);

    // A shorter one never shortens it, and neither arms a second hold.
    try reportThrottle(&governor, test_spec, .fromMilliseconds(1));
    try std.testing.expectEqual(extended.nanoseconds, governor.hold.until.?.nanoseconds);
    try std.testing.expectEqual(@as(usize, 1), governor.hold.consecutive);
    try std.testing.expectEqual(@as(u64, 1), governor.stats.holds.load(.acquire));
}

test "a named delay is capped by max_hold" {
    var governor = testGovernor(10, 10, 50, 300_000);

    try reportThrottle(&governor, test_spec, .fromSeconds(3600));
    try std.testing.expectEqual(@as(i96, 50 * std.time.ns_per_ms), governor.hold.length.nanoseconds);
}

test "a second throttle inside the clean window doubles the backoff base" {
    var governor = testGovernor(10, 1000, 10_000, 300_000);

    try reportThrottle(&governor, test_spec, null);
    try std.testing.expectEqual(@as(i96, 10 * std.time.ns_per_ms), governor.hold.length.nanoseconds);

    // The hold has passed but the clean window has not: rewind it by hand
    // rather than sleeping through a race.
    const now: std.Io.Timestamp = .now(std.testing.io, .awake);
    governor.hold.until = now.subDuration(.fromMilliseconds(1));
    try reportThrottle(&governor, test_spec, null);
    try std.testing.expectEqual(@as(usize, 2), governor.hold.consecutive);
    // Full jitter over twice the base, floored at the initial delay.
    try std.testing.expect(governor.hold.length.nanoseconds >= 10 * std.time.ns_per_ms);
    try std.testing.expect(governor.hold.length.nanoseconds <= 20 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(u64, 2), governor.stats.holds.load(.acquire));
}

test "a clean window as long as the hold ends the episode" {
    var governor = testGovernor(10, 1000, 10_000, 300_000);

    try reportThrottle(&governor, test_spec, null);
    const length = governor.hold.length;
    const now: std.Io.Timestamp = .now(std.testing.io, .awake);
    governor.hold.until = now.subDuration(length).subDuration(.fromMilliseconds(1));

    try reportThrottle(&governor, test_spec, null);
    try std.testing.expectEqual(@as(usize, 1), governor.hold.consecutive);
    try std.testing.expectEqual(@as(i96, 10 * std.time.ns_per_ms), governor.hold.length.nanoseconds);
}

test "an episode longer than the throttle budget fails requests until a clean window" {
    var governor = testGovernor(10, 10, 10_000, 50);

    try reportThrottle(&governor, test_spec, .fromSeconds(30));
    const now: std.Io.Timestamp = .now(std.testing.io, .awake);
    governor.hold.episode_started = now.subDuration(.fromMilliseconds(51));

    try std.testing.expectError(error.RateLimited, reportThrottle(&governor, test_spec, null));
    try std.testing.expect(governor.hold.exhausted);
    // Every other request of this backend fails at once, without waiting the
    // hold out.
    try std.testing.expectError(error.RateLimited, admit(&governor, test_spec));

    // The next clean window clears the episode.
    governor.hold.until = now.subDuration(governor.hold.length).subDuration(.fromMilliseconds(1));
    try admit(&governor, test_spec);
    try std.testing.expect(governor.hold.until == null);
    try std.testing.expect(!governor.hold.exhausted);
}

test "admit returns at once when nothing is held" {
    var governor = testGovernor(10, 10, 1000, 300_000);

    const started: std.Io.Timestamp = .now(std.testing.io, .awake);
    try admit(&governor, test_spec);
    try std.testing.expect(started.durationTo(.now(std.testing.io, .awake)).nanoseconds < 5 * std.time.ns_per_ms);
    try std.testing.expectEqual(@as(u64, 0), governor.stats.hold_wait_ns.load(.acquire));
}

test "admit waits a hold out and counts the wait" {
    var governor = testGovernor(1, 1, 1000, 300_000);

    try reportThrottle(&governor, test_spec, .fromMilliseconds(15));
    const started: std.Io.Timestamp = .now(std.testing.io, .awake);
    try admit(&governor, test_spec);
    try std.testing.expect(started.durationTo(.now(std.testing.io, .awake)).nanoseconds >= 10 * std.time.ns_per_ms);
    try std.testing.expect(governor.stats.hold_wait_ns.load(.acquire) > 0);
}

test "a waiter blocked on a hold is cancelable" {
    const io = std.testing.io;
    var governor = testGovernor(1, 1, 100_000, 300_000);
    try reportThrottle(&governor, test_spec, .fromSeconds(60));

    var result: std.atomic.Value(u16) = .init(0);
    var started: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(governor_: *Governor, started_: *std.Io.Event, result_: *std.atomic.Value(u16)) void {
            started_.set(governor_.io);
            admit(governor_, test_spec) catch |err| {
                result_.store(@intFromError(err), .release);
                return;
            };
        }
    }.run, .{ &governor, &started, &result });
    try started.wait(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    group.cancel(io);
    try std.testing.expectEqual(@intFromError(error.Canceled), result.load(.acquire));
}

test "only rate limiting holds: a timeout and a server failure keep the per-request budget" {
    var governor = testGovernor(0, 0, 1000, 300_000);

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
    try std.testing.expectEqual(@as(usize, 3), try perform(usize, &governor, test_spec, &attempts, Attempts.run));
    try std.testing.expect(governor.hold.until == null);
    try std.testing.expectEqual(@as(u64, 0), governor.stats.holds.load(.acquire));
    try std.testing.expectEqual(@as(u64, 2), governor.stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 3), governor.stats.physical_requests.load(.acquire));
}

test "a throttle retries without charging the retry budget" {
    var governor = testGovernor(0, 0, 1000, 300_000);

    const Attempts = struct {
        seen: usize = 0,
        fn run(self: *@This(), current: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            // The retry budget is three; ten throttles do not exhaust it.
            if (self.seen <= 10) return .{ .retry = .{ .failure = .throttle, .status = .too_many_requests } };
            try std.testing.expectEqual(@as(?std.http.Status, .too_many_requests), current.previous_status);
            return .{ .done = self.seen };
        }
    };
    var attempts: Attempts = .{};
    try std.testing.expectEqual(@as(usize, 11), try perform(usize, &governor, test_spec, &attempts, Attempts.run));
    try std.testing.expectEqual(@as(u64, 0), governor.stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 10), governor.stats.throttles.load(.acquire));
    try std.testing.expectEqual(@as(u64, 11), governor.stats.physical_requests.load(.acquire));
    try std.testing.expect(governor.stats.holds.load(.acquire) >= 1);
}

test "the retry budget still bounds failures that are not rate limiting" {
    var governor = testGovernor(0, 0, 1000, 300_000);

    const Attempts = struct {
        seen: usize = 0,
        fn run(self: *@This(), _: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            return .{ .retry = .{ .failure = .transient } };
        }
    };
    var attempts: Attempts = .{};
    try std.testing.expectError(error.RetriesExhausted, perform(usize, &governor, test_spec, &attempts, Attempts.run));
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

test "absent and denied statuses map to one error each" {
    try std.testing.expect(openStatus(.not_found).? == error.FileNotFound);
    try std.testing.expect(openStatus(.gone).? == error.FileNotFound);
    try std.testing.expect(openStatus(.unauthorized).? == error.PermissionDenied);
    try std.testing.expect(openStatus(.forbidden).? == error.PermissionDenied);
    for ([_]std.http.Status{ .ok, .partial_content, .bad_request, .too_many_requests, .internal_server_error, .service_unavailable }) |status| {
        try std.testing.expect(openStatus(status) == null);
    }

    // The contract's own errors pass through unlogged; any other is logged
    // at err, which the test runner would count, so it is not exercised.
    const governor = testGovernor(10, 10, 1000, 300_000);
    try std.testing.expect(governor.openFailure("object", error.FileNotFound) == error.FileNotFound);
    try std.testing.expect(governor.openFailure("object", error.PermissionDenied) == error.PermissionDenied);
    try std.testing.expect(governor.openFailure("object", error.Canceled) == error.Canceled);
    try std.testing.expect(governor.openFailure("object", error.OutOfMemory) == error.SystemResources);
    try std.testing.expect(governor.openFailure("object", error.RequestFailed) == error.Unexpected);
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

test "a server-named retry delay sleeps at most max_delay" {
    var governor = testGovernor(1, 20, 20, 1000);
    const Failing = struct {
        seen: usize = 0,
        fn attempt(self: *@This(), _: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            // A failing server that asks for an hour.
            if (self.seen == 1) return .{ .retry = .{ .failure = .server_failure, .delay = .fromSeconds(3600) } };
            return .{ .done = self.seen };
        }
    };
    var failing: Failing = .{};
    try std.testing.expectEqual(@as(usize, 2), try perform(usize, &governor, test_spec, &failing, Failing.attempt));
    try std.testing.expectEqual(@as(u64, 20 * std.time.ns_per_ms), governor.stats.retry_delay_ns.load(.acquire));
}

test "an attempt that receives nothing is canceled and retried as a timeout" {
    var governor: Governor = .init(std.testing.io, undefined, "test", .throttle, .{
        .max_retries = 3,
        .initial_delay = .fromMilliseconds(1),
        .max_delay = .fromMilliseconds(1),
        .max_hold = .fromMilliseconds(1),
        .throttle_budget = .fromSeconds(1),
        .stall_timeout = .fromMilliseconds(120),
    });
    const Hang = struct {
        seen: usize = 0,
        progressed: u64 = 0,
        // Watched as `exchange` watches its round trip.
        fn attempt(self: *@This(), current: Attempt) anyerror!Outcome(usize) {
            return watched(usize, self, receive, current);
        }

        fn receive(self: *@This(), current: Attempt) anyerror!Outcome(usize) {
            self.seen += 1;
            // The first exchange hangs, the second one keeps receiving and
            // takes longer than the stall timeout overall.
            if (self.seen == 1) {
                try std.testing.io.sleep(.fromSeconds(60), .awake);
                return error.TestUnexpectedResult;
            }
            for (0..6) |_| {
                try std.testing.io.sleep(.fromMilliseconds(40), .awake);
                _ = current.progress.fetchAdd(1, .monotonic);
                self.progressed += 1;
            }
            return .{ .done = self.seen };
        }
    };
    var hang: Hang = .{};
    try std.testing.expectEqual(@as(usize, 2), try perform(usize, &governor, test_spec, &hang, Hang.attempt));
    try std.testing.expectEqual(@as(u64, 6), hang.progressed);
    try std.testing.expectEqual(@as(u64, 1), governor.stats.timeouts.load(.acquire));
}

test "an attempt waiting on a request of its own is not a stall" {
    var governor: Governor = .init(std.testing.io, undefined, "test", .throttle, .{
        .max_retries = 3,
        .initial_delay = .fromMilliseconds(1),
        .max_delay = .fromMilliseconds(1),
        .max_hold = .fromMilliseconds(1),
        .throttle_budget = .fromSeconds(1),
        .stall_timeout = .fromMilliseconds(120),
    });
    const Nested = struct {
        fn attempt(self: *@This(), current: Attempt) anyerror!Outcome(usize) {
            _ = try perform(usize, current.governor, test_spec, self, wait);
            return .{ .done = 1 };
        }

        // A re-resolve or token refresh waiting out a hold, outside any
        // exchange, for longer than the stall timeout.
        fn wait(_: *@This(), _: Attempt) anyerror!Outcome(usize) {
            try std.testing.io.sleep(.fromMilliseconds(300), .awake);
            return .{ .done = 0 };
        }
    };
    var nested: Nested = .{};
    try std.testing.expectEqual(@as(usize, 1), try perform(usize, &governor, test_spec, &nested, Nested.attempt));
    try std.testing.expectEqual(@as(u64, 0), governor.stats.timeouts.load(.acquire));
    try std.testing.expectEqual(@as(u64, 2), governor.stats.physical_requests.load(.acquire));
}

test "an exchange canceled while it waits for the head returns Canceled, not a retry" {
    const MockServer = @import("mock_server.zig").MockServer;
    const io = std.testing.io;

    // The GET waits unanswered for a second one that never comes.
    var server = try MockServer.init(io, "object", .{ .barrier_gets = 2 });
    defer server.deinit(io);
    try server.start(io);

    var client: std.http.Client = .{ .allocator = std.testing.allocator, .io = io };
    defer client.deinit();
    var governor: Governor = .init(io, &client, "test", .throttle, .{});
    var url_buffer: [64]u8 = undefined;
    const uri: std.Uri = try .parse(try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()}));

    const Exchange = struct {
        // Unwatched, so that the cancel lands in the read rather than in `watched`.
        fn run(governor_: *Governor, uri_: std.Uri, result: *anyerror!Outcome(void)) void {
            var progress: std.atomic.Value(u64) = .init(0);
            var head_buffer: [1024]u8 = undefined;
            result.* = roundTrip(void, .{ .governor = governor_, .spec = test_spec, .progress = &progress }, uri_, .{
                .extra_headers = &.{.{ .name = "Range", .value = "bytes=0-5" }},
                .head_buffer = &head_buffer,
            }, {}, consume);
        }

        fn consume(_: void, _: *std.http.Client.Response) anyerror!void {}
    };
    var result: anyerror!Outcome(void) = .{ .done = {} };
    var group: std.Io.Group = .init;
    try group.concurrent(io, Exchange.run, .{ &governor, uri, &result });
    try io.sleep(.fromMilliseconds(50), .awake);
    group.cancel(io);
    try std.testing.expectError(error.Canceled, result);

    try server.stop(io);
}

test "the request authority is the rate-limit scope" {
    const uri: std.Uri = try .parse("https://huggingface.co/api/models/x");
    try std.testing.expectEqualStrings("huggingface.co", authorityOf(uri));
}
