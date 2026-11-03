//! One governed HTTP request loop for every backend: send, classify, and
//! retry a retryable failure after a per-request full-jitter backoff, or
//! after the delay the server names. A backend's open, stat and read
//! failures map to one error contract (`OpenError`,
//! `Governor.readFailure`), each failure logged once.

const std = @import("std");

const log = @import("base.zig").scopedLog(.@"zml/vfs/request");

/// The retry knobs of one remote backend: its `Options.retry`.
pub const Retry = struct {
    /// Retries per request.
    max_retries: usize = 5,
    initial_delay: std.Io.Duration = .fromMilliseconds(500),
    /// Longest backoff before a request's next attempt, a server-named delay
    /// included.
    max_delay: std.Io.Duration = .fromSeconds(30),
};

/// One per backend instance: what each of its requests needs and its read
/// statistics.
pub const Governor = struct {
    pub const Stats = struct {
        physical_requests: std.atomic.Value(u64) = .init(0),
        physical_bytes: std.atomic.Value(u64) = .init(0),
        retries: std.atomic.Value(u64) = .init(0),
        transient_retries: std.atomic.Value(u64) = .init(0),
        timeouts: std.atomic.Value(u64) = .init(0),
        server_failures: std.atomic.Value(u64) = .init(0),
        throttles: std.atomic.Value(u64) = .init(0),
        retry_delay_ns: std.atomic.Value(u64) = .init(0),

        fn recordAttempt(self: *Stats) void {
            _ = self.physical_requests.fetchAdd(1, .monotonic);
        }

        pub fn recordSuccess(self: *Stats, request_size: usize) void {
            _ = self.physical_bytes.fetchAdd(@intCast(request_size), .monotonic);
        }

        fn recordFailure(self: *Stats, failure: ReadFailure) void {
            switch (failure) {
                .transient => _ = self.transient_retries.fetchAdd(1, .monotonic),
                .timeout => _ = self.timeouts.fetchAdd(1, .monotonic),
                .server_failure => _ = self.server_failures.fetchAdd(1, .monotonic),
                .throttle => _ = self.throttles.fetchAdd(1, .monotonic),
            }
        }

        fn recordRetry(self: *Stats) void {
            _ = self.retries.fetchAdd(1, .monotonic);
        }

        fn recordRetryDelay(self: *Stats, delay: std.Io.Duration) void {
            const delay_ns: u64 = @intCast(@max(delay.nanoseconds, 0));
            _ = self.retry_delay_ns.fetchAdd(delay_ns, .monotonic);
        }
    };

    io: std.Io,
    client: *std.http.Client,
    /// The backend's name, for log lines.
    name: []const u8,
    /// What a 503 means for this backend: rate limiting on the object stores
    /// (AWS `SlowDown`, GCS), a server failure elsewhere.
    unavailable: ReadFailure,
    retry: Retry,
    stats: Stats = .{},

    pub fn init(io: std.Io, client: *std.http.Client, name: []const u8, unavailable: ReadFailure, retry: Retry) Governor {
        std.debug.assert(retry.initial_delay.nanoseconds >= 0);
        std.debug.assert(retry.max_delay.nanoseconds >= retry.initial_delay.nanoseconds);
        return .{ .io = io, .client = client, .name = name, .unavailable = unavailable, .retry = retry };
    }

    /// Logs the read totals at debug, once, from the backend's `deinit`;
    /// nothing when the backend sent no request. Retries and failures are
    /// logged at warn as they happen.
    pub fn logTotals(self: *const Governor) void {
        const stats = &self.stats;
        const requests = stats.physical_requests.load(.acquire);
        if (requests == 0) return;
        // Under the VFS's scope, `zml/vfs`.
        std.log.scoped(.@"zml/vfs").debug("{s} read totals: requests={d}, bytes={Bi:.2}, retries={d}, transient_retries={d}, timeouts={d}, server_failures={d}, throttles={d}, retry_delay_ms={d:.3}", .{
            self.name,
            requests,
            stats.physical_bytes.load(.acquire),
            stats.retries.load(.acquire),
            stats.transient_retries.load(.acquire),
            stats.timeouts.load(.acquire),
            stats.server_failures.load(.acquire),
            stats.throttles.load(.acquire),
            @as(f64, @floatFromInt(stats.retry_delay_ns.load(.acquire))) / std.time.ns_per_ms,
        });
    }

    /// A failed open, stat or access of `path` as `OpenError` reports it.
    pub fn openFailure(self: *const Governor, path: []const u8, err: anyerror) OpenError {
        return switch (err) {
            error.FileNotFound => error.FileNotFound,
            error.PermissionDenied => error.PermissionDenied,
            error.Canceled => error.Canceled,
            // Logged by `exchange`.
            error.RequestFailed => error.Unexpected,
            error.OutOfMemory => error.SystemResources,
            else => {
                log.err("{s}: looking up {s} failed: {t}", .{ self.name, path, err });
                return error.Unexpected;
            },
        };
    }

    /// A failed read of `target` at `offset` as a backend's read reports it:
    /// a canceled task stays `Canceled`, since every wait in the governed
    /// loop is a cancellation point; any other failure is logged once at
    /// err, by `exchange` or here, and is `Unexpected`.
    pub fn readFailure(self: *const Governor, target: []const u8, offset: u64, err: anyerror) error{ Canceled, Unexpected } {
        return switch (err) {
            error.Canceled => error.Canceled,
            // Logged by `exchange`.
            error.RequestFailed => error.Unexpected,
            else => {
                log.err("{s}: reading {s} at {d} failed: {t}", .{ self.name, target, offset, err });
                return error.Unexpected;
            },
        };
    }
};

/// One attempt as a backend's request hook sees it.
pub const Attempt = struct {
    /// The backend and the request as `perform` was given them, which
    /// `exchange` sends and logs with.
    governor: *Governor,
    spec: RequestSpec,
    /// The status the previous attempt returned, for a hook that reacts to
    /// one (an expired signed URL answering 403).
    previous_status: ?std.http.Status = null,
};

/// Identity of one request, for the logs and for the classification.
pub const RequestSpec = struct {
    /// Object identity for log lines.
    target: []const u8,
    /// Statuses this request retries although they are not retryable in
    /// general, because its hook can do something about them: a signed
    /// download URL that expired answers 401 or 403, and the hook
    /// re-resolves it. Charged to the retry budget.
    retry_once: []const std.http.Status = &.{},
};

pub const ReadFailure = enum {
    transient,
    timeout,
    server_failure,
    throttle,
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
/// `exchange`); this drives the classification and the retry budget around
/// it. Every retryable failure, a throttle included, sleeps the per-request
/// backoff, or the delay the server named up to `max_delay`, and charges one
/// of `max_retries`. A cancel of the caller propagates.
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
        governor.stats.recordAttempt();
        switch (try attemptFn(context, .{ .governor = governor, .spec = spec, .previous_status = previous_status })) {
            .done => |value| return value,
            .retry => |failure| {
                governor.stats.recordFailure(failure.failure);
                previous_status = failure.status;
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

fn clampDelay(value: std.Io.Duration, low: std.Io.Duration, high: std.Io.Duration) std.Io.Duration {
    return .fromNanoseconds(std.math.clamp(value.nanoseconds, low.nanoseconds, high.nanoseconds));
}

/// The hops `Redirects.follow` takes: std's default.
pub const max_redirects = 3;

/// What happens to a redirect: the client chases it (`max_redirects` hops),
/// `consume` receives it because `HeadChain` walks the chain itself, or it
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
    authorization: std.http.Client.Request.Headers.Value = .default,
    content_type: std.http.Client.Request.Headers.Value = .default,
    extra_headers: []const std.http.Header = &.{},
    /// Sent as the whole request body; the method must allow one. The
    /// writer uses it as its own buffer, so it must be mutable.
    payload: ?[]u8 = null,
    /// An open, stat or access: an absent or denied object fails the
    /// request with its `OpenError` before `consume`, unlogged.
    open: bool = false,
    redirects: Redirects = .forbid,
};

/// One HTTP exchange of `attempt`'s request: send, receive the head,
/// classify it, and hand a 2xx (or a redirect `Redirects.surface` hands
/// back) to `consume` while the request is alive. Retryable connection,
/// send and receive failures and every classified status become `.retry`;
/// an unclassified status or any other connect or head failure fails with
/// `error.RequestFailed`, logged at err, except that an open request
/// (`ExchangeOptions.open`) fails an absent or denied object with
/// `FileNotFound` or `PermissionDenied` first, unlogged.
pub fn exchange(
    comptime T: type,
    attempt: Attempt,
    uri: std.Uri,
    options: ExchangeOptions,
    context: anytype,
    comptime consume: fn (@TypeOf(context), *std.http.Client.Response) anyerror!T,
) anyerror!Outcome(T) {
    const spec = attempt.spec;
    // A followed redirect keeps each hop's Location and merged path here,
    // 8 KiB of each, as much as a `HeadChain` URL; `req.uri` points into it.
    var redirect_buffer: [2 * max_redirects * 8 * 1024]u8 = undefined;
    var req = attempt.governor.client.request(options.method, uri, .{
        .redirect_behavior = options.redirects.behavior(),
        .headers = .{
            // Range offsets and Content-Length count the stored bytes.
            .accept_encoding = .{ .override = "identity" },
            .authorization = options.authorization,
            .content_type = options.content_type,
        },
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
        // The caller's cancel.
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

    var res = req.receiveHead(if (options.redirects == .follow) &redirect_buffer else &.{}) catch |err| {
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
            // The caller's cancel.
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

fn fatal(attempt: Attempt, stage: []const u8, err: anyerror) error{RequestFailed} {
    log.err("{s}: {s} for {s} failed: {}", .{ attempt.governor.name, stage, attempt.spec.target, err });
    return error.RequestFailed;
}

/// The URLs of a HEAD redirect chain walked by hand: std hands a HEAD its
/// redirect unfollowed. Three buffers in rotation: the current URL, the
/// `Location` it answered (copied out of a head that dies with the
/// response) with the path merged against it, and the resolved next URL.
pub const HeadChain = struct {
    urls: [3][8 * 1024]u8 = undefined,

    pub const End = struct {
        /// `start_url` itself, or a URL in the chain's buffers.
        url: []const u8,
        /// Its `Content-Length`, when it sent one.
        size: ?u64,
    };

    /// Follows `start_url` through at most `max_hops` redirects, absolute
    /// or relative, to the URL that answers a 2xx. Each hop is one governed
    /// HEAD, retried as a GET would be. `authorize` names the credentials
    /// each URL is sent. An absent or denied object fails the walk
    /// (`ExchangeOptions.open`).
    pub fn resolve(
        self: *HeadChain,
        governor: *Governor,
        target: []const u8,
        start_url: []const u8,
        max_hops: usize,
        context: anytype,
        comptime authorize: fn (@TypeOf(context), url: []const u8) std.http.Client.Request.Headers.Value,
    ) !End {
        const Hop = struct {
            context: @TypeOf(context),
            url: []const u8,
            uri: std.Uri,
            /// Where a redirect's `Location` is copied.
            location: []u8,

            const Result = union(enum) { end: ?u64, redirect: usize };

            fn attempt(hop: *@This(), current: Attempt) anyerror!Outcome(Result) {
                return exchange(Result, current, hop.uri, .{
                    .method = .HEAD,
                    .authorization = authorize(hop.context, hop.url),
                    .open = true,
                    .redirects = .surface,
                }, hop, consume);
            }

            fn consume(hop: *@This(), res: *std.http.Client.Response) anyerror!Result {
                if (res.head.status.class() == .success) return .{ .end = res.head.content_length };
                const location = res.head.location orelse return error.HttpRedirectLocationMissing;
                if (location.len > hop.location.len) return error.HttpRedirectLocationOversize;
                @memcpy(hop.location[0..location.len], location);
                return .{ .redirect = location.len };
            }
        };

        var url = start_url;
        // `start_url` is in none of the buffers.
        var current: usize = self.urls.len - 1;
        var hops: usize = 0;
        while (true) {
            const uri: std.Uri = try .parse(url);
            const scratch = (current + 1) % self.urls.len;
            var hop: Hop = .{ .context = context, .url = url, .uri = uri, .location = &self.urls[scratch] };
            const location_len = switch (try perform(Hop.Result, governor, .{ .target = target }, &hop, Hop.attempt)) {
                .end => |size| return .{ .url = url, .size = size },
                .redirect => |len| len,
            };

            if (hops == max_hops) return error.TooManyHttpRedirects;
            hops += 1;
            const output = (current + 2) % self.urls.len;
            var aux: []u8 = &self.urls[scratch];
            const resolved = try uri.resolveInPlace(location_len, &aux);
            url = try std.fmt.bufPrint(&self.urls[output], "{f}", .{resolved});
            current = output;
        }
    }
};

/// Retry classification of a non-2xx status; null when the status is not
/// retried. `unavailable` is what a 503 means for the backend: rate limiting
/// on the object stores (AWS `SlowDown`, GCS), a server failure elsewhere.
fn classifyStatus(status: std.http.Status, unavailable: ReadFailure) ?ReadFailure {
    return switch (status) {
        .request_timeout => .timeout,
        .too_many_requests => .throttle,
        .service_unavailable => unavailable,
        else => if (status.class() == .server_error) .server_failure else null,
    };
}

/// What an open, stat or access of a remote path fails with (vfs.zig's
/// header); `openStatus` maps the statuses, and anything else is logged
/// once, by `exchange` or `Governor.openFailure`.
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
fn serverRetryDelay(head: std.http.Client.Response.Head) ?std.Io.Duration {
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

fn fullJitterDelay(
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

fn testGovernor(initial_ms: i64, max_delay_ms: i64) Governor {
    // No attempt in these tests reaches the network: no client.
    return .init(std.testing.io, undefined, "test", .throttle, .{
        .max_retries = 3,
        .initial_delay = .fromMilliseconds(initial_ms),
        .max_delay = .fromMilliseconds(max_delay_ms),
    });
}

test "a timeout and a server failure each charge one retry" {
    var governor = testGovernor(0, 0);

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
    try std.testing.expectEqual(@as(u64, 2), governor.stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, 3), governor.stats.physical_requests.load(.acquire));
}

test "the retry budget bounds retryable failures" {
    var governor = testGovernor(0, 0);

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

    const governor = testGovernor(10, 10);
    try std.testing.expect(governor.openFailure("object", error.FileNotFound) == error.FileNotFound);
    try std.testing.expect(governor.openFailure("object", error.PermissionDenied) == error.PermissionDenied);
    try std.testing.expect(governor.openFailure("object", error.Canceled) == error.Canceled);
    try std.testing.expect(governor.openFailure("object", error.OutOfMemory) == error.SystemResources);
    try std.testing.expect(governor.openFailure("object", error.RequestFailed) == error.Unexpected);
    try std.testing.expect(governor.openFailure("object", error.InvalidContentRange) == error.Unexpected);
}

test "a failed read is Canceled or Unexpected" {
    const governor = testGovernor(10, 10);
    try std.testing.expect(governor.readFailure("object", 0, error.Canceled) == error.Canceled);
    try std.testing.expect(governor.readFailure("object", 0, error.RequestFailed) == error.Unexpected);
    try std.testing.expect(governor.readFailure("object", 0, error.InvalidContentRange) == error.Unexpected);
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
    var governor = testGovernor(1, 20);
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
        fn run(governor_: *Governor, uri_: std.Uri, result: *anyerror!Outcome(void)) void {
            result.* = exchange(void, .{ .governor = governor_, .spec = test_spec }, uri_, .{
                .extra_headers = &.{.{ .name = "Range", .value = "bytes=0-5" }},
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
