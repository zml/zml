const std = @import("std");

const stdx = @import("stdx");

pub const ReadHints = struct {
    /// Smallest source request expected to avoid excessive per-request cost
    /// or backend rate limiting.
    minimum_request_size: usize = 2 * 1024 * 1024,
    /// The backend benefits from keeping active source calls independent of a
    /// small bounded set of requests retained for downstream DMA completion.
    high_latency: bool = false,
};

pub const read_timing_bucket_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
};

pub const ReadTimingBucket = struct {
    attempts: u64 = 0,
    successes: u64 = 0,
    successful_bytes: u64 = 0,
    ttfb_ns: u64 = 0,
    body_ns: u64 = 0,
    transient_retries: u64 = 0,
    timeouts: u64 = 0,
    server_failures: u64 = 0,
    throttles: u64 = 0,
    retry_delay_ns: u64 = 0,

    fn sub(self: ReadTimingBucket, previous: ReadTimingBucket) ReadTimingBucket {
        return .{
            .attempts = self.attempts -| previous.attempts,
            .successes = self.successes -| previous.successes,
            .successful_bytes = self.successful_bytes -| previous.successful_bytes,
            .ttfb_ns = self.ttfb_ns -| previous.ttfb_ns,
            .body_ns = self.body_ns -| previous.body_ns,
            .transient_retries = self.transient_retries -| previous.transient_retries,
            .timeouts = self.timeouts -| previous.timeouts,
            .server_failures = self.server_failures -| previous.server_failures,
            .throttles = self.throttles -| previous.throttles,
            .retry_delay_ns = self.retry_delay_ns -| previous.retry_delay_ns,
        };
    }
};

pub const ReadStats = struct {
    physical_requests: u64 = 0,
    physical_bytes: u64 = 0,
    retries: u64 = 0,
    transient_retries: u64 = 0,
    timeouts: u64 = 0,
    server_failures: u64 = 0,
    throttles: u64 = 0,
    retry_delay_ns: u64 = 0,
    timing: [read_timing_bucket_sizes.len]ReadTimingBucket = @splat(.{}),

    pub fn sub(self: ReadStats, previous: ReadStats) ReadStats {
        var result: ReadStats = .{
            .physical_requests = self.physical_requests -| previous.physical_requests,
            .physical_bytes = self.physical_bytes -| previous.physical_bytes,
            .retries = self.retries -| previous.retries,
            .transient_retries = self.transient_retries -| previous.transient_retries,
            .timeouts = self.timeouts -| previous.timeouts,
            .server_failures = self.server_failures -| previous.server_failures,
            .throttles = self.throttles -| previous.throttles,
            .retry_delay_ns = self.retry_delay_ns -| previous.retry_delay_ns,
        };
        for (&result.timing, self.timing, previous.timing) |*bucket, current, old| {
            bucket.* = current.sub(old);
        }
        return result;
    }
};

pub const ReadFailure = enum {
    transient,
    timeout,
    server_failure,
    throttle,
};

pub const AtomicReadStats = struct {
    const AtomicTimingBucket = struct {
        attempts: std.atomic.Value(u64) = .init(0),
        successes: std.atomic.Value(u64) = .init(0),
        successful_bytes: std.atomic.Value(u64) = .init(0),
        ttfb_ns: std.atomic.Value(u64) = .init(0),
        body_ns: std.atomic.Value(u64) = .init(0),
        transient_retries: std.atomic.Value(u64) = .init(0),
        timeouts: std.atomic.Value(u64) = .init(0),
        server_failures: std.atomic.Value(u64) = .init(0),
        throttles: std.atomic.Value(u64) = .init(0),
        retry_delay_ns: std.atomic.Value(u64) = .init(0),

        fn snapshot(self: *const AtomicTimingBucket) ReadTimingBucket {
            return .{
                .attempts = self.attempts.load(.acquire),
                .successes = self.successes.load(.acquire),
                .successful_bytes = self.successful_bytes.load(.acquire),
                .ttfb_ns = self.ttfb_ns.load(.acquire),
                .body_ns = self.body_ns.load(.acquire),
                .transient_retries = self.transient_retries.load(.acquire),
                .timeouts = self.timeouts.load(.acquire),
                .server_failures = self.server_failures.load(.acquire),
                .throttles = self.throttles.load(.acquire),
                .retry_delay_ns = self.retry_delay_ns.load(.acquire),
            };
        }
    };

    physical_requests: std.atomic.Value(u64) = .init(0),
    physical_bytes: std.atomic.Value(u64) = .init(0),
    retries: std.atomic.Value(u64) = .init(0),
    transient_retries: std.atomic.Value(u64) = .init(0),
    timeouts: std.atomic.Value(u64) = .init(0),
    server_failures: std.atomic.Value(u64) = .init(0),
    throttles: std.atomic.Value(u64) = .init(0),
    retry_delay_ns: std.atomic.Value(u64) = .init(0),
    timing: [read_timing_bucket_sizes.len]AtomicTimingBucket = @splat(.{}),

    pub fn recordAttempt(self: *AtomicReadStats, request_size: usize) void {
        _ = self.physical_requests.fetchAdd(1, .monotonic);
        if (timingBucketIndex(request_size)) |index| {
            _ = self.timing[index].attempts.fetchAdd(1, .monotonic);
        }
    }

    pub fn recordSuccess(
        self: *AtomicReadStats,
        request_size: usize,
        ttfb_ns: u64,
        body_ns: u64,
        include_timing_sample: bool,
    ) void {
        _ = self.physical_bytes.fetchAdd(@intCast(request_size), .monotonic);
        if (!include_timing_sample) return;
        if (timingBucketIndex(request_size)) |index| {
            const bucket = &self.timing[index];
            _ = bucket.successes.fetchAdd(1, .monotonic);
            _ = bucket.successful_bytes.fetchAdd(@intCast(request_size), .monotonic);
            _ = bucket.ttfb_ns.fetchAdd(ttfb_ns, .monotonic);
            _ = bucket.body_ns.fetchAdd(body_ns, .monotonic);
        }
    }

    pub fn recordFailure(self: *AtomicReadStats, request_size: usize, failure: ReadFailure) void {
        const bucket = if (timingBucketIndex(request_size)) |index| &self.timing[index] else null;
        switch (failure) {
            .transient => {
                _ = self.transient_retries.fetchAdd(1, .monotonic);
                if (bucket) |b| _ = b.transient_retries.fetchAdd(1, .monotonic);
            },
            .timeout => {
                _ = self.timeouts.fetchAdd(1, .monotonic);
                if (bucket) |b| _ = b.timeouts.fetchAdd(1, .monotonic);
            },
            .server_failure => {
                _ = self.server_failures.fetchAdd(1, .monotonic);
                if (bucket) |b| _ = b.server_failures.fetchAdd(1, .monotonic);
            },
            .throttle => {
                _ = self.throttles.fetchAdd(1, .monotonic);
                if (bucket) |b| _ = b.throttles.fetchAdd(1, .monotonic);
            },
        }
    }

    pub fn recordRetry(self: *AtomicReadStats) void {
        _ = self.retries.fetchAdd(1, .monotonic);
    }

    pub fn recordRetryDelay(self: *AtomicReadStats, request_size: usize, delay: std.Io.Duration) void {
        const delay_ns: u64 = @intCast(@max(delay.nanoseconds, 0));
        _ = self.retry_delay_ns.fetchAdd(delay_ns, .monotonic);
        if (timingBucketIndex(request_size)) |index| {
            _ = self.timing[index].retry_delay_ns.fetchAdd(delay_ns, .monotonic);
        }
    }

    pub fn snapshot(self: *const AtomicReadStats) ReadStats {
        var result: ReadStats = .{
            .physical_requests = self.physical_requests.load(.acquire),
            .physical_bytes = self.physical_bytes.load(.acquire),
            .retries = self.retries.load(.acquire),
            .transient_retries = self.transient_retries.load(.acquire),
            .timeouts = self.timeouts.load(.acquire),
            .server_failures = self.server_failures.load(.acquire),
            .throttles = self.throttles.load(.acquire),
            .retry_delay_ns = self.retry_delay_ns.load(.acquire),
        };
        for (&result.timing, &self.timing) |*bucket, *atomic_bucket| {
            bucket.* = atomic_bucket.snapshot();
        }
        return result;
    }
};

pub fn timingBucketIndex(request_size: usize) ?usize {
    for (read_timing_bucket_sizes, 0..) |size, index| {
        if (request_size == size) return index;
    }
    return null;
}

test "atomic read stats retain typed exact-size timing buckets" {
    var stats: AtomicReadStats = .{};
    const request_size = 16 * 1024 * 1024;
    stats.recordAttempt(request_size);
    stats.recordFailure(request_size, .timeout);
    stats.recordRetry();
    stats.recordRetryDelay(request_size, .fromMilliseconds(25));
    stats.recordAttempt(request_size);
    stats.recordSuccess(request_size, 7, 11, true);

    const snapshot = stats.snapshot();
    const bucket = snapshot.timing[timingBucketIndex(request_size).?];
    try std.testing.expectEqual(@as(u64, 2), snapshot.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), snapshot.physical_bytes);
    try std.testing.expectEqual(@as(u64, 1), snapshot.retries);
    try std.testing.expectEqual(@as(u64, 1), snapshot.timeouts);
    try std.testing.expectEqual(@as(u64, 2), bucket.attempts);
    try std.testing.expectEqual(@as(u64, 1), bucket.successes);
    try std.testing.expectEqual(@as(u64, request_size), bucket.successful_bytes);
    try std.testing.expectEqual(@as(u64, 7), bucket.ttfb_ns);
    try std.testing.expectEqual(@as(u64, 11), bucket.body_ns);
    try std.testing.expectEqual(@as(u64, 1), bucket.timeouts);
    try std.testing.expectEqual(@as(u64, 25 * std.time.ns_per_ms), bucket.retry_delay_ns);
}

test "retried successful reads do not enter timing samples" {
    var stats: AtomicReadStats = .{};
    const request_size = 32 * 1024 * 1024;
    stats.recordAttempt(request_size);
    stats.recordFailure(request_size, .transient);
    stats.recordRetry();
    stats.recordAttempt(request_size);
    stats.recordSuccess(request_size, 13, 17, false);

    const snapshot = stats.snapshot();
    const bucket = snapshot.timing[timingBucketIndex(request_size).?];
    try std.testing.expectEqual(@as(u64, 2), snapshot.physical_requests);
    try std.testing.expectEqual(@as(u64, request_size), snapshot.physical_bytes);
    try std.testing.expectEqual(@as(u64, 2), bucket.attempts);
    try std.testing.expectEqual(@as(u64, 0), bucket.successes);
    try std.testing.expectEqual(@as(u64, 0), bucket.successful_bytes);
    try std.testing.expectEqual(@as(u64, 0), bucket.ttfb_ns);
    try std.testing.expectEqual(@as(u64, 0), bucket.body_ns);
}

pub const ReadStatsProvider = struct {
    userdata: *anyopaque,
    snapshotFn: *const fn (userdata: *anyopaque) ReadStats,

    pub fn snapshot(self: ReadStatsProvider) ReadStats {
        return self.snapshotFn(self.userdata);
    }
};

pub const Backend = struct {
    io: std.Io,
    read_hints: ReadHints = .{},
    read_stats: ?ReadStatsProvider = null,
};

pub const VFSBase = struct {
    inner: std.Io,

    pub fn init(io: std.Io) VFSBase {
        return .{ .inner = io };
    }

    pub fn vtable(overrides: anytype) std.Io.VTable {
        var new_vtable: std.Io.VTable = .{
            .crashHandler = crashHandler,
            .async = async,
            .concurrent = concurrent,
            .await = await,
            .cancel = cancel,
            .groupAsync = groupAsync,
            .groupConcurrent = groupConcurrent,
            .groupAwait = groupAwait,
            .groupCancel = groupCancel,
            .recancel = recancel,
            .swapCancelProtection = swapCancelProtection,
            .checkCancel = checkCancel,
            .futexWait = futexWait,
            .futexWaitUncancelable = futexWaitUncancelable,
            .futexWake = futexWake,
            .operate = operate,
            .batchAwaitAsync = batchAwaitAsync,
            .batchAwaitConcurrent = batchAwaitConcurrent,
            .batchCancel = batchCancel,
            .dirCreateDir = dirCreateDir,
            .dirCreateDirPath = dirCreateDirPath,
            .dirCreateDirPathOpen = dirCreateDirPathOpen,
            .dirOpenDir = dirOpenDir,
            .dirStat = dirStat,
            .dirStatFile = dirStatFile,
            .dirAccess = dirAccess,
            .dirCreateFile = dirCreateFile,
            .dirCreateFileAtomic = dirCreateFileAtomic,
            .dirOpenFile = dirOpenFile,
            .dirClose = dirClose,
            .dirRead = dirRead,
            .dirRealPath = dirRealPath,
            .dirRealPathFile = dirRealPathFile,
            .dirDeleteFile = dirDeleteFile,
            .dirDeleteDir = dirDeleteDir,
            .dirRename = dirRename,
            .dirRenamePreserve = dirRenamePreserve,
            .dirSymLink = dirSymLink,
            .dirReadLink = dirReadLink,
            .dirSetOwner = dirSetOwner,
            .dirSetFileOwner = dirSetFileOwner,
            .dirSetPermissions = dirSetPermissions,
            .dirSetFilePermissions = dirSetFilePermissions,
            .dirSetTimestamps = dirSetTimestamps,
            .dirHardLink = dirHardLink,
            .fileStat = fileStat,
            .fileLength = fileLength,
            .fileClose = fileClose,
            .fileWritePositional = fileWritePositional,
            .fileWriteFileStreaming = fileWriteFileStreaming,
            .fileWriteFilePositional = fileWriteFilePositional,
            .fileReadPositional = fileReadPositional,
            .fileSeekBy = fileSeekBy,
            .fileSeekTo = fileSeekTo,
            .fileSync = fileSync,
            .fileIsTty = fileIsTty,
            .fileEnableAnsiEscapeCodes = fileEnableAnsiEscapeCodes,
            .fileSupportsAnsiEscapeCodes = fileSupportsAnsiEscapeCodes,
            .fileSetLength = fileSetLength,
            .fileSetOwner = fileSetOwner,
            .fileSetPermissions = fileSetPermissions,
            .fileSetTimestamps = fileSetTimestamps,
            .fileLock = fileLock,
            .fileTryLock = fileTryLock,
            .fileUnlock = fileUnlock,
            .fileDowngradeLock = fileDowngradeLock,
            .fileRealPath = fileRealPath,
            .fileHardLink = fileHardLink,
            .fileMemoryMapCreate = fileMemoryMapCreate,
            .fileMemoryMapDestroy = fileMemoryMapDestroy,
            .fileMemoryMapSetLength = fileMemoryMapSetLength,
            .fileMemoryMapRead = fileMemoryMapRead,
            .fileMemoryMapWrite = fileMemoryMapWrite,
            .processExecutableOpen = processExecutableOpen,
            .processExecutablePath = processExecutablePath,
            .lockStderr = lockStderr,
            .tryLockStderr = tryLockStderr,
            .unlockStderr = unlockStderr,
            .processCurrentPath = processCurrentPath,
            .processSetCurrentDir = processSetCurrentDir,
            .processSetCurrentPath = processSetCurrentPath,
            .processReplace = processReplace,
            .processReplacePath = processReplacePath,
            .processSpawn = processSpawn,
            .processSpawnPath = processSpawnPath,
            .childWait = childWait,
            .childKill = childKill,
            .progressParentFile = progressParentFile,
            .now = now,
            .clockResolution = clockResolution,
            .sleep = sleep,
            .random = random,
            .randomSecure = randomSecure,
            .netListenIp = netListenIp,
            .netAccept = netAccept,
            .netBindIp = netBindIp,
            .netConnectIp = netConnectIp,
            .netListenUnix = netListenUnix,
            .netConnectUnix = netConnectUnix,
            .netSocketCreatePair = netSocketCreatePair,
            .netSend = netSend,
            .netRead = netRead,
            .netWrite = netWrite,
            .netWriteFile = netWriteFile,
            .netClose = netClose,
            .netShutdown = netShutdown,
            .netInterfaceNameResolve = netInterfaceNameResolve,
            .netInterfaceName = netInterfaceName,
            .netLookup = netLookup,
        };
        for (std.meta.fieldNames(@TypeOf(overrides))) |field_name| {
            @field(new_vtable, field_name) = @field(overrides, field_name);
        }
        return new_vtable;
    }

    pub fn as(userdata: ?*anyopaque) *VFSBase {
        return @ptrCast(@alignCast(userdata.?));
    }

    pub fn crashHandler(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.crashHandler(self.inner.userdata);
    }

    pub fn async(userdata: ?*anyopaque, result: []u8, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) ?*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.async(self.inner.userdata, result, result_alignment, context, context_alignment, start);
    }

    pub fn concurrent(userdata: ?*anyopaque, result_len: usize, result_alignment: std.mem.Alignment, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque, result: *anyopaque) void) std.Io.ConcurrentError!*std.Io.AnyFuture {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.concurrent(self.inner.userdata, result_len, result_alignment, context, context_alignment, start);
    }

    pub fn await(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.await(self.inner.userdata, any_future, result, result_alignment);
    }

    pub fn cancel(userdata: ?*anyopaque, any_future: *std.Io.AnyFuture, result: []u8, result_alignment: std.mem.Alignment) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.cancel(self.inner.userdata, any_future, result, result_alignment);
    }

    pub fn groupAsync(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque) void) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupAsync(self.inner.userdata, group, context, context_alignment, start);
    }

    pub fn groupConcurrent(userdata: ?*anyopaque, group: *std.Io.Group, context: []const u8, context_alignment: std.mem.Alignment, start: *const fn (context: *const anyopaque) void) std.Io.ConcurrentError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupConcurrent(self.inner.userdata, group, context, context_alignment, start);
    }

    pub fn groupAwait(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupAwait(self.inner.userdata, group, token);
    }

    pub fn groupCancel(userdata: ?*anyopaque, group: *std.Io.Group, token: *anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.groupCancel(self.inner.userdata, group, token);
    }

    pub fn recancel(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.recancel(self.inner.userdata);
    }

    pub fn swapCancelProtection(userdata: ?*anyopaque, new: std.Io.CancelProtection) std.Io.CancelProtection {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.swapCancelProtection(self.inner.userdata, new);
    }

    pub fn checkCancel(userdata: ?*anyopaque) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.checkCancel(self.inner.userdata);
    }

    pub fn futexWait(userdata: ?*anyopaque, ptr: *const u32, expected: u32, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWait(self.inner.userdata, ptr, expected, timeout);
    }

    pub fn futexWaitUncancelable(userdata: ?*anyopaque, ptr: *const u32, expected: u32) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWaitUncancelable(self.inner.userdata, ptr, expected);
    }

    pub fn futexWake(userdata: ?*anyopaque, ptr: *const u32, max_waiters: u32) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.futexWake(self.inner.userdata, ptr, max_waiters);
    }

    pub fn operate(userdata: ?*anyopaque, operation: std.Io.Operation) std.Io.Cancelable!std.Io.Operation.Result {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.operate(self.inner.userdata, operation);
    }

    pub fn batchAwaitAsync(userdata: ?*anyopaque, batch: *std.Io.Batch) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.batchAwaitAsync(self.inner.userdata, batch);
    }

    pub fn batchAwaitConcurrent(userdata: ?*anyopaque, batch: *std.Io.Batch, timeout: std.Io.Timeout) std.Io.Batch.AwaitConcurrentError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.batchAwaitConcurrent(self.inner.userdata, batch, timeout);
    }

    pub fn batchCancel(userdata: ?*anyopaque, batch: *std.Io.Batch) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.batchCancel(self.inner.userdata, batch);
    }

    pub fn dirCreateDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDir(self.inner.userdata, dir, sub_path, permissions);
    }

    pub fn dirCreateDirPath(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions) std.Io.Dir.CreateDirPathError!std.Io.Dir.CreatePathStatus {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDirPath(self.inner.userdata, dir, sub_path, permissions);
    }

    pub fn dirCreateDirPathOpen(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.Dir.Permissions, options: std.Io.Dir.OpenOptions) std.Io.Dir.CreateDirPathOpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateDirPathOpen(self.inner.userdata, dir, sub_path, permissions, options);
    }

    pub fn dirOpenDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.OpenOptions) std.Io.Dir.OpenError!std.Io.Dir {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirOpenDir(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirStat(userdata: ?*anyopaque, dir: std.Io.Dir) std.Io.Dir.StatError!std.Io.Dir.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirStat(self.inner.userdata, dir);
    }

    pub fn dirStatFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.StatFileOptions) std.Io.Dir.StatFileError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirStatFile(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirAccess(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.AccessOptions) std.Io.Dir.AccessError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirAccess(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirCreateFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.CreateFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateFile(self.inner.userdata, dir, sub_path, flags);
    }

    pub fn dirCreateFileAtomic(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.CreateFileAtomicOptions) std.Io.Dir.CreateFileAtomicError!std.Io.File.Atomic {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirCreateFileAtomic(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirOpenFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, flags: std.Io.File.OpenFlags) std.Io.File.OpenError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirOpenFile(self.inner.userdata, dir, sub_path, flags);
    }

    pub fn dirClose(userdata: ?*anyopaque, dirs: []const std.Io.Dir) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.dirClose(self.inner.userdata, dirs);
    }

    pub fn dirRead(userdata: ?*anyopaque, reader: *std.Io.Dir.Reader, entries: []std.Io.Dir.Entry) std.Io.Dir.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRead(self.inner.userdata, reader, entries);
    }

    pub fn dirRealPath(userdata: ?*anyopaque, dir: std.Io.Dir, out_buffer: []u8) std.Io.Dir.RealPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRealPath(self.inner.userdata, dir, out_buffer);
    }

    pub fn dirRealPathFile(userdata: ?*anyopaque, dir: std.Io.Dir, path_name: []const u8, out_buffer: []u8) std.Io.Dir.RealPathFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRealPathFile(self.inner.userdata, dir, path_name, out_buffer);
    }

    pub fn dirDeleteFile(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteFileError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirDeleteFile(self.inner.userdata, dir, sub_path);
    }

    pub fn dirDeleteDir(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8) std.Io.Dir.DeleteDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirDeleteDir(self.inner.userdata, dir, sub_path);
    }

    pub fn dirRename(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8) std.Io.Dir.RenameError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRename(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path);
    }

    pub fn dirRenamePreserve(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8) std.Io.Dir.RenamePreserveError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirRenamePreserve(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path);
    }

    pub fn dirSymLink(userdata: ?*anyopaque, dir: std.Io.Dir, target_path: []const u8, sym_link_path: []const u8, flags: std.Io.Dir.SymLinkFlags) std.Io.Dir.SymLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSymLink(self.inner.userdata, dir, target_path, sym_link_path, flags);
    }

    pub fn dirReadLink(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, buffer: []u8) std.Io.Dir.ReadLinkError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirReadLink(self.inner.userdata, dir, sub_path, buffer);
    }

    pub fn dirSetOwner(userdata: ?*anyopaque, dir: std.Io.Dir, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid) std.Io.Dir.SetOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetOwner(self.inner.userdata, dir, uid, gid);
    }

    pub fn dirSetFileOwner(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid, options: std.Io.Dir.SetFileOwnerOptions) std.Io.Dir.SetFileOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetFileOwner(self.inner.userdata, dir, sub_path, uid, gid, options);
    }

    pub fn dirSetPermissions(userdata: ?*anyopaque, dir: std.Io.Dir, permissions: std.Io.Dir.Permissions) std.Io.Dir.SetPermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetPermissions(self.inner.userdata, dir, permissions);
    }

    pub fn dirSetFilePermissions(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, permissions: std.Io.File.Permissions, options: std.Io.Dir.SetFilePermissionsOptions) std.Io.Dir.SetFilePermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetFilePermissions(self.inner.userdata, dir, sub_path, permissions, options);
    }

    pub fn dirSetTimestamps(userdata: ?*anyopaque, dir: std.Io.Dir, sub_path: []const u8, options: std.Io.Dir.SetTimestampsOptions) std.Io.Dir.SetTimestampsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirSetTimestamps(self.inner.userdata, dir, sub_path, options);
    }

    pub fn dirHardLink(userdata: ?*anyopaque, old_dir: std.Io.Dir, old_sub_path: []const u8, new_dir: std.Io.Dir, new_sub_path: []const u8, options: std.Io.Dir.HardLinkOptions) std.Io.Dir.HardLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.dirHardLink(self.inner.userdata, old_dir, old_sub_path, new_dir, new_sub_path, options);
    }

    pub fn fileStat(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.StatError!std.Io.File.Stat {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileStat(self.inner.userdata, file);
    }

    pub fn fileLength(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.LengthError!u64 {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileLength(self.inner.userdata, file);
    }

    pub fn fileClose(userdata: ?*anyopaque, files: []const std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.fileClose(self.inner.userdata, files);
    }

    pub fn fileWritePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, data: []const []const u8, splat: usize, offset: u64) std.Io.File.WritePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWritePositional(self.inner.userdata, file, header, data, splat, offset);
    }

    pub fn fileWriteFileStreaming(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.File.Writer.WriteFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteFileStreaming(self.inner.userdata, file, header, reader, limit);
    }

    pub fn fileWriteFilePositional(userdata: ?*anyopaque, file: std.Io.File, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit, offset: u64) std.Io.File.WriteFilePositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileWriteFilePositional(self.inner.userdata, file, header, reader, limit, offset);
    }

    pub fn fileReadPositional(userdata: ?*anyopaque, file: std.Io.File, data: []const []u8, offset: u64) std.Io.File.ReadPositionalError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileReadPositional(self.inner.userdata, file, data, offset);
    }

    pub fn fileSeekBy(userdata: ?*anyopaque, file: std.Io.File, relative_offset: i64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSeekBy(self.inner.userdata, file, relative_offset);
    }

    pub fn fileSeekTo(userdata: ?*anyopaque, file: std.Io.File, absolute_offset: u64) std.Io.File.SeekError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSeekTo(self.inner.userdata, file, absolute_offset);
    }

    pub fn fileSync(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.SyncError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSync(self.inner.userdata, file);
    }

    pub fn fileIsTty(userdata: ?*anyopaque, file: std.Io.File) std.Io.Cancelable!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileIsTty(self.inner.userdata, file);
    }

    pub fn fileEnableAnsiEscapeCodes(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.EnableAnsiEscapeCodesError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileEnableAnsiEscapeCodes(self.inner.userdata, file);
    }

    pub fn fileSupportsAnsiEscapeCodes(userdata: ?*anyopaque, file: std.Io.File) std.Io.Cancelable!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSupportsAnsiEscapeCodes(self.inner.userdata, file);
    }

    pub fn fileSetLength(userdata: ?*anyopaque, file: std.Io.File, length: u64) std.Io.File.SetLengthError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetLength(self.inner.userdata, file, length);
    }

    pub fn fileSetOwner(userdata: ?*anyopaque, file: std.Io.File, uid: ?std.Io.File.Uid, gid: ?std.Io.File.Gid) std.Io.File.SetOwnerError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetOwner(self.inner.userdata, file, uid, gid);
    }

    pub fn fileSetPermissions(userdata: ?*anyopaque, file: std.Io.File, permissions: std.Io.File.Permissions) std.Io.File.SetPermissionsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetPermissions(self.inner.userdata, file, permissions);
    }

    pub fn fileSetTimestamps(userdata: ?*anyopaque, file: std.Io.File, options: std.Io.File.SetTimestampsOptions) std.Io.File.SetTimestampsError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileSetTimestamps(self.inner.userdata, file, options);
    }

    pub fn fileLock(userdata: ?*anyopaque, file: std.Io.File, lock: std.Io.File.Lock) std.Io.File.LockError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileLock(self.inner.userdata, file, lock);
    }

    pub fn fileTryLock(userdata: ?*anyopaque, file: std.Io.File, lock: std.Io.File.Lock) std.Io.File.LockError!bool {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileTryLock(self.inner.userdata, file, lock);
    }

    pub fn fileUnlock(userdata: ?*anyopaque, file: std.Io.File) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileUnlock(self.inner.userdata, file);
    }

    pub fn fileDowngradeLock(userdata: ?*anyopaque, file: std.Io.File) std.Io.File.DowngradeLockError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileDowngradeLock(self.inner.userdata, file);
    }

    pub fn fileRealPath(userdata: ?*anyopaque, file: std.Io.File, out_buffer: []u8) std.Io.File.RealPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileRealPath(self.inner.userdata, file, out_buffer);
    }

    pub fn fileHardLink(userdata: ?*anyopaque, file: std.Io.File, dir: std.Io.Dir, path: []const u8, options: std.Io.File.HardLinkOptions) std.Io.File.HardLinkError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileHardLink(self.inner.userdata, file, dir, path, options);
    }

    pub fn fileMemoryMapCreate(userdata: ?*anyopaque, file: std.Io.File, options: std.Io.File.MemoryMap.CreateOptions) std.Io.File.MemoryMap.CreateError!std.Io.File.MemoryMap {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapCreate(self.inner.userdata, file, options);
    }

    pub fn fileMemoryMapDestroy(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.fileMemoryMapDestroy(self.inner.userdata, memory_map);
    }

    pub fn fileMemoryMapSetLength(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap, length: usize) std.Io.File.MemoryMap.SetLengthError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapSetLength(self.inner.userdata, memory_map, length);
    }

    pub fn fileMemoryMapRead(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap) std.Io.File.ReadPositionalError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapRead(self.inner.userdata, memory_map);
    }

    pub fn fileMemoryMapWrite(userdata: ?*anyopaque, memory_map: *std.Io.File.MemoryMap) std.Io.File.WritePositionalError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.fileMemoryMapWrite(self.inner.userdata, memory_map);
    }

    pub fn processExecutableOpen(userdata: ?*anyopaque, flags: std.Io.File.OpenFlags) std.process.OpenExecutableError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processExecutableOpen(self.inner.userdata, flags);
    }

    pub fn processExecutablePath(userdata: ?*anyopaque, buffer: []u8) std.process.ExecutablePathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processExecutablePath(self.inner.userdata, buffer);
    }

    pub fn lockStderr(userdata: ?*anyopaque, mode: ?std.Io.Terminal.Mode) std.Io.Cancelable!std.Io.LockedStderr {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.lockStderr(self.inner.userdata, mode);
    }

    pub fn tryLockStderr(userdata: ?*anyopaque, mode: ?std.Io.Terminal.Mode) std.Io.Cancelable!?std.Io.LockedStderr {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.tryLockStderr(self.inner.userdata, mode);
    }

    pub fn unlockStderr(userdata: ?*anyopaque) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.unlockStderr(self.inner.userdata);
    }

    pub fn processCurrentPath(userdata: ?*anyopaque, buffer: []u8) std.process.CurrentPathError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processCurrentPath(self.inner.userdata, buffer);
    }

    pub fn processSetCurrentDir(userdata: ?*anyopaque, dir: std.Io.Dir) std.process.SetCurrentDirError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSetCurrentDir(self.inner.userdata, dir);
    }

    pub fn processSetCurrentPath(userdata: ?*anyopaque, path: []const u8) std.process.SetCurrentPathError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSetCurrentPath(self.inner.userdata, path);
    }

    pub fn processReplace(userdata: ?*anyopaque, options: std.process.ReplaceOptions) std.process.ReplaceError {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processReplace(self.inner.userdata, options);
    }

    pub fn processReplacePath(userdata: ?*anyopaque, dir: std.Io.Dir, options: std.process.ReplaceOptions) std.process.ReplaceError {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processReplacePath(self.inner.userdata, dir, options);
    }

    pub fn processSpawn(userdata: ?*anyopaque, options: std.process.SpawnOptions) std.process.SpawnError!std.process.Child {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSpawn(self.inner.userdata, options);
    }

    pub fn processSpawnPath(userdata: ?*anyopaque, dir: std.Io.Dir, options: std.process.SpawnOptions) std.process.SpawnError!std.process.Child {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.processSpawnPath(self.inner.userdata, dir, options);
    }

    pub fn childWait(userdata: ?*anyopaque, child: *std.process.Child) std.process.Child.WaitError!std.process.Child.Term {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.childWait(self.inner.userdata, child);
    }

    pub fn childKill(userdata: ?*anyopaque, child: *std.process.Child) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.childKill(self.inner.userdata, child);
    }

    pub fn progressParentFile(userdata: ?*anyopaque) std.Progress.ParentFileError!std.Io.File {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.progressParentFile(self.inner.userdata);
    }

    pub fn now(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Timestamp {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.now(self.inner.userdata, clock);
    }

    pub fn clockResolution(userdata: ?*anyopaque, clock: std.Io.Clock) std.Io.Clock.ResolutionError!std.Io.Duration {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.clockResolution(self.inner.userdata, clock);
    }

    pub fn sleep(userdata: ?*anyopaque, timeout: std.Io.Timeout) std.Io.Cancelable!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.sleep(self.inner.userdata, timeout);
    }

    pub fn random(userdata: ?*anyopaque, buffer: []u8) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.random(self.inner.userdata, buffer);
    }

    pub fn randomSecure(userdata: ?*anyopaque, buffer: []u8) std.Io.RandomSecureError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.randomSecure(self.inner.userdata, buffer);
    }

    pub fn netListenIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.ListenOptions) std.Io.net.IpAddress.ListenError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netListenIp(self.inner.userdata, address, options);
    }

    pub fn netAccept(userdata: ?*anyopaque, server: std.Io.net.Socket.Handle, options: std.Io.net.Server.AcceptOptions) std.Io.net.Server.AcceptError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netAccept(self.inner.userdata, server, options);
    }

    pub fn netBindIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.BindOptions) std.Io.net.IpAddress.BindError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netBindIp(self.inner.userdata, address, options);
    }

    pub fn netConnectIp(userdata: ?*anyopaque, address: *const std.Io.net.IpAddress, options: std.Io.net.IpAddress.ConnectOptions) std.Io.net.IpAddress.ConnectError!std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netConnectIp(self.inner.userdata, address, options);
    }

    pub fn netListenUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress, options: std.Io.net.UnixAddress.ListenOptions) std.Io.net.UnixAddress.ListenError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netListenUnix(self.inner.userdata, address, options);
    }

    pub fn netConnectUnix(userdata: ?*anyopaque, address: *const std.Io.net.UnixAddress) std.Io.net.UnixAddress.ConnectError!std.Io.net.Socket.Handle {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netConnectUnix(self.inner.userdata, address);
    }

    pub fn netSocketCreatePair(userdata: ?*anyopaque, options: std.Io.net.Socket.CreatePairOptions) std.Io.net.Socket.CreatePairError![2]std.Io.net.Socket {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netSocketCreatePair(self.inner.userdata, options);
    }

    pub fn netSend(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, msgs: []std.Io.net.OutgoingMessage, flags: std.Io.net.SendFlags) struct { ?std.Io.net.Socket.SendError, usize } {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netSend(self.inner.userdata, handle, msgs, flags);
    }

    pub fn netRead(userdata: ?*anyopaque, src: std.Io.net.Socket.Handle, data: [][]u8) std.Io.net.Stream.Reader.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netRead(self.inner.userdata, src, data);
    }

    pub fn netWrite(userdata: ?*anyopaque, dest: std.Io.net.Socket.Handle, header: []const u8, data: []const []const u8, splat: usize) std.Io.net.Stream.Writer.Error!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netWrite(self.inner.userdata, dest, header, data, splat);
    }

    pub fn netWriteFile(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, header: []const u8, reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.net.Stream.Writer.WriteFileError!usize {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netWriteFile(self.inner.userdata, handle, header, reader, limit);
    }

    pub fn netClose(userdata: ?*anyopaque, handles: []const std.Io.net.Socket.Handle) void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        self.inner.vtable.netClose(self.inner.userdata, handles);
    }

    pub fn netShutdown(userdata: ?*anyopaque, handle: std.Io.net.Socket.Handle, how: std.Io.net.ShutdownHow) std.Io.net.ShutdownError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netShutdown(self.inner.userdata, handle, how);
    }

    pub fn netInterfaceNameResolve(userdata: ?*anyopaque, name: *const std.Io.net.Interface.Name) std.Io.net.Interface.Name.ResolveError!std.Io.net.Interface {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netInterfaceNameResolve(self.inner.userdata, name);
    }

    pub fn netInterfaceName(userdata: ?*anyopaque, iface: std.Io.net.Interface) std.Io.net.Interface.NameError!std.Io.net.Interface.Name {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netInterfaceName(self.inner.userdata, iface);
    }

    pub fn netLookup(userdata: ?*anyopaque, host: std.Io.net.HostName, q: *std.Io.Queue(std.Io.net.HostName.LookupResult), options: std.Io.net.HostName.LookupOptions) std.Io.net.HostName.LookupError!void {
        const self: *VFSBase = @ptrCast(@alignCast(userdata.?));
        return self.inner.vtable.netLookup(self.inner.userdata, host, q, options);
    }
};
