//! Whole-tensor staging for platforms that use Buffer.from.
//! The shared front end owns source lookup, executable bindings and admission.
const std = @import("std");
const log = @import("log.zig").load;
const stdx = @import("stdx");
const pjrt = @import("pjrt");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const backend = @import("backend.zig");
const LoadSpec = backend.LoadSpec;

/// Concurrent whole tensors staged on the host: the byte budget is this
/// many times the largest tensor submitted. Kept at the former start width
/// rather than the read width, which on a high-latency profile would
/// triple the host staging for nothing this path can use (a tensor already
/// splits its reads across the width).
const staging_tensors: usize = 12;

pub const Loader = struct {
    pub const InitError = std.mem.Allocator.Error;
    pub const AwaitError = std.mem.Allocator.Error || pjrt.ApiError || std.Io.File.OpenError || std.Io.File.Reader.SeekError || safetensors.TensorReader.ReadPositionalError || error{SourceSizeMismatch};
    /// A submission refuses on the sticky error, so it fails as an await does.
    pub const SubmitError = AwaitError;

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    group: stdx.Io.LimitedGroup,
    /// The load profile's minimum read size: one tensor becomes this many
    /// bytes per positional read.
    read_chunk_size: usize,
    /// The widest fan-out one tensor may take. One on a source whose reads
    /// are bandwidth-bound rather than round-trip bound: splitting there only
    /// spends tasks, and measurably so -- a replicated local load, whose long
    /// transfers keep permits free, lost time to helpers it had no use for.
    tensor_workers: usize,
    permits: ReadPermits,
    /// Concurrent tensors; sizes the staging budget.
    staging_slots: usize,
    admission: StagingAdmission = .{},
    staging: StagingPool = .{},
    first_error: std.atomic.Value(u16) = .init(0),

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        read_parallelism: usize,
        profile: VFS.LoadProfile,
    ) InitError!*Loader {
        const self = try allocator.create(Loader);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            // Tasks are capped by the read budget; the staging budget is
            // what actually decides how many run.
            .group = .init(read_parallelism),
            .staging_slots = @min(read_parallelism, staging_tensors),
            .read_chunk_size = profile.read_chunk_size,
            // A high-latency source may use spare read permits to split a
            // tensor already staging: more concurrency without more host memory.
            .tensor_workers = if (profile.high_latency) read_parallelism else 1,
            .permits = .init(read_parallelism),
        };
        return self;
    }

    /// Spawns one bounded read task per spec. Nothing runs when this fails.
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: ?*std.Progress.Node) SubmitError!*Batch {
        try self.checkOpen();
        var largest: usize = 0;
        for (specs) |spec| largest = @max(largest, spec.shape.byteSize());
        self.admission.widen(self.io, largest, self.staging_slots);
        const batch = try self.allocator.create(Batch);
        batch.* = .{ .pending = .init(1 + specs.len) };
        for (specs) |spec| {
            self.submitOne(batch, spec.source, spec.shape, spec.sharding, spec.output, progress);
        }
        // Every task is spawned: drop the sentinel.
        batch.finish(self.io);
        return batch;
    }

    /// Waits for the batch's tasks, frees it and returns the sticky error.
    pub fn awaitBatch(self: *Loader, batch: *Batch) AwaitError!void {
        batch.done.waitUncancelable(self.io);
        self.allocator.destroy(batch);
        try self.checkOpen();
    }

    /// Every batch was awaited, so the group is idle.
    pub fn destroy(self: *Loader) void {
        self.group.await(self.io) catch {};
        self.staging.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn recordError(self: *Loader, err: AwaitError, comptime fmt: []const u8, args: anytype) void {
        if (self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic) == null) {
            log.failure(err, fmt, args);
        }
    }

    fn checkOpen(self: *Loader) AwaitError!void {
        const code = self.first_error.load(.acquire);
        if (code != 0) return @errorCast(@errorFromInt(code));
    }

    fn loadOne(
        self: *Loader,
        source: *safetensors.Tensor,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
    ) AwaitError!void {
        if (self.first_error.load(.acquire) != 0) return;
        var stage: []const u8 = "validate";
        errdefer |err| self.recordError(err, "{s} tensor {s}: file={s}, offset={d}, source_bytes={d}, destination_bytes={d}", .{ stage, source.name, source.file_uri, source.offset, source.byteSize(), shape.byteSize() });
        const tensor_bytes = shape.byteSize();
        // The whole source becomes this tensor's bytes, so the two sizes are
        // the same thing. Staging is reused, so a source that does not fill
        // it would publish another tensor's bytes rather than the zeroes a
        // fresh allocation used to give.
        if (source.byteSize() != tensor_bytes) return error.SourceSizeMismatch;
        self.admission.reserve(self.io, tensor_bytes);
        defer self.admission.release(self.io, tensor_bytes);
        stage = "open source for";
        var reader = try source.reader(self.io, &.{}, .{});
        defer reader.deinit();
        stage = "allocate staging for";
        const staging = try self.staging.acquire(self.allocator, self.io, tensor_bytes);
        defer self.staging.release(self.allocator, self.io, staging);
        stage = "read";
        try self.readInto(&reader, staging[0..tensor_bytes]);
        stage = "transfer";
        output.* = try Buffer.from(
            self.io,
            self.platform,
            shape,
            sharding,
            staging[0..tensor_bytes],
            .{ .wait = true },
        );
    }

    /// Fills `destination` from `reader`, splitting it across helper tasks
    /// when the source permits are free. The permits are released before the
    /// caller's transfer, so a tensor staging to the device lends its read
    /// budget to one still reading.
    fn readInto(
        self: *Loader,
        reader: *const safetensors.TensorReader,
        destination: []u8,
    ) safetensors.TensorReader.ReadPositionalError!void {
        if (destination.len == 0) return;
        var work: ChunkedRead = .{
            .reader = reader,
            .destination = destination,
            .chunk_size = self.read_chunk_size,
        };
        // Splitting below the profile's chunk buys nothing, so that is the
        // most workers this tensor can use.
        const useful_workers = @min(
            self.tensor_workers,
            destination.len / self.read_chunk_size +
                @intFromBool(destination.len % self.read_chunk_size != 0),
        );

        self.permits.acquire(self.io);
        var held: usize = 1;
        var group: std.Io.Group = .init;
        while (true) {
            // Helpers only ever take permits that are already free, so a busy
            // source sees exactly the one read per tensor it saw before. The
            // check repeats between claims because the tensors that finish
            // first hand their budget to the one still reading -- which is
            // the long tensor that needed the help.
            // Nothing unclaimed means a helper would only start and stop.
            if (held < useful_workers and work.next.load(.monotonic) < destination.len) {
                const extra = self.permits.tryAcquire(self.io, useful_workers - held);
                if (extra != 0) {
                    held += extra;
                    work.workers.store(held, .monotonic);
                    for (0..extra) |_| group.async(self.io, ChunkedRead.run, .{&work});
                }
            }
            if (!work.readOne()) break;
        }
        group.await(self.io) catch {};
        self.permits.release(self.io, held);

        const code = work.failure.load(.acquire);
        if (code != 0) return @errorCast(@errorFromInt(code));
    }

    fn submitOne(
        self: *Loader,
        batch: *Batch,
        source: *safetensors.Tensor,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
        progress: ?*std.Progress.Node,
    ) void {
        self.group.async(self.io, struct {
            fn run(
                loader: *Loader,
                batch_: *Batch,
                source_: *safetensors.Tensor,
                shape_: Shape,
                sharding_: Sharding,
                output_: *Buffer,
                progress_: ?*std.Progress.Node,
            ) void {
                defer batch_.finish(loader.io);
                var node = if (progress_) |parent| parent.start(source_.name, 1) else null;
                defer if (node) |*n| n.end();
                loader.loadOne(source_, shape_, sharding_, output_) catch {};
            }
        }.run, .{ self, batch, source, shape, sharding, output, progress });
    }
};

/// One buffered submission: `pending` counts a publish sentinel plus one
/// unit per tensor task; the last one sets `done`.
pub const Batch = struct {
    pending: std.atomic.Value(usize),
    done: std.Io.Event = .unset,

    fn finish(self: *Batch, io: std.Io) void {
        if (self.pending.fetchSub(1, .acq_rel) == 1) self.done.set(io);
    }
};

/// Admits tensors by host bytes, up to the initial read width times the
/// largest tensor submitted. Small tensors can fill the read budget while
/// large tensors stay within the same staging budget.
const StagingAdmission = struct {
    mutex: std.Io.Mutex = .init,
    room: std.Io.Condition = .init,
    in_flight: usize = 0,
    /// Zero until the first submission reports its largest tensor.
    budget: usize = 0,

    /// Raises the budget to `slots` of `largest`. Submissions only ever widen
    /// it, so a later, bigger tensor cannot shrink what is already running.
    fn widen(self: *StagingAdmission, io: std.Io, largest: usize, slots: usize) void {
        const wanted = largest *| @max(1, slots);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (wanted <= self.budget) return;
        self.budget = wanted;
        self.room.broadcast(io);
    }

    fn reserve(self: *StagingAdmission, io: std.Io, bytes: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        // A tensor wider than the whole budget still has to run, so anything
        // goes when nothing else is staging.
        while (self.in_flight != 0 and self.in_flight + bytes > self.budget)
            self.room.waitUncancelable(io, &self.mutex);
        self.in_flight += bytes;
    }

    fn release(self: *StagingAdmission, io: std.Io, bytes: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.in_flight -= bytes;
        self.room.broadcast(io);
    }
};

/// Source reads in flight across every task of the buffered backend. A
/// tensor wider than the profile's `read_chunk_size` splits into concurrent
/// range reads, and these permits keep the split from putting more reads on
/// the source than `read_parallelism`. A high-latency backend needs the
/// split: on `hf://` the profile asks for 32 MiB chunks, so a one-gigabyte
/// tensor read one chunk at a time is a chain of thirty-odd round trips.
const ReadPermits = struct {
    mutex: std.Io.Mutex = .init,
    available: std.Io.Condition = .init,
    free: usize,

    fn init(count: usize) ReadPermits {
        return .{ .free = @max(1, count) };
    }

    fn acquire(self: *ReadPermits, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.free == 0) self.available.waitUncancelable(io, &self.mutex);
        self.free -= 1;
    }

    /// Up to `count` permits, without waiting: how many were free is how much
    /// concurrency this tensor gets, so a loaded source is left alone.
    fn tryAcquire(self: *ReadPermits, io: std.Io, count: usize) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const taken = @min(count, self.free);
        self.free -= taken;
        return taken;
    }

    fn release(self: *ReadPermits, io: std.Io, count: usize) void {
        if (count == 0) return;
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.free += count;
        for (0..count) |_| self.available.signal(io);
    }
};

/// One tensor's read, claimed from a shared byte cursor by the owning task
/// and its helpers. Positional reads do not move the reader, so they can run
/// concurrently on one `TensorReader`.
///
/// A claim is the tensor split `workers` ways, and never smaller than the
/// profile's `read_chunk_size` -- that field is a minimum request size, and
/// on a high-latency source a request costs a round trip whether or not it
/// runs beside another. So one worker takes the whole tensor in a single
/// read, exactly as before this splitting existed, and only the concurrency
/// the source can actually carry turns into extra requests.
const ChunkedRead = struct {
    reader: *const safetensors.TensorReader,
    destination: []u8,
    chunk_size: usize,
    /// Reads this tensor may have in flight; the owner raises it when it
    /// takes another permit.
    workers: std.atomic.Value(usize) = .init(1),
    /// Bytes already claimed.
    next: std.atomic.Value(usize) = .init(0),
    failure: std.atomic.Value(u16) = .init(0),

    fn run(self: *ChunkedRead) void {
        while (self.readOne()) {}
    }

    /// Reads the next unclaimed span. False once the tensor is fully claimed,
    /// or once a read failed.
    fn readOne(self: *ChunkedRead) bool {
        if (self.failure.load(.acquire) != 0) return false;
        const span = self.claim() orelse return false;
        self.reader.readPositionalAll(span.bytes, span.offset) catch |err| {
            _ = self.failure.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
            return false;
        };
        return true;
    }

    fn claim(self: *ChunkedRead) ?struct { offset: usize, bytes: []u8 } {
        var start = self.next.load(.monotonic);
        while (start < self.destination.len) {
            const workers = @max(1, self.workers.load(.monotonic));
            const share = self.destination.len / workers +
                @intFromBool(self.destination.len % workers != 0);
            const len = @min(@max(self.chunk_size, share), self.destination.len - start);
            if (self.next.cmpxchgWeak(start, start + len, .monotonic, .monotonic)) |actual| {
                start = actual;
                continue;
            }
            return .{ .offset = start, .bytes = self.destination[start..][0..len] };
        }
        return null;
    }
};

/// Host staging for the buffered backend: one buffer per running read task,
/// handed back on completion instead of freed. Mapping, faulting and then
/// unmapping a whole tensor's worth of pages for every tensor was half of a
/// serial buffered load of a 14.96 GiB checkpoint, and a fifth of a
/// concurrent one.
///
/// The pool retains at most as many bytes as the running tasks ever held at
/// once, so it at most doubles host staging the backend already had to fit.
const StagingPool = struct {
    mutex: std.Io.Mutex = .init,
    /// Buffers no task holds, and their total length.
    free: std.ArrayListUnmanaged([]u8) = .empty,
    retained_bytes: usize = 0,
    /// What the running tasks hold now, and the most they ever held.
    live_bytes: usize = 0,
    peak_live_bytes: usize = 0,

    /// A whole allocation of at least `size` bytes, with undefined contents:
    /// the allocation itself, not a slice of one. The caller slices it for
    /// its own use and returns this exact slice to `release`, whatever
    /// happens.
    fn acquire(self: *StagingPool, allocator: std.mem.Allocator, io: std.Io, size: usize) ![]u8 {
        if (self.take(io, size)) |buffer| return buffer;
        const fresh = try allocator.alloc(u8, size);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.addLive(fresh.len);
        return fresh;
    }

    /// `buffer` must be what `acquire` returned, not a slice of it.
    fn release(self: *StagingPool, allocator: std.mem.Allocator, io: std.Io, buffer: []u8) void {
        const retained = retained: {
            self.mutex.lockUncancelable(io);
            defer self.mutex.unlock(io);
            self.live_bytes -= buffer.len;
            // Over budget: drop retained buffers this one supersedes, so the
            // pool keeps the sizes that satisfy the most requests instead of
            // whatever happened to be released first.
            while (self.retained_bytes + buffer.len > self.peak_live_bytes) {
                const index = self.smallest() orelse break;
                if (self.free.items[index].len >= buffer.len) break;
                const dropped = self.free.swapRemove(index);
                self.retained_bytes -= dropped.len;
                allocator.free(dropped);
            }
            if (self.retained_bytes + buffer.len > self.peak_live_bytes) break :retained false;
            self.free.append(allocator, buffer) catch break :retained false;
            self.retained_bytes += buffer.len;
            break :retained true;
        };
        if (!retained) allocator.free(buffer);
    }

    /// Every task is done, so every buffer is back.
    fn deinit(self: *StagingPool, allocator: std.mem.Allocator) void {
        for (self.free.items) |buffer| allocator.free(buffer);
        self.free.deinit(allocator);
    }

    /// The smallest retained buffer that fits, so the big ones stay free for
    /// the big tensors.
    fn take(self: *StagingPool, io: std.Io, size: usize) ?[]u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        var best: ?usize = null;
        for (self.free.items, 0..) |buffer, i| {
            if (buffer.len < size) continue;
            if (best == null or buffer.len < self.free.items[best.?].len) best = i;
        }
        const buffer = self.free.swapRemove(best orelse return null);
        self.retained_bytes -= buffer.len;
        self.addLive(buffer.len);
        return buffer;
    }

    fn smallest(self: *const StagingPool) ?usize {
        if (self.free.items.len == 0) return null;
        var index: usize = 0;
        for (self.free.items, 0..) |buffer, i| {
            if (buffer.len < self.free.items[index].len) index = i;
        }
        return index;
    }

    /// Called with `mutex` held.
    fn addLive(self: *StagingPool, bytes: usize) void {
        self.live_bytes += bytes;
        self.peak_live_bytes = @max(self.peak_live_bytes, self.live_bytes);
    }
};

test "chunked read splits a tensor only as wide as it has workers" {
    var destination: [1000]u8 = undefined;
    var work: ChunkedRead = .{
        .reader = undefined,
        .destination = &destination,
        .chunk_size = 100,
    };

    // One worker reads the whole tensor, exactly as an unsplit read would.
    const whole = work.claim().?;
    try std.testing.expectEqual(@as(usize, 0), whole.offset);
    try std.testing.expectEqual(@as(usize, 1000), whole.bytes.len);
    try std.testing.expect(work.claim() == null);

    // Four workers take a quarter each.
    work.next = .init(0);
    work.workers = .init(4);
    for (0..4) |i| {
        const span = work.claim().?;
        try std.testing.expectEqual(i * 250, span.offset);
        try std.testing.expectEqual(@as(usize, 250), span.bytes.len);
    }
    try std.testing.expect(work.claim() == null);

    // Never below the profile's chunk, however many workers there are.
    work.next = .init(0);
    work.workers = .init(50);
    const floored = work.claim().?;
    try std.testing.expectEqual(@as(usize, 100), floored.bytes.len);
}

test "staging admission bounds concurrent tensors by bytes and always admits one" {
    const io = std.testing.io;
    var admission: StagingAdmission = .{};

    // Three slots of the largest tensor is the budget.
    admission.widen(io, 100, 3);
    try std.testing.expectEqual(@as(usize, 300), admission.budget);
    // A later, smaller submission cannot shrink what is running.
    admission.widen(io, 10, 3);
    try std.testing.expectEqual(@as(usize, 300), admission.budget);

    // Small tensors run far wider than the slot count they were sized from.
    for (0..30) |_| admission.reserve(io, 10);
    try std.testing.expectEqual(@as(usize, 300), admission.in_flight);
    for (0..30) |_| admission.release(io, 10);

    // A tensor wider than the whole budget still runs, alone.
    admission.reserve(io, 900);
    try std.testing.expectEqual(@as(usize, 900), admission.in_flight);
    admission.release(io, 900);
    try std.testing.expectEqual(@as(usize, 0), admission.in_flight);
}

test "staging pool retains up to the peak the tasks held and prefers the sizes that fit" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var pool: StagingPool = .{};
    defer pool.deinit(allocator);

    // Two buffers live at once sets the budget: the pool may retain 24 bytes.
    const small = try pool.acquire(allocator, io, 8);
    const large = try pool.acquire(allocator, io, 16);
    try std.testing.expectEqual(@as(usize, 24), pool.peak_live_bytes);
    pool.release(allocator, io, small);
    pool.release(allocator, io, large);
    try std.testing.expectEqual(@as(usize, 24), pool.retained_bytes);

    // The smallest buffer that fits comes back, so the 16 stays free for a
    // request the 8 cannot serve.
    const reused = try pool.acquire(allocator, io, 4);
    try std.testing.expectEqual(@as(usize, 8), reused.len);
    try std.testing.expectEqual(@as(usize, 16), pool.retained_bytes);
    const fits = try pool.acquire(allocator, io, 12);
    try std.testing.expectEqual(@as(usize, 16), fits.len);
    try std.testing.expectEqual(@as(usize, 0), pool.retained_bytes);
    pool.release(allocator, io, reused);
    pool.release(allocator, io, fits);

    // A buffer no retained one can serve evicts the ones it supersedes
    // rather than being dropped itself.
    const big = try pool.acquire(allocator, io, 24);
    try std.testing.expectEqual(@as(usize, 24), big.len);
    pool.release(allocator, io, big);
    try std.testing.expectEqual(@as(usize, 1), pool.free.items.len);
    try std.testing.expectEqual(@as(usize, 24), pool.free.items[0].len);
    try std.testing.expectEqual(@as(usize, 24), pool.retained_bytes);
}
