//! Whole-tensor staging for platforms that use Buffer.from: TPU, neuron,
//! metal and furiosa, until the direct path is checked on them. A
//! composite destination is assembled in the staging: its gaps zeroed, each
//! part read at its offset (a strided part reads its source range once and
//! copies each run where it lands), then the whole tensor is placed as any
//! other. Read permits and a byte budget bound the host staging; there is
//! no calibration and no pinned memory. The shared front end owns source
//! lookup, programs and admission.
const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");

const Buffer = @import("../buffer.zig").Buffer;
const Memory = @import("../platform.zig").Memory;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const backend = @import("backend.zig");
const load_testing = @import("testing.zig");
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const log = @import("log.zig").load;

/// Concurrent whole tensors staged on the host: the byte budget is this
/// many times the largest tensor submitted. Kept at the former start width
/// rather than the read width, which on a high-latency profile would
/// triple the host staging for nothing this path can use (a tensor already
/// splits its reads across the width).
const staging_tensors: usize = 12;

pub const Loader = struct {
    pub const AwaitError = std.mem.Allocator.Error || pjrt.ApiError || std.Io.File.OpenError || safetensors.ExactPositionalError;
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
        opts: backend.Options,
    ) !*Loader {
        const read_parallelism = opts.readWidth();
        const profile = opts.load_profile;
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

    /// Spawns one bounded read task per spec, which assembles its
    /// destination. The batch keeps a copy of every spec's parts and their
    /// dimensions. Nothing runs when this fails.
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: std.Progress.Node) SubmitError!*Batch {
        try self.checkOpen();
        var largest: usize = 0;
        var part_count: usize = 0;
        var dim_count: usize = 0;
        for (specs) |spec| {
            largest = @max(largest, spec.shape.byteSize());
            part_count += spec.parts.len;
            for (spec.parts) |part| dim_count += part.dims.len;
        }
        const batch = try self.allocator.create(Batch);
        errdefer self.allocator.destroy(batch);
        const parts = try self.allocator.alloc(LoadPart, part_count);
        errdefer self.allocator.free(parts);
        const dims = try self.allocator.alloc(LoadPart.Dim, dim_count);
        batch.* = .{ .pending = .init(1 + specs.len), .parts = parts, .dims = dims };
        self.admission.widen(self.io, largest, self.staging_slots);
        var next: usize = 0;
        var next_dim: usize = 0;
        for (specs) |spec| {
            const spec_parts = parts[next..][0..spec.parts.len];
            next += spec.parts.len;
            for (spec_parts, spec.parts) |*copy, part| {
                copy.* = part;
                copy.dims = dims[next_dim..][0..part.dims.len];
                @memcpy(dims[next_dim..][0..part.dims.len], part.dims);
                next_dim += part.dims.len;
            }
            self.submitOne(batch, .{
                .shape = spec.shape,
                .memory = spec.memory,
                .output = spec.output,
                .parts = spec_parts,
            }, progress);
        }
        // Every task is spawned: drop the sentinel.
        batch.finish(self.io);
        return batch;
    }

    /// Waits for the batch's tasks, frees it and returns the sticky error.
    pub fn awaitBatch(self: *Loader, batch: *Batch) AwaitError!void {
        batch.done.waitUncancelable(self.io);
        self.allocator.free(batch.dims);
        self.allocator.free(batch.parts);
        self.allocator.destroy(batch);
        try self.checkOpen();
    }

    /// Fails the loader unless it already failed: tasks that have not
    /// started their tensor return at once.
    pub fn abort(self: *Loader) void {
        self.recordError(error.Canceled, "loader aborted", .{});
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

    fn loadOne(self: *Loader, spec: LoadSpec) void {
        if (self.first_error.load(.acquire) != 0) return;
        std.debug.assert(spec.parts.len != 0);
        var stage: []const u8 = "allocate staging for";
        var part = &spec.parts[0];
        self.loadOneStaged(spec, &stage, &part) catch |err| {
            self.recordError(err, "{s} tensor {s}: file={s}, offset={d}, source_bytes={d}, parts={d}, destination_bytes={d}", .{ stage, part.source.name, part.source.file_uri, part.source.offset, part.source.byteSize(), spec.parts.len, spec.shape.byteSize() });
        };
    }

    /// `loadOne`'s steps; `stage` names the one that failed and `part` the
    /// part it was at.
    fn loadOneStaged(
        self: *Loader,
        spec: LoadSpec,
        stage: *[]const u8,
        part: **const LoadPart,
    ) AwaitError!void {
        const tensor_bytes = spec.shape.byteSize();
        self.admission.reserve(self.io, tensor_bytes);
        defer self.admission.release(self.io, tensor_bytes);
        const staging = try self.staging.acquire(self.allocator, self.io, tensor_bytes);
        defer self.staging.release(self.allocator, self.io, staging);
        const tensor = staging[0..tensor_bytes];
        // Staging is reused, so a byte no part writes would publish another
        // tensor's bytes rather than zero: the gaps are zeroed first.
        var strided = false;
        for (spec.parts) |covered| strided = strided or covered.dims.len != 0;
        if (strided) {
            @memset(tensor, 0);
        } else {
            // Only the gaps: the parts overwrite the rest.
            var covered_end: usize = 0;
            for (spec.parts) |covered| {
                const start: usize = @intCast(covered.dst_offset);
                @memset(tensor[covered_end..start], 0);
                covered_end = start + @as(usize, @intCast(covered.width));
            }
            @memset(tensor[covered_end..], 0);
        }
        for (spec.parts) |*current| {
            part.* = current;
            stage.* = "open source for";
            const file = try std.Io.Dir.openFile(.cwd(), self.io, current.source.file_uri, .{ .mode = .read_only });
            defer file.close(self.io);
            stage.* = "read";
            const start: usize = @intCast(current.dst_offset);
            if (current.dims.len == 0) {
                try self.readInto(current.source, file, current.src_offset, tensor[start..][0..@intCast(current.width)]);
                continue;
            }
            // A strided part: its source range once, then each run where it lands.
            const lattice = current.lattice();
            const source = try self.allocator.alloc(u8, @intCast(lattice.spans().?.src));
            defer self.allocator.free(source);
            try self.readInto(current.source, file, current.src_offset, source);
            lattice.copyRuns(0, lattice.runs(), @intFromPtr(tensor.ptr), @intFromPtr(source.ptr) -% @as(usize, @intCast(current.src_offset)));
        }
        stage.* = "transfer";
        // The shape's sharding places the bytes; a shape without a mesh is
        // replicated, as `Buffer.from` places it.
        spec.output.* = try Buffer.from(
            self.io,
            self.platform,
            spec.shape,
            tensor,
            .{ .wait = true, .memory = spec.memory },
        );
    }

    /// Fills `destination` from `source`'s bytes at `source_offset`, read
    /// from `file`, splitting it across helper tasks when the source permits
    /// are free. The permits are released before the caller's transfer, so a
    /// tensor staging to the device lends its read budget to one still
    /// reading.
    fn readInto(
        self: *Loader,
        source: *const safetensors.Tensor,
        file: std.Io.File,
        source_offset: u64,
        destination: []u8,
    ) safetensors.ExactPositionalError!void {
        if (destination.len == 0) return;
        var work: ChunkedRead = .{
            .io = self.io,
            .source = source,
            .file = file,
            .source_offset = source_offset,
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

    /// `spec.parts` belong to the batch.
    fn submitOne(
        self: *Loader,
        batch: *Batch,
        spec: LoadSpec,
        progress: std.Progress.Node,
    ) void {
        self.group.async(self.io, struct {
            fn run(
                loader: *Loader,
                batch_: *Batch,
                spec_: LoadSpec,
                progress_: std.Progress.Node,
            ) void {
                defer batch_.finish(loader.io);
                // One item on the caller's node, as the direct backend counts
                // it: a child node per tensor runs out of std.Progress's 127.
                defer progress_.completeOne();
                loader.loadOne(spec_);
            }
        }.run, .{ self, batch, spec, progress });
    }
};

/// One buffered submission: `pending` counts a publish sentinel plus one
/// unit per tensor task; the last one sets `done`.
pub const Batch = struct {
    pending: std.atomic.Value(usize),
    done: std.Io.Event = .unset,
    /// Every spec's parts and their dimensions, which the tasks read until
    /// the batch is done.
    parts: []LoadPart,
    dims: []LoadPart.Dim,

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
/// and its helpers. Positional reads leave the file as it was, so they can
/// run concurrently on one file.
///
/// A claim is the tensor split `workers` ways, and never smaller than the
/// profile's `read_chunk_size` -- that field is a minimum request size, and
/// on a high-latency source a request costs a round trip whether or not it
/// runs beside another. So one worker takes the whole tensor in a single
/// read, exactly as before this splitting existed, and only the concurrency
/// the source can actually carry turns into extra requests.
const ChunkedRead = struct {
    io: std.Io,
    source: *const safetensors.Tensor,
    file: std.Io.File,
    /// Where `destination` starts in the tensor's bytes.
    source_offset: u64 = 0,
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
        self.source.readRange(self.io, self.file, span.bytes, self.source_offset + span.offset) catch |err| {
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
    free: std.ArrayList([]u8) = .empty,
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
        .io = undefined,
        .source = undefined,
        .file = undefined,
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

test "buffered loader assembles a composite destination in reused staging" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    // Always on an owned CPU platform: Buffer.from places on the platform's
    // whole mesh, which a sub-mesh of an accelerator's devices is not.
    var devices: load_testing.Devices = undefined;
    try devices.initCpu(allocator, io, 2);
    defer devices.deinit();
    const platform = devices.platform;
    const mesh = try devices.mesh("tp");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    var a: [48]u8 = undefined;
    load_testing.pattern(&a, 1);
    var b: [32]u8 = undefined;
    load_testing.pattern(&b, 2);
    var full: [128]u8 = undefined;
    load_testing.pattern(&full, 3);
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "weights.safetensors", &registry, &.{
        .{ .name = "a", .dtype = "U16", .dims = &.{ 3, 8 }, .bytes = &a },
        .{ .name = "b", .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &b },
        .{ .name = "full", .dtype = "U16", .dims = &.{ 8, 8 }, .bytes = &full },
    });
    // Rows split over the two devices: b at row 1 and a's last two rows
    // (from byte 16 of a) at row 4 of 8, around a leading row, a gap and
    // a tail of zeros.
    const shape = Shape.init(.{ .row = 8, .col = 8 }, .u16).withPartitioning(&mesh, .{ .row = .model });
    const parts = [_]LoadPart{
        .{ .source = registry.tensors.getPtr("b").?, .dst_offset = 16, .width = 32 },
        .{ .source = registry.tensors.getPtr("a").?, .src_offset = 16, .dst_offset = 64, .width = 32 },
    };
    var expected: [128]u8 = @splat(0);
    @memcpy(expected[16..48], &b);
    @memcpy(expected[64..96], a[16..48]);
    const whole = [_]LoadPart{.whole(registry.tensors.getPtr("full").?)};

    const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.destroy();
    for ([_]Memory.Kind{ .default, .host_pinned }) |memory| {
        // A whole tensor first, so the composite reuses its staging buffer
        // and only the zeroing keeps its bytes out of the gaps.
        var first: Buffer = undefined;
        try loader.awaitBatch(try loader.submit(&.{.{ .shape = shape, .memory = memory, .output = &first, .parts = &whole }}, .none));
        defer first.deinit();
        try load_testing.expectShards(allocator, io, &first, &full);
        try expectMemory(platform, &first, memory);
        var output: Buffer = undefined;
        try loader.awaitBatch(try loader.submit(&.{.{ .shape = shape, .memory = memory, .output = &output, .parts = &parts }}, .none));
        defer output.deinit();
        try load_testing.expectShards(allocator, io, &output, &expected);
        try expectMemory(platform, &output, memory);
        try std.testing.expectEqual(@as(usize, 128), loader.staging.retained_bytes);
    }

    // A source whose file does not exist fails the worker, and the error
    // is sticky from then on.
    var missing = registry.tensors.get("full").?;
    missing.file_uri = "missing.safetensors";
    var never_written: Buffer = undefined;
    const batch = try loader.submit(&.{.{ .shape = shape, .output = &never_written, .parts = &.{.whole(&missing)} }}, .none);
    try std.testing.expectError(error.FileNotFound, loader.awaitBatch(batch));
    try std.testing.expectError(error.FileNotFound, loader.submit(&.{.{ .shape = shape, .output = &never_written, .parts = &whole }}, .none));
}

/// Every shard of `buffer` is in its device's `memory` space.
fn expectMemory(platform: *const Platform, buffer: *const Buffer, memory: Memory.Kind) !void {
    const devices = platform.placedShape(buffer.shape())._sharding.mesh.?.devicesInCanonicalOrder();
    try std.testing.expectEqual(devices.len, buffer._shards.len);
    for (devices, buffer._shards.constSlice()) |device, shard| {
        try std.testing.expect(shard.memory(platform.pjrt_api) == platform.devices[device.id].memory(memory).?.pjrt_memory);
    }
}
