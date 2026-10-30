//! Whole-tensor staging for platforms that use Buffer.from: TPU, neuron
//! and metal, until the direct path is checked on them. A composite
//! destination is assembled in the staging: its gaps zeroed, each
//! part read at its offset (a strided part reads its source range once and
//! copies each run where it lands), then the whole tensor is placed as any
//! other. Each batch runs up to the read width of workers over its specs;
//! read permits bound the source reads and one byte budget the host
//! staging, live and kept for reuse. There is no calibration and no pinned
//! memory. The shared front end owns source lookup, programs and admission.
const std = @import("std");

const stdx = @import("stdx");

const Buffer = @import("../buffer.zig").Buffer;
const Platform = @import("../platform.zig").Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const backend = @import("backend.zig");
const testing = @import("../testing.zig");
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const log = @import("log.zig").load;

/// The host staging budget in largest requests: this many times the
/// largest staging request submitted (`stagingBytes`), for the buffers in
/// use and those kept for reuse alike. Twelve rather than the read width:
/// at a high-latency width of 32 the read width would nearly triple the
/// host staging for nothing this path can use (a tensor already splits its
/// reads across the width).
const staging_tensors: usize = 12;

pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    /// Workers per batch, so the threads grow with the batches in flight:
    /// the read permits bound their reads and the staging budget their
    /// bytes, not their number.
    width: usize,
    /// The read hints' minimum read size: one tensor becomes this many
    /// bytes per positional read.
    read_chunk_size: usize,
    /// The widest fan-out one tensor may take. One on a source whose reads
    /// are bandwidth-bound rather than round-trip bound: splitting there only
    /// spends tasks, and measurably so -- a replicated local load, whose long
    /// transfers keep permits free, lost time to helpers it had no use for.
    tensor_workers: usize,
    permits: ReadPermits,
    /// The staging budget in largest requests (`staging_tensors`).
    staging_slots: usize,
    staging: Staging = .{},
    first_error: stdx.FirstError(anyerror) = .{},

    pub fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: backend.Options,
    ) !*Loader {
        const self = try allocator.create(Loader);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .width = opts.read_parallelism,
            .staging_slots = @min(opts.read_parallelism, staging_tensors),
            .read_chunk_size = opts.read_hints.read_chunk_size,
            // A high-latency source may use spare read permits to split a
            // tensor already staging: more concurrency without more host memory.
            .tensor_workers = if (opts.read_hints.high_latency) opts.read_parallelism else 1,
            .permits = .init(opts.read_parallelism),
        };
        return self;
    }

    /// Starts up to `width` workers over the specs, each assembling one
    /// destination at a time from its parts. The batch borrows `specs` and
    /// everything they name until it is awaited (`backend.LoadSpec`). On an
    /// io that starts no worker, the batch loads inside this call. Nothing
    /// runs when this fails.
    pub fn submit(self: *Loader, specs: []const LoadSpec, progress: std.Progress.Node) !*Batch {
        try self.checkOpen();
        var largest: usize = 0;
        for (specs) |spec| largest = @max(largest, stagingBytes(spec));
        const batch = try self.allocator.create(Batch);
        batch.* = .{ .specs = specs, .progress = progress };
        self.staging.widen(self.io, largest, self.staging_slots);
        // `concurrent`, never `async`: a busy Threaded io runs `async`
        // inline, and an inline first worker would drain the whole cursor
        // here.
        var started: usize = 0;
        for (0..@min(self.width, specs.len)) |_| {
            batch.workers.concurrent(self.io, work, .{ self, batch }) catch break;
            started += 1;
        }
        if (started == 0) self.work(batch);
        return batch;
    }

    /// Waits for the batch's workers, frees it and returns the sticky error.
    /// Uncancelable, as the direct backend's: the workers borrow the batch
    /// and its specs until they return.
    pub fn awaitBatch(self: *Loader, batch: *Batch) !void {
        const protection = self.io.swapCancelProtection(.blocked);
        batch.workers.await(self.io) catch |err| switch (err) {
            error.Canceled => unreachable,
        };
        _ = self.io.swapCancelProtection(protection);
        self.allocator.destroy(batch);
        try self.checkOpen();
    }

    /// Fails the loader unless it already failed: workers start no other
    /// tensor, and one waiting for its staging returns once it gets it.
    pub fn abort(self: *Loader) void {
        self.recordError(error.Canceled, "loader aborted", .{});
    }

    /// Every batch was awaited, so no worker runs.
    pub fn destroy(self: *Loader) void {
        self.staging.deinit(self.allocator);
        self.allocator.destroy(self);
    }

    fn recordError(self: *Loader, err: anyerror, comptime fmt: []const u8, args: anytype) void {
        if (self.first_error.set(err)) log.failure(err, fmt, args);
    }

    fn checkOpen(self: *Loader) !void {
        if (self.first_error.get()) |err| return err;
    }

    /// One of the batch's workers: loads the next unclaimed spec until none
    /// is left.
    fn work(self: *Loader, batch: *Batch) void {
        while (true) {
            const index = batch.next.fetchAdd(1, .monotonic);
            if (index >= batch.specs.len) return;
            self.loadOne(batch.specs[index]);
            // One item on the caller's node, as the direct backend counts
            // it: a child node per tensor runs out of std.Progress's 127.
            batch.progress.completeOne();
        }
    }

    fn loadOne(self: *Loader, spec: LoadSpec) void {
        if (self.first_error.get() != null) return;
        std.debug.assert(spec.parts.len != 0);
        const tensor_bytes = spec.shape.byteSize();
        const staging = self.staging.acquire(self.allocator, self.io, stagingBytes(spec)) catch |err|
            return self.fail(err, "allocate staging for", spec, &spec.parts[0]);
        defer self.staging.release(self.allocator, self.io, staging);
        // The staging may have taken a while: a failure meanwhile leaves
        // this tensor unstarted.
        if (self.first_error.get() != null) return;
        const tensor = staging[0..tensor_bytes];
        // Staging is reused, so a byte no part writes would publish another
        // tensor's bytes rather than zero: the gaps are zeroed first.
        var strided = false;
        for (spec.parts) |part| strided = strided or part.lattice.dims.len != 0;
        if (strided) {
            @memset(tensor, 0);
        } else {
            // Only the gaps: the parts overwrite the rest.
            var covered_end: usize = 0;
            for (spec.parts) |part| {
                const start: usize = @intCast(part.lattice.dst_offset);
                @memset(tensor[covered_end..start], 0);
                covered_end = start + @as(usize, @intCast(part.lattice.width));
            }
            @memset(tensor[covered_end..], 0);
        }
        for (spec.parts) |*part| {
            const file = std.Io.Dir.openFile(.cwd(), self.io, part.source.file_uri, .{ .mode = .read_only }) catch |err|
                return self.fail(err, "open source for", spec, part);
            defer file.close(self.io);
            const lattice = part.lattice;
            if (lattice.dims.len == 0) {
                const start: usize = @intCast(lattice.dst_offset);
                self.readInto(part.source, file, lattice.src_offset, tensor[start..][0..@intCast(lattice.width)]) catch |err|
                    return self.fail(err, "read", spec, part);
                continue;
            }
            // A strided part: its source range once, into the staging past
            // the tensor, then each run where it lands.
            const source = staging[tensor_bytes..][0..@intCast(lattice.spans().?.src)];
            self.readInto(part.source, file, lattice.src_offset, source) catch |err|
                return self.fail(err, "read", spec, part);
            lattice.copyRuns(0, lattice.runs(), @intFromPtr(tensor.ptr), @intFromPtr(source.ptr) -% @as(usize, @intCast(lattice.src_offset)));
        }
        // The front end asks no host kind of this backend
        // (`backend.supportsMemory`).
        std.debug.assert(spec.memory == .default);
        // The shape's sharding places the bytes; a shape without a mesh is
        // replicated, as `Buffer.from` places it.
        spec.output.* = Buffer.from(self.io, self.platform, spec.shape, tensor, .{ .wait = true }) catch |err|
            return self.fail(err, "transfer", spec, &spec.parts[0]);
    }

    /// Fails the loader with `err`, which `step` of `spec` met at `part`.
    fn fail(self: *Loader, err: anyerror, step: []const u8, spec: LoadSpec, part: *const LoadPart) void {
        self.recordError(err, "{s} tensor {s}: file={s}, offset={d}, source_bytes={d}, parts={d}, destination_bytes={d}", .{ step, part.source.name, part.source.file_uri, part.source.offset, part.source.byteSize(), spec.parts.len, spec.shape.byteSize() });
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
        var chunked: ChunkedRead = .{
            .io = self.io,
            .source = source,
            .file = file,
            .source_offset = source_offset,
            .destination = destination,
            .chunk_size = self.read_chunk_size,
        };
        // Splitting below the read chunk buys nothing, so that is the
        // most workers this tensor can use.
        var useful_workers = @min(
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
            if (held < useful_workers and chunked.next.load(.monotonic) < destination.len) {
                const extra = self.permits.tryAcquire(self.io, useful_workers - held);
                if (extra != 0) {
                    // Claims size themselves by the width, so it is raised
                    // before a helper can claim. `concurrent`, since a busy
                    // io runs `async` inline, on this task.
                    chunked.workers.store(held + extra, .monotonic);
                    var started: usize = 0;
                    while (started < extra) : (started += 1) {
                        group.concurrent(self.io, ChunkedRead.run, .{&chunked}) catch break;
                    }
                    held += started;
                    if (started < extra) {
                        // The io starts no more: the rest of the permits go
                        // back, and this tensor stops asking.
                        self.permits.release(self.io, extra - started);
                        chunked.workers.store(held, .monotonic);
                        useful_workers = held;
                    }
                }
            }
            if (!chunked.readOne()) break;
        }
        group.await(self.io) catch {};
        self.permits.release(self.io, held);

        if (chunked.failure.get()) |err| return err;
    }
};

/// The staging one spec takes: its tensor, then room for its widest strided
/// part's source range, which is read once before its runs are copied out.
fn stagingBytes(spec: LoadSpec) usize {
    var span: usize = 0;
    for (spec.parts) |part| {
        if (part.lattice.dims.len != 0) span = @max(span, @as(usize, @intCast(part.lattice.spans().?.src)));
    }
    return spec.shape.byteSize() + span;
}

/// One buffered submission: its workers claim specs from `next`.
pub const Batch = struct {
    specs: []const LoadSpec,
    progress: std.Progress.Node,
    next: std.atomic.Value(usize) = .init(0),
    workers: std.Io.Group = .init,
};

/// Source reads in flight across every task of the buffered backend. A
/// tensor wider than the hints' `read_chunk_size` splits into concurrent
/// range reads, and these permits keep the split from putting more reads on
/// the source than `read_parallelism`. A high-latency backend needs the
/// split: on `hf://` the hints ask for 32 MiB chunks, so a one-gigabyte
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
/// hints' `read_chunk_size` -- that field is a minimum request size, and
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
    /// Reads this tensor may have in flight: the owner raises it before it
    /// starts helpers on newly taken permits, and lowers it to the helpers
    /// that started.
    workers: std.atomic.Value(usize) = .init(1),
    /// Bytes already claimed.
    next: std.atomic.Value(usize) = .init(0),
    failure: stdx.FirstError(safetensors.ExactPositionalError) = .{},

    fn run(self: *ChunkedRead) void {
        while (self.readOne()) {}
    }

    /// Reads the next unclaimed span. False once the tensor is fully claimed,
    /// or once a read failed.
    fn readOne(self: *ChunkedRead) bool {
        if (self.failure.get() != null) return false;
        const span = self.claim() orelse return false;
        self.source.readRange(self.io, self.file, span.bytes, self.source_offset + span.offset) catch |err| {
            _ = self.failure.set(err);
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

/// Host staging for the buffered backend: one budget for the buffers the
/// running tensors hold and those kept for reuse, which are handed back on
/// completion instead of freed. Mapping, faulting and then unmapping a whole
/// tensor's worth of pages for every tensor was half of a serial buffered
/// load of a 14.96 GiB checkpoint, and a fifth of a concurrent one.
///
/// The budget is `staging_slots` times the largest request submitted, so
/// no request is wider than it. A request takes the smallest kept buffer
/// that fits, which counts in full while it is held: small tensors run far
/// wider than the slot count on small buffers, but once only large ones
/// are kept they take those, and run about `staging_slots` wide.
const Staging = struct {
    mutex: std.Io.Mutex = .init,
    room: std.Io.Condition = .init,
    /// Zero until the first submission reports its largest request.
    budget: usize = 0,
    /// What the running tensors hold.
    live_bytes: usize = 0,
    /// Buffers no tensor holds, and their total length.
    free: std.ArrayList([]u8) = .empty,
    retained_bytes: usize = 0,

    /// Raises the budget to `slots` of `largest`. Submissions only ever widen
    /// it: a later submission of smaller tensors cannot shrink what is
    /// already running.
    fn widen(self: *Staging, io: std.Io, largest: usize, slots: usize) void {
        const wanted = largest *| @max(1, slots);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (wanted <= self.budget) return;
        self.budget = wanted;
        self.room.broadcast(io);
    }

    /// A whole allocation of at least `size` bytes, with undefined contents:
    /// the allocation itself, not a slice of one. The caller slices it for
    /// its own use and returns this exact slice to `release`, whatever
    /// happens. Waits until a retained buffer fits, or `size` more bytes
    /// fit the budget beside the live ones. `size` is at most the budget:
    /// `widen` ran for the request's submission.
    fn acquire(self: *Staging, allocator: std.mem.Allocator, io: std.Io, size: usize) ![]u8 {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(size <= self.budget);
        while (true) {
            if (self.take(size)) |buffer| return buffer;
            if (self.live_bytes + size <= self.budget) break;
            self.room.waitUncancelable(io, &self.mutex);
        }
        // No retained buffer fits, so each is smaller than this request:
        // free them, smallest first, until it fits beside the rest.
        while (self.live_bytes + self.retained_bytes + size > self.budget) {
            const dropped = self.free.swapRemove(self.smallest());
            self.retained_bytes -= dropped.len;
            allocator.free(dropped);
        }
        const fresh = try allocator.alloc(u8, size);
        self.live_bytes += fresh.len;
        std.debug.assert(self.live_bytes + self.retained_bytes <= self.budget);
        return fresh;
    }

    /// `buffer` must be what `acquire` returned, not a slice of it. It is
    /// kept: the budget already counts it.
    fn release(self: *Staging, allocator: std.mem.Allocator, io: std.Io, buffer: []u8) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.live_bytes -= buffer.len;
        self.room.broadcast(io);
        self.free.append(allocator, buffer) catch return allocator.free(buffer);
        self.retained_bytes += buffer.len;
    }

    /// Every tensor is done, so every buffer is back.
    fn deinit(self: *Staging, allocator: std.mem.Allocator) void {
        std.debug.assert(self.live_bytes == 0);
        for (self.free.items) |buffer| allocator.free(buffer);
        self.free.deinit(allocator);
    }

    /// The smallest retained buffer that fits, now live, so the big ones
    /// stay free for the big tensors. Called with `mutex` held.
    fn take(self: *Staging, size: usize) ?[]u8 {
        var best: ?usize = null;
        for (self.free.items, 0..) |buffer, i| {
            if (buffer.len < size) continue;
            if (best == null or buffer.len < self.free.items[best.?].len) best = i;
        }
        const buffer = self.free.swapRemove(best orelse return null);
        self.retained_bytes -= buffer.len;
        self.live_bytes += buffer.len;
        return buffer;
    }

    /// Called with `mutex` held and a buffer retained.
    fn smallest(self: *const Staging) usize {
        var index: usize = 0;
        for (self.free.items, 0..) |buffer, i| {
            if (buffer.len < self.free.items[index].len) index = i;
        }
        return index;
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

    // Never below the read chunk, however many workers there are.
    work.next = .init(0);
    work.workers = .init(50);
    const floored = work.claim().?;
    try std.testing.expectEqual(@as(usize, 100), floored.bytes.len);
}

test "staging holds live and retained bytes within one budget" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var staging: Staging = .{};
    defer staging.deinit(allocator);

    // Three slots of the largest request is the budget, and a later,
    // smaller submission cannot shrink it.
    staging.widen(io, 16, 3);
    staging.widen(io, 4, 3);
    try std.testing.expectEqual(@as(usize, 48), staging.budget);

    // With nothing kept, small requests run far wider than the slot count,
    // and their buffers are kept.
    var small: [12][]u8 = undefined;
    for (&small) |*buffer| buffer.* = try staging.acquire(allocator, io, 4);
    try std.testing.expectEqual(@as(usize, 48), staging.live_bytes);
    for (small) |buffer| staging.release(allocator, io, buffer);
    try std.testing.expectEqual(@as(usize, 48), staging.retained_bytes);

    // A request no retained buffer fits frees as many as it needs room
    // from.
    const large = try staging.acquire(allocator, io, 16);
    try std.testing.expectEqual(@as(usize, 32), staging.retained_bytes);
    staging.release(allocator, io, large);
    // The smallest buffer that fits comes back to a smaller request, so the
    // 16 stays for one the 4s cannot serve. The first take moves the 16 to
    // the front of the list, ahead of the second's 4.
    var reused: [2][]u8 = undefined;
    for (&reused) |*buffer| {
        buffer.* = try staging.acquire(allocator, io, 3);
        try std.testing.expectEqual(@as(usize, 4), buffer.len);
    }
    try std.testing.expectEqual(@as(usize, 40), staging.retained_bytes);
    for (reused) |buffer| staging.release(allocator, io, buffer);
    try std.testing.expectEqual(@as(usize, 48), staging.retained_bytes);

    // A full budget holds the next request until a buffer comes back, then
    // hands it that buffer.
    var held: [3][]u8 = undefined;
    for (&held) |*buffer| buffer.* = try staging.acquire(allocator, io, 16);
    try std.testing.expectEqual(@as(usize, 48), staging.live_bytes + staging.retained_bytes);
    var waiting = try io.concurrent(Staging.acquire, .{ &staging, allocator, io, 16 });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expectEqual(@as(usize, 48), staging.live_bytes);
    staging.release(allocator, io, held[0]);
    held[0] = try waiting.await(io);
    try std.testing.expectEqual(@as(usize, 48), staging.live_bytes);
    for (held) |buffer| staging.release(allocator, io, buffer);

    // Once only large buffers are kept, small requests take them, each
    // counted in full: three fill the budget.
    var on_large: [3][]u8 = undefined;
    for (&on_large) |*buffer| buffer.* = try staging.acquire(allocator, io, 4);
    try std.testing.expectEqual(@as(usize, 48), staging.live_bytes);
    for (on_large) |buffer| staging.release(allocator, io, buffer);
    try std.testing.expectEqual(@as(usize, 48), staging.retained_bytes);
}

test "buffered loader assembles a composite destination in reused staging" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 2);
    defer devices.deinit();
    const platform = devices.platform;
    const mesh = try devices.mesh("tp");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    var a: [48]u8 = undefined;
    testing.pattern(&a, 1);
    var b: [32]u8 = undefined;
    testing.pattern(&b, 2);
    var full: [128]u8 = undefined;
    testing.pattern(&full, 3);
    var big: [256]u8 = undefined;
    testing.pattern(&big, 4);
    try testing.writeSafetensors(allocator, io, tmp.dir, "weights.safetensors", &registry, &.{
        .{ .name = "a", .dtype = "U16", .dims = &.{ 3, 8 }, .bytes = &a },
        .{ .name = "b", .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &b },
        .{ .name = "full", .dtype = "U16", .dims = &.{ 8, 8 }, .bytes = &full },
        .{ .name = "big", .dtype = "U16", .dims = &.{ 16, 8 }, .bytes = &big },
    });
    // Rows split over the two devices: b at row 1, a's last two rows (from
    // byte 16 of a) at row 4, and a strided part, the left halves of full's
    // rows 0 and 2, at rows 6 and 7, around a leading row, a gap and the
    // right halves of zeros.
    const shape = Shape.init(.{ .row = 8, .col = 8 }, .u16).withPartitioning(&mesh, .{ .row = .model });
    const columns = [_]backend.Lattice.Dim{.{ .count = 2, .src_stride = 32, .dst_stride = 16 }};
    const parts = [_]LoadPart{
        .{ .source = registry.tensors.getPtr("b").?, .lattice = .{ .dst_offset = 16, .width = 32 } },
        .{ .source = registry.tensors.getPtr("a").?, .lattice = .{ .src_offset = 16, .dst_offset = 64, .width = 32 } },
        .{ .source = registry.tensors.getPtr("full").?, .lattice = .{ .dst_offset = 96, .width = 8, .dims = &columns } },
    };
    try std.testing.expect(backend.checkParts(&parts, shape.byteSize()));
    var expected: [128]u8 = @splat(0);
    @memcpy(expected[16..48], &b);
    @memcpy(expected[64..96], a[16..48]);
    @memcpy(expected[96..104], full[0..8]);
    @memcpy(expected[112..120], full[32..40]);
    const whole = [_]LoadPart{.whole(registry.tensors.getPtr("full").?)};
    const big_shape = Shape.init(.{ .row = 16, .col = 8 }, .u16).withPartitioning(&mesh, .{ .row = .model });
    const big_whole = [_]LoadPart{.whole(registry.tensors.getPtr("big").?)};

    const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.destroy();
    // A bigger tensor first, so the composite, its strided source range
    // included, reuses its staging buffer and only the zeroing keeps its
    // bytes out of the gaps.
    var first: Buffer = undefined;
    try loader.awaitBatch(try loader.submit(&.{.{ .shape = big_shape, .output = &first, .parts = &big_whole }}, .none));
    defer first.deinit();
    try testing.expectShards(allocator, io, &first, &big, .default);
    var output: Buffer = undefined;
    try loader.awaitBatch(try loader.submit(&.{.{ .shape = shape, .output = &output, .parts = &parts }}, .none));
    defer output.deinit();
    try testing.expectShards(allocator, io, &output, &expected, .default);
    // One buffer was ever allocated, within the budget of two slots of the
    // largest request.
    try std.testing.expectEqual(@as(usize, 512), loader.staging.budget);
    try std.testing.expectEqual(@as(usize, 256), loader.staging.retained_bytes);

    // A source whose file does not exist fails the worker, and the error
    // is sticky from then on.
    var missing = registry.tensors.get("full").?;
    missing.file_uri = "missing.safetensors";
    const missing_parts = [_]LoadPart{.whole(&missing)};
    var never_written: Buffer = undefined;
    const batch = try loader.submit(&.{.{ .shape = shape, .output = &never_written, .parts = &missing_parts }}, .none);
    try std.testing.expectError(error.FileNotFound, loader.awaitBatch(batch));
    // `expectError` would print the batch, and every type its specs name.
    if (loader.submit(&.{.{ .shape = shape, .output = &never_written, .parts = &whole }}, .none)) |_|
        return error.TestUnexpectedResult
    else |err|
        try std.testing.expectEqual(error.FileNotFound, err);
}

test "a worker waiting for its staging loads nothing once the loader failed" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, io, 1);
    defer devices.deinit();

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    var bytes: [64]u8 = undefined;
    testing.pattern(&bytes, 1);
    try testing.writeSafetensors(allocator, io, tmp.dir, "weights.safetensors", &registry, &.{
        .{ .name = "x", .dtype = "U8", .dims = &.{ 8, 8 }, .bytes = &bytes },
    });
    const shape = Shape.init(.{ .row = 8, .col = 8 }, .u8);
    const parts = [_]LoadPart{.whole(registry.tensors.getPtr("x").?)};

    const loader = try Loader.create(allocator, io, devices.platform, .{ .read_parallelism = 1 });
    defer loader.destroy();
    // The test holds the whole budget, one slot of the tensor, so the
    // worker waits for its staging until the loader failed.
    loader.staging.widen(io, bytes.len, loader.staging_slots);
    const held = try loader.staging.acquire(allocator, io, bytes.len);
    var output: Buffer = .{ ._platform = devices.platform, ._shape = shape, ._shards = .empty };
    defer output.deinit();
    const batch = try loader.submit(&.{.{ .shape = shape, .output = &output, .parts = &parts }}, .none);
    try io.sleep(.fromMilliseconds(5), .awake);
    loader.abort();
    loader.staging.release(allocator, io, held);
    try std.testing.expectError(error.Canceled, loader.awaitBatch(batch));
    try std.testing.expectEqual(@as(usize, 0), output._shards.len);
}

test "buffered loader runs overlapping batches over their cursors, with or without concurrency" {
    const allocator = std.testing.allocator;
    var devices: testing.Devices = undefined;
    try devices.init(allocator, std.testing.io, 2);
    defer devices.deinit();
    const mesh = try devices.mesh("tp");

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    // Tensors of 2 to 32 rows of 8 u16, so the staging budget holds a
    // varying number of them.
    const count = 24;
    var bytes: [count][512]u8 = undefined;
    var names: [count][8]u8 = undefined;
    var dims: [count][2]i64 = undefined;
    var entries: [count]testing.TensorBytes = undefined;
    for (&bytes, &names, &dims, &entries, 0..) |*data, *name, *dim, *entry, i| {
        const rows = 2 * (i % 16 + 1);
        testing.pattern(data[0 .. rows * 16], @intCast(i + 1));
        dim.* = .{ @intCast(rows), 8 };
        entry.* = .{ .name = try std.fmt.bufPrint(name, "t{d}", .{i}), .dtype = "U16", .dims = dim, .bytes = data[0 .. rows * 16] };
    }
    try testing.writeSafetensors(allocator, std.testing.io, tmp.dir, "weights.safetensors", &registry, &entries);

    var parts: [count][1]LoadPart = undefined;
    var specs: [count]LoadSpec = undefined;
    var outputs: [count]Buffer = undefined;
    for (&parts, &specs, &outputs, entries) |*part, *spec, *output, entry| {
        part[0] = .whole(registry.tensors.getPtr(entry.name).?);
        const shape = Shape.init(.{ .row = entry.dims[0], .col = 8 }, .u16).withPartitioning(&mesh, .{ .row = .model });
        spec.* = .{ .shape = shape, .output = output, .parts = part };
    }

    var no_concurrency: std.Io.Threaded = .init(allocator, .{ .async_limit = .nothing, .concurrent_limit = .nothing });
    defer no_concurrency.deinit();
    for ([_]std.Io{ std.testing.io, no_concurrency.io() }) |io| {
        // A high-latency source with 32-byte chunks: helpers split each
        // tensor whenever a permit is free.
        const loader = try Loader.create(allocator, io, devices.platform, .{
            .read_parallelism = 4,
            .read_hints = .{ .read_chunk_size = 32, .high_latency = true },
        });
        defer loader.destroy();
        // Both batches are in flight before either is awaited, each with
        // its own workers: the permits bound their reads and the staging
        // budget their bytes (`Staging.acquire` asserts it).
        const first = try loader.submit(specs[0 .. count / 2], .none);
        const second = try loader.submit(specs[count / 2 ..], .none);
        try loader.awaitBatch(first);
        try loader.awaitBatch(second);
        defer for (&outputs) |*output| output.deinit();
        for (&outputs, entries) |*output, entry| try testing.expectShards(allocator, io, output, entry.bytes, .default);
        try std.testing.expect(loader.staging.retained_bytes <= loader.staging.budget);
        try std.testing.expectEqual(@as(usize, 0), loader.staging.live_bytes);
        try std.testing.expectEqual(@as(usize, 4), loader.permits.free);
    }
}
