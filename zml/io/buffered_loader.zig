//! Whole-tensor staging for platforms that use Buffer.from: TPU, neuron
//! and metal, until the direct path is checked on them. A composite
//! destination is assembled in the staging: its gaps zeroed, each
//! part read at its offset (a strided part reads its source range once and
//! copies each run where it lands), then the whole tensor is placed as any
//! other. Each batch runs up to the read width of workers over its specs;
//! each tensor allocates its host staging and frees it once placed. There
//! is no calibration and no pinned memory. The shared front end owns source
//! lookup, programs and admission.
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

pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    /// Workers per batch, so the threads grow with the batches in flight.
    width: usize,
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
        const batch = try self.allocator.create(Batch);
        batch.* = .{ .specs = specs, .progress = progress };
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
    /// Uncancelable: the workers borrow the batch and its specs until they
    /// return.
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
    /// tensor.
    pub fn abort(self: *Loader) void {
        self.recordError(error.Canceled, "loader aborted", .{});
    }

    /// Every batch was awaited, so no worker runs.
    pub fn destroy(self: *Loader) void {
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
            // One item on the caller's node: a child node per tensor runs
            // out of std.Progress's 127.
            batch.progress.completeOne();
        }
    }

    fn loadOne(self: *Loader, spec: LoadSpec) void {
        if (self.first_error.get() != null) return;
        std.debug.assert(spec.parts.len != 0);
        const tensor_bytes = spec.shape.byteSize();
        const staging = self.allocator.alloc(u8, stagingBytes(spec)) catch |err|
            return self.fail(err, "allocate staging for", spec, &spec.parts[0]);
        defer self.allocator.free(staging);
        const tensor = staging[0..tensor_bytes];
        // The staging's contents are undefined, so a byte no part writes
        // would publish garbage rather than zero: the gaps are zeroed first.
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
                part.source.readRange(self.io, file, tensor[start..][0..@intCast(lattice.width)], lattice.src_offset) catch |err|
                    return self.fail(err, "read", spec, part);
                continue;
            }
            // A strided part: its source range once, into the staging past
            // the tensor, then each run where it lands.
            const source = staging[tensor_bytes..][0..@intCast(lattice.spans().?.src)];
            part.source.readRange(self.io, file, source, lattice.src_offset) catch |err|
                return self.fail(err, "read", spec, part);
            lattice.copyRuns(0, lattice.runs(), @intFromPtr(tensor.ptr), @intFromPtr(source.ptr) -% @as(usize, @intCast(lattice.src_offset)));
        }
        // The front end asks no host kind of this backend.
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

test "buffered loader assembles a composite destination" {
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
    try testing.writeSafetensors(allocator, io, tmp.dir, "weights.safetensors", &registry, &.{
        .{ .name = "a", .dtype = "U16", .dims = &.{ 3, 8 }, .bytes = &a },
        .{ .name = "b", .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &b },
        .{ .name = "full", .dtype = "U16", .dims = &.{ 8, 8 }, .bytes = &full },
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

    const loader = try Loader.create(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.destroy();
    var output: Buffer = undefined;
    try loader.awaitBatch(try loader.submit(&.{.{ .shape = shape, .output = &output, .parts = &parts }}, .none));
    defer output.deinit();
    try testing.expectShards(allocator, io, &output, &expected, .default);

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

test "an aborted loader leaves each output as it was or whole, and stays failed" {
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
    // One worker over many specs, so the abort lands among them.
    var specs: [64]LoadSpec = undefined;
    var outputs: [64]Buffer = undefined;
    for (&specs, &outputs) |*spec, *output| {
        output.* = .{ ._platform = devices.platform, ._shape = shape, ._shards = .empty };
        spec.* = .{ .shape = shape, .output = output, .parts = &parts };
    }
    defer for (&outputs) |*output| output.deinit();

    const loader = try Loader.create(allocator, io, devices.platform, .{ .read_parallelism = 1 });
    defer loader.destroy();
    const batch = try loader.submit(&specs, .none);
    loader.abort();
    try std.testing.expectError(error.Canceled, loader.awaitBatch(batch));
    for (&outputs) |*output| {
        if (output._shards.len != 0) try testing.expectShards(allocator, io, output, &bytes, .default);
    }
    if (loader.submit(&specs, .none)) |_|
        return error.TestUnexpectedResult
    else |err|
        try std.testing.expectEqual(error.Canceled, err);
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
    // Tensors of 2 to 32 rows of 8 u16.
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
        const loader = try Loader.create(allocator, io, devices.platform, .{ .read_parallelism = 4 });
        defer loader.destroy();
        // Both batches are in flight before either is awaited, each with
        // its own workers.
        const first = try loader.submit(specs[0 .. count / 2], .none);
        const second = try loader.submit(specs[count / 2 ..], .none);
        try loader.awaitBatch(first);
        try loader.awaitBatch(second);
        defer for (&outputs) |*output| output.deinit();
        for (&outputs, entries) |*output, entry| try testing.expectShards(allocator, io, output, entry.bytes, .default);
    }
}
