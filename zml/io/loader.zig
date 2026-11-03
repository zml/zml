//! Checkpoint loading. `Loader.start` takes a whole model and publishes
//! every tensor it fills as one submission; `wait` retires it. Backends
//! only read sources into buffers; lookup and the checks stay in this
//! shared front end. The contract is in the pub doc comments below.
//!
//! From bindings to specs. `start` checks every tensor once, before
//! anything is published (`prepareSpec`, `checkPlacement`). It lowers each
//! bound source to bytes (`Source.Placed.lower`, the one rule), one part
//! per piece: a part of the destination's logical bytes (the row-major
//! bytes of its shape, before the sharding splits them), a run of `width`
//! bytes repeated over strided dimensions. A whole checkpoint tensor is one
//! part. Each `LoadSpec` is one destination, which the backends trust as
//! `backend.LoadSpec` states. Only `Source` makes bindings and its pieces
//! are disjoint by construction: nothing here checks that (Source's tests
//! do).
const std = @import("std");

const Buffer = @import("../buffer.zig").Buffer;
const mem = @import("../mem.zig");
const Bufferized = mem.Bufferized;
const meta = @import("../meta.zig");
const platform_mod = @import("../platform.zig");
const Memory = TensorStore.Memory;
const Platform = platform_mod.Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const testing = @import("../testing.zig");
const backend = @import("backend.zig");
const Backend = backend.Backend;
const LoadSpec = backend.LoadSpec;
const load_log = @import("log.zig").load;
const TensorStore = @import("TensorStore.zig");

const PrepareError = std.mem.Allocator.Error || error{InvalidBinding};

/// Loads models from their checkpoint sources.
///
/// What it borrows. The platform outlives the loader and every buffer it
/// loads. `io` outlives the loader: the backend's tasks run on it and open
/// the source files through it. `start` borrows the store, the buffers and
/// the progress node until `wait` or `cancel` returns, and reads the model
/// during the call only. The loaded buffers are the caller's and outlive
/// the loader.
///
/// Start and wait serially on the owning task; source reads and transfers
/// run concurrently inside the backend. Submissions are retired in publish
/// order.
///
/// What fails. Every error of `start` and `wait` fails the loader:
/// everything started is canceled (`cancel`), and every later call returns
/// the first error. Each is logged once, at err, where it is detected, but
/// for a plain allocation failure (`OutOfMemory` from an allocator) and
/// `Canceled`, the caller's own `cancel`, which are returned unlogged. A
/// model whose bindings are wrong is `InvalidBinding`, logged with the
/// tensor's field path, the key its source reads (none for a source that
/// reads nothing or a run inside a byte) and the rule it breaks; buffers
/// shaped unlike the model (`mem.pairBufferized`) too, logged with the
/// field path where they differ. The other errors a caller can act on: a
/// missing checkpoint file is `FileNotFound`; a denied one `AccessDenied`
/// locally and `PermissionDenied` remotely; a file that ends inside a
/// tensor `UnexpectedEndOfFile`, because it changed after its header was
/// read; host memory the backend cannot allocate `OutOfMemory`; device
/// memory that runs out PJRT's `ResourceExhausted`; `cancel` `Canceled`.
/// The rest (PJRT, a remote that failed) passes through as its source
/// names it.
pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    backend: Backend,
    /// Submissions published and not yet retired, oldest first.
    pending: std.Deque(PendingSubmission) = .empty,
    /// The first error; every later call returns it.
    failure: ?anyerror = null,
    stats: Stats = .{},

    pub const Options = struct {
        /// Concurrent source reads, 1 to `max_read_parallelism`, fixed for
        /// the load. Local checkpoints read fastest at 12 to 16 on every
        /// host measured.
        read_parallelism: usize = 16,
    };

    /// Concurrent source reads at most (`Options.read_parallelism`).
    pub const max_read_parallelism: usize = 128;

    /// Checks `opts` and creates the backend. A read width out of range is
    /// `InvalidOptions`, logged.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) !Loader {
        const backend_opts = try backendOptions(opts);
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .backend = try .init(allocator, io, platform, backend_opts),
        };
    }

    /// Gives up on everything started: aborts the backend (queued reads are
    /// dropped, reads in flight finish) and retires the pending
    /// submissions. Each buffer of the started loads is then empty or holds
    /// its shards, whose contents are unspecified (unwritten, errored, or
    /// partly written); read none, and free the tree as after a load
    /// (`mem.deinitBufferized`). The loader is failed afterwards. For an
    /// error between `start` and `wait`; nothing to do when nothing was
    /// started.
    pub fn cancel(self: *Loader) void {
        if (self.loadsStarted()) self.fail(error.Canceled);
    }

    /// Destroys the backend: its workers and its host staging. The buffers
    /// it loaded stay the caller's. Panics on loads neither waited for nor
    /// canceled.
    pub fn deinit(self: *Loader) void {
        if (self.loadsStarted()) std.debug.panic("Loader.deinit with loads started, whose store and buffers may be freed already: add `errdefer loader.cancel();` after `start`", .{});
        self.pending.deinit(self.allocator);
        self.backend.destroy();
        self.* = undefined;
    }

    /// Whether loads are started and neither waited for nor canceled.
    fn loadsStarted(self: *const Loader) bool {
        return self.pending.len != 0;
    }

    /// Starts loading every tensor of `model` into its buffer in `buffers`,
    /// as `store` binds it, and returns; `wait` returns once it is done.
    /// Each tensor is placed by its shape's sharding (replicated without a
    /// mesh), into the memory its binding names, assembled from the pieces
    /// of its source (`Source.Placed.bind`) with every byte no piece covers
    /// set to zero. All of these are published here, as one submission.
    ///
    /// Every tensor is checked once, before anything is published: a tensor
    /// `store` did not bind, a tensor a program computes
    /// (`Source.bindProgram`, which this loader does not run), a source
    /// that reads no bytes (an empty checkpoint tensor among them), a source
    /// that this loader cannot assemble (a run that starts or ends inside a
    /// byte, or strided runs: a transpose, a broadcast, an interleave or
    /// interior padding), a sub-byte last dimension that does not pack into
    /// whole bytes, a placement its mesh does not divide or on devices the
    /// platform lacks, and a memory kind the platform cannot take are
    /// `error.InvalidBinding`.
    ///
    /// `buffers` is `mem.bufferize` of `model`, whose buffers are empty. A
    /// tree of another structure (other union arms, optionals or slice
    /// lengths: `mem.pairBufferized`) is `InvalidBinding`, logged with the
    /// field path, before any buffer is written. `start` empties them
    /// again, and each stays empty until its load writes it whole. So
    /// after any error each buffer is empty or holds shards for the caller
    /// to free: `mem.deinitBufferized` frees the tree either way. `buffers`
    /// and `store` are borrowed until `wait` or `cancel` returns; do not end
    /// `progress` before then. `start` adds one item to `progress` for each
    /// tensor, a composite one once; the backend completes them.
    /// Several loads may be started before one `wait`, from one store or
    /// several.
    pub fn start(
        self: *Loader,
        model: anytype,
        buffers: *Bufferized(@TypeOf(model.*)),
        store: *const TensorStore,
        progress: std.Progress.Node,
    ) !void {
        if (self.failure) |err| return err;
        self.startModel(model, buffers, store, progress) catch |err| {
            self.fail(err);
            return err;
        };
    }

    fn startModel(
        self: *Loader,
        model: anytype,
        buffers: *Bufferized(@TypeOf(model.*)),
        store: *const TensorStore,
        progress: std.Progress.Node,
    ) !void {
        self.stats.begin(self.io);
        var prepared = try prepareModel(self.allocator, self.platform, store, model, buffers);
        errdefer prepared.arena.deinit();
        progress.increaseEstimatedTotalItems(prepared.specs.len);
        if (prepared.specs.len == 0) {
            prepared.arena.deinit();
            return;
        }
        try self.submit(prepared.specs, prepared.arena, progress);
    }

    /// Retires every submission in publish order. Returns once every
    /// started load is in its buffers, and logs one info line: the bytes
    /// loaded, the time since the first `start` and their rate. Idempotent:
    /// with nothing started it returns at once and logs nothing.
    pub fn wait(self: *Loader) !void {
        if (self.failure) |err| return err;
        self.drain() catch |err| {
            self.fail(err);
            return err;
        };
    }

    fn drain(self: *Loader) !void {
        while (self.pending.len != 0) try self.retireOldest();
        self.stats.end(self.io);
    }

    /// Records `err` unless the loader failed already, and gives up on
    /// everything started (`cancel`). The one writer of `failure`.
    fn fail(self: *Loader, err: anyerror) void {
        if (self.failure == null) self.failure = err;
        // Nothing still pending can succeed.
        if (self.pending.len != 0) self.backend.abort();
        while (self.pending.len != 0) self.retireOldest() catch {};
    }

    /// One submission over `specs`, which live in `arena`. Takes `arena`
    /// once the submission is published; on failure the caller still owns
    /// it.
    fn submit(
        self: *Loader,
        specs: []const LoadSpec,
        arena: std.heap.ArenaAllocator,
        progress: std.Progress.Node,
    ) !void {
        // One spec per destination, however many parts fill it: its bytes
        // count once.
        var logical_bytes: usize = 0;
        for (specs) |spec| logical_bytes += spec.shape.byteSize();
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const submission = try self.backend.submit(specs, progress);
        self.pending.pushBackAssumeCapacity(.{
            .submission = submission,
            .arena = arena,
            .logical_bytes = logical_bytes,
        });
        self.stats.submissions += 1;
    }

    /// Retires the oldest pending submission: waits for its reads and
    /// transfers, and counts its bytes once they landed.
    fn retireOldest(self: *Loader) !void {
        const oldest = self.pending.popFront().?;
        defer oldest.arena.deinit();
        try self.backend.await(oldest.submission);
        self.stats.bytes_loaded += oldest.logical_bytes;
    }
};

/// The loader's counters: read only by logs and tests; nothing here
/// decides.
const Stats = struct {
    /// When the first `start` since the last `wait` ran: `wait` logs the
    /// load's time from there.
    started_at: ?std.Io.Timestamp = null,
    /// Logical bytes of the submissions retired since that `start`: each
    /// destination's shape once, so a composite tensor counts its padding
    /// and not its sources one by one.
    bytes_loaded: usize = 0,
    /// Submissions published since `init`.
    submissions: usize = 0,

    /// Times and counts a load from the first `start` since the last
    /// `wait`; later starts add to it.
    fn begin(self: *Stats, io: std.Io) void {
        if (self.started_at != null) return;
        self.started_at = .now(io, .awake);
        self.bytes_loaded = 0;
    }

    /// Logs `wait`'s info line, unless nothing was started, and ends the
    /// load's timing.
    fn end(self: *Stats, io: std.Io) void {
        const started_at = self.started_at orelse return;
        const took = started_at.untilNow(io, .awake);
        const seconds = @as(f64, @floatFromInt(@max(took.nanoseconds, 1))) / std.time.ns_per_s;
        const rate: u64 = @intFromFloat(@as(f64, @floatFromInt(self.bytes_loaded)) / seconds);
        load_log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ self.bytes_loaded, took, rate });
        self.started_at = null;
    }
};

/// One published submission: what to wait for and what its accounting
/// owes.
const PendingSubmission = struct {
    submission: backend.Submission,
    /// The specs, which the backend reads until the submission is awaited
    /// (`backend.LoadSpec`).
    arena: std.heap.ArenaAllocator,
    logical_bytes: usize,
};

/// A model as `start` takes it: one spec per tensor, in `arena` with the
/// specs' parts and the tensors' field paths.
const PreparedModel = struct {
    arena: std.heap.ArenaAllocator,
    specs: []const LoadSpec,
};

/// Pairs each tensor of `model` with its buffer and binding: each tensor
/// takes one spec (`prepareSpec`). A tensor a program computes is
/// `InvalidBinding`: this loader runs no programs.
fn prepareModel(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    model: anytype,
    buffers: *Bufferized(@TypeOf(model.*)),
) PrepareError!PreparedModel {
    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();
    // Tensors and buffers pair by structure, before any buffer is written: a
    // buffer tree shaped otherwise is refused whole.
    var report_buffer: [256]u8 = undefined;
    var report: std.Io.Writer = .fixed(&report_buffer);
    const pairs = mem.pairBufferized(arena.allocator(), model, buffers, &report) catch |err| switch (err) {
        error.Mismatch => {
            load_log.err("buffers shaped unlike the model at model{s}; bufferize them from this model", .{report.buffered()});
            return error.InvalidBinding;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    // Empty until the backend writes it whole, so the caller can free the
    // tree whatever fails.
    for (pairs) |pair| pair.buffer.* = .{ ._platform = platform, ._shape = pair.tensor.shape(), ._shards = .empty };

    var specs: std.ArrayList(LoadSpec) = try .initCapacity(arena.allocator(), pairs.len);
    for (pairs) |pair| {
        const tensor = pair.tensor;
        const target: Target = .{ .path = pair.path, .shape = tensor.shape() };
        const binding = store.bindingOf(tensor.id) orelse {
            load_log.err("{f} has no checkpoint binding", .{target});
            return error.InvalidBinding;
        };
        switch (binding) {
            .load => |load| specs.appendAssumeCapacity(try prepareSpec(arena.allocator(), platform, target, load.source, load.memory, pair.buffer)),
            .compute => |compute| {
                load_log.err("{f} is computed by program {s}, which this loader does not run", .{ target, compute.program.name() });
                return error.InvalidBinding;
            },
        }
    }
    return .{ .arena = arena, .specs = specs.items };
}

/// What a spec fills, as the logs name it: the model's tensor at `path`
/// (`mem.Pair.path`) of `shape`.
const Target = struct {
    path: []const u8,
    shape: Shape,

    pub fn format(self: Target, writer: *std.Io.Writer) !void {
        try writer.print("model{s} {f}", .{ self.path, self.shape });
    }
};

/// The spec filling `output`, the buffer of `target`, from `source` into
/// `memory`, its parts in `arena`: the source lowered to bytes
/// (`Source.Placed.lower`), in destination order. The one check of what
/// the backends trust (`backend.LoadSpec`); a rule it breaks is
/// `InvalidBinding`, logged with the target, and with the key its source
/// reads once the source is lowered.
fn prepareSpec(
    arena: std.mem.Allocator,
    platform: *const Platform,
    target: Target,
    source: TensorStore.Source.Placed,
    memory: Memory,
    output: *Buffer,
) PrepareError!LoadSpec {
    const shape = target.shape;
    const parts = source.lower(arena) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.SubBytePosition => {
            load_log.err("{f}: its source {f} reads a run that starts or ends inside a byte, which the loader cannot place", .{ target, source.shape });
            return error.InvalidBinding;
        },
    };
    if (parts.len == 0) {
        load_log.err("{f} reads no checkpoint bytes", .{target});
        return error.InvalidBinding;
    }
    const key = parts[0].source.name;
    try checkPlacement(platform, target, key);
    if (!backend.supportsMemory(platform, memory)) {
        load_log.err("{f} from {s} is bound to {t} memory, which {t} cannot take", .{ target, key, memory, platform.target });
        return error.InvalidBinding;
    }
    for (parts) |part| {
        if (part.lattice.dims.len != 0) {
            load_log.err("{f} from {s}: a part of {d} strided runs of {d} bytes (a transpose, a broadcast, an interleave or interior padding), which this loader does not assemble", .{ target, part.source.name, part.lattice.runs(), part.lattice.width });
            return error.InvalidBinding;
        }
    }
    if (!backend.checkParts(parts, shape.byteSize())) {
        load_log.err("{f} from {s}: its parts leave their sources or its {d} bytes, pass the largest file offset, or come out of order", .{ target, key, shape.byteSize() });
        return error.InvalidBinding;
    }
    return .{ .shape = shape, .memory = memory, .output = output, .parts = parts };
}

/// Checks that `target`, from `from` (its source's key), places on
/// `platform`'s devices, in any memory space: a sub-byte last dimension
/// packs into whole bytes (`Shape.packedShape`), its mesh partitions divide
/// its dimensions, and every device of its mesh is one of the platform's.
fn checkPlacement(platform: *const Platform, target: Target, from: []const u8) error{InvalidBinding}!void {
    const shape = target.shape;
    const bits = shape.dtype().bitSizeOf();
    if (bits < 8 and (shape.rank() == 0 or @rem(shape.dim(-1), 8 / bits) != 0)) {
        load_log.err("{f} from {s}: its last dimension does not pack into whole bytes", .{ target, from });
        return error.InvalidBinding;
    }
    const placed = platform.placedShape(shape).packedShape();
    const placement = Sharding.Placement.init(&placed) catch {
        load_log.err("{f} from {s}: its dimensions are not divisible by its mesh partitions", .{ target, from });
        return error.InvalidBinding;
    };
    for (placement.mesh().?.devicesInCanonicalOrder()) |device| {
        if (device.id >= platform.devices.len) {
            load_log.err("{f} from {s} is placed on device {d}, but {t} has {d}", .{ target, from, device.id, platform.target, platform.devices.len });
            return error.InvalidBinding;
        }
    }
}

/// `opts` as the backends take them, with the local read hints. Every
/// option is checked here, once.
fn backendOptions(opts: Loader.Options) error{InvalidOptions}!backend.Options {
    if (opts.read_parallelism == 0 or opts.read_parallelism > Loader.max_read_parallelism) {
        load_log.err("invalid loader options: read_parallelism={d}, expected 1..{d}: InvalidOptions", .{ opts.read_parallelism, Loader.max_read_parallelism });
        return error.InvalidOptions;
    }
    return .{ .read_parallelism = opts.read_parallelism };
}

/// A four-byte tensor `value`, a four-byte `second`, a `missing` entry in a
/// file that does not exist, an `empty` tensor and a CPU platform. Pinned
/// after `init`.
const LoaderTestFixture = struct {
    const contents = [_]u8{ 1, 2, 3, 4 };
    const second_contents = [_]u8{ 5, 6, 7, 8 };

    tmp: std.testing.TmpDir,
    store: TensorStore,
    platform: *const Platform,
    value: Tensor,
    second: Tensor,
    missing: Tensor,
    empty: Tensor,

    fn init(self: *LoaderTestFixture, allocator: std.mem.Allocator, io: std.Io) !void {
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        const file = try self.tmp.dir.createFile(io, "weights.bin", .{ .read = true });
        try file.writePositionalAll(io, &(contents ++ second_contents), 0);
        var path_buffer: [1024]u8 = undefined;
        const path_len = try file.realPath(io, &path_buffer);
        file.close(io);
        const path = path_buffer[0..path_len];
        var missing_buffer: [1024]u8 = undefined;
        const missing_path = try std.fmt.bufPrint(&missing_buffer, "{s}.missing", .{path});

        var registry: safetensors.TensorRegistry = .init(allocator);
        defer registry.deinit();
        try registry.registerTensor(.{
            .file_uri = path,
            .name = "value",
            .shape = .init(.{contents.len}, .u8),
            .offset = 0,
        });
        try registry.registerTensor(.{
            .file_uri = path,
            .name = "second",
            .shape = .init(.{second_contents.len}, .u8),
            .offset = contents.len,
        });
        try registry.registerTensor(.{
            .file_uri = missing_path,
            .name = "missing",
            .shape = .init(.{contents.len}, .u8),
            .offset = 0,
        });
        try registry.registerTensor(.{
            .file_uri = path,
            .name = "empty",
            .shape = .init(.{0}, .u8),
            .offset = 0,
        });
        self.platform = testing.env();
        self.store = .init(allocator, &registry, .{ .meshes = &.{self.platform.replicated_mesh} });
        self.value = self.store.view().createReplicatedTensor("value", null);
        self.second = self.store.view().createReplicatedTensor("second", null);
        self.missing = self.store.view().createReplicatedTensor("missing", null);
        self.empty = self.store.view().createReplicatedTensor("empty", null);
    }

    fn deinit(self: *LoaderTestFixture) void {
        self.store.deinit();
        self.tmp.cleanup();
    }

    /// A tensor of `like`'s shape that `program` computes from the
    /// checkpoint tensor `name`.
    fn computed(self: *LoaderTestFixture, name: []const u8, like: Tensor, program: anytype) Tensor {
        const input = self.store.view().source(name, null).withPartitioning(.replicated, .{});
        return TensorStore.Source.bindProgram(&.{input}, like.shape(), program);
    }

    /// The program returning its one input.
    const Identity = struct {
        pub fn forward(_: Identity, _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0];
        }
    };

    /// Whether `loader` failed with `expected` and holds nothing started.
    fn expectFailed(loader: *const Loader, expected: anyerror) !void {
        try std.testing.expectEqual(expected, loader.failure.?);
        try std.testing.expectEqual(0, loader.pending.len);
    }
};

test "start publishes the model at once and wait retires it" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    const Model = struct { value: Tensor, second: Tensor };
    const model: Model = .{ .value = fixture.value, .second = fixture.second };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);

    try loader.start(&model, &buffers, &fixture.store, .none);
    errdefer loader.cancel();
    try std.testing.expectEqual(1, loader.pending.len);
    try loader.wait();
    try std.testing.expectEqual(0, loader.pending.len);
    try testing.expectShards(allocator, io, &buffers.value, &LoaderTestFixture.contents, .default);
    try testing.expectShards(allocator, io, &buffers.second, &LoaderTestFixture.second_contents, .default);
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, loader.stats.bytes_loaded);

    // Idempotent: a second wait neither retires nor recounts.
    try loader.wait();
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, loader.stats.bytes_loaded);

    // The next load counts and times from its own first start.
    try std.testing.expectEqual(null, loader.stats.started_at);
    var again: Buffer = undefined;
    try loader.start(&fixture.second, &again, &fixture.store, .none);
    try loader.wait();
    defer again.deinit();
    try std.testing.expectEqual(LoaderTestFixture.second_contents.len, loader.stats.bytes_loaded);
    try std.testing.expectEqual(null, loader.stats.started_at);
}

test "cancel gives up on everything started" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    const Model = struct { value: Tensor, second: Tensor };
    const model: Model = .{ .value = fixture.value, .second = fixture.second };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    loader.cancel();
    try LoaderTestFixture.expectFailed(&loader, error.Canceled);
    // The loader is failed: what it waits for and what it is given.
    try std.testing.expectError(error.Canceled, loader.wait());
    try std.testing.expectError(error.Canceled, loader.start(&model, &buffers, &fixture.store, .none));
    try std.testing.expectEqual(@as(usize, 0), loader.stats.bytes_loaded);
}

test "a read failure fails the loader for every later call" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    const Model = struct { broken: Tensor, value: Tensor };
    const model: Model = .{ .broken = fixture.missing, .value = fixture.value };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    try std.testing.expectError(error.FileNotFound, loader.wait());
    try LoaderTestFixture.expectFailed(&loader, error.FileNotFound);
    try std.testing.expectError(error.FileNotFound, loader.start(&model, &buffers, &fixture.store, .none));
    try std.testing.expectError(error.FileNotFound, loader.wait());
    try std.testing.expectEqual(@as(usize, 0), loader.stats.bytes_loaded);
}

test "after a failed load every buffer is empty or whole, and deinitBufferized frees the tree" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    // A tree with a slice and optionals, some of it in the file that loads,
    // and a tensor in the file that does not exist.
    const Layer = struct { weight: Tensor, bias: ?Tensor };
    const Model = struct { layers: []Layer, broken: Tensor };
    var layers = [_]Layer{
        .{ .weight = fixture.value, .bias = fixture.second },
        .{ .weight = fixture.second, .bias = null },
    };
    const model: Model = .{ .layers = &layers, .broken = fixture.missing };
    var buffers = try mem.bufferize(allocator, &model);
    // Frees whatever the failed load wrote, under the testing allocator.
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    try std.testing.expectError(error.FileNotFound, loader.wait());
    try LoaderTestFixture.expectFailed(&loader, error.FileNotFound);
    // Each buffer holds no shard or one per device.
    const devices = fixture.platform.placedShape(fixture.value.shape())._sharding.mesh.?.devicesInCanonicalOrder().len;
    var torn: usize = 0;
    meta.forEachVisit(&buffers, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer, shards: usize, count: *usize) void {
            if (buffer._shards.len != 0 and buffer._shards.len != shards) count.* += 1;
        }
    }.call, .{ devices, &torn });
    try std.testing.expectEqual(0, torn);
}

test "init checks every option and returns InvalidOptions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    for ([_]usize{ 0, Loader.max_read_parallelism + 1 }) |width| {
        if (Loader.init(allocator, io, platform, .{ .read_parallelism = width })) |value| {
            var loader = value;
            loader.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.InvalidOptions, err);
    }
}

test "a rejected binding fails the loader" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = fixture.platform;
    // A mesh of devices the platform does not have: the same tensors placed
    // on it, in device and host memory.
    var foreign_physical = platform.replicated_mesh.physical.*;
    const foreign_devices = try allocator.dupe(Sharding.Device, foreign_physical.devices_in_canonical_order);
    defer allocator.free(foreign_devices);
    for (foreign_devices) |*device| device.id += 100;
    foreign_physical.devices_in_canonical_order = foreign_devices;
    var foreign_mesh = platform.replicated_mesh.*;
    foreign_mesh.physical = &foreign_physical;
    const pinned = fixture.store.view().createReplicatedTensor("second", null);
    fixture.store.view().setMemory(pinned, .host_pinned);
    var foreign = [_]Tensor{ fixture.value, pinned };
    for (&foreign) |*tensor| tensor._shape._sharding.mesh = &foreign_mesh;

    // No binding, an empty source, a mesh of other devices, a host kind the
    // platform cannot take, a broadcast's strided runs and a program.
    const Case = struct { tensor: Tensor, skip: bool = false };
    const cases = [_]Case{
        .{ .tensor = Tensor.fromShape(fixture.value.shape()) },
        .{ .tensor = fixture.empty },
        .{ .tensor = foreign[0] },
        .{ .tensor = foreign[1] },
        .{ .tensor = pinned, .skip = backend.supportsMemory(platform, .host_pinned) },
        .{ .tensor = fixture.store.view().source("value", .{.x}).broad(Shape.init(.{ .y = 2, .x = 4 }, .u8)).withPartitioning(.replicated, .{}).bind() },
        .{ .tensor = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}) },
    };
    for (cases) |case| {
        if (case.skip) continue;
        var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
        defer loader.deinit();
        var output: Buffer = undefined;
        try std.testing.expectError(error.InvalidBinding, loader.start(&case.tensor, &output, &fixture.store, .none));
        try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
        // `start` emptied the buffer before it failed.
        try std.testing.expectEqual(0, output._shards.len);
        output.deinit();
        try std.testing.expectError(error.InvalidBinding, loader.start(&fixture.value, &output, &fixture.store, .none));
    }
}

test "buffers of another structure fail start before any buffer is written" {
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(std.testing.allocator, std.testing.io);
    defer fixture.deinit();
    const value = fixture.value;
    const second = fixture.second;

    // Each pair holds as many tensors on both sides, in another structure,
    // and a pair precedes each mismatch.
    const Optionals = struct { head: Tensor, a: ?Tensor, b: ?Tensor };
    try expectStructureMismatch(&fixture, Optionals, .{ .head = value, .a = value, .b = null }, .{ .head = value, .a = null, .b = value });
    const Arms = struct { head: Tensor, arm: union(enum) { a: Tensor, b: Tensor }, last: Tensor };
    try expectStructureMismatch(&fixture, Arms, .{ .head = value, .arm = .{ .a = value }, .last = second }, .{ .head = value, .arm = .{ .b = value }, .last = second });
    const Slices = struct { head: Tensor, first: []Tensor, second: []Tensor };
    var two = [_]Tensor{ value, second };
    var one = [_]Tensor{value};
    var other_one = [_]Tensor{second};
    var other_two = [_]Tensor{ second, value };
    try expectStructureMismatch(&fixture, Slices, .{ .head = value, .first = &two, .second = &one }, .{ .head = value, .first = &other_one, .second = &other_two });
}

/// Starts `model` into the buffers bufferized from `built`, which has as
/// many tensors in another structure: `InvalidBinding`, and every buffer as
/// it was.
fn expectStructureMismatch(fixture: *LoaderTestFixture, comptime Model: type, built: Model, model: Model) !void {
    const allocator = std.testing.allocator;
    var loader = try Loader.init(allocator, std.testing.io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var buffers = try mem.bufferize(allocator, &built);
    defer mem.deinitBufferized(allocator, &buffers);
    // No tensor has this shape, which `start` would replace with its
    // tensor's when it empties a buffer.
    const untouched: Shape = .init(.{ 1, 2, 3 }, .u8);
    meta.forEachVisit(&buffers, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer, shape: Shape) void {
            buffer._shape = shape;
        }
    }.call, .{untouched});
    try std.testing.expectError(error.InvalidBinding, loader.start(&model, &buffers, &fixture.store, .none));
    try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
    var written: usize = 0;
    meta.forEachVisit(&buffers, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer, shape: Shape, count: *usize) void {
            if (!buffer._shape.eql(shape) or buffer._shards.len != 0) count.* += 1;
        }
    }.call, .{ untouched, &written });
    try std.testing.expectEqual(0, written);
}
