//! Checkpoint loading. `Loader.start` takes a whole model: it starts
//! compiling the programs that compute some of its tensors
//! (`Source.bindProgram`) and publishes every tensor the loader fills
//! itself. `wait` then publishes the programs' inputs in groups, runs each
//! group's programs on the caller's task once its inputs landed, and
//! retires everything. Backends only read sources into buffers; lookup,
//! compilation, admission and execution stay in this shared front end. The
//! contract is in the pub doc comments below.
//!
//! From bindings to specs. `start` checks every tensor and program input
//! once, before anything is published (`prepareSpec`, `checkPlacement`).
//! It lowers each bound source to bytes (`Source.Placed.lower`, the one
//! rule), one part per piece: a part of the destination's logical bytes
//! (the row-major bytes of its shape, before the sharding splits them), a
//! run of `width` bytes repeated over strided dimensions. A whole
//! checkpoint tensor is one part. Each `LoadSpec` is one destination, which
//! the backends trust as `backend.LoadSpec` states. Only `Source` makes
//! bindings and its pieces are disjoint by construction: nothing here
//! checks that (Source's tests do).
//!
//! Order and admission. The tensors the loader fills go out first, as one
//! submission. A group holds the tensors of consecutive programs in model
//! order, up to `program_group_bytes` of outputs, so the inputs of one
//! layer's experts, adjacent in the checkpoint, are read together. Before
//! publishing a group, `wait` waits for its programs to compile, then
//! retires the oldest pending submissions until no program group is
//! pending: one program group is in flight at a time.
const std = @import("std");

const Buffer = @import("../buffer.zig").Buffer;
const Compiler = @import("../Compiler.zig");
const Exe = @import("../exe.zig").Exe;
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

/// Loads models from their checkpoint sources. A value: `init` returns at
/// once.
///
/// What it borrows. The platform outlives the loader and every buffer it
/// loads. `io` outlives the loader: the backend's tasks run on it and open
/// the source files through it. `start` borrows the store, the buffers and
/// the progress node until `wait` or `cancel` returns, and reads the model
/// during the call only. The loaded buffers are the caller's and outlive
/// the loader.
///
/// Start and wait serially on the owning task; source reads and transfers
/// run concurrently inside the backend, compiles on tasks of their own, or
/// inline when `io` cannot run one concurrently. Submissions are retired in
/// publish order.
///
/// What fails. Every error of `start` and `wait` fails the loader:
/// everything started is canceled (`cancel`), and every later call returns
/// the first error. Each is logged once, at err, where it is detected, but
/// for a plain allocation failure (`OutOfMemory` from an allocator) and
/// `Canceled`, the caller's own `cancel`, which are returned unlogged. A
/// model whose bindings are wrong is `InvalidBinding`, logged with the
/// tensor's field path, the key its source reads (none for a source that
/// reads nothing or a run inside a byte) and the rule it breaks, or for a
/// program whose output differs, the program's name and the shape it
/// computes; buffers shaped unlike the model (`mem.pairBufferized`) too,
/// logged with the field path where they differ. The other errors a caller
/// can act on: a missing checkpoint file is `FileNotFound`; a denied one
/// `AccessDenied` locally and `PermissionDenied` remotely; a file that ends
/// inside a tensor `UnexpectedEndOfFile`, because it changed after its
/// header was read; host memory the backend cannot allocate `OutOfMemory`;
/// device memory that runs out PJRT's `ResourceExhausted`; `cancel`
/// `Canceled`. The rest (PJRT, the compiler, a remote that failed) passes
/// through as its source names it.
pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    /// `init`'s task, which warms the device allocators and creates the
    /// backend, until the first `start` awaits it.
    setup: ?std.Io.Future(anyerror!Backend),
    /// What `setup` created; null before the first `start`, and after a
    /// failed set-up.
    backend: ?Backend = null,
    /// `Options.program_group_bytes`.
    program_group_bytes: usize,
    /// The programs of the loads started since the last `wait`, each
    /// compiled once.
    programs: std.ArrayList(*CompiledProgram) = .empty,
    /// Program groups started and not yet published, oldest first.
    queued: std.Deque(Group) = .empty,
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
        /// Output bytes of the program tensors one group holds at most, one
        /// tensor at least. A group's inputs are read together and freed
        /// once its programs ran.
        program_group_bytes: usize = 8 << 30,
    };

    /// Concurrent source reads at most (`Options.read_parallelism`).
    pub const max_read_parallelism: usize = 128;

    /// Checks `opts` and returns at once. Warming the device allocators and
    /// creating the backend run on a task of the loader's own, or here when
    /// `io` has no concurrency; the first `start` waits for that task and
    /// returns its error. A read width out of range is `InvalidOptions`,
    /// logged.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) error{InvalidOptions}!Loader {
        const backend_opts = try backendOptions(opts);
        const setup_args = .{ allocator, io, platform, backend_opts };
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .setup = io.concurrent(setUp, setup_args) catch io.async(setUp, setup_args),
            .program_group_bytes = opts.program_group_bytes,
        };
    }

    /// `init`'s task: the device allocators warmed, then the backend.
    fn setUp(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, opts: backend.Options) anyerror!Backend {
        platform.warmupDeviceAllocators(io) catch |err| {
            load_log.failure(err, "warm the device allocators", .{});
            return err;
        };
        return Backend.init(allocator, io, platform, opts);
    }

    /// Waits for `init`'s task the first time: its backend, or its error,
    /// for the caller to fail the loader with.
    fn awaitSetup(self: *Loader) !void {
        const setup = if (self.setup) |*task| task else return;
        const result = setup.await(self.io);
        self.setup = null;
        self.backend = try result;
    }

    /// Gives up on everything started: drops the queued groups, aborts the
    /// backend (queued reads are dropped, reads in flight finish), retires
    /// the pending submissions without running their programs, and waits
    /// for the compiles. Each buffer of the started loads is then empty or
    /// holds its shards, whose contents are unspecified (unwritten, errored,
    /// or partly written); read none, and free the tree as after a load
    /// (`mem.deinitBufferized`). The loader is failed afterwards. For an
    /// error between `start` and `wait`; nothing to do when nothing was
    /// started.
    pub fn cancel(self: *Loader) void {
        if (self.loadsStarted()) self.fail(error.Canceled);
    }

    /// Cancels `init`'s task when no `start` awaited it, destroying what it
    /// built, and destroys the backend: its workers and its host staging.
    /// The buffers it loaded stay the caller's. Panics on loads neither
    /// waited for nor canceled.
    pub fn deinit(self: *Loader) void {
        if (self.loadsStarted()) std.debug.panic("Loader.deinit with loads started, whose store and buffers may be freed already: add `errdefer loader.cancel();` after `start`", .{});
        self.queued.deinit(self.allocator);
        self.pending.deinit(self.allocator);
        self.programs.deinit(self.allocator);
        if (self.setup) |*setup| {
            if (setup.cancel(self.io)) |backend_| backend_.destroy() else |_| {}
        }
        if (self.backend) |backend_| backend_.destroy();
        self.* = undefined;
    }

    /// Whether loads are started and neither waited for nor canceled.
    fn loadsStarted(self: *const Loader) bool {
        return self.queued.len != 0 or self.pending.len != 0 or self.programs.items.len != 0;
    }

    /// Starts loading every tensor of `model` into its buffer in `buffers`,
    /// as `store` binds it, and returns; `wait` returns once it is done.
    /// Each tensor is placed by its shape's sharding (replicated without a
    /// mesh), into the memory its binding names, assembled from the pieces
    /// of its source (`Source.Placed.bind`) with every byte no piece covers
    /// set to zero. All of these are published here, as one submission. A
    /// tensor a program computes (`Source.bindProgram`) starts its program
    /// compiling, once for every tensor of an equal program over the same
    /// placed shapes; `wait` lands its inputs and runs it.
    ///
    /// Every tensor and program input is checked once, before anything is
    /// published: a tensor `store` did not bind, a source that reads no
    /// bytes (an empty checkpoint tensor among them), a source that this
    /// loader cannot assemble (a run that starts or ends inside a byte, or
    /// strided runs: a transpose, a broadcast, an interleave or interior
    /// padding), a sub-byte last dimension that does not pack into whole
    /// bytes, a placement its mesh does not divide or on devices the
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
    /// tensor the loader fills, a composite one once, and for each program
    /// input; the backend completes them.
    /// Several loads may be started before one `wait`, from one store or
    /// several. The first `start` waits for `init`'s task, and fails with
    /// its error.
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
        try self.awaitSetup();
        // The load's time starts once the set-up is done.
        self.stats.begin(self.io);
        var prepared = try prepareModel(self.allocator, self.platform, store, model, buffers);
        errdefer prepared.arena.deinit();
        var items = prepared.specs.len;
        for (prepared.computed) |item| items += item.compute.inputs.len;
        progress.increaseEstimatedTotalItems(items);
        // Queued first: their programs compile while the rest is read.
        try self.queueGroups(store, prepared.computed, progress);
        if (prepared.specs.len == 0) {
            prepared.arena.deinit();
            return;
        }
        try self.submit(prepared.specs, prepared.arena, &.{}, progress);
    }

    /// Publishes the queued program groups, oldest first, each once its
    /// programs compiled and no other program group is pending (`admit`),
    /// and retires every submission in publish order, running a group's
    /// programs on this task once its inputs landed. Returns once every
    /// started load is in its buffers, with the freed device memory, its
    /// programs' inputs and temporaries among it, given back to the devices
    /// (`Platform.releaseFreedDeviceMemory`), and logs one info line: the
    /// bytes loaded, the time since the first `start` and their rate, then
    /// what the programs computed and how long publishing waited for their
    /// compiles. A program that fails to compile or run fails it; one that
    /// does not return its tensor's shape, placed alike, is
    /// `InvalidBinding`. Idempotent: with nothing started it returns at
    /// once and logs nothing.
    pub fn wait(self: *Loader) !void {
        if (self.failure) |err| return err;
        const started = self.loadsStarted();
        self.drain() catch |err| {
            self.fail(err);
            return err;
        };
        self.releasePrograms();
        if (started) self.platform.releaseFreedDeviceMemory();
    }

    fn drain(self: *Loader) !void {
        var groups: usize = 0;
        var tensors: usize = 0;
        while (self.queued.popFront()) |group| {
            tensors += group.calls.len;
            try self.publish(group);
            groups += 1;
        }
        while (self.pending.len != 0) try self.retireOldest(true);
        self.stats.end(self.io, tensors, self.programs.items.len, groups);
    }

    /// Records `err` unless the loader failed already, and gives up on
    /// everything started (`cancel`). The one writer of `failure`.
    fn fail(self: *Loader, err: anyerror) void {
        if (self.failure == null) self.failure = err;
        while (self.queued.popFront()) |group| group.deinit(self.allocator);
        // Nothing still pending can succeed.
        if (self.pending.len != 0) self.backend.?.abort();
        while (self.pending.len != 0) self.retireOldest(false) catch {};
        self.releasePrograms();
    }

    /// Queues the tensors of `computed`, in model order, in groups of at
    /// most `program_group_bytes` of outputs, one tensor at least.
    fn queueGroups(self: *Loader, store: *const TensorStore, computed: []const Computed, progress: std.Progress.Node) !void {
        var first: usize = 0;
        while (first < computed.len) {
            var end = first + 1;
            var bytes = computed[first].shape.byteSize();
            while (end < computed.len and bytes + computed[end].shape.byteSize() <= self.program_group_bytes) : (end += 1) {
                bytes += computed[end].shape.byteSize();
            }
            try self.queued.ensureUnusedCapacity(self.allocator, 1);
            self.queued.pushBackAssumeCapacity(try self.prepareGroup(store, computed[first..end], progress));
            first = end;
        }
    }

    /// The group computing `computed`: each tensor's program, the input
    /// shells it runs over and the specs that fill them, one per input,
    /// checked as `start` checks every spec (`prepareSpec`) before the
    /// program's compile starts, which packs their shapes.
    fn prepareGroup(self: *Loader, store: *const TensorStore, computed: []const Computed, progress: std.Progress.Node) PrepareError!Group {
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        errdefer arena.deinit();
        const calls = try self.allocator.alloc(Call, computed.len);
        var prepared: usize = 0;
        errdefer {
            for (calls[0..prepared]) |*call| call.deinit(self.allocator);
            self.allocator.free(calls);
        }
        var specs: std.ArrayList(LoadSpec) = .empty;
        for (computed, calls) |item, *call| {
            const inputs = try self.allocator.alloc(Buffer, item.compute.inputs.len);
            errdefer self.allocator.free(inputs);
            for (item.compute.inputs, inputs, 0..) |source, *input, index| {
                input.* = .{ ._platform = self.platform, ._shape = source.shape, ._shards = .empty };
                const target: Target = .{ .path = item.path, .shape = item.shape, .input = .{ .index = index, .shape = source.shape } };
                const spec = try prepareSpec(arena.allocator(), self.platform, target, source, .default, input);
                try specs.append(arena.allocator(), spec);
            }
            const compiled = try self.compiledProgram(store, item.compute, item.shape);
            call.* = .{ .path = try arena.allocator().dupe(u8, item.path), .program = compiled, .inputs = inputs, .output = item.output };
            prepared += 1;
        }
        return .{ .calls = calls, .specs = specs.items, .arena = arena, .progress = progress };
    }

    /// The compiled program computing a tensor of `output` from `compute`:
    /// one started already for an equal program over the same placed
    /// shapes, else a compile started now.
    fn compiledProgram(self: *Loader, store: *const TensorStore, compute: TensorStore.Binding.Compute, output: Shape) std.mem.Allocator.Error!*CompiledProgram {
        for (self.programs.items) |compiled| {
            if (compiled.compute.sameExecutable(compiled.output, compute, output)) return compiled;
        }
        try self.programs.ensureUnusedCapacity(self.allocator, 1);
        const compiled = try CompiledProgram.start(self.allocator, self.io, self.platform, store.meshes, compute, output);
        self.programs.appendAssumeCapacity(compiled);
        return compiled;
    }

    fn releasePrograms(self: *Loader) void {
        for (self.programs.items) |compiled| compiled.destroy(self.allocator, self.io);
        self.programs.clearRetainingCapacity();
    }

    /// Publishes `group` once its programs compiled and no other program
    /// group is pending: its calls and its specs' arena move to the pending
    /// submission, or are freed when this fails.
    fn publish(self: *Loader, group: Group) !void {
        errdefer group.deinit(self.allocator);
        const started: std.Io.Timestamp = .now(self.io, .awake);
        for (group.calls) |call| try call.program.await(self.io, self.platform);
        self.stats.compile_wait_ns +|= @intCast(started.untilNow(self.io, .awake).toNanoseconds());
        try self.admit();
        try self.submit(group.specs, group.arena, group.calls, group.progress);
    }

    /// Retires pending submissions, oldest first, until no program group is
    /// pending, so one program group is in flight at a time. A bulk load
    /// published before the pending group is retired with it.
    fn admit(self: *Loader) !void {
        var retired: usize = 0;
        while (self.programGroupPending()) : (retired += 1) try self.retireOldest(true);
        self.stats.admission_retires += retired;
    }

    /// Whether a program group is pending: only retiring one frees room
    /// for the next (a bulk load's buffers stay).
    fn programGroupPending(self: *const Loader) bool {
        var it = self.pending.iterator();
        while (it.nextPtr()) |submission| {
            if (submission.calls.len != 0) return true;
        }
        return false;
    }

    /// One submission over `specs`, which live in `arena`. Takes `arena` and
    /// `calls` once the submission is published; on failure the caller still
    /// owns them.
    fn submit(
        self: *Loader,
        specs: []const LoadSpec,
        arena: std.heap.ArenaAllocator,
        calls: []Call,
        progress: std.Progress.Node,
    ) !void {
        // One spec per destination, however many parts fill it: its bytes
        // count once.
        var logical_bytes: usize = 0;
        for (specs) |spec| logical_bytes += spec.shape.byteSize();
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const submission = try self.backend.?.submit(specs, progress);
        self.pending.pushBackAssumeCapacity(.{
            .submission = submission,
            .arena = arena,
            .calls = calls,
            .logical_bytes = logical_bytes,
        });
        self.stats.submissions += 1;
    }

    /// Retires the oldest pending submission: waits for its reads and
    /// transfers, runs its programs when `execute`, frees its inputs either
    /// way and counts its bytes once they ran.
    fn retireOldest(self: *Loader, execute: bool) !void {
        var oldest = self.pending.popFront().?;
        defer self.release(&oldest);
        try self.backend.?.await(oldest.submission);
        if (!execute) return;
        for (oldest.calls) |*call| {
            call.execute(self.allocator, self.io) catch |err| {
                const target: Target = .{ .path = call.path, .shape = call.program.output };
                load_log.failure(err, "run program {s} computing {f}: inputs={d}", .{ call.program.name(), target, call.inputs.len });
                return err;
            };
        }
        self.stats.bytes_loaded += oldest.logical_bytes;
    }

    fn release(self: *Loader, oldest: *PendingSubmission) void {
        oldest.arena.deinit();
        for (oldest.calls) |*call| call.deinit(self.allocator);
        self.allocator.free(oldest.calls);
    }
};

/// The loader's counters: read only by logs and tests; nothing here
/// decides.
const Stats = struct {
    /// When the first `start` since the last `wait` ran: `wait` logs the
    /// load's time from there.
    started_at: ?std.Io.Timestamp = null,
    /// Logical bytes of the submissions retired with execution since that
    /// `start`: each destination's shape once, so a composite tensor counts
    /// its padding and not its sources one by one, and a program group
    /// counts its inputs.
    bytes_loaded: usize = 0,
    /// How long publishing waited for compiles since the last `wait`.
    compile_wait_ns: u64 = 0,
    /// Submissions published since `init`.
    submissions: usize = 0,
    /// Submissions retired to admit a program group, bulk loads included.
    admission_retires: usize = 0,

    /// Times and counts a load from the first `start` since the last
    /// `wait`; later starts add to it.
    fn begin(self: *Stats, io: std.Io) void {
        if (self.started_at != null) return;
        self.started_at = .now(io, .awake);
        self.bytes_loaded = 0;
    }

    /// Logs `wait`'s info line, unless nothing was started, and ends the
    /// load's timing. `programs` programs computed `tensors` tensors in
    /// `groups` groups.
    fn end(self: *Stats, io: std.Io, tensors: usize, programs: usize, groups: usize) void {
        const started_at = self.started_at orelse return;
        const took = started_at.untilNow(io, .awake);
        const seconds = @as(f64, @floatFromInt(@max(took.nanoseconds, 1))) / std.time.ns_per_s;
        const rate: u64 = @intFromFloat(@as(f64, @floatFromInt(self.bytes_loaded)) / seconds);
        if (groups == 0) {
            load_log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s]", .{ self.bytes_loaded, took, rate });
        } else {
            load_log.info("Loaded weights [{Bi:.2}, {f}, {Bi:.2}/s], computed {d} tensors with {d} programs in {d} groups, {d} ms waiting for compiles", .{
                self.bytes_loaded,
                took,
                rate,
                tensors,
                programs,
                groups,
                self.compile_wait_ns / std.time.ns_per_ms,
            });
        }
        self.started_at = null;
        self.compile_wait_ns = 0;
    }
};

/// One published submission: what to wait for, what to run afterwards and
/// what its accounting owes.
const PendingSubmission = struct {
    submission: backend.Submission,
    /// The specs, which the backend reads until the submission is awaited
    /// (`backend.LoadSpec`).
    arena: std.heap.ArenaAllocator,
    /// Run in order once the reads are done; freed with their inputs when
    /// the submission is retired. Empty for a bulk load.
    calls: []Call,
    logical_bytes: usize,
};

/// Program tensors whose inputs `wait` publishes as one submission.
const Group = struct {
    calls: []Call,
    /// One per program input, filling the calls' input shells; they and
    /// their parts live in `arena`.
    specs: []LoadSpec,
    arena: std.heap.ArenaAllocator,
    progress: std.Progress.Node,

    fn deinit(self: Group, allocator: std.mem.Allocator) void {
        for (self.calls) |*call| call.deinit(allocator);
        allocator.free(self.calls);
        self.arena.deinit();
    }
};

/// A program compiling, then compiled.
const CompiledProgram = struct {
    compute: TensorStore.Binding.Compute,
    output: Shape,
    future: std.Io.Future(Compiler.Error!Exe),
    /// Set by the first `await`.
    exe: ?Exe = null,

    fn start(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        meshes: []const *const Sharding.Mesh,
        compute: TensorStore.Binding.Compute,
        output: Shape,
    ) std.mem.Allocator.Error!*CompiledProgram {
        const self = try allocator.create(CompiledProgram);
        // Concurrent, as `io.async` may run the compile right here on a busy
        // threaded io, ahead of the reads `start` publishes after it.
        const args = .{ compute, allocator, io, platform, meshes, output };
        const compile = TensorStore.Binding.Compute.compile;
        self.* = .{
            .compute = compute,
            .output = output,
            .future = io.concurrent(compile, args) catch io.async(compile, args),
        };
        return self;
    }

    /// Waits for the compile the first time, then checks that the
    /// executable computes one tensor of `output`, placed alike
    /// (`checkOutput`).
    fn await(self: *CompiledProgram, io: std.Io, platform: *const Platform) !void {
        if (self.exe != null) return;
        self.exe = self.future.await(io) catch |err| {
            load_log.failure(err, "compile program {s} computing {f}", .{ self.name(), self.output });
            return err;
        };
        try checkOutput(platform, self.name(), self.output, &self.exe.?);
    }

    fn name(self: *const CompiledProgram) []const u8 {
        return self.compute.program.name();
    }

    fn destroy(self: *CompiledProgram, allocator: std.mem.Allocator, io: std.Io) void {
        if (self.exe) |*exe| {
            exe.deinit();
        } else if (self.future.cancel(io)) |exe| {
            exe.deinit();
        } else |_| {}
        allocator.destroy(self);
    }
};

/// One tensor of a program group, a pending `Exe.call`: its program, the
/// input shells the group's submission fills, and the model's buffer it
/// computes.
const Call = struct {
    /// The tensor's field path, in the group's arena: `prepareModel`'s
    /// arena may be freed before the call runs.
    path: []const u8,
    program: *CompiledProgram,
    inputs: []Buffer,
    output: *Buffer,

    /// Runs the program's executable, awaited when the group was
    /// published, over the loaded inputs on the calling task.
    /// `Exe.call` leaves every input handle to the caller, donated ones
    /// included (their contents are gone), so `deinit` frees them all,
    /// exactly once; a runner would destroy the donated handles itself.
    /// A failed execution, an input whose transfer failed included, is
    /// returned and leaves `output` unwritten.
    fn execute(self: *const Call, allocator: std.mem.Allocator, io: std.Io) !void {
        const exe = &self.program.exe.?;
        var args = try exe.args(allocator);
        defer args.deinit(allocator);
        var results = try exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{self.inputs});
        try exe.call(io, args, &results, .{ .wait = true });
        self.output.* = results.get(Buffer);
    }

    /// Frees the inputs, donated or not; a shell the loader never wrote owns
    /// no shards.
    fn deinit(self: *Call, allocator: std.mem.Allocator) void {
        for (self.inputs) |*input| input.deinit();
        allocator.free(self.inputs);
    }
};

/// A tensor a program computes, at `path` in the model, the buffer it
/// computes into, and its program and inputs.
const Computed = struct {
    path: []const u8,
    shape: Shape,
    output: *Buffer,
    compute: TensorStore.Binding.Compute,
};

/// A model as `start` takes it: one spec per tensor the loader fills, and
/// the tensors programs compute, all in `arena` with the specs' parts and
/// the tensors' field paths.
const PreparedModel = struct {
    arena: std.heap.ArenaAllocator,
    specs: []const LoadSpec,
    computed: []const Computed,
};

/// Pairs each tensor of `model` with its buffer and binding: a tensor the
/// loader fills takes one spec (`prepareSpec`), a tensor a program computes
/// is placed like any other (`checkPlacement`) and computed.
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
    // Empty until a backend or a program writes it whole, so the caller can
    // free the tree whatever fails.
    for (pairs) |pair| pair.buffer.* = .{ ._platform = platform, ._shape = pair.tensor.shape(), ._shards = .empty };

    var specs: std.ArrayList(LoadSpec) = try .initCapacity(arena.allocator(), pairs.len);
    var computed: std.ArrayList(Computed) = try .initCapacity(arena.allocator(), pairs.len);
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
                try checkPlacement(platform, target, compute.program.name());
                computed.appendAssumeCapacity(.{ .path = pair.path, .shape = tensor.shape(), .output = pair.buffer, .compute = compute });
            },
        }
    }
    return .{ .arena = arena, .specs = specs.items, .computed = computed.items };
}

/// What a spec fills, as the logs name it: the model's tensor at `path`
/// (`mem.Pair.path`) of `shape`, or `input` of the program computing it.
const Target = struct {
    path: []const u8,
    shape: Shape,
    input: ?Input = null,

    const Input = struct { index: usize, shape: Shape };

    /// The shape the spec fills: the input's, else the tensor's.
    fn destination(self: Target) Shape {
        return if (self.input) |input| input.shape else self.shape;
    }

    pub fn format(self: Target, writer: *std.Io.Writer) !void {
        if (self.input) |input| try writer.print("program input {d} {f} of ", .{ input.index, input.shape });
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
    const shape = target.destination();
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

/// Checks that `target`, from `from` (its source's key or its program),
/// places on `platform`'s devices, in any memory space: a sub-byte last
/// dimension packs into whole bytes (`Shape.packedShape`), its mesh
/// partitions divide its dimensions, and every device of its mesh is one of
/// the platform's.
fn checkPlacement(platform: *const Platform, target: Target, from: []const u8) error{InvalidBinding}!void {
    const shape = target.destination();
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

/// A program must compute one tensor of `expected`, placed on the same
/// devices, each holding the same slice. The comparison is per device, not
/// by mesh pointer and partition, so `.open` and `.replicated`, or a null
/// mesh and the replicated mesh, agree when they place alike. `expected`
/// places on the platform's devices (`checkPlacement`).
fn checkOutput(platform: *const Platform, name: []const u8, expected: Shape, exe: *const Exe) error{InvalidBinding}!void {
    if (exe.output_shapes.len != 1 or !expected.eql(exe.output_shapes[0])) {
        load_log.err("program {s} computing {f} returns {d} outputs, the first {f}", .{ name, expected, exe.output_shapes.len, if (exe.output_shapes.len != 0) exe.output_shapes[0] else expected });
        return error.InvalidBinding;
    }
    const actual = exe.output_shapes[0];
    const want_shape = platform.placedShape(expected).packedShape();
    const want = Sharding.Placement.init(&want_shape) catch |err|
        std.debug.panic("loader: {f} does not place, though checked: {s}", .{ expected, @errorName(err) });
    const got_shape = platform.placedShape(actual).packedShape();
    const got = Sharding.Placement.init(&got_shape) catch |err| {
        load_log.err("program {s}: placement of its output {f}: {s}", .{ name, actual, @errorName(err) });
        return error.InvalidBinding;
    };
    const want_devices = want.mesh().?.devicesInCanonicalOrder();
    const got_devices = got.mesh().?.devicesInCanonicalOrder();
    if (want_devices.len != got_devices.len) {
        load_log.err("program {s} places its output {f} on {d} devices, the tensor {f} on {d}", .{ name, actual, got_devices.len, expected, want_devices.len });
        return error.InvalidBinding;
    }
    for (want_devices, got_devices, 0..) |w, g, device_index| {
        const want_slices = want.slices(device_index, want_shape);
        const got_slices = got.slices(device_index, got_shape);
        const same_slices = want_slices.len == got_slices.len and for (want_slices.constSlice(), got_slices.constSlice()) |a, b| {
            if (a.start != b.start or a.size != b.size) break false;
        } else true;
        if (w.id != g.id or !same_slices) {
            load_log.err("program {s} places its output {f} unlike the tensor {f} on device {d}", .{ name, actual, expected, w.id });
            return error.InvalidBinding;
        }
    }
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

    /// The program adding `value` to every byte of its one input.
    const Add = struct {
        value: u8,

        pub fn forward(self: Add, _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0].add(Tensor.scalar(self.value, .u8));
        }
    };

    /// Whether `loader` failed with `expected` and holds nothing started.
    fn expectFailed(loader: *const Loader, expected: anyerror) !void {
        try std.testing.expectEqual(expected, loader.failure.?);
        try std.testing.expectEqual(0, loader.pending.len + loader.queued.len + loader.programs.items.len);
    }
};

test "start publishes the bulk at once and wait runs program groups one at a time" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    // One program tensor per group.
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2, .program_group_bytes = 1 });
    defer loader.deinit();

    const Model = struct { value: Tensor, first: Tensor, second: Tensor };
    const model: Model = .{
        .value = fixture.value,
        .first = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}),
        .second = fixture.computed("second", fixture.second, LoaderTestFixture.Identity{}),
    };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);

    try loader.start(&model, &buffers, &fixture.store, .none);
    errdefer loader.cancel();
    // The bulk is published; the groups wait for `wait`, their one program
    // (the same function and shapes) compiling meanwhile.
    try std.testing.expectEqual(1, loader.pending.len);
    try std.testing.expectEqual(2, loader.queued.len);
    try std.testing.expectEqual(1, loader.programs.items.len);
    try loader.wait();
    try std.testing.expectEqual(0, loader.pending.len + loader.queued.len + loader.programs.items.len);
    // Admission retired the bulk and the first group before it published
    // the second.
    try std.testing.expectEqual(2, loader.stats.admission_retires);
    try testing.expectShards(allocator, io, &buffers.value, &LoaderTestFixture.contents, .default);
    try testing.expectShards(allocator, io, &buffers.first, &LoaderTestFixture.contents, .default);
    try testing.expectShards(allocator, io, &buffers.second, &LoaderTestFixture.second_contents, .default);
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.stats.bytes_loaded);

    // Idempotent: a second wait neither reruns nor recounts.
    try loader.wait();
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.stats.bytes_loaded);

    // The next load counts and times from its own first start.
    try std.testing.expectEqual(null, loader.stats.started_at);
    var again: Buffer = undefined;
    try loader.start(&fixture.second, &again, &fixture.store, .none);
    try loader.wait();
    defer again.deinit();
    try std.testing.expectEqual(LoaderTestFixture.second_contents.len, loader.stats.bytes_loaded);
    try std.testing.expectEqual(null, loader.stats.started_at);
}

test "one loader takes loads from different stores before one wait" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var other_store = try testing.storeOf(allocator, &fixture.store.registry, .{ .meshes = &.{fixture.platform.replicated_mesh} });
    defer other_store.deinit();
    const other = other_store.view().createReplicatedTensor("second", null);

    var first: Buffer = undefined;
    var second: Buffer = undefined;
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    try loader.start(&fixture.value, &first, &fixture.store, .none);
    try loader.start(&other, &second, &other_store, .none);
    try loader.wait();
    defer first.deinit();
    defer second.deinit();
    try testing.expectShards(allocator, io, &first, &LoaderTestFixture.contents, .default);
    try testing.expectShards(allocator, io, &second, &LoaderTestFixture.second_contents, .default);
}

test "tensors of one program and equal args share one compile, and one group runs them all" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    const Add = LoaderTestFixture.Add;
    const Model = struct { a: Tensor, b: Tensor, c: Tensor };
    const model: Model = .{
        .a = fixture.computed("value", fixture.value, Add{ .value = 1 }),
        .b = fixture.computed("second", fixture.second, Add{ .value = 1 }),
        .c = fixture.computed("value", fixture.value, Add{ .value = 2 }),
    };
    try std.testing.expect(fixture.store.bindingOf(model.a.id).? == .compute);
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    errdefer loader.cancel();
    // Nothing for the loader to fill itself, and one group.
    try std.testing.expectEqual(0, loader.pending.len);
    try std.testing.expectEqual(1, loader.queued.len);
    try std.testing.expectEqual(2, loader.programs.items.len);
    try loader.wait();
    try testing.expectShards(allocator, io, &buffers.a, &.{ 2, 3, 4, 5 }, .default);
    try testing.expectShards(allocator, io, &buffers.b, &.{ 6, 7, 8, 9 }, .default);
    try testing.expectShards(allocator, io, &buffers.c, &.{ 3, 4, 5, 6 }, .default);
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.stats.bytes_loaded);
}

test "a program frees its donated and non-donated inputs once" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.writeFile(io, .{ .sub_path = "inputs", .data = std.mem.asBytes(&[4]i32{ 1, 2, 10, 20 }) });
    const path = try tmp.dir.realPathFileAlloc(io, "inputs", allocator);
    defer allocator.free(path);

    const shape: Shape = .init(.{2}, .i32);
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .name = "value", .file_uri = path, .shape = shape, .offset = 0 });
    try registry.registerTensor(.{ .name = "weight", .file_uri = path, .shape = shape, .offset = 2 * @sizeOf(i32) });
    var store: TensorStore = .init(allocator, &registry, .{ .meshes = &.{platform.replicated_mesh} });
    defer store.deinit();

    const Sum = struct {
        donate: bool,

        pub fn forward(self: @This(), _: Shape, inputs: []const Tensor) Tensor {
            const sum = inputs[0].add(inputs[1]);
            return if (self.donate) sum.reuseBuffer(inputs[0]) else sum;
        }
    };
    for ([_]bool{ false, true }) |donate| {
        var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
        defer loader.deinit();
        const tensor = TensorStore.Source.bindProgram(&.{
            store.view().source("value", null).withPartitioning(.replicated, .{}),
            store.view().source("weight", null).withPartitioning(.replicated, .{}),
        }, shape, Sum{ .donate = donate });
        var output: Buffer = undefined;
        try loader.start(&tensor, &output, &store, .none);
        errdefer loader.cancel();
        try loader.programs.items[0].await(io, platform);
        try std.testing.expectEqual(@as(usize, @intFromBool(donate)), loader.programs.items[0].exe.?.donated_input_indices.len);
        // Retiring runs the program, then frees every input handle once, the
        // donated one included.
        try loader.wait();
        defer output.deinit();
        try std.testing.expectEqual([2]i32{ 11, 22 }, try output.getValue([2]i32, io));
    }
}

test "cancel gives up on everything started without running its programs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    const Model = struct { value: Tensor, computed: Tensor };
    const model: Model = .{ .value = fixture.value, .computed = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}) };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    loader.cancel();
    try LoaderTestFixture.expectFailed(&loader, error.Canceled);
    try std.testing.expectEqual(@as(usize, 0), buffers.computed._shards.len);
    // The loader is failed: what it waits for and what it is given.
    try std.testing.expectError(error.Canceled, loader.wait());
    try std.testing.expectError(error.Canceled, loader.start(&model, &buffers, &fixture.store, .none));
    try std.testing.expectEqual(@as(usize, 0), loader.stats.bytes_loaded);
}

test "cancel frees a published group's inputs without running its program" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    const Model = struct { computed: Tensor };
    const model: Model = .{ .computed = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}) };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    errdefer loader.cancel();
    // Publish the group as `wait` does, then give up before it retires.
    try loader.publish(loader.queued.popFront().?);
    try std.testing.expect(loader.programGroupPending());
    loader.cancel();
    try LoaderTestFixture.expectFailed(&loader, error.Canceled);
    try std.testing.expect(!loader.programGroupPending());
    try std.testing.expectEqual(0, buffers.computed._shards.len);
}

test "a read failure fails the loader for every later call" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    // The broken group first, retired when the next one is admitted.
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2, .program_group_bytes = 1 });
    defer loader.deinit();

    const Model = struct { broken: Tensor, value: Tensor };
    const model: Model = .{
        .broken = fixture.computed("missing", fixture.missing, LoaderTestFixture.Identity{}),
        .value = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}),
    };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    try std.testing.expectError(error.FileNotFound, loader.wait());
    try LoaderTestFixture.expectFailed(&loader, error.FileNotFound);
    try std.testing.expectEqual(@as(usize, 0), buffers.value._shards.len);
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
    // a tensor in the file that does not exist, and a program over it.
    const Layer = struct { weight: Tensor, bias: ?Tensor };
    const Model = struct { layers: []Layer, broken: Tensor, computed: Tensor };
    var layers = [_]Layer{
        .{ .weight = fixture.value, .bias = fixture.second },
        .{ .weight = fixture.second, .bias = null },
    };
    const model: Model = .{
        .layers = &layers,
        .broken = fixture.missing,
        .computed = fixture.computed("missing", fixture.missing, LoaderTestFixture.Identity{}),
    };
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
    try std.testing.expectEqual(0, buffers.computed._shards.len);
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
    // Nothing started, so `deinit` cancels the set-up or destroys what it
    // built.
    var loader = try Loader.init(allocator, io, platform, .{});
    loader.deinit();
}

test "a failed set-up fails the first start, and deinit frees the rest" {
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(std.testing.allocator, io);
    defer fixture.deinit();
    // `init` allocates nothing; the set-up's first allocation, inside the
    // backend, fails.
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 0 });
    var loader = try Loader.init(failing.allocator(), io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var output: Buffer = undefined;
    try std.testing.expectError(error.OutOfMemory, loader.start(&fixture.value, &output, &fixture.store, .none));
    try LoaderTestFixture.expectFailed(&loader, error.OutOfMemory);
    try std.testing.expect(loader.backend == null);
    try std.testing.expectError(error.OutOfMemory, loader.wait());
}

test "without concurrency init sets up in place" {
    const allocator = std.testing.allocator;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, std.testing.io);
    defer fixture.deinit();
    // The testing io, but `io.concurrent` refuses and `io.async` runs the
    // function in place.
    var vtable = std.testing.io.vtable.*;
    vtable.concurrent = struct {
        fn concurrent(_: ?*anyopaque, _: usize, _: std.mem.Alignment, _: []const u8, _: std.mem.Alignment, _: *const fn (*const anyopaque, *anyopaque) void) std.Io.ConcurrentError!*std.Io.AnyFuture {
            return error.ConcurrencyUnavailable;
        }
    }.concurrent;
    vtable.async = struct {
        fn async(_: ?*anyopaque, result: []u8, _: std.mem.Alignment, context: []const u8, _: std.mem.Alignment, start: *const fn (*const anyopaque, *anyopaque) void) ?*std.Io.AnyFuture {
            start(context.ptr, result.ptr);
            return null;
        }
    }.async;
    const io: std.Io = .{ .userdata = std.testing.io.userdata, .vtable = &vtable };

    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    try std.testing.expect(loader.setup.?.any_future == null);
    // Nothing started: neither has anything to do.
    loader.cancel();
    try loader.wait();
    try std.testing.expectEqual(null, loader.failure);

    var output: Buffer = undefined;
    try loader.start(&fixture.value, &output, &fixture.store, .none);
    try loader.wait();
    defer output.deinit();
    try testing.expectShards(allocator, io, &output, &LoaderTestFixture.contents, .default);
}

test "a rejected binding or a program of the wrong output fails the loader" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = fixture.platform;
    // Two rows of two, not the four bytes the tensor holds.
    const Rows = struct {
        pub fn forward(_: @This(), _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0].reshape(.{ 2, 2 });
        }
    };
    // A mesh of devices the platform does not have: the same tensors placed
    // on it, in device and host memory, and a program's output.
    var foreign_physical = platform.replicated_mesh.physical.*;
    const foreign_devices = try allocator.dupe(Sharding.Device, foreign_physical.devices_in_canonical_order);
    defer allocator.free(foreign_devices);
    for (foreign_devices) |*device| device.id += 100;
    foreign_physical.devices_in_canonical_order = foreign_devices;
    var foreign_mesh = platform.replicated_mesh.*;
    foreign_mesh.physical = &foreign_physical;
    const pinned = fixture.store.view().createReplicatedTensor("second", null);
    fixture.store.view().setMemory(pinned, .host_pinned);
    var foreign = [_]Tensor{ fixture.value, pinned, fixture.value };
    for (&foreign) |*tensor| tensor._shape._sharding.mesh = &foreign_mesh;
    // The program places its output replicated, the tensor expects its
    // bytes split over the devices: same dims, another placement. It needs
    // devices that split four bytes.
    var misplaced = fixture.value;
    misplaced._shape = Shape.init(.{ .x = 4 }, .u8).withPartitioning(platform.meshes.get("model").?, .{ .x = .model });
    const splits = platform.devices.len > 1 and 4 % platform.devices.len == 0;

    // No binding, an empty source or program input, a mesh of other
    // devices, a host kind the platform cannot take, a broadcast's strided
    // runs, and a program of the wrong shape or placement.
    const Case = struct { tensor: Tensor, at_wait: bool = false, skip: bool = false };
    const cases = [_]Case{
        .{ .tensor = Tensor.fromShape(fixture.value.shape()) },
        .{ .tensor = fixture.empty },
        .{ .tensor = fixture.computed("empty", fixture.empty, LoaderTestFixture.Identity{}) },
        .{ .tensor = foreign[0] },
        .{ .tensor = foreign[1] },
        .{ .tensor = fixture.computed("value", foreign[2], LoaderTestFixture.Identity{}) },
        .{ .tensor = pinned, .skip = backend.supportsMemory(platform, .host_pinned) },
        .{ .tensor = fixture.store.view().source("value", .{.x}).broad(Shape.init(.{ .y = 2, .x = 4 }, .u8)).withPartitioning(.replicated, .{}).bind() },
        .{ .tensor = fixture.computed("value", fixture.value, Rows{}), .at_wait = true },
        .{ .tensor = fixture.computed("value", misplaced, LoaderTestFixture.Identity{}), .at_wait = true, .skip = !splits },
    };
    for (cases) |case| {
        if (case.skip) continue;
        var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
        defer loader.deinit();
        var output: Buffer = undefined;
        if (case.at_wait) {
            try loader.start(&case.tensor, &output, &fixture.store, .none);
            try std.testing.expectError(error.InvalidBinding, loader.wait());
        } else {
            try std.testing.expectError(error.InvalidBinding, loader.start(&case.tensor, &output, &fixture.store, .none));
        }
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

test "a program takes its inputs from sources, in order" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const Source = TensorStore.Source;
    const view = fixture.store.view();

    // value's first half then second's last half, assembled by the loader,
    // and second as it is: the program subtracts the second input from the
    // first, which tells them apart.
    const Sub = struct {
        pub fn forward(_: @This(), _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0].sub(inputs[1]);
        }
    };
    const assembled = Source.concatenate(&.{
        view.source("value", .{.x}).slice(.x, .{ .end = 2 }),
        view.source("second", .{.x}).slice(.x, .{ .start = 2 }),
    }, .x);
    const second = view.source("second", .{.x});
    const inputs = [_]Source.Placed{ assembled.withPartitioning(.replicated, .{}), second.withPartitioning(.replicated, .{}) };
    const tensor = Source.bindProgram(&inputs, second.shape(), Sub{});
    const compute = fixture.store.bindingOf(tensor.id).?.compute;
    try std.testing.expectEqual(2, compute.inputs.len);
    try std.testing.expect(compute.program.as(Sub) != null and compute.program.as(LoaderTestFixture.Identity) == null);
    try std.testing.expect(compute.inputs[0].whole() == null);
    try std.testing.expect(compute.inputs[1].whole() != null);
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var output: Buffer = undefined;
    try loader.start(&tensor, &output, &fixture.store, .none);
    try loader.wait();
    defer output.deinit();
    // { 1, 2, 7, 8 } - { 5, 6, 7, 8 }, wrapping.
    try testing.expectShards(allocator, io, &output, &.{ 252, 252, 0, 0 }, .default);
}
