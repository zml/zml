//! Checkpoint loading. `Loader.start` takes a whole model: it starts
//! compiling the programs that compute some of its tensors
//! (`Source.bindProgram`) and publishes every tensor the loader fills
//! itself. `wait` then publishes the programs' inputs in groups the devices
//! have room for, runs each group's programs on the caller's task once its
//! inputs landed, and retires everything. Backends only read sources into
//! buffers; lookup, compilation, admission and execution stay in this
//! shared front end. The contract is in the pub doc comments below;
//! docs/learn/loader.md is the user guide.
//!
//! From bindings to specs. `start` checks every tensor and program input
//! once, before anything is published (`prepareSpec`, `checkPlacement`).
//! It lowers each piece of a bound source to bytes (`Source.lowerPiece`, the
//! one rule): a part of the destination's logical bytes (the row-major bytes
//! of its shape, before the sharding splits them), a run of `width` bytes
//! repeated over strided dimensions. A whole checkpoint tensor is one part.
//! A strided part of at most `max_part_runs` runs, or of runs of at least
//! `large_run_bytes`, becomes one part per run; a spec with any other
//! strided part (row interleaves, transposes, broadcasts) goes to the
//! backend as strided parts, which the direct backend gathers. Each
//! `LoadSpec` is one destination: its placed shape, its memory kind, the
//! caller's output and the parts that fill it. The backends trust it. Only
//! `Source` makes bindings and its pieces are disjoint by construction:
//! nothing here checks that (Source's tests do).
//!
//! Order and admission. The tensors the loader fills go out first, as one
//! submission, so every program group is admitted with them counted. A
//! group holds the tensors of consecutive programs in model order, up to
//! `program_group_bytes` of outputs, so the inputs of one layer's experts,
//! adjacent in the checkpoint, are read together. Before publishing a
//! group, `wait` waits for its programs to compile, then retires the oldest
//! pending submissions until the group fits the room every device reports
//! (`admission`): its limit, minus what is in use, minus the bytes the
//! direct backend published but has not allocated yet, minus a 64 MiB
//! reserve. A group costs its inputs, its outputs and the largest of its
//! programs' temporaries per device, since they run one after the other.
//! Once no program group is pending, retiring frees nothing more, and the
//! group goes anyway (one warning). Where a device reports no limit (CPU)
//! or the backend does not count its allocations (buffered), nothing is
//! measured and one program group is in flight at a time.
//!
//! Accounting. Each spec is one progress item, which its backend
//! completes; a program's output adds none. `wait`'s line counts each
//! destination's bytes once: a composite counts its padding and not its
//! sources, and a group counts its programs' inputs.
const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Compiler = @import("../Compiler.zig");
const Exe = @import("../exe.zig").Exe;
const mem = @import("../mem.zig");
const Bufferized = mem.Bufferized;
const meta = @import("../meta.zig");
const ops = @import("../ops.zig");
const platform_mod = @import("../platform.zig");
const Memory = platform_mod.Memory;
const Platform = platform_mod.Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const testing = @import("../testing.zig");
const backend = @import("backend.zig");
const Backend = backend.Backend;
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const dma_calibration = @import("dma_calibration.zig");
const load_log = @import("log.zig").load;
const load_testing = @import("testing.zig");
const TensorStore = @import("TensorStore.zig");

const PrepareError = std.mem.Allocator.Error || error{InvalidBinding};

/// Loads models from their checkpoint sources, and computes the tensors
/// programs compute from theirs. A value: `init` returns at once, and the
/// loader may be moved until its first `start`.
///
/// What it borrows. The platform outlives the loader and every buffer it
/// loads. `io`, and the VFS behind it, outlive the loader: its tasks run on
/// `io`, and the direct backend opens each source file through it and
/// keeps it open until `deinit`, under a copy of the file's path, so
/// nothing of the store is kept past `wait`. `start` borrows the store, the
/// buffers and the progress node until `wait` or `cancel` returns, and
/// reads the model during the call only. The loaded buffers are the
/// caller's and outlive the loader.
///
/// Start and wait serially on the owning task; source reads and transfers
/// run concurrently inside the selected backend, compiles on tasks of
/// their own. Submissions are retired in publish order.
pub const Loader = struct {
    /// Every error `start` and `wait` return. Each one fails the loader:
    /// everything started is canceled (`cancel`), and every later call
    /// returns the first error. A model whose bindings are wrong is
    /// `InvalidBinding`, logged at err with the tensor, the key its source
    /// reads (a source that reads nothing has none) and the rule it
    /// breaks. Buffers shaped unlike the model (`mem.pairBufferized`) are
    /// `InvalidBinding` too, logged with the field path. The others come
    /// from reading the checkpoint (a missing file is `FileNotFound`; a
    /// denied one `AccessDenied` locally and `PermissionDenied` remotely; a
    /// file that ends inside a tensor `UnexpectedEndOfFile`, because it
    /// changed after its header was read or the tensor's shape needs more
    /// bytes than its `data_offsets` give), the devices, the compiler, or
    /// `cancel` (`Canceled`), and from `init`'s task: a platform with more
    /// devices than the direct backend addresses is `UnsupportedPlatform`,
    /// an io that cannot run its tasks concurrently
    /// `ConcurrencyUnavailable`. Host memory the backend cannot allocate,
    /// pin or fit under its mapped ceiling is `OutOfMemory`, logged with
    /// the cause; device memory that runs out is PJRT's `ResourceExhausted`.
    pub const Error = error{ InvalidBinding, Canceled, UnsupportedPlatform } || std.mem.Allocator.Error || pjrt.ApiError || std.Io.File.OpenError ||
        safetensors.ExactPositionalError || Compiler.Error || std.Io.ConcurrentError;

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    /// `init`'s task, which warms the device allocators and creates the
    /// backend, until the first `start` awaits it. The future holds what
    /// the task builds and the task holds no pointer into the loader, so
    /// the loader can be moved before that `start`.
    setup: ?std.Io.Future(Error!Backend),
    /// What `setup` created; null before the first `start`, and after a
    /// failed set-up.
    backend: ?Backend = null,
    /// Output bytes of the program tensors one group publishes at most. A
    /// group's inputs are planned together, so the weights and scales of one
    /// layer's experts, adjacent in the checkpoint, coalesce into full
    /// requests instead of one read per expert scale; they are freed once
    /// the group's programs ran.
    program_group_bytes: usize = 8 << 30,
    /// The programs of the loads started since the last `wait`, each
    /// compiled once, on a task of its own.
    programs: std.ArrayList(*CompiledProgram) = .empty,
    /// Program groups started and not yet published, oldest first.
    queued: std.Deque(Group) = .empty,
    /// Submissions published and not yet retired, oldest first.
    pending: std.Deque(PendingSubmission) = .empty,
    /// The first error; every later call returns it.
    failure: ?Error = null,
    /// When the first `start` since the last `wait` ran: `wait` logs the
    /// load's time from there.
    started_at: ?std.Io.Timestamp = null,
    /// Logical bytes of the submissions retired with execution since that
    /// `start`: each destination's shape once, so a composite tensor counts
    /// its padding and not its sources one by one, and a program group
    /// counts its inputs.
    bytes_loaded: usize = 0,
    /// Whether admission can measure the room: the backend counts its
    /// allocations and every device reported an allocator limit at the
    /// first `start`. Otherwise one program group is in flight at a time.
    memory_supported: bool = false,
    /// Output and temporary bytes of the pending program groups, per
    /// device: what retiring them will take before it frees anything.
    pending_execution: []u64,
    pending_executes: usize = 0,
    submissions: usize = 0,
    /// Submissions retired to admit a program group, bulk loads included.
    admission_retires: usize = 0,
    /// How long publishing waited for compiles since the last `wait`.
    compile_wait_ns: u64 = 0,
    min_room_seen: ?u64 = null,
    oversized_logged: bool = false,
    temp_unavailable_logged: bool = false,
    /// Per-device scratch for admission, `platform.devices.len` each.
    scratch: Scratch,

    const Scratch = struct {
        /// The four `u64` slices below, one allocation.
        words: []u64,
        room: []u64,
        unlanded: []u64,
        inputs: []u64,
        temporaries: []u64,
    };

    pub const Options = struct {
        /// The checkpoint's path or URI, read by `init` only. When `io` is a
        /// VFS, the VFS backend serving that path sizes the reads (its
        /// `VFS.LoadProfile`). Null, or a plain `io`: the local sizing.
        checkpoint: ?[]const u8 = null,
        /// Concurrent source reads, 1 to `max_read_parallelism`. Null takes
        /// the checkpoint's default (`backend.Options.readWidth`: 16
        /// locally, 32 on a high-latency source), clipped to one less than
        /// what the pre-grown pinned set holds. Fixed for the load: a source
        /// that rate limits holds the VFS instead (`vfs/request.zig`).
        read_parallelism: ?usize = null,
        /// The DMA block sizes calibration screens, each at least the direct
        /// I/O alignment (`VFS.direct_io_alignment`, 4 KiB, when `io` is a
        /// VFS and `direct_io` is not `off`; else 1) and at most 32 MiB
        /// (`backend.max_read_request_size`). With that alignment, a request
        /// (the larger of a block and the checkpoint's read chunk) exceeds
        /// two units.
        /// Ignored by the buffered backend; CPU takes the default sizing
        /// without measuring. Read by `init`'s task: the slice must live
        /// until the first `start` or `deinit`.
        dma_block_sizes: []const usize = &dma_calibration.default_block_sizes,
        /// Direct I/O for local source files, decided per file by the
        /// VFS that opens it: `auto` reads a file past the page cache when it
        /// is mostly not cached at the first decision, `on` whenever the
        /// filesystem allows it, `off` never. The checkpoint plays no part:
        /// a local file beside a remote checkpoint may be read directly, and
        /// a remote file never is, nor opened to ask. The planner widens a
        /// direct file's reads to `VFS.direct_io_alignment`, at most two
        /// alignment units per request. A direct read never fills the page
        /// cache, so under `auto` a cold file stays cold and comes from the
        /// disk on every load; on a host whose warm buffered reads beat its
        /// disk, a model loaded repeatedly is better served by `off`
        /// (measurements in vfs/direct_io.zig). Nothing changes when `io` is
        /// not a VFS. The loader keeps each file open until `deinit`, so a
        /// file's decision holds for every load of this loader.
        direct_io: VFS.DirectIo = .auto,
    };

    /// Concurrent source reads at most (`Options.read_parallelism`).
    pub const max_read_parallelism: usize = 128;

    /// Checks `opts` and returns at once. Warming the device allocators and
    /// creating the backend (for the direct one: DMA calibration and the
    /// pinned pre-growth) run on a task of the loader's own, or here when
    /// `io` has no concurrency; the first `start` waits for that task and
    /// returns its error. A read width or a block size out of range, an
    /// empty block list, a checkpoint whose scheme the VFS does not serve,
    /// a read chunk that a block calibration may select does not divide, or
    /// a request too short for a widened direct read is `InvalidOptions`,
    /// logged.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) error{ InvalidOptions, OutOfMemory }!Loader {
        const backend_opts = try backendOptions(io, opts);

        const n = platform.devices.len;

        const pending_execution = try allocator.alloc(u64, n);
        errdefer allocator.free(pending_execution);
        @memset(pending_execution, 0);

        const words = try allocator.alloc(u64, 4 * n);

        // Started last, so that nothing fails after it.
        const setup_args = .{ allocator, io, platform, backend_opts };
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .setup = io.concurrent(setUp, setup_args) catch io.async(setUp, setup_args),
            .pending_execution = pending_execution,
            .scratch = .{
                .words = words,
                .room = words[0..n],
                .unlanded = words[n .. 2 * n],
                .inputs = words[2 * n .. 3 * n],
                .temporaries = words[3 * n .. 4 * n],
            },
        };
    }

    /// `init`'s task: the device allocators warmed, then the backend.
    /// `InvalidOptions` means `init` let through options a backend refuses.
    fn setUp(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, opts: backend.Options) Error!Backend {
        platform.warmupDeviceAllocators(io) catch |err| {
            load_log.failure(err, "warm the device allocators", .{});
            return err;
        };
        return Backend.init(allocator, io, platform, opts) catch |err| switch (err) {
            error.InvalidOptions => stdx.debug.panic("loader backend: InvalidOptions after init checked them", .{}),
            else => |e| e,
        };
    }

    /// Waits for `init`'s task the first time: its backend, then whether
    /// admission can measure the devices; or its error, for the caller to
    /// fail the loader with.
    fn awaitSetup(self: *Loader) Error!void {
        const setup = if (self.setup) |*task| task else return;
        const result = setup.await(self.io);
        self.setup = null;
        self.backend = try result;
        self.memory_supported = self.probeMemory();
    }

    /// Gives up on everything started: drops the queued groups, aborts the
    /// backend (queued reads are dropped, reads in flight finish), retires
    /// the pending submissions without running their programs, and waits
    /// for the compiles. Each buffer of the started loads is then empty or
    /// holds its shards, whose contents are unspecified (unwritten,
    /// errored, or partly written); read none, and free the tree as after a
    /// load (`mem.deinitBufferized`). The loader is failed afterwards. For
    /// an error between `start` and `wait`; nothing to do when nothing was
    /// started.
    pub fn cancel(self: *Loader) void {
        if (!self.loadsStarted()) return;
        if (self.failure == null) self.failure = error.Canceled;
        while (self.queued.popFront()) |group| group.deinit(self.allocator);
        if (self.pending.len != 0) self.backend.?.abort();
        while (self.pending.len != 0) self.retireOldest(false) catch {};
        self.releasePrograms();
    }

    /// Cancels, with a warning, loads still started (`cancel`): a caller
    /// that succeeded waited for them, and an error path failed the loader
    /// or canceled it (`errdefer loader.cancel()`), so any left lack their
    /// `wait`. Then cancels `init`'s task when no `start` awaited it,
    /// destroying what it built, and destroys the backend: its pinned
    /// memory, its workers and the files it keeps open, closed through
    /// `io`. The buffers it loaded stay the caller's.
    pub fn deinit(self: *Loader) void {
        if (self.loadsStarted()) load_log.warn("Loader.deinit cancels loads started without a wait", .{});
        self.cancel();
        self.logAdmission();
        self.queued.deinit(self.allocator);
        self.pending.deinit(self.allocator);
        self.programs.deinit(self.allocator);
        self.allocator.free(self.pending_execution);
        self.allocator.free(self.scratch.words);
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

    /// Decided once: the CPU plugin logs an unimplemented warning on every
    /// stats call, so admission never asks a device that answered nothing.
    fn probeMemory(self: *Loader) bool {
        if (self.backend.? != .direct) return false;
        for (self.platform.devices) |device| {
            if (device.memoryStats().bytes_limit == null) return false;
        }
        return true;
    }

    /// Starts loading every tensor of `model` into its buffer in `buffers`,
    /// as `store` binds it, and returns; `wait` returns once it is done.
    /// Each tensor is placed by its shape's sharding (replicated without a
    /// mesh), into the memory space its binding names, assembled from the
    /// pieces of its source (`Source.bind`) with every byte no piece covers
    /// set to zero. All of these are published here, as one planned
    /// submission. A tensor a program computes (`Source.bindProgram`) starts
    /// its program compiling, once for every tensor of an equal program
    /// over the same placed shapes; `wait` lands its inputs and runs it.
    ///
    /// Every tensor and program input is checked once, before anything is
    /// published: a tensor `store` did not bind, a source of no bytes or of
    /// an empty checkpoint tensor, a source that does not fill its tensor's
    /// bytes or that this loader cannot assemble (a run that starts or ends
    /// inside a byte, strided parts into host memory), a placement its mesh
    /// does not divide or on devices the platform lacks, and a memory kind
    /// the platform cannot take are `error.InvalidBinding`. On the direct
    /// backend, a broadcast or transpose whose source range does not fit
    /// one read is `InvalidBinding` when its submission is published: here
    /// for the tensors the loader fills, in `wait` for a program's inputs.
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
    /// input; the backends complete them.
    /// Several loads may be started before one `wait`, from one store or
    /// several. The first `start` waits for `init`'s task, and fails with
    /// its error.
    pub fn start(
        self: *Loader,
        model: anytype,
        buffers: *Bufferized(@TypeOf(model.*)),
        store: *const TensorStore,
        progress: std.Progress.Node,
    ) Error!void {
        if (self.failure) |err| return err;
        self.startModel(@TypeOf(model.*), model, buffers, store, progress) catch |err| {
            self.fail(err);
            return err;
        };
    }

    fn startModel(
        self: *Loader,
        comptime Model: type,
        model: *const Model,
        buffers: *Bufferized(Model),
        store: *const TensorStore,
        progress: std.Progress.Node,
    ) Error!void {
        try self.awaitSetup();
        // The load's time starts once the set-up is done.
        if (self.started_at == null) {
            self.started_at = .now(self.io, .awake);
            self.bytes_loaded = 0;
        }
        var prepared = try prepareModel(self.allocator, self.platform, store, Model, model, buffers);
        defer prepared.deinit();
        // The backends complete one item per spec: one per tensor the loader
        // fills and, in the groups, one per program input.
        var items = prepared.specs.len;
        for (prepared.computed) |item| items += item.compute.inputs.len;
        progress.increaseEstimatedTotalItems(items);
        // Queued first: their programs compile while the rest is read.
        try self.queueGroups(store, prepared.computed, progress);
        if (prepared.specs.len != 0) try self.submit(prepared.specs, &.{}, &.{}, progress);
    }

    /// Publishes the queued program groups, oldest first, each once its
    /// programs compiled and the devices have room for it (`admit`), and
    /// retires every submission in publish order, running a group's
    /// programs on this task once its inputs landed. Returns once every
    /// started load is in its buffers, and logs one info line: the bytes
    /// loaded, the time since the first `start` and their rate, then what
    /// the programs computed and how long publishing waited for their
    /// compiles. A program that fails to compile or run fails it; one that
    /// does not return its tensor's shape, placed alike, is
    /// `InvalidBinding`, and so is a program input the direct backend
    /// cannot gather in one read (`start`). Idempotent: with nothing started
    /// it returns at once and logs nothing.
    pub fn wait(self: *Loader) Error!void {
        if (self.failure) |err| return err;
        self.drain() catch |err| {
            self.fail(err);
            return err;
        };
        self.releasePrograms();
    }

    fn drain(self: *Loader) Error!void {
        var groups: usize = 0;
        var tensors: usize = 0;
        while (self.queued.frontPtr()) |group| {
            tensors += group.executables.len;
            try self.publish(group);
            self.queued.popFront().?.deinit(self.allocator);
            groups += 1;
        }
        while (self.pending.len != 0) try self.retireOldest(true);
        if (self.started_at) |started_at| {
            const took = started_at.untilNow(self.io, .awake);
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
                    self.programs.items.len,
                    groups,
                    self.compile_wait_ns / std.time.ns_per_ms,
                });
            }
        }
        self.started_at = null;
        self.compile_wait_ns = 0;
    }

    /// Records `err` unless the loader failed already, and cancels
    /// everything started.
    fn fail(self: *Loader, err: Error) void {
        if (self.failure == null) self.failure = err;
        self.cancel();
    }

    /// Queues the tensors of `computed`, in model order, in groups of at
    /// most `program_group_bytes` of outputs, one tensor at least.
    fn queueGroups(self: *Loader, store: *const TensorStore, computed: []const Computed, progress: std.Progress.Node) Error!void {
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
    /// checked as `start` checks every spec (`prepareSpec`).
    fn prepareGroup(self: *Loader, store: *const TensorStore, computed: []const Computed, progress: std.Progress.Node) Error!Group {
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        errdefer arena.deinit();
        const executables = try self.allocator.alloc(BoundExecutable, computed.len);
        var prepared: usize = 0;
        errdefer {
            for (executables[0..prepared]) |*executable| executable.deinit(self.allocator);
            self.allocator.free(executables);
        }
        var specs: std.ArrayList(LoadSpec) = .empty;
        for (computed, executables) |item, *executable| {
            const compiled = try self.compiledProgram(store, item.compute, item.shape);
            const inputs = try self.allocator.alloc(Buffer, item.compute.inputs.len);
            for (inputs, item.compute.inputs) |*input, source| {
                input.* = .{ ._platform = self.platform, ._shape = source.shape, ._shards = .empty };
            }
            executable.* = .{ .program = compiled, .inputs = inputs, .output = item.output };
            prepared += 1;
            for (item.compute.inputs, inputs, 0..) |source, *input, index| {
                const target: Target = .{ .id = item.id, .shape = item.shape, .input = .{ .index = index, .shape = source.shape } };
                const spec = try prepareSpec(arena.allocator(), self.platform, target, source, .default, input);
                try specs.append(arena.allocator(), spec);
            }
        }
        return .{ .executables = executables, .specs = specs.items, .arena = arena, .progress = progress };
    }

    /// The compiled program computing a tensor of `output` from `compute`:
    /// one started already for an equal program over the same placed
    /// shapes, else a compile started now.
    fn compiledProgram(self: *Loader, store: *const TensorStore, compute: TensorStore.Binding.Compute, output: Shape) Error!*CompiledProgram {
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

    /// Publishes `group` once its programs compiled and the devices have
    /// room for it; its executables move to the pending submission.
    fn publish(self: *Loader, group: *Group) Error!void {
        const started: std.Io.Timestamp = .now(self.io, .awake);
        for (group.executables) |*executable| executable.exe = try executable.program.await(self.io, self.platform);
        self.compile_wait_ns +|= @intCast(started.untilNow(self.io, .awake).toNanoseconds());
        const execution = try self.allocator.alloc(u64, self.platform.devices.len);
        errdefer self.allocator.free(execution);
        try self.admit(group.executables, execution);
        try self.submit(group.specs, group.executables, execution, group.progress);
        group.executables = &.{};
    }

    /// Fills `execution` with the group's output and temporary bytes per
    /// device, then retires pending submissions, oldest first, until the
    /// group fits the measured room or no program group is pending: a bulk
    /// load frees nothing, and once nothing can be retired for it the group
    /// is admitted whatever its size. Without memory measurement one
    /// program group is in flight at a time.
    fn admit(self: *Loader, executables: []const BoundExecutable, execution: []u64) Error!void {
        const inputs = self.scratch.inputs;
        const temporaries = self.scratch.temporaries;
        @memset(inputs, 0);
        @memset(temporaries, 0);
        @memset(execution, 0);
        for (executables) |*executable| {
            const exe = executable.exe;
            for (exe.input_shapes) |shape| addPlacementBytes(inputs, self.platform, shape);
            // The output's memory space is the program's to choose; it is
            // charged to the devices like the default space.
            addPlacementBytes(execution, self.platform, exe.output_shapes[0]);
            // The group's programs run one after the other, so only the
            // largest temporaries are live at once.
            const temp = self.compiledTempBytes(exe);
            const output_mesh = self.platform.placedShape(exe.output_shapes[0])._sharding.mesh.?;
            for (output_mesh.devicesInCanonicalOrder()) |device| temporaries[device.id] = @max(temporaries[device.id], temp);
        }
        for (execution, temporaries) |*bytes, temp| bytes.* +|= temp;
        var retired: usize = 0;
        var fit = self.measureFit(inputs, execution);
        while (fit != .fits and self.pending_executes != 0) : (retired += 1) {
            try self.retireOldest(true);
            fit = self.measureFit(inputs, execution);
        }
        if (fit == .exceeds and !self.oversized_logged) {
            self.oversized_logged = true;
            load_log.warn("program group exceeds the device room alone: inputs={Bi:.2} execution={Bi:.2} room={Bi:.2} (device 0); admitted anyway", .{
                inputs[0],
                execution[0],
                self.scratch.room[0],
            });
        }
        self.admission_retires += retired;
        if (retired != 0) {
            load_log.debug("execute admission: retired={d} before submission {d}: measured={}, device 0 room={Bi:.2} pending_execution={Bi:.2} inputs={Bi:.2} execution={Bi:.2}", .{
                retired,
                self.submissions,
                self.memory_supported,
                if (self.memory_supported) self.scratch.room[0] else 0,
                self.pending_execution[0],
                inputs[0],
                execution[0],
            });
        }
    }

    const Fit = enum { unmeasured, fits, exceeds };

    /// Whether the group fits beside the pending executions in the room the
    /// devices report now.
    fn measureFit(self: *Loader, inputs: []const u64, execution: []const u64) Fit {
        if (!self.memory_supported or !self.readRoom()) return .unmeasured;
        return if (admission.admits(self.scratch.room, self.pending_execution, inputs, execution)) .fits else .exceeds;
    }

    /// Refreshes the per-device room; false when a device reports no memory
    /// limit. Only called when `memory_supported`, which implies the direct
    /// backend (`probeMemory`).
    fn readRoom(self: *Loader) bool {
        self.backend.?.direct.unlandedBytesPerDevice(self.scratch.unlanded);
        for (self.scratch.room, self.platform.devices, self.scratch.unlanded) |*out, device, unlanded| {
            const reported = device.memoryStats();
            out.* = admission.room(reported.bytes_limit, reported.bytes_in_use, unlanded, admission.reserve_bytes) orelse return false;
            self.min_room_seen = @min(self.min_room_seen orelse out.*, out.*);
        }
        return true;
    }

    /// Temporaries the executable takes on each of its devices, from its
    /// compiled memory stats (per partition, i.e. per device). Zero when
    /// the plugin cannot answer.
    fn compiledTempBytes(self: *Loader, exe: *const Exe) u64 {
        const api = self.platform.pjrt_api;
        const executable = exe.exe.executable(api) catch |err| return self.tempUnavailable(err);
        defer executable.deinit(api);
        const stats = executable.getCompiledMemoryStats(api) catch |err| return self.tempUnavailable(err);
        return stats.temp_size_in_bytes;
    }

    fn tempUnavailable(self: *Loader, err: pjrt.ApiError) u64 {
        if (!self.temp_unavailable_logged) {
            self.temp_unavailable_logged = true;
            load_log.debug("executable memory stats unavailable ({s}): temporaries counted as zero", .{@errorName(err)});
        }
        return 0;
    }

    /// One submission over `specs`. Takes `executables` and `execution`
    /// once the submission is published; on failure the caller still owns
    /// them. `execution` holds a program group's output and temporary bytes
    /// per device, empty for a bulk load.
    fn submit(self: *Loader, specs: []const LoadSpec, executables: []BoundExecutable, execution: []u64, progress: std.Progress.Node) Error!void {
        // One spec per destination, however many parts fill it: its bytes
        // count once.
        var logical_bytes: usize = 0;
        for (specs) |spec| logical_bytes += spec.shape.byteSize();
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const submission = self.backend.?.submit(specs, progress) catch |err| return backendError(err);
        self.pending.pushBackAssumeCapacity(.{
            .submission = submission,
            .executables = executables,
            .execution = execution,
            .logical_bytes = logical_bytes,
        });
        self.submissions += 1;
        if (executables.len != 0) {
            for (self.pending_execution, execution) |*pending, bytes| pending.* +|= bytes;
            self.pending_executes += 1;
        }
    }

    /// Retires the oldest pending submission: waits for its reads and DMA,
    /// runs its programs when `execute`, frees its inputs either way and
    /// counts its bytes once it ran. The first error is kept in `failure`
    /// and aborts the backend: nothing still pending can succeed.
    fn retireOldest(self: *Loader, execute: bool) Error!void {
        var oldest = self.pending.popFront().?;
        defer self.release(&oldest);
        self.retire(&oldest, execute) catch |err| {
            if (self.failure == null) {
                self.failure = err;
                self.backend.?.abort();
            }
            return err;
        };
    }

    fn retire(self: *Loader, oldest: *PendingSubmission, execute: bool) Error!void {
        self.backend.?.await(oldest.submission) catch |err| return backendError(err);
        if (!execute) return;
        for (oldest.executables) |*executable| {
            executable.execute(self.allocator, self.io) catch |err| {
                load_log.err("run program {s}: inputs={d}, output={f}: {s}", .{ executable.program.name(), executable.inputs.len, executable.program.output, @errorName(err) });
                return err;
            };
        }
        self.bytes_loaded += oldest.logical_bytes;
    }

    fn release(self: *Loader, oldest: *PendingSubmission) void {
        for (oldest.executables) |*executable| executable.deinit(self.allocator);
        if (oldest.executables.len != 0) {
            for (self.pending_execution, oldest.execution) |*pending, bytes| pending.* -= bytes;
            self.pending_executes -= 1;
        }
        self.allocator.free(oldest.executables);
        self.allocator.free(oldest.execution);
    }

    fn logAdmission(self: *const Loader) void {
        if (self.submissions == 0) return;
        load_log.debug("loader admission: submissions={d}, execute_admission_retires={d}, memory_supported={}, min_room_seen={Bi:.2}, reserve={Bi:.2}", .{
            self.submissions,
            self.admission_retires,
            self.memory_supported,
            self.min_room_seen orelse 0,
            admission.reserve_bytes,
        });
    }
};

/// One published submission: what to wait for, what to run afterwards and
/// what its accounting owes.
const PendingSubmission = struct {
    submission: backend.Submission,
    /// Run in order once the reads are done; freed with their inputs when
    /// the submission is retired. Empty for a bulk load.
    executables: []BoundExecutable,
    /// Output and temporary bytes per device; empty for a bulk load.
    execution: []u64,
    logical_bytes: usize,
};

/// Program tensors whose inputs `wait` publishes as one submission.
const Group = struct {
    executables: []BoundExecutable,
    /// One per program input, filling the executables' input shells; they
    /// and their parts live in `arena`.
    specs: []LoadSpec,
    arena: std.heap.ArenaAllocator,
    progress: std.Progress.Node,

    fn deinit(self: Group, allocator: std.mem.Allocator) void {
        for (self.executables) |*executable| executable.deinit(allocator);
        allocator.free(self.executables);
        self.arena.deinit();
    }
};

/// A program compiling on a task of its own, then compiled.
const CompiledProgram = struct {
    compute: TensorStore.Binding.Compute,
    output: Shape,
    /// The compile task's arguments, borrowed until it completes.
    inputs: []Tensor,
    future: std.Io.Future(Compiler.Error!Exe),
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
        errdefer allocator.destroy(self);
        const inputs = try allocator.alloc(Tensor, compute.inputs.len);
        for (inputs, compute.inputs) |*input, source| input.* = .fromShape(source.shape);
        // Concurrent, as `io.async` may run the compile right here on a busy
        // threaded io, ahead of the reads `start` publishes after it.
        const args = .{ compute.program, allocator, io, platform, meshes, output, inputs };
        self.* = .{
            .compute = compute,
            .output = output,
            .inputs = inputs,
            .future = io.concurrent(compile, args) catch io.async(compile, args),
        };
        return self;
    }

    fn compile(
        program: TensorStore.Program,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        meshes: []const *const Sharding.Mesh,
        output: Shape,
        inputs: []const Tensor,
    ) Compiler.Error!Exe {
        return program.vtable.compile(program.value, allocator, io, platform, .{
            .meshes = meshes,
            .program_name = program.vtable.name,
        }, output, inputs);
    }

    /// The program's type name.
    fn name(self: *const CompiledProgram) []const u8 {
        return self.compute.program.vtable.name;
    }

    /// The executable, waiting for the compile the first time. It must
    /// compute one tensor of `output`, placed alike (`checkOutput`).
    fn await(self: *CompiledProgram, io: std.Io, platform: *const Platform) Loader.Error!*const Exe {
        if (self.exe == null) {
            self.exe = self.future.await(io) catch |err| {
                load_log.err("compile program {s} computing {f}: {s}", .{ self.name(), self.output, @errorName(err) });
                return err;
            };
            try checkOutput(platform, self.name(), self.output, &self.exe.?);
        }
        return &self.exe.?;
    }

    fn destroy(self: *CompiledProgram, allocator: std.mem.Allocator, io: std.Io) void {
        if (self.exe) |*exe| {
            exe.deinit();
        } else if (self.future.cancel(io)) |exe| {
            exe.deinit();
        } else |_| {}
        allocator.free(self.inputs);
        allocator.destroy(self);
    }
};

/// One tensor of a program group: its program, the input shells the
/// group's submission fills, and the model's buffer it computes.
const BoundExecutable = struct {
    program: *CompiledProgram,
    inputs: []Buffer,
    output: *Buffer,
    /// The program's executable, once its group is published.
    exe: *const Exe = undefined,

    /// Runs the executable over the loaded inputs on the calling task.
    /// `Exe.call` leaves every input handle to the caller, donated ones
    /// included (their contents are gone), so `deinit` frees them all,
    /// exactly once; a runner would destroy the donated handles itself.
    /// A failed execution, an input whose transfer failed included, is
    /// returned and leaves `output` unwritten.
    fn execute(self: *const BoundExecutable, allocator: std.mem.Allocator, io: std.Io) !void {
        var args = try self.exe.args(allocator);
        defer args.deinit(allocator);
        var results = try self.exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{self.inputs});
        try self.exe.call(io, args, &results, .{ .wait = true });
        self.output.* = results.get(Buffer);
    }

    /// Frees the inputs, donated or not; a shell the loader never wrote owns
    /// no shards.
    fn deinit(self: *BoundExecutable, allocator: std.mem.Allocator) void {
        for (self.inputs) |*input| input.deinit();
        allocator.free(self.inputs);
    }
};

/// The arithmetic behind program admission: what the devices have left,
/// and whether the next group fits beside the ones still pending.
/// Everything is per device (index = `platform.devices` index) in bytes;
/// nothing here touches a device, so the rules are tested without one.
const admission = struct {
    /// Allocator slack per device. BFC rounds every allocation up to 256 bytes;
    /// a 64-pack, 16-source, 8-device load is about 1e5 shards, under 32 MiB of
    /// rounding, and nothing else runs on the devices during a load.
    const reserve_bytes: u64 = 64 << 20;

    /// Bytes the loader may still take on one device: the limit (null when the
    /// plugin reports none, as CPU does) minus what is in use, minus its own
    /// bytes published but not allocated yet (`unlanded`), minus the reserve.
    fn room(limit: ?u64, in_use: u64, unlanded: u64, reserve: u64) ?u64 {
        return (limit orelse return null) -| in_use -| unlanded -| reserve;
    }

    /// Whether one submission's `inputs` (placements, counted here until they
    /// are published) and `execution` (output placement plus the executable's
    /// temporaries) fit beside the pending executions on every device.
    fn admits(rooms: []const u64, pending: []const u64, inputs: []const u64, execution: []const u64) bool {
        for (rooms, pending, inputs, execution) |r, p, i, e| {
            if (p +| i +| e > r) return false;
        }
        return true;
    }
};

test "room subtracts use, unlanded bytes and the reserve, saturating" {
    try std.testing.expectEqual(null, admission.room(null, 0, 0, 0));
    try std.testing.expectEqual(55, admission.room(100, 30, 10, 5).?);
    // More in use than the limit admits: no room, no underflow.
    try std.testing.expectEqual(0, admission.room(100, 130, 0, 0).?);
    try std.testing.expectEqual(0, admission.room(100, 90, 0, 20).?);
}

test "admits needs room on every device" {
    try std.testing.expect(admission.admits(&.{ 40, 40 }, &.{ 20, 20 }, &.{ 10, 10 }, &.{ 5, 5 }));
    // The second device is one byte short.
    try std.testing.expect(!admission.admits(&.{ 40, 34 }, &.{ 20, 20 }, &.{ 10, 10 }, &.{ 5, 5 }));
    try std.testing.expect(admission.admits(&.{}, &.{}, &.{}, &.{}));
}

/// Adds the bytes `shape` occupies on each device it is placed on, as the
/// direct backend allocates them. `shape` places on the platform's devices
/// (`checkPlacement`).
fn addPlacementBytes(out: []u64, platform: *const Platform, shape: Shape) void {
    const placement = Sharding.Placement.init(platform.placedShape(shape).packedShape()) catch |err|
        std.debug.panic("loader: {f} does not place, though checked: {s}", .{ shape, @errorName(err) });
    const bytes: u64 = placement.shape.byteSize();
    for (placement.mesh().?.devicesInCanonicalOrder()) |device| out[device.id] += bytes;
}

/// A backend's error as `start` and `wait` return it. A binding the backend
/// refused, which it logged, is `InvalidBinding`. The other names it keeps
/// to itself mean a broken invariant: a backend returns the error that
/// closed it, never `Closed`; `InvalidOptions` was checked by `init`; and
/// `Internal` is the direct backend's own bookkeeping.
fn backendError(err: backend.SubmitError) Loader.Error {
    return switch (err) {
        error.UnsupportedLayout,
        error.InvalidTensorRange,
        error.UnsupportedMemoryKind,
        error.IncompatibleSharding,
        => error.InvalidBinding,
        error.Closed, error.InvalidOptions, error.Internal => stdx.debug.panic("loader backend: {s}", .{@errorName(err)}),
        else => |e| e,
    };
}

/// A tensor a program computes, the buffer it computes into, and its
/// program and inputs.
const Computed = struct {
    id: Tensor.Id,
    shape: Shape,
    output: *Buffer,
    compute: TensorStore.Binding.Compute,
};

/// A model as `start` takes it: one spec per tensor the loader fills, and
/// the tensors programs compute, all in `arena` with the specs' parts.
const PreparedModel = struct {
    arena: std.heap.ArenaAllocator,
    specs: []const LoadSpec,
    computed: []const Computed,

    fn deinit(self: *PreparedModel) void {
        self.arena.deinit();
    }
};

/// Pairs each tensor of `model` with its buffer and binding: a tensor the
/// loader fills takes one spec (`prepareSpec`), a tensor a program computes
/// is placed like any other (`checkPlacement`) and computed.
fn prepareModel(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    comptime Model: type,
    model: *const Model,
    buffers: *Bufferized(Model),
) PrepareError!PreparedModel {
    // Tensors and buffers pair by structure, before any buffer is written: a
    // buffer tree shaped otherwise is refused whole.
    const Pair = struct { tensor: *const Tensor, buffer: *Buffer };
    var pairs: std.ArrayList(Pair) = try .initCapacity(allocator, meta.count(Tensor, model));
    defer pairs.deinit(allocator);
    var report_buffer: [256]u8 = undefined;
    var report: std.Io.Writer = .fixed(&report_buffer);
    mem.pairBufferized(Model, model, buffers, &report, &pairs, struct {
        fn call(list: *std.ArrayList(Pair), tensor: *const Tensor, buffer: *Buffer) void {
            list.appendAssumeCapacity(.{ .tensor = tensor, .buffer = buffer });
        }
    }.call) catch {
        load_log.err("buffers shaped unlike the model at model{s}; bufferize them from this model", .{report.buffered()});
        return error.InvalidBinding;
    };
    // Empty until a backend or a program writes it whole, so the caller can
    // free the tree whatever fails.
    for (pairs.items) |pair| pair.buffer.* = .{ ._platform = platform, ._shape = pair.tensor.shape(), ._shards = .empty };

    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();
    var specs: std.ArrayList(LoadSpec) = try .initCapacity(arena.allocator(), pairs.items.len);
    var computed: std.ArrayList(Computed) = try .initCapacity(arena.allocator(), pairs.items.len);
    for (pairs.items) |pair| {
        const tensor = pair.tensor;
        const binding = store.bindingOf(tensor.id) orelse {
            load_log.err("tensor {} {f} has no checkpoint binding", .{ tensor.id, tensor.shape() });
            return error.InvalidBinding;
        };
        const target: Target = .{ .id = tensor.id, .shape = tensor.shape() };
        switch (binding) {
            .load => |load| specs.appendAssumeCapacity(try prepareSpec(arena.allocator(), platform, target, load.source, load.memory, pair.buffer)),
            .compute => |compute| {
                try checkPlacement(platform, target, compute.program.vtable.name);
                computed.appendAssumeCapacity(.{ .id = tensor.id, .shape = tensor.shape(), .output = pair.buffer, .compute = compute });
            },
        }
    }
    return .{ .arena = arena, .specs = specs.items, .computed = computed.items };
}

/// A strided part with few runs, or with large ones, expands into one part
/// per run, each read and transferred on its own: at most `max_part_runs`
/// runs, or runs of at least `large_run_bytes`. A spec with any other
/// strided part goes to the backend whole, as strided parts
/// (`backend.Lattice`), which the direct backend reads by destination
/// ranges.
const max_part_runs = 64;
const large_run_bytes = 1 << 20;

fn expandable(part: LoadPart) bool {
    return part.dims.len == 0 or part.lattice().runs() <= max_part_runs or part.width >= large_run_bytes;
}

/// What a spec fills, as the logs name it: tensor `id` of `shape`, or
/// `input` of the program computing it.
const Target = struct {
    id: Tensor.Id,
    shape: Shape,
    input: ?Input = null,

    const Input = struct { index: usize, shape: Shape };

    /// The shape the spec fills: the input's, else the tensor's.
    fn destination(self: Target) Shape {
        return if (self.input) |input| input.shape else self.shape;
    }

    pub fn format(self: Target, writer: *std.Io.Writer) !void {
        if (self.input) |input| try writer.print("program input {d} {f} of ", .{ input.index, input.shape });
        try writer.print("tensor {} {f}", .{ self.id, self.shape });
    }
};

/// The spec filling `output`, the buffer of `target`, from `source` into
/// `memory`, its parts in `arena`: each piece lowered to bytes
/// (`Source.lowerPiece`), expanded into its runs unless a part does not
/// expand, in destination order. The one check of what the backends trust
/// (`backend.LoadSpec`); a rule it breaks is `InvalidBinding`, logged with
/// the target and the key its source reads, if it reads any.
fn prepareSpec(
    arena: std.mem.Allocator,
    platform: *const Platform,
    target: Target,
    source: TensorStore.Source.Bound,
    memory: Memory.Kind,
    output: *Buffer,
) PrepareError!LoadSpec {
    const shape = target.destination();
    if (source._pieces.len == 0) {
        load_log.err("{f} reads no checkpoint bytes", .{target});
        return error.InvalidBinding;
    }
    const key = source._pieces[0].tensor.name;
    for (source._pieces) |piece| {
        if (piece.tensor.byteSize() == 0) {
            load_log.err("{f} reads {s} in {s}, which is empty", .{ target, piece.tensor.name, piece.tensor.file_uri });
            return error.InvalidBinding;
        }
    }
    if (source.shape.byteSize() != shape.byteSize()) {
        load_log.err("{f} of {d} bytes from {s}: its source fills {f}, {d} bytes", .{ target, shape.byteSize(), key, source.shape, source.shape.byteSize() });
        return error.InvalidBinding;
    }
    const lowered = try arena.alloc(LoadPart, source._pieces.len);
    var expands = true;
    var runs: u64 = 0;
    for (source._pieces, lowered) |piece, *part| {
        part.* = TensorStore.Source.lowerPiece(arena, source.shape, piece) catch |err| switch (err) {
            error.OutOfMemory => |e| return e,
            error.SubBytePosition => {
                load_log.err("{f}: its source {f} reads a run of {s} that starts or ends inside a byte, which the loader cannot place", .{ target, source.shape, piece.tensor.name });
                return error.InvalidBinding;
            },
        };
        expands = expands and expandable(part.*);
        runs += part.lattice().runs();
    }
    try checkPlacement(platform, target, key);
    if (!backend.supportsMemory(platform, memory)) {
        load_log.err("{f} from {s} is bound to {t} memory, which {t} cannot take", .{ target, key, memory, platform.target });
        return error.InvalidBinding;
    }
    if (!expands and memory.isHost()) {
        load_log.err("{f} from {s}: strided parts into {t} memory, which this loader does not assemble", .{ target, key, memory });
        return error.InvalidBinding;
    }
    const parts = if (!expands) lowered else expanded: {
        if (runs == lowered.len) break :expanded lowered;
        const expanded = try arena.alloc(LoadPart, @intCast(runs));
        var next: usize = 0;
        for (lowered) |part| {
            var cursor: backend.Lattice.Cursor = .init(part.lattice(), 0, part.lattice().runs());
            while (cursor.next()) |run| : (next += 1) {
                expanded[next] = .{ .source = part.source, .src_offset = run.src, .dst_offset = run.dst, .width = part.width };
            }
        }
        break :expanded expanded;
    };
    std.mem.sort(LoadPart, parts, {}, struct {
        fn lessThan(_: void, lhs: LoadPart, rhs: LoadPart) bool {
            return lhs.dst_offset < rhs.dst_offset;
        }
    }.lessThan);
    if (!backend.checkParts(parts, shape.byteSize())) {
        load_log.err("{f} from {s}: its parts leave their sources or its {d} bytes, or come out of order", .{ target, key, shape.byteSize() });
        return error.InvalidBinding;
    }
    return .{ .shape = shape, .memory = memory, .output = output, .parts = parts };
}

/// Checks that `target`, from `from` (its source's key or its program),
/// places on `platform`'s devices, in any memory space: its mesh
/// partitions divide its dimensions, and every device of its mesh is one of
/// the platform's.
fn checkPlacement(platform: *const Platform, target: Target, from: []const u8) error{InvalidBinding}!void {
    const placement = Sharding.Placement.init(platform.placedShape(target.destination()).packedShape()) catch {
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

/// `opts` as the backends take them, the checkpoint resolved to the read
/// hints of the VFS backend serving it. Every option is checked here, once.
fn backendOptions(io: std.Io, opts: Loader.Options) error{InvalidOptions}!backend.Options {
    const profile: VFS.LoadProfile = profile: {
        const path = opts.checkpoint orelse break :profile .local;
        const vfs = VFS.fromIo(io) orelse break :profile .local;
        break :profile vfs.loadProfile(path) catch |err| {
            load_log.err("invalid loader options: checkpoint={s}: {s}", .{ path, @errorName(err) });
            return error.InvalidOptions;
        };
    };
    if (opts.read_parallelism) |width| {
        if (width == 0 or width > Loader.max_read_parallelism) {
            load_log.err("invalid loader options: read_parallelism={d}, expected 1..{d}: InvalidOptions", .{ width, Loader.max_read_parallelism });
            return error.InvalidOptions;
        }
    }
    const max_request = backend.max_read_request_size;
    if (profile.read_chunk_size == 0 or profile.read_chunk_size > max_request) {
        load_log.err("invalid loader options: profile={s}, read_chunk_size={d}, expected 1..{d}: InvalidOptions", .{ profile.name, profile.read_chunk_size, max_request });
        return error.InvalidOptions;
    }
    if (opts.dma_block_sizes.len == 0) {
        load_log.err("invalid loader options: dma_block_sizes is empty: InvalidOptions", .{});
        return error.InvalidOptions;
    }
    // A direct read is widened by up to an alignment unit at each end,
    // which a job of smaller blocks cannot hold (`Planner.maximumJobLen`).
    const alignment = backend.readAlignment(io, opts.direct_io);
    const min_block = @max(alignment, 1);
    for (opts.dma_block_sizes) |block_size| {
        if (block_size < min_block or block_size > max_request) {
            load_log.err("invalid loader options: profile={s}, dma block size {d}, expected {d}..{d}: InvalidOptions", .{ profile.name, block_size, min_block, max_request });
            return error.InvalidOptions;
        }
    }
    // A source request is the larger of the read chunk and the block
    // calibration selected, and a whole number of blocks (the direct
    // `Loader.create`); with an alignment, it holds a direct read widened at
    // each end. Checked for every block it may select: a candidate, or on
    // CPU the default.
    for ([_][]const usize{ opts.dma_block_sizes, &.{dma_calibration.Result.default.block_size} }) |block_sizes| {
        for (block_sizes) |block_size| {
            const request = @max(profile.read_chunk_size, block_size);
            if (request % block_size != 0) {
                load_log.err("invalid loader options: profile={s}, read_chunk_size={d} is not a multiple of the dma block size {d}: InvalidOptions", .{ profile.name, profile.read_chunk_size, block_size });
                return error.InvalidOptions;
            }
            if (alignment != 0 and request <= 2 * alignment) {
                load_log.err("invalid loader options: profile={s}, request of {d} cannot hold a direct read widened by {d} at each end: InvalidOptions", .{ profile.name, request, alignment });
                return error.InvalidOptions;
            }
        }
    }
    return .{
        .read_parallelism = opts.read_parallelism,
        .load_profile = profile,
        .dma_block_sizes = opts.dma_block_sizes,
        .direct_io = opts.direct_io,
    };
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
    const want = Sharding.Placement.init(platform.placedShape(expected).packedShape()) catch |err|
        std.debug.panic("loader: {f} does not place, though checked: {s}", .{ expected, @errorName(err) });
    const got = Sharding.Placement.init(platform.placedShape(actual).packedShape()) catch |err| {
        load_log.err("program {s}: placement of its output {f}: {s}", .{ name, actual, @errorName(err) });
        return error.InvalidBinding;
    };
    const want_devices = want.mesh().?.devicesInCanonicalOrder();
    const got_devices = got.mesh().?.devicesInCanonicalOrder();
    if (want_devices.len != got_devices.len) {
        load_log.err("program {s} places its output {f} on {d} devices, the tensor {f} on {d}", .{ name, actual, got_devices.len, expected, want_devices.len });
        return error.InvalidBinding;
    }
    for (want_devices, got_devices) |w, g| {
        const want_slices = want.slices(w.coords);
        const got_slices = got.slices(g.coords);
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
    path_buffer: [1024]u8,
    path_len: usize,
    missing_buffer: [1024]u8,
    missing_len: usize,
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
        self.path_len = try file.realPath(io, &self.path_buffer);
        file.close(io);
        const path = self.path_buffer[0..self.path_len];
        const missing_path = try std.fmt.bufPrint(&self.missing_buffer, "{s}.missing", .{path});
        self.missing_len = missing_path.len;

        var registry: safetensors.TensorRegistry = .init(allocator);
        {
            errdefer registry.deinit();
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
        }
        self.platform = testing.env();
        self.store = .init(allocator, registry, .{ .meshes = &.{self.platform.replicated_mesh} });
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
        return self.store.view().maybeCreateBinding(&.{name}, like.shape(), program).?;
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

    fn expectContents(allocator: std.mem.Allocator, io: std.Io, buffer: *const Buffer, expected: []const u8) !void {
        const loaded = try buffer.toSliceAlloc(allocator, io);
        defer loaded.free(allocator);
        try std.testing.expectEqualSlices(u8, expected, loaded.constData());
    }

    /// Whether `loader` failed with `expected` and holds nothing started.
    fn expectFailed(loader: *const Loader, expected: anyerror) !void {
        try std.testing.expectEqual(expected, loader.failure.?);
        try std.testing.expectEqual(0, loader.pending.len + loader.queued.len + loader.programs.items.len);
    }
};

test "start publishes the bulk at once and wait runs program groups one at a time without memory measurement" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    // Exercise serial admission regardless of the platform's memory stats,
    // one program tensor per group: decided once the set-up is done.
    try loader.awaitSetup();
    loader.memory_supported = false;
    loader.program_group_bytes = 1;

    const Model = struct { value: Tensor, first: Tensor, second: Tensor };
    const model: Model = .{
        .value = fixture.value,
        .first = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}),
        .second = fixture.computed("second", fixture.second, LoaderTestFixture.Identity{}),
    };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);

    try loader.start(&model, &buffers, &fixture.store, .none);
    // The bulk is published; the groups wait for `wait`, their one program
    // (the same function and shapes) compiling meanwhile.
    try std.testing.expectEqual(1, loader.pending.len);
    try std.testing.expectEqual(2, loader.queued.len);
    try std.testing.expectEqual(1, loader.programs.items.len);
    try loader.wait();
    try std.testing.expectEqual(0, loader.pending.len + loader.queued.len + loader.programs.items.len);
    // Serial admission retired the bulk and the first group before it
    // published the second.
    try std.testing.expectEqual(2, loader.admission_retires);
    try LoaderTestFixture.expectContents(allocator, io, &buffers.value, &LoaderTestFixture.contents);
    try LoaderTestFixture.expectContents(allocator, io, &buffers.first, &LoaderTestFixture.contents);
    try LoaderTestFixture.expectContents(allocator, io, &buffers.second, &LoaderTestFixture.second_contents);
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytes_loaded);

    // Idempotent: a second wait neither reruns nor recounts.
    try loader.wait();
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytes_loaded);

    // The next load counts and times from its own first start.
    try std.testing.expectEqual(null, loader.started_at);
    var again: Buffer = undefined;
    try loader.start(&fixture.second, &again, &fixture.store, .none);
    try loader.wait();
    defer again.deinit();
    try std.testing.expectEqual(LoaderTestFixture.second_contents.len, loader.bytes_loaded);
    try std.testing.expectEqual(null, loader.started_at);
}

test "one loader takes loads from different stores before one wait" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var other_store = try load_testing.storeOf(allocator, &fixture.store.registry, &.{fixture.platform.replicated_mesh});
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
    try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);
    try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
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
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    // Nothing for the loader to fill itself, and one group.
    try std.testing.expectEqual(0, loader.pending.len);
    try std.testing.expectEqual(1, loader.queued.len);
    try std.testing.expectEqual(2, loader.programs.items.len);
    try loader.wait();
    try LoaderTestFixture.expectContents(allocator, io, &buffers.a, &.{ 2, 3, 4, 5 });
    try LoaderTestFixture.expectContents(allocator, io, &buffers.b, &.{ 6, 7, 8, 9 });
    try LoaderTestFixture.expectContents(allocator, io, &buffers.c, &.{ 3, 4, 5, 6 });
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytes_loaded);
}

test "loader delivers a host-pinned binding into host pinned memory without charging the devices" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = fixture.platform;
    const pinned = fixture.store.view().createReplicatedTensor("second", null);
    fixture.store.view().setMemory(pinned, .host_pinned);
    try std.testing.expectEqual(Memory.Kind.host_pinned, fixture.store.bindingOf(pinned.id).?.load.memory);

    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    const Model = struct { device: Tensor, host: Tensor };
    const model: Model = .{ .device = fixture.value, .host = pinned };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);

    if (!backend.supportsMemory(platform, .host_pinned)) {
        // Refused before anything is submitted.
        try std.testing.expectError(error.InvalidBinding, loader.start(&model, &buffers, &fixture.store, .none));
        try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
        return;
    }
    try loader.start(&model, &buffers, &fixture.store, .none);
    try loader.wait();
    try LoaderTestFixture.expectContents(allocator, io, &buffers.device, &LoaderTestFixture.contents);
    try LoaderTestFixture.expectContents(allocator, io, &buffers.host, &LoaderTestFixture.second_contents);

    // One shard per device, each in that device's host pinned memory space.
    // On CPU/neuron/metal, Memory.isOfKind matches every kind, so
    // memory(.host_pinned) resolves to the default "device" memory and the
    // routing cannot be observed there. Only CUDA/ROCm/oneAPI exercise it.
    const devices = platform.placedShape(pinned.shape())._sharding.mesh.?.devicesInCanonicalOrder();
    try std.testing.expectEqual(devices.len, buffers.host._shards.len);
    for (devices, buffers.host._shards.constSlice()) |device, shard| {
        const pinned_memory = platform.devices[device.id].memory(.host_pinned).?;
        const default_memory = platform.devices[device.id].memory(.default).?;
        if (pinned_memory.pjrt_memory == default_memory.pjrt_memory) continue;
        try std.testing.expect(shard.memory(platform.pjrt_api) == pinned_memory.pjrt_memory);
        try std.testing.expect(shard.memory(platform.pjrt_api) != default_memory.pjrt_memory);
    }

    // Only the tensor in device memory counts against the devices, on both
    // sides of the admission ledger.
    if (loader.backend.? == .direct) {
        const allocated = try allocator.alloc(u64, platform.devices.len);
        defer allocator.free(allocated);
        loader.backend.?.direct.allocatedBytesPerDevice(allocated);
        try std.testing.expectEqualSlices(u64, loader.backend.?.direct.submitted_bytes, allocated);
        for (devices) |device| try std.testing.expectEqual(LoaderTestFixture.contents.len, allocated[device.id]);
    }
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
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try registry.registerTensor(.{ .name = "value", .file_uri = path, .shape = shape, .offset = 0 });
        try registry.registerTensor(.{ .name = "weight", .file_uri = path, .shape = shape, .offset = 2 * @sizeOf(i32) });
        break :store .init(allocator, registry, .{ .meshes = &.{platform.replicated_mesh} });
    };
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
        const tensor = store.view().maybeCreateBinding(&.{ "value", "weight" }, shape, Sum{ .donate = donate }) orelse unreachable;
        var output: Buffer = undefined;
        try loader.start(&tensor, &output, &store, .none);
        const exe = try loader.programs.items[0].await(io, platform);
        try std.testing.expectEqual(@as(usize, @intFromBool(donate)), exe.donated_input_indices.len);
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
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    loader.cancel();
    try LoaderTestFixture.expectFailed(&loader, error.Canceled);
    try std.testing.expectEqual(@as(usize, 0), buffers.computed._shards.len);
    // The loader is failed: what it waits for and what it is given.
    try std.testing.expectError(error.Canceled, loader.wait());
    try std.testing.expectError(error.Canceled, loader.start(&model, &buffers, &fixture.store, .none));
    try std.testing.expectEqual(@as(usize, 0), loader.bytes_loaded);
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
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    // Publish the group as `wait` does, then give up before it retires.
    try loader.publish(loader.queued.frontPtr().?);
    loader.queued.popFront().?.deinit(allocator);
    try std.testing.expectEqual(1, loader.pending_executes);
    loader.cancel();
    try LoaderTestFixture.expectFailed(&loader, error.Canceled);
    try std.testing.expectEqual(0, loader.pending_executes);
    for (loader.pending_execution) |bytes| try std.testing.expectEqual(0, bytes);
    try std.testing.expectEqual(0, buffers.computed._shards.len);
}

test "a read failure fails the loader for every later call" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    // The broken group first, retired when the next one is admitted.
    try loader.awaitSetup();
    loader.memory_supported = false;
    loader.program_group_bytes = 1;

    const Model = struct { broken: Tensor, value: Tensor };
    const model: Model = .{
        .broken = fixture.computed("missing", fixture.missing, LoaderTestFixture.Identity{}),
        .value = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}),
    };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    try std.testing.expectError(error.FileNotFound, loader.wait());
    try LoaderTestFixture.expectFailed(&loader, error.FileNotFound);
    try std.testing.expectEqual(@as(usize, 0), buffers.value._shards.len);
    try std.testing.expectError(error.FileNotFound, loader.start(&model, &buffers, &fixture.store, .none));
    try std.testing.expectError(error.FileNotFound, loader.wait());
    try std.testing.expectEqual(@as(usize, 0), loader.bytes_loaded);
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
    var buffers = try mem.bufferize(allocator, Model, &model);
    // Frees whatever the failed load wrote, under the testing allocator.
    defer mem.deinitBufferized(allocator, Model, &buffers);
    // The direct backend may open the file as it plans, in `start`.
    const result = if (loader.start(&model, &buffers, &fixture.store, .none)) loader.wait() else |err| err;
    try std.testing.expectError(error.FileNotFound, result);
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

test "the direct backend allocates what it was submitted once everything landed" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const placed = try allocator.alloc(u64, fixture.platform.devices.len);
    defer allocator.free(placed);
    @memset(placed, 0);
    addPlacementBytes(placed, fixture.platform, fixture.value.shape());
    for (placed) |bytes| try std.testing.expectEqual(LoaderTestFixture.contents.len, bytes);

    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    const Model = struct { value: Tensor, computed: Tensor };
    const model: Model = .{ .value = fixture.value, .computed = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}) };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
    try loader.wait();
    if (loader.backend.? != .direct) return;
    const allocated = try allocator.alloc(u64, fixture.platform.devices.len);
    defer allocator.free(allocated);
    loader.backend.?.direct.allocatedBytesPerDevice(allocated);
    // Replicated program inputs and bulk outputs occupy every device.
    for (allocated) |bytes| try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, bytes);
    try std.testing.expectEqualSlices(u64, loader.backend.?.direct.submitted_bytes, allocated);
    const unlanded = try allocator.alloc(u64, fixture.platform.devices.len);
    defer allocator.free(unlanded);
    loader.backend.?.direct.unlandedBytesPerDevice(unlanded);
    for (unlanded) |bytes| try std.testing.expectEqual(0, bytes);
}

test "init checks every option and returns InvalidOptions" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const platform = testing.env();
    var vfs: VFS = try .init(allocator, io);
    defer vfs.deinit();
    const too_large = [_]usize{backend.max_read_request_size + 1};
    const Case = struct { io: std.Io, opts: Loader.Options };
    const cases = [_]Case{
        .{ .io = io, .opts = .{ .read_parallelism = 0 } },
        .{ .io = io, .opts = .{ .read_parallelism = Loader.max_read_parallelism + 1 } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &.{} } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &.{0} } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &too_large } },
        // The local profile reads 8 MiB at least: not a whole number of
        // 3 MiB blocks.
        .{ .io = io, .opts = .{ .dma_block_sizes = &.{3 << 20} } },
        // No backend of the VFS serves the scheme.
        .{ .io = vfs.io(), .opts = .{ .checkpoint = "missing://bucket/model" } },
    };
    for (cases) |case| {
        if (Loader.init(allocator, case.io, platform, case.opts)) |value| {
            var loader = value;
            loader.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.InvalidOptions, err);
    }
    // Under a VFS, a local file may be read directly: a block holds an
    // alignment unit at least.
    if (VFS.direct_io_alignment != 0) {
        const small = [_]usize{VFS.direct_io_alignment / 2};
        if (Loader.init(allocator, vfs.io(), platform, .{ .checkpoint = "/model.safetensors", .dma_block_sizes = &small })) |value| {
            var loader = value;
            loader.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.InvalidOptions, err);
        // Whatever the checkpoint: a request of one alignment unit cannot
        // hold a direct read widened at each end.
        try vfs.registerBackend("tiny", .{ .io = io, .read_hints = .{ .read_chunk_size = VFS.direct_io_alignment } });
        const unit = [_]usize{VFS.direct_io_alignment};
        if (Loader.init(allocator, vfs.io(), platform, .{ .checkpoint = "tiny://model", .dma_block_sizes = &unit })) |value| {
            var loader = value;
            loader.deinit();
            return error.TestUnexpectedResult;
        } else |err| try std.testing.expectEqual(error.InvalidOptions, err);
    }
    // A plain io has no VFS to ask: local sizing. Nothing started, so
    // `deinit` cancels the set-up or destroys what it built.
    var loader = try Loader.init(allocator, io, platform, .{ .checkpoint = "missing://bucket/model" });
    loader.deinit();
}

test "a failed set-up fails the first start, and deinit frees the rest" {
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(std.testing.allocator, io);
    defer fixture.deinit();
    // `init` allocates its two per-device slices, then the set-up's first
    // allocation, the backend, fails.
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 2 });
    var loader = try Loader.init(failing.allocator(), io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var output: Buffer = undefined;
    try std.testing.expectError(error.OutOfMemory, loader.start(&fixture.value, &output, &fixture.store, .none));
    try LoaderTestFixture.expectFailed(&loader, error.OutOfMemory);
    try std.testing.expect(loader.backend == null);
    try std.testing.expectError(error.OutOfMemory, loader.wait());
}

test "without concurrency init sets up in place, and the loader moves before its first start" {
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
    try std.testing.expect(loader.setup.?.any_future == null);
    // Nothing started: neither has anything to do.
    loader.cancel();
    try loader.wait();
    try std.testing.expectEqual(null, loader.failure);

    var moved = loader;
    defer moved.deinit();
    var output: Buffer = undefined;
    try moved.start(&fixture.value, &output, &fixture.store, .none);
    try moved.wait();
    defer output.deinit();
    try LoaderTestFixture.expectContents(allocator, io, &output, &LoaderTestFixture.contents);
}

test "a rate-limited HTTP source loads through the VFS hold" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const tensor_bytes = 1024;
    const tensors = 4;

    // Four tensor ranges with a gap between them, so the planner keeps four
    // jobs and the load makes four concurrent GETs.
    var object: [(2 * tensors - 1) * tensor_bytes]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 31 +% 7);

    var server = try VFS.mock_server.MockServer.init(io, &object, .{
        // Two GETs per 100 ms: the four readers are throttled repeatedly.
        .throttle = .{ .window = .{ .gets = 2, .per_ms = 100 } },
    });
    var server_group: std.Io.Group = .init;
    try VFS.mock_server.startMockServer(&server, &server_group, io);
    var server_joined = false;
    defer VFS.mock_server.cleanupMockServer(&server, &server_group, io, server_joined);

    var client: std.http.Client = .{ .allocator = allocator, .io = io };
    defer client.deinit();
    var http: VFS.HTTP = try .init(allocator, io, &client, .{ .protocol = .http, .retry = .{
        .max_retries = 5,
        .initial_delay = .fromMilliseconds(20),
        .max_delay = .fromMilliseconds(20),
        .max_hold = .fromSeconds(5),
        .throttle_budget = .fromSeconds(30),
    } });
    defer http.deinit();
    var vfs: VFS = try .init(allocator, io);
    defer vfs.deinit();
    try vfs.registerBackend("http", http.backend());
    const stats = &http.read_stats;

    var url_buffer: [128]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});

    const platform = testing.env();
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        const names = [tensors][]const u8{ "a", "b", "c", "d" };
        for (names, 0..) |name, index| {
            try registry.registerTensor(.{
                .file_uri = url,
                .name = name,
                .shape = .init(.{tensor_bytes}, .u8),
                .offset = 2 * index * tensor_bytes,
            });
        }
        break :store .init(allocator, registry, .{ .meshes = &.{platform.replicated_mesh} });
    };
    defer store.deinit();

    const Model = struct { a: Tensor, b: Tensor, c: Tensor, d: Tensor };
    const model: Model = .{
        .a = store.view().createReplicatedTensor("a", null),
        .b = store.view().createReplicatedTensor("b", null),
        .c = store.view().createReplicatedTensor("c", null),
        .d = store.view().createReplicatedTensor("d", null),
    };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);

    var loader = try Loader.init(allocator, vfs.io(), platform, .{
        .checkpoint = url,
        .read_parallelism = 4,
    });
    defer loader.deinit();
    try loader.start(&model, &buffers, &store, .none);
    try loader.wait();

    inline for (comptime std.meta.fieldNames(Model), 0..) |field_name, index| {
        try LoaderTestFixture.expectContents(
            allocator,
            io,
            &@field(buffers, field_name),
            object[2 * index * tensor_bytes ..][0..tensor_bytes],
        );
    }

    server_group.cancel(io);
    server_joined = true;
    try server.check();

    // The server rate limited the load, the backend held every reader, and
    // no reader burned its retry budget for it.
    const snapshot = stats.snapshot();
    try std.testing.expect(snapshot.throttles > 0);
    try std.testing.expect(snapshot.holds > 0);
    try std.testing.expect(snapshot.hold_wait_ns > 0);
    try std.testing.expectEqual(@as(u64, 0), snapshot.retries);
    try std.testing.expectEqual(@as(u64, tensors * tensor_bytes), snapshot.physical_bytes);

    // Holding costs time, not credits: every lifecycle credit, with its
    // pinned blocks, came back.
    const credits = &loader.backend.?.direct.credits;
    try std.testing.expectEqual(credits.free.len, credits.free_len);

    // The checkpoint's VFS backend, not the local profile (8 MiB), sized the
    // reads: 16 MiB requests over the CPU's 4 MiB blocks. Its file was read
    // exact: a remote file is never direct.
    const direct = loader.backend.?.direct;
    try std.testing.expectEqual(@as(usize, 16 << 20), direct.plan_config.request_size);
    try std.testing.expectEqual(@as(usize, 0), direct.source_slots.get(url).?.alignment orelse 0);
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

    // An empty checkpoint tensor binds one piece of it, as a tensor or as a
    // program's input, by which `start` names it.
    const empty_entry = fixture.store.view().get("empty").?;
    const empty_input = fixture.computed("empty", fixture.empty, LoaderTestFixture.Identity{});
    try std.testing.expectEqual(empty_entry, fixture.store.bindingOf(fixture.empty.id).?.load.source._pieces[0].tensor);
    try std.testing.expectEqual(empty_entry, fixture.store.bindingOf(empty_input.id).?.compute.inputs[0]._pieces[0].tensor);
    // No binding, an empty source or program input, a mesh of other
    // devices, and a program of the wrong shape or placement.
    const Case = struct { tensor: Tensor, at_wait: bool = false, skip: bool = false };
    const cases = [_]Case{
        .{ .tensor = Tensor.fromShape(fixture.value.shape()) },
        .{ .tensor = fixture.empty },
        .{ .tensor = empty_input },
        .{ .tensor = foreign[0] },
        .{ .tensor = foreign[1] },
        .{ .tensor = fixture.computed("value", foreign[2], LoaderTestFixture.Identity{}) },
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
    var buffers = try mem.bufferize(allocator, Model, &built);
    defer mem.deinitBufferized(allocator, Model, &buffers);
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

test "loader reports source size mismatches" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    // Eight bytes from a source of four.
    var tensor = fixture.value;
    tensor._shape = .init(.{8}, .u8);
    var output: Buffer = undefined;
    // Checked before anything is published, whatever the backend.
    try std.testing.expectError(error.InvalidBinding, loader.start(&tensor, &output, &fixture.store, .none));
    try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
    try std.testing.expectError(error.InvalidBinding, loader.wait());
}

/// Two safetensors files for composite loads, every row 16 bytes (8 u16):
/// `first` holds `a` [3, 8] then `b` [2, 8], adjacent, then `spacer`, then
/// `q` [8, 8], `k` [4, 8] and `v` [4, 8]; `second` holds `plain` [4, 8],
/// then `c` [4, 8], the experts `e{0..3}.w1` and `e{0..3}.w3` [2, 8] and
/// `wide` [64, 8]. `empty` [0, 8] is registered without bytes. Pinned after
/// `init`. Each store takes a copy of `registry` (`load_testing.storeOf`).
const CompositeFixture = struct {
    tmp: std.testing.TmpDir,
    registry: safetensors.TensorRegistry,
    a: [48]u8,
    b: [32]u8,
    c: [64]u8,
    plain: [64]u8,
    q: [128]u8,
    k: [64]u8,
    v: [64]u8,
    w1: [4][32]u8,
    w3: [4][32]u8,
    wide: [1024]u8,

    const expert_names = [2][4][]const u8{
        .{ "e0.w1", "e1.w1", "e2.w1", "e3.w1" },
        .{ "e0.w3", "e1.w3", "e2.w3", "e3.w3" },
    };

    fn init(self: *CompositeFixture, allocator: std.mem.Allocator, io: std.Io) !void {
        self.tmp = std.testing.tmpDir(.{});
        errdefer self.tmp.cleanup();
        self.registry = .init(allocator);
        errdefer self.registry.deinit();
        load_testing.pattern(&self.a, 1);
        load_testing.pattern(&self.b, 2);
        load_testing.pattern(&self.c, 3);
        load_testing.pattern(&self.plain, 4);
        load_testing.pattern(&self.q, 9);
        load_testing.pattern(&self.k, 10);
        load_testing.pattern(&self.v, 11);
        for (&self.w1, &self.w3, 0..) |*w1, *w3, e| {
            load_testing.pattern(w1, @intCast(20 + e));
            load_testing.pattern(w3, @intCast(30 + e));
        }
        var spacer: [16]u8 = undefined;
        load_testing.pattern(&spacer, 5);
        load_testing.pattern(&self.wide, 6);
        try load_testing.writeSafetensors(allocator, io, self.tmp.dir, "first.safetensors", &self.registry, &.{
            .{ .name = "a", .dtype = "U16", .dims = &.{ 3, 8 }, .bytes = &self.a },
            .{ .name = "b", .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.b },
            .{ .name = "spacer", .dtype = "U16", .dims = &.{ 1, 8 }, .bytes = &spacer },
            .{ .name = "q", .dtype = "U16", .dims = &.{ 8, 8 }, .bytes = &self.q },
            .{ .name = "k", .dtype = "U16", .dims = &.{ 4, 8 }, .bytes = &self.k },
            .{ .name = "v", .dtype = "U16", .dims = &.{ 4, 8 }, .bytes = &self.v },
        });
        try load_testing.writeSafetensors(allocator, io, self.tmp.dir, "second.safetensors", &self.registry, &.{
            .{ .name = "plain", .dtype = "U16", .dims = &.{ 4, 8 }, .bytes = &self.plain },
            .{ .name = "c", .dtype = "U16", .dims = &.{ 4, 8 }, .bytes = &self.c },
            .{ .name = expert_names[0][0], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w1[0] },
            .{ .name = expert_names[1][0], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w3[0] },
            .{ .name = expert_names[0][1], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w1[1] },
            .{ .name = expert_names[1][1], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w3[1] },
            .{ .name = expert_names[0][2], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w1[2] },
            .{ .name = expert_names[1][2], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w3[2] },
            .{ .name = expert_names[0][3], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w1[3] },
            .{ .name = expert_names[1][3], .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.w3[3] },
            .{ .name = "wide", .dtype = "U16", .dims = &.{ 64, 8 }, .bytes = &self.wide },
        });
        try self.registry.registerTensor(.{
            .file_uri = self.registry.tensors.get("a").?.file_uri,
            .name = "empty",
            .shape = .init(.{ 0, 8 }, .u16),
            .offset = 0,
        });
    }

    fn deinit(self: *CompositeFixture) void {
        self.registry.deinit();
        self.tmp.cleanup();
    }

    fn rows(view: TensorStore.View, name: []const u8) TensorStore.Source {
        return view.source(name, .{ .row, .col });
    }
};

test "loader assembles concatenated and padded sources on every placement and memory kind" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    // a, b and c end to end over 16 rows; c alone over 8 rows; b then a
    // over 6 rows. The rest is zero.
    var concatenated: [256]u8 = @splat(0);
    @memcpy(concatenated[0..48], &fixture.a);
    @memcpy(concatenated[48..80], &fixture.b);
    @memcpy(concatenated[80..144], &fixture.c);
    var padded: [128]u8 = @splat(0);
    @memcpy(padded[0..64], &fixture.c);
    var reversed: [96]u8 = @splat(0);
    @memcpy(reversed[0..32], &fixture.b);
    @memcpy(reversed[32..80], &fixture.a);

    const Source = TensorStore.Source;
    const rows = CompositeFixture.rows;
    const Model = struct { concatenated: Tensor, padded: Tensor, reversed: Tensor, plain: Tensor };
    for ([_]usize{ 2, 4 }) |device_count| {
        var devices: load_testing.Devices = undefined;
        try devices.init(allocator, io, device_count);
        defer devices.deinit();
        const platform = devices.platform;
        const tp = try devices.mesh("tp");
        var store = try load_testing.storeOf(allocator, &fixture.registry, &.{&tp});
        defer store.deinit();
        // Rows over the devices (b and c straddle shards on four), every
        // row split over them, and every byte on each of them.
        inline for (.{ .{ .row = .model }, .{ .col = .model }, .{} }) |partitioning| {
            const shards: u64 = if (@typeInfo(@TypeOf(partitioning)).@"struct".field_names.len == 0) 1 else device_count;
            for ([_]Memory.Kind{ .default, .host_pinned }) |memory| {
                if (!backend.supportsMemory(platform, memory)) continue;
                const view = store.view();
                const model: Model = .{
                    .concatenated = Source.concatenate(&.{ rows(view, "a"), rows(view, "b"), rows(view, "c") }, .row)
                        .pad(0, .{ .row = Source.Pad{ .high = 7 } })
                        .withPartitioning(.tp, partitioning).bind(),
                    // The padded engram-style table: one source.
                    .padded = rows(view, "c").pad(0, .{ .row = Source.Pad{ .high = 4 } }).withPartitioning(.tp, partitioning).bind(),
                    // Out of file order, on the platform's replicated mesh.
                    .reversed = Source.concatenate(&.{ rows(view, "b"), rows(view, "a") }, .row)
                        .pad(0, .{ .row = Source.Pad{ .high = 1 } })
                        .withPartitioning(.replicated, .{}).bind(),
                    .plain = rows(view, "plain").withPartitioning(.tp, partitioning).bind(),
                };
                inline for (comptime std.meta.fieldNames(Model)) |name| view.setMemory(@field(model, name), memory);
                try std.testing.expect(store.bindingOf(model.plain.id).?.load.source.whole() != null);
                var buffers = try mem.bufferize(allocator, Model, &model);
                defer mem.deinitBufferized(allocator, Model, &buffers);
                var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
                defer loader.deinit();
                try loader.start(&model, &buffers, &store, .none);
                try loader.wait();

                try load_testing.expectShards(allocator, io, &buffers.concatenated, &concatenated);
                try load_testing.expectShards(allocator, io, &buffers.padded, &padded);
                try load_testing.expectShards(allocator, io, &buffers.reversed, &reversed);
                try load_testing.expectShards(allocator, io, &buffers.plain, &fixture.plain);
                // Each destination once, padding included.
                try std.testing.expectEqual(@as(usize, 256 + 128 + 96 + 64), loader.bytes_loaded);
                inline for (comptime std.meta.fieldNames(Model)) |field| {
                    const buffer = &@field(buffers, field);
                    const ordered = platform.placedShape(buffer.shape())._sharding.mesh.?.devicesInCanonicalOrder();
                    for (ordered, buffer._shards.constSlice()) |device, shard| {
                        const wanted = platform.devices[device.id].memory(memory).?;
                        const default_memory = platform.devices[device.id].memory(.default).?;
                        // On CPU/neuron/metal isOfKind matches every kind, so
                        // memory(.host_pinned) is the default memory and the
                        // routing cannot be observed; CUDA/ROCm/oneAPI check it.
                        try std.testing.expect(shard.memory(platform.pjrt_api) == wanted.pjrt_memory);
                        if (memory != .default and wanted.pjrt_memory != default_memory.pjrt_memory)
                            try std.testing.expect(shard.memory(platform.pjrt_api) != default_memory.pjrt_memory);
                    }
                }
                // A destination is allocated once per device, whatever its
                // parts; host memory counts on neither side.
                if (loader.backend.? == .direct) {
                    const allocated = try allocator.alloc(u64, platform.devices.len);
                    defer allocator.free(allocated);
                    loader.backend.?.direct.allocatedBytesPerDevice(allocated);
                    try std.testing.expectEqualSlices(u64, loader.backend.?.direct.submitted_bytes, allocated);
                    for (devices.physical.devices_in_canonical_order) |device| {
                        const expected: u64 = if (memory == .host_pinned) 0 else (256 + 128 + 64) / shards + 96;
                        try std.testing.expectEqual(expected, allocated[device.id]);
                    }
                }
            }
        }
    }
}

test "loader assembles interleaved, stacked and split sources" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const Source = TensorStore.Source;
    const rows = CompositeFixture.rows;
    const row = 16;

    // Experts with their two projections concatenated, then row by row.
    var concatenated_experts: [256]u8 = undefined;
    var interleaved_experts: [256]u8 = undefined;
    for (0..4) |e| {
        @memcpy(concatenated_experts[e * 64 ..][0..32], &fixture.w1[e]);
        @memcpy(concatenated_experts[e * 64 + 32 ..][0..32], &fixture.w3[e]);
        for (0..2) |r| {
            @memcpy(interleaved_experts[e * 64 + r * 2 * row ..][0..row], fixture.w1[e][r * row ..][0..row]);
            @memcpy(interleaved_experts[e * 64 + (r * 2 + 1) * row ..][0..row], fixture.w3[e][r * row ..][0..row]);
        }
    }

    const Model = struct { qkv: Tensor, concatenated: Tensor, interleaved: Tensor, head: Tensor, tail: Tensor };
    for ([_]usize{ 2, 4 }) |device_count| {
        var devices: load_testing.Devices = undefined;
        try devices.init(allocator, io, device_count);
        defer devices.deinit();
        const platform = devices.platform;
        const tp = try devices.mesh("tp");
        var store = try load_testing.storeOf(allocator, &fixture.registry, &.{&tp});
        defer store.deinit();
        const view = store.view();
        const groups: i64 = @intCast(device_count);

        // q, k and v cut in one group per device: each device holds its rows
        // of every member, contiguous. Every part has `groups` runs.
        var members: [3]Source = undefined;
        for (&members, [_][]const u8{ "q", "k", "v" }) |*member, name| {
            member.* = rows(view, name).splitAxis(.row, .{ .group = groups, .r = -1 });
        }
        var experts: [2]Source = undefined;
        for (&experts, CompositeFixture.expert_names) |*stacked, names| {
            var per_expert: [4]Source = undefined;
            for (&per_expert, names) |*expert, name| expert.* = rows(view, name);
            stacked.* = Source.stack(&per_expert, 0, .expert);
        }
        const fused = Source.concatenate(&.{ rows(view, "q"), rows(view, "k") }, .row);
        // A part of q, and the rest of q with all of k.
        const head, const tail = fused.split(.row, &.{ 4, 8 })[0..2].*;
        const model: Model = .{
            .qkv = Source.concatenate(&members, .r).merge(.{ .row = .{ .group, .r } }).withPartitioning(.tp, .{ .row = .model }).bind(),
            // Experts are whole rows of the merged axis, one or two per device.
            .concatenated = Source.concatenate(&experts, .row)
                .merge(.{ .rows = .{ .expert, .row } })
                .withPartitioning(.tp, .{ .rows = .model }).bind(),
            .interleaved = Source.stack(&experts, 2, .proj)
                .merge(.{ .rows = .{ .expert, .row, .proj } })
                .withPartitioning(.tp, .{ .rows = .model }).bind(),
            .head = head.withPartitioning(.replicated, .{}).bind(),
            .tail = tail.withPartitioning(.tp, .{ .row = .model }).bind(),
        };
        const qkv_source = store.bindingOf(model.qkv.id).?.load.source;
        try std.testing.expectEqual(@as(usize, 3), qkv_source._pieces.len);
        for (qkv_source._pieces) |piece| {
            const part = try Source.lowerPiece(allocator, qkv_source.shape, piece);
            defer allocator.free(part.dims);
            try std.testing.expectEqual(@as(u64, device_count), part.lattice().runs());
        }

        var buffers = try mem.bufferize(allocator, Model, &model);
        defer mem.deinitBufferized(allocator, Model, &buffers);
        var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
        defer loader.deinit();
        try loader.start(&model, &buffers, &store, .none);
        try loader.wait();

        var qkv: [256]u8 = undefined;
        var at: usize = 0;
        for (0..device_count) |g| {
            for ([_][]const u8{ &fixture.q, &fixture.k, &fixture.v }) |member| {
                const per_group = member.len / device_count;
                @memcpy(qkv[at..][0..per_group], member[g * per_group ..][0..per_group]);
                at += per_group;
            }
        }
        try load_testing.expectShards(allocator, io, &buffers.qkv, &qkv);
        try load_testing.expectShards(allocator, io, &buffers.concatenated, &concatenated_experts);
        try load_testing.expectShards(allocator, io, &buffers.interleaved, &interleaved_experts);
        try load_testing.expectShards(allocator, io, &buffers.head, fixture.q[0..64]);
        var tail_bytes: [128]u8 = undefined;
        @memcpy(tail_bytes[0..64], fixture.q[64..128]);
        @memcpy(tail_bytes[64..], &fixture.k);
        try load_testing.expectShards(allocator, io, &buffers.tail, &tail_bytes);
    }
}

test "loader gathers strided sources: a row interleave across files, broadcasts and a transpose" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const Source = TensorStore.Source;
    // Rows of 16 bytes: g and u interleave row by row, 96 runs each; s0 and
    // s1 repeat 100 and 60 times; z [24, 4] is transposed, 96 runs of two
    // bytes, then padded with two zero rows. Every part has more runs than
    // expand, so the direct backend gathers them, its jobs reading g's rows
    // from one file and u's from the other.
    var g: [96 * 16]u8 = undefined;
    var u: [96 * 16]u8 = undefined;
    var z: [24 * 8]u8 = undefined;
    var s0: [4]u8 = undefined;
    var s1: [4]u8 = undefined;
    load_testing.pattern(&g, 1);
    load_testing.pattern(&u, 2);
    load_testing.pattern(&z, 3);
    load_testing.pattern(&s0, 4);
    load_testing.pattern(&s1, 5);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "first.safetensors", &registry, &.{
        .{ .name = "g", .dtype = "U16", .dims = &.{ 96, 8 }, .bytes = &g },
        .{ .name = "s0", .dtype = "F32", .dims = &.{}, .bytes = &s0 },
        .{ .name = "z", .dtype = "U16", .dims = &.{ 24, 4 }, .bytes = &z },
    });
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "second.safetensors", &registry, &.{
        .{ .name = "s1", .dtype = "F32", .dims = &.{}, .bytes = &s1 },
        .{ .name = "u", .dtype = "U16", .dims = &.{ 96, 8 }, .bytes = &u },
    });

    var interleaved: [192 * 16]u8 = undefined;
    for (0..96) |r| {
        @memcpy(interleaved[2 * r * 16 ..][0..16], g[r * 16 ..][0..16]);
        @memcpy(interleaved[(2 * r + 1) * 16 ..][0..16], u[r * 16 ..][0..16]);
    }
    var scales: [160 * 4]u8 = undefined;
    for (0..160) |i| @memcpy(scales[i * 4 ..][0..4], if (i < 100) &s0 else &s1);
    var transposed: [6 * 48]u8 = @splat(0);
    for (0..4) |column| {
        for (0..24) |row| @memcpy(transposed[column * 48 + row * 2 ..][0..2], z[row * 8 + column * 2 ..][0..2]);
    }

    const Model = struct { by_rows: Tensor, by_columns: Tensor, scales_by_rows: Tensor, scales_by_columns: Tensor, transposed: Tensor, transposed_by_columns: Tensor };
    for ([_]usize{ 1, 2, 4 }) |device_count| {
        for ([_]usize{ 0, 64 }) |alignment| {
            var devices: load_testing.Devices = undefined;
            try devices.init(allocator, io, device_count);
            defer devices.deinit();
            const tp = try devices.mesh("tp");
            var store = try load_testing.storeOf(allocator, &registry, &.{&tp});
            defer store.deinit();
            const view = store.view();
            const rows = Source.stack(&.{ view.source("g", .{ .row, .col }), view.source("u", .{ .row, .col }) }, 1, .proj)
                .merge(.{ .row = .{ .row, .proj } });
            const repeated = Source.concatenate(&.{
                view.source("s0", .{}).broad(Shape.init(.{ .c = 100 }, .f32)),
                view.source("s1", .{}).broad(Shape.init(.{ .c = 60 }, .f32)),
            }, .c).splitAxis(.c, .{ .row = 40, .col = 4 });
            const flipped = view.source("z", .{ .row, .col }).transpose(.{ .col, .row }).withTags(.{ .row, .col })
                .pad(0, .{ .row = Tensor.Pad{ .high = 2 } });
            const model: Model = .{
                .by_rows = rows.withPartitioning(.tp, .{ .row = .model }).bind(),
                .by_columns = rows.withPartitioning(.tp, .{ .col = .model }).bind(),
                .scales_by_rows = repeated.withPartitioning(.tp, .{ .row = .model }).bind(),
                .scales_by_columns = repeated.withPartitioning(.tp, .{ .col = .model }).bind(),
                .transposed = flipped.withPartitioning(.replicated, .{}).bind(),
                .transposed_by_columns = flipped.withPartitioning(.tp, .{ .col = .model }).bind(),
            };
            var buffers = try mem.bufferize(allocator, Model, &model);
            defer mem.deinitBufferized(allocator, Model, &buffers);
            var loader = try Loader.init(allocator, io, devices.platform, .{ .read_parallelism = 2 });
            defer loader.deinit();
            // Jobs of 128 bytes: every destination takes several.
            try loader.awaitSetup();
            switch (loader.backend.?) {
                .direct => |direct| {
                    direct.plan_config.request_size = 256;
                    direct.plan_config.alignment = alignment;
                },
                .buffered => {},
            }
            try loader.start(&model, &buffers, &store, .none);
            try loader.wait();
            try load_testing.expectShards(allocator, io, &buffers.by_rows, &interleaved);
            try load_testing.expectShards(allocator, io, &buffers.by_columns, &interleaved);
            try load_testing.expectShards(allocator, io, &buffers.scales_by_rows, &scales);
            try load_testing.expectShards(allocator, io, &buffers.scales_by_columns, &scales);
            try load_testing.expectShards(allocator, io, &buffers.transposed, &transposed);
            try load_testing.expectShards(allocator, io, &buffers.transposed_by_columns, &transposed);
        }
    }
}

test "bindings the loader cannot fill fail it before submission" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    // Three nibbles: whole bytes hold no element boundary past the first two.
    try fixture.registry.registerTensor(.{
        .file_uri = fixture.registry.tensors.get("a").?.file_uri,
        .name = "nibbles",
        .shape = .init(.{3}, .u4),
        .offset = 0,
    });
    const platform = testing.env();
    var store = try load_testing.storeOf(allocator, &fixture.registry, &.{platform.replicated_mesh});
    defer store.deinit();
    const Source = TensorStore.Source;
    const view = store.view();
    const rows = CompositeFixture.rows;

    const rejected = [_]Tensor{
        // Nothing to read: an empty tensor padded.
        rows(view, "empty").pad(0, .{ .row = Source.Pad{ .high = 8 } }).withPartitioning(.replicated, .{}).bind(),
        // Twelve bits: its one run ends inside a byte (since R10; no known
        // checkpoint has one).
        view.createReplicatedTensor("nibbles", null),
        // Strided parts into host memory.
        host: {
            const tensor = rows(view, "wide").transpose(.{}).withPartitioning(.replicated, .{}).bind();
            view.setMemory(tensor, .host_pinned);
            break :host tensor;
        },
    };
    var output: Buffer = undefined;
    for (&rejected) |*tensor| {
        var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
        defer loader.deinit();
        try std.testing.expectError(error.InvalidBinding, loader.start(tensor, &output, &store, .none));
        try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
    }
}

test "loader fills gaps, reordered and strided rows, and a transpose" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = testing.env();
    var store = try load_testing.storeOf(allocator, &fixture.registry, &.{platform.replicated_mesh});
    defer store.deinit();
    const Source = TensorStore.Source;
    const Pad = Source.Pad;
    const view = store.view();
    const rows = CompositeFixture.rows;
    const a = rows(view, "a");

    // b's second row past a gap, the rest zero; b then a, each padded; a's
    // last row before its first two, one source feeding two parts; a's
    // rows every other row, a strided part of few runs, expanded; and a
    // transpose, 512 runs of two bytes, gathered.
    const Model = struct { sparse: Tensor, padded: Tensor, swapped: Tensor, strided: Tensor, transposed: Tensor };
    const model: Model = .{
        .sparse = rows(view, "b").slice(.row, .{ .start = 1 }).pad(0, .{ .row = Pad{ .low = 4, .high = 3 } }).withPartitioning(.replicated, .{}).bind(),
        .padded = Source.concatenate(&.{
            rows(view, "b").pad(0, .{ .row = Pad{ .high = 2 } }),
            a.pad(0, .{ .row = Pad{ .high = 1 } }),
        }, .row).withPartitioning(.replicated, .{}).bind(),
        .swapped = Source.concatenate(&.{ a.slice(.row, .{ .start = 2 }), a.slice(.row, .{ .end = 2 }) }, .row)
            .pad(0, .{ .row = Pad{ .high = 5 } }).withPartitioning(.replicated, .{}).bind(),
        .strided = a.pad(0, .{ .row = Pad{ .interior = 1, .high = 3 } }).withPartitioning(.replicated, .{}).bind(),
        .transposed = rows(view, "wide").transpose(.{}).withPartitioning(.replicated, .{}).bind(),
    };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    try loader.start(&model, &buffers, &store, .none);
    try loader.wait();

    var expected: [128]u8 = @splat(0);
    @memcpy(expected[64..80], fixture.b[16..32]);
    try load_testing.expectShards(allocator, io, &buffers.sparse, &expected);
    expected = @splat(0);
    @memcpy(expected[0..32], &fixture.b);
    @memcpy(expected[64..112], &fixture.a);
    try load_testing.expectShards(allocator, io, &buffers.padded, &expected);
    expected = @splat(0);
    @memcpy(expected[0..16], fixture.a[32..48]);
    @memcpy(expected[16..48], fixture.a[0..32]);
    try load_testing.expectShards(allocator, io, &buffers.swapped, &expected);
    expected = @splat(0);
    for (0..3) |i| @memcpy(expected[i * 32 ..][0..16], fixture.a[i * 16 ..][0..16]);
    try load_testing.expectShards(allocator, io, &buffers.strided, &expected);
    var transposed_bytes: [1024]u8 = undefined;
    for (0..8) |column| {
        for (0..64) |row| @memcpy(transposed_bytes[column * 128 + row * 2 ..][0..2], fixture.wide[row * 16 + column * 2 ..][0..2]);
    }
    try load_testing.expectShards(allocator, io, &buffers.transposed, &transposed_bytes);
}

test "a program takes its inputs from sources, in order" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = testing.env();
    var store = try load_testing.storeOf(allocator, &fixture.registry, &.{platform.replicated_mesh});
    defer store.deinit();
    const Source = TensorStore.Source;
    const view = store.view();

    // a then b, assembled by the loader, and c as it is: the program adds
    // c to the first four rows, the way an FP8 dequantization takes a
    // fused weight and its scales.
    const AddRows = struct {
        pub fn forward(_: @This(), _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0].slice(0, .{ .end = 4 }).add(inputs[1]);
        }
    };
    const assembled = Source.concatenate(&.{ CompositeFixture.rows(view, "a"), CompositeFixture.rows(view, "b") }, .row);
    const c = CompositeFixture.rows(view, "c");
    const inputs = [_]Source{ assembled.withPartitioning(.replicated, .{}), c.withPartitioning(.replicated, .{}) };
    const tensor = Source.bindProgram(&inputs, c.shape(), AddRows{});
    const compute = store.bindingOf(tensor.id).?.compute;
    try std.testing.expectEqual(2, compute.inputs.len);
    try std.testing.expect(compute.program.as(AddRows) != null and compute.program.as(LoaderTestFixture.Identity) == null);
    try std.testing.expect(compute.inputs[0].whole() == null);
    try std.testing.expect(compute.inputs[1].whole() != null);
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var output: Buffer = undefined;
    try loader.start(&tensor, &output, &store, .none);
    try loader.wait();
    defer output.deinit();
    var expected: [64]u8 = undefined;
    for (0..32) |i| {
        const row = if (i < 24) fixture.a[i * 2 ..][0..2] else fixture.b[(i - 24) * 2 ..][0..2];
        const value = std.mem.readInt(u16, row, .little) +% std.mem.readInt(u16, fixture.c[i * 2 ..][0..2], .little);
        std.mem.writeInt(u16, expected[i * 2 ..][0..2], value, .little);
    }
    try load_testing.expectShards(allocator, io, &output, &expected);
}

test "a padded host-pinned table stays in host memory and serves device gathers through device views" {
    // The engram arrangement: one source of 264-byte rows, an odd row count
    // padded to a multiple of 8, rows sharded over every device, delivered
    // into host pinned memory. A program then reads it through device-kind
    // views of that memory (`Buffer.createView`), as llmd does, and gathers
    // rows from each device's local shard.
    try expectHostTableGathers(.rows);
}

test "a replicated host-pinned table is one allocation that every device reads through device views" {
    // The same table on every device: where the platform shares host memory
    // across devices (`direct_loader.sharesHostMemory`), the loader makes it
    // one allocation, each device's shard a view of it, and every device's
    // kernels gather from it.
    try expectHostTableGathers(.replicated);
}

/// Loads a padded table into host pinned memory, rows sharded over every
/// device or replicated on each, wraps it in device-kind views and gathers
/// rows of each device's shard on that device.
fn expectHostTableGathers(placement: enum { rows, replicated }) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const rows: usize = 100_003;
    const row_bytes: usize = 264;
    const gathered: usize = 256;

    // The accelerator under test, or four CPU devices.
    const shared = testing.env();
    const owned: ?*Platform = if (shared.target == .cpu)
        try Platform.init(allocator, io, .cpu, .{ .cpu = .{ .device_count = 4 } })
    else
        null;
    defer if (owned) |platform| platform.deinit(allocator, io);
    const platform: *const Platform = if (owned) |p| p else shared;
    const mesh: *const Sharding.Mesh = if (owned) |p|
        try p.registerMesh("model", .mesh(.{ .model = .high_bandwidth }))
    else
        shared.meshes.get("model").?;
    if (!backend.supportsMemory(platform, .host_pinned)) return error.SkipZigTest;
    const devices = mesh.devicesInCanonicalOrder();
    const shard_count = devices.len;
    // Host memory is distinct from device memory on CUDA/ROCm/oneAPI only;
    // elsewhere every kind resolves to the one memory and the residency
    // checks cannot observe anything.
    const distinct_memory = platform.devices[devices[0].id].memory(.host_pinned).?.pjrt_memory !=
        platform.devices[devices[0].id].memory(.default).?.pjrt_memory;

    const padded_rows = std.mem.alignForward(usize, rows, 8);
    const shard_rows = if (placement == .rows) padded_rows / shard_count else padded_rows;
    if (placement == .rows) try std.testing.expectEqual(padded_rows, shard_rows * shard_count);
    const padded = try allocator.alloc(u8, padded_rows * row_bytes);
    defer allocator.free(padded);
    @memset(padded, 0);
    for (0..rows) |row| {
        const bytes = padded[row * row_bytes ..][0..row_bytes];
        // The row's number first, so a misplaced row cannot pass for another.
        std.mem.writeInt(u32, bytes[0..4], @intCast(row + 1), .little);
        for (bytes[4..], 4..) |*byte, column| byte.* = @as(u8, @truncate(row *% 131 +% column *% 7)) | 1;
    }

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var store: TensorStore = store: {
        var registry: safetensors.TensorRegistry = .init(allocator);
        errdefer registry.deinit();
        try load_testing.writeSafetensors(allocator, io, tmp.dir, "table.safetensors", &registry, &.{
            .{ .name = "table", .dtype = "U8", .dims = &.{ @intCast(rows), @intCast(row_bytes) }, .bytes = padded[0 .. rows * row_bytes] },
        });
        break :store .init(allocator, registry, .{ .meshes = &.{mesh} });
    };
    defer store.deinit();
    const source = store.view().source("table", .{ .row, .channel })
        .pad(0, .{ .row = TensorStore.Source.Pad{ .high = @intCast(padded_rows - rows) } });
    const table = switch (placement) {
        .rows => source.withPartitioning(.model, .{ .row = .model }).bind(),
        .replicated => source.withPartitioning(.model, .{}).bind(),
    };
    store.view().setMemory(table, .host_pinned);
    try std.testing.expectEqual(@as(i64, @intCast(padded_rows)), table.dim(.row));

    var in_use_before: [Platform.MAX_NUM_DEVICES]u64 = undefined;
    if (distinct_memory) for (devices) |device| {
        in_use_before[device.id] = platform.devices[device.id].memoryStats().bytes_in_use;
    };

    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 4 });
    defer loader.deinit();
    var host_table: Buffer = undefined;
    try loader.start(&table, &host_table, &store, .none);
    try loader.wait();
    defer host_table.deinit();
    try std.testing.expectEqual(padded.len, loader.bytes_loaded);
    try load_testing.expectShards(allocator, io, &host_table, padded);

    // Device-kind views of the pinned shards; `host_table` owns the memory
    // and outlives them.
    var device_table = try host_table.createView();
    defer device_table.deinit();
    const shard_bytes = shard_rows * row_bytes;
    const first_pointer = try host_table._shards.get(0).opaqueDeviceMemoryDataPointer(platform.pjrt_api);
    for (devices, host_table._shards.constSlice(), device_table._shards.constSlice()) |device, owner, view| {
        // Replicas share one allocation where the platform shares host
        // memory across devices; row shards never do.
        const pointer = try owner.opaqueDeviceMemoryDataPointer(platform.pjrt_api);
        const shares = placement == .replicated and shard_count > 1 and switch (platform.target) {
            .cpu, .cuda, .rocm, .oneapi => true,
            else => false,
        };
        if (device.id != devices[0].id) try std.testing.expectEqual(shares, pointer == first_pointer);
        const pinned_memory = platform.devices[device.id].memory(.host_pinned).?.pjrt_memory;
        const default_memory = platform.devices[device.id].memory(.default).?.pjrt_memory;
        try std.testing.expect(owner.memory(platform.pjrt_api) == pinned_memory);
        try std.testing.expect(view.memory(platform.pjrt_api) == default_memory);
        try std.testing.expectEqual(
            try owner.opaqueDeviceMemoryDataPointer(platform.pjrt_api),
            try view.opaqueDeviceMemoryDataPointer(platform.pjrt_api),
        );
        // The table took no device memory.
        if (distinct_memory) {
            const in_use = platform.devices[device.id].memoryStats().bytes_in_use;
            try std.testing.expect(in_use -| in_use_before[device.id] < shard_bytes / 4);
        }
    }

    // Each device gathers `gathered` rows of its own shard, the padding
    // rows of the last one included (of every replica, replicated).
    const Gather = struct {
        fn forward(table_: Tensor, indices: Tensor) Tensor {
            const output: Shape = Shape.init(.{ .shard = indices.dim(.shard), .bytes = indices.dim(.k) * table_.dim(.channel) }, .u8)
                .withPartitioning(table_.shape()._sharding.mesh.?, .{ .shard = .model });
            return ops.manualComputation((struct {
                fn body(inputs: struct { table: Tensor, indices: Tensor }, local: Shape) Tensor {
                    return inputs.table.gather(.{ .row = inputs.indices }, .{}).reshape(local);
                }
            }).body, .{ .table = table_, .indices = indices }, output);
        }
    };
    const indices_shape = Shape.init(.{ .shard = shard_count, .k = gathered }, .u32).withPartitioning(mesh, .{ .shard = .model });
    const indices = try allocator.alloc(u32, shard_count * gathered);
    defer allocator.free(indices);
    for (0..shard_count) |shard| {
        const local = indices[shard * gathered ..][0..gathered];
        for (local, 0..) |*index, k| index.* = @intCast((k * 7919 + shard * 104_729) % shard_rows);
        // The first and last rows of every shard; on the last device these
        // are padding.
        local[0] = 0;
        local[1] = @intCast(shard_rows - 1);
        local[2] = @intCast(shard_rows - 2);
    }
    var indices_buffer: Buffer = try .from(io, platform, indices_shape, std.mem.sliceAsBytes(indices), .{});
    defer indices_buffer.deinit();

    const exe = try platform.compileFn(allocator, io, Gather.forward, .{ Tensor.fromShape(table.shape()), Tensor.fromShape(indices_shape) }, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ device_table, indices_buffer });
    try exe.call(io, args, &results, .{ .wait = true });
    var output = results.get(Buffer);
    defer output.deinit();

    const expected = try allocator.alloc(u8, shard_count * gathered * row_bytes);
    defer allocator.free(expected);
    for (0..shard_count) |shard| {
        for (0..gathered) |k| {
            const first_row = if (placement == .rows) shard * shard_rows else 0;
            const row = first_row + indices[shard * gathered + k];
            @memcpy(expected[(shard * gathered + k) * row_bytes ..][0..row_bytes], padded[row * row_bytes ..][0..row_bytes]);
        }
    }
    try load_testing.expectShards(allocator, io, &output, expected);

    // The control: the same table in device memory does show up in the
    // devices' usage, so the residency check above can fail.
    if (!distinct_memory) return;
    const device_copy = switch (placement) {
        .rows => source.withPartitioning(.model, .{ .row = .model }).bind(),
        .replicated => source.withPartitioning(.model, .{}).bind(),
    };
    var in_use_control: [Platform.MAX_NUM_DEVICES]u64 = undefined;
    for (devices) |device| in_use_control[device.id] = platform.devices[device.id].memoryStats().bytes_in_use;
    var device_buffer: Buffer = undefined;
    try loader.start(&device_copy, &device_buffer, &store, .none);
    try loader.wait();
    defer device_buffer.deinit();
    for (devices) |device| {
        const in_use = platform.devices[device.id].memoryStats().bytes_in_use;
        // Zero: the plugin reports no usage, and neither check means anything.
        if (in_use == 0) continue;
        try std.testing.expect(in_use -| in_use_control[device.id] >= shard_bytes);
    }
}
