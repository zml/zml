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
//! It lowers each bound source to bytes (`Source.Placed.lower`, the one
//! rule), one part per piece: a part of the destination's logical bytes
//! (the row-major bytes of its shape, before the sharding splits them), a
//! run of `width` bytes repeated over strided dimensions. A whole
//! checkpoint tensor is one part.
//! A strided part of at most `max_part_runs` runs, or of runs of at least
//! `large_run_bytes`, becomes one part per run; a spec with any other
//! strided part (row interleaves, transposes, broadcasts) goes to the
//! backend as strided parts, which the direct backend gathers. Each
//! `LoadSpec` is one destination: its placed shape, where it goes, the
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
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Compiler = @import("../Compiler.zig");
const Exe = @import("../exe.zig").Exe;
const mem = @import("../mem.zig");
const Bufferized = mem.Bufferized;
const meta = @import("../meta.zig");
const ops = @import("../ops.zig");
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
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const dma_calibration = @import("dma_calibration.zig");
const load_log = @import("log.zig").load;
const TensorStore = @import("TensorStore.zig");

const PrepareError = std.mem.Allocator.Error || error{InvalidBinding};

/// Bytes per device, by `platform.devices` index: `Platform.init` refuses
/// more devices than it holds.
const DeviceBytes = [Platform.MAX_NUM_DEVICES]u64;

/// Loads models from their checkpoint sources, and computes the tensors
/// programs compute from theirs. A value: `init` returns at once.
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
///
/// What fails. Every error of `start` and `wait` fails the loader:
/// everything started is canceled (`cancel`), and every later call returns
/// the first error. Each is logged once, at err, where it is detected, but
/// for a plain allocation failure (`OutOfMemory` from an allocator) and
/// `Canceled`, the caller's own `cancel`, which are returned unlogged. A
/// model whose bindings are wrong is `InvalidBinding`, logged with the
/// tensor, the key its source reads (none for a source that reads nothing
/// or a run inside a byte) and the rule it breaks; buffers shaped unlike
/// the model (`mem.pairBufferized`) too, logged with the field path. The
/// other errors a caller can act on: a missing checkpoint file is
/// `FileNotFound`; a denied one `AccessDenied` locally and
/// `PermissionDenied` remotely; a file that ends inside a tensor
/// `UnexpectedEndOfFile`, because it changed after its header was read or
/// the tensor's shape needs more bytes than its `data_offsets` give; host
/// memory the backend cannot allocate, pin or fit under its mapped ceiling
/// `OutOfMemory`; device memory that runs out PJRT's `ResourceExhausted`;
/// an io that cannot run the loader's tasks concurrently
/// `ConcurrencyUnavailable`; `cancel` `Canceled`. The rest
/// (PJRT, the compiler, a remote that failed) passes through as its source
/// names it.
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
    /// compiled once, on a task of its own.
    programs: std.ArrayList(*CompiledProgram) = .empty,
    /// Program groups started and not yet published, oldest first.
    queued: std.Deque(Group) = .empty,
    /// Submissions published and not yet retired, oldest first.
    pending: std.Deque(PendingSubmission) = .empty,
    /// The first error; every later call returns it.
    failure: ?anyerror = null,
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
    submissions: usize = 0,
    /// Submissions retired to admit a program group, bulk loads included.
    admission_retires: usize = 0,
    /// How long publishing waited for compiles since the last `wait`.
    compile_wait_ns: u64 = 0,
    min_room_seen: ?u64 = null,
    oversized_logged: bool = false,
    temp_unavailable_logged: bool = false,

    pub const Options = struct {
        /// The checkpoint's path or URI, read by `init` only. When `io` is a
        /// VFS, the VFS backend serving that path sizes the reads
        /// (`VFS.readHints`). Null, or a plain `io`: the local sizing.
        checkpoint: ?[]const u8 = null,
        /// Concurrent source reads, 1 to `max_read_parallelism`. Null takes
        /// the checkpoint's default, 16 locally and 32 on a high-latency
        /// source. The direct backend clips it so `width + 1` requests fit
        /// under the mapped ceiling beside its DMA reserve. Fixed for the
        /// load: a source that rate limits holds the VFS instead
        /// (`vfs/request.zig`).
        read_parallelism: ?usize = null,
        /// The DMA block sizes calibration screens: 1 to
        /// `backend.max_dma_block_sizes` of them, copied by `init`, each at
        /// least the direct I/O alignment (`VFS.direct_io_alignment`, 4 KiB,
        /// when `io` is a VFS and `direct_io` is not `off`; else 1) and at
        /// most 32 MiB (`backend.max_read_request_size`). With that
        /// alignment, a request (the larger of a block and the checkpoint's
        /// read chunk) exceeds two units. Ignored by the buffered backend;
        /// CPU takes the default sizing without measuring.
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
        /// Output bytes of the program tensors one group publishes at most. A
        /// group's inputs are planned together, so the weights and scales of
        /// one layer's experts, adjacent in the checkpoint, coalesce into
        /// full requests instead of one read per expert scale; they are freed
        /// once the group's programs ran.
        program_group_bytes: usize = 8 << 30,

        /// The default read widths by latency class, from the recorded width
        /// sweeps. Local sources knee at 12 to 16 on every host measured: the
        /// B70 peaks at 12 to 16 and loses 11% at 24, one MI300X prefers 12,
        /// the RTX 5090 host 16 over 12, and the GB300 plateau from 12 to 48
        /// has 16 within 5% of its best rung, so 16 is the width with the
        /// smallest regret. Remote sources plateau from 24 to 32: hf:// reads
        /// 32 and 64 alike and AWS was flat from 24, while pinned memory grows
        /// with the width (32 x 32 MiB requests pin 1.5 GiB).
        const local_read_parallelism: usize = 16;
        const high_latency_read_parallelism: usize = 32;
    };

    /// Concurrent source reads at most (`Options.read_parallelism`).
    pub const max_read_parallelism: usize = 128;

    /// Checks `opts` and returns at once. Warming the device allocators and
    /// creating the backend (for the direct one: DMA calibration and the
    /// pinned pre-growth) run on a task of the loader's own, or here when
    /// `io` has no concurrency; the first `start` waits for that task and
    /// returns its error. A read width or a block size out of range, an
    /// empty or too long block list, a checkpoint whose scheme the VFS does
    /// not serve, a read chunk that a block calibration may select does not
    /// divide, or a request too short for a widened direct read is
    /// `InvalidOptions`, logged.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) error{InvalidOptions}!Loader {
        const backend_opts = try backendOptions(io, opts);
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

    /// Waits for `init`'s task the first time: its backend, then whether
    /// admission can measure the devices; or its error, for the caller to
    /// fail the loader with.
    fn awaitSetup(self: *Loader) !void {
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
        if (self.loadsStarted()) self.fail(error.Canceled);
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
    /// mesh), into the memory its binding names, assembled from the
    /// pieces of its source (`Source.Placed.bind`) with every byte no piece
    /// covers set to zero. All of these are published here, as one planned
    /// submission. A tensor a program computes (`Source.bindProgram`) starts
    /// its program compiling, once for every tensor of an equal program
    /// over the same placed shapes; `wait` lands its inputs and runs it.
    ///
    /// Every tensor and program input is checked once, before anything is
    /// published: a tensor `store` did not bind, a source that reads no
    /// bytes (an empty checkpoint tensor among them), a source that does
    /// not fill its tensor's bytes or that this loader cannot assemble (a
    /// run that starts or ends inside a byte, strided parts into host
    /// memory), a placement its mesh does not divide or on devices the
    /// platform lacks, and a host kind the platform cannot take are
    /// `error.InvalidBinding`. On the direct backend, a broadcast or
    /// transpose whose source range does not fit one read is
    /// `InvalidBinding` when its submission is published: here for the
    /// tensors the loader fills, in `wait` for a program's inputs.
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
        if (self.started_at == null) {
            self.started_at = .now(self.io, .awake);
            self.bytes_loaded = 0;
        }
        var prepared = try prepareModel(self.allocator, self.platform, store, model, buffers);
        errdefer prepared.arena.deinit();
        // The backends complete one item per spec: one per tensor the loader
        // fills and, in the groups, one per program input.
        var items = prepared.specs.len;
        for (prepared.computed) |item| items += item.compute.inputs.len;
        progress.increaseEstimatedTotalItems(items);
        // Queued first: their programs compile while the rest is read.
        try self.queueGroups(store, prepared.computed, progress);
        if (prepared.specs.len == 0) {
            prepared.arena.deinit();
            return;
        }
        try self.submit(prepared.specs, prepared.arena, &.{}, @splat(0), progress);
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
    pub fn wait(self: *Loader) !void {
        if (self.failure) |err| return err;
        self.drain() catch |err| {
            self.fail(err);
            return err;
        };
        self.releasePrograms();
    }

    fn drain(self: *Loader) !void {
        var groups: usize = 0;
        var tensors: usize = 0;
        while (self.queued.popFront()) |group| {
            tensors += group.executables.len;
            try self.publish(group);
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
    /// checked as `start` checks every spec (`prepareSpec`).
    fn prepareGroup(self: *Loader, store: *const TensorStore, computed: []const Computed, progress: std.Progress.Node) PrepareError!Group {
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

    /// Publishes `group` once its programs compiled and the devices have
    /// room for it: its executables and its specs' arena move to the
    /// pending submission, or are freed when this fails.
    fn publish(self: *Loader, group: Group) !void {
        errdefer group.deinit(self.allocator);
        const started: std.Io.Timestamp = .now(self.io, .awake);
        for (group.executables) |executable| try executable.program.await(self.io, self.platform);
        self.compile_wait_ns +|= @intCast(started.untilNow(self.io, .awake).toNanoseconds());
        const execution = try self.admit(group.executables);
        try self.submit(group.specs, group.arena, group.executables, execution, group.progress);
    }

    /// Retires pending submissions, oldest first, until the group of
    /// `executables` fits the measured room or no program group is pending:
    /// a bulk load frees nothing, and once nothing can be retired for it the
    /// group is admitted whatever its size. Without memory measurement one
    /// program group is in flight at a time. Returns the group's output and
    /// temporary bytes per device, zero while unmeasured.
    fn admit(self: *Loader, executables: []const BoundExecutable) !DeviceBytes {
        var inputs: DeviceBytes = @splat(0);
        var execution: DeviceBytes = @splat(0);
        var room: ?DeviceBytes = null;
        if (self.memory_supported) {
            var temporaries: DeviceBytes = @splat(0);
            for (executables) |executable| {
                const exe = &executable.program.exe.?;
                for (exe.input_shapes) |shape| addPlacementBytes(&inputs, self.platform, shape);
                // The output's memory space is the program's to choose; it is
                // charged to the devices like the default space.
                addPlacementBytes(&execution, self.platform, exe.output_shapes[0]);
                // The group's programs run one after the other, so only the
                // largest temporaries are live at once.
                const temp = self.compiledTempBytes(exe);
                const output_mesh = self.platform.placedShape(exe.output_shapes[0])._sharding.mesh.?;
                for (output_mesh.devicesInCanonicalOrder()) |device| temporaries[device.id] = @max(temporaries[device.id], temp);
            }
            for (&execution, temporaries) |*bytes, temp| bytes.* +|= temp;
            room = self.readRoom();
        }
        var retired: usize = 0;
        while (room == null or !self.fits(&room.?, &inputs, &execution)) : (retired += 1) {
            if (!self.programGroupPending()) {
                if (room != null and !self.oversized_logged) {
                    self.oversized_logged = true;
                    load_log.warn("program group exceeds the device room alone: inputs={Bi:.2} execution={Bi:.2} room={Bi:.2} (device 0); admitted anyway", .{
                        inputs[0],
                        execution[0],
                        room.?[0],
                    });
                }
                break;
            }
            try self.retireOldest(true);
            if (self.memory_supported) room = self.readRoom();
        }
        self.admission_retires += retired;
        if (retired != 0) {
            // Unmeasured, the group's bytes are not computed: only the retires are logged.
            if (self.memory_supported) {
                load_log.debug("execute admission: retired={d} before submission {d}: measured=true, device 0 room={Bi:.2} pending_execution={Bi:.2} inputs={Bi:.2} execution={Bi:.2}", .{
                    retired,
                    self.submissions,
                    if (room) |measured| measured[0] else 0,
                    self.pendingExecution()[0],
                    inputs[0],
                    execution[0],
                });
            } else {
                load_log.debug("execute admission: retired={d} before submission {d}: measured=false", .{ retired, self.submissions });
            }
        }
        return execution;
    }

    /// Whether a group of `inputs` and `execution` fits beside the pending
    /// program groups in `room`, on every device.
    fn fits(self: *const Loader, room: *const DeviceBytes, inputs: *const DeviceBytes, execution: *const DeviceBytes) bool {
        const n = self.platform.devices.len;
        const pending = self.pendingExecution();
        return admission.admits(room[0..n], pending[0..n], inputs[0..n], execution[0..n]);
    }

    /// Output and temporary bytes of the pending program groups, per
    /// device: what retiring them will take before it frees anything.
    fn pendingExecution(self: *const Loader) DeviceBytes {
        var sum: DeviceBytes = @splat(0);
        var it = self.pending.iterator();
        while (it.nextPtr()) |submission| {
            for (&sum, submission.execution) |*bytes, more| bytes.* +|= more;
        }
        return sum;
    }

    /// Whether a program group is pending: only retiring one frees room
    /// for the next (a bulk load's buffers stay).
    fn programGroupPending(self: *const Loader) bool {
        var it = self.pending.iterator();
        while (it.nextPtr()) |submission| {
            if (submission.executables.len != 0) return true;
        }
        return false;
    }

    /// The room each device reports now; null when one reports no memory
    /// limit. Only called when `memory_supported`, which implies the direct
    /// backend (`probeMemory`).
    fn readRoom(self: *Loader) ?DeviceBytes {
        const n = self.platform.devices.len;
        var unlanded: DeviceBytes = undefined;
        self.backend.?.direct.unlandedBytesPerDevice(unlanded[0..n]);
        var room: DeviceBytes = undefined;
        for (room[0..n], self.platform.devices, unlanded[0..n]) |*out, device, still_to_land| {
            const reported = device.memoryStats();
            out.* = admission.room(reported.bytes_limit, reported.bytes_in_use, still_to_land, admission.reserve_bytes) orelse return null;
            self.min_room_seen = @min(self.min_room_seen orelse out.*, out.*);
        }
        return room;
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

    /// One submission over `specs`, which live in `arena`. Takes `arena` and
    /// `executables` once the submission is published; on failure the caller
    /// still owns them. `execution` is a program group's output and
    /// temporary bytes per device (`admit`), zero for a bulk load.
    fn submit(
        self: *Loader,
        specs: []const LoadSpec,
        arena: std.heap.ArenaAllocator,
        executables: []BoundExecutable,
        execution: DeviceBytes,
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
            .executables = executables,
            .execution = execution,
            .logical_bytes = logical_bytes,
        });
        self.submissions += 1;
    }

    /// Retires the oldest pending submission: waits for its reads and DMA,
    /// runs its programs when `execute`, frees its inputs either way and
    /// counts its bytes once they ran.
    fn retireOldest(self: *Loader, execute: bool) !void {
        var oldest = self.pending.popFront().?;
        defer self.release(&oldest);
        try self.backend.?.await(oldest.submission);
        if (!execute) return;
        for (oldest.executables) |*executable| {
            executable.execute(self.allocator, self.io) catch |err| {
                load_log.failure(err, "run program {s}: inputs={d}, output={f}", .{ executable.program.compute.program.name(), executable.inputs.len, executable.program.output });
                return err;
            };
        }
        self.bytes_loaded += oldest.logical_bytes;
    }

    fn release(self: *Loader, oldest: *PendingSubmission) void {
        oldest.arena.deinit();
        for (oldest.executables) |*executable| executable.deinit(self.allocator);
        self.allocator.free(oldest.executables);
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
    /// The specs, which the backend reads until the submission is awaited
    /// (`backend.LoadSpec`).
    arena: std.heap.ArenaAllocator,
    /// Run in order once the reads are done; freed with their inputs when
    /// the submission is retired. Empty for a bulk load.
    executables: []BoundExecutable,
    /// Output and temporary bytes per device of a program group, while
    /// admission measures (`admit`); zero otherwise and for a bulk load.
    execution: DeviceBytes,
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
        const name = self.compute.program.name();
        self.exe = self.future.await(io) catch |err| {
            load_log.failure(err, "compile program {s} computing {f}", .{ name, self.output });
            return err;
        };
        try checkOutput(platform, name, self.output, &self.exe.?);
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

/// One tensor of a program group: its program, the input shells the
/// group's submission fills, and the model's buffer it computes.
const BoundExecutable = struct {
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
    fn execute(self: *const BoundExecutable, allocator: std.mem.Allocator, io: std.Io) !void {
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
fn addPlacementBytes(out: *DeviceBytes, platform: *const Platform, shape: Shape) void {
    const placed = platform.placedShape(shape).packedShape();
    const placement = Sharding.Placement.init(&placed) catch |err|
        std.debug.panic("loader: {f} does not place, though checked: {s}", .{ shape, @errorName(err) });
    const bytes: u64 = placement.localShape(placed).byteSize();
    for (placement.mesh().?.devicesInCanonicalOrder()) |device| out[device.id] += bytes;
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
    // Tensors and buffers pair by structure, before any buffer is written: a
    // buffer tree shaped otherwise is refused whole.
    var report_buffer: [256]u8 = undefined;
    var report: std.Io.Writer = .fixed(&report_buffer);
    const pairs = mem.pairBufferized(allocator, model, buffers, &report) catch |err| switch (err) {
        error.Mismatch => {
            load_log.err("buffers shaped unlike the model at model{s}; bufferize them from this model", .{report.buffered()});
            return error.InvalidBinding;
        },
        error.OutOfMemory => return error.OutOfMemory,
    };
    defer allocator.free(pairs);
    // Empty until a backend or a program writes it whole, so the caller can
    // free the tree whatever fails.
    for (pairs) |pair| pair.buffer.* = .{ ._platform = platform, ._shape = pair.tensor.shape(), ._shards = .empty };

    var arena: std.heap.ArenaAllocator = .init(allocator);
    errdefer arena.deinit();
    var specs: std.ArrayList(LoadSpec) = try .initCapacity(arena.allocator(), pairs.len);
    var computed: std.ArrayList(Computed) = try .initCapacity(arena.allocator(), pairs.len);
    for (pairs) |pair| {
        const tensor = pair.tensor;
        const binding = store.bindingOf(tensor.id) orelse {
            load_log.err("tensor {} {f} has no checkpoint binding", .{ tensor.id, tensor.shape() });
            return error.InvalidBinding;
        };
        const target: Target = .{ .id = tensor.id, .shape = tensor.shape() };
        switch (binding) {
            .load => |load| specs.appendAssumeCapacity(try prepareSpec(arena.allocator(), platform, target, load.source, load.memory, pair.buffer)),
            .compute => |compute| {
                try checkPlacement(platform, target, compute.program.name());
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

fn expandable(lattice: backend.Lattice) bool {
    return lattice.dims.len == 0 or lattice.runs() <= max_part_runs or lattice.width >= large_run_bytes;
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
/// `memory`, its parts in `arena`: the source lowered to bytes
/// (`Source.Placed.lower`), expanded into its runs unless a part does not
/// expand, in destination order. The one check of what the backends trust
/// (`backend.LoadSpec`); a rule it breaks is `InvalidBinding`, logged with
/// the target, and with the key its source reads once the source is
/// lowered.
fn prepareSpec(
    arena: std.mem.Allocator,
    platform: *const Platform,
    target: Target,
    source: TensorStore.Source.Placed,
    memory: Memory,
    output: *Buffer,
) PrepareError!LoadSpec {
    const shape = target.destination();
    const lowered = source.lower(arena) catch |err| switch (err) {
        error.OutOfMemory => |e| return e,
        error.SubBytePosition => {
            load_log.err("{f}: its source {f} reads a run that starts or ends inside a byte, which the loader cannot place", .{ target, source.shape });
            return error.InvalidBinding;
        },
    };
    if (lowered.len == 0) {
        load_log.err("{f} reads no checkpoint bytes", .{target});
        return error.InvalidBinding;
    }
    const key = lowered[0].source.name;
    if (source.shape.byteSize() != shape.byteSize()) {
        load_log.err("{f} of {d} bytes from {s}: its source fills {f}, {d} bytes", .{ target, shape.byteSize(), key, source.shape, source.shape.byteSize() });
        return error.InvalidBinding;
    }
    var expands = true;
    var runs: u64 = 0;
    for (lowered) |part| {
        expands = expands and expandable(part.lattice);
        runs += part.lattice.runs();
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
            var cursor: backend.Lattice.Cursor = .init(part.lattice, 0, part.lattice.runs());
            while (cursor.next()) |run| : (next += 1) {
                expanded[next] = .{ .source = part.source, .lattice = .{ .src_offset = run.src, .dst_offset = run.dst, .width = part.lattice.width } };
            }
        }
        break :expanded expanded;
    };
    std.mem.sort(LoadPart, parts, {}, struct {
        fn lessThan(_: void, lhs: LoadPart, rhs: LoadPart) bool {
            return lhs.lattice.dst_offset < rhs.lattice.dst_offset;
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
    const placed = platform.placedShape(target.destination()).packedShape();
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

/// `opts` as the backends take them, the checkpoint resolved to the read
/// hints of the VFS backend serving it and the read width to its default.
/// Every option is checked here, once.
fn backendOptions(io: std.Io, opts: Loader.Options) error{InvalidOptions}!backend.Options {
    const hints: VFS.ReadHints = hints: {
        const path = opts.checkpoint orelse break :hints .local;
        const vfs = VFS.fromIo(io) orelse break :hints .local;
        break :hints vfs.readHints(path) catch |err| {
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
    if (hints.read_chunk_size == 0 or hints.read_chunk_size > max_request) {
        load_log.err("invalid loader options: checkpoint={?s}, read_chunk_size={d}, expected 1..{d}: InvalidOptions", .{ opts.checkpoint, hints.read_chunk_size, max_request });
        return error.InvalidOptions;
    }
    if (opts.dma_block_sizes.len == 0) {
        load_log.err("invalid loader options: dma_block_sizes is empty: InvalidOptions", .{});
        return error.InvalidOptions;
    }
    const dma_block_sizes = backend.DmaBlockSizes.fromSlice(opts.dma_block_sizes) catch {
        load_log.err("invalid loader options: {d} dma block sizes, expected at most {d}: InvalidOptions", .{ opts.dma_block_sizes.len, backend.max_dma_block_sizes });
        return error.InvalidOptions;
    };
    // A direct read is widened by up to an alignment unit at each end,
    // which a job of smaller blocks cannot hold (`Planner.maximumJobLen`).
    const alignment = backend.readAlignment(io, opts.direct_io);
    const min_block = @max(alignment, 1);
    for (opts.dma_block_sizes) |block_size| {
        if (block_size < min_block or block_size > max_request) {
            load_log.err("invalid loader options: dma block size {d}, expected {d}..{d}: InvalidOptions", .{ block_size, min_block, max_request });
            return error.InvalidOptions;
        }
    }
    // A source request is the larger of the read chunk and the block
    // calibration selected, and a whole number of blocks (the direct
    // `Loader.create`); with an alignment, it holds a direct read widened at
    // each end. Checked for every block it may select: a candidate, or on
    // CPU the default.
    for ([_][]const usize{ opts.dma_block_sizes, &.{dma_calibration.cpu_block_size} }) |block_sizes| {
        for (block_sizes) |block_size| {
            const request = @max(hints.read_chunk_size, block_size);
            if (request % block_size != 0) {
                load_log.err("invalid loader options: checkpoint={?s}, read_chunk_size={d} is not a multiple of the dma block size {d}: InvalidOptions", .{ opts.checkpoint, hints.read_chunk_size, block_size });
                return error.InvalidOptions;
            }
            if (alignment != 0 and request <= 2 * alignment) {
                load_log.err("invalid loader options: checkpoint={?s}, request of {d} cannot hold a direct read widened by {d} at each end: InvalidOptions", .{ opts.checkpoint, request, alignment });
                return error.InvalidOptions;
            }
        }
    }
    return .{
        .read_parallelism = opts.read_parallelism orelse if (hints.high_latency)
            Loader.Options.high_latency_read_parallelism
        else
            Loader.Options.local_read_parallelism,
        .read_hints = hints,
        .dma_block_sizes = dma_block_sizes,
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

test "start publishes the bulk at once and wait runs program groups one at a time without memory measurement" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    // One program tensor per group.
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2, .program_group_bytes = 1 });
    defer loader.deinit();
    // Exercise serial admission regardless of the platform's memory stats:
    // decided once the set-up is done.
    try loader.awaitSetup();
    loader.memory_supported = false;

    const Model = struct { value: Tensor, first: Tensor, second: Tensor };
    const model: Model = .{
        .value = fixture.value,
        .first = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}),
        .second = fixture.computed("second", fixture.second, LoaderTestFixture.Identity{}),
    };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);

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
    try testing.expectShards(allocator, io, &buffers.value, &LoaderTestFixture.contents, .default);
    try testing.expectShards(allocator, io, &buffers.first, &LoaderTestFixture.contents, .default);
    try testing.expectShards(allocator, io, &buffers.second, &LoaderTestFixture.second_contents, .default);
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
    // Nothing for the loader to fill itself, and one group.
    try std.testing.expectEqual(0, loader.pending.len);
    try std.testing.expectEqual(1, loader.queued.len);
    try std.testing.expectEqual(2, loader.programs.items.len);
    try loader.wait();
    try testing.expectShards(allocator, io, &buffers.a, &.{ 2, 3, 4, 5 }, .default);
    try testing.expectShards(allocator, io, &buffers.b, &.{ 6, 7, 8, 9 }, .default);
    try testing.expectShards(allocator, io, &buffers.c, &.{ 3, 4, 5, 6 }, .default);
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytes_loaded);
}

test "loader delivers host bindings into pinned host memory without charging the devices" {
    for ([_]Memory{ .host_pinned, .host_mapped }) |kind| try expectHostDelivery(kind);
}

fn expectHostDelivery(kind: Memory) !void {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = fixture.platform;
    const pinned = fixture.store.view().createReplicatedTensor("second", null);
    fixture.store.view().setMemory(pinned, kind);
    try std.testing.expectEqual(kind, fixture.store.bindingOf(pinned.id).?.load.memory);

    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    const Model = struct { device: Tensor, host: Tensor };
    const model: Model = .{ .device = fixture.value, .host = pinned };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);

    if (!backend.supportsMemory(platform, kind)) {
        // Refused before anything is submitted.
        try std.testing.expectError(error.InvalidBinding, loader.start(&model, &buffers, &fixture.store, .none));
        try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
        return;
    }
    try loader.start(&model, &buffers, &fixture.store, .none);
    try loader.wait();
    try testing.expectShards(allocator, io, &buffers.device, &LoaderTestFixture.contents, .default);
    // One shard per device, each in that device's pinned host memory space
    // or, `host_mapped`, a view of pinned memory in its default space.
    try testing.expectShards(allocator, io, &buffers.host, &LoaderTestFixture.second_contents, kind);

    // Only the tensor in device memory counts against the devices, on both
    // sides of the admission ledger.
    if (loader.backend.? == .direct) {
        const allocated = try allocator.alloc(u64, platform.devices.len);
        defer allocator.free(allocated);
        loader.backend.?.direct.allocatedBytesPerDevice(allocated);
        try std.testing.expectEqualSlices(u64, loader.backend.?.direct.submitted_bytes, allocated);
        for (allocated) |bytes| try std.testing.expectEqual(LoaderTestFixture.contents.len, bytes);
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
        const tensor = TensorStore.Source.bindProgram(&.{
            store.view().source("value", null).withPartitioning(.replicated, .{}),
            store.view().source("weight", null).withPartitioning(.replicated, .{}),
        }, shape, Sum{ .donate = donate });
        var output: Buffer = undefined;
        try loader.start(&tensor, &output, &store, .none);
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
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    try loader.start(&model, &buffers, &fixture.store, .none);
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
    try loader.awaitSetup();
    loader.memory_supported = false;

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
    var buffers = try mem.bufferize(allocator, &model);
    // Frees whatever the failed load wrote, under the testing allocator.
    defer mem.deinitBufferized(allocator, &buffers);
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
    var placed: DeviceBytes = @splat(0);
    addPlacementBytes(&placed, fixture.platform, fixture.value.shape());
    for (placed[0..fixture.platform.devices.len]) |bytes| try std.testing.expectEqual(LoaderTestFixture.contents.len, bytes);

    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    const Model = struct { value: Tensor, computed: Tensor };
    const model: Model = .{ .value = fixture.value, .computed = fixture.computed("value", fixture.value, LoaderTestFixture.Identity{}) };
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
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
    const too_many: [backend.max_dma_block_sizes + 1]usize = @splat(4 << 20);
    const Case = struct { io: std.Io, opts: Loader.Options };
    const cases = [_]Case{
        .{ .io = io, .opts = .{ .read_parallelism = 0 } },
        .{ .io = io, .opts = .{ .read_parallelism = Loader.max_read_parallelism + 1 } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &.{} } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &.{0} } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &too_large } },
        .{ .io = io, .opts = .{ .dma_block_sizes = &too_many } },
        // The local hints read 8 MiB at least: not a whole number of
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

test "a rate-limited HTTP source loads through the VFS hold" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const tensor_bytes = 1024;
    const tensors = 4;

    // Four tensor ranges with a gap between them, so the planner keeps four
    // jobs and the load makes four concurrent GETs.
    var object: [(2 * tensors - 1) * tensor_bytes]u8 = undefined;
    for (&object, 0..) |*byte, index| byte.* = @truncate(index *% 31 +% 7);

    var server = try VFS.MockServer.init(io, &object, .{
        // Two GETs per 100 ms: the four readers are throttled repeatedly.
        .throttle = .{ .window = .{ .gets = 2, .per_ms = 100 } },
    });
    defer server.deinit(io);
    try server.start(io);

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
    const stats = &http.governor.stats;

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
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);

    var loader = try Loader.init(allocator, vfs.io(), platform, .{
        .checkpoint = url,
        .read_parallelism = 4,
    });
    defer loader.deinit();
    try loader.start(&model, &buffers, &store, .none);
    try loader.wait();

    inline for (comptime std.meta.fieldNames(Model), 0..) |field_name, index| {
        try testing.expectShards(allocator, io, &@field(buffers, field_name), object[2 * index * tensor_bytes ..][0..tensor_bytes], .default);
    }

    try server.stop(io);

    // The server rate limited the load, the backend held every reader, and
    // no reader burned its retry budget for it.
    try std.testing.expect(stats.throttles.load(.acquire) > 0);
    try std.testing.expect(stats.holds.load(.acquire) > 0);
    try std.testing.expect(stats.hold_wait_ns.load(.acquire) > 0);
    try std.testing.expectEqual(@as(u64, 0), stats.retries.load(.acquire));
    try std.testing.expectEqual(@as(u64, tensors * tensor_bytes), stats.physical_bytes.load(.acquire));

    // Holding costs time, not credits: every lifecycle credit, with its
    // pinned blocks, came back.
    const credits = &loader.backend.?.direct.credits;
    try std.testing.expectEqual(credits.free.len, credits.free_len);

    // The checkpoint's VFS backend, not the local hints (8 MiB), sized the
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

    // No binding, an empty source or program input, a mesh of other
    // devices, and a program of the wrong shape or placement.
    const Case = struct { tensor: Tensor, at_wait: bool = false, skip: bool = false };
    const cases = [_]Case{
        .{ .tensor = Tensor.fromShape(fixture.value.shape()) },
        .{ .tensor = fixture.empty },
        .{ .tensor = fixture.computed("empty", fixture.empty, LoaderTestFixture.Identity{}) },
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
/// `init`. Each store takes a copy of `registry` (`testing.storeOf`).
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
        testing.pattern(&self.a, 1);
        testing.pattern(&self.b, 2);
        testing.pattern(&self.c, 3);
        testing.pattern(&self.plain, 4);
        testing.pattern(&self.q, 9);
        testing.pattern(&self.k, 10);
        testing.pattern(&self.v, 11);
        for (&self.w1, &self.w3, 0..) |*w1, *w3, e| {
            testing.pattern(w1, @intCast(20 + e));
            testing.pattern(w3, @intCast(30 + e));
        }
        var spacer: [16]u8 = undefined;
        testing.pattern(&spacer, 5);
        testing.pattern(&self.wide, 6);
        try testing.writeSafetensors(allocator, io, self.tmp.dir, "first.safetensors", &self.registry, &.{
            .{ .name = "a", .dtype = "U16", .dims = &.{ 3, 8 }, .bytes = &self.a },
            .{ .name = "b", .dtype = "U16", .dims = &.{ 2, 8 }, .bytes = &self.b },
            .{ .name = "spacer", .dtype = "U16", .dims = &.{ 1, 8 }, .bytes = &spacer },
            .{ .name = "q", .dtype = "U16", .dims = &.{ 8, 8 }, .bytes = &self.q },
            .{ .name = "k", .dtype = "U16", .dims = &.{ 4, 8 }, .bytes = &self.k },
            .{ .name = "v", .dtype = "U16", .dims = &.{ 4, 8 }, .bytes = &self.v },
        });
        try testing.writeSafetensors(allocator, io, self.tmp.dir, "second.safetensors", &self.registry, &.{
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
        var devices: testing.Devices = undefined;
        try devices.init(allocator, io, device_count);
        defer devices.deinit();
        const platform = devices.platform;
        const tp = try devices.mesh("tp");
        var store = try testing.storeOf(allocator, &fixture.registry, .{ .meshes = &.{&tp} });
        defer store.deinit();
        // Rows over the devices (b and c straddle shards on four), every
        // row split over them, and every byte on each of them.
        inline for (.{ .{ .row = .model }, .{ .col = .model }, .{} }) |partitioning| {
            const shards: u64 = if (@typeInfo(@TypeOf(partitioning)).@"struct".field_names.len == 0) 1 else device_count;
            for ([_]Memory{ .default, .host_pinned, .host_mapped }) |memory| {
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
                var buffers = try mem.bufferize(allocator, &model);
                defer mem.deinitBufferized(allocator, &buffers);
                var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
                defer loader.deinit();
                try loader.start(&model, &buffers, &store, .none);
                try loader.wait();

                try testing.expectShards(allocator, io, &buffers.concatenated, &concatenated, memory);
                try testing.expectShards(allocator, io, &buffers.padded, &padded, memory);
                try testing.expectShards(allocator, io, &buffers.reversed, &reversed, memory);
                try testing.expectShards(allocator, io, &buffers.plain, &fixture.plain, memory);
                // Each destination once, padding included.
                try std.testing.expectEqual(@as(usize, 256 + 128 + 96 + 64), loader.bytes_loaded);
                // A destination is allocated once per device, whatever its
                // parts; host memory counts on neither side.
                if (loader.backend.? == .direct) {
                    const allocated = try allocator.alloc(u64, platform.devices.len);
                    defer allocator.free(allocated);
                    loader.backend.?.direct.allocatedBytesPerDevice(allocated);
                    try std.testing.expectEqualSlices(u64, loader.backend.?.direct.submitted_bytes, allocated);
                    for (devices.physical.devices_in_canonical_order) |device| {
                        const expected: u64 = if (memory.isHost()) 0 else (256 + 128 + 64) / shards + 96;
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
        var devices: testing.Devices = undefined;
        try devices.init(allocator, io, device_count);
        defer devices.deinit();
        const platform = devices.platform;
        const tp = try devices.mesh("tp");
        var store = try testing.storeOf(allocator, &fixture.registry, .{ .meshes = &.{&tp} });
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
        var lowering: std.heap.ArenaAllocator = .init(allocator);
        defer lowering.deinit();
        const qkv_parts = try store.bindingOf(model.qkv.id).?.load.source.lower(lowering.allocator());
        try std.testing.expectEqual(@as(usize, 3), qkv_parts.len);
        for (qkv_parts) |part| try std.testing.expectEqual(@as(u64, device_count), part.lattice.runs());

        var buffers = try mem.bufferize(allocator, &model);
        defer mem.deinitBufferized(allocator, &buffers);
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
        try testing.expectShards(allocator, io, &buffers.qkv, &qkv, .default);
        try testing.expectShards(allocator, io, &buffers.concatenated, &concatenated_experts, .default);
        try testing.expectShards(allocator, io, &buffers.interleaved, &interleaved_experts, .default);
        try testing.expectShards(allocator, io, &buffers.head, fixture.q[0..64], .default);
        var tail_bytes: [128]u8 = undefined;
        @memcpy(tail_bytes[0..64], fixture.q[64..128]);
        @memcpy(tail_bytes[64..], &fixture.k);
        try testing.expectShards(allocator, io, &buffers.tail, &tail_bytes, .default);
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
    testing.pattern(&g, 1);
    testing.pattern(&u, 2);
    testing.pattern(&z, 3);
    testing.pattern(&s0, 4);
    testing.pattern(&s1, 5);
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try testing.writeSafetensors(allocator, io, tmp.dir, "first.safetensors", &registry, &.{
        .{ .name = "g", .dtype = "U16", .dims = &.{ 96, 8 }, .bytes = &g },
        .{ .name = "s0", .dtype = "F32", .dims = &.{}, .bytes = &s0 },
        .{ .name = "z", .dtype = "U16", .dims = &.{ 24, 4 }, .bytes = &z },
    });
    try testing.writeSafetensors(allocator, io, tmp.dir, "second.safetensors", &registry, &.{
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
            var devices: testing.Devices = undefined;
            try devices.init(allocator, io, device_count);
            defer devices.deinit();
            const tp = try devices.mesh("tp");
            var store = try testing.storeOf(allocator, &registry, .{ .meshes = &.{&tp} });
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
            var buffers = try mem.bufferize(allocator, &model);
            defer mem.deinitBufferized(allocator, &buffers);
            var loader = try Loader.init(allocator, io, devices.platform, .{ .read_parallelism = 2 });
            defer loader.deinit();
            // Jobs of 128 bytes: every destination takes several.
            try loader.awaitSetup();
            switch (loader.backend.?) {
                .direct => |direct| {
                    direct.plan_config.request_size = 256;
                    direct.reads.alignment = alignment;
                },
                .buffered => {},
            }
            try loader.start(&model, &buffers, &store, .none);
            try loader.wait();
            try testing.expectShards(allocator, io, &buffers.by_rows, &interleaved, .default);
            try testing.expectShards(allocator, io, &buffers.by_columns, &interleaved, .default);
            try testing.expectShards(allocator, io, &buffers.scales_by_rows, &scales, .default);
            try testing.expectShards(allocator, io, &buffers.scales_by_columns, &scales, .default);
            try testing.expectShards(allocator, io, &buffers.transposed, &transposed, .default);
            try testing.expectShards(allocator, io, &buffers.transposed_by_columns, &transposed, .default);
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
    var store = try testing.storeOf(allocator, &fixture.registry, .{ .meshes = &.{platform.replicated_mesh} });
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
    var store = try testing.storeOf(allocator, &fixture.registry, .{ .meshes = &.{platform.replicated_mesh} });
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
    var buffers = try mem.bufferize(allocator, &model);
    defer mem.deinitBufferized(allocator, &buffers);
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    try loader.start(&model, &buffers, &store, .none);
    try loader.wait();

    var expected: [128]u8 = @splat(0);
    @memcpy(expected[64..80], fixture.b[16..32]);
    try testing.expectShards(allocator, io, &buffers.sparse, &expected, .default);
    expected = @splat(0);
    @memcpy(expected[0..32], &fixture.b);
    @memcpy(expected[64..112], &fixture.a);
    try testing.expectShards(allocator, io, &buffers.padded, &expected, .default);
    expected = @splat(0);
    @memcpy(expected[0..16], fixture.a[32..48]);
    @memcpy(expected[16..48], fixture.a[0..32]);
    try testing.expectShards(allocator, io, &buffers.swapped, &expected, .default);
    expected = @splat(0);
    for (0..3) |i| @memcpy(expected[i * 32 ..][0..16], fixture.a[i * 16 ..][0..16]);
    try testing.expectShards(allocator, io, &buffers.strided, &expected, .default);
    var transposed_bytes: [1024]u8 = undefined;
    for (0..8) |column| {
        for (0..64) |row| @memcpy(transposed_bytes[column * 128 + row * 2 ..][0..2], fixture.wide[row * 16 + column * 2 ..][0..2]);
    }
    try testing.expectShards(allocator, io, &buffers.transposed, &transposed_bytes, .default);
}

test "a program takes its inputs from sources, in order" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = testing.env();
    var store = try testing.storeOf(allocator, &fixture.registry, .{ .meshes = &.{platform.replicated_mesh} });
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
    const inputs = [_]Source.Placed{ assembled.withPartitioning(.replicated, .{}), c.withPartitioning(.replicated, .{}) };
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
    try testing.expectShards(allocator, io, &output, &expected, .default);
}

test "a padded host-mapped table stays in host memory and serves device gathers in place" {
    // The engram arrangement: one source of 264-byte rows, an odd row count
    // padded to a multiple of 8, rows sharded over every device, delivered
    // as device-kind views of pinned host memory. A program gathers rows
    // from each device's local shard.
    try expectHostTableGathers(.rows);
}

test "a replicated host-mapped table is one allocation that every device reads in place" {
    // The same table on every device: the direct loader makes it one
    // allocation (`direct_loader.TensorTransfer.initHost`), each device's
    // shard a view of it, and every device's kernels gather from it.
    try expectHostTableGathers(.replicated);
}

/// Loads a padded table as `host_mapped`, rows sharded over every device or
/// replicated on each, and gathers rows of each device's shard on that
/// device. The table is freed while the gather is queued behind matrix
/// products: its memory must outlive the read, and go once it is done.
fn expectHostTableGathers(placement: enum { rows, replicated }) !void {
    const direct_loader = @import("direct_loader.zig");
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
    const api = platform.pjrt_api;
    const mesh: *const Sharding.Mesh = if (owned) |p|
        try p.registerMesh("model", .mesh(.{ .model = .high_bandwidth }))
    else
        shared.meshes.get("model").?;
    if (!backend.supportsMemory(platform, .host_mapped)) return error.SkipZigTest;
    const devices = mesh.devicesInCanonicalOrder();
    const shard_count = devices.len;

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
        try testing.writeSafetensors(allocator, io, tmp.dir, "table.safetensors", &registry, &.{
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
    store.view().setMemory(table, .host_mapped);
    try std.testing.expectEqual(@as(i64, @intCast(padded_rows)), table.dim(.row));

    const live_before = direct_loader.liveHostAllocations();
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 4 });
    defer loader.deinit();
    // The device allocators' counts, once the set-up's calibration has
    // allocated on the devices. Bytes in use would not do: CUDA's counts
    // what its pool holds, which freed memory leaves room in. CPU's plugin
    // counts nothing.
    try loader.awaitSetup();
    var allocations: [Platform.MAX_NUM_DEVICES]?u64 = undefined;
    for (devices) |device| allocations[device.id] = platform.devices[device.id].memoryStats().num_allocs;
    var device_table: Buffer = undefined;
    try loader.start(&table, &device_table, &store, .none);
    try loader.wait();
    var freed = false;
    defer if (!freed) device_table.deinit();
    // The table took no device memory: nothing was allocated on a device.
    for (devices) |device| if (allocations[device.id]) |before| {
        try std.testing.expectEqual(before, platform.devices[device.id].memoryStats().num_allocs.?);
    };
    try std.testing.expectEqual(padded.len, loader.bytes_loaded);
    try testing.expectShards(allocator, io, &device_table, padded, .host_mapped);
    // A row shard is one allocation per device, a replicated table one.
    try std.testing.expectEqual(live_before + if (placement == .rows) shard_count else 1, direct_loader.liveHostAllocations());

    const first_pointer = try device_table._shards.get(0).opaqueDeviceMemoryDataPointer(api);
    for (devices, device_table._shards.constSlice()) |device, shard| {
        // Replicas share one allocation; row shards never do.
        const pointer = try shard.opaqueDeviceMemoryDataPointer(api);
        const shares = placement == .replicated and shard_count > 1;
        if (device.id != devices[0].id) try std.testing.expectEqual(shares, pointer == first_pointer);
    }

    // Each device gathers `gathered` rows of its own shard, the padding
    // rows of the last one included (of every replica, replicated), once
    // a chain of matrix products is done: the indices depend on them.
    const Gather = struct {
        fn forward(table_: Tensor, indices: Tensor, ballast: Tensor) Tensor {
            var product = ballast;
            for (0..8) |_| product = product.dot(product.withTags(.{ .k, .j }), .k).withTags(.{ .i, .k });
            // Zero, as every product keeps the ballast's positive entries.
            const offset: Tensor = .select(
                product.sum(.i).sum(.k).reshape(Shape.init(.{}, .f32)).cmp(.LT, .scalar(0, .f32)),
                .scalar(1, .u32),
                .scalar(0, .u32),
            );
            const output: Shape = Shape.init(.{ .shard = indices.dim(.shard), .bytes = indices.dim(.k) * table_.dim(.channel) }, .u8)
                .withPartitioning(table_.shape()._sharding.mesh.?, .{ .shard = .model });
            return ops.manualComputation((struct {
                fn body(inputs: struct { table: Tensor, indices: Tensor, offset: Tensor }, local: Shape) Tensor {
                    const shifted = inputs.indices.add(inputs.offset.broad(inputs.indices.shape()));
                    return inputs.table.gather(.{ .row = shifted }, .{}).reshape(local);
                }
            }).body, .{ .table = table_, .indices = indices, .offset = offset }, output);
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
    // Milliseconds of products on an accelerator; a token amount on CPU.
    const ballast_dim: usize = if (platform.target == .cpu) 32 else 4096;
    const ballast_shape = Shape.init(.{ .i = ballast_dim, .k = ballast_dim }, .f32).withPartitioning(mesh, .{});
    const ballast = try allocator.alloc(f32, ballast_dim * ballast_dim);
    defer allocator.free(ballast);
    @memset(ballast, 1.0 / @as(f32, @floatFromInt(ballast_dim)));
    var ballast_buffer: Buffer = try .from(io, platform, ballast_shape, std.mem.sliceAsBytes(ballast), .{});
    defer ballast_buffer.deinit();

    const exe = try platform.compileFn(allocator, io, Gather.forward, .{ Tensor.fromShape(table.shape()), Tensor.fromShape(indices_shape), Tensor.fromShape(ballast_shape) }, .{ .meshes = &.{mesh} });
    defer exe.deinit();
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{ device_table, indices_buffer, ballast_buffer });
    try exe.call(io, args, &results, .{});
    var output = results.get(Buffer);
    defer output.deinit();

    // The table goes while the gather may still have to read it. Pinned
    // buffers of its size and another pattern would take its memory had it
    // gone early, and the gather would return their bytes.
    device_table.deinit();
    freed = true;
    const decoy_bytes = try allocator.alloc(u8, padded.len);
    defer allocator.free(decoy_bytes);
    @memset(decoy_bytes, 0xee);
    var decoy: Buffer = try .from(io, platform, table.shape(), decoy_bytes, .{ .memory = .host_pinned });
    defer decoy.deinit();

    const expected = try allocator.alloc(u8, shard_count * gathered * row_bytes);
    defer allocator.free(expected);
    for (0..shard_count) |shard| {
        for (0..gathered) |k| {
            const first_row = if (placement == .rows) shard * shard_rows else 0;
            const row = first_row + indices[shard * gathered + k];
            @memcpy(expected[(shard * gathered + k) * row_bytes ..][0..row_bytes], padded[row * row_bytes ..][0..row_bytes]);
        }
    }
    try testing.expectShards(allocator, io, &output, expected, .default);

    // The allocations go with their last views, once nothing reads them;
    // PJRT may drop the views on its own threads just after the output is
    // ready.
    var polls: usize = 0;
    while (direct_loader.liveHostAllocations() != live_before) : (polls += 1) {
        if (polls == 5000) return error.TestUnexpectedResult;
        try io.sleep(.fromMilliseconds(1), .awake);
    }

    // The control: the same table in device memory is allocated on every
    // device, so the check after the load above can fail.
    const device_copy = switch (placement) {
        .rows => source.withPartitioning(.model, .{ .row = .model }).bind(),
        .replicated => source.withPartitioning(.model, .{}).bind(),
    };
    for (devices) |device| allocations[device.id] = platform.devices[device.id].memoryStats().num_allocs;
    var device_buffer: Buffer = undefined;
    try loader.start(&device_copy, &device_buffer, &store, .none);
    try loader.wait();
    defer device_buffer.deinit();
    for (devices) |device| if (allocations[device.id]) |before| {
        try std.testing.expect(platform.devices[device.id].memoryStats().num_allocs.? > before);
    };
}
