//! Checkpoint loading. `Loader.start` takes a whole model: it starts
//! compiling the programs that compute some of its tensors
//! (`Source.bindProgram`) and publishes every tensor the loader fills
//! itself. `wait` then publishes the programs' inputs in groups the devices
//! have room for, runs each group's programs on the caller's task once its
//! inputs landed, and retires everything. Backends only read sources into
//! buffers; lookup, compilation, admission and execution stay in this
//! shared front end.
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
/// programs compute from theirs. The platform and options' borrowed values
/// must outlive the loader. Start and wait serially on the owning task;
/// source reads and transfers run concurrently inside the selected backend,
/// compiles on tasks of their own. Submissions are retired in publish
/// order.
pub const Loader = struct {
    /// Every error `start` and `wait` return. Each one fails the loader:
    /// everything started is canceled (`cancel`), and every later call
    /// returns the first error. A model whose bindings are wrong is
    /// `InvalidBinding`, logged at err with the tensor, its key and the rule
    /// it breaks. The others come from reading the checkpoint, the devices,
    /// the compiler, or `cancel`, and from `init`'s task: a platform with
    /// more devices than the direct backend addresses is
    /// `UnsupportedPlatform`, an io that cannot run its tasks concurrently
    /// `ConcurrencyUnavailable`.
    pub const Error = error{ InvalidBinding, Canceled, UnsupportedPlatform } || std.mem.Allocator.Error || pjrt.ApiError || std.Io.File.OpenError ||
        std.Io.File.Reader.SeekError || safetensors.ExactPositionalError || Compiler.Error || std.Io.ConcurrentError;

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
    /// Placement bytes of every published submission, per device.
    submitted_bytes: []u64,
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
        allocated: []u64,
        inputs: []u64,
        placed: []u64,
    };

    pub const Options = struct {
        /// The checkpoint's path or URI, read by `init` only. When `io` is a
        /// VFS, the VFS backend serving that path sizes the reads (its
        /// `VFS.LoadProfile`), whose name and read statistics the loader
        /// borrows: the VFS and that backend must outlive the loader. Null,
        /// or a plain `io`: local files without direct I/O.
        checkpoint: ?[]const u8 = null,
        /// Concurrent source reads, 1 to `max_read_parallelism`. Null takes
        /// the checkpoint's default (`backend.Options.readWidth`: 16
        /// locally, 32 on a high-latency source), clipped to one less than
        /// what the pre-grown pinned set holds. Fixed for the load: a source
        /// that rate limits holds the VFS instead (`vfs/request.zig`).
        read_parallelism: ?usize = null,
        /// The DMA block sizes calibration screens, each at least the
        /// checkpoint's direct I/O alignment (4 KiB for a local file under a
        /// VFS, else 1) and at most 32 MiB (`backend.max_read_request_size`).
        /// Ignored by the buffered backend; CPU takes the default sizing
        /// without measuring.
        dma_block_sizes: []const usize = &dma_calibration.default_block_sizes,
        /// Direct I/O for local source files, decided per file by the
        /// VFS that opens it: `auto` reads a file past the page cache when it
        /// is mostly not cached at the first decision, `on` whenever the
        /// filesystem allows it, `off` never. The planner widens a direct file's
        /// reads to the profile's alignment, at most two alignment units per
        /// request. A direct read never fills the page cache, so under
        /// `auto` a cold file stays cold and comes from the disk on every
        /// load (in a file of many pages per sample window, the residency probe
        /// lands on other pages every time, so the ones it cached do not count;
        /// a file of at most 32 pages is sampled whole and turns buffered after
        /// its first load); on a host whose warm buffered reads beat its disk, a model
        /// loaded repeatedly is better served by `off`. Nothing changes for
        /// a profile without alignment.
        /// Warm replicated Llama-3.1-8B on eight MI300X took ~1.10 s buffered but
        /// ~4.53 s forced direct from a slow storage extent; on four GB300
        /// with four NVMe drives in RAID0, direct took ~0.30 s versus ~0.32 s
        /// warm buffered. These are loader times, not disk-only rates.
        /// Residency alone cannot predict which path wins. The loader retains
        /// each open file, so its decision is not remeasured on every submission
        /// if cache residency changes.
        direct_io: VFS.DirectIo = .auto,
    };

    /// Concurrent source reads at most (`Options.read_parallelism`).
    pub const max_read_parallelism: usize = 128;

    /// Checks `opts` and returns at once. Warming the device allocators and
    /// creating the backend (for the direct one: DMA calibration and the
    /// pinned pre-growth) run on a task of the loader's own, or here when
    /// `io` has no concurrency; the first `start` waits for that task and
    /// returns its error. A value out of range, or a checkpoint whose
    /// scheme the VFS does not serve, is `InvalidOptions`, logged.
    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) error{ InvalidOptions, OutOfMemory }!Loader {
        const backend_opts = try backendOptions(io, opts);

        const n = platform.devices.len;

        const submitted_bytes = try allocator.alloc(u64, n);
        errdefer allocator.free(submitted_bytes);
        @memset(submitted_bytes, 0);

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
            .submitted_bytes = submitted_bytes,
            .pending_execution = pending_execution,
            .scratch = .{
                .words = words,
                .room = words[0..n],
                .allocated = words[n .. 2 * n],
                .inputs = words[2 * n .. 3 * n],
                .placed = words[3 * n .. 4 * n],
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
        if (self.queued.len == 0 and self.pending.len == 0 and self.programs.items.len == 0) return;
        if (self.failure == null) self.failure = error.Canceled;
        while (self.queued.popFront()) |group| group.deinit(self.allocator);
        if (self.pending.len != 0) self.backend.?.abort();
        while (self.pending.len != 0) self.retireOldest(false) catch {};
        self.releasePrograms();
    }

    /// Cancels what is still started (`cancel`): a caller that succeeded
    /// waited for it, so only an error path leaves any. Then cancels
    /// `init`'s task when no `start` awaited it, destroying what it built,
    /// and destroys the backend.
    pub fn deinit(self: *Loader) void {
        self.cancel();
        self.logAdmission();
        self.queued.deinit(self.allocator);
        self.pending.deinit(self.allocator);
        self.programs.deinit(self.allocator);
        self.allocator.free(self.submitted_bytes);
        self.allocator.free(self.pending_execution);
        self.allocator.free(self.scratch.words);
        if (self.setup) |*setup| {
            if (setup.cancel(self.io)) |backend_| backend_.destroy() else |_| {}
        }
        if (self.backend) |backend_| backend_.destroy();
        self.* = undefined;
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
    /// mesh), into the memory space its binding names, filled from its one
    /// source or assembled from its parts (`TensorStore.Layout`) with every
    /// byte no part covers set to zero. All of these are published here, as
    /// one planned submission. A tensor a program computes
    /// (`Source.bindProgram`) starts its program compiling, once for every
    /// tensor of the same program; `wait` lands its inputs and runs it.
    /// Before anything is published, a tensor without a binding, a
    /// zero-byte source, a composite layout that does not fit its tensor or
    /// that this loader cannot assemble yet, and a memory kind the platform
    /// cannot take (`backend.supportsMemory`) are `error.InvalidBinding`.
    /// Several loads may be started before one `wait`. `buffers` is
    /// `mem.bufferize` of `model`, whose buffers are empty; `start` empties
    /// them again, and each stays empty until its load writes it whole. So
    /// after any error each buffer is empty or holds shards for the caller
    /// to free: `mem.deinitBufferized` frees the tree either way. `buffers`
    /// and `store` are borrowed until `wait` or `cancel` returns; do not end
    /// `progress` before then. `start` adds one item to `progress` for each
    /// tensor the loader fills, a composite one once, and for each program
    /// input; the backends complete them. The first `start` waits for
    /// `init`'s task first, and fails with its error.
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
        const prepared = try prepareModel(self.allocator, self.platform, store, Model, model, buffers);
        defer prepared.deinit(self.allocator);
        // The backends complete one item per spec: one per tensor the loader
        // fills and, in the groups, one per program input.
        var items = prepared.specs.len;
        for (prepared.computed) |item| items += item.binding.program.?.inputs.len;
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
    /// compiles. Idempotent.
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
    /// shells it runs over and the specs that fill them, one per input.
    fn prepareGroup(self: *Loader, store: *const TensorStore, computed: []const Computed, progress: std.Progress.Node) Error!Group {
        const executables = try self.allocator.alloc(BoundExecutable, computed.len);
        var prepared: usize = 0;
        errdefer {
            for (executables[0..prepared]) |*executable| executable.deinit(self.allocator);
            self.allocator.free(executables);
        }
        var part_count: usize = 0;
        var spec_count: usize = 0;
        for (computed, executables) |item, *executable| {
            const program = item.binding.program.?;
            const compiled = try self.compiledProgram(store, program, item.shape);
            const inputs = try self.allocator.alloc(Buffer, program.inputs.len);
            for (inputs, program.inputs) |*input, layout| {
                input.* = .{ ._platform = self.platform, ._shape = layout.shape, ._shards = .empty };
            }
            executable.* = .{ .program = compiled, .inputs = inputs, .output = item.output };
            prepared += 1;
            spec_count += inputs.len;
            for (program.inputs) |*layout| part_count += layoutPartCount(layout);
        }
        const specs = try self.allocator.alloc(LoadSpec, spec_count);
        errdefer self.allocator.free(specs);
        var parts: std.ArrayList(LoadPart) = .empty;
        errdefer parts.deinit(self.allocator);
        // Exact, so the specs' slices stay where they point.
        try parts.ensureTotalCapacityPrecise(self.allocator, part_count);
        var next: usize = 0;
        for (computed, executables) |item, executable| {
            const tensors = item.binding.tensors;
            for (item.binding.program.?.inputs, executable.inputs) |*layout, *input| {
                const first = parts.items.len;
                const whole = layout.whole(tensors);
                if (whole) |source| {
                    if (source.byteSize() == 0) {
                        load_log.err("program input {s} in {s} of tensor {} {f} is empty", .{ source.name, source.file_uri, item.id, item.shape });
                        return error.InvalidBinding;
                    }
                    parts.appendAssumeCapacity(.whole(source));
                } else try appendCompositeParts(self.allocator, &parts, item.id, layout.shape, tensors, .default, layout);
                specs[next] = .{
                    .shape = layout.shape,
                    .memory = .default,
                    .output = input,
                    .parts = parts.items[first..],
                    .zero_fill = whole == null,
                };
                next += 1;
            }
        }
        std.debug.assert(next == specs.len and parts.items.len == parts.capacity);
        return .{ .executables = executables, .specs = specs, .parts = parts.items, .progress = progress };
    }

    /// The compiled `program` computing a tensor of `output`: one started
    /// already for the same function, equal args and the same placed
    /// shapes, else a compile started now.
    fn compiledProgram(self: *Loader, store: *const TensorStore, program: *const TensorStore.Program, output: Shape) Error!*CompiledProgram {
        for (self.programs.items) |compiled| {
            if (compiled.program.sameExecutable(compiled.output, program.*, output)) return compiled;
        }
        try self.programs.ensureUnusedCapacity(self.allocator, 1);
        const compiled = try CompiledProgram.start(self.allocator, self.io, self.platform, store.meshes, program, output);
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
        @memset(inputs, 0);
        @memset(execution, 0);
        for (executables, 0..) |*executable, index| {
            const exe = executable.exe;
            const name = executable.program.program.vtable.name;
            for (exe.input_shapes) |shape| {
                try addPlacementBytes(inputs, self.platform, shape, .default, name);
            }
            // The output's memory space is the program's to choose; it is
            // charged to the devices like the default space.
            try addPlacementBytes(execution, self.platform, exe.output_shapes[0], .default, name);
            // Tensors sharing a program run one after the other, so its
            // temporaries are charged once.
            const seen = for (executables[0..index]) |*earlier| {
                if (earlier.exe == exe) break true;
            } else false;
            if (seen) continue;
            const temp = self.compiledTempBytes(exe);
            const output_mesh = backend.placedShape(self.platform, exe.output_shapes[0])._sharding.mesh.?;
            for (output_mesh.devicesInCanonicalOrder()) |device| execution[device.id] +|= temp;
        }
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

    /// Refreshes the per-device room; false when backend accounting or device
    /// memory statistics are unavailable.
    fn readRoom(self: *Loader) bool {
        switch (self.backend.?) {
            .direct => |direct| direct.allocatedBytesPerDevice(self.scratch.allocated),
            .buffered => return false,
        }
        for (self.scratch.room, self.platform.devices, self.submitted_bytes, self.scratch.allocated) |*out, device, submitted, allocated| {
            const reported = device.memoryStats();
            out.* = admission.room(reported.bytes_limit, reported.bytes_in_use, submitted, allocated, admission.reserve_bytes) orelse return false;
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
        var logical_bytes: usize = 0;
        const placed = self.scratch.placed;
        @memset(placed, 0);
        // One spec per destination, however many parts fill it: its bytes
        // count once, on each device it is placed on.
        for (specs) |spec| {
            logical_bytes += spec.shape.byteSize();
            try addPlacementBytes(placed, self.platform, spec.shape, spec.memory, spec.parts[0].source.name);
        }
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const submission = self.backend.?.submit(specs, progress) catch |err| return backendError(err);
        self.pending.pushBackAssumeCapacity(.{
            .submission = submission,
            .executables = executables,
            .execution = execution,
            .logical_bytes = logical_bytes,
        });
        for (self.submitted_bytes, placed) |*submitted, bytes| submitted.* +|= bytes;
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
        oldest.submission.await() catch |err| return backendError(err);
        if (!execute) return;
        for (oldest.executables) |*executable| {
            executable.execute(self.allocator, self.io) catch |err| {
                load_log.err("run program {s}: inputs={d}, output={f}: {s}", .{ executable.program.program.vtable.name, executable.inputs.len, executable.program.output, @errorName(err) });
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
    /// One per program input, filling the executables' input shells.
    specs: []LoadSpec,
    parts: []LoadPart,
    progress: std.Progress.Node,

    fn deinit(self: Group, allocator: std.mem.Allocator) void {
        for (self.executables) |*executable| executable.deinit(allocator);
        allocator.free(self.executables);
        allocator.free(self.specs);
        allocator.free(self.parts);
    }
};

/// A program compiling on a task of its own, then compiled.
const CompiledProgram = struct {
    program: *const TensorStore.Program,
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
        program: *const TensorStore.Program,
        output: Shape,
    ) std.mem.Allocator.Error!*CompiledProgram {
        const self = try allocator.create(CompiledProgram);
        errdefer allocator.destroy(self);
        const inputs = try allocator.alloc(Tensor, program.inputs.len);
        for (inputs, program.inputs) |*input, layout| input.* = .fromShape(layout.shape);
        // Concurrent, as `io.async` may run the compile right here on a busy
        // threaded io, ahead of the reads `start` publishes after it.
        const args = .{ program, allocator, io, platform, meshes, output, inputs };
        self.* = .{
            .program = program,
            .output = output,
            .inputs = inputs,
            .future = io.concurrent(compile, args) catch io.async(compile, args),
        };
        return self;
    }

    fn compile(
        program: *const TensorStore.Program,
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

    /// The executable, waiting for the compile the first time. It must
    /// compute one tensor of `output`, placed alike (`checkOutput`).
    fn await(self: *CompiledProgram, io: std.Io, platform: *const Platform) Loader.Error!*const Exe {
        if (self.exe == null) {
            self.exe = self.future.await(io) catch |err| {
                load_log.err("compile program {s} computing {f}: {s}", .{ self.program.vtable.name, self.output, @errorName(err) });
                return err;
            };
            try checkOutput(platform, self.program.vtable.name, self.output, &self.exe.?);
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
    /// submitted-but-unallocated bytes (workers allocate tensors lazily, so
    /// `allocated` may briefly exceed `submitted`), minus the reserve.
    fn room(limit: ?u64, in_use: u64, submitted: u64, allocated: u64, reserve: u64) ?u64 {
        return (limit orelse return null) -| in_use -| (submitted -| allocated) -| reserve;
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
    try std.testing.expectEqual(null, admission.room(null, 0, 0, 0, 0));
    try std.testing.expectEqual(55, admission.room(100, 30, 50, 40, 5).?);
    // More in use than the limit admits: no room, no underflow.
    try std.testing.expectEqual(0, admission.room(100, 130, 0, 0, 0).?);
    // A worker allocated ahead of the front end's accounting: nothing unlanded.
    try std.testing.expectEqual(70, admission.room(100, 30, 10, 40, 0).?);
    try std.testing.expectEqual(0, admission.room(100, 90, 0, 0, 20).?);
}

test "admits needs room on every device" {
    try std.testing.expect(admission.admits(&.{ 40, 40 }, &.{ 20, 20 }, &.{ 10, 10 }, &.{ 5, 5 }));
    // The second device is one byte short.
    try std.testing.expect(!admission.admits(&.{ 40, 34 }, &.{ 20, 20 }, &.{ 10, 10 }, &.{ 5, 5 }));
    try std.testing.expect(admission.admits(&.{}, &.{}, &.{}, &.{}));
}

/// Adds the bytes `shape` occupies on each device it is placed on, the same
/// expression the direct backend allocates by, so submitted and allocated
/// bytes cancel exactly once everything landed. Bytes delivered into a host
/// memory space take no device memory and count nowhere. A shape its mesh
/// cannot divide is a binding error in any memory space; `name`, its source
/// or program, names it in the log.
fn addPlacementBytes(out: []u64, platform: *const Platform, shape: Shape, memory: Memory.Kind, name: []const u8) error{InvalidBinding}!void {
    const placement = Sharding.Placement.init(backend.placedShape(platform, shape).packedShape()) catch {
        load_log.err("tensor {f} from {s}: its dimensions are not divisible by its mesh partitions", .{ shape, name });
        return error.InvalidBinding;
    };
    if (!backend.countsAgainstDevice(memory)) return;
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
        error.SourceSizeMismatch,
        error.InvalidLayout,
        error.UnsupportedLayout,
        error.InvalidTensorRange,
        error.EmptyTensor,
        error.UnsupportedMemoryKind,
        error.IncompatibleSharding,
        => error.InvalidBinding,
        error.Closed, error.InvalidOptions, error.Internal => stdx.debug.panic("loader backend: {s}", .{@errorName(err)}),
        else => |e| e,
    };
}

/// A tensor a program computes, the buffer it computes into and its
/// binding.
const Computed = struct {
    id: Tensor.Id,
    shape: Shape,
    output: *Buffer,
    binding: TensorStore.Binding,
};

/// A model as `start` takes it: one spec per tensor the loader fills, the
/// parts they slice, and the tensors programs compute. The caller frees
/// all three.
const PreparedModel = struct {
    specs: []LoadSpec,
    parts: []LoadPart,
    computed: []Computed,

    fn deinit(self: PreparedModel, allocator: std.mem.Allocator) void {
        allocator.free(self.specs);
        allocator.free(self.parts);
        allocator.free(self.computed);
    }
};

/// Pairs each tensor of `model` with its buffer and binding: a plain binding
/// is one whole-source part, a composite one expands into one part per
/// source, all sharing the tensor's shape, memory space and output buffer;
/// a program binding is computed.
fn prepareModel(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    comptime Model: type,
    model: *const Model,
    buffers: *Bufferized(Model),
) PrepareError!PreparedModel {
    const tensor_count = meta.count(Tensor, model);
    // Tensors and buffers pair by visit order: a buffer tree whose optionals,
    // slices or unions differ from the model's would write past `flattened`
    // or leave buffers unpaired.
    const buffer_count = meta.count(Buffer, buffers);
    stdx.debug.assert(buffer_count == tensor_count, "Loader.start: the model has {d} tensors but its buffers {d}; bufferize the buffers from this model", .{ tensor_count, buffer_count });
    const flattened = try allocator.alloc(*Buffer, tensor_count);
    defer allocator.free(flattened);
    meta.forEachVisit(buffers, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
            output[i] = buffer;
        }
    }.call, .{flattened});
    // Empty until a backend or a program writes it whole, so the caller can
    // free the tree whatever fails.
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, output: []*Buffer, platform_: *const Platform) void {
            output[i].* = .{ ._platform = platform_, ._shape = tensor.shape(), ._shards = .empty };
        }
    }.call, .{ flattened, platform });

    // Sized first, so the specs can slice the parts as they are appended.
    var part_count: usize = 0;
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(_: usize, tensor: *const Tensor, store_: *const TensorStore, count: *usize) void {
            const binding = store_.getSourcesById(tensor.id) orelse return;
            if (binding.program == null) count.* += loadPartCount(binding);
        }
    }.call, .{ store, &part_count });

    var specs: std.ArrayList(LoadSpec) = .empty;
    errdefer specs.deinit(allocator);
    try specs.ensureTotalCapacityPrecise(allocator, tensor_count);
    var computed: std.ArrayList(Computed) = .empty;
    errdefer computed.deinit(allocator);
    try computed.ensureTotalCapacityPrecise(allocator, tensor_count);
    var parts: std.ArrayList(LoadPart) = .empty;
    errdefer parts.deinit(allocator);
    try parts.ensureTotalCapacityPrecise(allocator, part_count);
    const Ctx = struct {
        platform: *const Platform,
        store: *const TensorStore,
        allocator: std.mem.Allocator,
        buffers: []*Buffer,
        specs: *std.ArrayList(LoadSpec),
        computed: *std.ArrayList(Computed),
        parts: *std.ArrayList(LoadPart),
        err: ?PrepareError = null,
    };
    var ctx: Ctx = .{
        .platform = platform,
        .store = store,
        .allocator = allocator,
        .buffers = flattened,
        .specs = &specs,
        .computed = &computed,
        .parts = &parts,
    };
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, context: *Ctx) void {
            if (context.err != null) return;
            const binding = context.store.getSourcesById(tensor.id) orelse {
                load_log.err("tensor {} {f} has no checkpoint binding", .{ tensor.id, tensor.shape() });
                context.err = error.InvalidBinding;
                return;
            };
            if (binding.program != null) {
                context.computed.appendAssumeCapacity(.{ .id = tensor.id, .shape = tensor.shape(), .output = context.buffers[i], .binding = binding });
                return;
            }
            const first_part = context.parts.items.len;
            if (binding.layout) |layout| {
                appendCompositeParts(context.allocator, context.parts, tensor.id, tensor.shape(), binding.tensors, binding.memory, layout) catch |err| {
                    context.err = err;
                    return;
                };
            } else {
                std.debug.assert(binding.tensors.len == 1);
                if (binding.tensors[0].byteSize() == 0) {
                    load_log.err("tensor {} has empty source {s} in {s}", .{ tensor.id, binding.tensors[0].name, binding.tensors[0].file_uri });
                    context.err = error.InvalidBinding;
                    return;
                }
                context.parts.appendAssumeCapacity(.whole(binding.tensors[0]));
            }
            if (!backend.supportsMemory(context.platform, binding.memory)) {
                load_log.err("tensor {} from {s} is bound to {t} memory, which {t} cannot take", .{ tensor.id, binding.tensors[0].name, binding.memory, context.platform.target });
                context.err = error.InvalidBinding;
                return;
            }
            context.specs.appendAssumeCapacity(.{
                .shape = tensor.shape(),
                .memory = binding.memory,
                .output = context.buffers[i],
                .parts = context.parts.items[first_part..],
                .zero_fill = binding.layout != null,
            });
        }
    }.call, .{&ctx});
    if (ctx.err) |err| return err;
    // The capacity was exact, so the specs' slices point into the very
    // allocation returned.
    std.debug.assert(parts.items.len == parts.capacity);
    const owned_specs = try specs.toOwnedSlice(allocator);
    errdefer allocator.free(owned_specs);
    return .{
        .specs = owned_specs,
        .parts = parts.items,
        .computed = try computed.toOwnedSlice(allocator),
    };
}

/// A strided part with few chunks, or with large ones, expands into one
/// part per chunk, each read and transferred on its own: at most
/// `max_part_chunks` chunks, or chunks of at least `large_chunk_bytes`. A
/// composite with any other strided part goes to the backend whole, as
/// strided parts (`backend.Lattice`), which the direct backend reads by
/// destination ranges.
const max_part_chunks = 64;
const large_chunk_bytes = 1 << 20;

fn expandable(part: TensorStore.Part) bool {
    return part.dims.len == 0 or part.chunks() <= max_part_chunks or part.width >= large_chunk_bytes;
}

fn expandsAll(layout: *const TensorStore.Layout) bool {
    for (layout.parts) |part| {
        if (!expandable(part)) return false;
    }
    return true;
}

/// The load parts a bound tensor takes, one for a plain binding.
fn loadPartCount(binding: TensorStore.Binding) usize {
    const layout = binding.layout orelse return 1;
    return layoutPartCount(layout);
}

/// The load parts `layout` takes: its chunks when every part expands, else
/// one per part.
fn layoutPartCount(layout: *const TensorStore.Layout) usize {
    if (!expandsAll(layout)) return layout.parts.len;
    var n: usize = 0;
    for (layout.parts) |part| n += part.chunks();
    return n;
}

/// Appends a composite layout's parts in destination order, each expanded
/// into its contiguous chunks or, when one of them does not expand, all as
/// strided parts. They are checked against the `shape` they fill: the
/// layout's size and sources (`tensors`), then what the backends check
/// themselves (`checkParts`: inside each source and the tensor, disjoint).
/// Strided parts must not go to a host `memory` space.
fn appendCompositeParts(
    allocator: std.mem.Allocator,
    parts: *std.ArrayList(LoadPart),
    id: Tensor.Id,
    shape: Shape,
    tensors: []const *safetensors.Tensor,
    memory: Memory.Kind,
    layout: *const TensorStore.Layout,
) error{InvalidBinding}!void {
    const key = if (tensors.len != 0) tensors[0].name else "no source";
    if (layout.shape.byteSize() != shape.byteSize()) {
        load_log.err("composite tensor {} {f} from {s}: its layout fills {f}", .{ id, shape, key, layout.shape });
        return error.InvalidBinding;
    }
    if (layout.parts.len == 0) {
        load_log.err("composite tensor {} {f} from {s} has no parts", .{ id, shape, key });
        return error.InvalidBinding;
    }
    const strided = !expandsAll(layout);
    if (strided and !backend.countsAgainstDevice(memory)) {
        load_log.err("composite tensor {} {f} from {s}: strided parts into {t} memory, which this loader does not assemble", .{ id, shape, key, memory });
        return error.InvalidBinding;
    }
    const first = parts.items.len;
    for (layout.parts) |part| {
        if (part.source >= tensors.len) {
            load_log.err("composite tensor {} {f} from {s}: a part reads source {d} of {d}", .{ id, shape, key, part.source, tensors.len });
            return error.InvalidBinding;
        }
        const source = tensors[part.source];
        if (source.byteSize() == 0) {
            load_log.err("composite tensor {} has empty source {s} in {s}", .{ id, source.name, source.file_uri });
            return error.InvalidBinding;
        }
        if (strided) {
            parts.appendAssumeCapacity(.{
                .source = source,
                .src_offset = part.src_offset,
                .dst_offset = part.dst_offset,
                .width = part.width,
                .dims = part.dims,
            });
            continue;
        }
        var chunk: u64 = 0;
        while (chunk < part.chunks()) : (chunk += 1) {
            var rest = chunk;
            var src_offset = part.src_offset;
            var dst_offset = part.dst_offset;
            var d = part.dims.len;
            while (d > 0) {
                d -= 1;
                const index = rest % part.dims[d].count;
                rest /= part.dims[d].count;
                src_offset += index * part.dims[d].src_stride;
                dst_offset += index * part.dims[d].dst_stride;
            }
            parts.appendAssumeCapacity(.{
                .source = source,
                .src_offset = src_offset,
                .dst_offset = dst_offset,
                .width = part.width,
            });
        }
    }
    const appended = parts.items[first..];
    std.mem.sort(LoadPart, appended, {}, struct {
        fn lessThan(_: void, lhs: LoadPart, rhs: LoadPart) bool {
            return lhs.dst_offset < rhs.dst_offset;
        }
    }.lessThan);
    backend.checkParts(allocator, appended, shape.byteSize(), true) catch |err| {
        load_log.err("composite tensor {} {f} of {d} bytes from {s}: {s}", .{ id, shape, shape.byteSize(), key, switch (err) {
            error.UnsupportedLayout => "strided parts this loader cannot prove disjoint",
            else => "parts overlap or leave the sources or the tensor",
        } });
        return error.InvalidBinding;
    };
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
    const min_block = @max(profile.direct_io_alignment orelse 1, 1);
    for (opts.dma_block_sizes) |block_size| {
        if (block_size < min_block or block_size > max_request) {
            load_log.err("invalid loader options: profile={s}, dma block size {d}, expected {d}..{d}: InvalidOptions", .{ profile.name, block_size, min_block, max_request });
            return error.InvalidOptions;
        }
    }
    // A source request is the larger of the read chunk and the block
    // calibration selected, and a whole number of blocks (direct `Sizing`).
    // Checked for every block it may select: a candidate, or on CPU the
    // default.
    for ([_][]const usize{ opts.dma_block_sizes, &.{dma_calibration.Result.default.block_size} }) |block_sizes| {
        for (block_sizes) |block_size| {
            if (@max(profile.read_chunk_size, block_size) % block_size != 0) {
                load_log.err("invalid loader options: profile={s}, read_chunk_size={d} is not a multiple of the dma block size {d}: InvalidOptions", .{ profile.name, profile.read_chunk_size, block_size });
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
/// mesh and the replicated mesh, agree when they place alike.
fn checkOutput(platform: *const Platform, name: []const u8, expected: Shape, exe: *const Exe) error{InvalidBinding}!void {
    if (exe.output_shapes.len != 1 or !expected.eql(exe.output_shapes[0])) {
        load_log.err("program {s} computing {f} returns {d} outputs, the first {f}", .{ name, expected, exe.output_shapes.len, if (exe.output_shapes.len != 0) exe.output_shapes[0] else expected });
        return error.InvalidBinding;
    }
    const actual = exe.output_shapes[0];
    const want = Sharding.Placement.init(backend.placedShape(platform, expected).packedShape()) catch |err| {
        load_log.err("program {s}: placement of {f}: {s}", .{ name, expected, @errorName(err) });
        return error.InvalidBinding;
    };
    const got = Sharding.Placement.init(backend.placedShape(platform, actual).packedShape()) catch |err| {
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
    registry: safetensors.TensorRegistry,
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

        self.registry = .init(allocator);
        errdefer self.registry.deinit();
        try self.registry.registerTensor(.{
            .file_uri = path,
            .name = "value",
            .shape = .init(.{contents.len}, .u8),
            .offset = 0,
        });
        try self.registry.registerTensor(.{
            .file_uri = path,
            .name = "second",
            .shape = .init(.{second_contents.len}, .u8),
            .offset = contents.len,
        });
        try self.registry.registerTensor(.{
            .file_uri = missing_path,
            .name = "missing",
            .shape = .init(.{contents.len}, .u8),
            .offset = 0,
        });
        try self.registry.registerTensor(.{
            .file_uri = path,
            .name = "empty",
            .shape = .init(.{0}, .u8),
            .offset = 0,
        });
        self.platform = testing.env();
        self.store = .fromRegistry(allocator, &self.registry, &.{self.platform.replicated_mesh});
        self.value = self.store.view().createReplicatedTensor("value", null);
        self.second = self.store.view().createReplicatedTensor("second", null);
        self.missing = self.store.view().createReplicatedTensor("missing", null);
        self.empty = self.store.view().createReplicatedTensor("empty", null);
    }

    fn deinit(self: *LoaderTestFixture) void {
        self.store.deinit();
        self.registry.deinit();
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
    var other_store = TensorStore.fromRegistry(allocator, &fixture.registry, &.{fixture.platform.replicated_mesh});
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
    try std.testing.expect(fixture.store.getSourcesById(model.a.id).?.program != null);
    try std.testing.expect(fixture.store.sourceOf(model.a) == null);
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
    const pinned = fixture.store.view().createHostPinnedTensor("second", null, .replicated, .{});
    try std.testing.expectEqual(Memory.Kind.host_pinned, fixture.store.getSourcesById(pinned.id).?.memory);

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
    const devices = backend.placedShape(platform, pinned.shape())._sharding.mesh.?.devicesInCanonicalOrder();
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
        try std.testing.expectEqualSlices(u64, loader.submitted_bytes, allocated);
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
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try registry.registerTensor(.{ .name = "value", .file_uri = path, .shape = shape, .offset = 0 });
    try registry.registerTensor(.{ .name = "weight", .file_uri = path, .shape = shape, .offset = 2 * @sizeOf(i32) });
    var store: TensorStore = .fromRegistry(allocator, &registry, &.{platform.replicated_mesh});
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
    const devices = backend.placedShape(fixture.platform, fixture.value.shape())._sharding.mesh.?.devicesInCanonicalOrder().len;
    var torn: usize = 0;
    meta.forEachVisit(&buffers, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer, shards: usize, count: *usize) void {
            if (buffer._shards.len != 0 and buffer._shards.len != shards) count.* += 1;
        }
    }.call, .{ devices, &torn });
    try std.testing.expectEqual(0, torn);
    try std.testing.expectEqual(0, buffers.computed._shards.len);
}

test "submitted bytes match the backend's allocations once everything landed" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const placed = try allocator.alloc(u64, fixture.platform.devices.len);
    defer allocator.free(placed);
    @memset(placed, 0);
    try addPlacementBytes(placed, fixture.platform, fixture.value.shape(), .default, "value");
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
    try std.testing.expectEqualSlices(u64, loader.submitted_bytes, allocated);
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
    if ((try vfs.loadProfile("/model.safetensors")).direct_io_alignment) |alignment| {
        const small = [_]usize{alignment / 2};
        if (Loader.init(allocator, vfs.io(), platform, .{ .checkpoint = "/model.safetensors", .dma_block_sizes = &small })) |value| {
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
    // `init` allocates its three per-device slices, then the set-up's
    // first allocation, the backend, fails.
    var failing: std.testing.FailingAllocator = .init(std.testing.allocator, .{ .fail_index = 3 });
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
    var http: VFS.HTTP = try .initWithOptions(allocator, io, &client, .http, .{
        .max_retries = 5,
        .retry_initial_delay = .fromMilliseconds(20),
        .retry_max_delay = .fromMilliseconds(20),
        .max_hold = .fromSeconds(5),
        .throttle_budget = .fromSeconds(30),
    });
    defer http.deinit();
    var vfs: VFS = try .init(allocator, io);
    defer vfs.deinit();
    try vfs.registerBackend("http", http.backend());
    const stats = http.backend().read_stats.?;

    var url_buffer: [128]u8 = undefined;
    const url = try std.fmt.bufPrint(&url_buffer, "http://127.0.0.1:{d}/object", .{server.port()});

    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    const names = [tensors][]const u8{ "a", "b", "c", "d" };
    for (names, 0..) |name, index| {
        try registry.registerTensor(.{
            .file_uri = url,
            .name = name,
            .shape = .init(.{tensor_bytes}, .u8),
            .offset = 2 * index * tensor_bytes,
        });
    }
    const platform = testing.env();

    var store: TensorStore = .fromRegistry(allocator, &registry, &.{platform.replicated_mesh});
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

    // The checkpoint's VFS backend, not the local profile, sized the load.
    const profile = loader.backend.?.direct.load_profile;
    try std.testing.expectEqualStrings("http", profile.name);
    try std.testing.expect(profile.high_latency and profile.stats != null);
}

test "a rejected binding or a program of the wrong output fails the loader" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    // Two rows of two, not the four bytes the tensor holds.
    const Rows = struct {
        pub fn forward(_: @This(), _: Shape, inputs: []const Tensor) Tensor {
            return inputs[0].reshape(.{ 2, 2 });
        }
    };
    // The program places its output on the platform's devices, the tensor
    // expects other ones: same dims, different placement.
    var foreign_physical = fixture.platform.replicated_mesh.physical.*;
    const foreign_devices = try allocator.dupe(Sharding.Device, foreign_physical.devices_in_canonical_order);
    defer allocator.free(foreign_devices);
    for (foreign_devices) |*device| device.id += 100;
    foreign_physical.devices_in_canonical_order = foreign_devices;
    var foreign_mesh = fixture.platform.replicated_mesh.*;
    foreign_mesh.physical = &foreign_physical;
    var misplaced = fixture.value;
    misplaced._shape._sharding.mesh = &foreign_mesh;

    // No binding, an empty source, and a program of the wrong shape or
    // placement.
    const Case = struct { tensor: Tensor, at_wait: bool = false };
    const cases = [_]Case{
        .{ .tensor = Tensor.fromShape(fixture.value.shape()) },
        .{ .tensor = fixture.empty },
        .{ .tensor = fixture.computed("value", fixture.value, Rows{}), .at_wait = true },
        .{ .tensor = fixture.computed("value", misplaced, LoaderTestFixture.Identity{}), .at_wait = true },
    };
    for (cases) |case| {
        var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
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

test "loader reports source size mismatches" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var tensor = fixture.value;
    tensor._shape = .init(.{8}, .u8);
    var output: Buffer = undefined;
    try loader.awaitSetup();
    // The direct backend checks sizes as it plans, the buffered one as it
    // reads.
    if (loader.backend.? == .direct) {
        try std.testing.expectError(error.InvalidBinding, loader.start(&tensor, &output, &fixture.store, .none));
    } else {
        try loader.start(&tensor, &output, &fixture.store, .none);
        try std.testing.expectError(error.InvalidBinding, loader.wait());
    }
    try LoaderTestFixture.expectFailed(&loader, error.InvalidBinding);
    try std.testing.expectError(error.InvalidBinding, loader.wait());
}

/// Two safetensors files for composite loads, every row 16 bytes (8 u16):
/// `first` holds `a` [3, 8] then `b` [2, 8], adjacent, then `spacer`, then
/// `q` [8, 8], `k` [4, 8] and `v` [4, 8]; `second` holds `plain` [4, 8],
/// then `c` [4, 8], the experts `e{0..3}.w1` and `e{0..3}.w3` [2, 8] and
/// `wide` [64, 8]. `empty` [0, 8] is registered without bytes. Pinned after
/// `init`.
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
        var store: TensorStore = .fromRegistry(allocator, &fixture.registry, &.{&tp});
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
                        .bind(.tp, partitioning, .{ .memory = memory }),
                    // The padded engram-style table: one source.
                    .padded = rows(view, "c").pad(0, .{ .row = Source.Pad{ .high = 4 } }).bind(.tp, partitioning, .{ .memory = memory }),
                    // Out of file order, on the platform's replicated mesh.
                    .reversed = Source.concatenate(&.{ rows(view, "b"), rows(view, "a") }, .row)
                        .pad(0, .{ .row = Source.Pad{ .high = 1 } })
                        .bind(.replicated, .{}, .{ .memory = memory }),
                    .plain = rows(view, "plain").bind(.tp, partitioning, .{ .memory = memory }),
                };
                try std.testing.expect(store.getSourcesById(model.plain.id).?.layout == null);
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
                    const ordered = backend.placedShape(platform, buffer.shape())._sharding.mesh.?.devicesInCanonicalOrder();
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
                    try std.testing.expectEqualSlices(u64, loader.submitted_bytes, allocated);
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
        var store: TensorStore = .fromRegistry(allocator, &fixture.registry, &.{&tp});
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
            .qkv = Source.concatenate(&members, .r).merge(.{ .row = .{ .group, .r } }).bind(.tp, .{ .row = .model }, .{}),
            // Experts are whole rows of the merged axis, one or two per device.
            .concatenated = Source.concatenate(&experts, .row)
                .merge(.{ .rows = .{ .expert, .row } })
                .bind(.tp, .{ .rows = .model }, .{}),
            .interleaved = Source.stack(&experts, 2, .proj)
                .merge(.{ .rows = .{ .expert, .row, .proj } })
                .bind(.tp, .{ .rows = .model }, .{}),
            .head = head.bind(.replicated, .{}, .{}),
            .tail = tail.bind(.tp, .{ .row = .model }, .{}),
        };
        const qkv_layout = store.getSourcesById(model.qkv.id).?.layout.?;
        try std.testing.expectEqual(@as(usize, 3), qkv_layout.parts.len);
        for (qkv_layout.parts) |part| try std.testing.expectEqual(@as(u64, device_count), part.chunks());

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
            var store: TensorStore = .fromRegistry(allocator, &registry, &.{&tp});
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
                .by_rows = rows.bind(.tp, .{ .row = .model }, .{}),
                .by_columns = rows.bind(.tp, .{ .col = .model }, .{}),
                .scales_by_rows = repeated.bind(.tp, .{ .row = .model }, .{}),
                .scales_by_columns = repeated.bind(.tp, .{ .col = .model }, .{}),
                .transposed = flipped.bind(.replicated, .{}, .{}),
                .transposed_by_columns = flipped.bind(.tp, .{ .col = .model }, .{}),
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

test "composite layouts that do not fit their tensor fail the loader before submission" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = testing.env();
    var store: TensorStore = .fromRegistry(allocator, &fixture.registry, &.{platform.replicated_mesh});
    defer store.deinit();
    const destination: Shape = .init(.{ 8, 8 }, .u16);
    const Part = TensorStore.Part;
    const Composite = struct {
        fn bind(store_: *TensorStore, registry: *safetensors.TensorRegistry, names: []const []const u8, shape: Shape, parts: []const Part) Tensor {
            return bindAs(store_, registry, names, shape, shape, parts);
        }

        /// A tensor of `shape` whose layout fills `filled`.
        fn bindAs(store_: *TensorStore, registry: *safetensors.TensorRegistry, names: []const []const u8, shape: Shape, filled: Shape, parts: []const Part) Tensor {
            const arena = store_.arena.allocator();
            const tensors = arena.alloc(*safetensors.Tensor, names.len) catch unreachable;
            for (tensors, names) |*tensor, name| tensor.* = registry.tensors.getPtr(name).?;
            const layout = arena.create(TensorStore.Layout) catch unreachable;
            layout.* = .{ .shape = filled, .parts = parts };
            const tensor: Tensor = .fromShape(shape);
            store_.putSourcesNoClobber(tensor.id, .{ .tensors = tensors, .layout = layout }) catch unreachable;
            return tensor;
        }
    };
    const s = &store;
    const r = &fixture.registry;

    const rejected = [_]Tensor{
        // No part at all, and a part of a source the binding lacks.
        Composite.bind(s, r, &.{"a"}, destination, &.{}),
        Composite.bind(s, r, &.{"a"}, destination, &.{.{ .source = 1, .dst_offset = 0, .width = 16 }}),
        // A layout of another size than its tensor.
        Composite.bindAs(s, r, &.{"a"}, destination, .init(.{ 4, 8 }, .u16), &.{.{ .dst_offset = 0, .width = 48 }}),
        // Past the end of its source, and of the tensor.
        Composite.bind(s, r, &.{"a"}, destination, &.{.{ .src_offset = 16, .dst_offset = 0, .width = 48 }}),
        Composite.bind(s, r, &.{"a"}, destination, &.{.{ .dst_offset = 96, .width = 48 }}),
        // Overlapping parts.
        Composite.bind(s, r, &.{ "a", "b" }, destination, &.{
            .{ .source = 0, .dst_offset = 0, .width = 48 },
            .{ .source = 1, .dst_offset = 32, .width = 32 },
        }),
        Composite.bind(s, r, &.{ "a", "empty" }, destination, &.{
            .{ .source = 0, .dst_offset = 0, .width = 48 },
            .{ .source = 1, .dst_offset = 48, .width = 16 },
        }),
        // Strided parts whose runs overlap; two of over a thousand runs
        // each that interleave with different strides, which the checks
        // cannot prove disjoint.
        Composite.bind(s, r, &.{"wide"}, .init(.{ 8, 64 }, .u16), &.{
            .{ .dst_offset = 0, .width = 2, .dims = &.{.{ .count = 128, .src_stride = 8, .dst_stride = 4 }} },
            .{ .dst_offset = 2, .width = 4, .dims = &.{.{ .count = 128, .src_stride = 8, .dst_stride = 4 }} },
        }),
        Composite.bind(s, r, &.{"wide"}, .init(.{ 64, 64 }, .u16), &.{
            .{ .dst_offset = 0, .width = 1, .dims = &.{.{ .count = 2000, .src_stride = 0, .dst_stride = 4 }} },
            .{ .dst_offset = 2, .width = 1, .dims = &.{.{ .count = 1300, .src_stride = 0, .dst_stride = 6 }} },
        }),
        // Strided parts into host memory.
        host: {
            const tensor = Composite.bind(s, r, &.{"wide"}, .init(.{ 8, 64 }, .u16), &.{.{ .dst_offset = 0, .width = 2, .dims = &.{
                .{ .count = 8, .src_stride = 2, .dst_stride = 128 },
                .{ .count = 64, .src_stride = 16, .dst_stride = 2 },
            } }});
            s.setMemory(tensor, .host_pinned);
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

    // A part placed past a gap, the rest zero; parts listed out of
    // destination order; one source feeding two parts; a strided part of few
    // runs, expanded; and a transpose, 512 runs of two bytes, gathered.
    const sparse = Composite.bind(s, r, &.{"b"}, destination, &.{.{ .src_offset = 16, .dst_offset = 64, .width = 16 }});
    const descending = Composite.bind(s, r, &.{ "a", "b" }, destination, &.{
        .{ .source = 0, .dst_offset = 64, .width = 48 },
        .{ .source = 1, .dst_offset = 0, .width = 32 },
    });
    const swapped = Composite.bind(s, r, &.{"a"}, destination, &.{
        .{ .src_offset = 32, .dst_offset = 0, .width = 16 },
        .{ .src_offset = 0, .dst_offset = 16, .width = 32 },
    });
    const strided = Composite.bind(s, r, &.{"a"}, destination, &.{.{ .dst_offset = 0, .width = 16, .dims = &.{
        .{ .count = 3, .src_stride = 16, .dst_stride = 32 },
    } }});
    const transposed = Composite.bind(s, r, &.{"wide"}, .init(.{ 8, 64 }, .u16), &.{.{ .dst_offset = 0, .width = 2, .dims = &.{
        .{ .count = 8, .src_stride = 2, .dst_stride = 128 },
        .{ .count = 64, .src_stride = 16, .dst_stride = 2 },
    } }});
    const Model = struct { sparse: Tensor, descending: Tensor, swapped: Tensor, strided: Tensor, transposed: Tensor };
    const model: Model = .{ .sparse = sparse, .descending = descending, .swapped = swapped, .strided = strided, .transposed = transposed };
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
    try load_testing.expectShards(allocator, io, &buffers.descending, &expected);
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
    var store: TensorStore = .fromRegistry(allocator, &fixture.registry, &.{platform.replicated_mesh});
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
    const inputs = [_]Source.Input{ assembled.place(.replicated, .{}), c.place(.replicated, .{}) };
    const tensor = Source.bindProgram(&inputs, c.shape(), AddRows{});
    const binding = store.getSourcesById(tensor.id).?;
    const program = binding.program.?;
    try std.testing.expectEqual(2, program.inputs.len);
    try std.testing.expect(program.as(AddRows) != null and program.as(LoaderTestFixture.Identity) == null);
    try std.testing.expect(program.inputs[0].whole(binding.tensors) == null);
    try std.testing.expect(program.inputs[1].whole(binding.tensors) != null);
    try std.testing.expect(store.sourceOf(tensor) == null);
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
    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try load_testing.writeSafetensors(allocator, io, tmp.dir, "table.safetensors", &registry, &.{
        .{ .name = "table", .dtype = "U8", .dims = &.{ @intCast(rows), @intCast(row_bytes) }, .bytes = padded[0 .. rows * row_bytes] },
    });
    var store: TensorStore = .fromRegistry(allocator, &registry, &.{mesh});
    defer store.deinit();
    const source = store.view().source("table", .{ .row, .channel })
        .pad(0, .{ .row = TensorStore.Source.Pad{ .high = @intCast(padded_rows - rows) } });
    const table = switch (placement) {
        .rows => source.bind(.model, .{ .row = .model }, .{ .memory = .host_pinned }),
        .replicated => source.bind(.model, .{}, .{ .memory = .host_pinned }),
    };
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
        .rows => source.bind(.model, .{ .row = .model }, .{}),
        .replicated => source.bind(.model, .{}, .{}),
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
