//! Checkpoint loading: `Loader` publishes submissions in FIFO order, admits
//! executable submissions against the device room it measures, and retires
//! them on the caller's task (`loadExecute` admission, `awaitAll`). Backends
//! only read sources into buffers; lookup, admission and execution stay in
//! this shared front end.
const std = @import("std");

const pjrt = @import("pjrt");
const VFS = @import("vfs");

const Buffer = @import("../buffer.zig").Buffer;
const Exe = @import("../exe.zig").Exe;
const mem = @import("../mem.zig");
const Bufferized = mem.Bufferized;
const meta = @import("../meta.zig");
const platform_mod = @import("../platform.zig");
const Platform = platform_mod.Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const admission = @import("execute_admission.zig");
const backend = @import("backend.zig");
const Backend = backend.Backend;
const LoadSpec = backend.LoadSpec;
const dma_calibration = @import("dma_calibration.zig");
const limits = @import("limits.zig");
const load_log = @import("log.zig").load;
const TensorStore = @import("TensorStore.zig");

const DeliveryMap = std.AutoHashMapUnmanaged(Tensor.Id, void);

const PrepareError = std.mem.Allocator.Error || error{ TensorNotFound, EmptyTensor, TransformedTensorNotDelivered };
const BindingError = error{ ExecutablePlatformMismatch, ExecutableInputCountMismatch, ExecutableOutputCountMismatch, ExecutableInputShapeMismatch, ExecutableOutputShapeMismatch, ExecutablePlacementMismatch };
/// Placing the sources on the devices, then the backend's own submission.
const SubmitError = Sharding.Error || backend.SubmitError || error{Overflow};

/// Loads checkpoint sources and optionally executes bindings over them.
/// The platform and options' borrowed values must outlive the loader. Each
/// submission borrows its store's source metadata, outputs, and optional
/// progress parent until `awaitAll` or `deinit` returns.
/// Submit and await serially on the owning task; source reads and transfers
/// run concurrently inside the selected backend. Submissions are retired in
/// publish order: `loadExecute` retires older ones when its own does not fit
/// the room the devices report, and `awaitAll` retires the rest.
pub const Loader = struct {
    pub const AwaitError = backend.AwaitError || std.mem.Allocator.Error;
    pub const LoadError = PrepareError || SubmitError || AwaitError;
    pub const LoadExecuteError = BindingError || SubmitError || AwaitError || error{ TensorNotFound, EmptyTensor };

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    backend: Backend,
    /// Submissions published and not yet retired, oldest first.
    pending: std.Deque(PendingSubmission) = .empty,
    /// Transformed tensors a `loadExecute` was published for. `load` skips
    /// them; their bytes reach the caller's buffer when that submission is
    /// retired, before any bulk submitted after it.
    delivered: DeliveryMap = .empty,
    /// The first error a retire returned; every later call reports it.
    failure: ?AwaitError = null,
    /// Logical bytes of every submission retired with execution.
    bytes_loaded: usize = 0,
    /// Whether admission can measure the room: the backend counts its
    /// allocations and every device reported an allocator limit at init.
    /// Otherwise `loadExecute` retires everything pending before it
    /// publishes, the old synchronous order.
    memory_supported: bool,
    /// Placement bytes of every published submission, per device.
    submitted_bytes: []u64,
    /// Output and temporary bytes of the pending executable submissions,
    /// per device: what retiring them will take before it frees anything.
    pending_execution: []u64,
    pending_executes: usize = 0,
    submissions: usize = 0,
    admission_retires: usize = 0,
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

    pub const Options = backend.Options;

    /// One executable over a binding. `tensor`'s sources are loaded into
    /// fresh input buffers; when the submission is retired, `exe` runs over
    /// them and its result is written to `output.*`.
    pub const Binding = struct {
        tensor: Tensor,
        output: *Buffer,
        exe: *const Exe,
    };

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        opts: Options,
    ) !Loader {
        try validateOptions(opts);
        try platform.warmupDeviceAllocators(io);
        const selected = try Backend.init(allocator, io, platform, opts);
        errdefer selected.destroy();
        return initWithBackend(allocator, io, platform, selected);
    }

    /// Takes the backend on success only; the caller destroys it otherwise.
    fn initWithBackend(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, selected: Backend) !Loader {
        const devices = platform.devices.len;
        const submitted_bytes = try allocator.alloc(u64, devices);
        errdefer allocator.free(submitted_bytes);
        @memset(submitted_bytes, 0);
        const pending_execution = try allocator.alloc(u64, devices);
        errdefer allocator.free(pending_execution);
        @memset(pending_execution, 0);
        const words = try allocator.alloc(u64, 4 * devices);
        errdefer allocator.free(words);
        var self: Loader = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .backend = selected,
            .memory_supported = false,
            .submitted_bytes = submitted_bytes,
            .pending_execution = pending_execution,
            .scratch = .{
                .words = words,
                .room = words[0..devices],
                .allocated = words[devices .. 2 * devices],
                .inputs = words[2 * devices .. 3 * devices],
                .placed = words[3 * devices .. 4 * devices],
            },
        };
        self.memory_supported = self.probeMemory();
        return self;
    }

    /// Decided once: the CPU plugin logs an unimplemented warning on every
    /// stats call, so admission never asks a device that answered nothing.
    fn probeMemory(self: *Loader) bool {
        if (self.backend != .direct) return false;
        for (self.platform.devices) |device| {
            if (device.memoryStats().bytes_limit == null) return false;
        }
        return true;
    }

    /// Sizing selected during initialization; absent for buffered loading.
    pub fn calibration(self: *const Loader) ?dma_calibration.Result {
        return self.backend.calibration();
    }

    /// Submits every single-source tensor of `model` as one planned
    /// submission. Work may start before this returns. Transformed tensors
    /// must have a `loadExecute` published for them; they are skipped, and
    /// a missing one fails before any work is submitted. A zero-byte source
    /// is `error.EmptyTensor`. Shardings are selected during this call; an
    /// empty slice uses replicated placement. Progress counts loaded
    /// sources, excluding skipped transformed tensors. The caller owns the
    /// estimated total. Never gated by admission: the outputs are the model.
    pub fn load(
        self: *Loader,
        comptime ModelType: type,
        model: *const ModelType,
        buffers: *Bufferized(ModelType),
        store: *const TensorStore,
        shardings: []const Sharding,
        progress: ?*std.Progress.Node,
    ) LoadError!void {
        if (self.failure) |err| return err;
        const specs = try prepareModelLoad(
            self.allocator,
            self.platform,
            store,
            shardings,
            &self.delivered,
            ModelType,
            model,
            buffers,
        );
        defer self.allocator.free(specs);
        try self.submit(specs, &.{}, &.{}, progress);
    }

    /// Submits the sources of every binding as one planned submission, so
    /// adjacent sources of different bindings coalesce into shared reads.
    /// Before publishing, retires the oldest pending submissions (running
    /// their executables on this task) until this one fits the room the
    /// devices report; one submission is always admitted. Retiring the
    /// submission runs the executables in binding order, writes each
    /// output and frees the inputs. Input and output placement come from
    /// each executable. Progress counts input sources across all bindings,
    /// before execution. The caller owns the estimated total.
    pub fn loadExecute(self: *Loader, store: *const TensorStore, bindings: []const Binding, progress: ?*std.Progress.Node) LoadExecuteError!void {
        if (self.failure) |err| return err;
        const executables = try self.allocator.alloc(BoundExecutable, bindings.len);
        var prepared: usize = 0;
        errdefer {
            for (executables[0..prepared]) |*executable| executable.deinit(self.allocator);
            self.allocator.free(executables);
        }
        var source_count: usize = 0;
        for (bindings, executables) |binding, *executable| {
            executable.* = try BoundExecutable.init(self.allocator, self.platform, store, binding);
            prepared += 1;
            source_count += executable.sources.len;
        }
        const execution = try self.allocator.alloc(u64, self.platform.devices.len);
        errdefer self.allocator.free(execution);
        try self.admit(executables, execution);
        try self.delivered.ensureUnusedCapacity(self.allocator, @intCast(bindings.len));
        const specs = try self.allocator.alloc(LoadSpec, source_count);
        defer self.allocator.free(specs);
        var next: usize = 0;
        for (executables) |executable| {
            for (
                executable.sources,
                executable.exe.input_shapes,
                executable.exe.input_shardings,
                executable.inputs,
            ) |source, shape, sharding, *input| {
                if (source.byteSize() == 0) {
                    load_log.debug("executable input {s} in {s} is empty", .{ source.name, source.file_uri });
                    return error.EmptyTensor;
                }
                specs[next] = .{
                    .source = source,
                    .shape = shape,
                    .sharding = sharding.resolve(self.platform),
                    .output = input,
                };
                next += 1;
            }
        }
        try self.submit(specs, executables, execution, progress);
        for (bindings) |binding| self.delivered.putAssumeCapacity(binding.tensor.id, {});
    }

    /// Retires every pending submission in publish order, running their
    /// executables, and returns the first error seen by this loader.
    /// Idempotent.
    pub fn awaitAll(self: *Loader) AwaitError!void {
        while (self.pending.len != 0) self.retireOldest(true) catch {};
        if (self.failure) |err| return err;
    }

    /// Logical bytes of every submission retired with execution so far.
    pub fn bytesLoaded(self: *const Loader) usize {
        return self.bytes_loaded;
    }

    /// Awaits every pending submission without running executables (their
    /// outputs stay unwritten, their inputs are freed), then destroys the
    /// backend.
    pub fn deinit(self: *Loader) void {
        while (self.pending.len != 0) self.retireOldest(false) catch {};
        self.logAdmission();
        self.pending.deinit(self.allocator);
        self.delivered.deinit(self.allocator);
        self.allocator.free(self.submitted_bytes);
        self.allocator.free(self.pending_execution);
        self.allocator.free(self.scratch.words);
        self.backend.destroy();
        self.* = undefined;
    }

    /// Fills `execution` with the submission's output and temporary bytes
    /// per device, then retires pending submissions, oldest first, until
    /// the submission fits the measured room or nothing executable is
    /// pending: a bulk load frees nothing, and once nothing can be retired
    /// for it the submission is admitted whatever its size. Without memory
    /// measurement one executable submission is in flight at a time, the
    /// old synchronous order.
    fn admit(self: *Loader, executables: []const BoundExecutable, execution: []u64) !void {
        const inputs = self.scratch.inputs;
        @memset(inputs, 0);
        @memset(execution, 0);
        for (executables, 0..) |*executable, index| {
            const exe = executable.exe;
            for (exe.input_shapes, exe.input_shardings) |shape, sharding| {
                try addPlacementBytes(inputs, sharding.resolve(self.platform), shape);
            }
            const output_sharding = exe.output_shardings[0].resolve(self.platform);
            try addPlacementBytes(execution, output_sharding, exe.output_shapes[0]);
            // Bindings sharing an executable execute one after the other, so
            // its temporaries are charged once.
            const seen = for (executables[0..index]) |*earlier| {
                if (earlier.exe == exe) break true;
            } else false;
            if (seen) continue;
            const temp = self.compiledTempBytes(exe);
            for (output_sharding.devicesInCanonicalOrder()) |device| execution[device.id] +|= temp;
        }
        var retired: usize = 0;
        var fit = self.measureFit(inputs, execution);
        while (fit != .fits and self.pending_executes != 0) : (retired += 1) {
            try self.retireOldest(true);
            fit = self.measureFit(inputs, execution);
        }
        if (fit == .exceeds and !self.oversized_logged) {
            self.oversized_logged = true;
            load_log.warn("executable submission exceeds the device room alone: inputs={Bi:.2} execution={Bi:.2} room={Bi:.2} (device 0); admitted anyway", .{
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

    /// Whether the submission fits beside the pending executions in the room
    /// the devices report now.
    fn measureFit(self: *Loader, inputs: []const u64, execution: []const u64) Fit {
        if (!self.memory_supported or !self.readRoom()) return .unmeasured;
        return if (admission.admits(self.scratch.room, self.pending_execution, inputs, execution)) .fits else .exceeds;
    }

    /// Refreshes the per-device room; false when a device stopped answering.
    fn readRoom(self: *Loader) bool {
        self.backend.allocatedBytesPerDevice(self.scratch.allocated);
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
    /// them. `execution` holds the output and temporary bytes per device of
    /// an executable submission, empty for a bulk one.
    fn submit(self: *Loader, specs: []const LoadSpec, executables: []BoundExecutable, execution: []u64, progress: ?*std.Progress.Node) SubmitError!void {
        var logical_bytes: usize = 0;
        const placed = self.scratch.placed;
        @memset(placed, 0);
        for (specs) |spec| {
            logical_bytes = try std.math.add(usize, logical_bytes, spec.source.shape.byteSize());
            try addPlacementBytes(placed, spec.sharding, spec.shape);
        }
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const submission = try self.backend.submit(specs, progress);
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
    /// runs its executables when `execute`, frees its inputs either way and
    /// counts its bytes once it ran. The first error is kept in `failure`.
    fn retireOldest(self: *Loader, execute: bool) AwaitError!void {
        var oldest = self.pending.popFront().?;
        defer self.release(&oldest);
        self.retire(&oldest, execute) catch |err| {
            self.failure = self.failure orelse err;
            return err;
        };
    }

    fn retire(self: *Loader, oldest: *PendingSubmission, execute: bool) AwaitError!void {
        try oldest.submission.await();
        if (!execute) return;
        for (oldest.executables) |*executable| {
            executable.execute(self.allocator, self.io) catch |err| {
                if (self.failure == null) load_log.err("prepare executable arguments/results: inputs={d}, output={f}: {s}", .{ executable.inputs.len, executable.exe.output_shapes[0], @errorName(err) });
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
    /// Run in binding order once the reads are done; freed with their
    /// inputs when the submission is retired. Empty for a bulk load.
    executables: []BoundExecutable,
    /// Output and temporary bytes per device; empty for a bulk load.
    execution: []u64,
    logical_bytes: usize,
};

/// Adds the bytes `shape` occupies on each device of `resolved`, the same
/// expression the backend allocates by, so submitted and allocated bytes
/// cancel exactly once everything landed.
fn addPlacementBytes(out: []u64, resolved: Sharding, shape: Shape) !void {
    const bytes: u64 = (try resolved.placement(shape.packedShape())).shape.byteSize();
    for (resolved.devicesInCanonicalOrder()) |device| {
        out[device.id] = try std.math.add(u64, out[device.id], bytes);
    }
}

fn prepareModelLoad(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    shardings: []const Sharding,
    delivered: *const DeliveryMap,
    comptime ModelType: type,
    model: *const ModelType,
    buffers: *Bufferized(ModelType),
) PrepareError![]LoadSpec {
    const tensor_count = meta.count(Tensor, model);
    const flattened = try allocator.alloc(*Buffer, tensor_count);
    defer allocator.free(flattened);
    meta.forEachVisit(buffers, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
            output[i] = buffer;
        }
    }.call, .{flattened});

    var specs: std.ArrayListUnmanaged(LoadSpec) = .empty;
    errdefer specs.deinit(allocator);
    try specs.ensureTotalCapacityPrecise(allocator, tensor_count);
    const Ctx = struct {
        platform: *const Platform,
        store: *const TensorStore,
        shardings: []const Sharding,
        delivered: *const DeliveryMap,
        allocator: std.mem.Allocator,
        buffers: []*Buffer,
        specs: *std.ArrayListUnmanaged(LoadSpec),
        err: ?PrepareError = null,
    };
    var ctx: Ctx = .{
        .platform = platform,
        .store = store,
        .shardings = shardings,
        .delivered = delivered,
        .allocator = allocator,
        .buffers = flattened,
        .specs = &specs,
    };
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, context: *Ctx) void {
            if (context.err != null) return;
            const sources = context.store.getSourcesById(tensor.id) orelse {
                load_log.debug("tensor {} {f} has no checkpoint binding", .{ tensor.id, tensor.shape() });
                context.err = error.TensorNotFound;
                return;
            };
            if (sources.transformed) {
                if (!context.delivered.contains(tensor.id)) {
                    logNotDelivered(context.allocator, tensor, sources.tensors);
                    context.err = error.TransformedTensorNotDelivered;
                }
                return;
            }
            std.debug.assert(sources.tensors.len == 1);
            if (sources.tensors[0].byteSize() == 0) {
                load_log.debug("tensor {} has empty source {s} in {s}", .{ tensor.id, sources.tensors[0].name, sources.tensors[0].file_uri });
                context.err = error.EmptyTensor;
                return;
            }
            const shape = tensor.shape();
            context.specs.appendAssumeCapacity(.{
                .source = sources.tensors[0],
                .shape = shape,
                .sharding = (Sharding.pickSharding(
                    context.shardings,
                    shape,
                    .explicit_axis_binding,
                ) orelse context.platform.replicated_sharding).resolve(context.platform),
                .output = context.buffers[i],
            });
        }
    }.call, .{&ctx});
    if (ctx.err) |err| return err;
    return specs.toOwnedSlice(allocator);
}

fn logNotDelivered(arena: std.mem.Allocator, tensor: *const Tensor, sources: []const *safetensors.Tensor) void {
    const max_names = 8;
    var names: std.Io.Writer.Allocating = .init(arena);
    defer names.deinit();
    for (sources[0..@min(sources.len, max_names)], 0..) |source, i| {
        names.writer.print("{s}{s}", .{ if (i != 0) ", " else "", source.name }) catch break;
    }

    if (sources.len > max_names) {
        names.writer.print(" and {} more", .{sources.len - max_names}) catch {};
    }

    load_log.debug("Transformed tensor {} {f} has no loadExecute submitted before load; sources: {s}", .{ tensor.id, tensor.shape(), names.written() });
}

fn validateOptions(opts: Loader.Options) error{InvalidOptions}!void {
    const read_chunk_size = opts.load_profile.read_chunk_size;
    _ = limits.effectiveSourceRequestSize(read_chunk_size, 0) catch |err| {
        load_log.err("invalid loader options: profile={s}, read_chunk_size={d}, expected 1..{d}: {s}", .{ opts.load_profile.name, read_chunk_size, limits.max_read_request_size, @errorName(err) });
        return err;
    };
    if (opts.read_parallelism) |width| {
        if (width == 0 or width > limits.max_read_parallelism) {
            load_log.err("invalid loader options: read_parallelism={d}, expected 1..{d}: InvalidOptions", .{ width, limits.max_read_parallelism });
            return error.InvalidOptions;
        }
    }
}

/// One executable of a `loadExecute` submission and the input shells its
/// sources are loaded into.
const BoundExecutable = struct {
    sources: []const *safetensors.Tensor,
    inputs: []Buffer,
    output: *Buffer,
    exe: *const Exe,

    fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        store: *const TensorStore,
        binding: Loader.Binding,
    ) !BoundExecutable {
        const sources = (store.getSourcesById(binding.tensor.id) orelse {
            load_log.debug("executable binding tensor {} {f} has no checkpoint sources", .{ binding.tensor.id, binding.tensor.shape() });
            return error.TensorNotFound;
        }).tensors;
        try validateExecutableBinding(platform, binding.tensor, sources, binding.exe);
        const inputs = try allocator.alloc(Buffer, sources.len);
        for (inputs, binding.exe.input_shapes, binding.exe.input_shardings) |*input, shape, sharding| {
            input.* = .{
                ._platform = platform,
                ._shape = shape,
                ._sharding = sharding.resolve(platform),
                ._shards = .empty,
            };
        }
        return .{
            .sources = sources,
            .inputs = inputs,
            .output = binding.output,
            .exe = binding.exe,
        };
    }

    /// Runs the executable over the loaded inputs on the calling task.
    fn execute(self: *const BoundExecutable, allocator: std.mem.Allocator, io: std.Io) !void {
        var args = try self.exe.args(allocator);
        defer args.deinit(allocator);
        var results = try self.exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{self.inputs});
        self.exe.callOpts(io, args, &results, .{ .wait = true });
        self.output.* = results.get(Buffer);
    }

    /// Frees the inputs; a shell the loader never wrote owns no shards.
    fn deinit(self: *BoundExecutable, allocator: std.mem.Allocator) void {
        for (self.inputs) |*input| input.deinit();
        allocator.free(self.inputs);
    }
};

fn validateExecutableBinding(
    platform: *const Platform,
    tensor: Tensor,
    sources: []const *safetensors.Tensor,
    exe: *const Exe,
) BindingError!void {
    errdefer |err| load_log.debug("executable binding for tensor {} {f}: {s}; sources={d}, input_shapes={d}, input_shardings={d}, output_shapes={d}, output_shardings={d}, platform_matches={}", .{ tensor.id, tensor.shape(), @errorName(err), sources.len, exe.input_shapes.len, exe.input_shardings.len, exe.output_shapes.len, exe.output_shardings.len, exe.platform == platform });
    if (exe.platform != platform) return error.ExecutablePlatformMismatch;
    if (exe.output_shapes.len != 1 or exe.output_shardings.len != 1)
        return error.ExecutableOutputCountMismatch;
    if (exe.input_shapes.len != sources.len or exe.input_shardings.len != sources.len)
        return error.ExecutableInputCountMismatch;
    if (!tensor.shape().eql(exe.output_shapes[0])) {
        load_log.debug("executable output shape: expected {f}, actual {f}", .{ tensor.shape(), exe.output_shapes[0] });
        return error.ExecutableOutputShapeMismatch;
    }
    for (sources, exe.input_shapes, exe.input_shardings) |source, shape, sharding| {
        if (!source.shape.eql(shape)) {
            load_log.debug("executable input {s}: source shape {f}, executable shape {f}", .{ source.name, source.shape, shape });
            return error.ExecutableInputShapeMismatch;
        }
        try validateExecutableSharding(platform, sharding, exe.num_devices);
    }
    try validateExecutableSharding(platform, exe.output_shardings[0], exe.num_devices);
}

fn validateExecutableSharding(
    platform: *const Platform,
    unresolved: Sharding,
    expected_devices: usize,
) error{ExecutablePlacementMismatch}!void {
    const sharding = unresolved.resolve(platform);
    const devices = sharding.devicesInCanonicalOrder();
    if (devices.len != expected_devices) {
        load_log.debug("executable placement: expected {d} devices, actual {d}", .{ expected_devices, devices.len });
        return error.ExecutablePlacementMismatch;
    }
    for (devices) |device| {
        if (device.id >= platform.devices.len) {
            load_log.debug("executable placement device {d} exceeds platform device count {d}", .{ device.id, platform.devices.len });
            return error.ExecutablePlacementMismatch;
        }
    }
}

/// A four-byte tensor `value`, a four-byte `second`, a `missing` entry in a
/// file that does not exist, an `empty` tensor, a CPU platform and an
/// identity executable over `value`'s shape. Pinned after `init`.
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
    platform: *Platform,
    exe: Exe,
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
        self.store = .fromRegistry(allocator, &self.registry);
        errdefer self.store.deinit();
        self.value = self.store.view().createTensor("value", null, .replicated);
        self.second = self.store.view().createTensor("second", null, .replicated);
        self.missing = self.store.view().createTensor("missing", null, .replicated);
        self.empty = self.store.view().createTensor("empty", null, .replicated);

        self.platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
            return error.SkipZigTest;
        errdefer self.platform.deinit(allocator, io);
        const Identity = struct {
            fn call(input: Tensor) Tensor {
                return input;
            }
        };
        self.exe = try self.platform.compileFn(allocator, io, Identity.call, .{self.value}, .{});
    }

    fn deinit(self: *LoaderTestFixture, allocator: std.mem.Allocator, io: std.Io) void {
        self.exe.deinit();
        self.platform.deinit(allocator, io);
        self.store.deinit();
        self.registry.deinit();
        self.tmp.cleanup();
    }

    const BackendKind = enum { direct, buffered };
    /// The platform picks the backend; the tests run both on the CPU
    /// platform, since nothing else in the tree constructs the buffered one.
    const backends = [_]BackendKind{ .direct, .buffered };

    fn loader(self: *LoaderTestFixture, allocator: std.mem.Allocator, io: std.Io, kind: BackendKind) !Loader {
        const opts: Loader.Options = .{ .read_parallelism = 2 };
        return switch (kind) {
            .direct => try Loader.init(allocator, io, self.platform, opts),
            .buffered => buffered: {
                const selected = try Backend.initBuffered(
                    allocator,
                    io,
                    self.platform,
                    opts.readWidth(),
                    opts.load_profile,
                );
                errdefer selected.destroy();
                break :buffered try Loader.initWithBackend(allocator, io, self.platform, selected);
            },
        };
    }

    fn binding(self: *const LoaderTestFixture, tensor: Tensor, output: *Buffer) Loader.Binding {
        return .{ .tensor = tensor, .output = output, .exe = &self.exe };
    }

    /// An unexpected success leaves a submission pending over borrowed
    /// outputs: retire it before the outputs go out of scope.
    fn expectLoadError(subject: *Loader, expected: anyerror, result: anyerror!void) !void {
        if (result) |_| subject.awaitAll() catch {} else |_| {}
        try std.testing.expectError(expected, result);
    }

    fn expectContents(allocator: std.mem.Allocator, io: std.Io, buffer: *const Buffer, expected: []const u8) !void {
        const loaded = try buffer.toSliceAlloc(allocator, io);
        defer loaded.free(allocator);
        try std.testing.expectEqualSlices(u8, expected, loaded.constData());
    }
};

test "loader retires submissions in FIFO order and counts bytes once" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();
        if (kind == .direct) {
            try std.testing.expectEqual(dma_calibration.Result.default, loader.calibration().?);
        } else {
            try std.testing.expect(loader.calibration() == null);
        }
        // The CPU plugin reports no allocator limit: admission is serial.
        try std.testing.expect(!loader.memory_supported);

        const Model = struct { value: Tensor };
        const model: Model = .{ .value = fixture.value };
        var buffers = try mem.bufferize(allocator, Model, &model);
        defer mem.deinitBufferized(allocator, Model, &buffers);

        var first: Buffer = undefined;
        try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &first)}, null);
        try std.testing.expectEqual(1, loader.pending.len);
        var second: Buffer = undefined;
        // Serial admission retires the first submission inside the second call.
        try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.second, &second)}, null);
        defer first.deinit();
        try std.testing.expectEqual(1, loader.pending.len);
        try std.testing.expectEqual(1, loader.admission_retires);
        try std.testing.expectEqual(LoaderTestFixture.contents.len, loader.bytesLoaded());
        try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);

        // The bulk is never gated: it queues behind the pending pack.
        try loader.load(Model, &model, &buffers, &fixture.store, &.{}, null);
        try std.testing.expectEqual(2, loader.pending.len);
        try loader.awaitAll();
        defer second.deinit();
        try std.testing.expectEqual(0, loader.pending.len);
        try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
        try LoaderTestFixture.expectContents(allocator, io, &buffers.value, &LoaderTestFixture.contents);
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytesLoaded());

        // Idempotent: a second await neither reruns nor recounts.
        try loader.awaitAll();
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytesLoaded());

        const Empty = struct { empty: Tensor };
        const empty_model: Empty = .{ .empty = fixture.empty };
        var empty_buffers = try mem.bufferize(allocator, Empty, &empty_model);
        defer mem.deinitBufferized(allocator, Empty, &empty_buffers);
        try LoaderTestFixture.expectLoadError(&loader, error.EmptyTensor, loader.load(Empty, &empty_model, &empty_buffers, &fixture.store, &.{}, null));
    }
}

test "one loader accepts pending submissions from different stores" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    var other_store = TensorStore.fromRegistry(allocator, &fixture.registry);
    defer other_store.deinit();
    const other = other_store.view().createTensor("second", null, .replicated);

    for (LoaderTestFixture.backends) |kind| {
        var first: Buffer = undefined;
        var second: Buffer = undefined;
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();
        try loader.load(Tensor, &fixture.value, &first, &fixture.store, &.{}, null);
        try loader.load(Tensor, &other, &second, &other_store, &.{}, null);
        try loader.awaitAll();
        defer first.deinit();
        defer second.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);
        try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
    }
}

test "bulk loading queues behind a submitted loadExecute of a transformed tensor" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    const transformed = fixture.store.view().maybeCreateBinding(&.{"value"}, fixture.value.shape()).?;
    try std.testing.expect(fixture.store.getSourcesById(transformed.id).?.transformed);
    try std.testing.expect(!fixture.store.getSourcesById(fixture.value.id).?.transformed);

    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();
        var output: Buffer = undefined;
        try loader.loadExecute(&fixture.store, &.{fixture.binding(transformed, &output)}, null);
        try std.testing.expect(loader.delivered.contains(transformed.id));
        // Delivered at submission: the bulk skips the tensor and queues
        // behind the pack instead of waiting for it.
        try loader.load(Tensor, &transformed, &output, &fixture.store, &.{}, null);
        try std.testing.expectEqual(2, loader.pending.len);
        try loader.awaitAll();
        defer output.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &output, &LoaderTestFixture.contents);
        try std.testing.expectEqual(LoaderTestFixture.contents.len, loader.bytesLoaded());
    }

    // Without a loadExecute, a transformed tensor cannot be loaded in bulk.
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();
        var never_written: Buffer = undefined;
        try LoaderTestFixture.expectLoadError(&loader, error.TransformedTensorNotDelivered, loader.load(Tensor, &transformed, &never_written, &fixture.store, &.{}, null));
    }
}

test "loader runs every binding of one submission" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();

        var outputs: [2]Buffer = undefined;
        try loader.loadExecute(&fixture.store, &.{
            fixture.binding(fixture.value, &outputs[0]),
            fixture.binding(fixture.second, &outputs[1]),
        }, null);
        try loader.awaitAll();
        defer for (&outputs) |*output| output.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &outputs[0], &LoaderTestFixture.contents);
        try LoaderTestFixture.expectContents(allocator, io, &outputs[1], &LoaderTestFixture.second_contents);
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, loader.bytesLoaded());
    }
}

test "loader deinit awaits pending submissions without running their executables" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);

        const Model = struct { value: Tensor };
        const model: Model = .{ .value = fixture.value };
        var buffers = try mem.bufferize(allocator, Model, &model);
        defer mem.deinitBufferized(allocator, Model, &buffers);
        var never_written: Buffer = undefined;
        try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &never_written)}, null);
        try loader.load(Model, &model, &buffers, &fixture.store, &.{}, null);
        loader.deinit();
    }
}

test "loader read failure fails later submissions and awaitAll" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();

        const Model = struct { value: Tensor };
        const model: Model = .{ .value = fixture.value };
        var buffers = try mem.bufferize(allocator, Model, &model);
        defer mem.deinitBufferized(allocator, Model, &buffers);

        var broken_output: Buffer = undefined;
        try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.missing, &broken_output)}, null);
        var never_written: Buffer = undefined;
        // Serial admission retires the broken submission first and reports it.
        try std.testing.expectError(error.FileNotFound, loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &never_written)}, null));
        try std.testing.expectEqual(0, loader.pending.len);
        try std.testing.expectError(error.FileNotFound, loader.load(Model, &model, &buffers, &fixture.store, &.{}, null));
        try std.testing.expectError(error.FileNotFound, loader.awaitAll());
        try std.testing.expectError(error.FileNotFound, loader.awaitAll());
        try std.testing.expectEqual(@as(usize, 0), loader.bytesLoaded());
    }
}

test "submitted bytes match the backend's allocations once everything landed" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    var placed = [_]u64{0};
    try addPlacementBytes(&placed, fixture.exe.input_shardings[0].resolve(fixture.platform), fixture.exe.input_shapes[0]);
    try std.testing.expectEqual(LoaderTestFixture.contents.len, placed[0]);

    var loader = try fixture.loader(allocator, io, .direct);
    defer loader.deinit();
    const Model = struct { value: Tensor };
    const model: Model = .{ .value = fixture.value };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    var first: Buffer = undefined;
    try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &first)}, null);
    try loader.load(Model, &model, &buffers, &fixture.store, &.{}, null);
    try loader.awaitAll();
    defer first.deinit();
    var allocated = [_]u64{0};
    loader.backend.allocatedBytesPerDevice(&allocated);
    // The executable's input shell and the bulk output, on the one device.
    try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, allocated[0]);
    try std.testing.expectEqual(loader.submitted_bytes[0], allocated[0]);
}

test "loader initialization releases its workspace on an invalid profile" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    // The alignment is checked after the pool is calibrated, so this covers
    // the backend's post-sizing errdefer path through the front end.
    var profile: VFS.LoadProfile = .local;
    profile.direct_io_alignment = 3;
    const result: anyerror!void = if (Loader.init(allocator, io, fixture.platform, .{
        .load_profile = profile,
        .direct_io = .on,
    })) |value| unexpected: {
        var loader = value;
        loader.deinit();
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.InvalidOptions, result);
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
    var store: TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();

    const platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    const Model = struct { a: Tensor, b: Tensor, c: Tensor, d: Tensor };
    const model: Model = .{
        .a = store.view().createTensor("a", null, .replicated),
        .b = store.view().createTensor("b", null, .replicated),
        .c = store.view().createTensor("c", null, .replicated),
        .d = store.view().createTensor("d", null, .replicated),
    };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);

    var loader = try Loader.init(allocator, vfs.io(), platform, .{
        .read_parallelism = 4,
        .load_profile = try vfs.loadProfile(url),
    });
    defer loader.deinit();
    try loader.load(Model, &model, &buffers, &store, &.{}, null);
    try loader.awaitAll();

    inline for (@typeInfo(Model).@"struct".fields, 0..) |field, index| {
        try LoaderTestFixture.expectContents(
            allocator,
            io,
            &@field(buffers, field.name),
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

    // Holding costs time, not pinned memory or credits: the pool never grew
    // beyond the width and every lifecycle credit came back.
    const direct = loader.backend.direct;
    try std.testing.expect(direct.pool.high_water <= 4);
    try std.testing.expectEqual(@as(usize, 0), direct.request_gate.in_use);
    try std.testing.expectEqual(@as(usize, direct.pool.capacity), direct.pool.free_blocks.items.len);
}

test "validation rejections leave the loader usable" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();
        var output: Buffer = undefined;
        var foreign: Platform = undefined;
        const different_shapes = [_]Shape{.init(.{8}, .u8)};
        const failures = [_]BindingError{
            error.ExecutablePlatformMismatch,
            error.ExecutableInputCountMismatch,
            error.ExecutableOutputCountMismatch,
            error.ExecutableInputShapeMismatch,
            error.ExecutableOutputShapeMismatch,
            error.ExecutablePlacementMismatch,
        };
        for (failures) |expected| {
            var exe = fixture.exe;
            switch (expected) {
                error.ExecutablePlatformMismatch => exe.platform = &foreign,
                error.ExecutableInputCountMismatch => exe.input_shapes = &.{},
                error.ExecutableOutputCountMismatch => exe.output_shapes = &.{},
                error.ExecutableInputShapeMismatch => exe.input_shapes = &different_shapes,
                error.ExecutableOutputShapeMismatch => exe.output_shapes = &different_shapes,
                error.ExecutablePlacementMismatch => exe.num_devices += 1,
            }
            try std.testing.expectError(expected, loader.loadExecute(&fixture.store, &.{.{ .tensor = fixture.value, .output = &output, .exe = &exe }}, null));

            try std.testing.expect(loader.failure == null);
            try std.testing.expectEqual(0, loader.pending.len);
        }
        const missing = Tensor.fromShape(fixture.value.shape());
        try std.testing.expectError(error.TensorNotFound, loader.load(Tensor, &missing, &output, &fixture.store, &.{}, null));
        try std.testing.expectError(error.EmptyTensor, loader.load(Tensor, &fixture.empty, &output, &fixture.store, &.{}, null));
        const transformed = fixture.store.view().maybeCreateBinding(&.{"value"}, fixture.value.shape()).?;
        try std.testing.expectError(error.TransformedTensorNotDelivered, loader.load(Tensor, &transformed, &output, &fixture.store, &.{}, null));

        try loader.load(Tensor, &fixture.value, &output, &fixture.store, &.{}, null);
        try loader.awaitAll();
        defer output.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &output, &LoaderTestFixture.contents);
    }
}

test "both backends report source size mismatches without changing publication semantics" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();
        var tensor = fixture.value;
        tensor._shape = .init(.{8}, .u8);
        var output: Buffer = undefined;
        if (kind == .direct) {
            try std.testing.expectError(error.SourceSizeMismatch, loader.load(Tensor, &tensor, &output, &fixture.store, &.{}, null));

            try loader.load(Tensor, &fixture.value, &output, &fixture.store, &.{}, null);
            try loader.awaitAll();
            output.deinit();
        } else {
            try loader.load(Tensor, &tensor, &output, &fixture.store, &.{}, null);
            try std.testing.expectError(error.SourceSizeMismatch, loader.awaitAll());
            try std.testing.expectError(error.SourceSizeMismatch, loader.awaitAll());
        }
    }
}
