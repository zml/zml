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
const ops = @import("../ops.zig");
const platform_mod = @import("../platform.zig");
const Memory = platform_mod.Memory;
const Platform = platform_mod.Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;
const testing = @import("../testing.zig");
const admission = @import("execute_admission.zig");
const backend = @import("backend.zig");
const Backend = backend.Backend;
const LoadPart = backend.LoadPart;
const LoadSpec = backend.LoadSpec;
const dma_calibration = @import("dma_calibration.zig");
const limits = @import("limits.zig");
const load_log = @import("log.zig").load;
const load_testing = @import("testing.zig");
const TensorStore = @import("TensorStore.zig");

const DeliveryMap = std.AutoHashMapUnmanaged(Tensor.Id, void);

const PrepareError = std.mem.Allocator.Error || error{ TensorNotFound, EmptyTensor, TransformedTensorNotDelivered, UnsupportedMemoryKind } || LayoutError;
/// A composite binding the loader cannot place: `InvalidLayout` when its
/// parts do not fit the tensor, `UnsupportedLayout` for a layout this loader
/// does not assemble yet (a strided part of many small chunks).
const LayoutError = error{ InvalidLayout, UnsupportedLayout };
const BindingError = error{ ExecutablePlatformMismatch, ExecutableInputCountMismatch, ExecutableOutputCountMismatch, ExecutableInputShapeMismatch, ExecutableOutputShapeMismatch, ExecutablePlacementMismatch };
/// Placing the sources on the devices, then the backend's own submission.
const SubmitError = error{IncompatibleSharding} || backend.SubmitError || error{Overflow};

/// Loads checkpoint sources and optionally executes bindings over them.
/// The platform and options' borrowed values must outlive the loader. Each
/// submission borrows its store's source metadata and outputs until
/// `awaitAll` or `deinit` returns; it copies its progress node, which must
/// not end before then.
/// Submit and await serially on the owning task; source reads and transfers
/// run concurrently inside the selected backend. Submissions are retired in
/// publish order: `loadExecute` retires older ones when its own does not fit
/// the room the devices report, and `awaitAll` retires the rest.
pub const Loader = struct {
    pub const AwaitError = backend.AwaitError || std.mem.Allocator.Error;
    pub const LoadError = PrepareError || SubmitError || AwaitError;
    pub const LoadExecuteError = BindingError || SubmitError || AwaitError || LayoutError || error{ TensorNotFound, EmptyTensor };

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
    /// Logical bytes of every submission retired with execution: each
    /// destination's shape once (`Loader.bytesLoaded`).
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
    /// fresh input buffers, one per source, or, for a composite binding, the
    /// assembled tensor into the one input; when the submission is retired,
    /// `exe` runs over them and its result is written to `output.*`.
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

        const backend_ = try Backend.init(allocator, io, platform, opts);
        errdefer backend_.destroy();

        const n = platform.devices.len;

        const submitted_bytes = try allocator.alloc(u64, n);
        errdefer allocator.free(submitted_bytes);
        @memset(submitted_bytes, 0);

        const pending_execution = try allocator.alloc(u64, n);
        errdefer allocator.free(pending_execution);
        @memset(pending_execution, 0);

        const words = try allocator.alloc(u64, 4 * n);
        errdefer allocator.free(words);

        var self: Loader = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .backend = backend_,
            .memory_supported = false,
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

        self.memory_supported = self.probeMemory();

        return self;
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

    /// Decided once: the CPU plugin logs an unimplemented warning on every
    /// stats call, so admission never asks a device that answered nothing.
    fn probeMemory(self: *Loader) bool {
        if (self.backend != .direct) return false;
        for (self.platform.devices) |device| {
            if (device.memoryStats().bytes_limit == null) return false;
        }
        return true;
    }

    /// Submits every tensor of `model` that is not transformed as one
    /// planned submission: a plain tensor from its one source, a composite
    /// one (`TensorStore.Layout`) from its parts, which the backend writes
    /// into the one destination buffer with every byte no part covers set to
    /// zero. Work may start before this returns. Transformed tensors must
    /// have a `loadExecute` published for them; they are skipped, and a
    /// missing one fails before any work is submitted. A zero-byte source is
    /// `error.EmptyTensor`; a composite layout that does not fit its tensor
    /// is `error.InvalidLayout`, one this loader cannot assemble yet is
    /// `error.UnsupportedLayout`. Each tensor is placed by its shape's
    /// sharding (replicated without a mesh) into the memory space its store
    /// binding names; a kind the platform cannot take
    /// (`backend.supportsMemory`) is `error.UnsupportedMemoryKind`. All of
    /// these fail before any work is submitted. Progress counts loaded
    /// tensors, a composite one once, excluding skipped transformed tensors.
    /// The caller owns the estimated total. Never gated by admission: the
    /// outputs are the model.
    pub fn load(
        self: *Loader,
        comptime ModelType: type,
        model: *const ModelType,
        buffers: *Bufferized(ModelType),
        store: *const TensorStore,
        progress: ?*std.Progress.Node,
    ) LoadError!void {
        if (self.failure) |err| return err;
        const prepared = try prepareModelLoad(
            self.allocator,
            self.platform,
            store,
            &self.delivered,
            ModelType,
            model,
            buffers,
        );
        defer prepared.deinit(self.allocator);
        try self.submit(prepared.specs, &.{}, &.{}, progress);
    }

    /// Submits the inputs of every binding as one planned submission, so
    /// adjacent sources of different bindings coalesce into shared reads.
    /// A program's binding (`TensorStore.Binding.inputs`) lands each of its
    /// inputs; any other binding is one input, the tensor as `load` fills
    /// it. Before publishing, retires the oldest pending submissions
    /// (running their executables on this task) until this one fits the
    /// room the devices report; one submission is always admitted.
    /// Retiring the submission runs the executables in binding order,
    /// writes each output and frees the inputs. Input and output placement
    /// come from each executable's shapes; inputs land in default device
    /// memory. Progress counts inputs across all bindings, before
    /// execution. The caller owns the estimated total.
    pub fn loadExecute(self: *Loader, store: *const TensorStore, bindings: []const Binding, progress: ?*std.Progress.Node) LoadExecuteError!void {
        if (self.failure) |err| return err;
        const executables = try self.allocator.alloc(BoundExecutable, bindings.len);
        var prepared: usize = 0;
        errdefer {
            for (executables[0..prepared]) |*executable| executable.deinit(self.allocator);
            self.allocator.free(executables);
        }
        var spec_count: usize = 0;
        var part_count: usize = 0;
        for (bindings, executables) |binding, *executable| {
            executable.* = try BoundExecutable.init(self.allocator, self.platform, store, binding);
            prepared += 1;
            spec_count += executable.inputs.len;
            if (executable.binding.inputs) |inputs| {
                for (inputs) |*layout| part_count += layoutPartCount(layout);
            } else part_count += loadPartCount(executable.binding);
        }
        const execution = try self.allocator.alloc(u64, self.platform.devices.len);
        errdefer self.allocator.free(execution);
        try self.admit(executables, execution);
        try self.delivered.ensureUnusedCapacity(self.allocator, @intCast(bindings.len));
        const specs = try self.allocator.alloc(LoadSpec, spec_count);
        defer self.allocator.free(specs);
        var parts: std.ArrayList(LoadPart) = .empty;
        defer parts.deinit(self.allocator);
        // Exact, so the specs' slices stay where they point.
        try parts.ensureTotalCapacityPrecise(self.allocator, part_count);
        var next: usize = 0;
        for (bindings, executables) |binding, executable| {
            const sources = executable.binding;
            for (executable.exe.input_shapes, executable.inputs, 0..) |shape, *input, i| {
                const layout: ?*const TensorStore.Layout = if (sources.inputs) |inputs| &inputs[i] else sources.layout;
                const whole: ?*safetensors.Tensor = if (layout) |l| l.whole(sources.tensors) else sources.tensors[0];
                const first = parts.items.len;
                if (whole) |source| {
                    if (source.byteSize() == 0) {
                        load_log.debug("executable input {s} in {s} is empty", .{ source.name, source.file_uri });
                        return error.EmptyTensor;
                    }
                    parts.appendAssumeCapacity(.whole(source));
                } else try appendCompositeParts(self.allocator, &parts, binding.tensor.id, shape, sources.tensors, .default, layout.?);
                specs[next] = .{
                    .shape = shape,
                    .memory = .default,
                    .output = input,
                    .parts = parts.items[first..],
                    .zero_fill = whole == null,
                };
                next += 1;
            }
        }
        std.debug.assert(next == specs.len and parts.items.len == part_count);
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

    /// Logical bytes of every submission retired with execution so far:
    /// each destination's shape once, so a composite tensor counts its
    /// padding and not its sources one by one, and an executable submission
    /// counts its inputs.
    pub fn bytesLoaded(self: *const Loader) usize {
        return self.bytes_loaded;
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
            for (exe.input_shapes) |shape| {
                try addPlacementBytes(inputs, self.platform, shape, .default);
            }
            // The output's memory space is the executable's to choose; it
            // is charged to the devices like the default space.
            try addPlacementBytes(execution, self.platform, exe.output_shapes[0], .default);
            // Bindings sharing an executable execute one after the other, so
            // its temporaries are charged once.
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

    /// Refreshes the per-device room; false when backend accounting or device
    /// memory statistics are unavailable.
    fn readRoom(self: *Loader) bool {
        switch (self.backend) {
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
    /// them. `execution` holds the output and temporary bytes per device of
    /// an executable submission, empty for a bulk one.
    fn submit(self: *Loader, specs: []const LoadSpec, executables: []BoundExecutable, execution: []u64, progress: ?*std.Progress.Node) SubmitError!void {
        var logical_bytes: usize = 0;
        const placed = self.scratch.placed;
        @memset(placed, 0);
        // One spec per destination, however many parts fill it: its bytes
        // count once, on each device it is placed on.
        for (specs) |spec| {
            logical_bytes = try std.math.add(usize, logical_bytes, spec.shape.byteSize());
            try addPlacementBytes(placed, self.platform, spec.shape, spec.memory);
        }
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const submission = try self.backend.submit(specs, if (progress) |node| node.* else .none);
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

/// Adds the bytes `shape` occupies on each device it is placed on, the same
/// expression the direct backend allocates by, so submitted and allocated
/// bytes cancel exactly once everything landed. Bytes delivered into a host
/// memory space take no device memory and count nowhere.
fn addPlacementBytes(out: []u64, platform: *const Platform, shape: Shape, memory: Memory.Kind) error{ IncompatibleSharding, Overflow }!void {
    if (!backend.countsAgainstDevice(memory)) return;
    const placement: Sharding.Placement = try .init(backend.placedShape(platform, shape).packedShape());
    const bytes: u64 = placement.shape.byteSize();
    for (placement.mesh().?.devicesInCanonicalOrder()) |device| {
        out[device.id] = try std.math.add(u64, out[device.id], bytes);
    }
}

/// A bulk load's specs and the parts they slice. The caller frees both.
const PreparedLoad = struct {
    specs: []LoadSpec,
    parts: []LoadPart,

    fn deinit(self: PreparedLoad, allocator: std.mem.Allocator) void {
        allocator.free(self.specs);
        allocator.free(self.parts);
    }
};

/// One spec per tensor that is not transformed: a plain binding is one
/// whole-source part, a composite one expands into one part per source, all
/// sharing the tensor's shape, memory space and output buffer.
fn prepareModelLoad(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    delivered: *const DeliveryMap,
    comptime ModelType: type,
    model: *const ModelType,
    buffers: *Bufferized(ModelType),
) PrepareError!PreparedLoad {
    const tensor_count = meta.count(Tensor, model);
    const flattened = try allocator.alloc(*Buffer, tensor_count);
    defer allocator.free(flattened);
    meta.forEachVisit(buffers, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
            output[i] = buffer;
        }
    }.call, .{flattened});

    // Sized first, so the specs can slice the parts as they are appended.
    var part_count: usize = 0;
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(_: usize, tensor: *const Tensor, store_: *const TensorStore, count: *usize) void {
            const sources = store_.getSourcesById(tensor.id) orelse return;
            if (sources.inputs == null) count.* += loadPartCount(sources);
        }
    }.call, .{ store, &part_count });

    var specs: std.ArrayList(LoadSpec) = .empty;
    errdefer specs.deinit(allocator);
    try specs.ensureTotalCapacityPrecise(allocator, tensor_count);
    var parts: std.ArrayList(LoadPart) = .empty;
    errdefer parts.deinit(allocator);
    try parts.ensureTotalCapacityPrecise(allocator, part_count);
    const Ctx = struct {
        platform: *const Platform,
        store: *const TensorStore,
        delivered: *const DeliveryMap,
        allocator: std.mem.Allocator,
        buffers: []*Buffer,
        specs: *std.ArrayList(LoadSpec),
        parts: *std.ArrayList(LoadPart),
        err: ?PrepareError = null,
    };
    var ctx: Ctx = .{
        .platform = platform,
        .store = store,
        .delivered = delivered,
        .allocator = allocator,
        .buffers = flattened,
        .specs = &specs,
        .parts = &parts,
    };
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, context: *Ctx) void {
            if (context.err != null) return;
            const sources = context.store.getSourcesById(tensor.id) orelse {
                load_log.debug("tensor {} {f} has no checkpoint binding", .{ tensor.id, tensor.shape() });
                context.err = error.TensorNotFound;
                return;
            };
            if (sources.inputs != null) {
                if (!context.delivered.contains(tensor.id)) {
                    logNotDelivered(context.allocator, tensor, sources.tensors);
                    context.err = error.TransformedTensorNotDelivered;
                }
                return;
            }
            const first_part = context.parts.items.len;
            if (sources.layout) |layout| {
                appendCompositeParts(context.allocator, context.parts, tensor.id, tensor.shape(), sources.tensors, sources.memory, layout) catch |err| {
                    context.err = err;
                    return;
                };
            } else {
                std.debug.assert(sources.tensors.len == 1);
                if (sources.tensors[0].byteSize() == 0) {
                    load_log.debug("tensor {} has empty source {s} in {s}", .{ tensor.id, sources.tensors[0].name, sources.tensors[0].file_uri });
                    context.err = error.EmptyTensor;
                    return;
                }
                context.parts.appendAssumeCapacity(.whole(sources.tensors[0]));
            }
            if (!backend.supportsMemory(context.platform, sources.memory)) {
                load_log.debug("tensor {} from {s} is bound to {t} memory, which {t} cannot take", .{ tensor.id, sources.tensors[0].name, sources.memory, context.platform.target });
                context.err = error.UnsupportedMemoryKind;
                return;
            }
            context.specs.appendAssumeCapacity(.{
                .shape = tensor.shape(),
                .memory = sources.memory,
                .output = context.buffers[i],
                .parts = context.parts.items[first_part..],
                .zero_fill = sources.layout != null,
            });
        }
    }.call, .{&ctx});
    if (ctx.err) |err| return err;
    // The capacity was exact, so the specs' slices point into the very
    // allocation returned.
    std.debug.assert(parts.items.len == parts.capacity);
    return .{
        .specs = try specs.toOwnedSlice(allocator),
        .parts = parts.items,
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
) (LayoutError || error{EmptyTensor})!void {
    if (layout.shape.byteSize() != shape.byteSize()) {
        load_log.debug("composite tensor {} {f}: its layout fills {f}", .{ id, shape, layout.shape });
        return error.InvalidLayout;
    }
    if (layout.parts.len == 0) {
        load_log.debug("composite tensor {} {f} has no parts", .{ id, shape });
        return error.InvalidLayout;
    }
    const strided = !expandsAll(layout);
    if (strided and !backend.countsAgainstDevice(memory)) {
        load_log.debug("composite tensor {} {f}: strided parts into {t} memory, which this loader does not assemble", .{ id, shape, memory });
        return error.UnsupportedLayout;
    }
    const first = parts.items.len;
    for (layout.parts) |part| {
        if (part.source >= tensors.len) {
            load_log.debug("composite tensor {} {f}: a part reads source {d} of {d}", .{ id, shape, part.source, tensors.len });
            return error.InvalidLayout;
        }
        const source = tensors[part.source];
        if (source.byteSize() == 0) {
            load_log.debug("composite tensor {} has empty source {s} in {s}", .{ id, source.name, source.file_uri });
            return error.EmptyTensor;
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
        load_log.debug("composite tensor {} {f} of {d} bytes: parts overlap or leave the sources or the tensor ({s})", .{ id, shape, shape.byteSize(), @errorName(err) });
        return if (err == error.UnsupportedLayout) error.UnsupportedLayout else error.InvalidLayout;
    };
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
    binding: TensorStore.Binding,
    inputs: []Buffer,
    output: *Buffer,
    exe: *const Exe,

    fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        store: *const TensorStore,
        binding: Loader.Binding,
    ) !BoundExecutable {
        const sources = store.getSourcesById(binding.tensor.id) orelse {
            load_log.debug("executable binding tensor {} {f} has no checkpoint sources", .{ binding.tensor.id, binding.tensor.shape() });
            return error.TensorNotFound;
        };
        try validateExecutableBinding(platform, binding.tensor, sources, binding.exe);
        const inputs = try allocator.alloc(Buffer, binding.exe.input_shapes.len);
        for (inputs, binding.exe.input_shapes) |*input, shape| {
            input.* = .{
                ._platform = platform,
                ._shape = shape,
                ._shards = .empty,
            };
        }
        return .{
            .binding = sources,
            .inputs = inputs,
            .output = binding.output,
            .exe = binding.exe,
        };
    }

    /// Runs the executable over the loaded inputs on the calling task.
    /// `Exe.call` leaves every input handle to the caller, donated ones
    /// included (their contents are gone), so `deinit` frees them all,
    /// exactly once; a runner would destroy the donated handles itself.
    fn execute(self: *const BoundExecutable, allocator: std.mem.Allocator, io: std.Io) !void {
        var args = try self.exe.args(allocator);
        defer args.deinit(allocator);
        var results = try self.exe.results(allocator);
        defer results.deinit(allocator);
        args.set(.{self.inputs});
        self.exe.call(io, args, &results, .{ .wait = true });
        self.output.* = results.get(Buffer);
    }

    /// Frees the inputs, donated or not; a shell the loader never wrote owns
    /// no shards.
    fn deinit(self: *BoundExecutable, allocator: std.mem.Allocator) void {
        for (self.inputs) |*input| input.deinit();
        allocator.free(self.inputs);
    }
};

fn validateExecutableBinding(
    platform: *const Platform,
    tensor: Tensor,
    binding: TensorStore.Binding,
    exe: *const Exe,
) BindingError!void {
    checkExecutableBinding(platform, tensor, binding, exe) catch |err| {
        load_log.debug("executable binding for tensor {} {f}: {s}; sources={d}, composite={}, input_shapes={d}, output_shapes={d}, platform_matches={}", .{ tensor.id, tensor.shape(), @errorName(err), binding.tensors.len, binding.layout != null, exe.input_shapes.len, exe.output_shapes.len, exe.platform == platform });
        return err;
    };
}

/// The executable's inputs and output carry their placement in their
/// shapes' shardings, as compiled. A program's binding is one input per
/// layout, of its shape and placed like it; any other binding is one input,
/// the tensor as its binding fills it.
fn checkExecutableBinding(
    platform: *const Platform,
    tensor: Tensor,
    binding: TensorStore.Binding,
    exe: *const Exe,
) BindingError!void {
    if (exe.platform != platform) return error.ExecutablePlatformMismatch;
    if (exe.output_shapes.len != 1)
        return error.ExecutableOutputCountMismatch;
    if (!tensor.shape().eql(exe.output_shapes[0])) {
        load_log.debug("executable output shape: expected {f}, actual {f}", .{ tensor.shape(), exe.output_shapes[0] });
        return error.ExecutableOutputShapeMismatch;
    }
    if (binding.inputs) |inputs| {
        if (exe.input_shapes.len != inputs.len)
            return error.ExecutableInputCountMismatch;
        for (inputs, exe.input_shapes, 0..) |layout, shape, i| {
            if (!layout.shape.eql(shape)) {
                load_log.debug("executable input {d}: bound shape {f}, executable shape {f}", .{ i, layout.shape, shape });
                return error.ExecutableInputShapeMismatch;
            }
            try validateExecutablePlacement(platform, shape, exe.num_devices);
            try validateSamePlacement(platform, layout.shape, shape);
        }
    } else {
        if (exe.input_shapes.len != 1)
            return error.ExecutableInputCountMismatch;
        const filled = if (binding.layout) |layout| layout.shape else binding.tensors[0].shape;
        if (!filled.eql(exe.input_shapes[0])) {
            load_log.debug("executable input: bound shape {f}, executable shape {f}", .{ filled, exe.input_shapes[0] });
            return error.ExecutableInputShapeMismatch;
        }
        try validateExecutablePlacement(platform, exe.input_shapes[0], exe.num_devices);
    }
    try validateExecutablePlacement(platform, exe.output_shapes[0], exe.num_devices);
    try validateSamePlacement(platform, tensor.shape(), exe.output_shapes[0]);
}

/// The executable must place its output where the model expects the bound
/// tensor, and its inputs where their bindings place them: the same
/// devices, each holding the same slice. The comparison is
/// per device, not by mesh pointer and partition, so `.open` and `.replicated`,
/// or a null mesh and the replicated mesh, agree when they place alike.
fn validateSamePlacement(platform: *const Platform, expected: Shape, actual: Shape) error{ExecutablePlacementMismatch}!void {
    const want = Sharding.Placement.init(backend.placedShape(platform, expected).packedShape()) catch |err| {
        load_log.debug("binding tensor placement of {f}: {s}", .{ expected, @errorName(err) });
        return error.ExecutablePlacementMismatch;
    };
    // `actual` was already validated by validateExecutablePlacement.
    const got = Sharding.Placement.init(backend.placedShape(platform, actual).packedShape()) catch return error.ExecutablePlacementMismatch;
    const want_devices = want.mesh().?.devicesInCanonicalOrder();
    const got_devices = got.mesh().?.devicesInCanonicalOrder();
    if (want_devices.len != got_devices.len) {
        load_log.debug("executable shape {f} spans {d} devices, binding {f} spans {d}", .{ actual, got_devices.len, expected, want_devices.len });
        return error.ExecutablePlacementMismatch;
    }
    for (want_devices, got_devices) |w, g| {
        const want_slices = want.slices(w.coords);
        const got_slices = got.slices(g.coords);
        const same_slices = want_slices.len == got_slices.len and for (want_slices.constSlice(), got_slices.constSlice()) |a, b| {
            if (a.start != b.start or a.size != b.size) break false;
        } else true;
        if (w.id != g.id or !same_slices) {
            load_log.debug("executable placement {f} differs from binding {f} on device {d}", .{ actual, expected, w.id });
            return error.ExecutablePlacementMismatch;
        }
    }
}

/// The shape's mesh (the replicated one when it names none) must place it,
/// over exactly the executable's devices, all of them on this platform.
fn validateExecutablePlacement(
    platform: *const Platform,
    shape: Shape,
    expected_devices: usize,
) error{ExecutablePlacementMismatch}!void {
    const placement = Sharding.Placement.init(backend.placedShape(platform, shape).packedShape()) catch |err| {
        load_log.debug("executable placement of {f}: {s}", .{ shape, @errorName(err) });
        return error.ExecutablePlacementMismatch;
    };
    const devices = placement.mesh().?.devicesInCanonicalOrder();
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
    platform: *const Platform,
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
        self.platform = testing.env();
        self.store = .fromRegistry(allocator, &self.registry, &.{self.platform.replicated_mesh});
        errdefer self.store.deinit();
        self.value = self.store.view().createReplicatedTensor("value", null);
        self.second = self.store.view().createReplicatedTensor("second", null);
        self.missing = self.store.view().createReplicatedTensor("missing", null);
        self.empty = self.store.view().createReplicatedTensor("empty", null);
        const Identity = struct {
            fn call(input: Tensor) Tensor {
                return input;
            }
        };
        self.exe = try self.platform.compileFn(allocator, io, Identity.call, .{self.value}, .{});
    }

    fn deinit(self: *LoaderTestFixture) void {
        self.exe.deinit();
        self.store.deinit();
        self.registry.deinit();
        self.tmp.cleanup();
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
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    // Exercise serial admission regardless of the platform's memory stats.
    loader.memory_supported = false;

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
    try loader.load(Model, &model, &buffers, &fixture.store, null);
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
    try LoaderTestFixture.expectLoadError(&loader, error.EmptyTensor, loader.load(Empty, &empty_model, &empty_buffers, &fixture.store, null));
}

test "one loader accepts pending submissions from different stores" {
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
    try loader.load(Tensor, &fixture.value, &first, &fixture.store, null);
    try loader.load(Tensor, &other, &second, &other_store, null);
    try loader.awaitAll();
    defer first.deinit();
    defer second.deinit();
    try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);
    try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
}

test "bulk loading queues behind a submitted loadExecute of a transformed tensor" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const transformed = fixture.store.view().maybeCreateBinding(&.{"value"}, fixture.value.shape()).?;
    try std.testing.expect(fixture.store.getSourcesById(transformed.id).?.inputs != null);
    try std.testing.expect(fixture.store.getSourcesById(fixture.value.id).?.inputs == null);

    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var output: Buffer = undefined;
    try loader.loadExecute(&fixture.store, &.{fixture.binding(transformed, &output)}, null);
    try std.testing.expect(loader.delivered.contains(transformed.id));
    // Delivered at submission: the bulk skips the tensor and queues
    // behind the pack instead of waiting for it.
    try loader.load(Tensor, &transformed, &output, &fixture.store, null);
    try std.testing.expectEqual(2, loader.pending.len);
    try loader.awaitAll();
    defer output.deinit();
    try LoaderTestFixture.expectContents(allocator, io, &output, &LoaderTestFixture.contents);
    try std.testing.expectEqual(LoaderTestFixture.contents.len, loader.bytesLoaded());

    // Without a loadExecute, a transformed tensor cannot be loaded in bulk.
    var fresh_loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer fresh_loader.deinit();
    var never_written: Buffer = undefined;
    try LoaderTestFixture.expectLoadError(&fresh_loader, error.TransformedTensorNotDelivered, fresh_loader.load(Tensor, &transformed, &never_written, &fixture.store, null));
}

test "loader runs every binding of one submission" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
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
        // Refused before anything is submitted, and the loader stays usable.
        try LoaderTestFixture.expectLoadError(&loader, error.UnsupportedMemoryKind, loader.load(Model, &model, &buffers, &fixture.store, null));
        try std.testing.expectEqual(0, loader.pending.len);
        try std.testing.expect(loader.failure == null);
        return;
    }
    try loader.load(Model, &model, &buffers, &fixture.store, null);
    try loader.awaitAll();
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
    if (loader.backend == .direct) {
        const allocated = try allocator.alloc(u64, platform.devices.len);
        defer allocator.free(allocated);
        loader.backend.direct.allocatedBytesPerDevice(allocated);
        try std.testing.expectEqualSlices(u64, loader.submitted_bytes, allocated);
        for (devices) |device| try std.testing.expectEqual(LoaderTestFixture.contents.len, allocated[device.id]);
    }
}

test "loadExecute frees donated and non-donated inputs once" {
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
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();

    inline for (.{ false, true }) |donate| {
        const Functions = struct {
            fn forward(value: Tensor, weight: Tensor) Tensor {
                const sum = value.add(weight);
                return if (donate) sum.reuseBuffer(value) else sum;
            }
        };
        const tensor = store.view().maybeCreateBinding(&.{ "value", "weight" }, shape) orelse unreachable;
        const exe = try platform.compileFn(allocator, io, Functions.forward, .{ Tensor.fromShape(shape), Tensor.fromShape(shape) }, .{});
        defer exe.deinit();
        try std.testing.expectEqual(@as(usize, @intFromBool(donate)), exe.donated_input_indices.len);

        var output: Buffer = undefined;
        try loader.loadExecute(&store, &.{.{ .tensor = tensor, .output = &output, .exe = &exe }}, null);
        // Retiring runs the executable, then frees every input handle once,
        // the donated one included.
        try loader.awaitAll();
        defer output.deinit();
        try std.testing.expectEqual([2]i32{ 11, 22 }, try output.getValue([2]i32, io));
        try std.testing.expect(loader.delivered.contains(tensor.id));
    }
}

test "loader deinit awaits pending submissions without running their executables" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });

    const Model = struct { value: Tensor };
    const model: Model = .{ .value = fixture.value };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    var never_written: Buffer = undefined;
    try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &never_written)}, null);
    try loader.load(Model, &model, &buffers, &fixture.store, null);
    loader.deinit();
}

test "loader read failure fails later submissions and awaitAll" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    // Retire the broken submission when admitting the next one.
    loader.memory_supported = false;

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
    try std.testing.expectError(error.FileNotFound, loader.load(Model, &model, &buffers, &fixture.store, null));
    try std.testing.expectError(error.FileNotFound, loader.awaitAll());
    try std.testing.expectError(error.FileNotFound, loader.awaitAll());
    try std.testing.expectEqual(@as(usize, 0), loader.bytesLoaded());
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
    try addPlacementBytes(placed, fixture.platform, fixture.exe.input_shapes[0], .default);
    for (placed) |bytes| try std.testing.expectEqual(LoaderTestFixture.contents.len, bytes);

    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    const Model = struct { value: Tensor };
    const model: Model = .{ .value = fixture.value };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    var first: Buffer = undefined;
    try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &first)}, null);
    try loader.load(Model, &model, &buffers, &fixture.store, null);
    try loader.awaitAll();
    defer first.deinit();
    const allocated = try allocator.alloc(u64, fixture.platform.devices.len);
    defer allocator.free(allocated);
    loader.backend.direct.allocatedBytesPerDevice(allocated);
    // Replicated executable inputs and bulk outputs occupy every device.
    for (allocated) |bytes| try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, bytes);
    try std.testing.expectEqualSlices(u64, loader.submitted_bytes, allocated);
}

test "loader initialization releases its workspace on an invalid profile" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
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
        .read_parallelism = 4,
        .load_profile = try vfs.loadProfile(url),
    });
    defer loader.deinit();
    try loader.load(Model, &model, &buffers, &store, null);
    try loader.awaitAll();

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
    defer fixture.deinit();
    var loader = try Loader.init(allocator, io, fixture.platform, .{ .read_parallelism = 2 });
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
    {
        // The executable places its output on the platform's devices, the
        // binding tensor expects other ones: same dims, different placement.
        var foreign_physical = fixture.platform.replicated_mesh.physical.*;
        const foreign_devices = try allocator.dupe(Sharding.Device, foreign_physical.devices_in_canonical_order);
        defer allocator.free(foreign_devices);
        for (foreign_devices) |*device| device.id += 100;
        foreign_physical.devices_in_canonical_order = foreign_devices;
        var foreign_mesh = fixture.platform.replicated_mesh.*;
        foreign_mesh.physical = &foreign_physical;
        var misplaced = fixture.value;
        misplaced._shape._sharding.mesh = &foreign_mesh;
        try std.testing.expectError(error.ExecutablePlacementMismatch, loader.loadExecute(&fixture.store, &.{.{ .tensor = misplaced, .output = &output, .exe = &fixture.exe }}, null));

        try std.testing.expect(loader.failure == null);
        try std.testing.expectEqual(0, loader.pending.len);
    }
    const missing = Tensor.fromShape(fixture.value.shape());
    try std.testing.expectError(error.TensorNotFound, loader.load(Tensor, &missing, &output, &fixture.store, null));
    try std.testing.expectError(error.EmptyTensor, loader.load(Tensor, &fixture.empty, &output, &fixture.store, null));
    const transformed = fixture.store.view().maybeCreateBinding(&.{"value"}, fixture.value.shape()).?;
    try std.testing.expectError(error.TransformedTensorNotDelivered, loader.load(Tensor, &transformed, &output, &fixture.store, null));

    try loader.load(Tensor, &fixture.value, &output, &fixture.store, null);
    try loader.awaitAll();
    defer output.deinit();
    try LoaderTestFixture.expectContents(allocator, io, &output, &LoaderTestFixture.contents);
}

test "loader reports source size mismatches without changing publication semantics" {
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
    if (loader.backend == .direct) {
        try std.testing.expectError(error.SourceSizeMismatch, loader.load(Tensor, &tensor, &output, &fixture.store, null));

        try loader.load(Tensor, &fixture.value, &output, &fixture.store, null);
        try loader.awaitAll();
        output.deinit();
    } else {
        try loader.load(Tensor, &tensor, &output, &fixture.store, null);
        try std.testing.expectError(error.SourceSizeMismatch, loader.awaitAll());
        try std.testing.expectError(error.SourceSizeMismatch, loader.awaitAll());
    }
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
                try loader.load(Model, &model, &buffers, &store, null);
                try loader.awaitAll();

                try load_testing.expectShards(allocator, io, &buffers.concatenated, &concatenated);
                try load_testing.expectShards(allocator, io, &buffers.padded, &padded);
                try load_testing.expectShards(allocator, io, &buffers.reversed, &reversed);
                try load_testing.expectShards(allocator, io, &buffers.plain, &fixture.plain);
                // Each destination once, padding included.
                try std.testing.expectEqual(@as(usize, 256 + 128 + 96 + 64), loader.bytesLoaded());
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
                if (loader.backend == .direct) {
                    const allocated = try allocator.alloc(u64, platform.devices.len);
                    defer allocator.free(allocated);
                    loader.backend.direct.allocatedBytesPerDevice(allocated);
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
        try loader.load(Model, &model, &buffers, &store, null);
        try loader.awaitAll();

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
            switch (loader.backend) {
                .direct => |direct| {
                    direct.plan_config.request_size = 256;
                    direct.plan_config.alignment = alignment;
                },
                .buffered => {},
            }
            try loader.load(Model, &model, &buffers, &store, null);
            try loader.awaitAll();
            try load_testing.expectShards(allocator, io, &buffers.by_rows, &interleaved);
            try load_testing.expectShards(allocator, io, &buffers.by_columns, &interleaved);
            try load_testing.expectShards(allocator, io, &buffers.scales_by_rows, &scales);
            try load_testing.expectShards(allocator, io, &buffers.scales_by_columns, &scales);
            try load_testing.expectShards(allocator, io, &buffers.transposed, &transposed);
            try load_testing.expectShards(allocator, io, &buffers.transposed_by_columns, &transposed);
        }
    }
}

test "composite layouts that do not fit their tensor fail before submission" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: CompositeFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit();
    const platform = testing.env();
    var store: TensorStore = .fromRegistry(allocator, &fixture.registry, &.{platform.replicated_mesh});
    defer store.deinit();
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
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

    const Case = struct { tensor: Tensor, expected: anyerror };
    const cases = [_]Case{
        // No part at all, and a part of a source the binding lacks.
        .{ .tensor = Composite.bind(s, r, &.{"a"}, destination, &.{}), .expected = error.InvalidLayout },
        .{ .tensor = Composite.bind(s, r, &.{"a"}, destination, &.{.{ .source = 1, .dst_offset = 0, .width = 16 }}), .expected = error.InvalidLayout },
        // A layout of another size than its tensor.
        .{ .tensor = Composite.bindAs(s, r, &.{"a"}, destination, .init(.{ 4, 8 }, .u16), &.{.{ .dst_offset = 0, .width = 48 }}), .expected = error.InvalidLayout },
        // Past the end of its source, and of the tensor.
        .{ .tensor = Composite.bind(s, r, &.{"a"}, destination, &.{.{ .src_offset = 16, .dst_offset = 0, .width = 48 }}), .expected = error.InvalidLayout },
        .{ .tensor = Composite.bind(s, r, &.{"a"}, destination, &.{.{ .dst_offset = 96, .width = 48 }}), .expected = error.InvalidLayout },
        // Overlapping parts.
        .{ .tensor = Composite.bind(s, r, &.{ "a", "b" }, destination, &.{
            .{ .source = 0, .dst_offset = 0, .width = 48 },
            .{ .source = 1, .dst_offset = 32, .width = 32 },
        }), .expected = error.InvalidLayout },
        .{ .tensor = Composite.bind(s, r, &.{ "a", "empty" }, destination, &.{
            .{ .source = 0, .dst_offset = 0, .width = 48 },
            .{ .source = 1, .dst_offset = 48, .width = 16 },
        }), .expected = error.EmptyTensor },
        // Strided parts whose runs overlap; two of over a thousand runs
        // each that interleave with different strides, which the checks
        // cannot prove disjoint.
        .{ .tensor = Composite.bind(s, r, &.{"wide"}, .init(.{ 8, 64 }, .u16), &.{
            .{ .dst_offset = 0, .width = 2, .dims = &.{.{ .count = 128, .src_stride = 8, .dst_stride = 4 }} },
            .{ .dst_offset = 2, .width = 4, .dims = &.{.{ .count = 128, .src_stride = 8, .dst_stride = 4 }} },
        }), .expected = error.InvalidLayout },
        .{ .tensor = Composite.bind(s, r, &.{"wide"}, .init(.{ 64, 64 }, .u16), &.{
            .{ .dst_offset = 0, .width = 1, .dims = &.{.{ .count = 2000, .src_stride = 0, .dst_stride = 4 }} },
            .{ .dst_offset = 2, .width = 1, .dims = &.{.{ .count = 1300, .src_stride = 0, .dst_stride = 6 }} },
        }), .expected = error.UnsupportedLayout },
        // Strided parts into host memory.
        .{ .tensor = host: {
            const tensor = Composite.bind(s, r, &.{"wide"}, .init(.{ 8, 64 }, .u16), &.{.{ .dst_offset = 0, .width = 2, .dims = &.{
                .{ .count = 8, .src_stride = 2, .dst_stride = 128 },
                .{ .count = 64, .src_stride = 16, .dst_stride = 2 },
            } }});
            s.setMemory(tensor, .host_pinned);
            break :host tensor;
        }, .expected = error.UnsupportedLayout },
    };
    var output: Buffer = undefined;
    for (cases) |case| {
        try LoaderTestFixture.expectLoadError(&loader, case.expected, loader.load(Tensor, &case.tensor, &output, &store, null));
        try std.testing.expectEqual(0, loader.pending.len);
        try std.testing.expect(loader.failure == null);
    }

    // The loader stays usable. A part placed past a gap, the rest zero; parts
    // listed out of destination order; one source feeding two parts; a
    // strided part of few runs, expanded; and a transpose, 512 runs of two
    // bytes, gathered.
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
    try loader.load(Model, &model, &buffers, &store, null);
    try loader.awaitAll();

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
    const assembled = Source.concatenate(&.{ CompositeFixture.rows(view, "a"), CompositeFixture.rows(view, "b") }, .row);
    const c = CompositeFixture.rows(view, "c");
    const inputs = [_]Source.Input{ assembled.place(.replicated, .{}), c.place(.replicated, .{}) };
    const tensor = Source.bindProgram(&inputs, c.shape());
    const binding = store.getSourcesById(tensor.id).?;
    try std.testing.expectEqual(2, binding.inputs.?.len);
    try std.testing.expect(binding.inputs.?[0].whole(binding.tensors) == null);
    try std.testing.expect(binding.inputs.?[1].whole(binding.tensors) != null);
    try std.testing.expect(store.sourceOf(tensor) == null);
    const AddRows = struct {
        fn call(rows: Tensor, other: Tensor) Tensor {
            return rows.slice(0, .{ .end = 4 }).add(other);
        }
    };
    var exe = try platform.compileFn(allocator, io, AddRows.call, .{ Tensor.fromShape(inputs[0].shape), Tensor.fromShape(inputs[1].shape) }, .{});
    defer exe.deinit();
    var loader = try Loader.init(allocator, io, platform, .{ .read_parallelism = 2 });
    defer loader.deinit();
    var output: Buffer = undefined;
    try loader.loadExecute(&store, &.{.{ .tensor = tensor, .output = &output, .exe = &exe }}, null);
    try loader.awaitAll();
    defer output.deinit();
    var expected: [64]u8 = undefined;
    for (0..32) |i| {
        const row = if (i < 24) fixture.a[i * 2 ..][0..2] else fixture.b[(i - 24) * 2 ..][0..2];
        const value = std.mem.readInt(u16, row, .little) +% std.mem.readInt(u16, fixture.c[i * 2 ..][0..2], .little);
        std.mem.writeInt(u16, expected[i * 2 ..][0..2], value, .little);
    }
    try load_testing.expectShards(allocator, io, &output, &expected);

    // An executable of other inputs does not take them.
    const Single = struct {
        fn call(rows: Tensor) Tensor {
            return rows.slice(0, .{ .end = 4 });
        }
    };
    var single = try platform.compileFn(allocator, io, Single.call, .{Tensor.fromShape(inputs[0].shape)}, .{});
    defer single.deinit();
    var swapped = try platform.compileFn(allocator, io, AddRows.call, .{ Tensor.fromShape(.init(.{ 6, 8 }, .u16)), Tensor.fromShape(inputs[1].shape) }, .{});
    defer swapped.deinit();
    var unused: Buffer = undefined;
    try std.testing.expectError(
        error.ExecutableInputCountMismatch,
        loader.loadExecute(&store, &.{.{ .tensor = tensor, .output = &unused, .exe = &single }}, null),
    );
    try std.testing.expectError(
        error.ExecutableInputShapeMismatch,
        loader.loadExecute(&store, &.{.{ .tensor = tensor, .output = &unused, .exe = &swapped }}, null),
    );
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
    try loader.load(Tensor, &table, &host_table, &store, null);
    try loader.awaitAll();
    defer host_table.deinit();
    try std.testing.expectEqual(padded.len, loader.bytesLoaded());
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
    exe.call(io, args, &results, .{ .wait = true });
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
    try loader.load(Tensor, &device_copy, &device_buffer, &store, null);
    try loader.awaitAll();
    defer device_buffer.deinit();
    for (devices) |device| {
        const in_use = platform.devices[device.id].memoryStats().bytes_in_use;
        // Zero: the plugin reports no usage, and neither check means anything.
        if (in_use == 0) continue;
        try std.testing.expect(in_use -| in_use_control[device.id] >= shard_bytes);
    }
}
