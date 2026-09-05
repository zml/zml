//! Checkpoint loading: Loader submits work, Handle awaits it, and Window
//! bounds the caller's outstanding executable inputs. Backends only read
//! sources into buffers; lookup and execution stay in this shared front end.
const std = @import("std");

const VFS = @import("vfs");
const backend = @import("backend.zig");
const Buffer = @import("../buffer.zig").Buffer;
const Bufferized = mem.Bufferized;
const platform_mod = @import("../platform.zig");
const Exe = @import("../exe.zig").Exe;
const mem = @import("../mem.zig");
const meta = @import("../meta.zig");
const Platform = platform_mod.Platform;
const safetensors = @import("../safetensors.zig");
const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Tensor = @import("../tensor.zig").Tensor;

const load_log = std.log.scoped(.@"zml/io/load");

const dma_calibration = @import("dma_calibration.zig");
const limits = @import("limits.zig");
const TensorStore = @import("TensorStore.zig");
pub const Parallelism = @import("source_concurrency.zig").Parallelism;
const Backend = backend.Backend;
const LoadSpec = backend.LoadSpec;
const DeliveryMap = std.AutoHashMapUnmanaged(Tensor.Id, *bool);

/// Loads checkpoint sources and optionally executes bindings over them.
/// The platform and options' borrowed values must outlive the loader. Each
/// submission borrows its store's source metadata, outputs, and optional progress
/// parent until completion.
/// Submit and await handles serially on the owning task; source reads and
/// transfers run concurrently inside the selected backend.
pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    backend: Backend,
    /// Every handle this loader created, in publish order. `deinit` awaits
    /// the open ones and frees them all; a `Handle` is invalid afterwards.
    handles: std.ArrayListUnmanaged(*HandleState) = .empty,
    /// Stable completion flags shared with executable handles. Only a successful
    /// execution delivers a transformed tensor; submitting its reads does not.
    delivered: DeliveryMap = .empty,

    pub const Options = struct {
        pub const auto: Options = .{};

        /// Concurrent positional source requests.
        read_parallelism: Parallelism = .{ .adaptive = .{
            .initial = 12,
            .maximum = limits.max_read_parallelism,
        } },
        /// Model-wide source tuning prepared from the VFS path. The default is
        /// generic for callers that do not have an explicit VFS profile.
        load_profile: VFS.LoadProfile = .default,
        /// Calibrate transfer sizing during initialization. Ignored by buffered
        /// backends; CPU uses the default sizing without measurement.
        dma: dma_calibration.Options = .{},
        /// Upper bound for the direct backend's host arenas, not a growth target.
        max_host_bytes: usize = 16 * 1024 * 1024 * 1024,
        numa: mem.dma.NumaPlacement = .memory_nodes,
        /// Direct I/O for local source files, decided per file by the
        /// profile's VFS: `auto` reads a file past the page cache when it is
        /// mostly not cached at load time, `on` whenever the filesystem
        /// allows it, `off` never. The planner widens the reads of a direct
        /// file to the profile's alignment, at most two alignment units per
        /// request. A direct read never fills the page cache, so under
        /// `auto` a cold file stays cold and comes from the disk on every
        /// load; on a host whose warm buffered reads beat its disk, a model
        /// loaded repeatedly is better served by `off`. Nothing changes for
        /// a profile without alignment.
        direct_io: VFS.DirectIo = .auto,
    };

    /// One executable over a binding. `tensor`'s sources are loaded into
    /// fresh input buffers; `Handle.await` runs `exe` over them and writes
    /// its result to `output.*`.
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
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .backend = try Backend.init(allocator, io, platform, .{
                .read_parallelism = opts.read_parallelism,
                .load_profile = opts.load_profile,
                .dma = opts.dma,
                .max_host_bytes = opts.max_host_bytes,
                .numa = opts.numa,
                .direct_io = opts.direct_io,
            }),
        };
    }

    /// Sizing selected during initialization; absent for buffered loading.
    pub fn calibration(self: *const Loader) ?dma_calibration.Result {
        return self.backend.calibration();
    }

    /// Submits every single-source tensor of `model` as one planned
    /// submission. Work may start before this returns. Transformed tensors must
    /// already have been delivered by awaiting `loadExecute`; they are skipped.
    /// Missing deliveries fail before any work is submitted. A zero-byte source
    /// is `error.EmptyTensor`. Shardings are selected during this call; an empty
    /// slice uses replicated placement. Progress counts loaded sources, excluding
    /// skipped transformed tensors. The caller owns the estimated total.
    pub fn load(
        self: *Loader,
        comptime ModelType: type,
        model: *const ModelType,
        buffers: *Bufferized(ModelType),
        store: *const TensorStore,
        shardings: []const Sharding,
        progress: ?*std.Progress.Node,
    ) !Handle {
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
        return self.submit(specs, &.{}, progress);
    }

    /// Submits one tensor through the same validation and handle lifecycle as load.
    pub fn loadBuffer(self: *Loader, tensor: Tensor, output: *Buffer, store: *const TensorStore, shardings: []const Sharding, progress: ?*std.Progress.Node) !Handle {
        return self.load(Tensor, &tensor, output, store, shardings, progress);
    }

    /// Submits the sources of every binding as one planned submission, so
    /// adjacent sources of different bindings coalesce into shared reads.
    /// `Handle.await` runs the executables in binding order on the awaiting
    /// task and frees their inputs. Input and output placement come from each
    /// executable. Progress counts input sources across all bindings, before
    /// execution. The caller owns the estimated total.
    pub fn loadExecute(self: *Loader, store: *const TensorStore, bindings: []const Binding, progress: ?*std.Progress.Node) !Handle {
        const executables = try self.allocator.alloc(BoundExecutable, bindings.len);
        var prepared: usize = 0;
        errdefer {
            for (executables[0..prepared]) |*executable| executable.deinit(self.allocator);
            self.allocator.free(executables);
        }
        var source_count: usize = 0;
        for (bindings, executables) |binding, *executable| {
            executable.* = try BoundExecutable.init(
                self.allocator,
                self.platform,
                store,
                binding,
                try self.deliveryFor(binding.tensor.id),
            );
            prepared += 1;
            source_count += executable.sources.len;
        }
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
                if (source.byteSize() == 0) return error.EmptyTensor;
                specs[next] = .{
                    .source = source,
                    .shape = shape,
                    .sharding = sharding.resolve(self.platform),
                    .output = input,
                };
                next += 1;
            }
        }
        return self.submit(specs, executables, progress);
    }

    /// Awaits every handle in publish order, running their executables.
    /// Returns the first error after awaiting all of them.
    pub fn awaitAll(self: *Loader) !void {
        var first_error: ?anyerror = null;
        for (self.handles.items) |state| {
            state.await(true) catch |err| {
                first_error = first_error orelse err;
            };
        }
        if (first_error) |err| return err;
    }

    /// Logical bytes of every submission awaited successfully so far.
    pub fn bytesLoaded(self: *const Loader) usize {
        return self.backend.bytesLoaded();
    }

    /// Bytes the inputs of `exe` occupy on each device: the size of every
    /// input's per-device placement, summed. Sizes a `Window` budget.
    pub fn executeInputBytesPerDevice(self: *const Loader, exe: *const Exe) !usize {
        var total: usize = 0;
        for (exe.input_shapes, exe.input_shardings) |shape, sharding| {
            const placement = try sharding.resolve(self.platform).placement(shape);
            total = try std.math.add(usize, total, placement.shape.byteSize());
        }
        return total;
    }

    /// Awaits every open handle without running executables (their outputs
    /// stay unwritten, their inputs are freed), frees every handle, then
    /// destroys the backend.
    pub fn deinit(self: *Loader) void {
        for (self.handles.items) |state| {
            state.await(false) catch {};
            self.allocator.destroy(state);
        }
        self.handles.deinit(self.allocator);
        var deliveries = self.delivered.valueIterator();
        while (deliveries.next()) |flag| self.allocator.destroy(flag.*);
        self.delivered.deinit(self.allocator);
        self.backend.destroy();
        self.* = undefined;
    }

    fn deliveryFor(self: *Loader, id: Tensor.Id) !*bool {
        const entry = try self.delivered.getOrPut(self.allocator, id);
        if (entry.found_existing) return entry.value_ptr.*;
        errdefer self.delivered.removeByPtr(entry.key_ptr);
        const delivered = try self.allocator.create(bool);
        delivered.* = false;
        entry.value_ptr.* = delivered;
        return delivered;
    }

    /// One submission over `specs`. The handle owns `executables` once the
    /// submission is published; on failure the caller still does.
    fn submit(self: *Loader, specs: []const LoadSpec, executables: []BoundExecutable, progress: ?*std.Progress.Node) !Handle {
        var logical_bytes: usize = 0;
        for (specs) |spec| {
            logical_bytes = try std.math.add(usize, logical_bytes, spec.source.shape.byteSize());
        }
        try self.handles.ensureUnusedCapacity(self.allocator, 1);
        const state = try self.allocator.create(HandleState);
        errdefer self.allocator.destroy(state);
        state.* = .{
            .allocator = self.allocator,
            .io = self.io,
            .executables = executables,
            .logical_bytes = logical_bytes,
            .submission = try self.backend.submit(specs, progress),
        };
        self.handles.appendAssumeCapacity(state);
        return .{ .state = state };
    }
};

/// One submission of a `Loader`: a whole-model `load` or a `loadExecute`.
/// A copyable value, valid until `Loader.deinit`.
pub const Handle = struct {
    state: *HandleState,

    /// Waits for the submission's reads and DMA. For `loadExecute`, then
    /// runs each executable in binding order on this task with `.wait = true`,
    /// writes its output and frees its inputs. On success the submission's
    /// logical bytes join `Loader.bytesLoaded`. Fails with the loader's
    /// sticky error when its pipeline failed. Idempotent: later calls return
    /// the cached outcome.
    pub fn await(self: Handle) !void {
        return self.state.await(true);
    }

    /// Whether source reads and transfers have finished. `await` may still
    /// run the submission's executables before returning.
    pub fn isDone(self: Handle) bool {
        return self.state.isDone();
    }

    /// Logical bytes of every source in the submission.
    pub fn logicalBytes(self: Handle) usize {
        return self.state.logical_bytes;
    }
};

/// Caller-side concurrency policy for `loadExecute`: at most `max_handles`
/// pending submissions whose executable inputs (per device, from
/// `Loader.executeInputBytesPerDevice`) sum to at most `budget_bytes`.
/// `submit` awaits the oldest pending handle, running its executables and
/// freeing its inputs, until the next submission fits; one submission is
/// always admitted, even above the budget. A window of one serializes
/// submissions like a synchronous `loadExecute`.
pub const Window = struct {
    const Pending = struct {
        handle: Handle,
        input_bytes: usize,
    };

    allocator: std.mem.Allocator,
    budget_bytes: usize,
    max_handles: usize,
    pending: std.ArrayListUnmanaged(Pending) = .empty,
    pending_bytes: usize = 0,

    pub fn init(allocator: std.mem.Allocator, budget_bytes: usize, max_handles: usize) Window {
        std.debug.assert(max_handles > 0);
        return .{
            .allocator = allocator,
            .budget_bytes = budget_bytes,
            .max_handles = max_handles,
        };
    }

    pub fn submit(self: *Window, loader: *Loader, store: *const TensorStore, bindings: []const Loader.Binding, progress: ?*std.Progress.Node) !void {
        var input_bytes: usize = 0;
        for (bindings) |binding| {
            input_bytes = try std.math.add(
                usize,
                input_bytes,
                try loader.executeInputBytesPerDevice(binding.exe),
            );
        }
        while (self.pending.items.len != 0 and
            (self.pending.items.len == self.max_handles or
                self.pending_bytes +| input_bytes > self.budget_bytes))
        {
            try self.awaitOldest();
        }
        try self.pending.ensureUnusedCapacity(self.allocator, 1);
        const handle = try loader.loadExecute(store, bindings, progress);
        self.pending.appendAssumeCapacity(.{ .handle = handle, .input_bytes = input_bytes });
        self.pending_bytes += input_bytes;
    }

    /// Awaits every pending handle, oldest first. Returns the first error
    /// after awaiting all of them.
    pub fn drain(self: *Window) !void {
        var first_error: ?anyerror = null;
        while (self.pending.items.len != 0) {
            self.awaitOldest() catch |err| {
                first_error = first_error orelse err;
            };
        }
        if (first_error) |err| return err;
    }

    /// Drains, dropping any error (the loader keeps its sticky error). A
    /// handle cannot be awaited without running its executables, so this
    /// runs whatever is still pending; call `drain` first to observe errors.
    pub fn deinit(self: *Window) void {
        self.drain() catch {};
        self.pending.deinit(self.allocator);
        self.* = undefined;
    }

    fn awaitOldest(self: *Window) !void {
        const oldest = self.pending.orderedRemove(0);
        self.pending_bytes -= oldest.input_bytes;
        try oldest.handle.await();
    }
};

fn prepareModelLoad(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    shardings: []const Sharding,
    delivered: *const DeliveryMap,
    comptime ModelType: type,
    model: *const ModelType,
    buffers: *Bufferized(ModelType),
) ![]LoadSpec {
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
        err: ?anyerror = null,
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
                context.err = error.NotFound;
                return;
            };
            if (sources.transformed) {
                const delivered_flag = context.delivered.get(tensor.id);
                if (delivered_flag == null or !delivered_flag.?.*) {
                    logNotDelivered(context.allocator, tensor, sources.tensors);
                    context.err = error.TransformedTensorNotDelivered;
                }
                return;
            }
            std.debug.assert(sources.tensors.len == 1);
            if (sources.tensors[0].byteSize() == 0) {
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

    load_log.err("Transformed tensor {} {f} was not delivered by loadExecute before load; sources: {s}", .{ tensor.id, tensor.shape(), names.written() });
}

const HandleState = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    submission: backend.Submission,
    /// Run in binding order once the reads are done; freed with their
    /// inputs by the first await.
    executables: []BoundExecutable,
    logical_bytes: usize,
    awaited: bool = false,
    failure: ?anyerror = null,

    fn isDone(self: *const HandleState) bool {
        return self.awaited or self.submission.isDone();
    }

    /// The first call waits for the reads, runs the executables when
    /// `execute`, frees the inputs either way and commits the bytes on
    /// success; later calls return the cached outcome.
    fn await(self: *HandleState, execute: bool) !void {
        if (self.awaited) {
            if (self.failure) |err| return err;
            return;
        }
        self.awaited = true;
        defer self.releaseExecutables();
        self.submission.await() catch |err| {
            self.failure = err;
            return err;
        };
        if (execute) for (self.executables) |*executable| {
            executable.execute(self.allocator, self.io) catch |err| {
                self.failure = err;
                return err;
            };
            executable.deinit(self.allocator);
        };
        self.submission.commitBytes(self.logical_bytes);
    }

    fn releaseExecutables(self: *HandleState) void {
        for (self.executables) |*executable| executable.deinit(self.allocator);
        self.allocator.free(self.executables);
        self.executables = &.{};
    }
};

fn validateOptions(opts: Loader.Options) !void {
    _ = try limits.effectiveSourceRequestSize(opts.load_profile.read_chunk_size, 0);
    const initial = opts.read_parallelism.initial();
    const maximum = opts.read_parallelism.maximum();
    if (initial == 0 or maximum < initial or maximum > limits.max_read_parallelism)
        return error.InvalidLoadParallelism;
}

/// One executable of a `loadExecute` submission and the input shells its
/// sources are loaded into.
const BoundExecutable = struct {
    sources: []const *safetensors.Tensor,
    inputs: []Buffer,
    output: *Buffer,
    exe: *const Exe,
    delivered: *bool,

    fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        store: *const TensorStore,
        binding: Loader.Binding,
        delivered: *bool,
    ) !BoundExecutable {
        const sources = (store.getSourcesById(binding.tensor.id) orelse return error.NotFound).tensors;
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
            .delivered = delivered,
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
        self.delivered.* = true;
    }

    /// Frees the inputs; a shell the loader never wrote owns no shards.
    /// Idempotent.
    fn deinit(self: *BoundExecutable, allocator: std.mem.Allocator) void {
        for (self.inputs) |*input| input.deinit();
        allocator.free(self.inputs);
        self.inputs = &.{};
    }
};

fn validateExecutableBinding(
    platform: *const Platform,
    tensor: Tensor,
    sources: []const *safetensors.Tensor,
    exe: *const Exe,
) !void {
    if (exe.platform != platform) return error.ExecutablePlatformMismatch;
    if (exe.output_shapes.len != 1 or exe.output_shardings.len != 1)
        return error.InvalidExecutableOutputs;
    if (exe.input_shapes.len != sources.len or exe.input_shardings.len != sources.len)
        return error.InvalidExecutableInputs;
    if (!tensor.shape().eql(exe.output_shapes[0])) return error.ExecutableOutputShapeMismatch;
    for (sources, exe.input_shapes, exe.input_shardings) |source, shape, sharding| {
        if (!source.shape.eql(shape)) return error.ExecutableInputShapeMismatch;
        try validateExecutableSharding(platform, sharding, exe.num_devices);
    }
    try validateExecutableSharding(platform, exe.output_shardings[0], exe.num_devices);
}

fn validateExecutableSharding(
    platform: *const Platform,
    unresolved: Sharding,
    expected_devices: usize,
) !void {
    const sharding = unresolved.resolve(platform);
    const devices = sharding.devicesInCanonicalOrder();
    if (devices.len != expected_devices) return error.ExecutablePlacementMismatch;
    for (devices) |device| {
        if (device.id >= platform.devices.len) return error.ExecutablePlacementMismatch;
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
        const opts: Loader.Options = .{ .read_parallelism = .{ .fixed = 2 } };
        return switch (kind) {
            .direct => try Loader.init(allocator, io, self.platform, opts),
            .buffered => .{
                .allocator = allocator,
                .io = io,
                .platform = self.platform,
                .backend = try Backend.initBuffered(
                    allocator,
                    io,
                    self.platform,
                    opts.read_parallelism,
                    opts.load_profile,
                ),
            },
        };
    }

    fn binding(self: *const LoaderTestFixture, tensor: Tensor, output: *Buffer) Loader.Binding {
        return .{ .tensor = tensor, .output = output, .exe = &self.exe };
    }

    fn expectLoadError(expected: anyerror, result: anyerror!Handle) !void {
        // Formatting Handle recursively instantiates every platform backend.
        // Await an unexpected submission before releasing its borrowed outputs.
        const outcome: anyerror!void = if (result) |handle| success: {
            handle.state.await(false) catch {};
            break :success {};
        } else |err| err;
        try std.testing.expectError(expected, outcome);
    }

    fn expectContents(allocator: std.mem.Allocator, io: std.Io, buffer: *const Buffer, expected: []const u8) !void {
        const loaded = try buffer.toSliceAlloc(allocator, io);
        defer loaded.free(allocator);
        try std.testing.expectEqualSlices(u8, expected, loaded.constData());
    }
};

test "loader handles complete out of order and count bytes once each" {
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

        const Model = struct { value: Tensor };
        const model: Model = .{ .value = fixture.value };
        var buffers = try mem.bufferize(allocator, Model, &model);
        defer mem.deinitBufferized(allocator, Model, &buffers);

        var first: Buffer = undefined;
        const a = try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &first)}, null);
        var second: Buffer = undefined;
        const b = try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.second, &second)}, null);
        const c = try loader.load(Model, &model, &buffers, &fixture.store, &.{}, null);
        try std.testing.expectEqual(LoaderTestFixture.contents.len, a.logicalBytes());

        try b.await();
        defer second.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
        try a.await();
        defer first.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);
        try c.await();
        try LoaderTestFixture.expectContents(allocator, io, &buffers.value, &LoaderTestFixture.contents);
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytesLoaded());

        // Idempotent: a second await neither reruns nor recounts.
        try std.testing.expect(a.isDone());
        try a.await();
        try loader.awaitAll();
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 3, loader.bytesLoaded());

        const Empty = struct { empty: Tensor };
        const empty_model: Empty = .{ .empty = fixture.empty };
        var empty_buffers = try mem.bufferize(allocator, Empty, &empty_model);
        defer mem.deinitBufferized(allocator, Empty, &empty_buffers);
        try LoaderTestFixture.expectLoadError(error.EmptyTensor, loader.load(Empty, &empty_model, &empty_buffers, &fixture.store, &.{}, null));
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
        const a = try loader.loadBuffer(fixture.value, &first, &fixture.store, &.{}, null);
        const b = try loader.loadBuffer(other, &second, &other_store, &.{}, null);
        try b.await();
        defer second.deinit();
        try a.await();
        defer first.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);
        try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
    }
}

test "bulk loading preserves transformed tensors after their executable completes" {
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
        const handle = try loader.loadExecute(&fixture.store, &.{fixture.binding(transformed, &output)}, null);
        try std.testing.expect(!loader.delivered.get(transformed.id).?.*);
        // Other submissions can grow the map while this handle is pending.
        for (0..32) |_| {
            const other = fixture.store.view().maybeCreateBinding(&.{"value"}, fixture.value.shape()).?;
            _ = try loader.deliveryFor(other.id);
        }
        try handle.await();
        defer output.deinit();
        try std.testing.expect(loader.delivered.get(transformed.id).?.*);
        const bulk = try loader.loadBuffer(transformed, &output, &fixture.store, &.{}, null);
        try std.testing.expectEqual(0, bulk.logicalBytes());
        try bulk.await();
        try LoaderTestFixture.expectContents(allocator, io, &output, &LoaderTestFixture.contents);
        try std.testing.expectEqual(LoaderTestFixture.contents.len, loader.bytesLoaded());
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
        const handle = try loader.loadExecute(&fixture.store, &.{
            fixture.binding(fixture.value, &outputs[0]),
            fixture.binding(fixture.second, &outputs[1]),
        }, null);
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, handle.logicalBytes());
        try handle.await();
        defer for (&outputs) |*output| output.deinit();
        try LoaderTestFixture.expectContents(allocator, io, &outputs[0], &LoaderTestFixture.contents);
        try LoaderTestFixture.expectContents(allocator, io, &outputs[1], &LoaderTestFixture.second_contents);
        try std.testing.expectEqual(LoaderTestFixture.contents.len * 2, loader.bytesLoaded());
    }
}

test "loader deinit awaits open handles without running their executables" {
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
        _ = try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &never_written)}, null);
        const bulk = try loader.load(Model, &model, &buffers, &fixture.store, &.{}, null);
        loader.deinit();
        _ = bulk;
    }
}

test "loader window awaits the oldest handle before exceeding its budget" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    for (LoaderTestFixture.backends) |kind| {
        var loader = try fixture.loader(allocator, io, kind);
        defer loader.deinit();

        const input_bytes = try loader.executeInputBytesPerDevice(&fixture.exe);
        try std.testing.expectEqual(LoaderTestFixture.contents.len, input_bytes);
        var window: Window = .init(allocator, input_bytes, 4);
        defer window.deinit();

        var first: Buffer = undefined;
        try window.submit(&loader, &fixture.store, &.{fixture.binding(fixture.value, &first)}, null);
        try std.testing.expectEqual(@as(usize, 1), window.pending.items.len);
        var second: Buffer = undefined;
        // The budget holds one submission: the first is awaited before the second
        // is submitted.
        try window.submit(&loader, &fixture.store, &.{fixture.binding(fixture.second, &second)}, null);
        defer first.deinit();
        try std.testing.expectEqual(@as(usize, 1), window.pending.items.len);
        try std.testing.expectEqual(LoaderTestFixture.contents.len, loader.bytesLoaded());
        try LoaderTestFixture.expectContents(allocator, io, &first, &LoaderTestFixture.contents);
        try window.drain();
        defer second.deinit();
        try std.testing.expectEqual(@as(usize, 0), window.pending.items.len);
        try std.testing.expectEqual(@as(usize, 0), window.pending_bytes);
        try LoaderTestFixture.expectContents(allocator, io, &second, &LoaderTestFixture.second_contents);
    }
}

test "loader read failure fails every pending handle and later submissions" {
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
        const Broken = struct { missing: Tensor };
        const broken_model: Broken = .{ .missing = fixture.missing };
        var broken_buffers = try mem.bufferize(allocator, Broken, &broken_model);
        defer mem.deinitBufferized(allocator, Broken, &broken_buffers);

        var never_written: Buffer = undefined;
        const good = try loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &never_written)}, null);
        const broken = try loader.load(Broken, &broken_model, &broken_buffers, &fixture.store, &.{}, null);
        try std.testing.expectError(error.FileNotFound, broken.await());
        try std.testing.expectError(error.FileNotFound, good.await());
        try std.testing.expectError(error.FileNotFound, good.await());
        try LoaderTestFixture.expectLoadError(error.FileNotFound, loader.load(Model, &model, &buffers, &fixture.store, &.{}, null));
        try LoaderTestFixture.expectLoadError(error.FileNotFound, loader.loadExecute(&fixture.store, &.{fixture.binding(fixture.value, &never_written)}, null));
        try std.testing.expectError(error.FileNotFound, loader.awaitAll());
        try std.testing.expectEqual(@as(usize, 0), loader.bytesLoaded());
    }
}

test "loader initialization releases its workspace on invalid host budget" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var fixture: LoaderTestFixture = undefined;
    try fixture.init(allocator, io);
    defer fixture.deinit(allocator, io);
    const result: anyerror!void = if (Loader.init(allocator, io, fixture.platform, .{
        .max_host_bytes = 0,
    })) |value| unexpected: {
        var loader = value;
        loader.deinit();
        break :unexpected {};
    } else |err| err;
    try std.testing.expectError(error.InvalidDmaLoadConfig, result);
}
