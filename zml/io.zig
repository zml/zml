const std = @import("std");

const stdx = @import("stdx");
pub const VFS = @import("vfs");

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const dma = @import("io/dma_calibration.zig");
const direct_loader = @import("io/direct_loader.zig");
const load_limits = @import("io/limits.zig");
const loader_types = @import("io/loader_types.zig");
const platform_mod = @import("platform.zig");
const Exe = @import("exe.zig").Exe;
const mem = @import("mem.zig");
const meta = @import("meta.zig");
const Platform = platform_mod.Platform;
const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Tensor = @import("tensor.zig").Tensor;

const load_log = std.log.scoped(.@"zml/io/load");

pub const TensorStore = struct {
    registry: *safetensors.TensorRegistry,
    id_to_sources: std.AutoHashMapUnmanaged(Tensor.Id, []*safetensors.Tensor),
    allocator: std.mem.Allocator,
    arena: std.heap.ArenaAllocator,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry) TensorStore {
        const arena: std.heap.ArenaAllocator = .init(allocator);
        return .{
            .registry = registry,
            .id_to_sources = .empty,
            .allocator = allocator,
            .arena = arena,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.id_to_sources.deinit(self.allocator);
        self.arena.deinit();
    }

    fn putSourcesNoClobber(self: *TensorStore, id: Tensor.Id, sources: []*safetensors.Tensor) std.mem.Allocator.Error!void {
        const gop = try self.id_to_sources.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Id {} already has associated sources", .{id});
        }
        errdefer self.id_to_sources.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = sources;
    }

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn dupeSource(self: *TensorStore, key: []const u8) ?*safetensors.Tensor {
        const entry = self.getPtrFromKey(key) orelse return null;

        const copy = self.arena.allocator().create(safetensors.Tensor) catch @panic("OOM");
        copy.* = entry.*;

        return copy;
    }

    fn getPtrFromId(self: *const TensorStore, id: Tensor.Id) ?*safetensors.Tensor {
        const sources = self.id_to_sources.get(id) orelse return null;
        stdx.debug.assert(sources.len == 1, "Expect tensor with id {} to have only one source, got {}", .{ id, sources.len });
        return sources[0];
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: Tensor.Id, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const sources = self.id_to_sources.get(id) orelse return error.NotFound;
        stdx.debug.assert(sources.len == 1, "Expect tensor with id {} to have only one source, got {}", .{ id, sources.len });

        return sources[0].reader(io, buffer, .{});
    }

    pub fn getSourcesById(self: *const TensorStore, id: Tensor.Id) ?[]*safetensors.Tensor {
        return self.id_to_sources.get(id);
    }

    pub fn getShape(self: *const TensorStore, key: []const u8) ?Shape {
        const entry_ptr = self.getPtrFromKey(key) orelse return null;
        return entry_ptr.shape;
    }

    pub fn view(self: *TensorStore) View {
        return .{ .store = self };
    }

    pub const View = struct {
        store: *TensorStore,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn root(self: *const View) View {
            return .{
                .store = self.store,
            };
        }

        pub fn parent(self: *const View) View {
            const slice = self.prefix() orelse unreachable;
            const index = std.mem.lastIndexOfScalar(u8, slice[0 .. slice.len - 1], '.') orelse return self.root();
            var buffer: [256]u8 = undefined;
            @memcpy(buffer[0 .. index + 1], slice[0 .. index + 1]);
            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = index + 1,
            };
        }

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = makeKey(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ });

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        pub fn withLayer(self: *const View, index: usize) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = makeKey(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index });

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        pub fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn hasKey(self: *const View, subkey: []const u8) bool {
            var buffer: [256]u8 = undefined;
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            return for (self.store.registry.tensors.keys()) |k| {
                if (std.mem.startsWith(u8, k, key)) break true;
            } else false;
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            const source = self.store.dupeSource(key) orelse return null;

            const sources = self.store.arena.allocator().alloc(*safetensors.Tensor, 1) catch |e| std.debug.panic("Not handling {} errors", .{e});
            errdefer self.store.arena.allocator().free(sources);
            sources[0] = source;

            var shape = source.shape;
            shape = applyTags(shape, tagz);
            shape = applyPartitioning(shape, partitioning);

            const tensor: Tensor = .fromShape(shape);
            self.store.putSourcesNoClobber(tensor.id, sources) catch |e| std.debug.panic("Not handling {} errors", .{e});

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) Tensor {
            return self.maybeCreateTensor(subkey, tagz, partitioning) orelse
                stdx.debug.panic("Checkpoint has no tensor named {s}{s}", .{ self.prefix() orelse "", subkey });
        }

        fn applyTags(shape_: Shape, tagz: anytype) Shape {
            var shape = shape_;
            if (@TypeOf(tagz) != @TypeOf(null)) {
                switch (@typeInfo(@TypeOf(tagz))) {
                    .optional => if (tagz) |t| {
                        shape = shape.withTags(t);
                    },
                    else => shape = shape.withTags(tagz),
                }
            }
            return shape;
        }

        fn applyPartitioning(shape_: Shape, partitioning: anytype) Shape {
            var shape = shape_;

            if (@TypeOf(partitioning) == @TypeOf(null)) {
                @compileError("TensorStore.View.createTensor partitioning cannot be null; pass .replicated or an explicit partitioning");
            }

            switch (@typeInfo(@TypeOf(partitioning))) {
                .optional => @compileError("TensorStore.View.createTensor partitioning cannot be optional; pass .replicated or an explicit partitioning"),
                .enum_literal => switch (partitioning) {
                    .replicated => shape = shape.withReplicatedPartitioning(),
                    else => @compileError("Only .replicated is supported as a standalone partitioning enum literal"),
                },
                else => shape = shape.withPartitioning(partitioning),
            }

            return shape;
        }

        pub fn maybeCreateBinding(self: View, sources: []const []const u8, shape: Shape) ?Tensor {
            const arena = self.store.arena.allocator();

            var tensor_list = std.ArrayList(*safetensors.Tensor).initCapacity(arena, sources.len) catch |e| std.debug.panic("Not handling {} errors", .{e});
            defer tensor_list.deinit(arena);

            var buffer: [256]u8 = undefined;
            for (sources) |subkey| {
                const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
                const tensor = self.store.dupeSource(key) orelse return null;
                tensor_list.appendAssumeCapacity(tensor);
            }

            const tensors = tensor_list.toOwnedSlice(arena) catch unreachable;
            errdefer arena.free(tensors);

            const tensor: Tensor = .fromShape(shape);
            self.store.putSourcesNoClobber(tensor.id, tensors) catch |e| std.debug.panic("Not handling {} errors", .{e});

            return tensor;
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            return self.store.getShape(key);
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b makeKey(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            };
            return self.store.getShape(key);
        }

        pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
            var key_buffer: [256]u8 = undefined;
            const key = makeKey(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey });
            return self.store.getReader(key, io, buffer);
        }

        pub fn count(self: View) usize {
            var count_: usize = 0;
            const prefix_ = self.prefix() orelse "";
            var it = self.store.registry.tensors.iterator();
            while (it.next()) |item| {
                const key = item.key_ptr.*;
                if (std.mem.startsWith(u8, key, prefix_)) {
                    count_ += 1;
                }
            }
            return count_;
        }

        fn makeKey(buffer: []u8, comptime fmt: []const u8, args: anytype) []const u8 {
            const key = std.fmt.bufPrint(buffer, fmt, args) catch
                std.debug.panic("Expected key to be less than {} characters", .{buffer.len});
            return key;
        }
    };
};

const BufferedMemoryWriter = struct {
    io: std.Io,
    platform: *const Platform,
    shape: Shape,
    sharding: Sharding,
    buffer: *Buffer,
    interface: std.Io.Writer,

    fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding, buffer: *Buffer) !BufferedMemoryWriter {
        return .{
            .io = io,
            .platform = platform,
            .shape = shape,
            .sharding = sharding,
            .buffer = buffer,
            .interface = .{
                .buffer = try allocator.alloc(u8, shape.byteSize()),
                .vtable = &.{
                    .drain = std.Io.Writer.fixedDrain,
                    .flush = flush,
                    .rebase = std.Io.Writer.failingRebase,
                },
            },
        };
    }

    fn deinit(self: *BufferedMemoryWriter, allocator: std.mem.Allocator) void {
        if (self.interface.buffer.len > 0) {
            allocator.free(self.interface.buffer);
        }
    }

    fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *BufferedMemoryWriter = @alignCast(@fieldParentPtr("interface", w));

        self.buffer.* = Buffer.from(
            self.io,
            self.platform,
            self.shape,
            self.sharding,
            @ptrCast(self.interface.buffer),
            .{ .wait = true },
        ) catch return std.Io.Writer.Error.WriteFailed;
    }
};

const effectiveSourceRequestSize = load_limits.effectiveSourceRequestSize;

pub const default_dma_benchmark_block_sizes = dma.default_dma_benchmark_block_sizes;
pub const BenchTransferOptions = dma.BenchTransferOptions;
pub const max_load_read_parallelism = load_limits.max_read_parallelism;
pub const max_load_dma_parallelism = load_limits.max_dma_parallelism;
pub const max_load_read_request_size = load_limits.max_read_request_size;
const isDirectTransferPlatform = dma.isDirectTransferPlatform;
pub const initPlatformDma = dma.initPlatformDma;
pub const platformTransferSettings = dma.platformTransferSettings;
pub const deinitPlatformDma = dma.deinitPlatformDma;
pub const benchTransfer = dma.benchTransfer;

pub const Parallelism = loader_types.Parallelism;
const DirectLoader = direct_loader.DirectLoader;
const LoaderLoadSpec = direct_loader.LoadSpec;

fn prepareModelLoad(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    store: *const TensorStore,
    opts: Loader.Opts,
    comptime ModelType: type,
    model: *const ModelType,
    buffers: *Bufferized(ModelType),
) ![]LoaderLoadSpec {
    const tensor_count = meta.count(Tensor, model);
    const flattened = try allocator.alloc(*Buffer, tensor_count);
    defer allocator.free(flattened);
    meta.forEachVisit(buffers, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
            output[i] = buffer;
        }
    }.call, .{flattened});

    var specs: std.ArrayListUnmanaged(LoaderLoadSpec) = .empty;
    errdefer specs.deinit(allocator);
    try specs.ensureTotalCapacityPrecise(allocator, tensor_count);
    const Ctx = struct {
        platform: *const Platform,
        store: *const TensorStore,
        opts: Loader.Opts,
        buffers: []*Buffer,
        specs: *std.ArrayListUnmanaged(LoaderLoadSpec),
        err: ?anyerror = null,
    };
    var ctx: Ctx = .{
        .platform = platform,
        .store = store,
        .opts = opts,
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
            if (sources.len != 1) {
                load_log.debug("skipping fused tensor with {} sources; load it with Loader.loadExecute", .{sources.len});
                return;
            }
            const shape = tensor.shape();
            context.specs.appendAssumeCapacity(.{
                .source = sources[0],
                .shape = shape,
                .sharding = Sharding.pickSharding(
                    context.opts.shardings,
                    shape,
                    .explicit_axis_binding,
                ) orelse context.platform.replicated_sharding,
                .output = context.buffers[i],
            });
        }
    }.call, .{&ctx});
    if (ctx.err) |err| return err;
    return specs.toOwnedSlice(allocator);
}

pub const Loader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: Opts,
    backend: Backend,

    const Backend = union(enum) {
        direct: *DirectLoader,
        buffered: *BufferedLoader,
    };

    pub const Opts = loader_types.LoaderOptions;

    pub fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        store: *const TensorStore,
        opts: Opts,
    ) !Loader {
        try validateLoaderOpts(opts);
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .store = store,
            .opts = opts,
            .backend = if (isDirectTransferPlatform(platform))
                .{ .direct = try DirectLoader.create(allocator, io, platform, opts) }
            else
                .{ .buffered = try BufferedLoader.create(
                    allocator,
                    io,
                    platform,
                    opts.read_parallelism.initial(),
                ) },
        };
    }

    /// Atomically publishes all single-source tensors in `model`. Work may
    /// start before this method returns. Multi-source bindings remain explicit
    /// and must be submitted with `loadExecute`.
    pub fn load(
        self: *Loader,
        comptime ModelType: type,
        model: *const ModelType,
        buffers: *Bufferized(ModelType),
    ) !void {
        try self.checkOpen();
        const specs = try prepareModelLoad(
            self.allocator,
            self.platform,
            self.store,
            self.opts,
            ModelType,
            model,
            buffers,
        );
        defer self.allocator.free(specs);
        return switch (self.backend) {
            .direct => |direct| direct.loadPrepared(specs),
            .buffered => |buffered| buffered.loadPrepared(specs),
        };
    }

    /// Loads every source associated with `tensor`, drains the loader-wide
    /// epoch, and executes `exe` synchronously. The returned output is ready.
    pub fn loadExecute(
        self: *Loader,
        tensor: Tensor,
        output: *Buffer,
        exe: *const Exe,
    ) !void {
        try self.checkOpen();
        var binding = try PreparedExecutableBinding.init(
            self.allocator,
            self.platform,
            self.store,
            self.opts,
            tensor,
            exe,
        );
        defer binding.deinit();
        switch (self.backend) {
            .direct => |direct| try direct.loadBinding(binding.sources, binding.inputs, exe),
            .buffered => |buffered| try buffered.loadBinding(binding.sources, binding.inputs, exe),
        }
        try executeLoadedBinding(self.allocator, self.io, binding.inputs, output, exe);
    }

    /// Drains the current epoch and reopens the loader for later submissions.
    pub fn await(self: *Loader) !void {
        return switch (self.backend) {
            .direct => |direct| direct.await(),
            .buffered => |buffered| buffered.await(),
        };
    }

    pub fn bytesLoaded(self: *const Loader) usize {
        return switch (self.backend) {
            .direct => |direct| direct.bytesLoaded(),
            .buffered => |buffered| buffered.bytesLoaded(),
        };
    }

    pub fn deinit(self: *Loader) void {
        switch (self.backend) {
            .direct => |direct| direct.destroy(),
            .buffered => |buffered| buffered.destroy(),
        }
        self.* = undefined;
    }

    fn checkOpen(self: *Loader) !void {
        return switch (self.backend) {
            .direct => |direct| direct.checkOpen(),
            .buffered => |buffered| buffered.checkOpen(),
        };
    }
};

fn validateLoaderOpts(opts: Loader.Opts) !void {
    _ = try effectiveSourceRequestSize(opts.load_profile.read_chunk_size, 0);
    const initial = opts.read_parallelism.initial();
    const maximum = opts.read_parallelism.maximum();
    if (initial == 0 or maximum < initial or maximum > max_load_read_parallelism)
        return error.InvalidLoadParallelism;
}

const PreparedExecutableBinding = struct {
    allocator: std.mem.Allocator,
    sources: []const *safetensors.Tensor,
    inputs: []Buffer,

    fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        store: *const TensorStore,
        opts: Loader.Opts,
        tensor: Tensor,
        exe: *const Exe,
    ) !PreparedExecutableBinding {
        const sources = store.getSourcesById(tensor.id) orelse return error.NotFound;
        const output_sharding = Sharding.pickSharding(
            opts.shardings,
            tensor.shape(),
            .explicit_axis_binding,
        ) orelse platform.replicated_sharding;
        try validateExecutableBinding(platform, tensor, sources, exe, output_sharding);
        const inputs = try allocator.alloc(Buffer, sources.len);
        for (inputs, exe.input_shapes, exe.input_shardings) |*input, shape, sharding| {
            input.* = .{
                ._platform = platform,
                ._shape = shape,
                ._sharding = sharding.resolve(platform),
                ._shards = .empty,
            };
        }
        return .{ .allocator = allocator, .sources = sources, .inputs = inputs };
    }

    fn deinit(self: *PreparedExecutableBinding) void {
        for (self.inputs) |*input| input.deinit();
        self.allocator.free(self.inputs);
        self.* = undefined;
    }
};

fn validateExecutableBinding(
    platform: *const Platform,
    tensor: Tensor,
    sources: []const *safetensors.Tensor,
    exe: *const Exe,
    expected_output_sharding: Sharding,
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
    try validateSamePlacement(
        tensor.shape(),
        expected_output_sharding.resolve(platform),
        exe.output_shardings[0].resolve(platform),
    );
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

fn validateSamePlacement(shape: Shape, expected: Sharding, actual: Sharding) !void {
    const expected_devices = expected.devicesInCanonicalOrder();
    const actual_devices = actual.devicesInCanonicalOrder();
    if (expected_devices.len != actual_devices.len) return error.ExecutablePlacementMismatch;
    const expected_placement = try expected.placement(shape);
    const actual_placement = try actual.placement(shape);
    if (!expected_placement.shape.eql(actual_placement.shape))
        return error.ExecutablePlacementMismatch;
    for (expected_devices, actual_devices) |expected_device, actual_device| {
        const expected_slices = expected_placement.slices(expected_device.coords);
        const actual_slices = actual_placement.slices(actual_device.coords);
        if (expected_device.id != actual_device.id or expected_slices.len != actual_slices.len)
            return error.ExecutablePlacementMismatch;
        for (expected_slices.constSlice(), actual_slices.constSlice()) |expected_slice, actual_slice| {
            if (expected_slice.start != actual_slice.start or expected_slice.size != actual_slice.size)
                return error.ExecutablePlacementMismatch;
        }
    }
}

const BufferedLoader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    group: stdx.Io.LimitedGroup,
    bytes_loaded: std.atomic.Value(usize) = .init(0),
    epoch_logical_bytes: usize = 0,
    first_error: std.atomic.Value(u16) = .init(0),
    epoch_active: bool = false,

    fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        read_parallelism: usize,
    ) !*BufferedLoader {
        const self = try allocator.create(BufferedLoader);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .group = .init(read_parallelism),
        };
        return self;
    }

    fn recordError(self: *BufferedLoader, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn checkOpen(self: *BufferedLoader) !void {
        const code = self.first_error.load(.acquire);
        if (code != 0) return @errorFromInt(code);
        if (self.epoch_active) return error.LoaderEpochActive;
    }

    fn submitOne(
        self: *BufferedLoader,
        source: *safetensors.Tensor,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
    ) void {
        self.group.async(self.io, struct {
            fn run(
                loader: *BufferedLoader,
                source_: *safetensors.Tensor,
                shape_: Shape,
                sharding_: Sharding,
                output_: *Buffer,
            ) void {
                if (loader.first_error.load(.acquire) != 0) return;
                var reader = source_.reader(loader.io, &.{}, .{}) catch |err| {
                    loader.recordError(err);
                    return;
                };
                defer reader.deinit();
                var writer = BufferedMemoryWriter.init(
                    loader.allocator,
                    loader.io,
                    loader.platform,
                    shape_,
                    sharding_,
                    output_,
                ) catch |err| {
                    loader.recordError(err);
                    return;
                };
                defer writer.deinit(loader.allocator);
                _ = reader.interface.streamRemaining(&writer.interface) catch |err| {
                    loader.recordError(err);
                    return;
                };
                writer.interface.flush() catch |err| {
                    loader.recordError(err);
                    return;
                };
            }
        }.run, .{ self, source, shape, sharding, output });
    }

    fn loadPrepared(self: *BufferedLoader, specs: []const LoaderLoadSpec) !void {
        try self.checkOpen();
        var logical_bytes: usize = 0;
        for (specs) |item| {
            logical_bytes = try std.math.add(usize, logical_bytes, item.source.shape.byteSize());
        }
        for (specs) |item| {
            self.submitOne(item.source, item.shape, item.sharding, item.output);
        }
        self.epoch_logical_bytes = logical_bytes;
        self.epoch_active = true;
    }

    fn await(self: *BufferedLoader) !void {
        if (!self.epoch_active) {
            try self.checkOpen();
            return;
        }
        self.group.await(self.io) catch |err| self.recordError(err);
        self.epoch_active = false;
        const code = self.first_error.load(.acquire);
        if (code == 0) _ = self.bytes_loaded.fetchAdd(self.epoch_logical_bytes, .monotonic);
        self.epoch_logical_bytes = 0;
        if (code != 0) return @errorFromInt(code);
    }

    fn bytesLoaded(self: *const BufferedLoader) usize {
        return self.bytes_loaded.load(.acquire);
    }

    fn loadBinding(
        self: *BufferedLoader,
        sources: []const *safetensors.Tensor,
        inputs: []Buffer,
        exe: *const Exe,
    ) !void {
        try self.checkOpen();
        var logical_bytes: usize = 0;
        for (sources) |source| {
            logical_bytes = try std.math.add(usize, logical_bytes, source.shape.byteSize());
        }
        for (sources, exe.input_shapes, exe.input_shardings, inputs) |source, shape, sharding, *input| {
            self.submitOne(source, shape, sharding.resolve(self.platform), input);
        }
        self.epoch_logical_bytes = logical_bytes;
        self.epoch_active = true;
        try self.await();
    }

    fn destroy(self: *BufferedLoader) void {
        self.await() catch {};
        self.allocator.destroy(self);
    }
};

fn executeLoadedBinding(
    allocator: std.mem.Allocator,
    io: std.Io,
    inputs: []Buffer,
    output: *Buffer,
    exe: *const Exe,
) !void {
    var args = try exe.args(allocator);
    defer args.deinit(allocator);
    var results = try exe.results(allocator);
    defer results.deinit(allocator);
    args.set(.{inputs});
    exe.callOpts(io, args, &results, .{ .wait = true });
    output.* = results.get(Buffer);
}

test "loader supports repeated synchronous and explicit epochs" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    const contents = [_]u8{ 1, 2, 3, 4 };

    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    const file = try tmp.dir.createFile(io, "weights.bin", .{ .read = true });
    try file.writePositionalAll(io, &contents, 0);
    var path_buffer: [1024]u8 = undefined;
    const path_len = try file.realPath(io, &path_buffer);
    file.close(io);

    var registry: safetensors.TensorRegistry = .init(allocator);
    defer registry.deinit();
    try registry.registerTensor(.{
        .file_uri = path_buffer[0..path_len],
        .name = "value",
        .shape = .init(.{contents.len}, .u8),
        .offset = 0,
    });
    var store: TensorStore = .fromRegistry(allocator, &registry);
    defer store.deinit();
    const tensor = store.view().createTensor("value", null, .replicated);

    const platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);
    const Identity = struct {
        fn call(input: Tensor) Tensor {
            return input;
        }
    };
    var exe = try platform.compileFn(allocator, io, Identity.call, .{tensor}, .{});
    defer exe.deinit();
    var loader = try Loader.init(allocator, io, platform, &store, .{
        .read_parallelism = .{ .fixed = 1 },
    });
    defer loader.deinit();

    var first: Buffer = undefined;
    try loader.loadExecute(tensor, &first, &exe);
    defer first.deinit();
    const first_loaded = try first.toSliceAlloc(allocator, io);
    defer first_loaded.free(allocator);
    try std.testing.expectEqualSlices(u8, &contents, first_loaded.constData());
    var second: Buffer = undefined;
    try loader.loadExecute(tensor, &second, &exe);
    defer second.deinit();

    const Model = struct { value: Tensor };
    const model: Model = .{ .value = tensor };
    var buffers = try mem.bufferize(allocator, Model, &model);
    defer mem.deinitBufferized(allocator, Model, &buffers);
    try loader.load(Model, &model, &buffers);
    try std.testing.expectError(error.LoaderEpochActive, loader.load(Model, &model, &buffers));
    try loader.await();

    try std.testing.expectEqual(contents.len * 3, loader.bytesLoaded());
    const loaded = try buffers.value.toSliceAlloc(allocator, io);
    defer loaded.free(allocator);
    try std.testing.expectEqualSlices(u8, &contents, loaded.constData());
}
