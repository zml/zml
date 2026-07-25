const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Exe = @import("exe.zig").Exe;
const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const CreateOptions = @import("platform.zig").CreateOptions;
const mem = @import("mem.zig");
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
const Platform = @import("platform.zig").Platform;
const tracer = @import("profiling/tracer.zig");
const safetensors = @import("safetensors.zig");
const Shape = @import("shape.zig").Shape;
const Sharding = @import("Sharding.zig");
const Placement = Sharding.Placement;
const Tensor = @import("tensor.zig").Tensor;

const log = std.log.scoped(.@"zml/io");
const load_log = std.log.scoped(.@"zml/io/load");

pub const TensorStore = struct {
    registry: *safetensors.TensorRegistry,
    id_to_sources: std.AutoHashMapUnmanaged(usize, []*safetensors.Tensor),
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

    fn putSourcesNoClobber(self: *TensorStore, id: usize, sources: []*safetensors.Tensor) std.mem.Allocator.Error!void {
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

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*safetensors.Tensor {
        const sources = self.id_to_sources.get(id) orelse return null;
        stdx.debug.assert(sources.len == 1, "Expect tensor with id {} to have only one source, got {}", .{ id, sources.len });
        return sources[0];
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const sources = self.id_to_sources.get(id) orelse return error.NotFound;
        stdx.debug.assert(sources.len == 1, "Expect tensor with id {} to have only one source, got {}", .{ id, sources.len });

        return sources[0].reader(io, buffer, .{});
    }

    pub fn getSourcesById(self: *const TensorStore, id: usize) ?[]*safetensors.Tensor {
        return self.id_to_sources.get(id);
    }

    pub fn getShape(self: *const TensorStore, key: []const u8) ?Shape {
        const entry_ptr = self.getPtrFromKey(key) orelse return null;
        return entry_ptr.shape;
    }

    fn getBorrowedPositionalReaderById(self: *const TensorStore, id: usize, io: std.Io, file: std.Io.File) !safetensors.TensorReader {
        const tensor_desc = self.getPtrFromId(id) orelse return error.NotFound;
        return .initBorrowedPositional(io, tensor_desc.*, file);
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

        fn prefix(self: *const View) ?[]const u8 {
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
            const source = self.store.getPtrFromKey(key) orelse return null;

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
            return self.maybeCreateTensor(subkey, tagz, partitioning).?;
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
                const tensor = self.store.getPtrFromKey(key) orelse return null;
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

pub const Loader = struct {
    allocator: std.mem.Allocator,
    platform: *const Platform,
    dma_allocators: []const mem.DmaAllocator,
    dma_chunk_size: usize,
    pinned_buffer_pools: []mem.DynamicBufferPool,
    group: stdx.Io.LimitedGroup,
    bytes_loaded: std.atomic.Value(usize) = .init(0),

    pub const Opts = struct {
        pub const default: Opts = .{
            .parallelism = 1,
            .dma_chunks = 2,
            .dma_chunk_size = 4096,
        };
        parallelism: usize,
        dma_chunks: usize,
        dma_chunk_size: usize,
    };

    pub fn init(allocator: std.mem.Allocator, platform: *const Platform, opts: Opts) !Loader {
        const pool_count = platform.devices.len;
        const dma_allocators = try allocator.alloc(mem.DmaAllocator, pool_count);
        errdefer allocator.free(dma_allocators);
        for (platform.devices, 0..) |*device, i| {
            dma_allocators[i] = .init(allocator, device);
        }

        const buffer_pools = try allocator.alloc(mem.DynamicBufferPool, pool_count);
        errdefer allocator.free(buffer_pools);
        for (buffer_pools) |*pool_| {
            pool_.* = .init(opts.dma_chunks, opts.dma_chunk_size);
        }
        errdefer for (buffer_pools, 0..) |*pool_, i| {
            pool_.deinit(dma_allocators[i].allocator());
        };

        return .{
            .allocator = allocator,
            .platform = platform,
            .dma_allocators = dma_allocators,
            .dma_chunk_size = opts.dma_chunk_size,
            .pinned_buffer_pools = buffer_pools,
            .group = .init(opts.parallelism),
        };
    }

    pub fn deinit(self: Loader) void {
        for (self.pinned_buffer_pools, 0..) |*pool, i| pool.deinit(self.dma_allocators[i].allocator());
        self.allocator.free(self.pinned_buffer_pools);
        self.allocator.free(self.dma_allocators);
    }

    pub fn await(self: *Loader, io: std.Io) std.Io.Cancelable!void {
        return self.group.await(io);
    }

    pub const LoadOpts = struct {
        progress: ?*std.Progress.Node = null,
    };

    pub fn load(self: *Loader, io: std.Io, comptime T: type, model: *const T, buffers: *Bufferized(T), store: *const TensorStore, shardings: []const Sharding, opts: LoadOpts) void {
        const tensor_count = meta.count(Tensor, model);

        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();

        const flattened_buffers = arena.allocator().alloc(*Buffer, tensor_count) catch @panic("Errors can't be handled in `loadInner`");
        meta.forEachVisit(buffers, *Buffer, struct {
            fn call(i: usize, buffer: *Buffer, flattened_buffers_: []*Buffer) void {
                flattened_buffers_[i] = buffer;
            }
        }.call, .{flattened_buffers});

        const Ctx = struct {
            self: *Loader,
            io: std.Io,
            store: *const TensorStore,
            shardings: []const Sharding,
            buffers: []*Buffer,
            opts: LoadOpts,
        };

        var ctx: Ctx = .{
            .self = self,
            .io = io,
            .store = store,
            .shardings = shardings,
            .buffers = flattened_buffers,
            .opts = opts,
        };

        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, ctx_: *Ctx) void {
                ctx_.self.group.async(ctx_.io, defaultCallback, .{ ctx_.self, ctx_.io, tensor, ctx_.buffers[i], ctx_.store, ctx_.shardings, ctx_.opts });
            }
        }.call, .{&ctx});
    }

    fn defaultCallback(self: *Loader, io: std.Io, tensor: *const Tensor, buffer: *Buffer, store: *const TensorStore, shardings: []const Sharding, opts: LoadOpts) void {
        const sources = store.getSourcesById(tensor.id) orelse {
            std.log.warn("Failed to get sources for tensor with id: {}", .{tensor.id});
            return;
        };

        if (sources.len != 1) {
            std.debug.panic("Expected loaded tensor to have only 1 source, got {}", .{sources.len});
        }

        self.loadSingleInner(io, sources[0], tensor.shape(), buffer, shardings, opts) catch |e| {
            log.err("Errors are not handled in `defaultCallback`, got {}", .{e});
            unreachable;
        };
    }

    fn loadSingle(self: *Loader, io: std.Io, source: *safetensors.Tensor, shape: Shape, buffer: *Buffer, loaded: *bool, shardings: []const Sharding, opts: LoadOpts) void {
        self.loadSingleInner(io, source, shape, buffer, shardings, opts) catch |e| {
            log.err("Failed to load tensor {s}: {}", .{ source.name, e });
            loaded.* = false;
        };
        loaded.* = true;
    }

    fn loadSingleInner(self: *Loader, io: std.Io, source: *safetensors.Tensor, shape: Shape, buffer: *Buffer, shardings: []const Sharding, opts: LoadOpts) !void {
        var reader = try source.reader(io, &.{}, .{});
        defer reader.deinit();

        const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse blk: {
            log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ reader.tensor.name, shape });
            break :blk self.platform.replicated_sharding;
        };

        var writer = try BufferedMemoryWriter.init(self.allocator, io, self.platform, shape, sharding, buffer);
        defer writer.deinit(self.allocator);

        const scale = 1024;

        if (opts.progress) |progress| {
            var node = progress.start(reader.tensor.name, reader.tensor.shape.byteSize() / scale);
            defer node.end();
            var progress_writer: ProgressWriter = .init(&writer.interface, &node, .{ .scale = scale });
            const total = try reader.interface.streamRemaining(&progress_writer.interface);
            try progress_writer.interface.flush();
            _ = self.bytes_loaded.fetchAdd(total, .monotonic);
        } else {
            const total = try reader.interface.streamRemaining(&writer.interface);
            try writer.interface.flush();
            _ = self.bytes_loaded.fetchAdd(total, .monotonic);
        }
    }

    pub fn loadExecute(
        self: *Loader,
        arena: std.mem.Allocator,
        io: std.Io,
        tensor: Tensor,
        buffer: *Buffer,
        store: *const TensorStore,
        shardings: []const Sharding,
        exe: *const Exe,
        opts: LoadOpts,
    ) !void {
        const sources = store.getSourcesById(tensor.id) orelse return error.NotFound;
        const buffers = try arena.alloc(Buffer, sources.len);
        const loaded = try arena.alloc(bool, sources.len);
        @memset(loaded, false);
        defer for (buffers, loaded) |*b, l| if (l) b.deinit();

        var node = if (opts.progress) |progress| b: {
            var writer = std.Io.Writer.Allocating.init(arena);
            try writer.writer.writeAll("Running executable on ");
            for (sources, 0..) |source, i| {
                try writer.writer.print("{s}{s}", .{ if (i != 0) ", " else "", source.name });
            }

            break :b progress.start(writer.written(), 1);
        } else null;
        defer if (node) |*n| n.end();

        for (sources, 0..) |source, i| {
            self.group.async(io, loadSingle, .{ self, io, source, source.shape, &buffers[i], &loaded[i], shardings, .{} });
        }
        try self.group.await(io);

        if (std.mem.findScalar(bool, loaded, false)) |_| {
            return error.LoadFailed;
        }

        var args = try exe.args(arena);
        var results = try exe.results(arena);

        args.set(.{buffers});
        exe.call(args, &results);

        buffer.* = results.get(Buffer);
    }
};

pub const ProgressWriter = struct {
    inner: *std.Io.Writer,
    progress: *std.Progress.Node,
    interface: std.Io.Writer,
    total: usize = 0,
    scale: usize,

    pub const InitOpts = struct {
        scale: usize = 1,
    };

    pub fn init(inner_: *std.Io.Writer, progress_: *std.Progress.Node, opts: InitOpts) ProgressWriter {
        return .{
            .inner = inner_,
            .progress = progress_,
            .scale = opts.scale,
            .interface = .{
                .buffer = inner_.buffer,
                .end = inner_.end,
                .vtable = &.{
                    .drain = drain,
                    .flush = flush,
                    .rebase = rebase,
                    .sendFile = sendFile,
                },
            },
        };
    }

    pub fn pre(self: *ProgressWriter) usize {
        self.inner.buffer = self.interface.buffer;
        self.inner.end = self.interface.end;
        return self.inner.end;
    }

    pub fn post(self: *ProgressWriter, len_pre: usize, total: usize) void {
        self.interface.buffer = self.inner.buffer;
        self.interface.end = self.inner.end;
        const drained_pre = len_pre -| self.interface.end;
        self.total += drained_pre + total;
        self.progress.setCompletedItems(self.total / self.scale);
    }

    pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        errdefer self.post(len_pre, 0);
        const total = try self.inner.vtable.drain(self.inner, data, splat);
        self.post(len_pre, total);
        return total;
    }

    pub fn sendFile(w: *std.Io.Writer, file_reader: *std.Io.File.Reader, limit: std.Io.Limit) std.Io.Writer.FileError!usize {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        errdefer self.post(len_pre, 0);
        const total = try self.inner.vtable.sendFile(self.inner, file_reader, limit);
        self.post(len_pre, total);
        return total;
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        defer self.post(len_pre, 0);
        try self.inner.vtable.flush(self.inner);
    }

    pub fn rebase(w: *std.Io.Writer, preserve: usize, capacity: usize) std.Io.Writer.Error!void {
        const self: *ProgressWriter = @alignCast(@fieldParentPtr("interface", w));
        const len_pre = self.pre();
        defer self.post(len_pre, 0);
        try self.inner.vtable.rebase(self.inner, preserve, capacity);
    }
};

pub const BufferedMemoryWriter = struct {
    io: std.Io,
    platform: *const Platform,
    shape: Shape,
    sharding: Sharding,
    buffer: *Buffer,
    interface: std.Io.Writer,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, platform: *const Platform, shape: Shape, sharding: Sharding, buffer: *Buffer) !BufferedMemoryWriter {
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

    pub fn deinit(self: *BufferedMemoryWriter, allocator: std.mem.Allocator) void {
        if (self.interface.buffer.len > 0) {
            allocator.free(self.interface.buffer);
        }
    }

    pub fn flush(w: *std.Io.Writer) std.Io.Writer.Error!void {
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

const DispatchSpans = struct {
    const DispatchSpan = struct {
        start: usize,
        end: usize,
        writer_offset: usize,
        primary_writer: usize,
        mirror_writer_start: usize,
        mirror_writer_len: usize,
    };

    const PlacementSpan = struct {
        writer_index: usize,
        start: usize,
        len: usize,
        order: usize,
    };

    spans: []DispatchSpan,
    mirror_writers: []usize,

    fn init(allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) !DispatchSpans {
        const placement = try sharding.placement(shape);
        const ordered_devices = sharding.devicesInCanonicalOrder();

        var placement_span_count: usize = 0;
        for (ordered_devices) |device| {
            placement_span_count += placementSpanCount(shape, placement.slices(device.coords).constSlice());
        }

        var placement_spans: std.ArrayList(PlacementSpan) = try .initCapacity(allocator, placement_span_count);
        defer placement_spans.deinit(allocator);

        const byte_strides = shape.computeByteStrides();

        for (ordered_devices, 0..) |device, writer_index| {
            appendShardPlacementSpans(&placement_spans, shape, placement.slices(device.coords).constSlice(), byte_strides.constSlice(), writer_index);
        }

        std.debug.assert(placement_spans.items.len == placement_span_count);

        var spans: std.ArrayList(DispatchSpan) = try .initCapacity(allocator, placement_spans.items.len);
        errdefer spans.deinit(allocator);

        var mirror_writers: std.ArrayList(usize) = try .initCapacity(allocator, placement_spans.items.len);
        errdefer mirror_writers.deinit(allocator);

        try deduplicateByRange(allocator, placement_spans.items, shape.byteSize(), &spans, &mirror_writers);

        // Record the final packed offset once, while spans are still in global
        // file order. Positional request tasks can then finish out of order
        // without mutating a writer cursor.
        const writer_offsets = try allocator.alloc(usize, ordered_devices.len);
        defer allocator.free(writer_offsets);
        @memset(writer_offsets, 0);
        for (spans.items) |*span| {
            span.writer_offset = writer_offsets[span.primary_writer];
            writer_offsets[span.primary_writer] += span.end - span.start;
            const mirror_end = span.mirror_writer_start + span.mirror_writer_len;
            for (mirror_writers.items[span.mirror_writer_start..mirror_end]) |writer_index| {
                if (writer_offsets[writer_index] != span.writer_offset) return error.InconsistentReplicaLayout;
                writer_offsets[writer_index] += span.end - span.start;
            }
        }

        const spans_ = try spans.toOwnedSlice(allocator);
        errdefer allocator.free(spans_);

        const mirror_writers_ = try mirror_writers.toOwnedSlice(allocator);
        errdefer allocator.free(mirror_writers_);

        return .{
            .spans = spans_,
            .mirror_writers = mirror_writers_,
        };
    }

    fn deinit(self: DispatchSpans, allocator: std.mem.Allocator) void {
        allocator.free(self.spans);
        allocator.free(self.mirror_writers);
    }

    fn writerMask(self: DispatchSpans, span: DispatchSpan) u64 {
        var mask = @as(u64, 1) << @intCast(span.primary_writer);
        const mirror_end = span.mirror_writer_start + span.mirror_writer_len;
        for (self.mirror_writers[span.mirror_writer_start..mirror_end]) |writer_index| {
            mask |= @as(u64, 1) << @intCast(writer_index);
        }
        return mask;
    }

    fn spanIndexAt(self: DispatchSpans, offset: usize) ?usize {
        var low: usize = 0;
        var high = self.spans.len;
        while (low < high) {
            const middle = low + (high - low) / 2;
            const span = self.spans[middle];
            if (offset < span.start) {
                high = middle;
            } else if (offset >= span.end) {
                low = middle + 1;
            } else {
                return middle;
            }
        }
        return null;
    }

    fn deduplicateByRange(
        allocator: std.mem.Allocator,
        placement_spans: []PlacementSpan,
        total_bytes: usize,
        spans: *std.ArrayList(DispatchSpan),
        mirror_writers: *std.ArrayList(usize),
    ) !void {
        const SortContext = struct {
            fn lessThan(_: void, lhs: PlacementSpan, rhs: PlacementSpan) bool {
                if (lhs.start != rhs.start) return lhs.start < rhs.start;
                if (lhs.len != rhs.len) return lhs.len < rhs.len;
                return lhs.order < rhs.order;
            }
        };

        std.mem.sort(PlacementSpan, placement_spans, {}, SortContext.lessThan);

        var i: usize = 0;
        var cursor: usize = 0;
        while (i < placement_spans.len) {
            const span = placement_spans[i];
            if (span.start != cursor) return error.NonContiguousShardPlacement;

            const mirror_writer_start = mirror_writers.items.len;
            var j = i + 1;
            while (j < placement_spans.len) : (j += 1) {
                const mirror = placement_spans[j];
                if (mirror.start != span.start or mirror.len != span.len) break;
                try mirror_writers.append(allocator, mirror.writer_index);
            }

            try spans.append(allocator, .{
                .start = span.start,
                .end = span.start + span.len,
                .writer_offset = 0,
                .primary_writer = span.writer_index,
                .mirror_writer_start = mirror_writer_start,
                .mirror_writer_len = j - i - 1,
            });
            cursor += span.len;
            i = j;
        }

        if (cursor != total_bytes) return error.NonContiguousShardPlacement;
    }

    fn appendPlacementSpan(placement_spans: *std.ArrayList(PlacementSpan), writer_index: usize, start: usize, len: usize) void {
        placement_spans.appendAssumeCapacity(.{
            .writer_index = writer_index,
            .start = start,
            .len = len,
            .order = placement_spans.items.len,
        });
    }

    fn appendShardPlacementSpans(
        placement_spans: *std.ArrayList(PlacementSpan),
        shape: Shape,
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        writer_index: usize,
    ) void {
        if (shape.rank() == 0) {
            appendPlacementSpan(placement_spans, writer_index, 0, shape.byteSize());
            return;
        }

        appendShardAxisPlacementSpans(placement_spans, slices, byte_strides, writer_index, 0, contiguousSliceAxis(shape, slices), 0);
    }

    fn appendShardAxisPlacementSpans(
        placement_spans: *std.ArrayList(PlacementSpan),
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        writer_index: usize,
        axis: usize,
        contiguous_axis: usize,
        base_start: i64,
    ) void {
        const slice = slices[axis];
        if (slice.size == 0) return;

        if (axis == contiguous_axis) {
            const span_start: usize = @intCast(base_start + slice.start * byte_strides[axis]);
            const span_len: usize = @intCast(slice.size * byte_strides[axis]);
            appendPlacementSpan(placement_spans, writer_index, span_start, span_len);
            return;
        }

        var i: i64 = 0;
        while (i < slice.size) : (i += 1) {
            const child_start = base_start + (slice.start + i) * byte_strides[axis];
            appendShardAxisPlacementSpans(placement_spans, slices, byte_strides, writer_index, axis + 1, contiguous_axis, child_start);
        }
    }

    fn placementSpanCount(shape: Shape, slices: []const Placement.Slice1d) usize {
        if (shape.rank() == 0) return 1;

        const contiguous_axis = contiguousSliceAxis(shape, slices);
        var count: usize = 1;
        for (slices[0..contiguous_axis]) |slice| {
            count *= @intCast(slice.size);
        }
        return count;
    }

    fn contiguousSliceAxis(shape: Shape, slices: []const Placement.Slice1d) usize {
        var axis = shape.rank() - 1;
        while (axis > 0) {
            const slice = slices[axis];
            if (slice.start != 0 or slice.size != shape.dim(axis)) break;
            axis -= 1;
        }
        return axis;
    }
};

/// Pure description of one source request. `segments` are in file order and
/// identify where each source fragment lands inside a DMA block. `blocks` are
/// independently contiguous in every destination selected by `writer_mask`.
const VectoredRequestPlan = struct {
    const Block = struct {
        writer_mask: u64,
        destination_offset: usize,
        len: usize = 0,
    };

    const Segment = struct {
        block_index: usize,
        block_offset: usize,
        len: usize,
    };

    const Builder = struct {
        writer_mask: u64,
        current_block: ?usize = null,
        used: usize = 0,
        next_destination: usize,
    };

    blocks: []Block,
    segments: []Segment,

    fn init(
        allocator: std.mem.Allocator,
        dispatch_spans: DispatchSpans,
        source_offset: usize,
        request_len: usize,
        block_size: usize,
    ) !VectoredRequestPlan {
        if (block_size == 0) return error.InvalidBlockSize;
        const total = if (dispatch_spans.spans.len == 0) 0 else dispatch_spans.spans[dispatch_spans.spans.len - 1].end;
        const request_end = std.math.add(usize, source_offset, request_len) catch return error.OutOfBounds;
        if (source_offset > total or request_end > total) return error.OutOfBounds;

        var blocks: std.ArrayList(Block) = .empty;
        errdefer blocks.deinit(allocator);
        var segments: std.ArrayList(Segment) = .empty;
        errdefer segments.deinit(allocator);
        if (request_len == 0) {
            const owned_blocks = try blocks.toOwnedSlice(allocator);
            errdefer allocator.free(owned_blocks);
            return .{
                .blocks = owned_blocks,
                .segments = try segments.toOwnedSlice(allocator),
            };
        }

        var builders: [Platform.MAX_NUM_DEVICES]Builder = undefined;
        var builder_count: usize = 0;
        var cursor = source_offset;
        var span_index = dispatch_spans.spanIndexAt(cursor) orelse return error.OutOfBounds;
        while (cursor < request_end) {
            const span = dispatch_spans.spans[span_index];
            const span_offset = cursor - span.start;
            var remaining = @min(request_end, span.end) - cursor;
            const writer_mask = dispatch_spans.writerMask(span);
            const destination = span.writer_offset + span_offset;

            var builder_index: usize = 0;
            while (builder_index < builder_count and builders[builder_index].writer_mask != writer_mask) : (builder_index += 1) {}
            if (builder_index == builder_count) {
                if (builder_count == builders.len) return error.TooManyDestinationSets;
                builders[builder_count] = .{
                    .writer_mask = writer_mask,
                    .next_destination = destination,
                };
                builder_count += 1;
            }
            const builder = &builders[builder_index];
            if (builder.next_destination != destination) return error.NonContiguousShardPlacement;

            while (remaining > 0) {
                if (builder.current_block == null or builder.used == block_size) {
                    try blocks.append(allocator, .{
                        .writer_mask = writer_mask,
                        .destination_offset = builder.next_destination,
                    });
                    builder.current_block = blocks.items.len - 1;
                    builder.used = 0;
                }
                const block_index = builder.current_block.?;
                const take = @min(remaining, block_size - builder.used);
                if (segments.items.len > 0) {
                    const previous = &segments.items[segments.items.len - 1];
                    if (previous.block_index == block_index and previous.block_offset + previous.len == builder.used) {
                        previous.len += take;
                    } else {
                        try segments.append(allocator, .{
                            .block_index = block_index,
                            .block_offset = builder.used,
                            .len = take,
                        });
                    }
                } else {
                    try segments.append(allocator, .{
                        .block_index = block_index,
                        .block_offset = builder.used,
                        .len = take,
                    });
                }
                builder.used += take;
                builder.next_destination += take;
                blocks.items[block_index].len += take;
                remaining -= take;
                cursor += take;
            }
            if (cursor == span.end) span_index += 1;
        }

        const owned_blocks = try blocks.toOwnedSlice(allocator);
        errdefer allocator.free(owned_blocks);
        return .{
            .blocks = owned_blocks,
            .segments = try segments.toOwnedSlice(allocator),
        };
    }

    fn deinit(self: VectoredRequestPlan, allocator: std.mem.Allocator) void {
        allocator.free(self.blocks);
        allocator.free(self.segments);
    }
};

const VectoredLoadMetrics = struct {
    read_operations: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    read_ns: std.atomic.Value(u64) = .init(0),
    pool_waits: std.atomic.Value(u64) = .init(0),
    pool_wait_ns: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    submitted_bytes: std.atomic.Value(u64) = .init(0),
    committed_bytes: std.atomic.Value(u64) = .init(0),
    dma_ns: std.atomic.Value(u64) = .init(0),
    active_reads: std.atomic.Value(usize) = .init(0),
    peak_reads: std.atomic.Value(usize) = .init(0),

    fn beginRead(self: *VectoredLoadMetrics) void {
        const active = self.active_reads.fetchAdd(1, .acq_rel) + 1;
        var peak = self.peak_reads.load(.acquire);
        while (active > peak) {
            peak = self.peak_reads.cmpxchgWeak(peak, active, .release, .acquire) orelse break;
        }
    }
};

const VectoredTensorTransfer = struct {
    const Target = struct {
        manager: *pjrt.AsyncHostToDeviceTransferManager,
        pjrt_buffer: *pjrt.Buffer,
        device_index: usize,
        total: usize,
        submitted_bytes: std.atomic.Value(usize) = .init(0),
        final_submitted: bool = false,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    reader: safetensors.TensorReader,
    dispatch_spans: DispatchSpans,
    targets: []Target,
    total: usize,
    completed_read_bytes: std.atomic.Value(usize) = .init(0),
    progress: ?std.Progress.Node = null,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        store: *const TensorStore,
        tensor: *const Tensor,
        source_file: std.Io.File,
        shardings: []const Sharding,
        output: *Buffer,
        progress_parent: ?*std.Progress.Node,
    ) !VectoredTensorTransfer {
        var reader = try store.getBorrowedPositionalReaderById(tensor.id, io, source_file);
        errdefer reader.deinit();

        const shape = tensor.shape();
        const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse blk: {
            log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ reader.tensor.name, shape });
            break :blk platform.replicated_sharding;
        };
        const dispatch_spans = try DispatchSpans.init(allocator, shape, sharding);
        errdefer dispatch_spans.deinit(allocator);

        const placement = try sharding.placement(shape);
        const ordered_devices = sharding.devicesInCanonicalOrder();
        const targets = try allocator.alloc(Target, ordered_devices.len);
        errdefer allocator.free(targets);

        var pjrt_buffers: Buffer.Shards = .empty;
        var initialized: usize = 0;
        errdefer {
            for (targets[0..initialized]) |target| {
                target.manager.deinit(platform.pjrt_api);
                target.pjrt_buffer.deinit(platform.pjrt_api);
            }
        }

        const shape_spec: pjrt.ShapeSpec = .init(placement.shape.dims(), pjrtx.bufferTypeFromDtype(placement.shape.dtype()));
        for (ordered_devices, 0..) |device, i| {
            const memory = platform.devices[device.id].memory(.default).?;
            const manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(platform.pjrt_api);
            const pjrt_buffer = try manager.retrieveBuffer(platform.pjrt_api, 0);
            targets[i] = .{
                .manager = manager,
                .pjrt_buffer = pjrt_buffer,
                .device_index = device.id,
                .total = placement.shape.byteSize(),
            };
            initialized += 1;
            pjrt_buffers.appendAssumeCapacity(pjrt_buffer);
        }

        output.* = .fromPjrtBuffers(platform, shape, sharding, pjrt_buffers.constSlice());
        const progress = if (progress_parent) |parent|
            parent.start(reader.tensor.name, std.math.divCeil(usize, shape.byteSize(), 1024) catch unreachable)
        else
            null;

        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .reader = reader,
            .dispatch_spans = dispatch_spans,
            .targets = targets,
            .total = shape.byteSize(),
            .progress = progress,
        };
    }

    fn deinit(self: *VectoredTensorTransfer) void {
        if (self.progress) |*progress| progress.end();
        for (self.targets) |target| target.manager.deinit(self.platform.pjrt_api);
        self.allocator.free(self.targets);
        self.dispatch_spans.deinit(self.allocator);
        self.reader.deinit();
    }

    fn recordReadProgress(self: *VectoredTensorTransfer, bytes: usize) void {
        const completed = self.completed_read_bytes.fetchAdd(bytes, .acq_rel) + bytes;
        if (self.progress) |*progress| {
            progress.setCompletedItems(std.math.divCeil(usize, completed, 1024) catch unreachable);
        }
    }
};

const RequestGate = struct {
    limit: usize,
    in_use: usize = 0,
    peak: usize = 0,
    closed: std.atomic.Value(bool) = .init(false),
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    fn acquire(self: *RequestGate, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.closed.load(.acquire) and self.in_use >= self.limit) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        if (self.closed.load(.acquire)) return false;
        self.in_use += 1;
        self.peak = @max(self.peak, self.in_use);
        return true;
    }

    fn release(self: *RequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.in_use > 0);
        self.in_use -= 1;
        self.condition.signal(io);
    }

    fn close(self: *RequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed.store(true, .release);
        self.condition.broadcast(io);
    }

    fn peakUse(self: *RequestGate, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.peak;
    }
};

const VectoredLoadPipeline = struct {
    const RequestContext = struct {
        pipeline: *VectoredLoadPipeline,
        pending: std.atomic.Value(usize) = .init(1),

        fn addBlock(self: *RequestContext) void {
            _ = self.pending.fetchAdd(1, .acq_rel);
        }

        fn finishScheduling(self: *RequestContext) void {
            self.completeOne();
        }

        fn completeBlock(self: *RequestContext) void {
            self.completeOne();
        }

        fn completeOne(self: *RequestContext) void {
            const previous = self.pending.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous != 1) return;
            self.pipeline.lifecycle_gate.release(self.pipeline.io);
            self.pipeline.allocator.destroy(self);
        }
    };

    const BlockContext = struct {
        request: *RequestContext,
        lease: mem.DmaBlockPool.Lease,
        completion_reported: std.atomic.Value(bool) = .init(false),

        fn complete(self: *BlockContext) void {
            self.lease.complete();
            if (self.lease.isComplete() and
                self.completion_reported.cmpxchgStrong(false, true, .acq_rel, .acquire) == null)
            {
                self.request.completeBlock();
            }
        }
    };

    const ReadyTransfer = struct {
        target: *VectoredTensorTransfer.Target,
        block: *BlockContext,
        destination_offset: usize,
        len: usize,
    };

    const EventContext = struct {
        pipeline: *VectoredLoadPipeline,
        block: *BlockContext,
        pjrt_event: *pjrt.Event,
        err: ?*pjrt.Error = null,
        submitted_at: std.Io.Timestamp,
        device_index: usize,
        bytes: usize,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    pool: *mem.DmaBlockPool,
    lifecycle_gate: *RequestGate,
    block_size: usize,
    dma_limit: usize,
    metrics: *VectoredLoadMetrics,
    first_error: std.atomic.Value(u16) = .init(0),
    metadata_mutex: std.Io.Mutex = .init,
    blocks: std.ArrayListUnmanaged(*BlockContext) = .empty,
    ready_queues: []std.ArrayListUnmanaged(ReadyTransfer),
    events: std.ArrayListUnmanaged(*EventContext) = .empty,
    active_by_device: []usize,
    peak_by_device: []usize,
    next_device: usize = 0,
    pumping: bool = false,
    active_events: usize = 0,
    ready_entries: usize = 0,
    reads_finished: bool = false,
    dma_done: std.Io.Event = .unset,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pool: *mem.DmaBlockPool,
        lifecycle_gate: *RequestGate,
        block_size: usize,
        dma_limit: usize,
        metrics: *VectoredLoadMetrics,
    ) !VectoredLoadPipeline {
        const ready_queues = try allocator.alloc(std.ArrayListUnmanaged(ReadyTransfer), platform.devices.len);
        errdefer allocator.free(ready_queues);
        @memset(ready_queues, .empty);
        const active_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(active_by_device);
        @memset(active_by_device, 0);
        const peak_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(peak_by_device);
        @memset(peak_by_device, 0);
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .pool = pool,
            .lifecycle_gate = lifecycle_gate,
            .block_size = block_size,
            .dma_limit = dma_limit,
            .metrics = metrics,
            .ready_queues = ready_queues,
            .active_by_device = active_by_device,
            .peak_by_device = peak_by_device,
        };
    }

    fn deinit(self: *VectoredLoadPipeline) void {
        std.debug.assert(self.active_events == 0);
        std.debug.assert(self.ready_entries == 0);
        for (self.events.items) |ctx| {
            ctx.pjrt_event.deinit(self.platform.pjrt_api);
            if (ctx.err) |err| err.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
        }
        for (self.blocks.items) |block| {
            std.debug.assert(block.lease.isComplete());
            std.debug.assert(block.completion_reported.load(.acquire));
            self.allocator.destroy(block);
        }
        for (self.ready_queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.ready_queues);
        self.allocator.free(self.active_by_device);
        self.allocator.free(self.peak_by_device);
        self.events.deinit(self.allocator);
        self.blocks.deinit(self.allocator);
    }

    fn failed(self: *const VectoredLoadPipeline) bool {
        return self.first_error.load(.acquire) != 0;
    }

    fn errorValue(self: *const VectoredLoadPipeline) ?anyerror {
        const value = self.first_error.load(.acquire);
        return if (value == 0) null else @errorFromInt(value);
    }

    fn recordError(self: *VectoredLoadPipeline, err: anyerror) void {
        if (self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic) == null) {
            self.pool.close(self.io);
            self.lifecycle_gate.close(self.io);
        }
    }

    fn registerRequest(self: *VectoredLoadPipeline) !*RequestContext {
        const request = try self.allocator.create(RequestContext);
        request.* = .{ .pipeline = self };
        return request;
    }

    fn registerBlock(
        self: *VectoredLoadPipeline,
        request: *RequestContext,
        data: []u8,
        references: usize,
    ) !*BlockContext {
        const block = try self.allocator.create(BlockContext);
        errdefer self.allocator.destroy(block);
        block.* = .{
            .request = request,
            .lease = .init(self.pool, self.io, data, references),
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.blocks.append(self.allocator, block);
        request.addBlock();
        return block;
    }

    fn transferReady(self: *const VectoredLoadPipeline, transfer: ReadyTransfer) bool {
        _ = self;
        return transfer.destination_offset + transfer.len != transfer.target.total or
            transfer.target.submitted_bytes.load(.acquire) == transfer.destination_offset;
    }

    fn enqueueBlock(
        self: *VectoredLoadPipeline,
        tensor: *VectoredTensorTransfer,
        block: *BlockContext,
        writer_mask: u64,
        destination_offset: usize,
        len: usize,
    ) !void {
        var reservations: [Platform.MAX_NUM_DEVICES]usize = @splat(0);
        var mask = writer_mask;
        while (mask != 0) {
            const writer_index: usize = @intCast(@ctz(mask));
            mask &= mask - 1;
            reservations[tensor.targets[writer_index].device_index] += 1;
        }

        self.metadata_mutex.lockUncancelable(self.io);
        errdefer self.metadata_mutex.unlock(self.io);
        for (self.ready_queues, reservations[0..self.ready_queues.len]) |*queue, count| {
            try queue.ensureUnusedCapacity(self.allocator, count);
        }
        mask = writer_mask;
        while (mask != 0) {
            const writer_index: usize = @intCast(@ctz(mask));
            mask &= mask - 1;
            const target = &tensor.targets[writer_index];
            self.ready_queues[target.device_index].appendAssumeCapacity(.{
                .target = target,
                .block = block,
                .destination_offset = destination_offset,
                .len = len,
            });
            self.ready_entries += 1;
        }
        self.metadata_mutex.unlock(self.io);
        self.requestPump();
    }

    fn requestPump(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        if (self.pumping or self.failed()) {
            self.metadata_mutex.unlock(self.io);
            return;
        }
        self.pumping = true;
        self.metadata_mutex.unlock(self.io);
        self.pump();
    }

    fn pump(self: *VectoredLoadPipeline) void {
        while (true) {
            var selected: ?ReadyTransfer = null;
            var stalled = false;
            self.metadata_mutex.lockUncancelable(self.io);
            for (0..self.ready_queues.len) |offset| {
                const device_index = (self.next_device + offset) % self.ready_queues.len;
                if (self.active_by_device[device_index] >= self.dma_limit) continue;
                const queue = &self.ready_queues[device_index];
                for (queue.items, 0..) |transfer, i| {
                    if (!self.transferReady(transfer)) continue;
                    selected = queue.swapRemove(i);
                    self.next_device = (device_index + 1) % self.ready_queues.len;
                    self.active_by_device[device_index] += 1;
                    self.peak_by_device[device_index] = @max(
                        self.peak_by_device[device_index],
                        self.active_by_device[device_index],
                    );
                    self.active_events += 1;
                    self.ready_entries -= 1;
                    break;
                }
                if (selected != null) break;
            }
            if (selected == null) {
                self.pumping = false;
                stalled = self.reads_finished and self.active_events == 0 and self.ready_entries != 0 and !self.failed();
                self.maybeDoneLocked();
                self.metadata_mutex.unlock(self.io);
                if (stalled) {
                    self.recordError(error.IncompleteTransferPlan);
                    self.abortReady();
                }
                return;
            }
            self.metadata_mutex.unlock(self.io);
            self.submitOne(selected.?);
        }
    }

    fn submitOne(self: *VectoredLoadPipeline, transfer: ReadyTransfer) void {
        const device_index = transfer.target.device_index;
        const is_last = transfer.destination_offset + transfer.len == transfer.target.total;
        const submitted_at: std.Io.Timestamp = .now(self.io, .awake);
        const event = transfer.target.manager.transferData(
            self.platform.pjrt_api,
            0,
            transfer.block.lease.data[0..transfer.len],
            @intCast(transfer.destination_offset),
            is_last,
        ) catch |err| {
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(device_index);
            return;
        };
        if (is_last) transfer.target.final_submitted = true;
        _ = transfer.target.submitted_bytes.fetchAdd(transfer.len, .release);

        const ctx = self.allocator.create(EventContext) catch {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.recordError(error.OutOfMemory);
            transfer.block.complete();
            self.eventCompleted(device_index);
            return;
        };
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .submitted_at = submitted_at,
            .device_index = device_index,
            .bytes = transfer.len,
        };

        self.metadata_mutex.lockUncancelable(self.io);
        self.events.append(self.allocator, ctx) catch {
            self.metadata_mutex.unlock(self.io);
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
            self.recordError(error.OutOfMemory);
            transfer.block.complete();
            self.eventCompleted(device_index);
            return;
        };
        self.metadata_mutex.unlock(self.io);

        _ = self.metrics.dma_submissions.fetchAdd(1, .monotonic);
        _ = self.metrics.submitted_bytes.fetchAdd(transfer.len, .monotonic);
        event.onReady(self.platform.pjrt_api, EventContext, struct {
            fn call(err: ?*pjrt.Error, ctx_: *EventContext) void {
                ctx_.err = err;
                if (err) |pjrt_error| {
                    ctx_.pipeline.recordError(pjrt_error.getCode(ctx_.pipeline.platform.pjrt_api).toApiError());
                } else {
                    const elapsed = ctx_.submitted_at.untilNow(ctx_.pipeline.io, .awake);
                    _ = ctx_.pipeline.metrics.committed_bytes.fetchAdd(ctx_.bytes, .monotonic);
                    _ = ctx_.pipeline.metrics.dma_ns.fetchAdd(@intCast(@max(elapsed.nanoseconds, 0)), .monotonic);
                }
                ctx_.block.complete();
                ctx_.pipeline.eventCompleted(ctx_.device_index);
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(device_index);
        };
    }

    fn eventCompleted(self: *VectoredLoadPipeline, device_index: usize) void {
        self.metadata_mutex.lockUncancelable(self.io);
        std.debug.assert(self.active_events > 0);
        std.debug.assert(self.active_by_device[device_index] > 0);
        self.active_events -= 1;
        self.active_by_device[device_index] -= 1;
        self.maybeDoneLocked();
        self.metadata_mutex.unlock(self.io);
        if (self.failed()) {
            self.abortReady();
        } else {
            self.requestPump();
        }
    }

    fn abortReady(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        for (self.ready_queues) |*queue| {
            for (queue.items) |transfer| {
                transfer.block.complete();
                self.ready_entries -= 1;
            }
            queue.clearRetainingCapacity();
        }
        self.maybeDoneLocked();
        self.metadata_mutex.unlock(self.io);
    }

    fn finishReads(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        self.reads_finished = true;
        self.maybeDoneLocked();
        self.metadata_mutex.unlock(self.io);
        self.requestPump();
    }

    fn peakDma(self: *VectoredLoadPipeline) usize {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        var peak: usize = 0;
        for (self.peak_by_device) |value| peak = @max(peak, value);
        return peak;
    }

    fn maybeDoneLocked(self: *VectoredLoadPipeline) void {
        if (self.reads_finished and self.ready_entries == 0 and self.active_events == 0) {
            self.dma_done.set(self.io);
        }
    }
};

const VectoredReadRequest = struct {
    fn run(
        request: *VectoredLoadPipeline.RequestContext,
        tensor: *VectoredTensorTransfer,
        pipeline: *VectoredLoadPipeline,
        source_offset: usize,
        request_len: usize,
    ) void {
        defer request.finishScheduling();
        if (pipeline.failed()) return;

        const plan = VectoredRequestPlan.init(
            pipeline.allocator,
            tensor.dispatch_spans,
            source_offset,
            request_len,
            pipeline.block_size,
        ) catch |err| {
            pipeline.recordError(err);
            return;
        };
        defer plan.deinit(pipeline.allocator);
        if (plan.blocks.len == 0) return;

        const leased = pipeline.allocator.alloc([]u8, plan.blocks.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(leased);
        @memset(leased, &.{});

        const wait_ns = pipeline.pool.acquireMany(pipeline.io, leased) catch |err| {
            pipeline.recordError(err);
            return;
        };
        if (wait_ns > 0) _ = pipeline.metrics.pool_waits.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.pool_wait_ns.fetchAdd(wait_ns, .monotonic);
        defer for (leased) |block| {
            if (block.len != 0) pipeline.pool.release(pipeline.io, block);
        };
        if (pipeline.failed()) return;

        const iovecs = pipeline.allocator.alloc([]u8, plan.segments.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(iovecs);
        for (plan.segments, iovecs) |segment, *iovec| {
            iovec.* = leased[segment.block_index][segment.block_offset..][0..segment.len];
        }

        pipeline.metrics.beginRead();
        const read_started: std.Io.Timestamp = .now(pipeline.io, .awake);
        tensor.reader.readPositionalAllV(iovecs, source_offset) catch |err| {
            _ = pipeline.metrics.active_reads.fetchSub(1, .acq_rel);
            pipeline.recordError(err);
            return;
        };
        const read_elapsed = read_started.untilNow(pipeline.io, .awake);
        _ = pipeline.metrics.active_reads.fetchSub(1, .acq_rel);
        _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.read_bytes.fetchAdd(request_len, .monotonic);
        _ = pipeline.metrics.read_ns.fetchAdd(@intCast(@max(read_elapsed.nanoseconds, 0)), .monotonic);
        tensor.recordReadProgress(request_len);

        if (pipeline.failed()) return;
        for (plan.blocks, 0..) |block_plan, i| {
            const references: usize = @popCount(block_plan.writer_mask);
            const block = pipeline.registerBlock(request, leased[i], references) catch {
                pipeline.recordError(error.OutOfMemory);
                return;
            };
            leased[i] = &.{};
            pipeline.enqueueBlock(
                tensor,
                block,
                block_plan.writer_mask,
                block_plan.destination_offset,
                block_plan.len,
            ) catch |err| {
                pipeline.recordError(err);
                var remaining = references;
                while (remaining > 0) : (remaining -= 1) block.complete();
                return;
            };
        }
    }
};

fn loadVectored(
    comptime ModelType: type,
    model: *const ModelType,
    bufferized: *Bufferized(ModelType),
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: VectoredLoadOpts,
    load_started: std.Io.Timestamp,
) !usize {
    const tensor_count = meta.count(Tensor, model);
    const tensors = try allocator.alloc(*const Tensor, tensor_count);
    defer allocator.free(tensors);
    const buffers = try allocator.alloc(*Buffer, tensor_count);
    defer allocator.free(buffers);
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, output: []*const Tensor) void {
            output[i] = tensor;
        }
    }.call, .{tensors});
    meta.forEachVisit(bufferized, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
            output[i] = buffer;
        }
    }.call, .{buffers});

    var pool = try mem.DmaBlockPool.init(allocator, platform, opts.dma_block_size, opts.max_pinned_bytes);
    defer pool.deinit();

    const SourceSlot = struct {
        const uninitialized = 0;
        const initializing = 1;
        const ready = 2;
        const failed = 3;

        uri: []const u8,
        file: std.Io.File = undefined,
        status: std.atomic.Value(u8) = .init(uninitialized),
        error_code: std.atomic.Value(u16) = .init(0),
        initialized: std.Io.Event = .unset,

        fn ensure(self: *@This(), io_: std.Io) !std.Io.File {
            while (true) switch (self.status.load(.acquire)) {
                uninitialized => {
                    if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                    self.file = std.Io.Dir.openFile(.cwd(), io_, self.uri, .{ .mode = .read_only }) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(failed, .release);
                        self.initialized.set(io_);
                        return err;
                    };
                    self.status.store(ready, .release);
                    self.initialized.set(io_);
                    return self.file;
                },
                initializing => self.initialized.waitUncancelable(io_),
                ready => return self.file,
                failed => return @errorFromInt(self.error_code.load(.acquire)),
                else => unreachable,
            };
        }
    };

    var source_slots: std.ArrayListUnmanaged(SourceSlot) = .empty;
    defer {
        for (source_slots.items) |*slot| {
            if (slot.status.load(.acquire) == SourceSlot.ready) slot.file.close(io);
        }
        source_slots.deinit(allocator);
    }
    const tensor_source_indices = try allocator.alloc(usize, tensor_count);
    defer allocator.free(tensor_source_indices);
    for (tensors, tensor_source_indices) |tensor, *source_index| {
        const descriptor = store.getPtrFromId(tensor.id) orelse return error.NotFound;
        source_index.* = for (source_slots.items, 0..) |slot, index| {
            if (std.mem.eql(u8, slot.uri, descriptor.file_uri)) break index;
        } else blk: {
            const index = source_slots.items.len;
            try source_slots.append(allocator, .{ .uri = descriptor.file_uri });
            break :blk index;
        };
    }

    const StateSlot = struct {
        const uninitialized = 0;
        const initializing = 1;
        const ready = 2;
        const failed = 3;

        state: VectoredTensorTransfer = undefined,
        status: std.atomic.Value(u8) = .init(uninitialized),
        error_code: std.atomic.Value(u16) = .init(0),
        initialized: std.Io.Event = .unset,

        fn ensure(
            self: *@This(),
            allocator_: std.mem.Allocator,
            io_: std.Io,
            platform_: *const Platform,
            store_: *const TensorStore,
            tensor_: *const Tensor,
            source_file_: std.Io.File,
            shardings_: []const Sharding,
            buffer_: *Buffer,
            progress_: ?*std.Progress.Node,
        ) !*VectoredTensorTransfer {
            while (true) switch (self.status.load(.acquire)) {
                uninitialized => {
                    if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                    self.state = VectoredTensorTransfer.init(
                        allocator_,
                        io_,
                        platform_,
                        store_,
                        tensor_,
                        source_file_,
                        shardings_,
                        buffer_,
                        progress_,
                    ) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(failed, .release);
                        self.initialized.set(io_);
                        return err;
                    };
                    self.status.store(ready, .release);
                    self.initialized.set(io_);
                    return &self.state;
                },
                initializing => self.initialized.waitUncancelable(io_),
                ready => return &self.state,
                failed => return @errorFromInt(self.error_code.load(.acquire)),
                else => unreachable,
            };
        }
    };

    const state_slots = try allocator.alloc(StateSlot, tensor_count);
    defer allocator.free(state_slots);
    for (state_slots) |*slot| slot.* = .{};
    defer for (state_slots) |*slot| {
        if (slot.status.load(.acquire) == StateSlot.ready) slot.state.deinit();
    };

    const coordinator_started_at: std.Io.Timestamp = .now(io, .awake);
    load_log.debug("vectored coordinator started: tensors={d}, elapsed={d:.3}s", .{
        tensor_count,
        @as(f64, @floatFromInt(load_started.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s,
    });

    var lifecycle_gate: RequestGate = .{
        .limit = opts.read_parallelism +| 8,
    };
    var metrics: VectoredLoadMetrics = .{};
    var pipeline = try VectoredLoadPipeline.init(
        allocator,
        io,
        platform,
        &pool,
        &lifecycle_gate,
        opts.dma_block_size,
        opts.dma_parallelism,
        &metrics,
    );
    defer pipeline.deinit();

    const ReadJob = struct {
        tensor_index: usize,
        source_offset: usize,
        len: usize,
    };
    var request_count: usize = 0;
    for (tensors) |tensor| {
        const count = std.math.divCeil(usize, tensor.byteSize(), opts.read_request_size) catch unreachable;
        request_count += count;
    }
    const jobs = try allocator.alloc(ReadJob, request_count);
    defer allocator.free(jobs);
    const offsets = try allocator.alloc(usize, tensor_count);
    defer allocator.free(offsets);
    @memset(offsets, 0);

    var job_count: usize = 0;
    var scheduled = true;
    while (scheduled) {
        scheduled = false;
        for (tensors, offsets, 0..) |tensor, *offset, tensor_index| {
            const tensor_size = tensor.byteSize();
            if (offset.* >= tensor_size) continue;
            scheduled = true;
            const request_len = @min(opts.read_request_size, tensor_size - offset.*);
            jobs[job_count] = .{
                .tensor_index = tensor_index,
                .source_offset = offset.*,
                .len = request_len,
            };
            job_count += 1;
            offset.* += request_len;
        }
    }
    std.debug.assert(job_count == request_count);

    var next_job: std.atomic.Value(usize) = .init(0);
    var read_group: std.Io.Group = .init;
    const worker_count = @min(opts.read_parallelism, request_count);
    for (0..worker_count) |_| {
        read_group.concurrent(io, struct {
            fn run(
                jobs_: []const ReadJob,
                next: *std.atomic.Value(usize),
                pipeline_: *VectoredLoadPipeline,
                slots_: []StateSlot,
                tensors_: []const *const Tensor,
                buffers_: []*Buffer,
                source_slots_: []SourceSlot,
                source_indices_: []const usize,
                allocator_: std.mem.Allocator,
                io_: std.Io,
                platform_: *const Platform,
                store_: *const TensorStore,
                shardings_: []const Sharding,
                progress_: ?*std.Progress.Node,
            ) void {
                while (true) {
                    if (pipeline_.failed()) return;
                    if (!pipeline_.lifecycle_gate.acquire(io_)) return;
                    const index = next.fetchAdd(1, .monotonic);
                    if (index >= jobs_.len) {
                        pipeline_.lifecycle_gate.release(io_);
                        return;
                    }
                    const job = jobs_[index];
                    const source_file = source_slots_[source_indices_[job.tensor_index]].ensure(io_) catch |err| {
                        pipeline_.lifecycle_gate.release(io_);
                        pipeline_.recordError(err);
                        return;
                    };
                    const tensor = slots_[job.tensor_index].ensure(
                        allocator_,
                        io_,
                        platform_,
                        store_,
                        tensors_[job.tensor_index],
                        source_file,
                        shardings_,
                        buffers_[job.tensor_index],
                        progress_,
                    ) catch |err| {
                        pipeline_.lifecycle_gate.release(io_);
                        pipeline_.recordError(err);
                        return;
                    };
                    const request = pipeline_.registerRequest() catch |err| {
                        pipeline_.lifecycle_gate.release(io_);
                        pipeline_.recordError(err);
                        return;
                    };
                    VectoredReadRequest.run(request, tensor, pipeline_, job.source_offset, job.len);
                }
            }
        }.run, .{
            jobs,
            &next_job,
            &pipeline,
            state_slots,
            tensors,
            buffers,
            source_slots.items,
            tensor_source_indices,
            allocator,
            io,
            platform,
            store,
            opts.shardings,
            opts.progress,
        }) catch |err| {
            pipeline.recordError(err);
            break;
        };
    }
    read_group.await(io) catch |err| pipeline.recordError(err);
    const reads_finished_at: std.Io.Timestamp = .now(io, .awake);
    load_log.debug("vectored reads submitted: elapsed={d:.3}s, read_phase={d:.3}s, committed={Bi:.2}", .{
        @as(f64, @floatFromInt(load_started.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s,
        @as(f64, @floatFromInt(coordinator_started_at.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s,
        metrics.committed_bytes.load(.acquire),
    });

    pipeline.finishReads();
    if (pipeline.failed()) {
        pipeline.abortReady();
        for (state_slots) |*slot| {
            if (slot.status.load(.acquire) != StateSlot.ready) continue;
            for (slot.state.targets) |*target| {
                if (!target.final_submitted) {
                    target.manager.setBufferErrorUnknown(platform.pjrt_api, 0, "vectored load failed") catch {};
                }
            }
        }
    }

    pipeline.dma_done.waitUncancelable(io);
    load_log.debug("vectored DMA drained: elapsed={d:.3}s, drain_phase={d:.3}s", .{
        @as(f64, @floatFromInt(load_started.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s,
        @as(f64, @floatFromInt(reads_finished_at.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s,
    });
    if (pipeline.errorValue()) |err| return err;

    var loaded_bytes: usize = 0;
    for (state_slots) |*slot| {
        std.debug.assert(slot.status.load(.acquire) == StateSlot.ready);
        for (slot.state.targets) |target| std.debug.assert(target.final_submitted);
        loaded_bytes += slot.state.total;
    }
    const elapsed = load_started.untilNow(io, .awake);
    const elapsed_seconds = @as(f64, @floatFromInt(elapsed.nanoseconds)) / std.time.ns_per_s;
    const goodput = if (elapsed_seconds > 0) @as(f64, @floatFromInt(loaded_bytes)) / elapsed_seconds else 0;
    const average_read = if (metrics.read_operations.load(.acquire) == 0) 0 else metrics.read_bytes.load(.acquire) / metrics.read_operations.load(.acquire);
    const average_dma = if (metrics.dma_submissions.load(.acquire) == 0) 0 else metrics.submitted_bytes.load(.acquire) / metrics.dma_submissions.load(.acquire);
    const read_operations = metrics.read_operations.load(.acquire);
    const dma_submissions = metrics.dma_submissions.load(.acquire);
    const average_read_ms = if (read_operations == 0) 0 else @as(f64, @floatFromInt(metrics.read_ns.load(.acquire))) / @as(f64, @floatFromInt(read_operations)) / std.time.ns_per_ms;
    const average_dma_ms = if (dma_submissions == 0) 0 else @as(f64, @floatFromInt(metrics.dma_ns.load(.acquire))) / @as(f64, @floatFromInt(dma_submissions)) / std.time.ns_per_ms;
    load_log.debug("completed: vectored=true, tensors={d}, logical_bytes={Bi:.2}, elapsed={d:.3}s, logical_goodput={d:.2}MiB/s, reads={d}, peak_reads={d}, peak_retained={d}, average_read={Bi:.2}, average_read_latency={d:.3}ms, dma_submissions={d}, peak_dma={d}, average_dma={Bi:.2}, average_dma_latency={d:.3}ms, submitted={Bi:.2}, committed={Bi:.2}, pinned_high_water={Bi:.2}, mapped={Bi:.2}, pool_waits={d}, pool_wait={d:.3}s", .{
        tensor_count,
        loaded_bytes,
        elapsed_seconds,
        goodput / (1024 * 1024),
        read_operations,
        metrics.peak_reads.load(.acquire),
        lifecycle_gate.peakUse(io),
        average_read,
        average_read_ms,
        dma_submissions,
        pipeline.peakDma(),
        average_dma,
        average_dma_ms,
        metrics.submitted_bytes.load(.acquire),
        metrics.committed_bytes.load(.acquire),
        pool.highWaterBytes(),
        pool.mappedBytes(),
        metrics.pool_waits.load(.acquire),
        @as(f64, @floatFromInt(metrics.pool_wait_ns.load(.acquire))) / std.time.ns_per_s,
    });
    return loaded_bytes;
}

pub const VectoredLoadOpts = struct {
    pub const auto: VectoredLoadOpts = .{};

    /// Hard maximum number of concurrent positional source requests.
    read_parallelism: usize = 12,
    /// Logical bytes gathered by one positional source request.
    read_request_size: usize = 2 * 1024 * 1024,
    /// Maximum number of physical DMA events in flight per device.
    dma_parallelism: usize = 8,
    /// Physical transfer and pool allocation unit.
    dma_block_size: usize = 2 * 1024 * 1024,
    /// Client-wide hard limit for registered host memory.
    max_pinned_bytes: usize = 128 * 1024 * 1024,
    shardings: []const Sharding = &.{},
    progress: ?*std.Progress.Node = null,
    total_bytes: ?*usize = null,
};

fn loadBuffered(
    comptime ModelType: type,
    model: *const ModelType,
    bufferized: *Bufferized(ModelType),
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: VectoredLoadOpts,
) !usize {
    const tensor_count = meta.count(Tensor, model);
    const Ctx = struct {
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        store: *const TensorStore,
        shardings: []const Sharding,
        progress: ?*std.Progress.Node,
        buffers: []*Buffer,
        group: stdx.Io.LimitedGroup,
        total: std.atomic.Value(usize) = .init(0),
        first_error: std.atomic.Value(u16) = .init(0),

        fn recordError(self: *@This(), err: anyerror) void {
            _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
        }
    };
    var ctx: Ctx = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .store = store,
        .shardings = opts.shardings,
        .progress = opts.progress,
        .buffers = try allocator.alloc(*Buffer, tensor_count),
        .group = .init(opts.read_parallelism),
    };
    defer allocator.free(ctx.buffers);
    meta.forEachVisit(bufferized, *Buffer, struct {
        fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
            output[i] = buffer;
        }
    }.call, .{ctx.buffers});

    meta.forEachVisit(model, *const Tensor, struct {
        fn call(i: usize, tensor: *const Tensor, context: *Ctx) void {
            context.group.concurrent(context.io, struct {
                fn run(i_: usize, tensor_: *const Tensor, context_: *Ctx) void {
                    if (context_.first_error.load(.acquire) != 0) return;
                    var reader = context_.store.getReaderById(tensor_.id, context_.io, &.{}) catch |err| {
                        context_.recordError(err);
                        return;
                    };
                    defer reader.deinit();
                    const shape = tensor_.shape();
                    const sharding = Sharding.pickSharding(context_.shardings, shape, .explicit_axis_binding) orelse context_.platform.replicated_sharding;
                    var writer = BufferedMemoryWriter.init(context_.allocator, context_.io, context_.platform, shape, sharding, context_.buffers[i_]) catch |err| {
                        context_.recordError(err);
                        return;
                    };
                    defer writer.deinit(context_.allocator);

                    const total = reader.interface.streamRemaining(&writer.interface) catch |err| {
                        context_.recordError(err);
                        return;
                    };
                    writer.interface.flush() catch |err| {
                        context_.recordError(err);
                        return;
                    };
                    _ = context_.total.fetchAdd(total, .monotonic);
                }
            }.run, .{ i, tensor, context }) catch |err| context.recordError(err);
        }
    }.call, .{&ctx});
    ctx.group.await(io) catch |err| ctx.recordError(err);
    const error_code = ctx.first_error.load(.acquire);
    if (error_code != 0) return @errorFromInt(error_code);
    return ctx.total.load(.acquire);
}

pub fn load(
    comptime ModelType: type,
    model: *const ModelType,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: VectoredLoadOpts,
) !Bufferized(ModelType) {
    stdx.debug.assert(opts.read_parallelism > 0, "zml.io.load read_parallelism must be greater than zero", .{});
    stdx.debug.assert(opts.read_request_size > 0, "zml.io.load read_request_size must be greater than zero", .{});
    stdx.debug.assert(opts.dma_parallelism > 0, "zml.io.load dma_parallelism must be greater than zero", .{});
    stdx.debug.assert(opts.dma_block_size > 0, "zml.io.load dma_block_size must be greater than zero", .{});
    stdx.debug.assert(opts.max_pinned_bytes >= opts.dma_block_size, "zml.io.load max_pinned_bytes must hold at least one DMA block", .{});

    const load_started: std.Io.Timestamp = .now(io, .awake);
    const tensor_count = meta.count(Tensor, model);
    var span = tracer.span("zml.io.load", .{ .tensor_count = tensor_count });
    defer span.end();

    var bufferized = try mem.bufferize(allocator, ModelType, model);
    errdefer meta.forEachVisit(&bufferized, *Buffer, struct {
        fn call(_: usize, buffer: *Buffer) void {
            buffer.deinit();
        }
    }.call, .{});

    var total_logical_bytes: u64 = 0;
    meta.forEachVisit(model, *const Tensor, struct {
        fn call(_: usize, tensor: *const Tensor, total: *u64) void {
            total.* += tensor.byteSize();
        }
    }.call, .{&total_logical_bytes});

    const direct = platform.target == .cuda or platform.target == .oneapi;
    load_log.debug("configured: target={s}, vectored={}, tensors={d}, read_parallelism={d}, read_request_size={Bi:.2}, dma_parallelism={d}, dma_block_size={Bi:.2}, max_pinned_bytes={Bi:.2}, logical_bytes={Bi:.2}", .{
        @tagName(platform.target),
        direct,
        tensor_count,
        opts.read_parallelism,
        opts.read_request_size,
        opts.dma_parallelism,
        opts.dma_block_size,
        opts.max_pinned_bytes,
        total_logical_bytes,
    });

    const loaded_bytes = if (direct)
        try loadVectored(ModelType, model, &bufferized, allocator, io, platform, store, opts, load_started)
    else
        try loadBuffered(ModelType, model, &bufferized, allocator, io, platform, store, opts);
    if (opts.total_bytes) |total_bytes| total_bytes.* = loaded_bytes;
    return bufferized;
}
fn buildMesh2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 4) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[0]),
            .device(devices[1]),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .device(devices[2]),
            .device(devices[3]),
        }),
    });

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

test "vectored final transfers wait for every prior destination submission" {
    var targets = [_]VectoredTensorTransfer.Target{
        .{ .manager = undefined, .pjrt_buffer = undefined, .device_index = 0, .total = 100 },
        .{ .manager = undefined, .pjrt_buffer = undefined, .device_index = 1, .total = 100 },
    };
    var block: VectoredLoadPipeline.BlockContext = undefined;
    var pipeline: VectoredLoadPipeline = undefined;
    var final: VectoredLoadPipeline.ReadyTransfer = .{
        .target = &targets[0],
        .block = &block,
        .destination_offset = 80,
        .len = 20,
    };

    try std.testing.expect(!pipeline.transferReady(final));
    final.target = &targets[1];
    try std.testing.expect(!pipeline.transferReady(final));
    targets[1].submitted_bytes.store(80, .release);
    try std.testing.expect(pipeline.transferReady(final));
    final.target = &targets[0];
    targets[0].submitted_bytes.store(60, .release);
    try std.testing.expect(!pipeline.transferReady(final));
    _ = targets[0].submitted_bytes.fetchAdd(20, .release);
    try std.testing.expect(pipeline.transferReady(final));

    const non_final: VectoredLoadPipeline.ReadyTransfer = .{
        .target = &targets[0],
        .block = &block,
        .destination_offset = 20,
        .len = 20,
    };
    targets[0].submitted_bytes.store(0, .release);
    try std.testing.expect(pipeline.transferReady(non_final));
}

fn buildMesh2x2x2(
    allocator: std.mem.Allocator,
    target: @import("platform.zig").Target,
    devices: []const @import("platform.zig").Device,
) !Sharding.PhysicalMesh {
    if (devices.len < 8) return error.NotEnoughDevices;
    const topology: Sharding.PhysicalMesh.Tree = .axis(.link_x, .{ .mesh = .torus }, &.{
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[0]),
                .device(devices[1]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[2]),
                .device(devices[3]),
            }),
        }),
        .axis(.link_y, .{ .mesh = .torus }, &.{
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[4]),
                .device(devices[5]),
            }),
            .axis(.link_z, .{ .mesh = .torus }, &.{
                .device(devices[6]),
                .device(devices[7]),
            }),
        }),
    });

    return Sharding.PhysicalMesh.fromTree(allocator, target, topology);
}

const VectoredRequestPlanTest = struct {
    const Scenario = struct {
        name: []const u8,
        device_count: u32,
        physical_mesh: CreateOptions.PhysicalMesh = .auto,
        shape: Shape,
        logical_mesh: Sharding.LogicalMesh,
        strategy: Sharding.Strategy,
        request_size: usize,
        block_size: usize,
    };

    fn run(scenario: Scenario) !void {
        const allocator = std.testing.allocator;
        const io = std.testing.io;
        var platform = Platform.auto(allocator, io, .{
            .physical_mesh = scenario.physical_mesh,
            .cpu = .{ .device_count = scenario.device_count },
        }) catch return error.SkipZigTest;
        defer platform.deinit(allocator, io);

        const sharding_data: Sharding.Data = try .init(
            scenario.name,
            &platform.physical_mesh,
            scenario.logical_mesh,
            scenario.strategy,
        );
        try expectLayout(allocator, scenario.shape, .{ .data = &sharding_data }, scenario.request_size, scenario.block_size);
    }

    fn expectLayout(
        allocator: std.mem.Allocator,
        shape: Shape,
        sharding: Sharding,
        request_size: usize,
        block_size: usize,
    ) !void {
        const dispatch_spans: DispatchSpans = try .init(allocator, shape, sharding);
        defer dispatch_spans.deinit(allocator);

        const writer_count = sharding.devicesInCanonicalOrder().len;
        const placement = try sharding.placement(shape);
        const writer_size = placement.shape.byteSize();
        const source = try allocator.alloc(u8, shape.byteSize());
        defer allocator.free(source);
        for (source, 0..) |*byte, i| byte.* = @truncate(i *% 131 +% 17);

        const expected = try allocator.alloc(u8, writer_count * writer_size);
        defer allocator.free(expected);
        @memset(expected, 0);
        for (dispatch_spans.spans) |span| {
            var mask = dispatch_spans.writerMask(span);
            while (mask != 0) {
                const writer_index: usize = @intCast(@ctz(mask));
                mask &= mask - 1;
                const len = span.end - span.start;
                @memcpy(expected[writer_index * writer_size + span.writer_offset ..][0..len], source[span.start..span.end]);
            }
        }

        const actual = try allocator.alloc(u8, expected.len);
        defer allocator.free(actual);
        @memset(actual, 0);

        const request_count = std.math.divCeil(usize, source.len, request_size) catch unreachable;
        var reverse_index = request_count;
        while (reverse_index > 0) {
            reverse_index -= 1;
            const source_offset = reverse_index * request_size;
            const request_len = @min(request_size, source.len - source_offset);
            const plan: VectoredRequestPlan = try .init(allocator, dispatch_spans, source_offset, request_len, block_size);
            defer plan.deinit(allocator);

            const block_storage = try allocator.alloc(u8, plan.blocks.len * block_size);
            defer allocator.free(block_storage);
            @memset(block_storage, 0);

            var source_cursor = source_offset;
            for (plan.segments) |segment| {
                try std.testing.expect(segment.block_index < plan.blocks.len);
                try std.testing.expect(segment.block_offset + segment.len <= block_size);
                const destination = block_storage[segment.block_index * block_size + segment.block_offset ..][0..segment.len];
                @memcpy(destination, source[source_cursor..][0..segment.len]);
                source_cursor += segment.len;
            }
            try std.testing.expectEqual(source_offset + request_len, source_cursor);

            for (plan.blocks, 0..) |block, block_index| {
                try std.testing.expect(block.len > 0 and block.len <= block_size);
                var mask = block.writer_mask;
                while (mask != 0) {
                    const writer_index: usize = @intCast(@ctz(mask));
                    mask &= mask - 1;
                    try std.testing.expect(block.destination_offset + block.len <= writer_size);
                    @memcpy(
                        actual[writer_index * writer_size + block.destination_offset ..][0..block.len],
                        block_storage[block_index * block_size ..][0..block.len],
                    );
                }
            }
        }
        try std.testing.expectEqualSlices(u8, expected, actual);
    }
};

test "vectored request planner validates empty and invalid ranges" {
    const spans = [_]DispatchSpans.DispatchSpan{.{
        .start = 0,
        .end = 16,
        .writer_offset = 0,
        .primary_writer = 0,
        .mirror_writer_start = 0,
        .mirror_writer_len = 0,
    }};
    const dispatch: DispatchSpans = .{ .spans = @constCast(&spans), .mirror_writers = &.{} };

    const empty: VectoredRequestPlan = try .init(std.testing.allocator, dispatch, 16, 0, 4);
    defer empty.deinit(std.testing.allocator);
    try std.testing.expectEqual(@as(usize, 0), empty.blocks.len);
    try std.testing.expectEqual(@as(usize, 0), empty.segments.len);
    try std.testing.expectError(error.OutOfBounds, VectoredRequestPlan.init(std.testing.allocator, dispatch, 17, 0, 4));
    try std.testing.expectError(error.OutOfBounds, VectoredRequestPlan.init(std.testing.allocator, dispatch, 15, 2, 4));
    try std.testing.expectError(error.OutOfBounds, VectoredRequestPlan.init(std.testing.allocator, dispatch, std.math.maxInt(usize), 2, 4));
    try std.testing.expectError(error.InvalidBlockSize, VectoredRequestPlan.init(std.testing.allocator, dispatch, 0, 1, 0));
}

test "vectored request planner handles replication and block/request boundaries" {
    try VectoredRequestPlanTest.run(.{
        .name = "replicated_boundaries",
        .device_count = 4,
        .shape = Shape.init(.{ .rows = 9, .cols = 257 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .cols = .replicated }),
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .request_size = 773,
        .block_size = 257,
    });
}

test "vectored request planner handles 1D mirrored and folded sharding" {
    try VectoredRequestPlanTest.run(.{
        .name = "mirrored_1d",
        .device_count = 4,
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .shape = Shape.init(.{ .rows = 7, .model = 1024 }, .f32)
            .withPartitioning(.{ .rows = .replicated, .model = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .model = .link_x }),
        .request_size = 2053,
        .block_size = 509,
    });
    try VectoredRequestPlanTest.run(.{
        .name = "folded_1d",
        .device_count = 4,
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .shape = Shape.init(.{ .model = 4096 }, .f32).withPartitioning(.{ .model = .model }),
        .logical_mesh = .mesh(.{ .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_y });
            break :blk strategy;
        },
        .request_size = 3001,
        .block_size = 997,
    });
}

test "vectored request planner handles 2D and 3D sharding" {
    try VectoredRequestPlanTest.run(.{
        .name = "batch_model_2d",
        .device_count = 4,
        .physical_mesh = .{ .custom = buildMesh2x2 },
        .shape = Shape.init(.{ .batch = 8, .model = 1024 }, .f32)
            .withPartitioning(.{ .batch = .batch, .model = .model }),
        .logical_mesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .strategy = .parseBindings(.{ .batch = .link_x, .model = .link_y }),
        .request_size = 4093,
        .block_size = 1021,
    });
    try VectoredRequestPlanTest.run(.{
        .name = "folded_model_3d",
        .device_count = 8,
        .physical_mesh = .{ .custom = buildMesh2x2x2 },
        .shape = Shape.init(.{ .batch = 16, .model = 4096 }, .f32)
            .withPartitioning(.{ .batch = .replicated, .model = .model }),
        .logical_mesh = .mesh(.{ .batch = .low_bandwidth, .model = .high_bandwidth }),
        .strategy = blk: {
            var strategy: Sharding.Strategy = .parseBindings(.{ .model = .link_x });
            strategy.addFold(.link_x, &.{ .link_x, .link_z });
            break :blk strategy;
        },
        .request_size = 8191,
        .block_size = 2039,
    });
}
