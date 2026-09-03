const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("vfs");

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const platform_mod = @import("platform.zig");
const CreateOptions = platform_mod.CreateOptions;
const Exe = @import("exe.zig").Exe;
const mem = @import("mem.zig");
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
const Memory = platform_mod.Memory;
const Platform = platform_mod.Platform;
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

    fn getBorrowedPositionalReaderById(
        self: *const TensorStore,
        id: Tensor.Id,
        io: std.Io,
        file: std.Io.File,
    ) !safetensors.TensorReader {
        const sources = self.id_to_sources.get(id) orelse return error.NotFound;
        if (sources.len != 1) return error.MultipleTensorSources;
        return .initBorrowedPositional(io, sources[0].*, file);
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

// Direct load writer state machine used by the compatibility `Loader` on
// CUDA and oneAPI. The model-wide vectored loader below has separate state and
// admission control; retaining this path preserves master's Loader behavior.
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
    source_calls: std.atomic.Value(u64) = .init(0),
    transfer_pieces: std.atomic.Value(u64) = .init(0),
    read_bytes: std.atomic.Value(u64) = .init(0),
    read_ns: std.atomic.Value(u64) = .init(0),
    weighted_read_latency_us: std.atomic.Value(u64) = .init(0),
    pool_waits: std.atomic.Value(u64) = .init(0),
    pool_wait_ns: std.atomic.Value(u64) = .init(0),
    dma_submissions: std.atomic.Value(u64) = .init(0),
    submitted_bytes: std.atomic.Value(u64) = .init(0),
    committed_bytes: std.atomic.Value(u64) = .init(0),
    dma_ns: std.atomic.Value(u64) = .init(0),
    weighted_dma_latency_us: std.atomic.Value(u64) = .init(0),
    ready_bytes: std.atomic.Value(u64) = .init(0),
    ready_blocks: std.atomic.Value(usize) = .init(0),
    weighted_ready_age_us: std.atomic.Value(u64) = .init(0),
    active_reads: std.atomic.Value(usize) = .init(0),
    peak_reads: std.atomic.Value(usize) = .init(0),
    outstanding_requests: std.atomic.Value(usize) = .init(0),
    pending_source_jobs: std.atomic.Value(usize) = .init(0),
    outstanding_request_bytes: std.atomic.Value(u64) = .init(0),
    request_high_water: std.atomic.Value(usize) = .init(0),
    post_read_bytes: std.atomic.Value(u64) = .init(0),
    retired_bytes: std.atomic.Value(u64) = .init(0),
    weighted_request_latency_us: std.atomic.Value(u64) = .init(0),
    config_epoch: std.atomic.Value(u64) = .init(0),
    probe_epoch: std.atomic.Value(u64) = .init(std.math.maxInt(u64)),
    probe_admission_start: u64 = std.math.maxInt(u64),
    probe_first_read_ns: std.atomic.Value(u64) = .init(0),
    probe_active_reads: std.atomic.Value(usize) = .init(0),
    probe_peak_reads: std.atomic.Value(usize) = .init(0),
    probe_full_read_operations: std.atomic.Value(u64) = .init(0),
    probe_read_bytes: std.atomic.Value(u64) = .init(0),
    probe_mutex: std.Io.Mutex = .init,

    const Snapshot = struct {
        probe_epoch: u64,
        active_reads: usize,
        probe_first_read_ns: u64,
        probe_active_reads: usize,
        probe_peak_reads: usize,
        probe_full_read_operations: u64,
        probe_read_bytes: u64,
    };

    fn snapshot(self: *VectoredLoadMetrics, io: std.Io) Snapshot {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        return .{
            .probe_epoch = self.probe_epoch.load(.acquire),
            .active_reads = self.active_reads.load(.acquire),
            .probe_first_read_ns = self.probe_first_read_ns.load(.acquire),
            .probe_active_reads = self.probe_active_reads.load(.acquire),
            .probe_peak_reads = self.probe_peak_reads.load(.acquire),
            .probe_full_read_operations = self.probe_full_read_operations.load(.acquire),
            .probe_read_bytes = self.probe_read_bytes.load(.acquire),
        };
    }

    fn beginRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64, admission_id: u64) void {
        const active = self.active_reads.fetchAdd(1, .acq_rel) + 1;
        var peak = self.peak_reads.load(.acquire);
        while (active > peak) {
            peak = self.peak_reads.cmpxchgWeak(peak, active, .release, .acquire) orelse break;
        }
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire) or
            admission_id < self.probe_admission_start) return;
        _ = self.probe_first_read_ns.cmpxchgStrong(
            0,
            @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1)),
            .release,
            .monotonic,
        );
        const probe_active = self.probe_active_reads.fetchAdd(1, .acq_rel) + 1;
        var probe_peak = self.probe_peak_reads.load(.acquire);
        while (probe_active > probe_peak) {
            probe_peak = self.probe_peak_reads.cmpxchgWeak(probe_peak, probe_active, .release, .acquire) orelse break;
        }
    }

    fn endRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64, admission_id: u64) void {
        _ = self.active_reads.fetchSub(1, .acq_rel);
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire) or
            admission_id < self.probe_admission_start) return;
        _ = self.probe_active_reads.fetchSub(1, .acq_rel);
    }

    fn recordProbeRead(
        self: *VectoredLoadMetrics,
        io: std.Io,
        epoch: u64,
        admission_id: u64,
        bytes: usize,
        full_request_size: usize,
    ) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire) or
            admission_id < self.probe_admission_start) return;
        _ = full_request_size;
        _ = self.probe_full_read_operations.fetchAdd(1, .monotonic);
        _ = self.probe_read_bytes.fetchAdd(@intCast(bytes), .monotonic);
    }

    fn resetReadPeak(self: *VectoredLoadMetrics) void {
        self.peak_reads.store(self.active_reads.load(.acquire), .release);
    }

    fn beginRequest(self: *VectoredLoadMetrics, bytes: usize) void {
        const active = self.outstanding_requests.fetchAdd(1, .acq_rel) + 1;
        _ = self.outstanding_request_bytes.fetchAdd(@intCast(bytes), .monotonic);
        var high_water = self.request_high_water.load(.acquire);
        while (active > high_water) {
            high_water = self.request_high_water.cmpxchgWeak(high_water, active, .release, .acquire) orelse break;
        }
    }

    fn endRequest(self: *VectoredLoadMetrics, bytes: usize) void {
        _ = self.outstanding_requests.fetchSub(1, .acq_rel);
        _ = self.outstanding_request_bytes.fetchSub(@intCast(bytes), .monotonic);
    }

    fn prepareProbe(
        self: *VectoredLoadMetrics,
        io: std.Io,
        epoch: u64,
        admission_start: u64,
    ) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        self.probe_epoch.store(std.math.maxInt(u64), .release);
        self.probe_first_read_ns.store(0, .release);
        self.probe_active_reads.store(0, .release);
        self.probe_peak_reads.store(0, .release);
        self.probe_full_read_operations.store(0, .release);
        self.probe_read_bytes.store(0, .release);
        self.probe_admission_start = admission_start;
        self.probe_epoch.store(epoch, .release);
        self.config_epoch.store(epoch, .release);
    }

    fn clearProbe(self: *VectoredLoadMetrics, io: std.Io, epoch: u64) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (self.probe_epoch.load(.acquire) != epoch) return;
        self.probe_epoch.store(std.math.maxInt(u64), .release);
        self.probe_admission_start = std.math.maxInt(u64);
        self.probe_first_read_ns.store(0, .release);
        self.probe_active_reads.store(0, .release);
        self.probe_peak_reads.store(0, .release);
        self.probe_full_read_operations.store(0, .release);
        self.probe_read_bytes.store(0, .release);
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
        const source = store.getPtrFromId(tensor.id) orelse return error.NotFound;
        const shape = tensor.shape();
        const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse blk: {
            log.debug("No sharding strategy found for tensor {s} with shape {f}, using replicated sharding", .{ source.name, shape });
            break :blk platform.replicated_sharding;
        };
        return initResolved(
            allocator,
            io,
            platform,
            source,
            source_file,
            shape,
            sharding,
            output,
            progress_parent,
        );
    }

    fn initResolved(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        source: *const safetensors.Tensor,
        source_file: std.Io.File,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
        progress_parent: ?*std.Progress.Node,
    ) !VectoredTensorTransfer {
        var reader: safetensors.TensorReader = .initBorrowedPositional(io, source.*, source_file);
        errdefer reader.deinit();

        const packed_shape = shape.packedShape();
        const dispatch_spans = try DispatchSpans.init(allocator, packed_shape, sharding);
        errdefer dispatch_spans.deinit(allocator);

        const packed_placement = try sharding.placement(packed_shape);
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

        const shape_spec: pjrt.ShapeSpec = .init(
            packed_placement.shape.dims(),
            pjrtx.bufferTypeFromDtype(packed_placement.shape.dtype()),
        );
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
                .total = packed_placement.shape.byteSize(),
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
            .total = packed_shape.byteSize(),
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

const LoaderSourceSlot = struct {
    const uninitialized = 0;
    const initializing = 1;
    const ready = 2;
    const failed = 3;

    uri: []const u8,
    file: std.Io.File = undefined,
    status: std.atomic.Value(u8) = .init(uninitialized),
    error_code: std.atomic.Value(u16) = .init(0),
    initialized: std.Io.Event = .unset,

    fn ensure(self: *LoaderSourceSlot, io: std.Io) !std.Io.File {
        while (true) switch (self.status.load(.acquire)) {
            uninitialized => {
                if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                self.file = std.Io.Dir.openFile(.cwd(), io, self.uri, .{ .mode = .read_only }) catch |err| {
                    self.error_code.store(@intFromError(err), .release);
                    self.status.store(failed, .release);
                    self.initialized.set(io);
                    return err;
                };
                self.status.store(ready, .release);
                self.initialized.set(io);
                return self.file;
            },
            initializing => self.initialized.waitUncancelable(io),
            ready => return self.file,
            failed => return @errorFromInt(self.error_code.load(.acquire)),
            else => unreachable,
        };
    }

    fn deinit(self: *LoaderSourceSlot, io: std.Io) void {
        if (self.status.load(.acquire) == ready) self.file.close(io);
    }
};

const LoaderLoadItem = struct {
    const StateSlot = struct {
        const uninitialized = 0;
        const initializing = 1;
        const ready = 2;
        const failed = 3;

        state: VectoredTensorTransfer = undefined,
        status: std.atomic.Value(u8) = .init(uninitialized),
        error_code: std.atomic.Value(u16) = .init(0),
        initialized: std.Io.Event = .unset,

        fn ensure(self: *StateSlot, item: *LoaderLoadItem, direct: *DirectLoader) !*VectoredTensorTransfer {
            while (true) switch (self.status.load(.acquire)) {
                uninitialized => {
                    if (self.status.cmpxchgStrong(uninitialized, initializing, .acq_rel, .acquire) != null) continue;
                    const source_file = item.source_slot.ensure(direct.io) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(failed, .release);
                        self.initialized.set(direct.io);
                        return err;
                    };
                    self.state = VectoredTensorTransfer.initResolved(
                        direct.allocator,
                        direct.io,
                        direct.platform,
                        item.source,
                        source_file,
                        item.shape,
                        item.sharding,
                        item.output,
                        direct.opts.progress,
                    ) catch |err| {
                        self.error_code.store(@intFromError(err), .release);
                        self.status.store(failed, .release);
                        self.initialized.set(direct.io);
                        return err;
                    };
                    self.status.store(ready, .release);
                    self.initialized.set(direct.io);
                    return &self.state;
                },
                initializing => self.initialized.waitUncancelable(direct.io),
                ready => return &self.state,
                failed => return @errorFromInt(self.error_code.load(.acquire)),
                else => unreachable,
            };
        }

        fn deinit(self: *StateSlot) void {
            if (self.status.load(.acquire) == ready) self.state.deinit();
        }
    };

    source: *const safetensors.Tensor,
    source_slot: *LoaderSourceSlot,
    shape: Shape,
    sharding: Sharding,
    output: *Buffer,
    state: StateSlot = .{},

    fn deinit(self: *LoaderLoadItem, allocator: std.mem.Allocator) void {
        self.state.deinit();
        allocator.destroy(self);
    }
};

const AdaptiveRequestGate = struct {
    limit: std.atomic.Value(usize),
    in_use: usize = 0,
    closed: std.atomic.Value(bool) = .init(false),
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    fn init(limit: usize) AdaptiveRequestGate {
        return .{ .limit = .init(limit) };
    }

    fn acquire(self: *AdaptiveRequestGate, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.closed.load(.acquire) and self.in_use >= self.limit.load(.acquire)) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        if (self.closed.load(.acquire)) return false;
        self.in_use += 1;
        return true;
    }

    fn waitUntilEnabled(self: *AdaptiveRequestGate, io: std.Io, index: usize) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (!self.closed.load(.acquire) and index >= self.limit.load(.acquire)) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.closed.load(.acquire);
    }

    fn release(self: *AdaptiveRequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.in_use > 0);
        self.in_use -= 1;
        // One release creates one admission slot. Waking every worker here
        // turns a high adaptive cap into a thundering herd even when the
        // active limit is small.
        self.condition.signal(io);
    }

    fn setLimit(self: *AdaptiveRequestGate, io: std.Io, new_limit: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        _ = self.limit.swap(new_limit, .acq_rel);
        self.condition.broadcast(io);
    }

    fn inUse(self: *AdaptiveRequestGate, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.in_use;
    }

    fn close(self: *AdaptiveRequestGate, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.closed.store(true, .release);
        self.condition.broadcast(io);
    }
};

const RequestGateLimits = struct {
    read: usize,
    lifecycle: usize,

    fn init(read: usize, feasible_width: usize) RequestGateLimits {
        std.debug.assert(feasible_width > 0);
        const effective_read = @min(read, feasible_width);
        return .{
            .read = effective_read,
            .lifecycle = @min(feasible_width, effective_read +| 1),
        };
    }
};

fn selectLoaderDmaDevice(
    active: []const usize,
    per_device_limit: usize,
    ready_mask: u64,
    next_device: usize,
) ?usize {
    std.debug.assert(active.len > 0 and active.len <= 64);
    std.debug.assert(per_device_limit > 0 and next_device < active.len);
    for (0..active.len) |offset| {
        const device_index = (next_device + offset) % active.len;
        if (ready_mask & (@as(u64, 1) << @intCast(device_index)) == 0 or
            active[device_index] >= per_device_limit)
        {
            continue;
        }
        return device_index;
    }
    return null;
}

const VectoredLoadPipeline = struct {
    const RequestContext = struct {
        pipeline: *VectoredLoadPipeline,
        started_at: std.Io.Timestamp,
        read_finished_at_ns: std.atomic.Value(u64) = .init(0),
        pending: std.atomic.Value(usize) = .init(1), // scheduling sentinel
        completed: std.atomic.Value(bool) = .init(false),
        successful: std.atomic.Value(bool) = .init(false),
        source_finished: std.atomic.Value(bool) = .init(false),
        read_epoch: u64,
        admission_id: u64 = 0,
        len: usize,
        epoch_tracked: bool = false,

        fn addBlock(self: *RequestContext) void {
            _ = self.pending.fetchAdd(1, .acq_rel);
        }

        fn markReadFinished(self: *RequestContext) void {
            const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(self.pipeline.io, .awake).nanoseconds, 1));
            self.read_finished_at_ns.store(now_ns, .release);
            _ = self.pipeline.metrics.post_read_bytes.fetchAdd(@intCast(self.len), .monotonic);
            self.finishSourceJob();
        }

        fn markSuccessful(self: *RequestContext) void {
            self.successful.store(true, .release);
        }

        fn finishScheduling(self: *RequestContext) void {
            self.finishSourceJob();
            self.completeOne();
        }

        fn finishSourceJob(self: *RequestContext) void {
            if (!self.source_finished.swap(true, .acq_rel)) {
                const previous = self.pipeline.metrics.pending_source_jobs.fetchSub(1, .acq_rel);
                std.debug.assert(previous > 0);
            }
        }

        fn completeBlock(self: *RequestContext) void {
            self.completeOne();
        }

        fn completeOne(self: *RequestContext) void {
            const previous = self.pending.fetchSub(1, .acq_rel);
            std.debug.assert(previous > 0);
            if (previous != 1) return;

            if (self.read_finished_at_ns.load(.acquire) != 0) {
                _ = self.pipeline.metrics.post_read_bytes.fetchSub(@intCast(self.len), .monotonic);
            }
            if (self.successful.load(.acquire)) {
                const elapsed = self.started_at.untilNow(self.pipeline.io, .awake);
                const elapsed_us: u64 = @intCast(@max(elapsed.nanoseconds, 0) / std.time.ns_per_us);
                _ = self.pipeline.metrics.retired_bytes.fetchAdd(@intCast(self.len), .monotonic);
                _ = self.pipeline.metrics.weighted_request_latency_us.fetchAdd(
                    elapsed_us *| @as(u64, @intCast(self.len)),
                    .monotonic,
                );
            }
            self.pipeline.metrics.endRequest(self.len);
            self.completed.store(true, .release);
            self.pipeline.request_gate.release(self.pipeline.io);
            if (self.epoch_tracked) self.pipeline.completeEpochJob();
        }
    };

    const BlockContext = struct {
        pipeline: *VectoredLoadPipeline,
        request: *RequestContext,
        lease: mem.DmaBlockPool.Lease,
        ready_at: std.Io.Timestamp,
        pending_submissions: usize,
        len: usize,
        ready_reported: bool = false,
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
        tensor: *VectoredTensorTransfer,
        target: *VectoredTensorTransfer.Target,
        block: *BlockContext,
        source_offset: usize,
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
    worker_gate: *AdaptiveRequestGate,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    block_size: usize,
    source_request_size: usize,
    device_pool_indices: []const usize,
    numa_explicit: bool,
    metrics: *VectoredLoadMetrics,
    next_read_admission: std.atomic.Value(u64) = .init(1),
    first_error: std.atomic.Value(u16) = .init(0),
    metadata_mutex: std.Io.Mutex = .init,
    requests: std.ArrayListUnmanaged(*RequestContext) = .empty,
    blocks: std.ArrayListUnmanaged(*BlockContext) = .empty,
    ready_queues: []std.ArrayListUnmanaged(ReadyTransfer),
    events: std.ArrayListUnmanaged(*EventContext) = .empty,
    active_by_device: []usize,
    peak_by_device: []usize,
    dma_limit: usize,
    next_device: usize = 0,
    pumping: bool = false,
    active_events: usize = 0,
    ready_entries: usize = 0,
    reads_finished: bool = false,
    dma_done: std.Io.Event = .unset,
    epoch_jobs: std.atomic.Value(usize) = .init(0),
    epoch_drained: std.Io.Event = .is_set,
    track_epoch_jobs: bool = false,
    live_scheduler: ?*FairVectoredReadScheduler = null,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pool: *mem.DmaBlockPool,
        worker_gate: *AdaptiveRequestGate,
        read_gate: *AdaptiveRequestGate,
        request_gate: *AdaptiveRequestGate,
        block_size: usize,
        source_request_size: usize,
        device_pool_indices: []const usize,
        numa_explicit: bool,
        metrics: *VectoredLoadMetrics,
        dma_limit: usize,
    ) !VectoredLoadPipeline {
        std.debug.assert(platform.devices.len <= 64);
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
            .worker_gate = worker_gate,
            .read_gate = read_gate,
            .request_gate = request_gate,
            .block_size = block_size,
            .source_request_size = source_request_size,
            .device_pool_indices = device_pool_indices,
            .numa_explicit = numa_explicit,
            .metrics = metrics,
            .ready_queues = ready_queues,
            .active_by_device = active_by_device,
            .peak_by_device = peak_by_device,
            .dma_limit = dma_limit,
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
        for (self.requests.items) |request| {
            std.debug.assert(request.completed.load(.acquire));
            self.allocator.destroy(request);
        }
        for (self.ready_queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.ready_queues);
        self.allocator.free(self.active_by_device);
        self.allocator.free(self.peak_by_device);
        self.events.deinit(self.allocator);
        self.blocks.deinit(self.allocator);
        self.requests.deinit(self.allocator);
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
            if (self.live_scheduler) |scheduler| {
                const abandoned = scheduler.fail(self.io);
                self.cancelEpochJobs(abandoned);
            }
            self.pool.close(self.io);
            self.worker_gate.close(self.io);
            self.read_gate.close(self.io);
            self.request_gate.close(self.io);
            self.abortReady();
        }
    }

    fn registerRequest(self: *VectoredLoadPipeline, len: usize) !*RequestContext {
        const request = try self.allocator.create(RequestContext);
        errdefer self.allocator.destroy(request);
        request.* = .{
            .pipeline = self,
            .started_at = .now(self.io, .awake),
            .read_epoch = 0,
            .len = len,
            .epoch_tracked = self.track_epoch_jobs,
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.requests.append(self.allocator, request);
        self.metrics.beginRequest(len);
        return request;
    }

    fn beginEpochJobs(self: *VectoredLoadPipeline, count: usize) void {
        if (count == 0) return;
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        std.debug.assert(self.track_epoch_jobs);
        if (self.epoch_jobs.load(.acquire) == 0) self.epoch_drained.reset();
        _ = self.epoch_jobs.fetchAdd(count, .acq_rel);
    }

    fn cancelEpochJobs(self: *VectoredLoadPipeline, count: usize) void {
        if (count == 0) return;
        const previous = self.epoch_jobs.fetchSub(count, .acq_rel);
        std.debug.assert(previous >= count);
        if (previous == count) self.epoch_drained.set(self.io);
    }

    fn completeEpochJob(self: *VectoredLoadPipeline) void {
        const previous = self.epoch_jobs.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
        if (previous == 1) self.epoch_drained.set(self.io);
    }

    fn waitEpochDrained(self: *VectoredLoadPipeline) void {
        self.epoch_drained.waitUncancelable(self.io);
    }

    fn reapCompleted(self: *VectoredLoadPipeline) void {
        std.debug.assert(self.epoch_jobs.load(.acquire) == 0);
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        std.debug.assert(self.active_events == 0 and self.ready_entries == 0);
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
        for (self.requests.items) |request| {
            std.debug.assert(request.completed.load(.acquire));
            self.allocator.destroy(request);
        }
        self.events.clearRetainingCapacity();
        self.blocks.clearRetainingCapacity();
        self.requests.clearRetainingCapacity();
        @memset(self.peak_by_device, 0);
    }

    fn reserveSourceJob(self: *VectoredLoadPipeline) void {
        _ = self.metrics.pending_source_jobs.fetchAdd(1, .acq_rel);
    }

    fn abandonSourceJob(self: *VectoredLoadPipeline) void {
        const previous = self.metrics.pending_source_jobs.fetchSub(1, .acq_rel);
        std.debug.assert(previous > 0);
    }

    fn registerBlock(self: *VectoredLoadPipeline, request: *RequestContext, data: []u8, references: usize, len: usize) !*BlockContext {
        const block = try self.allocator.create(BlockContext);
        errdefer self.allocator.destroy(block);
        block.* = .{
            .pipeline = self,
            .request = request,
            .lease = .init(self.pool, self.io, data, references),
            .ready_at = .now(self.io, .awake),
            .pending_submissions = references,
            .len = len,
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.blocks.append(self.allocator, block);
        request.addBlock();
        return block;
    }

    fn transferReady(self: *const VectoredLoadPipeline, transfer: ReadyTransfer) bool {
        _ = self;
        if (transfer.destination_offset + transfer.len == transfer.target.total and
            transfer.target.submitted_bytes.load(.acquire) != transfer.destination_offset)
        {
            return false;
        }
        return true;
    }

    fn enqueueBlock(
        self: *VectoredLoadPipeline,
        tensor: *VectoredTensorTransfer,
        block: *BlockContext,
        writer_mask: u64,
        source_offset: usize,
        destination_offset: usize,
        len: usize,
    ) !void {
        self.metadata_mutex.lockUncancelable(self.io);
        errdefer self.metadata_mutex.unlock(self.io);
        var reserve_mask = writer_mask;
        while (reserve_mask != 0) {
            const writer_index: usize = @intCast(@ctz(reserve_mask));
            reserve_mask &= reserve_mask - 1;
            const target = &tensor.targets[writer_index];
            try self.ready_queues[target.device_index].ensureUnusedCapacity(self.allocator, 1);
        }
        if (!block.ready_reported) {
            _ = self.metrics.ready_bytes.fetchAdd(block.len, .monotonic);
            _ = self.metrics.ready_blocks.fetchAdd(1, .monotonic);
            block.ready_reported = true;
        }
        var mask = writer_mask;
        while (mask != 0) {
            const writer_index: usize = @intCast(@ctz(mask));
            mask &= mask - 1;
            const target = &tensor.targets[writer_index];
            const transfer: ReadyTransfer = .{
                .tensor = tensor,
                .target = target,
                .block = block,
                .source_offset = source_offset,
                .destination_offset = destination_offset,
                .len = len,
            };
            const queue = &self.ready_queues[target.device_index];
            queue.appendAssumeCapacity(transfer);
            self.ready_entries += 1;
        }
        _ = self.metrics.transfer_pieces.fetchAdd(1, .monotonic);
        self.metadata_mutex.unlock(self.io);
        self.requestPump();
    }

    fn reserveReadyCapacity(self: *VectoredLoadPipeline, counts: []const usize) !void {
        std.debug.assert(counts.len == self.ready_queues.len);
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        for (self.ready_queues, counts) |*queue, count| {
            try queue.ensureUnusedCapacity(self.allocator, count);
        }
    }

    fn reserveBlockCapacity(self: *VectoredLoadPipeline, count: usize) !void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.blocks.ensureUnusedCapacity(self.allocator, count);
    }

    fn abandonSubmissions(
        self: *VectoredLoadPipeline,
        block: *BlockContext,
        count: usize,
    ) void {
        if (count == 0) return;
        self.metadata_mutex.lockUncancelable(self.io);
        std.debug.assert(block.pending_submissions >= count);
        block.pending_submissions -= count;
        if (block.pending_submissions == 0 and block.ready_reported) {
            _ = self.metrics.ready_bytes.fetchSub(block.len, .monotonic);
            _ = self.metrics.ready_blocks.fetchSub(1, .monotonic);
        }
        self.metadata_mutex.unlock(self.io);
        for (0..count) |_| block.complete();
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
            self.metadata_mutex.lockUncancelable(self.io);
            if (!self.failed()) {
                const limit = self.dma_limit;
                var ready_mask: u64 = 0;
                for (self.ready_queues, 0..) |queue, device_index| {
                    if (self.active_by_device[device_index] >= limit) continue;
                    for (queue.items) |transfer| {
                        if (self.transferReady(transfer)) {
                            ready_mask |= @as(u64, 1) << @intCast(device_index);
                            break;
                        }
                    }
                }
                const device_index = selectLoaderDmaDevice(
                    self.active_by_device,
                    limit,
                    ready_mask,
                    self.next_device,
                );
                if (device_index) |index| {
                    const queue = &self.ready_queues[index];
                    for (queue.items, 0..) |transfer, i| {
                        if (!self.transferReady(transfer)) continue;
                        selected = queue.orderedRemove(i);
                        break;
                    }
                    std.debug.assert(selected != null);
                    self.next_device = (index + 1) % self.ready_queues.len;
                    self.active_by_device[index] += 1;
                    std.debug.assert(self.active_by_device[index] <= limit);
                    self.peak_by_device[index] = @max(
                        self.peak_by_device[index],
                        self.active_by_device[index],
                    );
                    self.active_events += 1;
                    self.ready_entries -= 1;
                    const transfer = selected.?;
                    std.debug.assert(transfer.block.pending_submissions > 0);
                    transfer.block.pending_submissions -= 1;
                    if (transfer.block.pending_submissions == 0) {
                        _ = self.metrics.ready_bytes.fetchSub(transfer.block.len, .monotonic);
                        _ = self.metrics.ready_blocks.fetchSub(1, .monotonic);
                        const ready_elapsed = transfer.block.ready_at.untilNow(self.io, .awake);
                        const age_us: u64 = @intCast(@max(ready_elapsed.nanoseconds, 0) / std.time.ns_per_us);
                        _ = self.metrics.weighted_ready_age_us.fetchAdd(
                            age_us *| @as(u64, @intCast(transfer.block.len)),
                            .monotonic,
                        );
                    }
                }
            }
            if (selected == null) {
                self.pumping = false;
                const done = self.doneLocked();
                self.metadata_mutex.unlock(self.io);
                if (done) self.dma_done.set(self.io);
                return;
            }
            self.metadata_mutex.unlock(self.io);
            self.submitOne(selected.?);
        }
    }

    fn submitOne(self: *VectoredLoadPipeline, transfer: ReadyTransfer) void {
        const is_last = transfer.destination_offset + transfer.len == transfer.target.total;
        const submitted_at: std.Io.Timestamp = .now(self.io, .awake);
        const event = transfer.target.manager.transferData(
            self.platform.pjrt_api,
            0,
            transfer.block.lease.data[transfer.source_offset..][0..transfer.len],
            @intCast(transfer.destination_offset),
            is_last,
        ) catch |err| {
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
            return;
        };
        if (is_last) transfer.target.final_submitted = true;
        _ = transfer.target.submitted_bytes.fetchAdd(transfer.len, .release);

        const ctx = self.allocator.create(EventContext) catch {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.recordError(error.OutOfMemory);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
            return;
        };
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .submitted_at = submitted_at,
            .device_index = transfer.target.device_index,
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
            self.eventCompleted(transfer.target.device_index);
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
                    const elapsed_ns: u64 = @intCast(@max(elapsed.nanoseconds, 0));
                    const elapsed_us: u64 = elapsed_ns / std.time.ns_per_us;
                    _ = ctx_.pipeline.metrics.committed_bytes.fetchAdd(ctx_.bytes, .monotonic);
                    _ = ctx_.pipeline.metrics.dma_ns.fetchAdd(elapsed_ns, .monotonic);
                    _ = ctx_.pipeline.metrics.weighted_dma_latency_us.fetchAdd(elapsed_us *| @as(u64, @intCast(ctx_.bytes)), .monotonic);
                }
                ctx_.block.complete();
                ctx_.pipeline.eventCompleted(ctx_.device_index);
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index);
        };
    }

    fn eventCompleted(self: *VectoredLoadPipeline, device_index: usize) void {
        self.metadata_mutex.lockUncancelable(self.io);
        std.debug.assert(self.active_events > 0);
        std.debug.assert(self.active_by_device[device_index] > 0);
        self.active_events -= 1;
        self.active_by_device[device_index] -= 1;
        self.metadata_mutex.unlock(self.io);
        // A ready callback can be the first place an asynchronous PJRT error
        // becomes visible. Once outside the metadata lock, retire every
        // queued transfer so dma_done cannot wait forever on entries that the
        // failed pump will no longer submit.
        if (self.failed())
            self.abortReady()
        else
            self.requestPump();
    }

    fn abortReady(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        for (self.ready_queues) |*queue| {
            for (queue.items) |transfer| {
                std.debug.assert(transfer.block.pending_submissions > 0);
                transfer.block.pending_submissions -= 1;
                transfer.block.complete();
                self.ready_entries -= 1;
                if (transfer.block.pending_submissions == 0) {
                    _ = self.metrics.ready_bytes.fetchSub(transfer.block.len, .monotonic);
                    _ = self.metrics.ready_blocks.fetchSub(1, .monotonic);
                }
            }
            queue.clearRetainingCapacity();
        }
        const done = self.doneLocked();
        self.metadata_mutex.unlock(self.io);
        if (done) self.dma_done.set(self.io);
    }

    fn finishReads(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        self.reads_finished = true;
        const done = self.doneLocked();
        self.metadata_mutex.unlock(self.io);
        if (done) {
            self.dma_done.set(self.io);
            return;
        }
        self.requestPump();
    }

    fn peakDeviceActive(self: *VectoredLoadPipeline) usize {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        var peak_device_active: usize = 0;
        for (self.peak_by_device) |peak| {
            peak_device_active = @max(peak_device_active, peak);
        }
        return peak_device_active;
    }

    fn doneLocked(self: *const VectoredLoadPipeline) bool {
        return self.reads_finished and self.ready_entries == 0 and self.active_events == 0;
    }
};

const VectoredReadRequest = struct {
    const TransferSlice = struct {
        tensor: *VectoredTensorTransfer,
        block_index: usize,
        block_offset: usize,
        writer_mask: u64,
        destination_offset: usize,
        len: usize,
    };

    fn readAbsoluteAllV(
        allocator: std.mem.Allocator,
        io: std.Io,
        file: std.Io.File,
        buffers: []const []u8,
        file_offset: u64,
        metrics: *VectoredLoadMetrics,
    ) !void {
        std.debug.assert(buffers.len <= max_load_positional_iovecs);
        var total: usize = 0;
        for (buffers) |buffer| total = try std.math.add(usize, total, buffer.len);
        var completed: usize = 0;
        var buffer_index: usize = 0;
        var buffer_offset: usize = 0;
        const scratch = try allocator.alloc([]u8, buffers.len);
        defer allocator.free(scratch);
        while (completed < total) {
            const current = buffers[buffer_index..];
            scratch[0] = current[0][buffer_offset..];
            @memcpy(scratch[1..current.len], current[1..]);
            _ = metrics.source_calls.fetchAdd(1, .monotonic);
            const bytes_read = try file.readPositional(
                io,
                scratch[0..current.len],
                file_offset + completed,
            );
            if (bytes_read == 0) return error.UnexpectedEndOfFile;
            completed += bytes_read;
            var advance = bytes_read;
            while (advance > 0) {
                const available = buffers[buffer_index].len - buffer_offset;
                const take = @min(advance, available);
                buffer_offset += take;
                advance -= take;
                if (buffer_offset == buffers[buffer_index].len) {
                    buffer_index += 1;
                    buffer_offset = 0;
                }
            }
        }
    }

    fn fillAffinities(
        tensor: *const VectoredTensorTransfer,
        pipeline: *const VectoredLoadPipeline,
        blocks: []const VectoredRequestPlan.Block,
        affinities: []mem.DmaBlockPool.Affinity,
    ) void {
        std.debug.assert(blocks.len == affinities.len);
        for (blocks, affinities) |block_plan, *affinity| {
            if (!pipeline.numa_explicit) {
                affinity.* = .{};
                continue;
            }
            var eligible_nodes: u64 = 0;
            var writer_mask = block_plan.writer_mask;
            while (writer_mask != 0) {
                const writer_index: usize = @intCast(@ctz(writer_mask));
                writer_mask &= writer_mask - 1;
                const device_index = tensor.targets[writer_index].device_index;
                const node_index = pipeline.device_pool_indices[device_index];
                eligible_nodes |= @as(u64, 1) << @intCast(node_index);
            }
            std.debug.assert(eligible_nodes != 0);
            affinity.* = if (@popCount(eligible_nodes) == 1)
                .node(@ctz(eligible_nodes))
            else
                .replicated(eligible_nodes);
        }
    }

    fn recordPoolWait(pipeline: *VectoredLoadPipeline, pool_wait_ns: u64) void {
        if (pool_wait_ns > 0) _ = pipeline.metrics.pool_waits.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.pool_wait_ns.fetchAdd(pool_wait_ns, .monotonic);
    }

    fn enqueue(
        request: *VectoredLoadPipeline.RequestContext,
        tensor: *VectoredTensorTransfer,
        pipeline: *VectoredLoadPipeline,
        block_plan: VectoredRequestPlan.Block,
        leased: *[]u8,
    ) bool {
        const references: usize = @popCount(block_plan.writer_mask);
        const block = pipeline.registerBlock(request, leased.*, references, block_plan.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return false;
        };
        leased.* = &.{};
        pipeline.enqueueBlock(
            tensor,
            block,
            block_plan.writer_mask,
            0,
            block_plan.destination_offset,
            block_plan.len,
        ) catch |err| {
            var remaining = references;
            while (remaining > 0) : (remaining -= 1) block.complete();
            pipeline.recordError(err);
            return false;
        };
        return true;
    }

    fn beginRead(
        request: *VectoredLoadPipeline.RequestContext,
        pipeline: *VectoredLoadPipeline,
    ) bool {
        if (!pipeline.read_gate.acquire(pipeline.io)) return false;
        // Generation and admission identity belong to the source-call permit,
        // not to earlier job claim or pinned-block waits.
        request.read_epoch = pipeline.metrics.config_epoch.load(.acquire);
        request.admission_id = pipeline.next_read_admission.fetchAdd(1, .monotonic);
        pipeline.metrics.beginRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
        );
        return true;
    }

    fn endRead(
        request: *VectoredLoadPipeline.RequestContext,
        pipeline: *VectoredLoadPipeline,
    ) void {
        pipeline.metrics.endRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
        );
        pipeline.read_gate.release(pipeline.io);
    }

    fn recordReadSuccess(
        request: *VectoredLoadPipeline.RequestContext,
        tensor: *VectoredTensorTransfer,
        pipeline: *VectoredLoadPipeline,
        request_len: usize,
        read_elapsed: std.Io.Duration,
    ) void {
        pipeline.metrics.recordProbeRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
            request_len,
            pipeline.source_request_size,
        );
        const read_elapsed_ns: u64 = @intCast(@max(read_elapsed.nanoseconds, 0));
        const read_elapsed_us: u64 = read_elapsed_ns / std.time.ns_per_us;
        _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.read_bytes.fetchAdd(request_len, .monotonic);
        _ = pipeline.metrics.read_ns.fetchAdd(read_elapsed_ns, .monotonic);
        _ = pipeline.metrics.weighted_read_latency_us.fetchAdd(
            read_elapsed_us *| @as(u64, @intCast(request_len)),
            .monotonic,
        );
        tensor.recordReadProgress(request_len);
        request.markReadFinished();
    }

    fn readWhole(
        request: *VectoredLoadPipeline.RequestContext,
        tensor: *VectoredTensorTransfer,
        pipeline: *VectoredLoadPipeline,
        plan: VectoredRequestPlan,
        affinities: []const mem.DmaBlockPool.Affinity,
        leased: [][]u8,
        source_offset: usize,
        request_len: usize,
    ) bool {
        const pool_wait_ns = pipeline.pool.acquireMany(pipeline.io, leased, affinities) catch |err| {
            pipeline.recordError(err);
            return false;
        };
        recordPoolWait(pipeline, pool_wait_ns);
        if (pipeline.failed()) return false;

        const iovecs = pipeline.allocator.alloc([]u8, plan.segments.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return false;
        };
        defer pipeline.allocator.free(iovecs);
        for (plan.segments, iovecs) |segment, *iovec| {
            iovec.* = leased[segment.block_index][segment.block_offset..][0..segment.len];
        }

        if (!beginRead(request, pipeline)) return false;
        const read_started: std.Io.Timestamp = .now(pipeline.io, .awake);
        const read_result = tensor.reader.readPositionalAllV(iovecs, source_offset);
        const read_elapsed = read_started.untilNow(pipeline.io, .awake);
        read_result catch |err| {
            endRead(request, pipeline);
            pipeline.recordError(err);
            return false;
        };
        recordReadSuccess(request, tensor, pipeline, request_len, read_elapsed);
        endRead(request, pipeline);

        if (pipeline.failed()) return false;
        for (plan.blocks, 0..) |block_plan, i| {
            if (!enqueue(request, tensor, pipeline, block_plan, &leased[i])) return false;
        }
        return true;
    }

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
        if (plan.blocks.len == 0) {
            request.markReadFinished();
            request.markSuccessful();
            return;
        }

        const leased = pipeline.allocator.alloc([]u8, plan.blocks.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(leased);
        @memset(leased, &.{});

        const affinities = pipeline.allocator.alloc(mem.DmaBlockPool.Affinity, plan.blocks.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(affinities);
        fillAffinities(tensor, pipeline, plan.blocks, affinities);
        defer for (leased) |block| {
            if (block.len != 0) pipeline.pool.release(pipeline.io, block);
        };

        const successful = readWhole(request, tensor, pipeline, plan, affinities, leased, source_offset, request_len);
        if (successful) request.markSuccessful();
    }

    fn runCoalesced(
        request: *VectoredLoadPipeline.RequestContext,
        source_slot: *LoaderSourceSlot,
        pipeline: *VectoredLoadPipeline,
        file_offset: u64,
        request_len: usize,
        pieces: []const FairVectoredReadScheduler.SourcePiece,
        direct: *DirectLoader,
    ) void {
        defer request.finishScheduling();
        if (pipeline.failed()) return;

        const file = source_slot.ensure(pipeline.io) catch |err| {
            pipeline.recordError(err);
            return;
        };
        const block_count = std.math.divCeil(usize, request_len, pipeline.block_size) catch unreachable;
        if (block_count == 0) {
            request.markReadFinished();
            request.markSuccessful();
            return;
        }

        const leased = pipeline.allocator.alloc([]u8, block_count) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(leased);
        @memset(leased, &.{});
        defer for (leased) |block| {
            if (block.len != 0) pipeline.pool.release(pipeline.io, block);
        };

        const affinities = pipeline.allocator.alloc(mem.DmaBlockPool.Affinity, block_count) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(affinities);
        @memset(affinities, .{});
        const references = pipeline.allocator.alloc(usize, block_count) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(references);
        @memset(references, 0);

        var transfers: std.ArrayList(TransferSlice) = .empty;
        defer transfers.deinit(pipeline.allocator);
        const queue_counts = pipeline.allocator.alloc(usize, pipeline.platform.devices.len) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(queue_counts);
        @memset(queue_counts, 0);

        for (pieces) |piece| {
            const tensor = piece.item.state.ensure(piece.item, direct) catch |err| {
                pipeline.recordError(err);
                return;
            };
            const piece_end = std.math.add(usize, piece.tensor_offset, piece.len) catch {
                pipeline.recordError(error.InvalidLoaderJob);
                return;
            };
            var tensor_cursor = piece.tensor_offset;
            var span_index = tensor.dispatch_spans.spanIndexAt(tensor_cursor) orelse {
                pipeline.recordError(error.InvalidLoaderJob);
                return;
            };
            while (tensor_cursor < piece_end) {
                const span = tensor.dispatch_spans.spans[span_index];
                const absolute = std.math.add(u64, piece.item.source.offset, tensor_cursor) catch {
                    pipeline.recordError(error.InvalidLoaderJob);
                    return;
                };
                if (absolute < file_offset) {
                    pipeline.recordError(error.InvalidLoaderJob);
                    return;
                }
                const source_relative: usize = @intCast(absolute - file_offset);
                const block_index = source_relative / pipeline.block_size;
                const block_offset = source_relative % pipeline.block_size;
                const take = @min(
                    @min(piece_end - tensor_cursor, span.end - tensor_cursor),
                    pipeline.block_size - block_offset,
                );
                const writer_mask = tensor.dispatch_spans.writerMask(span);
                const destination_offset = span.writer_offset + tensor_cursor - span.start;
                var merged = false;
                if (transfers.items.len > 0) merge: {
                    const previous = &transfers.items[transfers.items.len - 1];
                    if (previous.tensor != tensor or previous.block_index != block_index or
                        previous.writer_mask != writer_mask or
                        previous.block_offset + previous.len != block_offset or
                        previous.destination_offset + previous.len != destination_offset)
                        break :merge;
                    previous.len += take;
                    merged = true;
                }
                if (!merged) {
                    transfers.append(pipeline.allocator, .{
                        .tensor = tensor,
                        .block_index = block_index,
                        .block_offset = block_offset,
                        .writer_mask = writer_mask,
                        .destination_offset = destination_offset,
                        .len = take,
                    }) catch {
                        pipeline.recordError(error.OutOfMemory);
                        return;
                    };
                    references[block_index] += @popCount(writer_mask);
                    var mask = writer_mask;
                    while (mask != 0) {
                        const writer_index: usize = @intCast(@ctz(mask));
                        mask &= mask - 1;
                        const target = &tensor.targets[writer_index];
                        queue_counts[target.device_index] += 1;
                        if (pipeline.numa_explicit) {
                            const node_index = pipeline.device_pool_indices[target.device_index];
                            affinities[block_index].eligible_nodes |= @as(u64, 1) << @intCast(node_index);
                        }
                    }
                }
                tensor_cursor += take;
                if (tensor_cursor == span.end) span_index += 1;
            }
        }

        pipeline.reserveReadyCapacity(queue_counts) catch |err| {
            pipeline.recordError(err);
            return;
        };
        pipeline.reserveBlockCapacity(block_count) catch |err| {
            pipeline.recordError(err);
            return;
        };
        const pool_wait_ns = pipeline.pool.acquireMany(pipeline.io, leased, affinities) catch |err| {
            pipeline.recordError(err);
            return;
        };
        recordPoolWait(pipeline, pool_wait_ns);
        if (pipeline.failed()) return;

        const iovecs = pipeline.allocator.alloc([]u8, block_count) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(iovecs);
        for (iovecs, leased, 0..) |*iovec, block, block_index| {
            const consumed = block_index * pipeline.block_size;
            const len = @min(pipeline.block_size, request_len - consumed);
            iovec.* = block[0..len];
        }

        if (!beginRead(request, pipeline)) return;
        const read_started: std.Io.Timestamp = .now(pipeline.io, .awake);
        const read_result = readAbsoluteAllV(
            pipeline.allocator,
            pipeline.io,
            file,
            iovecs,
            file_offset,
            pipeline.metrics,
        );
        const read_elapsed = read_started.untilNow(pipeline.io, .awake);
        read_result catch |err| {
            endRead(request, pipeline);
            pipeline.recordError(err);
            return;
        };
        pipeline.metrics.recordProbeRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
            request_len,
            pipeline.source_request_size,
        );
        const read_elapsed_ns: u64 = @intCast(@max(read_elapsed.nanoseconds, 0));
        const read_elapsed_us = read_elapsed_ns / std.time.ns_per_us;
        _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.read_bytes.fetchAdd(request_len, .monotonic);
        _ = pipeline.metrics.read_ns.fetchAdd(read_elapsed_ns, .monotonic);
        _ = pipeline.metrics.weighted_read_latency_us.fetchAdd(
            read_elapsed_us *| @as(u64, @intCast(request_len)),
            .monotonic,
        );
        for (pieces) |piece| piece.item.state.state.recordReadProgress(piece.len);
        request.markReadFinished();
        endRead(request, pipeline);
        if (pipeline.failed()) return;

        const blocks = pipeline.allocator.alloc(*VectoredLoadPipeline.BlockContext, block_count) catch {
            pipeline.recordError(error.OutOfMemory);
            return;
        };
        defer pipeline.allocator.free(blocks);
        var initialized_blocks: usize = 0;
        for (blocks, leased, references, iovecs) |*block, *lease, refs, iovec| {
            if (refs == 0) {
                for (blocks[0..initialized_blocks], references[0..initialized_blocks]) |initialized, initialized_refs| {
                    pipeline.abandonSubmissions(initialized, initialized_refs);
                }
                pipeline.recordError(error.InvalidLoaderJob);
                return;
            }
            block.* = pipeline.registerBlock(request, lease.*, refs, iovec.len) catch |err| {
                for (blocks[0..initialized_blocks], references[0..initialized_blocks]) |initialized, initialized_refs| {
                    pipeline.abandonSubmissions(initialized, initialized_refs);
                }
                pipeline.recordError(err);
                return;
            };
            lease.* = &.{};
            initialized_blocks += 1;
        }
        for (transfers.items, 0..) |transfer, transfer_index| {
            pipeline.enqueueBlock(
                transfer.tensor,
                blocks[transfer.block_index],
                transfer.writer_mask,
                transfer.block_offset,
                transfer.destination_offset,
                transfer.len,
            ) catch |err| {
                for (transfers.items[transfer_index..]) |remaining| {
                    pipeline.abandonSubmissions(
                        blocks[remaining.block_index],
                        @popCount(remaining.writer_mask),
                    );
                }
                pipeline.recordError(err);
                return;
            };
        }
        request.markSuccessful();
    }
};

/// Fixed-size source jobs scheduled by destination-device debt. A replicated
/// job is present in every destination queue but is claimed exactly once and
/// credits every device it serves.
const FairVectoredReadScheduler = struct {
    const SourcePiece = struct {
        item: *LoaderLoadItem,
        item_index: usize,
        tensor_offset: usize,
        len: usize,
    };

    const Job = struct {
        tensor_index: usize,
        source_slot: ?*LoaderSourceSlot = null,
        file_offset: u64,
        len: usize,
        pieces: []const SourcePiece = &.{},
    };

    const StoredJob = struct {
        tensor_index: usize,
        source_slot: ?*LoaderSourceSlot = null,
        file_offset: u64,
        len: usize,
        piece_start: usize,
        piece_len: usize,
        piece_storage: ?[]const SourcePiece = null,
        predecessor: ?usize,
        adaptive_sample: bool = false,
    };

    const TestJob = struct {
        tensor_index: usize,
        len: usize,
        physical_bytes: []const usize,
        block_count: usize = 1,
    };

    const Snapshot = struct {
        remaining_bytes: u64,
        remaining_jobs: usize,
        remaining_full_jobs: usize,
        has_unscheduled: bool,
    };

    const PreparedBatch = struct {
        allocator: std.mem.Allocator,
        jobs: []StoredJob,
        pieces: []SourcePiece,
        physical_bytes: []usize,
        queues: []std.ArrayListUnmanaged(usize),
        remaining_bytes: u64,
        remaining_full_jobs: usize,
        maximum_blocks_per_job: usize,
        source_runs: usize,

        fn deinit(self: *PreparedBatch) void {
            for (self.queues) |*queue| queue.deinit(self.allocator);
            self.allocator.free(self.queues);
            self.allocator.free(self.physical_bytes);
            self.allocator.free(self.pieces);
            self.allocator.free(self.jobs);
            self.* = undefined;
        }
    };

    allocator: std.mem.Allocator,
    device_count: usize,
    request_size: usize,
    jobs: std.ArrayListUnmanaged(StoredJob) = .empty,
    piece_batches: std.ArrayListUnmanaged([]SourcePiece) = .empty,
    physical_bytes: std.ArrayListUnmanaged(usize) = .empty,
    queues: []std.ArrayListUnmanaged(usize),
    cursors: []usize,
    claimed: std.ArrayListUnmanaged(bool) = .empty,
    scheduled_physical_bytes: []u64,
    remaining_bytes: u64,
    remaining_jobs: usize,
    remaining_full_jobs: usize,
    maximum_blocks_per_job: usize = 0,
    next_device: usize = 0,
    persistent: bool = false,
    sealed: bool = true,
    stopping: bool = false,
    mutex: std.Io.Mutex = .init,
    condition: std.Io.Condition = .init,

    fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        tensors: []const *const Tensor,
        shardings: []const Sharding,
        block_size: usize,
        request_size: usize,
    ) !FairVectoredReadScheduler {
        const device_count = platform.devices.len;
        if (device_count == 0) return error.DmaDeviceMismatch;
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        @memset(queues, .empty);
        const cursors = allocator.alloc(usize, device_count) catch |err| {
            allocator.free(queues);
            return err;
        };
        @memset(cursors, 0);
        const scheduled = allocator.alloc(u64, device_count) catch |err| {
            allocator.free(cursors);
            allocator.free(queues);
            return err;
        };
        @memset(scheduled, 0);
        var self: FairVectoredReadScheduler = .{
            .allocator = allocator,
            .device_count = device_count,
            .request_size = request_size,
            .queues = queues,
            .cursors = cursors,
            .scheduled_physical_bytes = scheduled,
            .remaining_bytes = 0,
            .remaining_jobs = 0,
            .remaining_full_jobs = 0,
        };
        errdefer self.deinit();

        const TensorPlan = struct {
            dispatch_spans: DispatchSpans,
            device_indices: []usize,
            total: usize,
        };
        const tensor_plans = try allocator.alloc(TensorPlan, tensors.len);
        var initialized_plans: usize = 0;
        defer {
            for (tensor_plans[0..initialized_plans]) |*plan| {
                plan.dispatch_spans.deinit(allocator);
                allocator.free(plan.device_indices);
            }
            allocator.free(tensor_plans);
        }
        for (tensors, tensor_plans) |tensor, *plan| {
            const shape = tensor.shape();
            const packed_shape = shape.packedShape();
            const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse
                platform.replicated_sharding;
            plan.* = .{
                .dispatch_spans = try .init(allocator, packed_shape, sharding),
                .device_indices = &.{},
                .total = packed_shape.byteSize(),
            };
            initialized_plans += 1;
            const ordered_devices = sharding.devicesInCanonicalOrder();
            plan.device_indices = try allocator.alloc(usize, ordered_devices.len);
            for (ordered_devices, plan.device_indices) |device, *device_index| {
                device_index.* = @intCast(device.id);
                if (device_index.* >= device_count) return error.DmaDeviceMismatch;
            }
        }

        const offsets = try allocator.alloc(usize, tensors.len);
        defer allocator.free(offsets);
        @memset(offsets, 0);
        const previous_jobs = try allocator.alloc(?usize, tensors.len);
        defer allocator.free(previous_jobs);
        @memset(previous_jobs, null);
        const next_active = try allocator.alloc(usize, tensors.len);
        defer allocator.free(next_active);
        // Retain unfinished tensors in a ring so job construction preserves
        // tensor round-robin order without rescanning completed tensors.
        var tensors_remaining: usize = 0;
        var first_active: ?usize = null;
        var last_active: ?usize = null;
        for (tensor_plans, 0..) |plan, tensor_index| {
            if (plan.total == 0) continue;
            if (first_active == null) first_active = tensor_index;
            if (last_active) |previous| next_active[previous] = tensor_index;
            last_active = tensor_index;
            tensors_remaining += 1;
        }
        if (last_active) |last| next_active[last] = first_active.?;
        var current_tensor = first_active orelse 0;
        var previous_tensor = last_active orelse 0;
        while (tensors_remaining != 0) {
            const tensor_index = current_tensor;
            const tensor_size = tensor_plans[tensor_index].total;
            const source_offset = offsets[tensor_index];
            const len = @min(request_size, tensor_size - source_offset);
            offsets[tensor_index] += len;

            const following_tensor = next_active[tensor_index];
            if (offsets[tensor_index] == tensor_size) {
                tensors_remaining -= 1;
                if (tensors_remaining != 0) {
                    next_active[previous_tensor] = following_tensor;
                    current_tensor = following_tensor;
                }
            } else {
                previous_tensor = tensor_index;
                current_tensor = following_tensor;
            }

            const job_index = self.jobs.items.len;
            try self.jobs.append(allocator, .{
                .tensor_index = tensor_index,
                .file_offset = source_offset,
                .len = len,
                .piece_start = 0,
                .piece_len = 0,
                .predecessor = previous_jobs[tensor_index],
                .adaptive_sample = len == request_size,
            });
            previous_jobs[tensor_index] = job_index;
            try self.physical_bytes.appendNTimes(allocator, 0, device_count);
            const row = self.physical_bytes.items[job_index * device_count ..][0..device_count];
            {
                const request_plan = try VectoredRequestPlan.init(
                    allocator,
                    tensor_plans[tensor_index].dispatch_spans,
                    source_offset,
                    len,
                    block_size,
                );
                defer request_plan.deinit(allocator);
                self.maximum_blocks_per_job = @max(
                    self.maximum_blocks_per_job,
                    request_plan.blocks.len,
                );
                for (request_plan.blocks) |block| {
                    var writer_mask = block.writer_mask;
                    while (writer_mask != 0) {
                        const writer_index: usize = @intCast(@ctz(writer_mask));
                        writer_mask &= writer_mask - 1;
                        const device_index = tensor_plans[tensor_index].device_indices[writer_index];
                        row[device_index] = try std.math.add(usize, row[device_index], block.len);
                    }
                }
            }
            for (row, self.queues) |bytes, *queue| {
                if (bytes != 0) try queue.append(allocator, job_index);
            }
            self.remaining_bytes +|= @intCast(len);
            self.remaining_jobs += 1;
            if (len == request_size) self.remaining_full_jobs += 1;
        }
        try self.claimed.appendNTimes(allocator, false, self.jobs.items.len);
        return self;
    }

    fn initAppendable(
        allocator: std.mem.Allocator,
        device_count: usize,
        request_size: usize,
    ) !FairVectoredReadScheduler {
        if (device_count == 0 or device_count > 64) return error.DmaDeviceMismatch;
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        errdefer allocator.free(queues);
        @memset(queues, .empty);
        const cursors = try allocator.alloc(usize, device_count);
        errdefer allocator.free(cursors);
        @memset(cursors, 0);
        const scheduled = try allocator.alloc(u64, device_count);
        errdefer allocator.free(scheduled);
        @memset(scheduled, 0);
        return .{
            .allocator = allocator,
            .device_count = device_count,
            .request_size = request_size,
            .queues = queues,
            .cursors = cursors,
            .scheduled_physical_bytes = scheduled,
            .remaining_bytes = 0,
            .remaining_jobs = 0,
            .remaining_full_jobs = 0,
            .persistent = true,
            .sealed = false,
        };
    }

    fn prepareBatch(
        allocator: std.mem.Allocator,
        device_count: usize,
        items: []const *LoaderLoadItem,
        block_size: usize,
        request_size: usize,
    ) !PreparedBatch {
        const scatter_limit = std.math.mul(
            usize,
            block_size,
            max_load_positional_iovecs,
        ) catch std.math.maxInt(usize);
        const maximum_job_len = @min(request_size, scatter_limit);
        if (maximum_job_len == 0) return error.InvalidLoaderJob;
        const TensorPlan = struct {
            dispatch_spans: DispatchSpans,
            device_indices: []usize,
            total: usize,
        };
        const plans = try allocator.alloc(TensorPlan, items.len);
        var initialized_plans: usize = 0;
        defer {
            for (plans[0..initialized_plans]) |*plan| {
                plan.dispatch_spans.deinit(allocator);
                allocator.free(plan.device_indices);
            }
            allocator.free(plans);
        }
        for (items, plans) |item, *plan| {
            const packed_shape = item.shape.packedShape();
            plan.* = .{
                .dispatch_spans = try .init(allocator, packed_shape, item.sharding),
                .device_indices = &.{},
                .total = packed_shape.byteSize(),
            };
            initialized_plans += 1;
            if (plan.total != item.source.byteSize()) return error.InvalidLoaderJob;
            const ordered_devices = item.sharding.devicesInCanonicalOrder();
            plan.device_indices = try allocator.alloc(usize, ordered_devices.len);
            for (ordered_devices, plan.device_indices) |device, *device_index| {
                device_index.* = @intCast(device.id);
                if (device_index.* >= device_count) return error.DmaDeviceMismatch;
            }
        }

        const order = try allocator.alloc(usize, items.len);
        defer allocator.free(order);
        for (order, 0..) |*index, i| index.* = i;
        const SortContext = struct {
            items: []const *LoaderLoadItem,

            fn lessThan(ctx: @This(), lhs: usize, rhs: usize) bool {
                const left = ctx.items[lhs];
                const right = ctx.items[rhs];
                const uri_order = std.mem.order(u8, left.source.file_uri, right.source.file_uri);
                if (uri_order != .eq) return uri_order == .lt;
                if (left.source.offset != right.source.offset)
                    return left.source.offset < right.source.offset;
                const left_size = left.source.byteSize();
                const right_size = right.source.byteSize();
                if (left_size != right_size) return left_size < right_size;
                return lhs < rhs;
            }
        };
        std.mem.sort(usize, order, SortContext{ .items = items }, SortContext.lessThan);

        var jobs_list: std.ArrayList(StoredJob) = .empty;
        defer jobs_list.deinit(allocator);
        var pieces_list: std.ArrayList(SourcePiece) = .empty;
        defer pieces_list.deinit(allocator);
        var physical_list: std.ArrayList(usize) = .empty;
        defer physical_list.deinit(allocator);
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        errdefer allocator.free(queues);
        @memset(queues, .empty);
        errdefer for (queues) |*queue| queue.deinit(allocator);
        var remaining_bytes: u64 = 0;
        var remaining_full_jobs: usize = 0;
        var maximum_blocks_per_job: usize = 0;
        var source_runs: usize = 0;
        var file_start: usize = 0;
        while (file_start < order.len) {
            const first_item = items[order[file_start]];
            var file_end = file_start + 1;
            while (file_end < order.len and std.mem.eql(
                u8,
                first_item.source.file_uri,
                items[order[file_end]].source.file_uri,
            )) : (file_end += 1) {}

            var previous_job: ?usize = null;
            var run_cursor = file_start;
            while (run_cursor < file_end) {
                const first_index = order[run_cursor];
                const first_offset = items[first_index].source.offset;
                var run_end = std.math.add(
                    u64,
                    first_offset,
                    items[first_index].source.byteSize(),
                ) catch return error.InvalidLoaderJob;
                if (run_end == first_offset) {
                    run_cursor += 1;
                    continue;
                }
                var run_item_end = run_cursor + 1;
                while (run_item_end < file_end) : (run_item_end += 1) {
                    const candidate = items[order[run_item_end]].source;
                    if (candidate.offset > run_end) break;
                    const candidate_end = std.math.add(u64, candidate.offset, candidate.byteSize()) catch
                        return error.InvalidLoaderJob;
                    run_end = @max(run_end, candidate_end);
                }
                source_runs += 1;

                var job_start = first_offset;
                var candidate_start = run_cursor;
                while (job_start < run_end) {
                    const job_end = @min(
                        run_end,
                        std.math.add(u64, job_start, maximum_job_len) catch run_end,
                    );
                    const job_len: usize = @intCast(job_end - job_start);
                    const job_index = jobs_list.items.len;
                    const piece_start = pieces_list.items.len;
                    var representative = first_index;
                    while (candidate_start < run_item_end) {
                        const candidate = items[order[candidate_start]].source;
                        const candidate_end = std.math.add(u64, candidate.offset, candidate.byteSize()) catch
                            return error.InvalidLoaderJob;
                        if (candidate_end > job_start) break;
                        candidate_start += 1;
                    }
                    for (order[candidate_start..run_item_end]) |item_index| {
                        const item = items[item_index];
                        if (item.source.offset >= job_end) break;
                        const item_end = std.math.add(u64, item.source.offset, item.source.byteSize()) catch
                            return error.InvalidLoaderJob;
                        const intersection_start = @max(job_start, item.source.offset);
                        const intersection_end = @min(job_end, item_end);
                        if (intersection_start >= intersection_end) continue;
                        if (pieces_list.items.len == piece_start) representative = item_index;
                        try pieces_list.append(allocator, .{
                            .item = item,
                            .item_index = item_index,
                            .tensor_offset = @intCast(intersection_start - item.source.offset),
                            .len = @intCast(intersection_end - intersection_start),
                        });
                    }
                    std.debug.assert(pieces_list.items.len > piece_start);
                    try jobs_list.append(allocator, .{
                        .tensor_index = representative,
                        .source_slot = first_item.source_slot,
                        .file_offset = job_start,
                        .len = job_len,
                        .piece_start = piece_start,
                        .piece_len = pieces_list.items.len - piece_start,
                        .predecessor = previous_job,
                        .adaptive_sample = true,
                    });
                    previous_job = job_index;

                    try physical_list.appendNTimes(allocator, 0, device_count);
                    const row = physical_list.items[job_index * device_count ..][0..device_count];
                    for (pieces_list.items[piece_start..]) |piece| {
                        const dispatch = plans[piece.item_index].dispatch_spans;
                        const piece_end = piece.tensor_offset + piece.len;
                        var cursor = piece.tensor_offset;
                        var span_index = dispatch.spanIndexAt(cursor) orelse
                            return error.InvalidLoaderJob;
                        while (cursor < piece_end) {
                            const span = dispatch.spans[span_index];
                            const take = @min(piece_end, span.end) - cursor;
                            var writer_mask = dispatch.writerMask(span);
                            while (writer_mask != 0) {
                                const writer_index: usize = @intCast(@ctz(writer_mask));
                                writer_mask &= writer_mask - 1;
                                const device_index = plans[piece.item_index].device_indices[writer_index];
                                row[device_index] = try std.math.add(usize, row[device_index], take);
                            }
                            cursor += take;
                            if (cursor == span.end) span_index += 1;
                        }
                    }
                    for (row, queues) |bytes, *queue| {
                        if (bytes != 0) try queue.append(allocator, job_index);
                    }
                    const block_count = std.math.divCeil(usize, job_len, block_size) catch unreachable;
                    maximum_blocks_per_job = @max(maximum_blocks_per_job, block_count);
                    remaining_bytes +|= @intCast(job_len);
                    // Coalesced jobs, including exact tails, are valid adaptive samples.
                    remaining_full_jobs += 1;
                    job_start = job_end;
                }
                run_cursor = run_item_end;
            }
            file_start = file_end;
        }

        const jobs = try jobs_list.toOwnedSlice(allocator);
        errdefer allocator.free(jobs);
        const pieces = try pieces_list.toOwnedSlice(allocator);
        errdefer allocator.free(pieces);
        const physical_bytes = try physical_list.toOwnedSlice(allocator);
        errdefer allocator.free(physical_bytes);
        return .{
            .allocator = allocator,
            .jobs = jobs,
            .pieces = pieces,
            .physical_bytes = physical_bytes,
            .queues = queues,
            .remaining_bytes = remaining_bytes,
            .remaining_full_jobs = remaining_full_jobs,
            .maximum_blocks_per_job = maximum_blocks_per_job,
            .source_runs = source_runs,
        };
    }

    fn appendPrepared(self: *FairVectoredReadScheduler, io: std.Io, batch: *PreparedBatch) !void {
        std.debug.assert(self.persistent);
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.stopping) return error.LoaderShuttingDown;
        if (self.sealed) return error.LoaderEpochSealed;

        try self.jobs.ensureUnusedCapacity(self.allocator, batch.jobs.len);
        if (batch.pieces.len != 0)
            try self.piece_batches.ensureUnusedCapacity(self.allocator, 1);
        try self.claimed.ensureUnusedCapacity(self.allocator, batch.jobs.len);
        try self.physical_bytes.ensureUnusedCapacity(self.allocator, batch.physical_bytes.len);
        for (self.queues, batch.queues) |*queue, prepared| {
            try queue.ensureUnusedCapacity(self.allocator, prepared.items.len);
        }

        const base = self.jobs.items.len;
        for (batch.jobs) |job| {
            var stored = job;
            if (stored.predecessor) |predecessor| stored.predecessor = base + predecessor;
            if (batch.pieces.len != 0) stored.piece_storage = batch.pieces;
            self.jobs.appendAssumeCapacity(stored);
            self.claimed.appendAssumeCapacity(false);
        }
        if (batch.pieces.len != 0) {
            self.piece_batches.appendAssumeCapacity(batch.pieces);
            batch.pieces = &.{};
        }
        self.physical_bytes.appendSliceAssumeCapacity(batch.physical_bytes);
        for (self.queues, batch.queues) |*queue, prepared| {
            for (prepared.items) |job_index| queue.appendAssumeCapacity(base + job_index);
        }
        self.remaining_bytes +|= batch.remaining_bytes;
        self.remaining_jobs += batch.jobs.len;
        self.remaining_full_jobs += batch.remaining_full_jobs;
        self.maximum_blocks_per_job = @max(self.maximum_blocks_per_job, batch.maximum_blocks_per_job);
        self.condition.broadcast(io);
    }

    fn seal(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.sealed = true;
        self.condition.broadcast(io);
    }

    fn reopen(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.persistent and self.remaining_jobs == 0 and !self.stopping);
        self.jobs.clearRetainingCapacity();
        for (self.piece_batches.items) |pieces| self.allocator.free(pieces);
        self.piece_batches.clearRetainingCapacity();
        self.claimed.clearRetainingCapacity();
        self.physical_bytes.clearRetainingCapacity();
        for (self.queues) |*queue| queue.clearRetainingCapacity();
        @memset(self.cursors, 0);
        @memset(self.scheduled_physical_bytes, 0);
        self.maximum_blocks_per_job = 0;
        self.next_device = 0;
        self.sealed = false;
        self.condition.broadcast(io);
    }

    fn stop(self: *FairVectoredReadScheduler, io: std.Io) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.stopping = true;
        self.sealed = true;
        self.condition.broadcast(io);
    }

    fn fail(self: *FairVectoredReadScheduler, io: std.Io) usize {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const abandoned = self.remaining_jobs;
        self.remaining_jobs = 0;
        self.remaining_bytes = 0;
        self.remaining_full_jobs = 0;
        self.stopping = true;
        self.sealed = true;
        self.condition.broadcast(io);
        return abandoned;
    }

    fn initForTest(
        allocator: std.mem.Allocator,
        device_count: usize,
        test_jobs: []const TestJob,
    ) !FairVectoredReadScheduler {
        const queues = try allocator.alloc(std.ArrayListUnmanaged(usize), device_count);
        @memset(queues, .empty);
        const cursors = allocator.alloc(usize, device_count) catch |err| {
            allocator.free(queues);
            return err;
        };
        @memset(cursors, 0);
        const scheduled = allocator.alloc(u64, device_count) catch |err| {
            allocator.free(cursors);
            allocator.free(queues);
            return err;
        };
        @memset(scheduled, 0);
        var self: FairVectoredReadScheduler = .{
            .allocator = allocator,
            .device_count = device_count,
            .request_size = max_load_read_request_size,
            .queues = queues,
            .cursors = cursors,
            .scheduled_physical_bytes = scheduled,
            .remaining_bytes = 0,
            .remaining_jobs = 0,
            .remaining_full_jobs = 0,
        };
        errdefer self.deinit();
        var previous_jobs = try allocator.alloc(?usize, test_jobs.len);
        defer allocator.free(previous_jobs);
        @memset(previous_jobs, null);
        for (test_jobs, 0..) |job, job_index| {
            if (job.physical_bytes.len != device_count or job.block_count == 0)
                return error.InvalidTestJob;
            try self.jobs.append(allocator, .{
                .tensor_index = job.tensor_index,
                .file_offset = 0,
                .len = job.len,
                .piece_start = 0,
                .piece_len = 0,
                .predecessor = if (job.tensor_index < previous_jobs.len)
                    previous_jobs[job.tensor_index]
                else
                    null,
                .adaptive_sample = job.len == self.request_size,
            });
            if (job.tensor_index < previous_jobs.len)
                previous_jobs[job.tensor_index] = job_index;
            try self.physical_bytes.appendSlice(allocator, job.physical_bytes);
            var destinations: usize = 0;
            for (job.physical_bytes, self.queues) |bytes, *queue| {
                if (bytes == 0) continue;
                try queue.append(allocator, job_index);
                destinations += 1;
            }
            if (destinations == 0) return error.InvalidTestJob;
            self.maximum_blocks_per_job = @max(
                self.maximum_blocks_per_job,
                job.block_count,
            );
            self.remaining_bytes +|= @intCast(job.len);
            self.remaining_jobs += 1;
            if (job.len == self.request_size) self.remaining_full_jobs += 1;
        }
        try self.claimed.appendNTimes(allocator, false, test_jobs.len);
        return self;
    }

    fn appendTestJobs(
        self: *FairVectoredReadScheduler,
        io: std.Io,
        test_jobs: []const TestJob,
    ) !void {
        std.debug.assert(self.persistent);
        const jobs = try self.allocator.alloc(StoredJob, test_jobs.len);
        defer self.allocator.free(jobs);
        const bytes = try self.allocator.alloc(usize, test_jobs.len * self.device_count);
        defer self.allocator.free(bytes);
        const queues = try self.allocator.alloc(std.ArrayListUnmanaged(usize), self.device_count);
        defer self.allocator.free(queues);
        @memset(queues, .empty);
        defer for (queues) |*queue| queue.deinit(self.allocator);
        const previous = try self.allocator.alloc(?usize, test_jobs.len);
        defer self.allocator.free(previous);
        @memset(previous, null);
        var remaining_bytes: u64 = 0;
        var remaining_full_jobs: usize = 0;
        var maximum_blocks: usize = 0;
        for (test_jobs, jobs, 0..) |job, *stored, job_index| {
            if (job.physical_bytes.len != self.device_count or job.block_count == 0)
                return error.InvalidTestJob;
            stored.* = .{
                .tensor_index = job.tensor_index,
                .file_offset = 0,
                .len = job.len,
                .piece_start = 0,
                .piece_len = 0,
                .predecessor = if (job.tensor_index < previous.len)
                    previous[job.tensor_index]
                else
                    null,
                .adaptive_sample = job.len == self.request_size,
            };
            if (job.tensor_index < previous.len) previous[job.tensor_index] = job_index;
            @memcpy(bytes[job_index * self.device_count ..][0..self.device_count], job.physical_bytes);
            var destinations: usize = 0;
            for (job.physical_bytes, queues) |physical, *queue| {
                if (physical == 0) continue;
                try queue.append(self.allocator, job_index);
                destinations += 1;
            }
            if (destinations == 0) return error.InvalidTestJob;
            remaining_bytes +|= @intCast(job.len);
            if (job.len == self.request_size) remaining_full_jobs += 1;
            maximum_blocks = @max(maximum_blocks, job.block_count);
        }
        var batch: PreparedBatch = .{
            .allocator = self.allocator,
            .jobs = jobs,
            .pieces = &.{},
            .physical_bytes = bytes,
            .queues = queues,
            .remaining_bytes = remaining_bytes,
            .remaining_full_jobs = remaining_full_jobs,
            .maximum_blocks_per_job = maximum_blocks,
            .source_runs = test_jobs.len,
        };
        try self.appendPrepared(io, &batch);
    }

    fn deinit(self: *FairVectoredReadScheduler) void {
        for (self.queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.queues);
        self.allocator.free(self.cursors);
        self.claimed.deinit(self.allocator);
        for (self.piece_batches.items) |pieces| self.allocator.free(pieces);
        self.piece_batches.deinit(self.allocator);
        self.allocator.free(self.scheduled_physical_bytes);
        self.jobs.deinit(self.allocator);
        self.physical_bytes.deinit(self.allocator);
        self.* = undefined;
    }

    fn claim(self: *FairVectoredReadScheduler, io: std.Io) ?Job {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.remaining_jobs == 0) return null;

        var selected_device: ?usize = null;
        var selected_job: ?usize = null;
        for (0..self.device_count) |offset| {
            const device_index = (self.next_device + offset) % self.device_count;
            const queue = &self.queues[device_index];
            while (self.cursors[device_index] < queue.items.len and
                self.claimed.items[queue.items[self.cursors[device_index]]])
            {
                self.cursors[device_index] += 1;
            }
            var candidate: ?usize = null;
            for (queue.items[self.cursors[device_index]..]) |job_index| {
                if (self.claimed.items[job_index]) continue;
                const predecessor = self.jobs.items[job_index].predecessor;
                if (predecessor == null or self.claimed.items[predecessor.?]) {
                    candidate = job_index;
                    break;
                }
            }
            if (candidate == null) continue;
            if (selected_device == null or
                self.scheduled_physical_bytes[device_index] <
                    self.scheduled_physical_bytes[selected_device.?])
            {
                selected_device = device_index;
                selected_job = candidate;
            }
        }
        const device_index = selected_device orelse unreachable;
        const job_index = selected_job.?;
        std.debug.assert(!self.claimed.items[job_index]);
        self.claimed.items[job_index] = true;
        self.remaining_jobs -= 1;
        const stored = self.jobs.items[job_index];
        if (stored.adaptive_sample) self.remaining_full_jobs -= 1;
        self.remaining_bytes -= stored.len;
        const row = self.physical_bytes.items[job_index * self.device_count ..][0..self.device_count];
        for (row, self.scheduled_physical_bytes) |bytes, *scheduled| {
            scheduled.* +|= @intCast(bytes);
        }
        self.next_device = (device_index + 1) % self.device_count;
        if (self.remaining_jobs == 0) self.condition.broadcast(io);
        return .{
            .tensor_index = stored.tensor_index,
            .source_slot = stored.source_slot,
            .file_offset = stored.file_offset,
            .len = stored.len,
            .pieces = if (stored.piece_storage) |pieces|
                pieces[stored.piece_start..][0..stored.piece_len]
            else
                &.{},
        };
    }

    fn waitForWork(self: *FairVectoredReadScheduler, io: std.Io) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        while (self.remaining_jobs == 0 and !self.stopping) {
            self.condition.waitUncancelable(io, &self.mutex);
        }
        return !self.stopping;
    }

    fn snapshot(self: *FairVectoredReadScheduler, io: std.Io) Snapshot {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{
            .remaining_bytes = self.remaining_bytes,
            .remaining_jobs = self.remaining_jobs,
            .remaining_full_jobs = self.remaining_full_jobs,
            .has_unscheduled = self.remaining_jobs != 0,
        };
    }

    fn maximumBlocksPerJob(self: *const FairVectoredReadScheduler) usize {
        return self.maximum_blocks_per_job;
    }
};

test "fair read scheduler rotates sharded devices by scheduled bytes" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 1, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 2, .len = 10, .physical_bytes = &.{ 0, 10 } },
        .{ .tensor_index = 3, .len = 10, .physical_bytes = &.{ 0, 10 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(usize, 0), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 1), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 3), scheduler.claim(io).?.tensor_index);
}

test "source batch coalesces exact adjacent and overlapping tensor ranges" {
    const allocator = std.testing.allocator;
    const io = std.testing.io;
    var platform = Platform.auto(allocator, io, .{ .cpu = .{ .device_count = 1 } }) catch
        return error.SkipZigTest;
    defer platform.deinit(allocator, io);

    var sources = [_]safetensors.Tensor{
        .{ .file_uri = "a", .name = "a0", .shape = .init(.{4}, .u8), .offset = 10 },
        .{ .file_uri = "a", .name = "a1", .shape = .init(.{4}, .u8), .offset = 14 },
        .{ .file_uri = "a", .name = "a0-copy", .shape = .init(.{4}, .u8), .offset = 10 },
        .{ .file_uri = "a", .name = "a-gap", .shape = .init(.{4}, .u8), .offset = 20 },
        .{ .file_uri = "b", .name = "b0", .shape = .init(.{12}, .u8), .offset = 3 },
    };
    var slots = [_]LoaderSourceSlot{
        .{ .uri = "a" },
        .{ .uri = "b" },
    };
    var outputs: [sources.len]Buffer = undefined;
    var items: [sources.len]LoaderLoadItem = undefined;
    var item_ptrs: [sources.len]*LoaderLoadItem = undefined;
    for (&items, &item_ptrs, 0..) |*item, *item_ptr, i| {
        item.* = .{
            .source = &sources[i],
            .source_slot = if (i == sources.len - 1) &slots[1] else &slots[0],
            .shape = sources[i].shape,
            .sharding = platform.replicated_sharding,
            .output = &outputs[i],
        };
        item_ptr.* = item;
    }
    var device_count: usize = 0;
    for (platform.replicated_sharding.devicesInCanonicalOrder()) |device| {
        device_count = @max(device_count, @as(usize, @intCast(device.id)) + 1);
    }

    var batch = try FairVectoredReadScheduler.prepareBatch(
        allocator,
        device_count,
        &item_ptrs,
        4,
        8,
    );
    defer batch.deinit();

    // a:[10,18) merges adjacency and the duplicate, a:[20,24) remains exact,
    // and b:[3,15) is split at the request-size boundary.
    try std.testing.expectEqual(@as(usize, 3), batch.source_runs);
    try std.testing.expectEqual(@as(usize, 4), batch.jobs.len);
    try std.testing.expectEqual(@as(u64, 24), batch.remaining_bytes);
    try std.testing.expectEqual(@as(usize, 6), batch.pieces.len);
    try std.testing.expectEqual(@as(u64, 10), batch.jobs[0].file_offset);
    try std.testing.expectEqual(@as(usize, 8), batch.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 3), batch.jobs[0].piece_len);
    try std.testing.expectEqual(@as(u64, 20), batch.jobs[1].file_offset);
    try std.testing.expectEqual(@as(u64, 3), batch.jobs[2].file_offset);
    try std.testing.expectEqual(@as(usize, 8), batch.jobs[2].len);
    try std.testing.expectEqual(@as(usize, 4), batch.jobs[3].len);
    try std.testing.expectEqual(@as(usize, 4), batch.remaining_full_jobs);

    var scheduler = try FairVectoredReadScheduler.initAppendable(allocator, device_count, 8);
    defer scheduler.deinit();
    try scheduler.appendPrepared(io, &batch);
    while (scheduler.claim(io)) |_| {}
    try std.testing.expectEqual(@as(usize, 0), scheduler.snapshot(io).remaining_full_jobs);

    var iov_source: safetensors.Tensor = .{
        .file_uri = "iov",
        .name = "iov0",
        .shape = .init(.{@as(i64, @intCast(max_load_positional_iovecs + 1))}, .u8),
        .offset = 0,
    };
    var iov_slot: LoaderSourceSlot = .{ .uri = "iov" };
    var iov_output: Buffer = undefined;
    var iov_item: LoaderLoadItem = .{
        .source = &iov_source,
        .source_slot = &iov_slot,
        .shape = iov_source.shape,
        .sharding = platform.replicated_sharding,
        .output = &iov_output,
    };
    var iov_batch = try FairVectoredReadScheduler.prepareBatch(
        allocator,
        device_count,
        &.{&iov_item},
        1,
        max_load_positional_iovecs + 1,
    );
    defer iov_batch.deinit();
    try std.testing.expectEqual(@as(usize, 2), iov_batch.jobs.len);
    try std.testing.expectEqual(max_load_positional_iovecs, iov_batch.jobs[0].len);
    try std.testing.expectEqual(@as(usize, 1), iov_batch.jobs[1].len);
}

test "fair read scheduler preserves per-tensor request order" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{ 1, 0 } },
        .{ .tensor_index = 0, .len = 2, .physical_bytes = &.{ 0, 2 } },
        .{ .tensor_index = 1, .len = 3, .physical_bytes = &.{ 0, 3 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(usize, 1), scheduler.claim(io).?.len);
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.len);
    try std.testing.expectEqual(@as(usize, 3), scheduler.claim(io).?.len);
}

test "fair read scheduler claims a replicated job once and credits every replica" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 20, .physical_bytes = &.{ 20, 20 } },
        .{ .tensor_index = 1, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 2, .len = 10, .physical_bytes = &.{ 0, 10 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(usize, 0), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqualSlices(u64, &.{ 20, 20 }, scheduler.scheduled_physical_bytes);
    // The replicated entry is skipped in device 1's queue; tie rotation gives
    // that device the next scheduling turn.
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 1), scheduler.claim(io).?.tensor_index);
    try std.testing.expect(scheduler.claim(io) == null);
}

test "fair read scheduler compares physical bytes rather than scheduling turns" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 4, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 1, .len = 4, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 2, .len = 4, .physical_bytes = &.{ 4, 0 } },
        .{ .tensor_index = 3, .len = 10, .physical_bytes = &.{ 0, 10 } },
        .{ .tensor_index = 4, .len = 10, .physical_bytes = &.{ 0, 10 } },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &jobs);
    defer scheduler.deinit();
    const io = std.testing.io;
    try std.testing.expectEqual(@as(usize, 0), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 3), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 1), scheduler.claim(io).?.tensor_index);
    // Device 0 receives another turn because it has 8 scheduled bytes while
    // device 1 has 10; a turn-count scheduler would alternate here.
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 4), scheduler.claim(io).?.tensor_index);
}

test "fair read scheduler tracks fixed jobs and tails" {
    const jobs = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = max_load_read_request_size, .physical_bytes = &.{max_load_read_request_size} },
        .{ .tensor_index = 1, .len = max_load_read_request_size, .physical_bytes = &.{max_load_read_request_size} },
        .{ .tensor_index = 2, .len = 7, .physical_bytes = &.{7} },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 1, &jobs);
    defer scheduler.deinit();
    const initial = scheduler.snapshot(std.testing.io);
    try std.testing.expectEqual(@as(usize, 3), initial.remaining_jobs);
    try std.testing.expectEqual(@as(usize, 2), initial.remaining_full_jobs);
    _ = scheduler.claim(std.testing.io).?;
    const after = scheduler.snapshot(std.testing.io);
    try std.testing.expectEqual(@as(usize, 2), after.remaining_jobs);
    try std.testing.expectEqual(@as(usize, 1), after.remaining_full_jobs);
}

test "fair read scheduler concurrent claims return every logical job once" {
    var job_storage: [32]FairVectoredReadScheduler.TestJob = undefined;
    for (&job_storage, 0..) |*job, index| job.* = .{
        .tensor_index = index,
        .len = 1,
        .physical_bytes = if (index % 3 == 0) &.{ 1, 1 } else if (index % 2 == 0) &.{ 1, 0 } else &.{ 0, 1 },
    };
    var scheduler = try FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &job_storage);
    defer scheduler.deinit();
    var seen: std.atomic.Value(u64) = .init(0);
    var claim_count: std.atomic.Value(usize) = .init(0);
    var duplicate: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    for (0..8) |_| try group.concurrent(std.testing.io, struct {
        fn run(
            scheduler_: *FairVectoredReadScheduler,
            seen_: *std.atomic.Value(u64),
            claim_count_: *std.atomic.Value(usize),
            duplicate_: *std.atomic.Value(bool),
        ) void {
            while (scheduler_.claim(std.testing.io)) |job| {
                const mask = @as(u64, 1) << @intCast(job.tensor_index);
                if (seen_.fetchOr(mask, .acq_rel) & mask != 0) duplicate_.store(true, .release);
                _ = claim_count_.fetchAdd(1, .monotonic);
            }
        }
    }.run, .{ &scheduler, &seen, &claim_count, &duplicate });
    try group.await(std.testing.io);
    try std.testing.expectEqual(std.math.maxInt(u32), @as(u32, @truncate(seen.load(.acquire))));
    try std.testing.expectEqual(job_storage.len, claim_count.load(.acquire));
    try std.testing.expect(!duplicate.load(.acquire));
}

test "fair read scheduler validates jobs and cleans up allocation failures" {
    const wrong_width = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{1} },
    };
    try std.testing.expectError(
        error.InvalidTestJob,
        FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &wrong_width),
    );
    const no_destination = [_]FairVectoredReadScheduler.TestJob{
        .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{ 0, 0 } },
    };
    try std.testing.expectError(
        error.InvalidTestJob,
        FairVectoredReadScheduler.initForTest(std.testing.allocator, 2, &no_destination),
    );

    const AllocationTest = struct {
        fn run(allocator: std.mem.Allocator) !void {
            const jobs = [_]FairVectoredReadScheduler.TestJob{
                .{ .tensor_index = 0, .len = 1, .physical_bytes = &.{ 1, 1 } },
                .{ .tensor_index = 1, .len = 1, .physical_bytes = &.{ 1, 0 } },
            };
            var scheduler = try FairVectoredReadScheduler.initForTest(allocator, 2, &jobs);
            defer scheduler.deinit();
        }
    };
    try std.testing.checkAllAllocationFailures(std.testing.allocator, AllocationTest.run, .{});
}

test "appendable fair scheduler admits late jobs by cumulative device debt" {
    const io = std.testing.io;
    var scheduler = try FairVectoredReadScheduler.initAppendable(
        std.testing.allocator,
        2,
        max_load_read_request_size,
    );
    defer scheduler.deinit();
    try scheduler.appendTestJobs(io, &.{
        .{ .tensor_index = 0, .len = 10, .physical_bytes = &.{ 10, 0 } },
        .{ .tensor_index = 1, .len = 10, .physical_bytes = &.{ 10, 0 } },
    });
    try std.testing.expectEqual(@as(usize, 0), scheduler.claim(io).?.tensor_index);
    try scheduler.appendTestJobs(io, &.{
        .{ .tensor_index = 2, .len = 10, .physical_bytes = &.{ 0, 10 } },
    });
    // The late device-1 job joins immediately because device 0 already owes
    // ten scheduled bytes. The already-claimed request is never pre-empted.
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.tensor_index);
    try std.testing.expectEqual(@as(usize, 1), scheduler.claim(io).?.tensor_index);
}

test "appendable fair scheduler resets debt only at a reusable epoch barrier" {
    const io = std.testing.io;
    var scheduler = try FairVectoredReadScheduler.initAppendable(
        std.testing.allocator,
        2,
        max_load_read_request_size,
    );
    defer scheduler.deinit();
    try scheduler.appendTestJobs(io, &.{
        .{ .tensor_index = 0, .len = 8, .physical_bytes = &.{ 8, 0 } },
        .{ .tensor_index = 1, .len = 8, .physical_bytes = &.{ 0, 8 } },
    });
    _ = scheduler.claim(io).?;
    _ = scheduler.claim(io).?;
    scheduler.seal(io);
    try std.testing.expectEqualSlices(u64, &.{ 8, 8 }, scheduler.scheduled_physical_bytes);
    scheduler.reopen(io);
    try std.testing.expectEqualSlices(u64, &.{ 0, 0 }, scheduler.scheduled_physical_bytes);
    try scheduler.appendTestJobs(io, &.{
        .{ .tensor_index = 2, .len = 4, .physical_bytes = &.{ 0, 4 } },
    });
    try std.testing.expectEqual(@as(usize, 2), scheduler.claim(io).?.tensor_index);
}

const read_width_ladder = [_]usize{ 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128 };

/// Source-only adaptive state. DMA width and request size never enter its
/// evidence or decisions.
const SourceReadWidthController = struct {
    const Phase = enum { baseline, upward, downward, pair_reference, pair_candidate, settled };

    const Evidence = struct {
        generation: u64,
        width: usize,
        completed_full_requests: usize,
        elapsed_ns: u64,
        bytes: u64,
        exercised_width: usize,
        clean: bool,
        remaining_full_jobs: usize,

        fn scoreable(self: Evidence) bool {
            return self.clean and self.generation != std.math.maxInt(u64) and
                self.exercised_width >= self.width and
                self.completed_full_requests >= @max(@as(usize, 8), self.width) and
                self.elapsed_ns >= 100 * std.time.ns_per_ms and self.bytes != 0;
        }

        fn bytesPerSecond(self: Evidence) f64 {
            if (self.elapsed_ns == 0) return 0;
            return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
                @as(f64, @floatFromInt(self.elapsed_ns));
        }
    };

    const Decision = struct {
        width: usize,
        generation: u64,
        changed: bool = false,
        settled: bool = false,
    };

    adaptive: bool,
    fixed_width: ?usize = null,
    maximum_index: usize,
    current_index: usize,
    selected_index: usize,
    peak_index: usize,
    generation: u64 = 0,
    phase: Phase,
    rates: [read_width_ladder.len]?f64 = @splat(null),
    unchanged_candidates: usize = 0,
    pair_resume: Phase = .upward,
    pair_resume_index: usize = 0,
    pair_prior_selected_index: usize = 0,
    pair_candidate_index: usize = 0,
    pair_reference_index: usize = 0,
    pair_candidate_total: f64 = 0,
    pair_reference_total: f64 = 0,
    pair_count: usize = 0,

    fn init(configured: Parallelism, pinned_feasible_width: usize) SourceReadWidthController {
        const configured_max = @min(configured.maximum(), pinned_feasible_width);
        var maximum_index: usize = 0;
        for (read_width_ladder, 0..) |candidate_width, index| {
            if (candidate_width > configured_max) break;
            maximum_index = index;
        }
        if (!configured.isAdaptive()) {
            const fixed = @min(configured.initial(), pinned_feasible_width);
            const fixed_index = widthIndexAtMost(fixed);
            return .{
                .adaptive = false,
                .fixed_width = @max(@as(usize, 1), fixed),
                .maximum_index = fixed_index,
                .current_index = fixed_index,
                .selected_index = fixed_index,
                .peak_index = fixed_index,
                .phase = .settled,
            };
        }
        const initial_index = @min(widthIndexAtMost(configured.initial()), maximum_index);
        return .{
            .adaptive = true,
            .maximum_index = maximum_index,
            .current_index = initial_index,
            .selected_index = initial_index,
            .peak_index = initial_index,
            .phase = .baseline,
        };
    }

    fn widthIndexAtMost(maximum: usize) usize {
        var result: usize = 0;
        for (read_width_ladder, 0..) |candidate_width, index| {
            if (candidate_width > maximum) break;
            result = index;
        }
        return result;
    }

    fn width(self: *const SourceReadWidthController) usize {
        return self.fixed_width orelse read_width_ladder[self.current_index];
    }

    fn selectedWidth(self: *const SourceReadWidthController) usize {
        return self.fixed_width orelse read_width_ladder[self.selected_index];
    }

    fn currentDecision(self: *const SourceReadWidthController) Decision {
        return .{
            .width = self.width(),
            .generation = self.generation,
            .settled = self.phase == .settled,
        };
    }

    fn probeCost(index: usize) usize {
        const candidate_width = read_width_ladder[index];
        return candidate_width +| @max(@as(usize, 8), candidate_width);
    }

    fn probeFitsTail(index: usize, remaining_full_jobs: usize) bool {
        return probeCost(index) *| 4 <= remaining_full_jobs;
    }

    fn pairFitsTail(candidate: usize, reference: usize, remaining_full_jobs: usize) bool {
        const three_pairs = (probeCost(candidate) +| probeCost(reference)) *| 3;
        return three_pairs *| 4 <= remaining_full_jobs;
    }

    fn restartFitsTail(self: *const SourceReadWidthController, remaining_full_jobs: usize) bool {
        return switch (self.phase) {
            .pair_reference => blk: {
                const remaining_pairs = 3 -| self.pair_count;
                const remaining_cost = remaining_pairs *|
                    (probeCost(self.pair_reference_index) +| probeCost(self.pair_candidate_index));
                break :blk remaining_cost *| 4 <= remaining_full_jobs;
            },
            .pair_candidate => blk: {
                const remaining_pairs = 3 -| self.pair_count;
                const remaining_cost = probeCost(self.pair_candidate_index) +|
                    (remaining_pairs -| 1) *|
                        (probeCost(self.pair_reference_index) +| probeCost(self.pair_candidate_index));
                break :blk remaining_cost *| 4 <= remaining_full_jobs;
            },
            .settled => true,
            else => probeFitsTail(self.current_index, remaining_full_jobs),
        };
    }

    fn blindGrow(
        self: *SourceReadWidthController,
        remaining_full_jobs: usize,
    ) ?Decision {
        if (!self.adaptive or self.phase != .baseline or self.current_index >= self.maximum_index)
            return null;
        const ceiling: usize = if (self.width() < 24) 24 else if (self.width() < 32) 32 else return null;
        const target = @min(widthIndexAtMost(ceiling), self.maximum_index);
        if (target <= self.current_index or !probeFitsTail(target, remaining_full_jobs)) return null;
        return self.changeTo(target);
    }

    fn observe(self: *SourceReadWidthController, evidence: Evidence) Decision {
        if (!self.adaptive or self.phase == .settled or
            evidence.generation != self.generation or evidence.width != self.width() or
            !evidence.scoreable())
            return self.currentDecision();
        const rate = evidence.bytesPerSecond();
        return switch (self.phase) {
            .baseline, .upward, .downward => self.finishScore(
                self.current_index,
                rate,
                evidence.remaining_full_jobs,
                true,
            ),
            .pair_reference => blk: {
                self.pair_reference_total += rate;
                self.phase = .pair_candidate;
                break :blk self.changeTo(self.pair_candidate_index);
            },
            .pair_candidate => blk: {
                self.pair_candidate_total += rate;
                self.pair_count += 1;
                if (self.pair_count < 3) {
                    self.phase = .pair_reference;
                    break :blk self.changeTo(self.pair_reference_index);
                }
                const reference_average = self.pair_reference_total / 3;
                const candidate_average = self.pair_candidate_total / 3;
                const reference_rate = self.rates[self.pair_reference_index] orelse reference_average;
                const normalized_candidate = if (reference_average == 0)
                    0
                else
                    reference_rate * candidate_average / reference_average;
                self.rates[self.pair_candidate_index] = normalized_candidate;
                self.recomputePeakAndSelection();
                self.phase = self.pair_resume;
                self.current_index = self.pair_resume_index;
                break :blk self.advanceAfterScore(
                    self.pair_resume_index,
                    self.pair_prior_selected_index,
                    evidence.remaining_full_jobs,
                );
            },
            .settled => self.currentDecision(),
        };
    }

    fn finishScore(
        self: *SourceReadWidthController,
        index: usize,
        rate: f64,
        remaining_full_jobs: usize,
        allow_pair: bool,
    ) Decision {
        const prior_selected = self.selected_index;
        self.rates[index] = rate;
        self.recomputePeakAndSelection();
        const peak_rate = self.rates[self.peak_index] orelse rate;
        const pair_candidate: ?usize = blk: {
            // Confirm the decision made by this score first. A downward
            // candidate just outside the band must not be hidden by an older
            // in-band selection. If this score establishes a new peak,
            // confirm the previously selected smaller width before discarding
            // it. The final entry covers an older unresolved boundary.
            for ([_]usize{ index, prior_selected, self.selected_index }) |candidate| {
                if (candidate == self.peak_index) continue;
                const candidate_rate = self.rates[candidate] orelse continue;
                const retention = if (peak_rate == 0) 0 else candidate_rate / peak_rate;
                if (@abs(retention - 0.97) <= 0.02) break :blk candidate;
            }
            break :blk null;
        };
        if (allow_pair and pair_candidate != null and
            pairFitsTail(pair_candidate.?, self.peak_index, remaining_full_jobs))
        {
            return self.startPair(
                pair_candidate.?,
                self.peak_index,
                self.phase,
                index,
                prior_selected,
            );
        }

        return self.advanceAfterScore(index, prior_selected, remaining_full_jobs);
    }

    fn advanceAfterScore(
        self: *SourceReadWidthController,
        index: usize,
        prior_selected: usize,
        remaining_full_jobs: usize,
    ) Decision {
        const peak_rate = self.rates[self.peak_index] orelse 0;
        const index_rate = self.rates[index] orelse 0;
        const retention = if (peak_rate == 0) 0 else index_rate / peak_rate;
        return switch (self.phase) {
            .baseline => blk: {
                self.phase = .upward;
                if (index < self.maximum_index and probeFitsTail(index + 1, remaining_full_jobs))
                    break :blk self.changeTo(index + 1);
                break :blk self.beginDownwardOrSettle(remaining_full_jobs);
            },
            .upward => blk: {
                if (self.selected_index == prior_selected)
                    self.unchanged_candidates += 1
                else
                    self.unchanged_candidates = 0;
                if (self.unchanged_candidates >= 2 or index == self.maximum_index)
                    break :blk self.beginDownwardOrSettle(remaining_full_jobs);
                if (probeFitsTail(index + 1, remaining_full_jobs))
                    break :blk self.changeTo(index + 1);
                break :blk self.beginDownwardOrSettle(remaining_full_jobs);
            },
            .downward => blk: {
                if (retention >= 0.97) self.selected_index = index;
                if (retention < 0.97 or index == 0 or
                    !probeFitsTail(index - 1, remaining_full_jobs))
                    break :blk self.settle();
                break :blk self.changeTo(index - 1);
            },
            else => self.currentDecision(),
        };
    }

    fn recomputePeakAndSelection(self: *SourceReadWidthController) void {
        var peak_index = self.peak_index;
        var peak_rate: f64 = self.rates[peak_index] orelse 0;
        for (self.rates, 0..) |maybe_rate, index| {
            const rate = maybe_rate orelse continue;
            if (rate > peak_rate) {
                peak_rate = rate;
                peak_index = index;
            }
        }
        self.peak_index = peak_index;
        var selected = peak_index;
        for (self.rates, 0..) |maybe_rate, index| {
            const rate = maybe_rate orelse continue;
            if (rate >= peak_rate * 0.97) {
                selected = index;
                break;
            }
        }
        self.selected_index = selected;
    }

    fn startPair(
        self: *SourceReadWidthController,
        candidate: usize,
        reference: usize,
        resume_phase: Phase,
        resume_index: usize,
        prior_selected: usize,
    ) Decision {
        self.pair_candidate_index = candidate;
        self.pair_reference_index = reference;
        self.pair_resume = resume_phase;
        self.pair_resume_index = resume_index;
        self.pair_prior_selected_index = prior_selected;
        self.pair_candidate_total = 0;
        self.pair_reference_total = 0;
        self.pair_count = 0;
        self.phase = .pair_reference;
        return self.changeToForced(reference);
    }

    fn beginDownwardOrSettle(
        self: *SourceReadWidthController,
        remaining_full_jobs: usize,
    ) Decision {
        self.phase = .downward;
        if (self.selected_index > 0 and probeFitsTail(self.selected_index - 1, remaining_full_jobs))
            return self.changeTo(self.selected_index - 1);
        return self.settle();
    }

    fn changeTo(self: *SourceReadWidthController, index: usize) Decision {
        const changed = index != self.current_index;
        self.current_index = index;
        if (changed) self.generation +|= 1;
        return .{
            .width = self.width(),
            .generation = self.generation,
            .changed = changed,
            .settled = self.phase == .settled,
        };
    }

    fn changeToForced(self: *SourceReadWidthController, index: usize) Decision {
        self.current_index = index;
        self.generation +|= 1;
        return .{
            .width = self.width(),
            .generation = self.generation,
            .changed = true,
            .settled = self.phase == .settled,
        };
    }

    fn settle(self: *SourceReadWidthController) Decision {
        self.phase = .settled;
        return self.changeTo(self.selected_index);
    }

    fn rollbackTail(self: *SourceReadWidthController) Decision {
        if (self.phase == .settled) return self.currentDecision();
        if (self.phase == .pair_reference or self.phase == .pair_candidate)
            self.selected_index = self.pair_prior_selected_index;
        return self.settle();
    }

    fn backoff(self: *SourceReadWidthController) Decision {
        if (!self.adaptive) return self.currentDecision();
        if (self.current_index == 0) {
            self.selected_index = 0;
            return self.settle();
        }

        // Never retain a width above the last clean selection after the source
        // reports pressure. Further feedback can keep walking a settled
        // controller down one rung at a time.
        self.selected_index = @min(self.current_index - 1, self.selected_index);
        self.peak_index = self.selected_index;
        self.phase = .settled;
        return self.changeTo(self.selected_index);
    }
};

fn sourceReadTestEvidence(
    controller: *const SourceReadWidthController,
    rate: u64,
    remaining_full_jobs: usize,
) SourceReadWidthController.Evidence {
    return .{
        .generation = controller.generation,
        .width = controller.width(),
        .completed_full_requests = @max(@as(usize, 8), controller.width()),
        .elapsed_ns = std.time.ns_per_s,
        .bytes = rate,
        .exercised_width = controller.width(),
        .clean = true,
        .remaining_full_jobs = remaining_full_jobs,
    };
}

test "source read controller bounds blind growth at 32" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        128,
    );
    try std.testing.expectEqual(@as(usize, 12), controller.width());
    try std.testing.expectEqual(@as(usize, 24), controller.blindGrow(10_000).?.width);
    try std.testing.expectEqual(@as(usize, 32), controller.blindGrow(10_000).?.width);
    try std.testing.expect(controller.blindGrow(10_000) == null);
}

test "source read controller clips infeasible adaptive and fixed widths" {
    var adaptive = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        10,
    );
    try std.testing.expectEqual(@as(usize, 8), adaptive.width());
    try std.testing.expect(adaptive.blindGrow(10_000) == null);

    const fixed = SourceReadWidthController.init(.{ .fixed = 20 }, 7);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.currentDecision().settled);

    const configured_initial = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 48, .maximum = 128 } },
        128,
    );
    try std.testing.expectEqual(@as(usize, 48), configured_initial.width());
}

test "source read controller isolates generation and requires exercised clean evidence" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    var stale = sourceReadTestEvidence(&controller, 100, 10_000);
    stale.generation +|= 1;
    try std.testing.expect(!controller.observe(stale).changed);
    var short = sourceReadTestEvidence(&controller, 100, 10_000);
    short.elapsed_ns = 99 * std.time.ns_per_ms;
    try std.testing.expect(!controller.observe(short).changed);
    var unexercised = sourceReadTestEvidence(&controller, 100, 10_000);
    unexercised.exercised_width -= 1;
    try std.testing.expect(!controller.observe(unexercised).changed);
    var dirty = sourceReadTestEvidence(&controller, 100, 10_000);
    dirty.clean = false;
    try std.testing.expect(!controller.observe(dirty).changed);
    try std.testing.expectEqual(
        @as(usize, 16),
        controller.observe(sourceReadTestEvidence(&controller, 100, 10_000)).width,
    );
}

test "source read controller selects plateau then refines downward" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 100_000));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 94, 100_000));
    try std.testing.expectEqual(@as(usize, 24), controller.width());
    _ = controller.observe(sourceReadTestEvidence(&controller, 94, 100_000));
    try std.testing.expectEqual(@as(usize, 8), controller.width());
    const settled = controller.observe(sourceReadTestEvidence(&controller, 80, 100_000));
    try std.testing.expect(settled.settled);
    try std.testing.expectEqual(@as(usize, 12), controller.selectedWidth());
}

test "source read controller confirms boundary with three alternating pairs" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.Phase.pair_reference, controller.phase);
    for (0..3) |_| {
        _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
        _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    }
    try std.testing.expectEqual(@as(usize, 3), controller.pair_count);
    try std.testing.expect(controller.phase != .pair_reference and controller.phase != .pair_candidate);
}

test "source read controller confirms a borderline out-of-band candidate" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 96, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.Phase.pair_reference, controller.phase);
    try std.testing.expectEqual(@as(usize, 16), read_width_ladder[controller.pair_candidate_index]);
    try std.testing.expectEqual(@as(usize, 12), read_width_ladder[controller.pair_reference_index]);
}

test "source read controller prioritizes the newly measured boundary" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    const width8 = SourceReadWidthController.widthIndexAtMost(8);
    const width12 = SourceReadWidthController.widthIndexAtMost(12);
    const width16 = SourceReadWidthController.widthIndexAtMost(16);
    controller.rates[width12] = 98;
    controller.rates[width16] = 100;
    controller.peak_index = width16;
    controller.selected_index = width12;
    controller.current_index = width8;
    controller.phase = .downward;

    _ = controller.observe(sourceReadTestEvidence(&controller, 96, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.Phase.pair_reference, controller.phase);
    try std.testing.expectEqual(width8, controller.pair_candidate_index);
    try std.testing.expectEqual(width16, controller.pair_reference_index);
}

test "source read controller confirms a prior selection displaced by a new peak" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    const width12 = SourceReadWidthController.widthIndexAtMost(12);
    const width16 = SourceReadWidthController.widthIndexAtMost(16);
    const width24 = SourceReadWidthController.widthIndexAtMost(24);
    controller.rates[width12] = 98;
    controller.rates[width16] = 100;
    controller.peak_index = width16;
    controller.selected_index = width12;
    controller.current_index = width24;
    controller.phase = .upward;

    _ = controller.observe(sourceReadTestEvidence(&controller, 102, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.Phase.pair_reference, controller.phase);
    try std.testing.expectEqual(width12, controller.pair_candidate_index);
    try std.testing.expectEqual(width24, controller.pair_reference_index);
}

test "source read controller rolls back an unfinished boundary pair" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.Phase.pair_reference, controller.phase);
    const rollback = controller.rollbackTail();
    try std.testing.expect(rollback.settled);
    try std.testing.expectEqual(@as(usize, 12), rollback.width);
}

test "source read controller charges only unfinished pair intervals on restart" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        32,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    for (0..2) |_| {
        _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
        _ = controller.observe(sourceReadTestEvidence(&controller, 97, 1_000_000));
    }
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 1_000_000));
    try std.testing.expectEqual(SourceReadWidthController.Phase.pair_candidate, controller.phase);
    const remaining_cost = SourceReadWidthController.probeCost(controller.pair_candidate_index);
    try std.testing.expect(controller.restartFitsTail(remaining_cost * 4));
    try std.testing.expect(!controller.restartFitsTail(remaining_cost * 4 - 1));
}

test "source read controller refines downward when an upward tail no longer fits" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    _ = controller.observe(sourceReadTestEvidence(&controller, 100, 100_000));
    try std.testing.expectEqual(@as(usize, 16), controller.width());
    const downward = controller.observe(sourceReadTestEvidence(&controller, 120, 100));
    try std.testing.expectEqual(SourceReadWidthController.Phase.downward, controller.phase);
    try std.testing.expectEqual(@as(usize, 12), downward.width);
}

test "source read controller rejects dirty probe restarts at a short tail" {
    var controller = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    try std.testing.expect(controller.restartFitsTail(96));
    try std.testing.expect(!controller.restartFitsTail(95));
}

test "source read controller keeps fixed width and rolls back a short tail" {
    var fixed = SourceReadWidthController.init(.{ .fixed = 7 }, 64);
    try std.testing.expectEqual(@as(usize, 7), fixed.width());
    try std.testing.expect(fixed.currentDecision().settled);

    var adaptive = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    const tail = adaptive.observe(sourceReadTestEvidence(&adaptive, 100, 10));
    try std.testing.expect(tail.settled);
    try std.testing.expectEqual(@as(usize, 12), tail.width);
}

test "source read controller backs off before and after convergence" {
    var probing = SourceReadWidthController.init(
        .{ .adaptive = .{ .initial = 12, .maximum = 64 } },
        64,
    );
    try std.testing.expectEqual(@as(usize, 24), probing.blindGrow(10_000).?.width);
    const probe_backoff = probing.backoff();
    try std.testing.expect(probe_backoff.changed);
    try std.testing.expect(probe_backoff.settled);
    try std.testing.expectEqual(@as(usize, 12), probe_backoff.width);

    const settled_backoff = probing.backoff();
    try std.testing.expect(settled_backoff.changed);
    try std.testing.expect(settled_backoff.settled);
    try std.testing.expectEqual(@as(usize, 8), settled_backoff.width);

    var fixed = SourceReadWidthController.init(.{ .fixed = 7 }, 64);
    const fixed_backoff = fixed.backoff();
    try std.testing.expect(!fixed_backoff.changed);
    try std.testing.expectEqual(@as(usize, 7), fixed_backoff.width);
}

const VectoredReadStatsSource = struct {
    provider: VFS.ReadStatsProvider,
    initial: VFS.ReadStats,
    previous: VFS.ReadStats,
};

const SourceTelemetry = struct {
    retries: u64 = 0,
    transient_retries: u64 = 0,
    timeouts: u64 = 0,
    server_failures: u64 = 0,
    throttles: u64 = 0,

    fn hasBackpressure(self: SourceTelemetry) bool {
        return self.retries != 0 or self.transient_retries != 0 or
            self.timeouts != 0 or self.server_failures != 0 or
            self.throttles != 0;
    }
};

fn takeSourceTelemetry(sources: []VectoredReadStatsSource) SourceTelemetry {
    var result: SourceTelemetry = .{};
    for (sources) |*source| {
        const current = source.provider.snapshot();
        const delta = current.sub(source.previous);
        source.previous = current;
        result.retries +|= delta.retries;
        result.transient_retries +|= delta.transient_retries;
        result.timeouts +|= delta.timeouts;
        result.server_failures +|= delta.server_failures;
        result.throttles +|= delta.throttles;
    }
    return result;
}

test "one load-profile feedback cursor reports only new backpressure" {
    const FakeProvider = struct {
        stats: VFS.ReadStats = .{},

        fn snapshot(userdata: *anyopaque) VFS.ReadStats {
            const self: *@This() = @ptrCast(@alignCast(userdata));
            return self.stats;
        }
    };

    var fake: FakeProvider = .{};
    const provider: VFS.ReadStatsProvider = .{
        .userdata = &fake,
        .snapshotFn = FakeProvider.snapshot,
    };
    var sources = [_]VectoredReadStatsSource{.{
        .provider = provider,
        .initial = provider.snapshot(),
        .previous = provider.snapshot(),
    }};

    fake.stats.retries = 2;
    fake.stats.throttles = 1;
    const first = takeSourceTelemetry(&sources);
    try std.testing.expectEqual(@as(u64, 2), first.retries);
    try std.testing.expectEqual(@as(u64, 1), first.throttles);
    try std.testing.expect(first.hasBackpressure());

    const second = takeSourceTelemetry(&sources);
    try std.testing.expectEqualDeep(SourceTelemetry{}, second);
    try std.testing.expect(!second.hasBackpressure());
}

const SourceReadRuntime = struct {
    controller: SourceReadWidthController,
    worker_gate: *AdaptiveRequestGate,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    metrics: *VectoredLoadMetrics,
    next_read_admission: *std.atomic.Value(u64),
    scheduler: *FairVectoredReadScheduler,
    pinned_feasible_width: usize,
    read_stats_sources: []VectoredReadStatsSource,
    source_bootstrap_enabled: bool,
    source_response_observed: bool = false,
    probe_transition_pending: bool = false,
    probe_measuring: bool = false,
    scoring_pending: bool = false,
    blind_admissions: bool = false,
    pending_read_limit: usize = 1,
    pending_evidence: SourceReadWidthController.Evidence = undefined,
    last_blind_growth_ns: u64 = 0,
    persistent: bool = false,
    scheduler_idle: bool = false,
    reported_width: std.atomic.Value(usize) = .init(1),
    epoch_barrier_requested: std.atomic.Value(bool) = .init(false),
    epoch_barrier_done: std.Io.Event = .unset,
    control: std.Io.Event = .unset,
    done: std.Io.Event = .unset,

    fn takeRemoteTelemetry(self: *SourceReadRuntime) SourceTelemetry {
        return takeSourceTelemetry(self.read_stats_sources);
    }

    fn applyDecision(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
        force_probe: bool,
    ) void {
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width);
        self.reported_width.store(decision.width, .release);
        self.worker_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        if (!decision.settled and (decision.changed or force_probe)) {
            self.read_gate.setLimit(io, 0);
            self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
            self.pending_read_limit = limits.read;
            self.probe_transition_pending = true;
            self.probe_measuring = false;
            self.scoring_pending = false;
            self.blind_admissions = false;
            _ = self.activatePendingProbe(io);
        } else if (decision.settled) {
            self.read_gate.setLimit(io, limits.read);
            self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
            self.metrics.config_epoch.store(decision.generation, .release);
            self.probe_transition_pending = false;
            self.probe_measuring = false;
            self.scoring_pending = false;
            self.blind_admissions = false;
        }
    }

    fn activatePendingProbe(self: *SourceReadRuntime, io: std.Io) bool {
        if (!self.probe_transition_pending or
            self.read_gate.inUse(io) != 0) return false;
        // Advance the diagnostic baseline at a generation boundary.
        _ = self.takeRemoteTelemetry();
        const admission_start = self.next_read_admission.load(.acquire);
        self.metrics.prepareProbe(io, self.controller.generation, admission_start);
        self.probe_transition_pending = false;
        self.probe_measuring = true;
        self.read_gate.setLimit(io, self.pending_read_limit);
        return true;
    }

    fn applyBlindGrowth(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
    ) void {
        const limits: RequestGateLimits = .init(decision.width, self.pinned_feasible_width);
        self.reported_width.store(decision.width, .release);
        self.worker_gate.setLimit(io, limits.read);
        self.read_gate.setLimit(io, limits.read);
        self.request_gate.setLimit(io, limits.lifecycle);
        self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
        self.metrics.config_epoch.store(decision.generation, .release);
        self.probe_transition_pending = false;
        self.probe_measuring = false;
        self.scoring_pending = false;
        self.blind_admissions = true;
    }

    fn currentEvidence(
        self: *SourceReadRuntime,
        io: std.Io,
        remaining_full_jobs: usize,
    ) SourceReadWidthController.Evidence {
        const probe = self.metrics.snapshot(io);
        const now_ns: u64 = @intCast(@max(
            std.Io.Timestamp.now(io, .awake).nanoseconds,
            1,
        ));
        return .{
            .generation = probe.probe_epoch,
            .width = self.controller.width(),
            .completed_full_requests = @intCast(probe.probe_full_read_operations),
            // Do not charge a candidate for prior-generation DMA drain before
            // its first source admission can begin.
            .elapsed_ns = if (probe.probe_first_read_ns == 0)
                0
            else
                now_ns -| probe.probe_first_read_ns,
            .bytes = probe.probe_read_bytes,
            .exercised_width = probe.probe_peak_reads,
            .clean = true,
            .remaining_full_jobs = remaining_full_jobs,
        };
    }

    fn finalize(self: *SourceReadRuntime, io: std.Io) void {
        std.debug.assert(self.read_gate.inUse(io) == 0);
        _ = self.takeRemoteTelemetry();
        if (self.persistent) {
            self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
            return;
        }
        if (self.controller.phase != .settled) {
            const remaining_full_jobs = self.scheduler.snapshot(io).remaining_full_jobs;
            if (self.scoring_pending) {
                self.pending_evidence.remaining_full_jobs = remaining_full_jobs;
                _ = self.controller.observe(self.pending_evidence);
            } else if (self.probe_measuring) {
                const evidence = self.currentEvidence(io, remaining_full_jobs);
                if (evidence.scoreable()) _ = self.controller.observe(evidence);
            }
        }
        if (self.controller.phase != .settled) _ = self.controller.rollbackTail();
        self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
    }

    fn finishIdleMeasurement(self: *SourceReadRuntime, io: std.Io) void {
        _ = self.takeRemoteTelemetry();
        if (self.controller.phase != .settled) {
            if (self.scoring_pending) {
                self.pending_evidence.remaining_full_jobs = std.math.maxInt(usize);
                _ = self.controller.observe(self.pending_evidence);
            } else if (self.probe_measuring) {
                const evidence = self.currentEvidence(io, std.math.maxInt(usize));
                if (evidence.scoreable()) _ = self.controller.observe(evidence);
            }
        }
        self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
        self.probe_transition_pending = false;
        self.probe_measuring = false;
        self.scoring_pending = false;
        self.blind_admissions = false;
        self.reported_width.store(self.controller.selectedWidth(), .release);
    }

    fn epochBarrier(self: *SourceReadRuntime, io: std.Io) void {
        self.epoch_barrier_done.reset();
        self.epoch_barrier_requested.store(true, .release);
        self.control.set(io);
        self.epoch_barrier_done.waitUncancelable(io);
    }

    fn run(self: *SourceReadRuntime, io: std.Io) std.Io.Cancelable!void {
        const started: std.Io.Timestamp = .now(io, .awake);
        self.applyDecision(io, self.controller.currentDecision(), self.controller.adaptive);
        while (true) {
            self.control.waitTimeout(io, .{ .duration = .{
                .raw = .fromMilliseconds(if (self.source_response_observed) 25 else 10),
                .clock = .awake,
            } }) catch |err| switch (err) {
                error.Timeout => {},
                error.Canceled => return error.Canceled,
            };
            if (self.control.isSet()) self.control.reset();
            if (self.done.isSet()) {
                self.finalize(io);
                break;
            }

            const telemetry = self.takeRemoteTelemetry();
            if (telemetry.hasBackpressure()) {
                self.applyDecision(io, self.controller.backoff(), false);
                continue;
            }
            if (self.metrics.read_bytes.load(.acquire) != 0) self.source_response_observed = true;
            const scheduler_snapshot = self.scheduler.snapshot(io);
            const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));

            if (self.persistent) {
                const idle = !scheduler_snapshot.has_unscheduled and
                    self.metrics.pending_source_jobs.load(.acquire) == 0 and
                    self.read_gate.inUse(io) == 0;
                if (idle) {
                    if (!self.scheduler_idle) {
                        self.finishIdleMeasurement(io);
                        self.scheduler_idle = true;
                    }
                    if (self.epoch_barrier_requested.swap(false, .acq_rel)) {
                        _ = self.takeRemoteTelemetry();
                        self.epoch_barrier_done.set(io);
                    }
                    continue;
                }
                if (self.scheduler_idle) {
                    self.scheduler_idle = false;
                    self.applyDecision(
                        io,
                        self.controller.currentDecision(),
                        self.controller.phase != .settled,
                    );
                    continue;
                }
            }

            // Blind admissions deliberately overlap generations so a remote
            // source can ramp before its first response. Once any response is
            // visible, close the read gate and start a clean generation only
            // after every blind admission has returned.
            if (self.blind_admissions and self.source_response_observed) {
                self.controller.generation +|= 1;
                var decision = self.controller.currentDecision();
                decision.changed = true;
                self.applyDecision(io, decision, true);
                continue;
            }

            if (!self.source_response_observed) {
                if (now_ns -| self.last_blind_growth_ns >= 10 * std.time.ns_per_ms and
                    shouldBootstrapSource(
                        self.source_bootstrap_enabled,
                        false,
                        self.metrics.read_bytes.load(.acquire),
                        self.metrics.outstanding_requests.load(.acquire),
                        self.controller.width(),
                        scheduler_snapshot.has_unscheduled,
                    ))
                {
                    self.last_blind_growth_ns = now_ns;
                    if (self.controller.blindGrow(scheduler_snapshot.remaining_full_jobs)) |decision| {
                        self.applyBlindGrowth(io, decision);
                    }
                }
                continue;
            }

            if (self.controller.phase == .settled) continue;

            // Hold a completed score until all calls admitted at that width
            // have drained, keeping generation attribution unambiguous.
            if (self.scoring_pending) {
                if (self.read_gate.inUse(io) != 0) continue;
                _ = self.takeRemoteTelemetry();
                self.pending_evidence.remaining_full_jobs = scheduler_snapshot.remaining_full_jobs;
                const decision = self.controller.observe(self.pending_evidence);
                self.scoring_pending = false;
                self.applyDecision(io, decision, !decision.settled);
                continue;
            }

            if (self.probe_transition_pending) {
                if (!scheduler_snapshot.has_unscheduled and
                    self.metrics.pending_source_jobs.load(.acquire) == 0 and
                    self.read_gate.inUse(io) == 0)
                {
                    if (self.persistent) continue;
                    self.applyDecision(io, self.controller.rollbackTail(), false);
                    continue;
                }
                _ = self.activatePendingProbe(io);
                continue;
            }

            if (self.probe_measuring) {
                const evidence = self.currentEvidence(
                    io,
                    scheduler_snapshot.remaining_full_jobs,
                );
                if (evidence.scoreable()) {
                    // Freeze a complete interval, then drain admissions that
                    // raced with the snapshot. Their bytes are excluded.
                    self.read_gate.setLimit(io, 0);
                    self.pending_evidence = evidence;
                    self.probe_measuring = false;
                    self.scoring_pending = true;
                    continue;
                }
            }

            if (!scheduler_snapshot.has_unscheduled and
                self.metrics.pending_source_jobs.load(.acquire) == 0 and
                self.read_gate.inUse(io) == 0 and
                self.controller.phase != .settled and !self.persistent)
            {
                const rollback = self.controller.rollbackTail();
                self.applyDecision(io, rollback, false);
            }
        }
    }
};

fn shouldBootstrapSource(
    enabled: bool,
    response_observed: bool,
    read_bytes: u64,
    outstanding_requests: usize,
    read_limit: usize,
    has_unscheduled: bool,
) bool {
    return enabled and !response_observed and read_bytes == 0 and
        outstanding_requests >= read_limit and has_unscheduled;
}

fn effectiveSourceReadParallelism(configured: Parallelism, high_latency: bool) Parallelism {
    _ = high_latency;
    return configured;
}

fn effectiveSourceRequestSize(read_chunk_size: usize, dma_block_size: usize) !usize {
    if (read_chunk_size == 0 or read_chunk_size > max_load_read_request_size)
        return error.InvalidLoadProfile;
    const selected = @max(read_chunk_size, dma_block_size);
    if (selected > max_load_read_request_size) return error.InvalidLoadProfile;
    return selected;
}

pub const default_dma_benchmark_block_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
};

const dma_benchmark_repeats = 3;

const DmaBenchmarkPhase = enum {
    block,
    block_confirmation,
    aggregate,
};

const DmaBenchmarkSample = struct {
    phase: DmaBenchmarkPhase,
    device_index: usize,
    block_size: usize,
    parallelism: usize,
    repeat: usize = 0,
    bytes: u64,
    transfers: u64,
    elapsed_ns: u64,
    total_latency_ns: u64,

    pub fn bytesPerSecond(self: DmaBenchmarkSample) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed_ns));
    }

    pub fn averageLatencyNs(self: DmaBenchmarkSample) f64 {
        if (self.transfers == 0) return 0;
        return @as(f64, @floatFromInt(self.total_latency_ns)) /
            @as(f64, @floatFromInt(self.transfers));
    }
};

const DeviceDmaRecommendation = struct {
    device_index: usize,
    device_id: u32,
    dma_block_size: usize,
    dma_parallelism: usize,
    measured_bytes_per_second: f64,
    average_latency_ns: f64,
    windows: usize = 0,
};

/// Immutable DMA settings shared by every device participating in one load.
/// The slices are owned by the enclosing platform settings.
const DmaLoadConfig = struct {
    device_kind: []const u8,
    device_ids: []const u32,
    device_numa_nodes: []const ?usize,
    block_size: usize,
    max_in_flight_per_device: usize,
    max_mapped_bytes: usize,
};

fn requiredDmaWorkspaceBytes(config: DmaLoadConfig) !usize {
    const request_blocks = std.math.divCeil(
        usize,
        max_load_read_request_size,
        config.block_size,
    ) catch return error.InvalidDmaLoadConfig;
    const maximum_request_blocks = std.math.add(
        usize,
        request_blocks,
        config.device_ids.len - 1,
    ) catch return error.InvalidDmaLoadConfig;
    var required_blocks: usize = 0;
    if (config.device_numa_nodes[0] == null) {
        const feed_blocks = std.math.mul(
            usize,
            config.device_ids.len,
            config.max_in_flight_per_device,
        ) catch return error.InvalidDmaLoadConfig;
        required_blocks = @max(feed_blocks, maximum_request_blocks);
    } else {
        for (config.device_numa_nodes, 0..) |maybe_node, index| {
            const node = maybe_node.?;
            var seen = false;
            for (config.device_numa_nodes[0..index]) |previous| {
                if (previous.? == node) {
                    seen = true;
                    break;
                }
            }
            if (seen) continue;
            var device_count: usize = 0;
            for (config.device_numa_nodes) |candidate| {
                if (candidate.? == node) device_count += 1;
            }
            const feed_blocks = std.math.mul(
                usize,
                device_count,
                config.max_in_flight_per_device,
            ) catch return error.InvalidDmaLoadConfig;
            required_blocks = std.math.add(
                usize,
                required_blocks,
                @max(feed_blocks, maximum_request_blocks),
            ) catch return error.InvalidDmaLoadConfig;
        }
    }
    return std.math.mul(usize, required_blocks, config.block_size) catch
        error.InvalidDmaLoadConfig;
}

fn validateDmaLoadConfig(config: DmaLoadConfig) !void {
    if (config.device_kind.len == 0 or config.device_ids.len == 0 or
        config.device_ids.len > 64 or
        config.device_ids.len != config.device_numa_nodes.len or
        config.block_size == 0 or config.max_in_flight_per_device == 0 or
        config.max_in_flight_per_device > max_load_dma_parallelism or
        config.block_size > max_load_read_request_size or
        config.max_mapped_bytes < config.block_size)
        return error.InvalidDmaLoadConfig;
    var known_numa_nodes: usize = 0;
    for (config.device_numa_nodes) |maybe_node| {
        if (maybe_node) |node| {
            known_numa_nodes += 1;
            if (node >= DmaBenchmarkNumaAllocator.max_nodes)
                return error.InvalidDmaLoadConfig;
        }
    }
    if (known_numa_nodes != 0 and known_numa_nodes != config.device_numa_nodes.len)
        return error.InvalidDmaLoadConfig;
    if (known_numa_nodes != 0 and builtin.os.tag != .linux)
        return error.DmaBenchmarkNumaUnsupported;
    if (try requiredDmaWorkspaceBytes(config) > config.max_mapped_bytes)
        return error.InvalidDmaLoadConfig;
    for (config.device_ids, 0..) |id, index| {
        for (config.device_ids[0..index]) |previous| {
            if (id == previous) return error.InvalidDmaLoadConfig;
        }
    }
}

fn dupeDmaLoadConfig(allocator: std.mem.Allocator, config: DmaLoadConfig) !DmaLoadConfig {
    try validateDmaLoadConfig(config);
    const kind = try allocator.dupe(u8, config.device_kind);
    errdefer allocator.free(kind);
    const ids = try allocator.dupe(u32, config.device_ids);
    errdefer allocator.free(ids);
    const nodes = try allocator.dupe(?usize, config.device_numa_nodes);
    return .{
        .device_kind = kind,
        .device_ids = ids,
        .device_numa_nodes = nodes,
        .block_size = config.block_size,
        .max_in_flight_per_device = config.max_in_flight_per_device,
        .max_mapped_bytes = config.max_mapped_bytes,
    };
}

fn freeDmaLoadConfig(allocator: std.mem.Allocator, config: DmaLoadConfig) void {
    allocator.free(config.device_kind);
    allocator.free(config.device_ids);
    allocator.free(config.device_numa_nodes);
}

test "DMA load config validates uniform caps, topology, and workspace budget" {
    const valid: DmaLoadConfig = .{
        .device_kind = "test",
        .device_ids = &.{ 11, 12 },
        .device_numa_nodes = &.{ null, null },
        .block_size = 4 * 1024 * 1024,
        .max_in_flight_per_device = 8,
        .max_mapped_bytes = 64 * 1024 * 1024,
    };
    try validateDmaLoadConfig(valid);

    var invalid = valid;
    invalid.device_ids = &.{ 11, 11 };
    try std.testing.expectError(error.InvalidDmaLoadConfig, validateDmaLoadConfig(invalid));
    invalid = valid;
    invalid.device_numa_nodes = &.{ 0, null };
    try std.testing.expectError(error.InvalidDmaLoadConfig, validateDmaLoadConfig(invalid));
    invalid = valid;
    invalid.max_mapped_bytes -= 1;
    try std.testing.expectError(error.InvalidDmaLoadConfig, validateDmaLoadConfig(invalid));
}

/// Owned, reusable host-DMA workspace. A workspace may be borrowed by only one
/// load at a time; all registered arenas remain mapped until `deinit`.
const DmaPlatformSettings = struct {
    config: DmaLoadConfig,

    allocator: std.mem.Allocator,
    platform: *const Platform,
    workspace: DmaBenchmarkSourcePools,
    calibrated: bool,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        config: DmaLoadConfig,
        calibrated: bool,
    ) !DmaPlatformSettings {
        try validateDmaLoadConfig(config);
        const canonical_ids = try allocator.alloc(u32, config.device_ids.len);
        defer allocator.free(canonical_ids);
        const canonical_nodes = try allocator.alloc(?usize, config.device_numa_nodes.len);
        defer allocator.free(canonical_nodes);
        var canonical_len: usize = 0;
        for (platform.devices) |device| {
            const input_index = for (config.device_ids, 0..) |device_id, index| {
                if (device.id() == device_id) break index;
            } else continue;
            if (!std.mem.eql(u8, device.kind(), config.device_kind))
                return error.DmaDeviceKindMismatch;
            canonical_ids[canonical_len] = config.device_ids[input_index];
            canonical_nodes[canonical_len] = config.device_numa_nodes[input_index];
            canonical_len += 1;
        }
        if (canonical_len != config.device_ids.len) return error.DmaDeviceMismatch;
        const canonical_config: DmaLoadConfig = .{
            .device_kind = config.device_kind,
            .device_ids = canonical_ids,
            .device_numa_nodes = canonical_nodes,
            .block_size = config.block_size,
            .max_in_flight_per_device = config.max_in_flight_per_device,
            .max_mapped_bytes = config.max_mapped_bytes,
        };
        const owned_config = try dupeDmaLoadConfig(allocator, canonical_config);
        errdefer freeDmaLoadConfig(allocator, owned_config);
        const full_nodes = try allocator.alloc(?usize, platform.devices.len);
        defer allocator.free(full_nodes);
        @memset(full_nodes, null);
        for (owned_config.device_ids, owned_config.device_numa_nodes) |device_id, node| {
            const device_index = for (platform.devices, 0..) |device, index| {
                if (device.id() == device_id) break index;
            } else return error.DmaDeviceMismatch;
            if (!std.mem.eql(u8, platform.devices[device_index].kind(), owned_config.device_kind))
                return error.DmaDeviceKindMismatch;
            full_nodes[device_index] = node;
        }
        var workspace = try DmaBenchmarkSourcePools.init(
            allocator,
            io,
            platform,
            full_nodes,
            owned_config.max_mapped_bytes,
        );
        errdefer workspace.deinit();
        return .{
            .config = owned_config,
            .allocator = allocator,
            .platform = platform,
            .workspace = workspace,
            .calibrated = calibrated,
        };
    }

    fn adopt(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        config: DmaLoadConfig,
        workspace: DmaBenchmarkSourcePools,
    ) !DmaPlatformSettings {
        const owned_config = try dupeDmaLoadConfig(allocator, config);
        return .{
            .config = owned_config,
            .allocator = allocator,
            .platform = platform,
            .workspace = workspace,
            .calibrated = true,
        };
    }

    fn validateLoad(
        self: *DmaPlatformSettings,
        platform: *const Platform,
        device_ids: []const u32,
    ) !void {
        try validateDmaLoadConfig(self.config);
        if (platform != self.platform) return error.DmaPlatformMismatch;
        for (device_ids) |device_id| {
            const device = for (platform.devices) |device| {
                if (device.id() == device_id) break device;
            } else return error.DmaDeviceMismatch;
            for (self.config.device_ids) |configured_id| {
                if (configured_id == device_id) break;
            } else return error.DmaDeviceMismatch;
            if (!std.mem.eql(u8, device.kind(), self.config.device_kind))
                return error.DmaDeviceKindMismatch;
        }
        if (self.workspace.allocatedBytes() > self.config.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;
    }

    fn retainedMappedBytes(self: *const DmaPlatformSettings) usize {
        return self.workspace.allocatedBytes();
    }

    fn numaPoolCount(self: *const DmaPlatformSettings) usize {
        return self.workspace.pools.len;
    }

    fn deinit(self: *DmaPlatformSettings) void {
        const io = self.workspace.io;
        const mapped_bytes = self.workspace.allocatedBytes();
        const started = std.Io.Timestamp.now(io, .awake);
        self.workspace.deinit();
        const elapsed_ns: u64 = @intCast(@max(
            started.untilNow(io, .awake).nanoseconds,
            0,
        ));
        log.debug("DMA load workspace teardown: mapped={Bi:.2}, elapsed_ms={d:.3}", .{
            mapped_bytes,
            @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
        });
        freeDmaLoadConfig(self.allocator, self.config);
        self.* = undefined;
    }
};

const dma_platform_idle = 0;
const dma_platform_inspecting = 1;
const dma_platform_calibrating = 2;
const dma_platform_loading = 3;
const dma_platform_destroying = 4;

fn isDirectTransferPlatform(platform: *const Platform) bool {
    return platform.target == .cuda or platform.target == .rocm or
        platform.target == .oneapi;
}

fn dmaSettingsFromOpaque(ptr: *anyopaque) *DmaPlatformSettings {
    return @ptrCast(@alignCast(ptr));
}

fn destroyDmaPlatformSettings(settings: *DmaPlatformSettings) void {
    const allocator = settings.allocator;
    settings.deinit();
    allocator.destroy(settings);
}

fn beginPlatformDmaOperation(platform: *const Platform, operation: u8) !void {
    const mutable: *Platform = @constCast(platform);
    if (mutable._dma.operation.cmpxchgStrong(
        dma_platform_idle,
        operation,
        .acq_rel,
        .acquire,
    ) != null) return error.DmaWorkspaceBusy;
}

fn endPlatformDmaOperation(platform: *const Platform, operation: u8) void {
    const mutable: *Platform = @constCast(platform);
    const previous = mutable._dma.operation.swap(dma_platform_idle, .release);
    std.debug.assert(previous == operation);
}

pub fn initPlatformDma(
    platform: *Platform,
    allocator: std.mem.Allocator,
    io: std.Io,
    defaults: platform_mod.TransferConfig,
) !void {
    if (!isDirectTransferPlatform(platform)) return;
    if (platform.devices.len == 0) return error.NoFeasibleDmaBenchmarkTuple;

    const device_kind = platform.devices[0].kind();
    for (platform.devices[1..]) |device| {
        if (!std.mem.eql(u8, device_kind, device.kind()))
            return error.HeterogeneousDmaUnsupported;
    }

    const device_indices = try allocator.alloc(usize, platform.devices.len);
    defer allocator.free(device_indices);
    const device_ids = try allocator.alloc(u32, platform.devices.len);
    defer allocator.free(device_ids);
    for (device_indices, device_ids, 0..) |*device_index, *device_id, index| {
        device_index.* = index;
        device_id.* = platform.devices[index].id();
    }
    const numa_nodes = try resolveDmaNumaNodes(
        allocator,
        platform,
        device_indices,
        defaults.device_numa_nodes,
    );
    defer allocator.free(numa_nodes);

    var settings = try DmaPlatformSettings.init(
        allocator,
        io,
        platform,
        .{
            .device_kind = device_kind,
            .device_ids = device_ids,
            .device_numa_nodes = numa_nodes,
            .block_size = defaults.block_size,
            .max_in_flight_per_device = defaults.max_in_flight_per_device,
            .max_mapped_bytes = defaults.max_mapped_bytes,
        },
        false,
    );
    errdefer settings.deinit();
    const owned = try allocator.create(DmaPlatformSettings);
    owned.* = settings;
    std.debug.assert(platform._dma.settings.swap(owned, .release) == null);
}

pub fn platformTransferSettings(platform: *Platform) !?platform_mod.TransferSettings {
    if (!isDirectTransferPlatform(platform)) return null;
    try beginPlatformDmaOperation(platform, dma_platform_inspecting);
    defer endPlatformDmaOperation(platform, dma_platform_inspecting);
    const raw = platform._dma.settings.load(.acquire) orelse return null;
    const settings = dmaSettingsFromOpaque(raw);
    return .{
        .calibrated = settings.calibrated,
        .block_size = settings.config.block_size,
        .max_in_flight_per_device = settings.config.max_in_flight_per_device,
        .max_mapped_bytes = settings.config.max_mapped_bytes,
        .retained_mapped_bytes = settings.retainedMappedBytes(),
        .numa_pool_count = settings.numaPoolCount(),
    };
}

pub fn deinitPlatformDma(platform: *Platform) void {
    if (!isDirectTransferPlatform(platform)) return;
    if (platform._dma.operation.cmpxchgStrong(
        dma_platform_idle,
        dma_platform_destroying,
        .acq_rel,
        .acquire,
    ) != null) @panic("Platform.deinit called while DMA state is borrowed");
    const raw = platform._dma.settings.swap(null, .acq_rel);
    if (raw) |ptr| destroyDmaPlatformSettings(dmaSettingsFromOpaque(ptr));
}

fn acquirePlatformDmaSettings(
    platform: *const Platform,
    device_ids: []const u32,
) !*DmaPlatformSettings {
    try beginPlatformDmaOperation(platform, dma_platform_loading);
    errdefer endPlatformDmaOperation(platform, dma_platform_loading);
    const mutable: *Platform = @constCast(platform);
    const raw = mutable._dma.settings.load(.acquire) orelse
        return error.DmaResourcesRequired;
    const settings = dmaSettingsFromOpaque(raw);
    try settings.validateLoad(platform, device_ids);
    return settings;
}

fn releasePlatformDmaSettings(platform: *const Platform) void {
    endPlatformDmaOperation(platform, dma_platform_loading);
}

const DmaBenchmarkReport = struct {
    allocator: std.mem.Allocator,
    resources: DmaPlatformSettings,
    devices: []DeviceDmaRecommendation,
    samples: []DmaBenchmarkSample,
    elapsed_ns: u64,
    setup_ns: u64 = 0,
    sampling_ns: u64 = 0,
    device_allocator_warmup_ns: u64 = 0,
    source_registration_ns: u64 = 0,
    benchmark_setup_ns: u64 = 0,
    benchmark_overhead_ns: u64 = 0,
    source_cleanup_ns: u64 = 0,
    calibration_ns: u64 = 0,
    windows: usize = 0,

    fn deinit(self: *DmaBenchmarkReport) void {
        self.resources.deinit();
        self.deinitReport();
    }

    fn deinitReport(self: *DmaBenchmarkReport) void {
        self.allocator.free(self.devices);
        self.allocator.free(self.samples);
        self.* = undefined;
    }
};

pub const BenchTransferOptions = struct {
    block_sizes: []const usize = &default_dma_benchmark_block_sizes,
    /// Fixed per-device width used by the block screen and the loader.
    block_parallelism: usize = 8,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until every participating device completes the transfer target.
    duration_ns: u64 = 2 * std.time.ns_per_ms,
    minimum_transfers_per_device: u64 = 32,
    /// Borderline local decisions receive longer alternating paired windows.
    confirmation_duration_ns: u64 = 25 * std.time.ns_per_ms,
    confirmation_minimum_transfers_per_device: u64 = 256,
    confirmation_margin: f64 = 0.02,
    /// Prefer a smaller transaction once it supplies enough headroom over the
    /// source pipeline instead of maximizing isolated copy-engine throughput.
    block_selection_tolerance: f64 = 0.08,
    max_mapped_bytes: usize = 2 * 1024 * 1024 * 1024,
    /// Optional device-index to NUMA-node override. When absent, complete PJRT
    /// `numa_node` attributes select local pools; incomplete or unsupported
    /// topology falls back to one shared DmaMapped pool.
    device_numa_nodes: []const usize = &.{},
};

const DmaBenchmarkOpts = BenchTransferOptions;

const DmaBenchmarkNumaAllocator = struct {
    const max_nodes = 1024;
    const mpol_bind = 2;

    parent: std.mem.Allocator,
    node: ?usize,

    fn allocator(self: *DmaBenchmarkNumaAllocator) std.mem.Allocator {
        return .{
            .ptr = self,
            .vtable = &.{
                .alloc = alloc,
                .resize = resize,
                .remap = remap,
                .free = free,
            },
        };
    }

    fn alloc(ctx: *anyopaque, len: usize, alignment: std.mem.Alignment, ret_addr: usize) ?[*]u8 {
        const self: *DmaBenchmarkNumaAllocator = @ptrCast(@alignCast(ctx));
        const allocation = self.parent.rawAlloc(len, alignment, ret_addr) orelse return null;
        const node = self.node orelse return allocation;
        if (comptime builtin.os.tag != .linux) {
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }

        const word_bits = @bitSizeOf(usize);
        var node_mask: [max_nodes / word_bits]usize = @splat(0);
        node_mask[node / word_bits] = @as(usize, 1) << @intCast(node % word_bits);
        const rc = std.os.linux.syscall6(
            .mbind,
            @intFromPtr(allocation),
            len,
            mpol_bind,
            @intFromPtr(&node_mask),
            // Linux get_nodes() decrements maxnode before copying the mask;
            // raw callers include the same extra sentinel bit as libnuma.
            node + 2,
            0,
        );
        if (std.os.linux.errno(rc) != .SUCCESS) {
            log.err("unable to bind DMA benchmark allocation ({Bi:.2}) to NUMA node {d}: {s}", .{
                len,
                node,
                @tagName(std.os.linux.errno(rc)),
            });
            self.parent.rawFree(allocation[0..len], alignment, ret_addr);
            return null;
        }
        return allocation;
    }

    fn resize(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) bool {
        return false;
    }

    fn remap(_: *anyopaque, _: []u8, _: std.mem.Alignment, _: usize, _: usize) ?[*]u8 {
        return null;
    }

    fn free(ctx: *anyopaque, buf: []u8, alignment: std.mem.Alignment, ret_addr: usize) void {
        const self: *DmaBenchmarkNumaAllocator = @ptrCast(@alignCast(ctx));
        self.parent.rawFree(buf, alignment, ret_addr);
    }
};

const PjrtPinnedHostAllocation = struct {
    buffer: *pjrt.Buffer,
    api: *const pjrt.Api,
    data: []u8,

    fn init(memory: *const Memory, size: usize) !PjrtPinnedHostAllocation {
        const api = memory.platform.pjrt_api;
        const buffer = try memory.platform.pjrt_client.createUninitializedBuffer(api, .{
            .dims = &.{@intCast(size)},
            .element_type = .u8,
            .layout = .{
                .tiled = .{
                    .minor_to_major = &.{0},
                    .tile_dims = &.{},
                    .tile_dims_sizes = &.{},
                },
            },
            .dst = .{ .memory = memory.pjrt_memory },
        });
        errdefer buffer.deinit(api);
        if (!buffer.isOnCpu(api)) return error.PinnedHostMemoryNotHostVisible;

        // The writable pointer is borrowed from PJRT. Keep both the external
        // reference and its owning buffer alive for the arena's whole lifetime.
        try buffer.increaseExternalReferenceCount(api);
        errdefer buffer.decreaseExternalReferenceCount(api) catch {};
        const ptr: [*]u8 = @ptrCast(try buffer.opaqueDeviceMemoryDataPointer(api));
        return .{
            .buffer = buffer,
            .api = api,
            .data = ptr[0..size],
        };
    }

    fn deinit(self: PjrtPinnedHostAllocation) void {
        self.buffer.decreaseExternalReferenceCount(self.api) catch unreachable;
        self.buffer.deinit(self.api);
    }
};

const DmaBenchmarkSourceAllocation = union(enum) {
    dma_map: []u8,
    pjrt_host: PjrtPinnedHostAllocation,

    fn data(self: *const DmaBenchmarkSourceAllocation) []u8 {
        return switch (self.*) {
            .dma_map => |bytes| bytes,
            .pjrt_host => |allocation| allocation.data,
        };
    }

    fn deinit(
        self: DmaBenchmarkSourceAllocation,
        dma_map_allocator: std.mem.Allocator,
    ) void {
        switch (self) {
            .dma_map => |bytes| dma_map_allocator.free(bytes),
            .pjrt_host => |allocation| allocation.deinit(),
        }
    }
};

const DmaBenchmarkSourcePool = struct {
    numa_allocator: DmaBenchmarkNumaAllocator,
    dma_map_allocator: mem.DmaMapAllocator,
    pjrt_host_memory: ?*const Memory,
    allocations: std.ArrayListUnmanaged(DmaBenchmarkSourceAllocation) = .empty,
    source: []u8 = &.{},
};

const DmaBenchmarkSourcePools = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    pools: []DmaBenchmarkSourcePool,
    device_pool_indices: []usize,
    device_sources: [][]const u8,
    registration_ns: u64 = 0,
    max_mapped_bytes: usize,
    allocated_bytes: std.atomic.Value(usize) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        device_numa_nodes: []const ?usize,
        max_mapped_bytes: usize,
    ) !DmaBenchmarkSourcePools {
        var unique_nodes: std.AutoHashMapUnmanaged(usize, void) = .empty;
        defer unique_nodes.deinit(allocator);
        for (device_numa_nodes) |maybe_node| {
            if (maybe_node) |node| try unique_nodes.put(allocator, node, {});
        }
        const pool_count = @max(@as(usize, 1), unique_nodes.count());
        const pools = try allocator.alloc(DmaBenchmarkSourcePool, pool_count);
        errdefer allocator.free(pools);
        const device_pool_indices = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(device_pool_indices);
        const device_sources = try allocator.alloc([]const u8, platform.devices.len);
        errdefer allocator.free(device_sources);
        @memset(device_sources, &.{});
        if (unique_nodes.count() == 0) {
            const pool = &pools[0];
            pool.numa_allocator = .{ .parent = allocator, .node = null };
            pool.dma_map_allocator = .init(pool.numa_allocator.allocator(), platform);
            pool.pjrt_host_memory = if (platform.target == .rocm)
                platform.devices[0].memory(.host_pinned) orelse
                    return error.PinnedHostMemoryUnavailable
            else
                null;
            pool.allocations = .empty;
            pool.source = &.{};
            @memset(device_pool_indices, 0);
            return .{
                .allocator = allocator,
                .io = io,
                .pools = pools,
                .device_pool_indices = device_pool_indices,
                .device_sources = device_sources,
                .max_mapped_bytes = max_mapped_bytes,
            };
        }

        var nodes: std.ArrayListUnmanaged(usize) = .empty;
        defer nodes.deinit(allocator);
        for (device_numa_nodes, 0..) |maybe_node, device_index| {
            const node = maybe_node orelse {
                device_pool_indices[device_index] = 0;
                continue;
            };
            var pool_index: ?usize = null;
            for (nodes.items, 0..) |existing, index| {
                if (existing == node) {
                    pool_index = index;
                    break;
                }
            }
            if (pool_index == null) {
                pool_index = nodes.items.len;
                try nodes.append(allocator, node);
                const pool = &pools[pool_index.?];
                pool.numa_allocator = .{ .parent = allocator, .node = node };
                pool.dma_map_allocator = .init(pool.numa_allocator.allocator(), platform);
                pool.pjrt_host_memory = if (platform.target == .rocm)
                    platform.devices[device_index].memory(.host_pinned) orelse
                        return error.PinnedHostMemoryUnavailable
                else
                    null;
                pool.allocations = .empty;
                pool.source = &.{};
            }
            device_pool_indices[device_index] = pool_index.?;
        }
        std.debug.assert(nodes.items.len == pools.len);
        return .{
            .allocator = allocator,
            .io = io,
            .pools = pools,
            .device_pool_indices = device_pool_indices,
            .device_sources = device_sources,
            .max_mapped_bytes = max_mapped_bytes,
        };
    }

    fn deinit(self: *DmaBenchmarkSourcePools) void {
        for (self.pools) |*pool| {
            for (pool.allocations.items) |allocation| {
                allocation.deinit(pool.dma_map_allocator.allocator());
            }
            pool.allocations.deinit(self.allocator);
        }
        self.allocator.free(self.device_sources);
        self.allocator.free(self.device_pool_indices);
        self.allocator.free(self.pools);
        self.* = undefined;
    }

    fn sourceForDevice(self: *const DmaBenchmarkSourcePools, device_index: usize) []const u8 {
        const assigned = self.device_sources[device_index];
        if (assigned.len != 0) return assigned;
        return self.pools[self.device_pool_indices[device_index]].source;
    }

    fn cleanupSourceForDevice(
        self: *const DmaBenchmarkSourcePools,
        device_index: usize,
        minimum_len: usize,
    ) []const u8 {
        const assigned = self.device_sources[device_index];
        if (assigned.len >= minimum_len) return assigned;
        const pool = &self.pools[self.device_pool_indices[device_index]];
        var index = pool.allocations.items.len;
        while (index != 0) {
            index -= 1;
            const arena = pool.allocations.items[index].data();
            if (arena.len >= minimum_len) return arena;
        }
        unreachable;
    }

    fn growPool(self: *DmaBenchmarkSourcePools, pool_index: usize, required_bytes: usize) !void {
        const pool = &self.pools[pool_index];
        if (required_bytes <= pool.source.len) return;
        if (try std.math.add(usize, self.allocatedBytes(), required_bytes) > self.max_mapped_bytes)
            return error.DmaBenchmarkPinnedBudgetExceeded;
        const started = std.Io.Timestamp.now(self.io, .awake);
        defer self.registration_ns +|= @intCast(@max(started.untilNow(self.io, .awake).nanoseconds, 0));
        try self.allocatePool(pool_index, required_bytes);
    }

    fn allocatePool(self: *DmaBenchmarkSourcePools, pool_index: usize, required_bytes: usize) !void {
        const pool = &self.pools[pool_index];
        const started = std.Io.Timestamp.now(self.io, .awake);
        var allocation: DmaBenchmarkSourceAllocation = if (pool.pjrt_host_memory) |host_memory|
            .{ .pjrt_host = try .init(host_memory, required_bytes) }
        else blk: {
            const dma_map_allocator = pool.dma_map_allocator.allocator();
            break :blk .{ .dma_map = try dma_map_allocator.alignedAlloc(
                u8,
                .fromByteUnits(std.heap.page_size_min),
                required_bytes,
            ) };
        };
        errdefer allocation.deinit(pool.dma_map_allocator.allocator());
        const mapped_at = std.Io.Timestamp.now(self.io, .awake);
        try pool.allocations.append(self.allocator, allocation);
        const replacement = allocation.data();
        _ = self.allocated_bytes.fetchAdd(replacement.len, .release);
        pool.source = replacement;
        const finished_at = std.Io.Timestamp.now(self.io, .awake);
        const map_ns: u64 = if (pool.pjrt_host_memory != null)
            0
        else
            @intCast(@max(started.durationTo(mapped_at).nanoseconds, 0));
        const elapsed_ns: u64 = @intCast(@max(started.durationTo(finished_at).nanoseconds, 0));
        const allocation_kind = if (pool.pjrt_host_memory != null) "pjrt_host" else "dma_map";
        if (pool.numa_allocator.node) |node| {
            log.info("DMA mapped arena numa_node={d} allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                node,
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        } else {
            log.info("DMA mapped arena numa_node=single allocator={s} address=0x{x} size={Bi:.2} allocation_ms={d:.3} dma_map_ms={d:.3}", .{
                allocation_kind,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                @as(f64, @floatFromInt(elapsed_ns)) / std.time.ns_per_ms,
                @as(f64, @floatFromInt(map_ns)) / std.time.ns_per_ms,
            });
        }
    }

    fn allocatedBytes(self: *const DmaBenchmarkSourcePools) usize {
        return self.allocated_bytes.load(.acquire);
    }

    /// Ensures every NUMA pool can feed its calibrated devices and hold one
    /// complete fixed-size source request. Independent nodes register their
    /// missing slabs concurrently; existing retained arenas are reused first.
    fn ensureLoadBlockReserves(
        self: *DmaBenchmarkSourcePools,
        block_size: usize,
        maximum_writer_groups: usize,
        calibrated_reserves: []const usize,
    ) !void {
        if (block_size == 0 or maximum_writer_groups == 0 or
            calibrated_reserves.len != self.pools.len)
            return error.InvalidDmaLoadConfig;
        const base_request_blocks = std.math.divCeil(
            usize,
            max_load_read_request_size,
            block_size,
        ) catch return error.InvalidDmaLoadConfig;
        const request_blocks = std.math.add(
            usize,
            base_request_blocks,
            maximum_writer_groups - 1,
        ) catch return error.InvalidDmaLoadConfig;
        return self.ensureBlockReserves(block_size, request_blocks, calibrated_reserves);
    }

    /// Grows each retained NUMA arena for the exact largest request produced
    /// by the completed scheduler. The caller has already accounted for every
    /// dispatch/writer boundary, so this function must not add another
    /// fixed-request or writer-group estimate.
    fn ensureExactLoadBlockReserves(
        self: *DmaBenchmarkSourcePools,
        block_size: usize,
        maximum_blocks_per_job: usize,
        calibrated_reserves: []const usize,
    ) !void {
        if (block_size == 0 or maximum_blocks_per_job == 0 or calibrated_reserves.len != self.pools.len)
            return error.InvalidDmaLoadConfig;
        return self.ensureBlockReserves(block_size, maximum_blocks_per_job, calibrated_reserves);
    }

    fn ensureBlockReserves(
        self: *DmaBenchmarkSourcePools,
        block_size: usize,
        request_blocks: usize,
        calibrated_reserves: []const usize,
    ) !void {
        const missing_bytes = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(missing_bytes);
        var missing_total: usize = 0;
        for (self.pools, calibrated_reserves, missing_bytes) |pool, reserve, *missing| {
            var usable_blocks: usize = 0;
            for (pool.allocations.items) |arena| {
                usable_blocks = std.math.add(
                    usize,
                    usable_blocks,
                    arena.data().len / block_size,
                ) catch return error.DmaMappedBudgetExceeded;
            }
            const required_blocks = @max(reserve, request_blocks);
            const missing_blocks = required_blocks -| usable_blocks;
            missing.* = std.math.mul(usize, missing_blocks, block_size) catch
                return error.DmaMappedBudgetExceeded;
            missing_total = std.math.add(usize, missing_total, missing.*) catch
                return error.DmaMappedBudgetExceeded;
        }
        if (missing_total == 0) return;
        const mapped_after_growth = std.math.add(
            usize,
            self.allocatedBytes(),
            missing_total,
        ) catch return error.DmaMappedBudgetExceeded;
        if (mapped_after_growth > self.max_mapped_bytes)
            return error.DmaMappedBudgetExceeded;

        const Worker = struct {
            pools: *DmaBenchmarkSourcePools,
            pool_index: usize,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                worker.pools.allocatePool(worker.pool_index, worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(
                        0,
                        @intFromError(err),
                        .release,
                        .monotonic,
                    );
                };
            }
        };
        var first_error: std.atomic.Value(u16) = .init(0);
        var group: std.Io.Group = .init;
        var group_error: ?anyerror = null;
        const registration_started = std.Io.Timestamp.now(self.io, .awake);
        for (missing_bytes, 0..) |bytes, pool_index| {
            if (bytes == 0) continue;
            group.concurrent(self.io, Worker.run, .{Worker{
                .pools = self,
                .pool_index = pool_index,
                .bytes = bytes,
                .first_error = &first_error,
            }}) catch |err| {
                group_error = err;
                break;
            };
        }
        group.await(self.io) catch |err| if (group_error == null) {
            group_error = err;
        };
        self.registration_ns +|= @intCast(@max(
            registration_started.untilNow(self.io, .awake).nanoseconds,
            0,
        ));
        if (group_error) |err| return err;
        const error_code = first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
    }

    fn blockPoolArenaProvider(self: *DmaBenchmarkSourcePools) mem.DmaBlockPool.ArenaProvider {
        return .{
            .context = self,
            .node_count = self.pools.len,
            .arenaCountFn = struct {
                fn call(context: *anyopaque, node_index: usize) usize {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.pools[node_index].allocations.items.len;
                }
            }.call,
            .arenaFn = struct {
                fn call(context: *anyopaque, node_index: usize, arena_index: usize) []u8 {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.pools[node_index].allocations.items[arena_index].data();
                }
            }.call,
            .allocateFn = struct {
                fn call(context: *anyopaque, node_index: usize, len: usize) ![]u8 {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    if (try std.math.add(usize, pools.allocatedBytes(), len) > pools.max_mapped_bytes)
                        return error.DmaMappedBudgetExceeded;
                    const started = std.Io.Timestamp.now(pools.io, .awake);
                    try pools.allocatePool(node_index, len);
                    pools.registration_ns +|= @intCast(@max(
                        started.untilNow(pools.io, .awake).nanoseconds,
                        0,
                    ));
                    return pools.pools[node_index].allocations.items[
                        pools.pools[node_index].allocations.items.len - 1
                    ].data();
                }
            }.call,
            .mappedBytesFn = struct {
                fn call(context: *anyopaque) usize {
                    const pools: *DmaBenchmarkSourcePools = @ptrCast(@alignCast(context));
                    return pools.allocatedBytes();
                }
            }.call,
        };
    }

    fn prepareAggregateSources(
        self: *DmaBenchmarkSourcePools,
        recommendations: []const DeviceDmaRecommendation,
        max_mapped_bytes: usize,
    ) !void {
        const missing = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(missing);
        @memset(missing, 0);
        const offsets = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(offsets);
        @memset(offsets, 0);

        // First use the largest current arena in each pool. Earlier arenas are
        // retained for manager teardown, while the aggregate path gets one
        // contiguous, disjoint ring per device.
        for (recommendations) |recommendation| {
            const bytes = try std.math.mul(
                usize,
                recommendation.dma_block_size,
                recommendation.dma_parallelism,
            );
            const pool_index = self.device_pool_indices[recommendation.device_index];
            const pool = &self.pools[pool_index];
            if (bytes <= pool.source.len -| offsets[pool_index]) {
                const end = offsets[pool_index] + bytes;
                self.device_sources[recommendation.device_index] =
                    pool.source[offsets[pool_index]..end];
                offsets[pool_index] = end;
            } else {
                missing[pool_index] = try std.math.add(usize, missing[pool_index], bytes);
            }
        }
        var missing_total: usize = 0;
        for (missing) |bytes| missing_total = try std.math.add(usize, missing_total, bytes);
        if (try std.math.add(usize, self.allocatedBytes(), missing_total) > max_mapped_bytes)
            return error.DmaBenchmarkPinnedBudgetExceeded;

        const Worker = struct {
            pools: *DmaBenchmarkSourcePools,
            pool_index: usize,
            bytes: usize,
            first_error: *std.atomic.Value(u16),

            fn run(worker: @This()) void {
                worker.pools.allocatePool(worker.pool_index, worker.bytes) catch |err| {
                    _ = worker.first_error.cmpxchgStrong(
                        0,
                        @intFromError(err),
                        .release,
                        .monotonic,
                    );
                };
            }
        };
        var first_error: std.atomic.Value(u16) = .init(0);
        var group: std.Io.Group = .init;
        const registration_started = std.Io.Timestamp.now(self.io, .awake);
        for (missing, 0..) |bytes, pool_index| {
            if (bytes == 0) continue;
            try group.concurrent(self.io, Worker.run, .{Worker{
                .pools = self,
                .pool_index = pool_index,
                .bytes = bytes,
                .first_error = &first_error,
            }});
            offsets[pool_index] = 0;
        }
        try group.await(self.io);
        self.registration_ns +|= @intCast(@max(
            registration_started.untilNow(self.io, .awake).nanoseconds,
            0,
        ));
        const error_code = first_error.load(.acquire);
        if (error_code != 0) return @errorFromInt(error_code);
        for (recommendations) |recommendation| {
            if (self.device_sources[recommendation.device_index].len != 0) continue;
            const bytes = try std.math.mul(
                usize,
                recommendation.dma_block_size,
                recommendation.dma_parallelism,
            );
            const pool_index = self.device_pool_indices[recommendation.device_index];
            const pool = &self.pools[pool_index];
            const end = offsets[pool_index] + bytes;
            self.device_sources[recommendation.device_index] =
                pool.source[offsets[pool_index]..end];
            offsets[pool_index] = end;
        }
    }
};

const DmaBenchmarkDistribution = struct {
    block_size: usize,

    fn init(
        _: std.mem.Allocator,
        block_size: usize,
    ) !DmaBenchmarkDistribution {
        return .{ .block_size = block_size };
    }

    fn deinit(self: *DmaBenchmarkDistribution, _: std.mem.Allocator) void {
        self.* = undefined;
    }

    fn at(self: DmaBenchmarkDistribution, _: u64) usize {
        return self.block_size;
    }
};

const DmaBenchmarkRunMetrics = struct {
    bytes: u64,
    transfers: u64,
    total_latency_ns: u64,
    elapsed_ns: u64,

    fn bytesPerSecond(self: DmaBenchmarkRunMetrics) f64 {
        if (self.elapsed_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.bytes)) * std.time.ns_per_s /
            @as(f64, @floatFromInt(self.elapsed_ns));
    }

    fn averageLatencyNs(self: DmaBenchmarkRunMetrics) f64 {
        if (self.transfers == 0) return 0;
        return @as(f64, @floatFromInt(self.total_latency_ns)) /
            @as(f64, @floatFromInt(self.transfers));
    }
};

const DmaBenchmarkAtomicMetrics = struct {
    bytes: std.atomic.Value(u64) = .init(0),
    transfers: std.atomic.Value(u64) = .init(0),
    total_latency_ns: std.atomic.Value(u64) = .init(0),
};

const DmaBenchmarkManager = struct {
    manager: *pjrt.AsyncHostToDeviceTransferManager,
    buffer: *pjrt.Buffer,
};

const ReusableDmaBenchmarkCohort = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    device_index: usize,
    block_size: usize,
    distribution: DmaBenchmarkDistribution,
    managers: std.ArrayListUnmanaged(DmaBenchmarkManager) = .empty,
    warmed_managers: usize = 0,
    next_transfer: std.atomic.Value(u64) = .init(0),
    first_error: std.atomic.Value(u16) = .init(0),

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        device_index: usize,
        block_size: usize,
    ) !ReusableDmaBenchmarkCohort {
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .device_index = device_index,
            .block_size = block_size,
            .distribution = try .init(allocator, block_size),
        };
    }

    fn recordError(self: *ReusableDmaBenchmarkCohort, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn transfer(
        self: *ReusableDmaBenchmarkCohort,
        source: []const u8,
        slot: usize,
        metrics: ?*DmaBenchmarkAtomicMetrics,
    ) void {
        const transfer_index = self.next_transfer.fetchAdd(1, .monotonic);
        const len = self.distribution.at(transfer_index);
        const source_offset = slot * self.block_size;
        const started = std.Io.Timestamp.now(self.io, .awake);
        const event = self.managers.items[slot].manager.transferData(
            self.platform.pjrt_api,
            0,
            source[source_offset..][0..len],
            0,
            false,
        ) catch |err| {
            self.recordError(err);
            return;
        };
        event.await(self.platform.pjrt_api, self.io) catch |err| {
            event.deinit(self.platform.pjrt_api);
            self.recordError(err);
            return;
        };
        event.deinit(self.platform.pjrt_api);
        if (metrics) |output| {
            const elapsed_ns: u64 = @intCast(@max(started.untilNow(self.io, .awake).nanoseconds, 0));
            _ = output.bytes.fetchAdd(@intCast(len), .monotonic);
            _ = output.transfers.fetchAdd(1, .monotonic);
            _ = output.total_latency_ns.fetchAdd(elapsed_ns, .monotonic);
        }
    }

    fn ensureReady(self: *ReusableDmaBenchmarkCohort, source: []const u8, parallelism: usize) !void {
        const required_bytes = std.math.mul(usize, self.block_size, parallelism) catch return error.OutOfMemory;
        if (required_bytes > source.len) return error.DmaBenchmarkPinnedBudgetExceeded;
        var dims = [_]i64{@intCast(self.block_size)};
        const shape_spec: pjrt.ShapeSpec = .init(&dims, .u8);
        const memory = self.platform.devices[self.device_index].memory(.default).?;
        while (self.managers.items.len < parallelism) {
            const manager = try self.platform.pjrt_client.createBuffersForAsyncHostToDevice(self.platform.pjrt_api, .{
                .shape_specs = &.{shape_spec},
                .memory = memory.pjrt_memory,
            });
            errdefer manager.deinit(self.platform.pjrt_api);
            const buffer = try manager.retrieveBuffer(self.platform.pjrt_api, 0);
            try self.managers.append(self.allocator, .{ .manager = manager, .buffer = buffer });
        }
        while (self.warmed_managers < parallelism) : (self.warmed_managers += 1) {
            const slot = self.warmed_managers;
            self.transfer(source, slot, null);
            self.transfer(source, slot, null);
            const error_code = self.first_error.load(.acquire);
            if (error_code != 0) return @errorFromInt(error_code);
        }
    }

    fn deinit(self: *ReusableDmaBenchmarkCohort, source: []const u8) void {
        for (self.managers.items) |manager| {
            const event = manager.manager.transferData(
                self.platform.pjrt_api,
                0,
                source[0..self.block_size],
                0,
                true,
            ) catch null;
            if (event) |done| {
                done.await(self.platform.pjrt_api, self.io) catch {};
                done.deinit(self.platform.pjrt_api);
            }
            manager.manager.deinit(self.platform.pjrt_api);
            manager.buffer.deinit(self.platform.pjrt_api);
        }
        self.managers.deinit(self.allocator);
        self.distribution.deinit(self.allocator);
        self.* = undefined;
    }
};

const ReusableDmaBenchmarkLane = struct {
    cohort: *ReusableDmaBenchmarkCohort,
    source: []const u8,
    parallelism: usize,
};

fn prepareReusableDmaBenchmarkLanes(io: std.Io, lanes: []const ReusableDmaBenchmarkLane) !void {
    const SetupWorker = struct {
        lane: ReusableDmaBenchmarkLane,
        first_error: *std.atomic.Value(u16),

        fn run(self: @This()) void {
            self.lane.cohort.ensureReady(self.lane.source, self.lane.parallelism) catch |err| {
                _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
            };
        }
    };
    var setup_error: std.atomic.Value(u16) = .init(0);
    var setup_group: std.Io.Group = .init;
    for (lanes) |lane| try setup_group.concurrent(io, SetupWorker.run, .{SetupWorker{
        .lane = lane,
        .first_error = &setup_error,
    }});
    try setup_group.await(io);
    const setup_error_code = setup_error.load(.acquire);
    if (setup_error_code != 0) return @errorFromInt(setup_error_code);
}

fn dmaBenchmarkWindowComplete(
    elapsed_ns: u64,
    minimum_duration_ns: u64,
    completed_transfers: u64,
    minimum_transfers: u64,
) bool {
    return elapsed_ns >= minimum_duration_ns and completed_transfers >= minimum_transfers;
}

fn runReusableDmaBenchmarkWindow(
    allocator: std.mem.Allocator,
    io: std.Io,
    lanes: []const ReusableDmaBenchmarkLane,
    duration_ns: u64,
    minimum_transfers_per_device: u64,
    setup_ns: *u64,
) ![]DmaBenchmarkRunMetrics {
    std.debug.assert(lanes.len > 0);
    const atomic_metrics = try allocator.alloc(DmaBenchmarkAtomicMetrics, lanes.len);
    defer allocator.free(atomic_metrics);
    for (atomic_metrics) |*metrics| metrics.* = .{};
    const setup_started = std.Io.Timestamp.now(io, .awake);
    try prepareReusableDmaBenchmarkLanes(io, lanes);
    setup_ns.* +|= @intCast(@max(setup_started.untilNow(io, .awake).nanoseconds, 0));

    const Worker = struct {
        lane: ReusableDmaBenchmarkLane,
        slot: usize,
        metrics: *DmaBenchmarkAtomicMetrics,
        ready: *std.atomic.Value(usize),
        start: *std.Io.Event,
        stop: *std.atomic.Value(bool),

        fn run(self: @This()) void {
            _ = self.ready.fetchAdd(1, .release);
            self.start.waitUncancelable(self.lane.cohort.io);
            while (!self.stop.load(.acquire)) {
                self.lane.cohort.transfer(self.lane.source, self.slot, self.metrics);
                if (self.lane.cohort.first_error.load(.acquire) != 0) return;
            }
        }
    };

    var ready: std.atomic.Value(usize) = .init(0);
    var start: std.Io.Event = .unset;
    var stop: std.atomic.Value(bool) = .init(false);
    var group: std.Io.Group = .init;
    var worker_count: usize = 0;
    for (lanes, atomic_metrics) |lane, *metrics| {
        for (0..lane.parallelism) |slot| {
            try group.concurrent(io, Worker.run, .{Worker{
                .lane = lane,
                .slot = slot,
                .metrics = metrics,
                .ready = &ready,
                .start = &start,
                .stop = &stop,
            }});
            worker_count += 1;
        }
    }
    while (ready.load(.acquire) != worker_count) try io.sleep(.fromMilliseconds(1), .awake);
    const measured_at = std.Io.Timestamp.now(io, .awake);
    start.set(io);
    while (true) {
        const elapsed_ns: u64 = @intCast(@max(measured_at.untilNow(io, .awake).nanoseconds, 0));
        var completed_transfers: u64 = std.math.maxInt(u64);
        for (lanes, atomic_metrics) |lane, *metrics| {
            const error_code = lane.cohort.first_error.load(.acquire);
            if (error_code != 0) {
                stop.store(true, .release);
                try group.await(io);
                return @errorFromInt(error_code);
            }
            completed_transfers = @min(completed_transfers, metrics.transfers.load(.acquire));
        }
        if (dmaBenchmarkWindowComplete(
            elapsed_ns,
            duration_ns,
            completed_transfers,
            minimum_transfers_per_device,
        )) break;
        try io.sleep(.fromMilliseconds(1), .awake);
    }
    stop.store(true, .release);
    try group.await(io);
    const elapsed_ns: u64 = @intCast(@max(measured_at.untilNow(io, .awake).nanoseconds, 1));

    const metrics = try allocator.alloc(DmaBenchmarkRunMetrics, lanes.len);
    for (lanes, atomic_metrics, metrics) |lane, atomic, *result| {
        const error_code = lane.cohort.first_error.load(.acquire);
        if (error_code != 0) {
            allocator.free(metrics);
            return @errorFromInt(error_code);
        }
        result.* = .{
            .bytes = atomic.bytes.load(.acquire),
            .transfers = atomic.transfers.load(.acquire),
            .total_latency_ns = atomic.total_latency_ns.load(.acquire),
            .elapsed_ns = elapsed_ns,
        };
    }
    return metrics;
}

const ReusableDmaBenchmarkSession = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    samples: *std.ArrayListUnmanaged(DmaBenchmarkSample),
    cohorts: std.ArrayListUnmanaged(*ReusableDmaBenchmarkCohort) = .empty,
    setup_ns: u64 = 0,
    sampling_ns: u64 = 0,
    windows: usize = 0,

    fn createCohort(
        self: *ReusableDmaBenchmarkSession,
        device_index: usize,
        block_size: usize,
    ) !*ReusableDmaBenchmarkCohort {
        const cohort = try self.allocator.create(ReusableDmaBenchmarkCohort);
        errdefer self.allocator.destroy(cohort);
        cohort.* = try .init(
            self.allocator,
            self.io,
            self.platform,
            device_index,
            block_size,
        );
        errdefer cohort.distribution.deinit(self.allocator);
        try self.cohorts.append(self.allocator, cohort);
        return cohort;
    }

    fn measure(
        self: *ReusableDmaBenchmarkSession,
        phase: DmaBenchmarkPhase,
        lanes: []const ReusableDmaBenchmarkLane,
        duration_ns: u64,
        minimum_transfers_per_device: u64,
        repeat: usize,
    ) ![]DmaBenchmarkRunMetrics {
        const metrics = try runReusableDmaBenchmarkWindow(
            self.allocator,
            self.io,
            lanes,
            duration_ns,
            minimum_transfers_per_device,
            &self.setup_ns,
        );
        errdefer self.allocator.free(metrics);
        self.sampling_ns +|= metrics[0].elapsed_ns;
        self.windows += 1;
        for (lanes, metrics) |lane, metric| {
            try self.samples.append(self.allocator, .{
                .phase = phase,
                .device_index = lane.cohort.device_index,
                .block_size = lane.cohort.block_size,
                .parallelism = lane.parallelism,
                .repeat = repeat,
                .bytes = metric.bytes,
                .transfers = metric.transfers,
                .elapsed_ns = metric.elapsed_ns,
                .total_latency_ns = metric.total_latency_ns,
            });
        }
        return metrics;
    }

    fn deinit(self: *ReusableDmaBenchmarkSession, source_pools: *const DmaBenchmarkSourcePools) void {
        for (self.cohorts.items) |cohort| {
            cohort.deinit(source_pools.cleanupSourceForDevice(
                cohort.device_index,
                cohort.block_size,
            ));
            self.allocator.destroy(cohort);
        }
        self.cohorts.deinit(self.allocator);
        self.* = undefined;
    }
};

fn finishDmaBenchmarkReport(
    result: *DmaBenchmarkReport,
    io: std.Io,
    benchmark_started: std.Io.Timestamp,
    device_allocator_warmup_ns: u64,
    session: *ReusableDmaBenchmarkSession,
    source_pools: *DmaBenchmarkSourcePools,
) void {
    const sampling_ns = session.sampling_ns;
    const benchmark_setup_ns = session.setup_ns;
    const windows = session.windows;
    const source_registration_ns = source_pools.registration_ns;
    session.deinit(source_pools);
    const elapsed_ns: u64 = @intCast(@max(
        benchmark_started.untilNow(io, .awake).nanoseconds,
        0,
    ));
    const calibration_ns = elapsed_ns -| device_allocator_warmup_ns -|
        source_registration_ns;
    result.elapsed_ns = elapsed_ns;
    result.setup_ns = device_allocator_warmup_ns +| source_registration_ns +| benchmark_setup_ns;
    result.sampling_ns = sampling_ns;
    result.device_allocator_warmup_ns = device_allocator_warmup_ns;
    result.source_registration_ns = source_registration_ns;
    result.benchmark_setup_ns = benchmark_setup_ns;
    result.benchmark_overhead_ns = calibration_ns -| sampling_ns -| benchmark_setup_ns;
    // Registered source arenas are now part of result.resources. Their
    // teardown is intentionally outside benchmark timing.
    result.source_cleanup_ns = 0;
    result.calibration_ns = calibration_ns;
    result.windows = windows;
}

pub const max_load_read_parallelism: usize = 128;
pub const max_load_dma_parallelism: usize = 32;
pub const max_load_read_request_size: usize = 32 * 1024 * 1024;
const max_load_positional_iovecs: usize = if (@TypeOf(std.posix.IOV_MAX) == void)
    64
else
    std.posix.IOV_MAX;

pub const Parallelism = union(enum) {
    adaptive: Adaptive,
    fixed: usize,

    pub const Adaptive = struct {
        initial: usize,
        maximum: usize,
    };

    fn initial(self: Parallelism) usize {
        return switch (self) {
            .adaptive => |adaptive| adaptive.initial,
            .fixed => |fixed| fixed,
        };
    }

    fn maximum(self: Parallelism) usize {
        return switch (self) {
            .adaptive => |adaptive| adaptive.maximum,
            .fixed => |fixed| fixed,
        };
    }

    fn isAdaptive(self: Parallelism) bool {
        return switch (self) {
            .adaptive => true,
            .fixed => false,
        };
    }
};

pub const Loader = struct {
    allocator: std.mem.Allocator,
    backend: Backend,

    const Backend = union(enum) {
        direct: *DirectLoader,
        buffered: *BufferedLoader,
    };

    pub const Opts = struct {
        pub const auto: Opts = .{};

        /// Concurrent positional source requests.
        read_parallelism: Parallelism = .{ .adaptive = .{ .initial = 12, .maximum = max_load_read_parallelism } },
        /// Model-wide source tuning prepared from the VFS path. The default is
        /// generic for callers that do not have an explicit VFS profile.
        load_profile: VFS.LoadProfile = .default,
        shardings: []const Sharding = &.{},
        progress: ?*std.Progress.Node = null,
    };

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
            .backend = if (isDirectTransferPlatform(platform))
                .{ .direct = try DirectLoader.create(allocator, io, platform, store, opts) }
            else
                .{ .buffered = try BufferedLoader.create(allocator, io, platform, store, opts) },
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
        return switch (self.backend) {
            .direct => |direct| direct.load(ModelType, model, buffers),
            .buffered => |buffered| buffered.load(ModelType, model, buffers),
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
        return switch (self.backend) {
            .direct => |direct| direct.loadExecute(tensor, output, exe),
            .buffered => |buffered| buffered.loadExecute(tensor, output, exe),
        };
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
            .direct => |direct| direct.bytes_loaded.load(.acquire),
            .buffered => |buffered| buffered.bytes_loaded.load(.acquire),
        };
    }

    pub fn deinit(self: *Loader) void {
        switch (self.backend) {
            .direct => |direct| direct.destroy(),
            .buffered => |buffered| buffered.destroy(),
        }
        self.* = undefined;
    }
};

fn validateLoaderOpts(opts: Loader.Opts) !void {
    _ = try effectiveSourceRequestSize(opts.load_profile.read_chunk_size, 0);
    const initial = opts.read_parallelism.initial();
    const maximum = opts.read_parallelism.maximum();
    if (initial == 0 or maximum < initial or maximum > max_load_read_parallelism)
        return error.InvalidLoadParallelism;
}

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
        if (expected_device.id != actual_device.id or
            !std.mem.eql(Placement.Slice1d, expected_slices.constSlice(), actual_slices.constSlice()))
        {
            return error.ExecutablePlacementMismatch;
        }
    }
}

const BufferedLoader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: Loader.Opts,
    group: stdx.Io.LimitedGroup,
    bytes_loaded: std.atomic.Value(usize) = .init(0),
    first_error: std.atomic.Value(u16) = .init(0),

    fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        store: *const TensorStore,
        opts: Loader.Opts,
    ) !*BufferedLoader {
        const self = try allocator.create(BufferedLoader);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .store = store,
            .opts = opts,
            .group = .init(opts.read_parallelism.initial()),
        };
        return self;
    }

    fn recordError(self: *BufferedLoader, err: anyerror) void {
        _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
    }

    fn checkOpen(self: *BufferedLoader) !void {
        const code = self.first_error.load(.acquire);
        if (code != 0) return @errorFromInt(code);
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
                const total = reader.interface.streamRemaining(&writer.interface) catch |err| {
                    loader.recordError(err);
                    return;
                };
                writer.interface.flush() catch |err| {
                    loader.recordError(err);
                    return;
                };
                _ = loader.bytes_loaded.fetchAdd(total, .monotonic);
            }
        }.run, .{ self, source, shape, sharding, output });
    }

    fn load(
        self: *BufferedLoader,
        comptime ModelType: type,
        model: *const ModelType,
        buffers: *Bufferized(ModelType),
    ) !void {
        try self.checkOpen();
        const tensor_count = meta.count(Tensor, model);
        var arena: std.heap.ArenaAllocator = .init(self.allocator);
        defer arena.deinit();
        const flattened = try arena.allocator().alloc(*Buffer, tensor_count);
        meta.forEachVisit(buffers, *Buffer, struct {
            fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
                output[i] = buffer;
            }
        }.call, .{flattened});
        const Prepared = struct {
            source: *safetensors.Tensor,
            shape: Shape,
            sharding: Sharding,
            output: *Buffer,
        };
        const prepared = try arena.allocator().alloc(Prepared, tensor_count);
        const Ctx = struct {
            loader: *BufferedLoader,
            buffers: []*Buffer,
            prepared: []Prepared,
            count: usize = 0,
            err: ?anyerror = null,
        };
        var ctx: Ctx = .{ .loader = self, .buffers = flattened, .prepared = prepared };
        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, context: *Ctx) void {
                if (context.err != null) return;
                const sources = context.loader.store.getSourcesById(tensor.id) orelse {
                    context.err = error.NotFound;
                    return;
                };
                if (sources.len != 1) {
                    load_log.debug("skipping fused tensor with {} sources; load it with Loader.loadExecute", .{sources.len});
                    return;
                }
                const shape = tensor.shape();
                const sharding = Sharding.pickSharding(
                    context.loader.opts.shardings,
                    shape,
                    .explicit_axis_binding,
                ) orelse context.loader.platform.replicated_sharding;
                context.prepared[context.count] = .{
                    .source = sources[0],
                    .shape = shape,
                    .sharding = sharding,
                    .output = context.buffers[i],
                };
                context.count += 1;
            }
        }.call, .{&ctx});
        if (ctx.err) |err| return err;
        for (prepared[0..ctx.count]) |item| {
            self.submitOne(item.source, item.shape, item.sharding, item.output);
        }
    }

    fn await(self: *BufferedLoader) !void {
        self.group.await(self.io) catch |err| self.recordError(err);
        try self.checkOpen();
    }

    fn loadExecute(self: *BufferedLoader, tensor: Tensor, output: *Buffer, exe: *const Exe) !void {
        try self.checkOpen();
        const sources = self.store.getSourcesById(tensor.id) orelse return error.NotFound;
        const output_sharding = Sharding.pickSharding(
            self.opts.shardings,
            tensor.shape(),
            .explicit_axis_binding,
        ) orelse self.platform.replicated_sharding;
        try validateExecutableBinding(self.platform, tensor, sources, exe, output_sharding);
        const inputs = try self.allocator.alloc(Buffer, sources.len);
        defer self.allocator.free(inputs);
        for (inputs, exe.input_shapes, exe.input_shardings) |*input, shape, sharding| {
            input.* = .{
                ._platform = self.platform,
                ._shape = shape,
                ._sharding = sharding.resolve(self.platform),
                ._shards = .empty,
            };
        }
        defer for (inputs) |*input| input.deinit();
        for (sources, exe.input_shapes, exe.input_shardings, inputs) |source, shape, sharding, *input| {
            self.submitOne(source, shape, sharding.resolve(self.platform), input);
        }
        try self.await();
        try executeLoadedBinding(self.allocator, self.io, inputs, output, exe);
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

const DirectLoader = struct {
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: Loader.Opts,
    dma_resources: *DmaPlatformSettings,
    pool: mem.DmaBlockPool,
    scheduler: FairVectoredReadScheduler,
    metrics: VectoredLoadMetrics = .{},
    worker_gate: AdaptiveRequestGate,
    read_gate: AdaptiveRequestGate,
    request_gate: AdaptiveRequestGate,
    pipeline: VectoredLoadPipeline,
    controller_runtime: SourceReadRuntime,
    worker_group: std.Io.Group = .init,
    controller_group: std.Io.Group = .init,
    source_slots: std.ArrayListUnmanaged(*LoaderSourceSlot) = .empty,
    epoch_items: std.ArrayListUnmanaged(*LoaderLoadItem) = .empty,
    bytes_loaded: std.atomic.Value(usize) = .init(0),
    epoch_logical_bytes: usize = 0,
    epoch_source_bytes: u64 = 0,
    epoch_source_jobs: usize = 0,
    epoch_source_runs: usize = 0,
    epoch_source_items: usize = 0,
    epoch_source_pieces: usize = 0,
    epoch_planning_ns: u64 = 0,
    epoch_started_at: ?std.Io.Timestamp = null,
    epoch_number: usize = 0,
    logged_read_operations: u64 = 0,
    logged_source_calls: u64 = 0,
    logged_transfer_pieces: u64 = 0,
    logged_dma_submissions: u64 = 0,
    diagnostic_stats: ?VFS.ReadStats = null,
    epoch_active: bool = false,
    source_request_size: usize,
    maximum_blocks_per_job: usize,
    effective_pinned_feasible_width: usize,
    read_stats_storage: [1]VectoredReadStatsSource = undefined,
    read_stats_len: usize = 0,
    workers_started: bool = false,
    controller_started: bool = false,
    cleaned: bool = false,

    fn create(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        store: *const TensorStore,
        opts: Loader.Opts,
    ) !*DirectLoader {
        if (platform.devices.len == 0 or platform.devices.len > 64)
            return error.DmaDeviceMismatch;
        const device_ids = try allocator.alloc(u32, platform.devices.len);
        defer allocator.free(device_ids);
        for (platform.devices, device_ids) |device, *device_id| device_id.* = device.id();
        const resources = try acquirePlatformDmaSettings(platform, device_ids);
        errdefer releasePlatformDmaSettings(platform);
        const config = resources.config;
        if (config.block_size > max_load_read_request_size or
            config.max_mapped_bytes < max_load_read_request_size)
            return error.InvalidDmaLoadConfig;

        const request_size = try effectiveSourceRequestSize(
            opts.load_profile.read_chunk_size,
            config.block_size,
        );
        const maximum_blocks_per_job = try std.math.add(
            usize,
            std.math.divCeil(usize, request_size, config.block_size) catch
                return error.InvalidDmaLoadConfig,
            platform.devices.len - 1,
        );
        const node_reserves = try allocator.alloc(usize, resources.workspace.pools.len);
        defer allocator.free(node_reserves);
        @memset(node_reserves, 0);
        for (platform.devices, 0..) |_, device_index| {
            const node_index = resources.workspace.device_pool_indices[device_index];
            node_reserves[node_index] = try std.math.add(
                usize,
                node_reserves[node_index],
                config.max_in_flight_per_device,
            );
        }
        // The per-node reserves are deliberately non-materialized. They keep
        // enough mapped-budget capacity available for devices that join a
        // later submission without paying their allocation cost at init.
        var pool = try mem.DmaBlockPool.initFromProvider(
            allocator,
            resources.workspace.blockPoolArenaProvider(),
            config.block_size,
            config.max_mapped_bytes,
            node_reserves,
        );
        var pool_moved = false;
        errdefer if (!pool_moved) pool.deinit();
        const aggregate_width = try pool.aggregatePotentialRequestWidth(maximum_blocks_per_job);
        const strict_width = try pool.minimumStrictAffinityRequestWidth(maximum_blocks_per_job);
        const strict_affinity = for (config.device_numa_nodes) |node| {
            if (node != null) break true;
        } else false;
        const feasible_width = if (strict_affinity)
            @min(aggregate_width, strict_width)
        else
            aggregate_width;
        if (feasible_width == 0) return error.DmaMappedBudgetExceeded;

        var scheduler = try FairVectoredReadScheduler.initAppendable(
            allocator,
            platform.devices.len,
            request_size,
        );
        var scheduler_moved = false;
        errdefer if (!scheduler_moved) scheduler.deinit();
        const source_parallelism = effectiveSourceReadParallelism(
            opts.read_parallelism,
            opts.load_profile.high_latency,
        );
        const controller = SourceReadWidthController.init(source_parallelism, feasible_width);
        const limits: RequestGateLimits = .init(controller.width(), feasible_width);

        const self = try allocator.create(DirectLoader);
        errdefer allocator.destroy(self);
        self.* = .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .store = store,
            .opts = opts,
            .dma_resources = resources,
            .pool = pool,
            .scheduler = scheduler,
            .worker_gate = .init(limits.read),
            .read_gate = .init(limits.read),
            .request_gate = .init(limits.lifecycle),
            .pipeline = undefined,
            .controller_runtime = undefined,
            .source_request_size = request_size,
            .maximum_blocks_per_job = maximum_blocks_per_job,
            .effective_pinned_feasible_width = feasible_width,
        };
        // Ownership moved into the stable heap object.
        pool_moved = true;
        scheduler_moved = true;
        errdefer {
            self.scheduler.deinit();
            self.pool.deinit();
        }

        self.pipeline = try VectoredLoadPipeline.init(
            allocator,
            io,
            platform,
            &self.pool,
            &self.worker_gate,
            &self.read_gate,
            &self.request_gate,
            config.block_size,
            request_size,
            resources.workspace.device_pool_indices,
            strict_affinity,
            &self.metrics,
            config.max_in_flight_per_device,
        );
        errdefer self.pipeline.deinit();
        self.pipeline.track_epoch_jobs = true;
        self.pipeline.live_scheduler = &self.scheduler;

        if (opts.load_profile.stats) |provider| {
            const initial = provider.snapshot();
            self.read_stats_storage[0] = .{
                .provider = provider,
                .initial = initial,
                .previous = initial,
            };
            self.read_stats_len = 1;
            self.diagnostic_stats = initial;
        }
        self.controller_runtime = .{
            .controller = controller,
            .worker_gate = &self.worker_gate,
            .read_gate = &self.read_gate,
            .request_gate = &self.request_gate,
            .metrics = &self.metrics,
            .next_read_admission = &self.pipeline.next_read_admission,
            .scheduler = &self.scheduler,
            .pinned_feasible_width = feasible_width,
            .read_stats_sources = self.read_stats_storage[0..self.read_stats_len],
            .source_bootstrap_enabled = opts.load_profile.high_latency,
            .persistent = true,
        };
        errdefer self.stopWorkers();
        try self.startWorkers(source_parallelism.maximum());
        load_log.debug("live loader ready: target={s}, profile={s}, request_size={Bi:.2}, dma_block_size={Bi:.2}, workers={d}, feasible_width={d}, mapped={Bi:.2}", .{
            @tagName(platform.target),
            opts.load_profile.name,
            request_size,
            config.block_size,
            source_parallelism.maximum(),
            feasible_width,
            self.pool.mappedBytes(),
        });
        return self;
    }

    fn startWorkers(self: *DirectLoader, worker_count: usize) !void {
        try self.controller_group.concurrent(
            self.io,
            SourceReadRuntime.run,
            .{ &self.controller_runtime, self.io },
        );
        self.controller_started = true;
        self.workers_started = true;
        for (0..worker_count) |worker_index| {
            self.worker_group.concurrent(self.io, workerMain, .{ self, worker_index }) catch |err| {
                self.stopWorkers();
                return err;
            };
        }
    }

    fn workerMain(self: *DirectLoader, worker_index: usize) void {
        while (true) {
            if (!self.scheduler.waitForWork(self.io)) return;
            if (!self.worker_gate.waitUntilEnabled(self.io, worker_index)) return;
            if (self.pipeline.failed()) return;
            if (!self.request_gate.acquire(self.io)) return;
            const job = self.scheduler.claim(self.io) orelse {
                self.request_gate.release(self.io);
                continue;
            };
            self.pipeline.reserveSourceJob();
            const request = self.pipeline.registerRequest(job.len) catch |err| {
                self.pipeline.abandonSourceJob();
                self.request_gate.release(self.io);
                self.pipeline.completeEpochJob();
                self.pipeline.recordError(err);
                return;
            };
            const source_slot = job.source_slot orelse {
                request.finishScheduling();
                self.pipeline.recordError(error.InvalidLoaderJob);
                return;
            };
            VectoredReadRequest.runCoalesced(
                request,
                source_slot,
                &self.pipeline,
                job.file_offset,
                job.len,
                job.pieces,
                self,
            );
        }
    }

    fn stopWorkers(self: *DirectLoader) void {
        self.scheduler.stop(self.io);
        self.worker_gate.close(self.io);
        self.read_gate.close(self.io);
        self.request_gate.close(self.io);
        if (self.controller_started) {
            self.controller_runtime.done.set(self.io);
            self.controller_runtime.control.set(self.io);
        }
        if (self.workers_started) self.worker_group.await(self.io) catch {};
        if (self.controller_started) self.controller_group.await(self.io) catch {};
        self.workers_started = false;
        self.controller_started = false;
    }

    fn checkOpen(self: *DirectLoader) !void {
        if (self.cleaned) return error.LoaderShuttingDown;
        if (self.pipeline.errorValue()) |err| return err;
        if (self.scheduler.sealed) return error.LoaderEpochSealed;
    }

    fn sourceSlot(self: *DirectLoader, uri: []const u8) !*LoaderSourceSlot {
        for (self.source_slots.items) |slot| {
            if (std.mem.eql(u8, slot.uri, uri)) return slot;
        }
        const slot = try self.allocator.create(LoaderSourceSlot);
        errdefer self.allocator.destroy(slot);
        slot.* = .{ .uri = uri };
        try self.source_slots.append(self.allocator, slot);
        return slot;
    }

    fn createItem(
        self: *DirectLoader,
        source: *safetensors.Tensor,
        shape: Shape,
        sharding: Sharding,
        output: *Buffer,
    ) !*LoaderLoadItem {
        const item = try self.allocator.create(LoaderLoadItem);
        errdefer self.allocator.destroy(item);
        item.* = .{
            .source = source,
            .source_slot = try self.sourceSlot(source.file_uri),
            .shape = shape,
            .sharding = sharding.resolve(self.platform),
            .output = output,
        };
        return item;
    }

    fn appendItems(self: *DirectLoader, items: []const *LoaderLoadItem) !void {
        try self.checkOpen();
        const planning_started: std.Io.Timestamp = .now(self.io, .awake);
        var batch = try FairVectoredReadScheduler.prepareBatch(
            self.allocator,
            self.platform.devices.len,
            items,
            self.dma_resources.config.block_size,
            self.source_request_size,
        );
        const planning_elapsed = planning_started.untilNow(self.io, .awake);
        defer batch.deinit();
        if (batch.maximum_blocks_per_job > self.maximum_blocks_per_job)
            return error.InvalidDmaLoadConfig;
        const batch_piece_count = batch.pieces.len;
        try self.epoch_items.ensureUnusedCapacity(self.allocator, items.len);

        var logical_bytes: usize = 0;
        for (items) |item| {
            const placement = try item.sharding.placement(item.shape.packedShape());
            if (placement.shape.byteSize() != 0) {
                for (item.sharding.devicesInCanonicalOrder()) |device| {
                    if (device.id >= self.platform.devices.len) return error.DmaDeviceMismatch;
                }
            }
            logical_bytes = try std.math.add(usize, logical_bytes, item.source.shape.byteSize());
        }
        const new_epoch_logical_bytes = try std.math.add(
            usize,
            self.epoch_logical_bytes,
            logical_bytes,
        );
        const new_epoch_source_bytes = try std.math.add(
            u64,
            self.epoch_source_bytes,
            batch.remaining_bytes,
        );
        if (!self.epoch_active) {
            self.epoch_started_at = .now(self.io, .awake);
            if (self.opts.load_profile.stats) |provider| {
                // Exclude aggregate backend traffic that happened while this
                // loader had no active epoch from the next diagnostic delta.
                self.diagnostic_stats = provider.snapshot();
            }
        }
        self.pipeline.beginEpochJobs(batch.jobs.len);
        self.scheduler.appendPrepared(self.io, &batch) catch |err| {
            self.pipeline.cancelEpochJobs(batch.jobs.len);
            return err;
        };
        for (items) |item| self.epoch_items.appendAssumeCapacity(item);
        self.epoch_logical_bytes = new_epoch_logical_bytes;
        self.epoch_source_bytes = new_epoch_source_bytes;
        self.epoch_source_jobs += batch.jobs.len;
        self.epoch_source_runs += batch.source_runs;
        self.epoch_source_items += items.len;
        self.epoch_source_pieces += batch_piece_count;
        self.epoch_planning_ns +|= @intCast(@max(planning_elapsed.nanoseconds, 0));
        self.epoch_active = true;
    }

    fn load(
        self: *DirectLoader,
        comptime ModelType: type,
        model: *const ModelType,
        buffers: *Bufferized(ModelType),
    ) !void {
        try self.checkOpen();
        const count = meta.count(Tensor, model);
        const flattened = try self.allocator.alloc(*Buffer, count);
        defer self.allocator.free(flattened);
        meta.forEachVisit(buffers, *Buffer, struct {
            fn call(i: usize, buffer: *Buffer, output: []*Buffer) void {
                output[i] = buffer;
            }
        }.call, .{flattened});
        var items: std.ArrayListUnmanaged(*LoaderLoadItem) = .empty;
        defer items.deinit(self.allocator);
        errdefer for (items.items) |item| item.deinit(self.allocator);
        const Ctx = struct {
            loader: *DirectLoader,
            buffers: []*Buffer,
            items: *std.ArrayListUnmanaged(*LoaderLoadItem),
            err: ?anyerror = null,
        };
        var ctx: Ctx = .{ .loader = self, .buffers = flattened, .items = &items };
        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, context: *Ctx) void {
                if (context.err != null) return;
                const sources = context.loader.store.getSourcesById(tensor.id) orelse {
                    context.err = error.NotFound;
                    return;
                };
                if (sources.len != 1) {
                    load_log.debug("skipping fused tensor with {} sources; load it with Loader.loadExecute", .{sources.len});
                    return;
                }
                const shape = tensor.shape();
                const sharding = Sharding.pickSharding(
                    context.loader.opts.shardings,
                    shape,
                    .explicit_axis_binding,
                ) orelse context.loader.platform.replicated_sharding;
                const item = context.loader.createItem(
                    sources[0],
                    shape,
                    sharding,
                    context.buffers[i],
                ) catch |err| {
                    context.err = err;
                    return;
                };
                context.items.append(context.loader.allocator, item) catch |err| {
                    item.deinit(context.loader.allocator);
                    context.err = err;
                };
            }
        }.call, .{&ctx});
        if (ctx.err) |err| return err;
        try self.appendItems(items.items);
        items.clearRetainingCapacity();
    }

    fn await(self: *DirectLoader) !void {
        if (self.cleaned) return error.LoaderShuttingDown;
        if (!self.epoch_active) {
            if (self.pipeline.errorValue()) |err| return err;
            return;
        }
        self.scheduler.seal(self.io);
        self.pipeline.waitEpochDrained();
        self.controller_runtime.epochBarrier(self.io);
        if (self.pipeline.errorValue()) |_| {
            for (self.epoch_items.items) |item| {
                if (item.state.status.load(.acquire) != LoaderLoadItem.StateSlot.ready) continue;
                for (item.state.state.targets) |*target| {
                    if (!target.final_submitted) {
                        target.manager.setBufferErrorUnknown(
                            self.platform.pjrt_api,
                            0,
                            "live loader failed",
                        ) catch {};
                    }
                }
            }
        }
        self.pipeline.reapCompleted();
        for (self.epoch_items.items) |item| item.deinit(self.allocator);
        self.epoch_items.clearRetainingCapacity();
        self.logEpoch(self.pipeline.errorValue() == null);
        if (self.pipeline.errorValue()) |err| {
            self.epoch_logical_bytes = 0;
            self.epoch_source_bytes = 0;
            self.epoch_source_jobs = 0;
            self.epoch_source_runs = 0;
            self.epoch_source_items = 0;
            self.epoch_source_pieces = 0;
            self.epoch_planning_ns = 0;
            self.epoch_started_at = null;
            self.epoch_active = false;
            return err;
        }
        _ = self.bytes_loaded.fetchAdd(self.epoch_logical_bytes, .monotonic);
        self.epoch_logical_bytes = 0;
        self.epoch_source_bytes = 0;
        self.epoch_source_jobs = 0;
        self.epoch_source_runs = 0;
        self.epoch_source_items = 0;
        self.epoch_source_pieces = 0;
        self.epoch_planning_ns = 0;
        self.epoch_started_at = null;
        self.epoch_active = false;
        self.scheduler.reopen(self.io);
    }

    fn logEpoch(self: *DirectLoader, successful: bool) void {
        const read_operations = self.metrics.read_operations.load(.acquire);
        const source_calls = self.metrics.source_calls.load(.acquire);
        const transfer_pieces = self.metrics.transfer_pieces.load(.acquire);
        const dma_submissions = self.metrics.dma_submissions.load(.acquire);
        const epoch_reads = read_operations -| self.logged_read_operations;
        const epoch_source_calls = source_calls -| self.logged_source_calls;
        const epoch_transfer_pieces = transfer_pieces -| self.logged_transfer_pieces;
        const epoch_dma = dma_submissions -| self.logged_dma_submissions;
        self.logged_read_operations = read_operations;
        self.logged_source_calls = source_calls;
        self.logged_transfer_pieces = transfer_pieces;
        self.logged_dma_submissions = dma_submissions;
        var source_requests: u64 = 0;
        var source_bytes: u64 = 0;
        var source_retries: u64 = 0;
        var source_throttles: u64 = 0;
        if (self.opts.load_profile.stats) |provider| {
            const current = provider.snapshot();
            if (self.diagnostic_stats) |previous| {
                const delta = current.sub(previous);
                source_requests = delta.physical_requests;
                source_bytes = delta.physical_bytes;
                source_retries = delta.retries;
                source_throttles = delta.throttles;
            }
            self.diagnostic_stats = current;
        }
        const elapsed_seconds: f64 = if (self.epoch_started_at) |started|
            @as(f64, @floatFromInt(started.untilNow(self.io, .awake).nanoseconds)) /
                std.time.ns_per_s
        else
            0;
        const average_read_size = if (self.epoch_source_jobs == 0)
            0
        else
            self.epoch_source_bytes / self.epoch_source_jobs;
        const coalescing_ratio = if (self.epoch_source_jobs == 0)
            0
        else
            @as(f64, @floatFromInt(self.epoch_source_items)) /
                @as(f64, @floatFromInt(self.epoch_source_jobs));
        load_log.debug("epoch completed: epoch={d}, successful={}, logical_bytes={Bi:.2}, planned_source_bytes={Bi:.2}, elapsed={d:.3}s, planning_elapsed={d:.3}s, reads={d}, physical_source_calls={d}, planned_source_jobs={d}, source_runs={d}, source_items={d}, source_slices={d}, tensor_transfer_pieces={d}, coalescing_ratio={d:.2}, average_read_size={Bi:.2}, selected_source_width={d}, request_size={Bi:.2}, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}, dma_submissions={d}, pinned_high_water={Bi:.2}, pinned_mapped={Bi:.2}", .{
            self.epoch_number,
            successful,
            self.epoch_logical_bytes,
            self.epoch_source_bytes,
            elapsed_seconds,
            @as(f64, @floatFromInt(self.epoch_planning_ns)) / std.time.ns_per_s,
            epoch_reads,
            epoch_source_calls,
            self.epoch_source_jobs,
            self.epoch_source_runs,
            self.epoch_source_items,
            self.epoch_source_pieces,
            epoch_transfer_pieces,
            coalescing_ratio,
            average_read_size,
            self.controller_runtime.reported_width.load(.acquire),
            self.source_request_size,
            source_requests,
            source_bytes,
            source_retries,
            source_throttles,
            epoch_dma,
            self.pool.highWaterBytes(),
            self.pool.mappedBytes(),
        });
        self.epoch_number += 1;
    }

    fn loadExecute(self: *DirectLoader, tensor: Tensor, output: *Buffer, exe: *const Exe) !void {
        try self.checkOpen();
        const sources = self.store.getSourcesById(tensor.id) orelse return error.NotFound;
        const output_sharding = Sharding.pickSharding(
            self.opts.shardings,
            tensor.shape(),
            .explicit_axis_binding,
        ) orelse self.platform.replicated_sharding;
        try validateExecutableBinding(self.platform, tensor, sources, exe, output_sharding);
        const inputs = try self.allocator.alloc(Buffer, sources.len);
        defer self.allocator.free(inputs);
        for (inputs, exe.input_shapes, exe.input_shardings) |*input, shape, sharding| {
            input.* = .{
                ._platform = self.platform,
                ._shape = shape,
                ._sharding = sharding.resolve(self.platform),
                ._shards = .empty,
            };
        }
        defer for (inputs) |*input| input.deinit();

        const items = try self.allocator.alloc(*LoaderLoadItem, sources.len);
        defer self.allocator.free(items);
        var initialized: usize = 0;
        errdefer for (items[0..initialized]) |item| item.deinit(self.allocator);
        for (sources, exe.input_shapes, exe.input_shardings, inputs, items) |source, shape, sharding, *input, *item| {
            item.* = try self.createItem(source, shape, sharding, input);
            initialized += 1;
        }
        try self.appendItems(items);
        initialized = 0;
        try self.await();
        try executeLoadedBinding(self.allocator, self.io, inputs, output, exe);
    }

    fn destroy(self: *DirectLoader) void {
        if (!self.cleaned) {
            if (self.epoch_active) self.await() catch {};
            self.stopWorkers();
            self.pipeline.reapCompleted();
            for (self.epoch_items.items) |item| item.deinit(self.allocator);
            self.epoch_items.deinit(self.allocator);
            for (self.source_slots.items) |slot| {
                slot.deinit(self.io);
                self.allocator.destroy(slot);
            }
            self.source_slots.deinit(self.allocator);
            self.pipeline.deinit();
            self.scheduler.deinit();
            self.pool.deinit();
            releasePlatformDmaSettings(self.platform);
            self.cleaned = true;
        }
        const allocator = self.allocator;
        allocator.destroy(self);
    }
};

const DmaBenchmarkCandidate = struct {
    value: usize,
    cohort: *ReusableDmaBenchmarkCohort,
    metrics: std.ArrayListUnmanaged(DmaBenchmarkRunMetrics) = .empty,

    fn median(self: DmaBenchmarkCandidate, allocator: std.mem.Allocator) !DmaBenchmarkRunMetrics {
        std.debug.assert(self.metrics.items.len > 0);
        const scratch = try allocator.dupe(DmaBenchmarkRunMetrics, self.metrics.items);
        defer allocator.free(scratch);
        std.mem.sort(DmaBenchmarkRunMetrics, scratch, {}, struct {
            fn lessThan(_: void, lhs: DmaBenchmarkRunMetrics, rhs: DmaBenchmarkRunMetrics) bool {
                return lhs.bytesPerSecond() < rhs.bytesPerSecond();
            }
        }.lessThan);
        return scratch[scratch.len / 2];
    }

    fn deinit(self: *DmaBenchmarkCandidate, allocator: std.mem.Allocator) void {
        self.metrics.deinit(allocator);
        self.* = undefined;
    }
};

const DmaBenchmarkDecision = struct {
    index: usize,
    metrics: DmaBenchmarkRunMetrics,
};

fn selectDmaBenchmarkCandidate(
    allocator: std.mem.Allocator,
    candidates: []const DmaBenchmarkCandidate,
    tolerance: f64,
) !DmaBenchmarkDecision {
    std.debug.assert(candidates.len > 0);
    const medians = try allocator.alloc(DmaBenchmarkRunMetrics, candidates.len);
    defer allocator.free(medians);
    var fastest_index: usize = 0;
    for (candidates, medians, 0..) |candidate, *median, index| {
        median.* = try candidate.median(allocator);
        if (median.bytesPerSecond() > medians[fastest_index].bytesPerSecond())
            fastest_index = index;
    }

    const floor = medians[fastest_index].bytesPerSecond() * (1.0 - tolerance);
    var selected_index = fastest_index;
    for (candidates, medians, 0..) |candidate, median, index| {
        if (median.bytesPerSecond() >= floor and
            candidate.value < candidates[selected_index].value)
            selected_index = index;
    }
    return .{ .index = selected_index, .metrics = medians[selected_index] };
}

fn dmaBenchmarkCandidateNeedsConfirmation(
    allocator: std.mem.Allocator,
    candidates: []const DmaBenchmarkCandidate,
    candidate_index: usize,
    peak_index: usize,
    tolerance: f64,
    margin: f64,
) !bool {
    if (candidate_index == peak_index) return false;
    const candidate = candidates[candidate_index];
    const peak = candidates[peak_index];
    std.debug.assert(candidate.metrics.items.len == peak.metrics.items.len);
    var qualified_once = false;
    var rejected_once = false;
    for (candidate.metrics.items, 0..) |metric, repeat| {
        var peak_rate: f64 = 0;
        for (candidates) |round_candidate| {
            std.debug.assert(round_candidate.metrics.items.len == candidate.metrics.items.len);
            peak_rate = @max(peak_rate, round_candidate.metrics.items[repeat].bytesPerSecond());
        }
        const ratio = if (peak_rate == 0) 0 else metric.bytesPerSecond() / peak_rate;
        if (ratio >= 1.0 - tolerance)
            qualified_once = true
        else
            rejected_once = true;
    }
    if (qualified_once and rejected_once) return true;
    const candidate_median = try candidate.median(allocator);
    const peak_median = try peak.median(allocator);
    const peak_rate = peak_median.bytesPerSecond();
    const ratio = if (peak_rate == 0) 0 else candidate_median.bytesPerSecond() / peak_rate;
    return @abs(ratio - (1.0 - tolerance)) <= margin;
}

fn medianDmaMetricRatioIndex(
    allocator: std.mem.Allocator,
    candidates: []const DmaBenchmarkRunMetrics,
    baselines: []const DmaBenchmarkRunMetrics,
) !usize {
    std.debug.assert(candidates.len == baselines.len and candidates.len > 0);
    const order = try allocator.alloc(usize, candidates.len);
    defer allocator.free(order);
    for (order, 0..) |*index, i| index.* = i;
    const Context = struct {
        candidates: []const DmaBenchmarkRunMetrics,
        baselines: []const DmaBenchmarkRunMetrics,
    };
    std.mem.sort(usize, order, Context{ .candidates = candidates, .baselines = baselines }, struct {
        fn lessThan(context: Context, lhs: usize, rhs: usize) bool {
            const lhs_baseline = context.baselines[lhs].bytesPerSecond();
            const rhs_baseline = context.baselines[rhs].bytesPerSecond();
            const lhs_ratio = if (lhs_baseline == 0) 0 else context.candidates[lhs].bytesPerSecond() / lhs_baseline;
            const rhs_ratio = if (rhs_baseline == 0) 0 else context.candidates[rhs].bytesPerSecond() / rhs_baseline;
            return lhs_ratio < rhs_ratio;
        }
    }.lessThan);
    return order[order.len / 2];
}

fn confirmAndSelectDmaBenchmarkCandidate(
    session: *ReusableDmaBenchmarkSession,
    opts: DmaBenchmarkOpts,
    phase: DmaBenchmarkPhase,
    candidates: []const DmaBenchmarkCandidate,
    source: []const u8,
    fixed_parallelism: ?usize,
    tolerance: f64,
) !DmaBenchmarkDecision {
    const medians = try session.allocator.alloc(DmaBenchmarkRunMetrics, candidates.len);
    defer session.allocator.free(medians);
    const ratios = try session.allocator.alloc(f64, candidates.len);
    defer session.allocator.free(ratios);
    const confirmed_metrics = try session.allocator.alloc(?DmaBenchmarkRunMetrics, candidates.len);
    defer session.allocator.free(confirmed_metrics);
    @memset(confirmed_metrics, null);

    var peak_index: usize = 0;
    for (candidates, medians, 0..) |candidate, *median, index| {
        median.* = try candidate.median(session.allocator);
        if (median.bytesPerSecond() > medians[peak_index].bytesPerSecond()) peak_index = index;
    }
    const peak_rate = medians[peak_index].bytesPerSecond();
    for (medians, ratios) |median, *ratio| {
        ratio.* = if (peak_rate == 0) 0 else median.bytesPerSecond() / peak_rate;
    }

    for (candidates, 0..) |_, candidate_index| {
        if (!try dmaBenchmarkCandidateNeedsConfirmation(
            session.allocator,
            candidates,
            candidate_index,
            peak_index,
            tolerance,
            opts.confirmation_margin,
        )) continue;
        var candidate_runs: [dma_benchmark_repeats]DmaBenchmarkRunMetrics = undefined;
        var baseline_runs: [dma_benchmark_repeats]DmaBenchmarkRunMetrics = undefined;
        for (0..dma_benchmark_repeats) |repeat| {
            const order = if (repeat % 2 == 0)
                [_]usize{ candidate_index, peak_index }
            else
                [_]usize{ peak_index, candidate_index };
            for (order) |measured_index| {
                const measured = candidates[measured_index];
                const parallelism = fixed_parallelism orelse measured.value;
                const lanes = [_]ReusableDmaBenchmarkLane{.{
                    .cohort = measured.cohort,
                    .source = source[0 .. measured.cohort.block_size * parallelism],
                    .parallelism = parallelism,
                }};
                const metrics = try session.measure(
                    phase,
                    &lanes,
                    opts.confirmation_duration_ns,
                    opts.confirmation_minimum_transfers_per_device,
                    repeat,
                );
                defer session.allocator.free(metrics);
                if (measured_index == candidate_index)
                    candidate_runs[repeat] = metrics[0]
                else
                    baseline_runs[repeat] = metrics[0];
            }
        }
        const representative = try medianDmaMetricRatioIndex(
            session.allocator,
            &candidate_runs,
            &baseline_runs,
        );
        const baseline_rate = baseline_runs[representative].bytesPerSecond();
        ratios[candidate_index] = if (baseline_rate == 0) 0 else candidate_runs[representative].bytesPerSecond() / baseline_rate;
        confirmed_metrics[candidate_index] = candidate_runs[representative];
    }

    var maximum_ratio: f64 = 1;
    for (ratios) |ratio| maximum_ratio = @max(maximum_ratio, ratio);
    const floor = maximum_ratio * (1.0 - tolerance);
    var selected_index = peak_index;
    for (candidates, ratios, 0..) |candidate, ratio, index| {
        if (ratio >= floor and candidate.value < candidates[selected_index].value)
            selected_index = index;
    }
    return .{
        .index = selected_index,
        .metrics = confirmed_metrics[selected_index] orelse medians[selected_index],
    };
}

fn deinitDmaBenchmarkCandidates(
    allocator: std.mem.Allocator,
    candidates: []DmaBenchmarkCandidate,
) void {
    for (candidates) |*candidate| candidate.deinit(allocator);
    allocator.free(candidates);
}

fn dmaBenchmarkTupleFeasible(source_len: usize, block_size: usize, parallelism: usize) bool {
    const bytes = std.math.mul(usize, block_size, parallelism) catch return false;
    return bytes <= source_len;
}

fn measureDmaBenchmarkCandidates(
    session: *ReusableDmaBenchmarkSession,
    phase: DmaBenchmarkPhase,
    candidates: []DmaBenchmarkCandidate,
    source: []const u8,
    fixed_parallelism: ?usize,
    duration_ns: u64,
    minimum_transfers_per_device: u64,
    repeats: usize,
) !void {
    for (0..repeats) |repeat| {
        for (0..candidates.len) |offset| {
            const index = (offset + repeat) % candidates.len;
            const candidate = &candidates[index];
            const parallelism = fixed_parallelism orelse candidate.value;
            const lanes = [_]ReusableDmaBenchmarkLane{.{
                .cohort = candidate.cohort,
                .source = source[0 .. candidate.cohort.block_size * parallelism],
                .parallelism = parallelism,
            }};
            const metrics = try session.measure(
                phase,
                &lanes,
                duration_ns,
                minimum_transfers_per_device,
                repeat,
            );
            defer session.allocator.free(metrics);
            try candidate.metrics.append(session.allocator, metrics[0]);
        }
    }
}

const TunedDmaDevice = struct {
    recommendation: DeviceDmaRecommendation,
    cohort: *ReusableDmaBenchmarkCohort,
};

fn tuneDmaBenchmarkDevice(
    session: *ReusableDmaBenchmarkSession,
    opts: DmaBenchmarkOpts,
    source_pools: *DmaBenchmarkSourcePools,
    device_index: usize,
) !TunedDmaDevice {
    const started_windows = session.windows;
    var block_count: usize = 0;
    var block_source_bytes: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_count += 1;
        block_source_bytes = @max(block_source_bytes, block_size * opts.block_parallelism);
    }
    if (block_count == 0) return error.NoFeasibleDmaBenchmarkTuple;
    const pool_index = source_pools.device_pool_indices[device_index];
    try source_pools.growPool(pool_index, block_source_bytes);
    const source_pool = &source_pools.pools[pool_index];

    const block_candidates = try session.allocator.alloc(DmaBenchmarkCandidate, block_count);
    errdefer session.allocator.free(block_candidates);
    var block_index: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (!dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            continue;
        block_candidates[block_index] = .{
            .value = block_size,
            .cohort = try session.createCohort(device_index, block_size),
        };
        block_index += 1;
    }
    defer deinitDmaBenchmarkCandidates(session.allocator, block_candidates);
    try measureDmaBenchmarkCandidates(
        session,
        .block,
        block_candidates,
        source_pool.source,
        opts.block_parallelism,
        opts.duration_ns,
        opts.minimum_transfers_per_device,
        dma_benchmark_repeats,
    );
    const block_decision = try confirmAndSelectDmaBenchmarkCandidate(
        session,
        opts,
        .block_confirmation,
        block_candidates,
        source_pool.source,
        opts.block_parallelism,
        opts.block_selection_tolerance,
    );
    const selected_cohort = block_candidates[block_decision.index].cohort;

    return .{
        .recommendation = .{
            .device_index = device_index,
            .device_id = session.platform.devices[device_index].id(),
            .dma_block_size = selected_cohort.block_size,
            .dma_parallelism = opts.block_parallelism,
            .measured_bytes_per_second = block_decision.metrics.bytesPerSecond(),
            .average_latency_ns = block_decision.metrics.averageLatencyNs(),
            .windows = session.windows - started_windows,
        },
        .cohort = selected_cohort,
    };
}

fn prepareDmaBenchmarkDevice(
    session: *ReusableDmaBenchmarkSession,
    device_index: usize,
    base: DeviceDmaRecommendation,
) !TunedDmaDevice {
    const cohort = try session.createCohort(device_index, base.dma_block_size);
    return .{
        .recommendation = .{
            .device_index = device_index,
            .device_id = session.platform.devices[device_index].id(),
            .dma_block_size = base.dma_block_size,
            .dma_parallelism = base.dma_parallelism,
            .measured_bytes_per_second = 0,
            .average_latency_ns = 0,
        },
        .cohort = cohort,
    };
}

fn warmupDmaBenchmarkDeviceAllocators(
    io: std.Io,
    platform: *const Platform,
    used_device_indices: []const usize,
) !void {
    const Worker = struct {
        platform: *const Platform,
        device_index: usize,
        first_error: *std.atomic.Value(u16),

        fn run(self: @This()) void {
            const dims: []const i64 = &.{1};
            const memory = self.platform.devices[self.device_index].memory(.default).?;
            const buffer = self.platform.pjrt_client.createUninitializedBuffer(self.platform.pjrt_api, .{
                .dims = dims,
                .element_type = pjrtx.bufferTypeFromDtype(.u8),
                .layout = self.platform.defaultMemoryLayout(dims, .u8),
                .dst = .{ .memory = memory.pjrt_memory },
            }) catch |err| {
                _ = self.first_error.cmpxchgStrong(0, @intFromError(err), .release, .monotonic);
                return;
            };
            buffer.deinit(self.platform.pjrt_api);
        }
    };
    var first_error: std.atomic.Value(u16) = .init(0);
    var group: std.Io.Group = .init;
    for (used_device_indices) |device_index| try group.concurrent(io, Worker.run, .{Worker{
        .platform = platform,
        .device_index = device_index,
        .first_error = &first_error,
    }});
    try group.await(io);
    const error_code = first_error.load(.acquire);
    if (error_code != 0) return @errorFromInt(error_code);
}

const DmaUsedDevices = struct {
    allocator: std.mem.Allocator,
    device_indices: []usize,
    device_ids: []u32,

    fn deinit(self: *DmaUsedDevices) void {
        self.allocator.free(self.device_indices);
        self.allocator.free(self.device_ids);
        self.* = undefined;
    }
};

fn dmaUsedDevicesForTensors(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    tensors: []const *const Tensor,
    shardings: []const Sharding,
) !DmaUsedDevices {
    const used = try allocator.alloc(bool, platform.devices.len);
    defer allocator.free(used);
    @memset(used, false);
    for (tensors) |tensor| {
        const shape = tensor.shape();
        const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse
            platform.replicated_sharding;
        const placement = try sharding.placement(shape);
        const physical_bytes = placement.shape.byteSize();
        if (physical_bytes == 0) continue;

        const ordered_devices = sharding.devicesInCanonicalOrder();
        for (ordered_devices) |device| {
            const device_index: usize = @intCast(device.id);
            if (device_index >= platform.devices.len) return error.DmaDeviceMismatch;
            used[device_index] = true;
        }
    }
    var count: usize = 0;
    for (used) |is_used| if (is_used) {
        count += 1;
    };
    const device_indices = try allocator.alloc(usize, count);
    errdefer allocator.free(device_indices);
    const device_ids = try allocator.alloc(u32, count);
    var next: usize = 0;
    for (used, 0..) |is_used, device_index| {
        if (!is_used) continue;
        device_indices[next] = device_index;
        device_ids[next] = platform.devices[device_index].id();
        next += 1;
    }
    return .{
        .allocator = allocator,
        .device_indices = device_indices,
        .device_ids = device_ids,
    };
}

fn resolveDmaNumaNodes(
    allocator: std.mem.Allocator,
    platform: *const Platform,
    used_device_indices: []const usize,
    override: []const usize,
) ![]?usize {
    const result = try allocator.alloc(?usize, platform.devices.len);
    @memset(result, null);
    if (override.len != 0) {
        if (override.len != platform.devices.len) return error.InvalidDmaBenchmarkOptions;
        if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
        for (override) |node| {
            if (node >= DmaBenchmarkNumaAllocator.max_nodes)
                return error.InvalidDmaBenchmarkOptions;
        }
        for (used_device_indices) |device_index| result[device_index] = override[device_index];
        return result;
    }

    for (used_device_indices) |device_index| {
        const node = platform.devices[device_index].numaNode() orelse {
            @memset(result, null);
            return result;
        };
        if (node >= DmaBenchmarkNumaAllocator.max_nodes) {
            @memset(result, null);
            return result;
        }
        result[device_index] = node;
    }
    if (comptime builtin.os.tag != .linux) @memset(result, null);
    return result;
}

/// Benchmarks synthetic DmaMapped PJRT transfers on every addressable device.
/// Candidate search and scoring are independent of model data and shardings.
fn benchmarkSyntheticDma(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    opts: DmaBenchmarkOpts,
) !DmaBenchmarkReport {
    const benchmark_started = std.Io.Timestamp.now(io, .awake);
    if (platform.target != .cuda and platform.target != .rocm and platform.target != .oneapi)
        return error.DmaBenchmarkUnsupported;
    if (platform.devices.len == 0 or opts.block_sizes.len == 0)
        return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.duration_ns == 0 or opts.confirmation_duration_ns == 0)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.block_parallelism == 0 or opts.block_parallelism > max_load_dma_parallelism)
        return error.InvalidDmaBenchmarkOptions;
    if (!(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1) or
        !(opts.confirmation_margin >= 0 and opts.confirmation_margin < 1))
        return error.InvalidDmaBenchmarkOptions;
    var maximum_feasible_block_size: usize = 0;
    for (opts.block_sizes) |block_size| {
        if (block_size == 0) return error.InvalidDmaBenchmarkOptions;
        if (dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            maximum_feasible_block_size = @max(maximum_feasible_block_size, block_size);
    }
    if (maximum_feasible_block_size == 0) return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.device_numa_nodes.len != 0 and
        opts.device_numa_nodes.len != platform.devices.len)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.device_numa_nodes.len != 0) {
        if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
        for (opts.device_numa_nodes) |node| {
            if (node >= DmaBenchmarkNumaAllocator.max_nodes)
                return error.InvalidDmaBenchmarkOptions;
        }
    }

    var used_devices: std.ArrayListUnmanaged(usize) = .empty;
    defer used_devices.deinit(allocator);
    const platform_device_ids = try allocator.alloc(u32, platform.devices.len);
    defer allocator.free(platform_device_ids);
    for (platform_device_ids, 0..) |*device_id, device_index| {
        try used_devices.append(allocator, device_index);
        device_id.* = platform.devices[device_index].id();
    }

    const representative_kind = platform.devices[used_devices.items[0]].kind();
    for (used_devices.items[1..]) |device_index| {
        if (!std.mem.eql(u8, representative_kind, platform.devices[device_index].kind()))
            return error.HeterogeneousDmaUnsupported;
    }
    const resolved_numa_nodes = try resolveDmaNumaNodes(
        allocator,
        platform,
        used_devices.items,
        opts.device_numa_nodes,
    );
    defer allocator.free(resolved_numa_nodes);

    const device_warmup_started = std.Io.Timestamp.now(io, .awake);
    try warmupDmaBenchmarkDeviceAllocators(io, platform, used_devices.items);
    const device_allocator_warmup_ns: u64 = @intCast(@max(
        device_warmup_started.untilNow(io, .awake).nanoseconds,
        0,
    ));
    var source_pools: DmaBenchmarkSourcePools = try .init(
        allocator,
        io,
        platform,
        resolved_numa_nodes,
        opts.max_mapped_bytes,
    );
    var source_pools_active = true;
    defer if (source_pools_active) source_pools.deinit();
    const calibration_source_bytes = std.math.mul(
        usize,
        maximum_feasible_block_size,
        opts.block_parallelism,
    ) catch return error.DmaBenchmarkPinnedBudgetExceeded;
    try source_pools.growPool(
        source_pools.device_pool_indices[used_devices.items[0]],
        calibration_source_bytes,
    );

    var samples: std.ArrayListUnmanaged(DmaBenchmarkSample) = .empty;
    errdefer samples.deinit(allocator);
    var session: ReusableDmaBenchmarkSession = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .samples = &samples,
    };
    var session_active = true;
    defer if (session_active) session.deinit(&source_pools);

    var tuned: std.ArrayListUnmanaged(TunedDmaDevice) = .empty;
    defer tuned.deinit(allocator);
    var recommendations: std.ArrayListUnmanaged(DeviceDmaRecommendation) = .empty;
    errdefer recommendations.deinit(allocator);
    for (used_devices.items, 0..) |device_index, used_index| {
        const device = if (used_index == 0)
            try tuneDmaBenchmarkDevice(
                &session,
                opts,
                &source_pools,
                device_index,
            )
        else
            try prepareDmaBenchmarkDevice(
                &session,
                device_index,
                tuned.items[0].recommendation,
            );
        try tuned.append(allocator, device);
        try recommendations.append(allocator, device.recommendation);
    }

    const uniform_block_size = recommendations.items[0].dma_block_size;
    const uniform_parallelism = recommendations.items[0].dma_parallelism;
    const config_numa_nodes = try allocator.alloc(?usize, used_devices.items.len);
    defer allocator.free(config_numa_nodes);
    for (used_devices.items, config_numa_nodes) |device_index, *node| {
        node.* = resolved_numa_nodes[device_index];
    }
    const calibrated_node_reserves = try allocator.alloc(usize, source_pools.pools.len);
    defer allocator.free(calibrated_node_reserves);
    @memset(calibrated_node_reserves, 0);
    for (used_devices.items) |device_index| {
        const pool_index = source_pools.device_pool_indices[device_index];
        calibrated_node_reserves[pool_index] = try std.math.add(
            usize,
            calibrated_node_reserves[pool_index],
            uniform_parallelism,
        );
    }

    if (tuned.items.len > 1) {
        var aggregate_source_bytes: usize = 0;
        for (recommendations.items) |recommendation| {
            const lane_source_bytes = try std.math.mul(
                usize,
                recommendation.dma_block_size,
                recommendation.dma_parallelism,
            );
            aggregate_source_bytes = try std.math.add(
                usize,
                aggregate_source_bytes,
                lane_source_bytes,
            );
        }
        if (aggregate_source_bytes > opts.max_mapped_bytes)
            return error.DmaBenchmarkPinnedBudgetExceeded;
        try source_pools.prepareAggregateSources(recommendations.items, opts.max_mapped_bytes);

        const lanes = try allocator.alloc(ReusableDmaBenchmarkLane, tuned.items.len);
        defer allocator.free(lanes);
        for (tuned.items, lanes) |device, *lane| {
            const lane_bytes = try std.math.mul(
                usize,
                device.recommendation.dma_block_size,
                device.recommendation.dma_parallelism,
            );
            lane.* = .{
                .cohort = device.cohort,
                .source = source_pools.sourceForDevice(device.recommendation.device_index)[0..lane_bytes],
                .parallelism = device.recommendation.dma_parallelism,
            };
        }

        const aggregate_started_windows = session.windows;
        const aggregate_metrics = try session.measure(
            .aggregate,
            lanes,
            opts.duration_ns,
            opts.minimum_transfers_per_device,
            0,
        );
        defer allocator.free(aggregate_metrics);
        const aggregate_windows = session.windows - aggregate_started_windows;
        for (recommendations.items, aggregate_metrics) |*recommendation, metrics| {
            recommendation.measured_bytes_per_second = metrics.bytesPerSecond();
            recommendation.average_latency_ns = metrics.averageLatencyNs();
            recommendation.windows += aggregate_windows;
        }
    }

    try source_pools.ensureLoadBlockReserves(
        uniform_block_size,
        used_devices.items.len,
        calibrated_node_reserves,
    );
    const owned_devices = try recommendations.toOwnedSlice(allocator);
    errdefer allocator.free(owned_devices);
    const owned_samples = try samples.toOwnedSlice(allocator);
    errdefer allocator.free(owned_samples);
    const resources = try DmaPlatformSettings.adopt(
        allocator,
        platform,
        .{
            .device_kind = representative_kind,
            .device_ids = platform_device_ids,
            .device_numa_nodes = config_numa_nodes,
            .block_size = uniform_block_size,
            .max_in_flight_per_device = uniform_parallelism,
            .max_mapped_bytes = opts.max_mapped_bytes,
        },
        source_pools,
    );
    source_pools_active = false;
    var result: DmaBenchmarkReport = .{
        .allocator = allocator,
        .resources = resources,
        .devices = owned_devices,
        .samples = owned_samples,
        .elapsed_ns = 0,
    };
    finishDmaBenchmarkReport(
        &result,
        io,
        benchmark_started,
        device_allocator_warmup_ns,
        &session,
        &source_pools,
    );
    session_active = false;
    source_pools_active = false;
    return result;
}

fn logDmaBenchmarkReport(platform: *const Platform, result: *const DmaBenchmarkReport) void {
    log.info("dma_bench version=12 synthetic=true numa_pools={d} retained_mapped_bytes={d} platform={s} devices={d} elapsed_ms={d:.3} calibration_ms={d:.3} allocator_warmup_ms={d:.3} source_registration_ms={d:.3} benchmark_setup_ms={d:.3} sampling_ms={d:.3} benchmark_overhead_ms={d:.3} windows={d}", .{
        result.resources.numaPoolCount(),
        result.resources.retainedMappedBytes(),
        @tagName(platform.target),
        result.devices.len,
        @as(f64, @floatFromInt(result.elapsed_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.calibration_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.device_allocator_warmup_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.source_registration_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.benchmark_setup_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.sampling_ns)) / std.time.ns_per_ms,
        @as(f64, @floatFromInt(result.benchmark_overhead_ns)) / std.time.ns_per_ms,
        result.windows,
    });
    for (result.devices) |recommendation| {
        const device = platform.devices[recommendation.device_index];
        log.info("dma_bench_numa device_index={d} device_id={d} numa_node={?d}", .{
            recommendation.device_index,
            recommendation.device_id,
            result.resources.config.device_numa_nodes[recommendation.device_index],
        });
        log.info("dma_bench_device device_index={d} device_id={d} kind=\"{s}\" block_bytes={d} parallelism={d} measured_gib_s={d:.3} average_latency_ms={d:.3} windows={d}", .{
            recommendation.device_index,
            recommendation.device_id,
            device.kind(),
            recommendation.dma_block_size,
            recommendation.dma_parallelism,
            recommendation.measured_bytes_per_second / (1024 * 1024 * 1024),
            recommendation.average_latency_ns / std.time.ns_per_ms,
            recommendation.windows,
        });
    }
    for (result.samples) |sample| {
        const device = platform.devices[sample.device_index];
        log.info("dma_bench_sample phase={s} device_index={d} device_id={d} block_bytes={d} parallelism={d} repeat={d} bytes={d} transfers={d} elapsed_ns={d} gib_s={d:.3} average_latency_ms={d:.3}", .{
            @tagName(sample.phase),
            sample.device_index,
            device.id(),
            sample.block_size,
            sample.parallelism,
            sample.repeat,
            sample.bytes,
            sample.transfers,
            sample.elapsed_ns,
            sample.bytesPerSecond() / (1024 * 1024 * 1024),
            sample.averageLatencyNs() / std.time.ns_per_ms,
        });
    }
}

/// Calibrates every addressable device and atomically replaces the platform's
/// private settings. The previous/default settings remain active on failure.
pub fn benchTransfer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *Platform,
    opts: BenchTransferOptions,
) !void {
    if (platform.target == .cpu) return;

    try beginPlatformDmaOperation(platform, dma_platform_calibrating);
    defer endPlatformDmaOperation(platform, dma_platform_calibrating);

    var result = try benchmarkSyntheticDma(allocator, io, platform, opts);
    errdefer result.deinit();
    logDmaBenchmarkReport(platform, &result);

    const replacement = try allocator.create(DmaPlatformSettings);
    replacement.* = result.resources;
    result.deinitReport();

    const old = platform._dma.settings.swap(replacement, .acq_rel);
    if (old) |ptr| destroyDmaPlatformSettings(dmaSettingsFromOpaque(ptr));
}

test "DMA benchmark completion target has no time cap" {
    try std.testing.expect(!dmaBenchmarkWindowComplete(9, 10, 128, 128));
    try std.testing.expect(!dmaBenchmarkWindowComplete(10, 10, 127, 128));
    try std.testing.expect(dmaBenchmarkWindowComplete(10, 10, 128, 128));
    try std.testing.expect(dmaBenchmarkWindowComplete(1_000, 10, 128, 128));
    try std.testing.expect(dmaBenchmarkWindowComplete(10, 10, 0, 0));
}

test "DMA benchmark synthetic distribution always transfers full blocks" {
    const allocator = std.testing.allocator;
    var distribution: DmaBenchmarkDistribution = try .init(allocator, 4);
    defer distribution.deinit(allocator);
    for (0..32) |index| try std.testing.expectEqual(
        @as(usize, 4),
        distribution.at(index),
    );
    try std.testing.expectEqual(@as(usize, 4), distribution.at(6));
}

test "DMA benchmark selection uses medians and prefers the smallest near-peak value" {
    const allocator = std.testing.allocator;
    var candidates = [_]DmaBenchmarkCandidate{
        .{ .value = 2, .cohort = undefined },
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
    };
    defer for (&candidates) |*candidate| candidate.deinit(allocator);
    const rates = [_][3]u64{
        .{ 60, 10, 62 },
        .{ 98, 99, 97 },
        .{ 100, 101, 99 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| {
            try candidate.metrics.append(allocator, .{
                .bytes = rate,
                .transfers = 1,
                .total_latency_ns = 1,
                .elapsed_ns = std.time.ns_per_s,
            });
        }
    }

    const decision = try selectDmaBenchmarkCandidate(allocator, &candidates, 0.05);
    try std.testing.expectEqual(@as(usize, 1), decision.index);
    try std.testing.expectEqual(@as(f64, 98), decision.metrics.bytesPerSecond());
}

test "DMA benchmark selection does not assume a unimodal response" {
    const allocator = std.testing.allocator;
    var candidates = [_]DmaBenchmarkCandidate{
        .{ .value = 2, .cohort = undefined },
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
        .{ .value = 16, .cohort = undefined },
    };
    defer for (&candidates) |*candidate| candidate.deinit(allocator);
    const rates = [_]u64{ 80, 100, 70, 99 };
    for (&candidates, rates) |*candidate, rate| {
        try candidate.metrics.append(allocator, .{
            .bytes = rate,
            .transfers = 1,
            .total_latency_ns = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    const decision = try selectDmaBenchmarkCandidate(allocator, &candidates, 0.02);
    try std.testing.expectEqual(@as(usize, 1), decision.index);
}

test "DMA benchmark tuple feasibility rejects pinned budget overflow" {
    try std.testing.expect(dmaBenchmarkTupleFeasible(128, 16, 8));
    try std.testing.expect(!dmaBenchmarkTupleFeasible(127, 16, 8));
    try std.testing.expect(!dmaBenchmarkTupleFeasible(std.math.maxInt(usize), std.math.maxInt(usize), 2));
}

test "DMA benchmark confirms a candidate when round qualification disagrees" {
    const allocator = std.testing.allocator;
    var candidates = [_]DmaBenchmarkCandidate{
        .{ .value = 4, .cohort = undefined },
        .{ .value = 8, .cohort = undefined },
    };
    defer for (&candidates) |*candidate| candidate.deinit(allocator);
    const rates = [_][3]u64{
        .{ 96, 80, 97 },
        .{ 100, 100, 100 },
    };
    for (&candidates, rates) |*candidate, candidate_rates| {
        for (candidate_rates) |rate| try candidate.metrics.append(allocator, .{
            .bytes = rate,
            .transfers = 1,
            .total_latency_ns = 1,
            .elapsed_ns = std.time.ns_per_s,
        });
    }
    try std.testing.expect(try dmaBenchmarkCandidateNeedsConfirmation(
        allocator,
        &candidates,
        0,
        1,
        0.05,
        0.02,
    ));
}

test "loader DMA admission rotates and respects per-device limits" {
    const all_ready: u64 = 0b1111;
    var active = [_]usize{ 0, 0, 0, 0 };
    var next_device: usize = 0;

    for ([_]usize{ 0, 1, 2, 3, 0, 1, 2, 3 }) |expected| {
        const selected = selectLoaderDmaDevice(
            &active,
            8,
            all_ready,
            next_device,
        ).?;
        try std.testing.expectEqual(expected, selected);
        next_device = (selected + 1) % active.len;
    }

    active = .{ 8, 0, 8, 0 };
    try std.testing.expectEqual(
        @as(?usize, 1),
        selectLoaderDmaDevice(&active, 8, all_ready, 0),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        selectLoaderDmaDevice(&active, 8, 0b0101, 0),
    );
}

test "source bootstrap requires a high-latency source with no observed response" {
    try std.testing.expect(shouldBootstrapSource(true, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(false, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, true, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, false, 1, 12, 12, true));
}

test "source profiles preserve adaptive read parallelism" {
    const automatic = effectiveSourceReadParallelism(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        false,
    );
    try std.testing.expect(automatic.isAdaptive());
    try std.testing.expectEqual(@as(usize, 12), automatic.initial());
    try std.testing.expectEqual(@as(usize, 128), automatic.maximum());

    const capped = effectiveSourceReadParallelism(
        .{ .adaptive = .{ .initial = 4, .maximum = 8 } },
        false,
    );
    try std.testing.expect(capped.isAdaptive());
    try std.testing.expectEqual(@as(usize, 4), capped.initial());
    try std.testing.expectEqual(@as(usize, 8), capped.maximum());

    const remote = effectiveSourceReadParallelism(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        true,
    );
    try std.testing.expect(remote.isAdaptive());

    const explicit = effectiveSourceReadParallelism(.{ .fixed = 7 }, false);
    try std.testing.expectEqual(@as(usize, 7), explicit.initial());
}

test "source request size combines the VFS floor with DMA granularity" {
    try std.testing.expectEqual(
        @as(usize, 8 * 1024 * 1024),
        try effectiveSourceRequestSize(8 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        try effectiveSourceRequestSize(8 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        try effectiveSourceRequestSize(16 * 1024 * 1024, 8 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        max_load_read_request_size,
        try effectiveSourceRequestSize(32 * 1024 * 1024, 16 * 1024 * 1024),
    );
    try std.testing.expectError(error.InvalidLoadProfile, effectiveSourceRequestSize(0, 8 * 1024 * 1024));
    try std.testing.expectError(
        error.InvalidLoadProfile,
        effectiveSourceRequestSize(max_load_read_request_size + 1, 8 * 1024 * 1024),
    );
}

test "probe source capacity counts active reads rather than retained requests" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 7, 10);
    for (0..48) |_| metrics.beginRequest(1);
    for (0..8) |index| metrics.beginRead(io, 7, 10 + @as(u64, @intCast(index)));

    try std.testing.expectEqual(@as(usize, 48), metrics.outstanding_requests.load(.acquire));
    const active = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 8), active.probe_peak_reads);
    try std.testing.expectEqual(@as(usize, 8), active.probe_active_reads);

    for (0..8) |index| metrics.endRead(io, 7, 10 + @as(u64, @intCast(index)));
    for (0..48) |_| metrics.endRequest(1);
    metrics.clearProbe(io, 7);
}

test "source probe excludes pre-boundary admissions" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.beginRead(io, 6, 40);
    metrics.prepareProbe(io, 7, 41);
    metrics.beginRead(io, 7, 40);
    metrics.recordProbeRead(io, 7, 40, max_load_read_request_size, max_load_read_request_size);
    metrics.beginRead(io, 7, 41);
    metrics.recordProbeRead(io, 7, 41, max_load_read_request_size, max_load_read_request_size);
    const admitted = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 3), admitted.active_reads);
    try std.testing.expect(admitted.probe_first_read_ns != 0);
    try std.testing.expectEqual(@as(usize, 1), admitted.probe_active_reads);
    try std.testing.expectEqual(@as(u64, 1), admitted.probe_full_read_operations);
    try std.testing.expectEqual(@as(u64, max_load_read_request_size), admitted.probe_read_bytes);
    metrics.endRead(io, 6, 40);
    metrics.endRead(io, 7, 40);
    const draining = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 1), draining.active_reads);
    try std.testing.expectEqual(@as(usize, 1), draining.probe_active_reads);
    metrics.endRead(io, 7, 41);
    const drained = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 0), drained.active_reads);
    try std.testing.expectEqual(@as(usize, 0), drained.probe_active_reads);
    metrics.clearProbe(io, 7);
}

test "partial source jobs contribute adaptive evidence" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 3, 1);
    metrics.beginRead(io, 3, 1);
    metrics.recordProbeRead(io, 3, 1, 256 * 1024, 16 * 1024 * 1024);
    metrics.endRead(io, 3, 1);
    const snapshot = metrics.snapshot(io);
    try std.testing.expectEqual(@as(u64, 1), snapshot.probe_full_read_operations);
    try std.testing.expectEqual(@as(u64, 256 * 1024), snapshot.probe_read_bytes);
}

test "request lifecycle gate permits one shared spare request" {
    const normal: RequestGateLimits = .init(12, 64);
    try std.testing.expectEqual(@as(usize, 12), normal.read);
    try std.testing.expectEqual(@as(usize, 13), normal.lifecycle);

    const clipped: RequestGateLimits = .init(32, 32);
    try std.testing.expectEqual(@as(usize, 32), clipped.read);
    try std.testing.expectEqual(@as(usize, 32), clipped.lifecycle);
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

test "adaptive request gate reductions drain without cancelling active requests" {
    const io = std.testing.io;
    var gate: AdaptiveRequestGate = .init(2);
    try std.testing.expect(gate.acquire(io));
    try std.testing.expect(gate.acquire(io));

    gate.setLimit(io, 1);
    var admitted: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(gate_: *AdaptiveRequestGate, io_: std.Io, admitted_: *std.Io.Event) void {
            if (!gate_.acquire(io_)) return;
            admitted_.set(io_);
            gate_.release(io_);
        }
    }.run, .{ &gate, io, &admitted });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!admitted.isSet());

    gate.release(io);
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!admitted.isSet());
    gate.release(io);
    try group.await(io);
    try std.testing.expect(admitted.isSet());
    try std.testing.expectEqual(@as(usize, 0), gate.inUse(io));
}

test "adaptive worker gate enables stable workers only as the limit grows" {
    const io = std.testing.io;
    var gate: AdaptiveRequestGate = .init(1);
    try std.testing.expect(gate.waitUntilEnabled(io, 0));

    var enabled: std.Io.Event = .unset;
    var group: std.Io.Group = .init;
    try group.concurrent(io, struct {
        fn run(gate_: *AdaptiveRequestGate, io_: std.Io, enabled_: *std.Io.Event) void {
            if (!gate_.waitUntilEnabled(io_, 1)) return;
            enabled_.set(io_);
        }
    }.run, .{ &gate, io, &enabled });
    try io.sleep(.fromMilliseconds(5), .awake);
    try std.testing.expect(!enabled.isSet());

    gate.setLimit(io, 2);
    try group.await(io);
    try std.testing.expect(enabled.isSet());
}

test "vectored final transfers wait for every prior destination submission" {
    var targets = [_]VectoredTensorTransfer.Target{
        .{ .manager = undefined, .pjrt_buffer = undefined, .device_index = 0, .total = 100 },
        .{ .manager = undefined, .pjrt_buffer = undefined, .device_index = 1, .total = 100 },
    };
    var tensor: VectoredTensorTransfer = undefined;
    tensor.targets = &targets;
    var block: VectoredLoadPipeline.BlockContext = undefined;
    var pipeline: VectoredLoadPipeline = undefined;
    var final: VectoredLoadPipeline.ReadyTransfer = .{
        .tensor = &tensor,
        .target = &targets[0],
        .block = &block,
        .source_offset = 0,
        .destination_offset = 80,
        .len = 20,
    };

    try std.testing.expect(!pipeline.transferReady(final));
    final.target = &targets[1];
    targets[1].submitted_bytes.store(80, .release);
    try std.testing.expect(pipeline.transferReady(final));
    final.target = &targets[0];
    targets[0].submitted_bytes.store(60, .release);
    try std.testing.expect(!pipeline.transferReady(final));
    _ = targets[0].submitted_bytes.fetchAdd(20, .release);
    try std.testing.expect(pipeline.transferReady(final));

    const non_final: VectoredLoadPipeline.ReadyTransfer = .{
        .tensor = &tensor,
        .target = &targets[0],
        .block = &block,
        .source_offset = 0,
        .destination_offset = 20,
        .len = 20,
    };
    targets[0].submitted_bytes.store(0, .release);
    try std.testing.expect(pipeline.transferReady(non_final));
}

test "late vectored callback failure drains and signals completion" {
    const io = std.testing.io;
    var queues = [_]std.ArrayListUnmanaged(VectoredLoadPipeline.ReadyTransfer){.empty};
    var active = [_]usize{1};
    var peak = [_]usize{1};
    var pipeline: VectoredLoadPipeline = .{
        .allocator = std.testing.allocator,
        .io = io,
        .platform = undefined,
        .pool = undefined,
        .worker_gate = undefined,
        .read_gate = undefined,
        .request_gate = undefined,
        .block_size = 1,
        .source_request_size = 1,
        .device_pool_indices = &.{0},
        .numa_explicit = false,
        .metrics = undefined,
        .ready_queues = &queues,
        .active_by_device = &active,
        .peak_by_device = &peak,
        .dma_limit = 1,
        .active_events = 1,
        .reads_finished = true,
    };
    pipeline.first_error.store(@intFromError(error.Unknown), .release);

    pipeline.eventCompleted(0);
    try std.testing.expect(pipeline.dma_done.isSet());
    try std.testing.expectEqual(@as(usize, 0), pipeline.active_events);
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

test "vectored request planner dispatches packed sub-byte storage" {
    const logical = Shape.init(.{ .rows = 9, .cols = 256 }, .u2)
        .withPartitioning(.{ .rows = .replicated, .cols = .replicated });
    const packed_shape = logical.packedShape();
    try std.testing.expectEqual(@as(usize, logical.byteSize()), packed_shape.byteSize());
    try VectoredRequestPlanTest.run(.{
        .name = "packed_u2",
        .device_count = 4,
        .shape = packed_shape,
        .logical_mesh = .mesh(.{ .x = .high_bandwidth }),
        .strategy = .parseBindings(.{ .x = .link_x }),
        .request_size = 131,
        .block_size = 67,
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
