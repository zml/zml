const std = @import("std");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

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
    id_map: std.AutoHashMapUnmanaged(usize, *safetensors.Tensor),
    allocator: std.mem.Allocator,

    pub fn fromRegistry(allocator: std.mem.Allocator, registry: *safetensors.TensorRegistry) TensorStore {
        return .{
            .registry = registry,
            .id_map = .empty,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *TensorStore) void {
        self.id_map.deinit(self.allocator);
    }

    fn bindIdToKey(self: *TensorStore, key: []const u8, id: usize) !void {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key).?;

        const gop = try self.id_map.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Key {s} already has an associated tensor (id: {})", .{ key, gop.key_ptr.* });
        }
        errdefer self.id_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = tensor_desc_ptr;
    }

    fn getPtrFromKey(self: *const TensorStore, key: []const u8) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const TensorStore, id: usize) ?*safetensors.Tensor {
        const tensor_desc_ptr = self.id_map.get(id) orelse return null;
        return tensor_desc_ptr;
    }

    pub fn getReader(self: *const TensorStore, key: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        return self.registry.reader(io, key, buffer);
    }

    pub fn getReaderById(self: *const TensorStore, id: usize, io: std.Io, buffer: []u8) !safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;

        return safetensors.TensorReader.init(io, tensor_desc.*, buffer, .{});
    }

    fn getBorrowedPositionalReaderById(self: *const TensorStore, id: usize, io: std.Io, file: std.Io.File) !safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;
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
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        pub fn withLayer(self: *const View, index: usize) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{d}.", .{ self.prefix() orelse "", index }) catch unreachable;

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
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            return for (self.store.registry.tensors.keys()) |k| {
                if (std.mem.startsWith(u8, k, key)) break true;
            } else false;
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            if (@TypeOf(tagz) != @TypeOf(null)) {
                switch (@typeInfo(@TypeOf(tagz))) {
                    .optional => if (tagz) |t| {
                        ptr.shape = ptr.shape.withTags(t);
                    },
                    else => ptr.shape = ptr.shape.withTags(tagz),
                }
            }

            if (@TypeOf(partitioning) == @TypeOf(null)) {
                @compileError("TensorStore.View.createTensor partitioning cannot be null; pass .replicated or an explicit partitioning");
            }

            switch (@typeInfo(@TypeOf(partitioning))) {
                .optional => @compileError("TensorStore.View.createTensor partitioning cannot be optional; pass .replicated or an explicit partitioning"),
                .enum_literal => switch (partitioning) {
                    .replicated => ptr.shape = ptr.shape.withReplicatedPartitioning(),
                    else => @compileError("Only .replicated is supported as a standalone partitioning enum literal"),
                },
                else => ptr.shape = ptr.shape.withPartitioning(partitioning),
            }

            const tensor: Tensor = .fromShape(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8, tagz: anytype, partitioning: anytype) Tensor {
            return self.maybeCreateTensor(subkey, tagz, partitioning).?;
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            };
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getReader(self: View, subkey: []const u8, io: std.Io, buffer: []u8) !safetensors.TensorReader {
            var key_buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&key_buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
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
    };
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

        const shape = reader.tensor.shape;
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

const VectoredLoadPipeline = struct {
    const DeferredTransfer = struct {
        tensor: *VectoredTensorTransfer,
        block: *mem.DmaBlockPool.Lease,
        writer_mask: u64,
        destination_offset: usize,
        len: usize,
    };

    const EventContext = struct {
        pipeline: *VectoredLoadPipeline,
        block: *mem.DmaBlockPool.Lease,
        pjrt_event: *pjrt.Event,
        err: ?*pjrt.Error = null,
        submitted_at: std.Io.Timestamp,
        bytes: usize,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    pool: *mem.DmaBlockPool,
    block_size: usize,
    metrics: *VectoredLoadMetrics,
    first_error: std.atomic.Value(u16) = .init(0),
    metadata_mutex: std.Io.Mutex = .init,
    blocks: std.ArrayListUnmanaged(*mem.DmaBlockPool.Lease) = .empty,
    deferred: std.ArrayListUnmanaged(DeferredTransfer) = .empty,
    events: std.ArrayListUnmanaged(*EventContext) = .empty,
    active_events: usize = 0,
    submissions_finished: bool = false,
    dma_done: std.Io.Event = .unset,

    fn deinit(self: *VectoredLoadPipeline) void {
        std.debug.assert(self.active_events == 0);
        for (self.events.items) |ctx| {
            ctx.pjrt_event.deinit(self.platform.pjrt_api);
            if (ctx.err) |err| err.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
        }
        for (self.blocks.items) |block| {
            std.debug.assert(block.isComplete());
            self.allocator.destroy(block);
        }
        self.events.deinit(self.allocator);
        self.deferred.deinit(self.allocator);
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
        }
    }

    fn registerBlock(self: *VectoredLoadPipeline, data: []u8, references: usize) !*mem.DmaBlockPool.Lease {
        const block = try self.allocator.create(mem.DmaBlockPool.Lease);
        errdefer self.allocator.destroy(block);
        block.* = .init(self.pool, self.io, data, references);
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.blocks.append(self.allocator, block);
        return block;
    }

    fn transferReady(self: *const VectoredLoadPipeline, transfer: DeferredTransfer) bool {
        _ = self;
        var mask = transfer.writer_mask;
        while (mask != 0) {
            const writer_index: usize = @intCast(@ctz(mask));
            mask &= mask - 1;
            const target = &transfer.tensor.targets[writer_index];
            if (transfer.destination_offset + transfer.len == target.total and
                target.submitted_bytes.load(.acquire) != transfer.destination_offset)
            {
                return false;
            }
        }
        return true;
    }

    fn submitOrDefer(self: *VectoredLoadPipeline, transfer: DeferredTransfer) void {
        if (self.failed()) {
            var remaining = @popCount(transfer.writer_mask);
            while (remaining > 0) : (remaining -= 1) transfer.block.complete();
            return;
        }
        var should_submit = false;
        self.metadata_mutex.lockUncancelable(self.io);
        if (self.transferReady(transfer)) {
            should_submit = true;
        } else {
            self.deferred.append(self.allocator, transfer) catch {
                self.metadata_mutex.unlock(self.io);
                self.recordError(error.OutOfMemory);
                var remaining = @popCount(transfer.writer_mask);
                while (remaining > 0) : (remaining -= 1) transfer.block.complete();
                return;
            };
        }
        self.metadata_mutex.unlock(self.io);
        if (should_submit) {
            self.submitTransfer(transfer);
            self.submitReadyDeferred();
        }
    }

    fn submitTransfer(self: *VectoredLoadPipeline, transfer: DeferredTransfer) void {
        var mask = transfer.writer_mask;
        while (mask != 0) {
            const writer_index: usize = @intCast(@ctz(mask));
            mask &= mask - 1;
            const target = &transfer.tensor.targets[writer_index];
            const is_last = transfer.destination_offset + transfer.len == target.total;
            if (self.submitOne(target, transfer.block, transfer.destination_offset, transfer.len, is_last)) {
                _ = target.submitted_bytes.fetchAdd(transfer.len, .release);
            }
        }
    }

    fn submitOne(
        self: *VectoredLoadPipeline,
        target: *VectoredTensorTransfer.Target,
        block: *mem.DmaBlockPool.Lease,
        destination_offset: usize,
        len: usize,
        is_last: bool,
    ) bool {
        const submitted_at: std.Io.Timestamp = .now(self.io, .awake);
        const event = target.manager.transferData(
            self.platform.pjrt_api,
            0,
            block.data[0..len],
            @intCast(destination_offset),
            is_last,
        ) catch |err| {
            self.recordError(err);
            block.complete();
            return false;
        };
        if (is_last) target.final_submitted = true;

        const ctx = self.allocator.create(EventContext) catch {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.recordError(error.OutOfMemory);
            block.complete();
            return true;
        };
        ctx.* = .{
            .pipeline = self,
            .block = block,
            .pjrt_event = event,
            .submitted_at = submitted_at,
            .bytes = len,
        };

        self.metadata_mutex.lockUncancelable(self.io);
        self.events.append(self.allocator, ctx) catch {
            self.metadata_mutex.unlock(self.io);
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.allocator.destroy(ctx);
            self.recordError(error.OutOfMemory);
            block.complete();
            return true;
        };
        self.active_events += 1;
        self.metadata_mutex.unlock(self.io);

        _ = self.metrics.dma_submissions.fetchAdd(1, .monotonic);
        _ = self.metrics.submitted_bytes.fetchAdd(len, .monotonic);
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
                ctx_.pipeline.eventCompleted();
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            self.recordError(err);
            block.complete();
            self.eventCompleted();
        };
        return true;
    }

    fn eventCompleted(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        std.debug.assert(self.active_events > 0);
        self.active_events -= 1;
        if (self.submissions_finished and self.active_events == 0) self.dma_done.set(self.io);
    }

    fn submitReadyDeferred(self: *VectoredLoadPipeline) void {
        while (true) {
            if (self.failed()) return;
            var ready: ?DeferredTransfer = null;
            self.metadata_mutex.lockUncancelable(self.io);
            for (self.deferred.items, 0..) |transfer, i| {
                if (self.transferReady(transfer)) {
                    ready = self.deferred.swapRemove(i);
                    break;
                }
            }
            self.metadata_mutex.unlock(self.io);
            if (ready) |transfer| {
                self.submitTransfer(transfer);
            } else {
                return;
            }
        }
    }

    fn abortDeferred(self: *VectoredLoadPipeline) void {
        for (self.deferred.items) |transfer| {
            var remaining = @popCount(transfer.writer_mask);
            while (remaining > 0) : (remaining -= 1) transfer.block.complete();
        }
        self.deferred.clearRetainingCapacity();
    }

    fn finishSubmissions(self: *VectoredLoadPipeline) void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        self.submissions_finished = true;
        if (self.active_events == 0) self.dma_done.set(self.io);
    }
};

const VectoredReadRequest = struct {
    fn run(
        tensor: *VectoredTensorTransfer,
        pipeline: *VectoredLoadPipeline,
        source_offset: usize,
        request_len: usize,
    ) void {
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
            const block = pipeline.registerBlock(leased[i], references) catch {
                pipeline.recordError(error.OutOfMemory);
                return;
            };
            leased[i] = &.{};
            const transfer: VectoredLoadPipeline.DeferredTransfer = .{
                .tensor = tensor,
                .block = block,
                .writer_mask = block_plan.writer_mask,
                .destination_offset = block_plan.destination_offset,
                .len = block_plan.len,
            };

            pipeline.submitOrDefer(transfer);
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
    opts: LoadOpts,
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

    var metrics: VectoredLoadMetrics = .{};
    var pipeline: VectoredLoadPipeline = .{
        .allocator = allocator,
        .io = io,
        .platform = platform,
        .pool = &pool,
        .block_size = opts.dma_block_size,
        .metrics = &metrics,
    };
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
                    const index = next.fetchAdd(1, .monotonic);
                    if (index >= jobs_.len) return;
                    const job = jobs_[index];
                    const source_file = source_slots_[source_indices_[job.tensor_index]].ensure(io_) catch |err| {
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
                        pipeline_.recordError(err);
                        return;
                    };
                    VectoredReadRequest.run(tensor, pipeline_, job.source_offset, job.len);
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

    if (pipeline.failed()) {
        pipeline.abortDeferred();
        for (state_slots) |*slot| {
            if (slot.status.load(.acquire) != StateSlot.ready) continue;
            for (slot.state.targets) |*target| {
                if (!target.final_submitted) {
                    target.manager.setBufferErrorUnknown(platform.pjrt_api, 0, "vectored load failed") catch {};
                }
            }
        }
    } else {
        pipeline.submitReadyDeferred();
        if (pipeline.deferred.items.len != 0) {
            pipeline.recordError(error.IncompleteTransferPlan);
            pipeline.abortDeferred();
            for (state_slots) |*slot| {
                if (slot.status.load(.acquire) != StateSlot.ready) continue;
                for (slot.state.targets) |*target| {
                    if (!target.final_submitted) {
                        target.manager.setBufferErrorUnknown(platform.pjrt_api, 0, "vectored load did not submit every final transfer") catch {};
                    }
                }
            }
        }
    }

    pipeline.finishSubmissions();
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
    load_log.debug("completed: vectored=true, tensors={d}, logical_bytes={Bi:.2}, elapsed={d:.3}s, logical_goodput={d:.2}MiB/s, reads={d}, peak_reads={d}, average_read={Bi:.2}, average_read_latency={d:.3}ms, dma_submissions={d}, average_dma={Bi:.2}, average_dma_latency={d:.3}ms, submitted={Bi:.2}, committed={Bi:.2}, pinned_high_water={Bi:.2}, mapped={Bi:.2}, pool_waits={d}, pool_wait={d:.3}s", .{
        tensor_count,
        loaded_bytes,
        elapsed_seconds,
        goodput / (1024 * 1024),
        read_operations,
        metrics.peak_reads.load(.acquire),
        average_read,
        average_read_ms,
        dma_submissions,
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

pub const LoadOpts = struct {
    pub const auto: LoadOpts = .{};

    /// Hard maximum number of concurrent positional source requests.
    read_parallelism: usize = 12,
    /// Logical bytes gathered by one positional source request.
    read_request_size: usize = 2 * 1024 * 1024,
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
    opts: LoadOpts,
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
                    const shape = reader.tensor.shape;
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
    opts: LoadOpts,
) !Bufferized(ModelType) {
    stdx.debug.assert(opts.read_parallelism > 0, "zml.io.load read_parallelism must be greater than zero", .{});
    stdx.debug.assert(opts.read_request_size > 0, "zml.io.load read_request_size must be greater than zero", .{});
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
    load_log.debug("configured: target={s}, vectored={}, tensors={d}, read_parallelism={d}, read_request_size={Bi:.2}, dma_block_size={Bi:.2}, max_pinned_bytes={Bi:.2}, logical_bytes={Bi:.2}", .{
        @tagName(platform.target),
        direct,
        tensor_count,
        opts.read_parallelism,
        opts.read_request_size,
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
        .{ .manager = undefined, .pjrt_buffer = undefined, .total = 100 },
        .{ .manager = undefined, .pjrt_buffer = undefined, .total = 100 },
    };
    var tensor: VectoredTensorTransfer = undefined;
    tensor.targets = &targets;
    var block: mem.DmaBlockPool.Lease = undefined;
    var pipeline: VectoredLoadPipeline = undefined;
    const final: VectoredLoadPipeline.DeferredTransfer = .{
        .tensor = &tensor,
        .block = &block,
        .writer_mask = 0b11,
        .destination_offset = 80,
        .len = 20,
    };

    try std.testing.expect(!pipeline.transferReady(final));
    targets[1].submitted_bytes.store(80, .release);
    try std.testing.expect(!pipeline.transferReady(final));
    targets[0].submitted_bytes.store(60, .release);
    try std.testing.expect(!pipeline.transferReady(final));
    _ = targets[0].submitted_bytes.fetchAdd(20, .release);
    try std.testing.expect(pipeline.transferReady(final));

    const non_final: VectoredLoadPipeline.DeferredTransfer = .{
        .tensor = &tensor,
        .block = &block,
        .writer_mask = 0b01,
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
