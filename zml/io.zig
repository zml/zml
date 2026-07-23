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

    fn getBorrowedPositionalReaderById(
        self: *const TensorStore,
        id: usize,
        io: std.Io,
        file: std.Io.File,
        batch_iovecs: bool,
    ) !safetensors.TensorReader {
        const tensor_desc = self.id_map.get(id) orelse return error.NotFound;
        return .initBorrowedPositionalWithOptions(io, tensor_desc.*, file, .{
            .batch_iovecs = batch_iovecs,
        });
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
    const ProbeDimension = enum(u8) { none, read, request_size, dma };
    const LocalReadTiming = struct {
        successes: u64 = 0,
        successful_bytes: u64 = 0,
        service_ns: u64 = 0,

        fn sub(self: LocalReadTiming, previous: LocalReadTiming) LocalReadTiming {
            return .{
                .successes = self.successes -| previous.successes,
                .successful_bytes = self.successful_bytes -| previous.successful_bytes,
                .service_ns = self.service_ns -| previous.service_ns,
            };
        }
    };

    read_operations: std.atomic.Value(u64) = .init(0),
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
    outstanding_request_bytes: std.atomic.Value(u64) = .init(0),
    request_high_water: std.atomic.Value(usize) = .init(0),
    post_read_bytes: std.atomic.Value(u64) = .init(0),
    retired_bytes: std.atomic.Value(u64) = .init(0),
    weighted_request_latency_us: std.atomic.Value(u64) = .init(0),
    config_epoch: std.atomic.Value(u64) = .init(0),
    probe_epoch: std.atomic.Value(u64) = .init(std.math.maxInt(u64)),
    probe_dimension: std.atomic.Value(u8) = .init(@intFromEnum(ProbeDimension.none)),
    probe_committed_bytes: std.atomic.Value(u64) = .init(0),
    probe_first_ns: std.atomic.Value(u64) = .init(0),
    probe_active_reads: std.atomic.Value(usize) = .init(0),
    probe_peak_reads: std.atomic.Value(usize) = .init(0),
    probe_read_operations: std.atomic.Value(u64) = .init(0),
    probe_read_bytes: std.atomic.Value(u64) = .init(0),
    local_timing_successes: [adaptive_request_sizes.len]std.atomic.Value(u64) = @splat(.init(0)),
    local_timing_bytes: [adaptive_request_sizes.len]std.atomic.Value(u64) = @splat(.init(0)),
    local_timing_service_ns: [adaptive_request_sizes.len]std.atomic.Value(u64) = @splat(.init(0)),
    probe_mutex: std.Io.Mutex = .init,

    const Snapshot = struct {
        read_operations: u64,
        read_bytes: u64,
        read_ns: u64,
        weighted_read_latency_us: u64,
        pool_waits: u64,
        pool_wait_ns: u64,
        dma_submissions: u64,
        submitted_bytes: u64,
        committed_bytes: u64,
        dma_ns: u64,
        weighted_dma_latency_us: u64,
        ready_bytes: u64,
        ready_blocks: usize,
        weighted_ready_age_us: u64,
        active_reads: usize,
        peak_reads: usize,
        outstanding_requests: usize,
        outstanding_request_bytes: u64,
        post_read_bytes: u64,
        retired_bytes: u64,
        weighted_request_latency_us: u64,
        config_epoch: u64,
        probe_epoch: u64,
        probe_committed_bytes: u64,
        probe_first_ns: u64,
        probe_read_operations: u64,
        probe_read_bytes: u64,

        fn sub(self: Snapshot, previous: Snapshot) Snapshot {
            return .{
                .read_operations = self.read_operations -| previous.read_operations,
                .read_bytes = self.read_bytes -| previous.read_bytes,
                .read_ns = self.read_ns -| previous.read_ns,
                .weighted_read_latency_us = self.weighted_read_latency_us -| previous.weighted_read_latency_us,
                .pool_waits = self.pool_waits -| previous.pool_waits,
                .pool_wait_ns = self.pool_wait_ns -| previous.pool_wait_ns,
                .dma_submissions = self.dma_submissions -| previous.dma_submissions,
                .submitted_bytes = self.submitted_bytes -| previous.submitted_bytes,
                .committed_bytes = self.committed_bytes -| previous.committed_bytes,
                .dma_ns = self.dma_ns -| previous.dma_ns,
                .weighted_dma_latency_us = self.weighted_dma_latency_us -| previous.weighted_dma_latency_us,
                .ready_bytes = self.ready_bytes,
                .ready_blocks = self.ready_blocks,
                .weighted_ready_age_us = self.weighted_ready_age_us -| previous.weighted_ready_age_us,
                .active_reads = self.active_reads,
                .peak_reads = self.peak_reads,
                .outstanding_requests = self.outstanding_requests,
                .outstanding_request_bytes = self.outstanding_request_bytes,
                .post_read_bytes = self.post_read_bytes,
                .retired_bytes = self.retired_bytes -| previous.retired_bytes,
                .weighted_request_latency_us = self.weighted_request_latency_us -| previous.weighted_request_latency_us,
                .config_epoch = self.config_epoch,
                .probe_epoch = self.probe_epoch,
                .probe_committed_bytes = self.probe_committed_bytes,
                .probe_first_ns = self.probe_first_ns,
                .probe_read_operations = self.probe_read_operations,
                .probe_read_bytes = self.probe_read_bytes,
            };
        }
    };

    fn snapshot(self: *const VectoredLoadMetrics) Snapshot {
        return .{
            .read_operations = self.read_operations.load(.acquire),
            .read_bytes = self.read_bytes.load(.acquire),
            .read_ns = self.read_ns.load(.acquire),
            .weighted_read_latency_us = self.weighted_read_latency_us.load(.acquire),
            .pool_waits = self.pool_waits.load(.acquire),
            .pool_wait_ns = self.pool_wait_ns.load(.acquire),
            .dma_submissions = self.dma_submissions.load(.acquire),
            .submitted_bytes = self.submitted_bytes.load(.acquire),
            .committed_bytes = self.committed_bytes.load(.acquire),
            .dma_ns = self.dma_ns.load(.acquire),
            .weighted_dma_latency_us = self.weighted_dma_latency_us.load(.acquire),
            .ready_bytes = self.ready_bytes.load(.acquire),
            .ready_blocks = self.ready_blocks.load(.acquire),
            .weighted_ready_age_us = self.weighted_ready_age_us.load(.acquire),
            .active_reads = self.active_reads.load(.acquire),
            .peak_reads = self.peak_reads.load(.acquire),
            .outstanding_requests = self.outstanding_requests.load(.acquire),
            .outstanding_request_bytes = self.outstanding_request_bytes.load(.acquire),
            .post_read_bytes = self.post_read_bytes.load(.acquire),
            .retired_bytes = self.retired_bytes.load(.acquire),
            .weighted_request_latency_us = self.weighted_request_latency_us.load(.acquire),
            .config_epoch = self.config_epoch.load(.acquire),
            .probe_epoch = self.probe_epoch.load(.acquire),
            .probe_committed_bytes = self.probe_committed_bytes.load(.acquire),
            .probe_first_ns = self.probe_first_ns.load(.acquire),
            .probe_read_operations = self.probe_read_operations.load(.acquire),
            .probe_read_bytes = self.probe_read_bytes.load(.acquire),
        };
    }

    fn beginRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64) void {
        const active = self.active_reads.fetchAdd(1, .acq_rel) + 1;
        var peak = self.peak_reads.load(.acquire);
        while (active > peak) {
            peak = self.peak_reads.cmpxchgWeak(peak, active, .release, .acquire) orelse break;
        }
        const dimension: ProbeDimension = @enumFromInt(self.probe_dimension.load(.acquire));
        if (dimension != .read and dimension != .request_size) return;
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire)) return;
        const probe_active = self.probe_active_reads.fetchAdd(1, .acq_rel) + 1;
        var probe_peak = self.probe_peak_reads.load(.acquire);
        while (probe_active > probe_peak) {
            probe_peak = self.probe_peak_reads.cmpxchgWeak(probe_peak, probe_active, .release, .acquire) orelse break;
        }
    }

    fn endRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64) void {
        _ = self.active_reads.fetchSub(1, .acq_rel);
        const dimension: ProbeDimension = @enumFromInt(self.probe_dimension.load(.acquire));
        if (dimension != .read and dimension != .request_size) return;
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire)) return;
        _ = self.probe_active_reads.fetchSub(1, .acq_rel);
    }

    fn recordProbeRead(self: *VectoredLoadMetrics, io: std.Io, epoch: u64, bytes: usize) void {
        const dimension: ProbeDimension = @enumFromInt(self.probe_dimension.load(.acquire));
        if (dimension != .read and dimension != .request_size) return;
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire)) return;
        _ = self.probe_read_operations.fetchAdd(1, .monotonic);
        _ = self.probe_read_bytes.fetchAdd(@intCast(bytes), .monotonic);
    }

    fn resetReadPeak(self: *VectoredLoadMetrics) void {
        self.peak_reads.store(self.active_reads.load(.acquire), .release);
    }

    fn beginRequest(self: *VectoredLoadMetrics, epoch: u64, bytes: usize) void {
        const active = self.outstanding_requests.fetchAdd(1, .acq_rel) + 1;
        _ = self.outstanding_request_bytes.fetchAdd(@intCast(bytes), .monotonic);
        var high_water = self.request_high_water.load(.acquire);
        while (active > high_water) {
            high_water = self.request_high_water.cmpxchgWeak(high_water, active, .release, .acquire) orelse break;
        }
        _ = epoch;
    }

    fn endRequest(self: *VectoredLoadMetrics, epoch: u64, bytes: usize) void {
        _ = self.outstanding_requests.fetchSub(1, .acq_rel);
        _ = self.outstanding_request_bytes.fetchSub(@intCast(bytes), .monotonic);
        _ = epoch;
    }

    fn prepareProbe(self: *VectoredLoadMetrics, io: std.Io, epoch: u64, dimension: ProbeDimension) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        self.probe_epoch.store(std.math.maxInt(u64), .release);
        self.probe_dimension.store(@intFromEnum(ProbeDimension.none), .release);
        self.probe_committed_bytes.store(0, .release);
        self.probe_first_ns.store(0, .release);
        self.probe_active_reads.store(0, .release);
        self.probe_peak_reads.store(0, .release);
        self.probe_read_operations.store(0, .release);
        self.probe_read_bytes.store(0, .release);
        // Candidate attribution begins when the tuple is installed. Capacity
        // activation gates scoring, but must not hide the tuple's fill cost.
        self.probe_dimension.store(@intFromEnum(dimension), .release);
        self.probe_epoch.store(epoch, .release);
        self.config_epoch.store(epoch, .release);
    }

    fn clearProbe(self: *VectoredLoadMetrics, io: std.Io, epoch: u64) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (self.probe_epoch.load(.acquire) != epoch) return;
        self.probe_epoch.store(std.math.maxInt(u64), .release);
        self.probe_dimension.store(@intFromEnum(ProbeDimension.none), .release);
        self.probe_committed_bytes.store(0, .release);
        self.probe_first_ns.store(0, .release);
        self.probe_active_reads.store(0, .release);
        self.probe_peak_reads.store(0, .release);
        self.probe_read_operations.store(0, .release);
        self.probe_read_bytes.store(0, .release);
    }

    fn recordProbeCommit(self: *VectoredLoadMetrics, io: std.Io, dimension: ProbeDimension, epoch: u64, bytes: usize) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        const probe_epoch = self.probe_epoch.load(.acquire);
        const active_dimension: ProbeDimension = @enumFromInt(self.probe_dimension.load(.acquire));
        const matching_source_dimension = dimension == .read and active_dimension == .request_size;
        if ((active_dimension != dimension and !matching_source_dimension) or epoch != probe_epoch) return;
        _ = self.probe_committed_bytes.fetchAdd(@intCast(bytes), .monotonic);
        const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(io, .awake).nanoseconds, 1));
        _ = self.probe_first_ns.cmpxchgStrong(0, now_ns, .release, .monotonic);
    }

    fn recordLocalReadTiming(self: *VectoredLoadMetrics, request_size: usize, actual_len: usize, elapsed_ns: u64) void {
        if (actual_len != request_size) return;
        const index = requestTimingBucketIndex(request_size) orelse return;
        _ = self.local_timing_successes[index].fetchAdd(1, .monotonic);
        _ = self.local_timing_bytes[index].fetchAdd(@intCast(actual_len), .monotonic);
        _ = self.local_timing_service_ns[index].fetchAdd(elapsed_ns, .monotonic);
    }

    fn localReadTiming(self: *const VectoredLoadMetrics, index: usize) LocalReadTiming {
        return .{
            .successes = self.local_timing_successes[index].load(.acquire),
            .successful_bytes = self.local_timing_bytes[index].load(.acquire),
            .service_ns = self.local_timing_service_ns[index].load(.acquire),
        };
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
        batch_iovecs: bool,
        shardings: []const Sharding,
        output: *Buffer,
        progress_parent: ?*std.Progress.Node,
    ) !VectoredTensorTransfer {
        var reader = try store.getBorrowedPositionalReaderById(tensor.id, io, source_file, batch_iovecs);
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

const PinnedGateLimits = struct {
    feasible_width: usize,
    read: usize,
    lifecycle: usize,

    fn init(read: usize, request_size: usize, max_pinned_bytes: usize, requested_slack: usize) PinnedGateLimits {
        std.debug.assert(request_size > 0 and request_size <= max_pinned_bytes);
        const feasible_width = @max(@as(usize, 1), max_pinned_bytes / request_size);
        const effective_read = @min(read, feasible_width);
        const slack = @min(requested_slack, feasible_width - effective_read);
        return .{
            .feasible_width = feasible_width,
            .read = effective_read,
            .lifecycle = effective_read + slack,
        };
    }
};

const VectoredLoadPipeline = struct {
    const RequestContext = struct {
        pipeline: *VectoredLoadPipeline,
        started_at: std.Io.Timestamp,
        read_finished_at_ns: std.atomic.Value(u64) = .init(0),
        pending: std.atomic.Value(usize) = .init(1), // scheduling sentinel
        completed: std.atomic.Value(bool) = .init(false),
        successful: std.atomic.Value(bool) = .init(false),
        read_epoch: u64,
        len: usize,

        fn addBlock(self: *RequestContext) void {
            _ = self.pending.fetchAdd(1, .acq_rel);
        }

        fn markReadFinished(self: *RequestContext) void {
            const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(self.pipeline.io, .awake).nanoseconds, 1));
            self.read_finished_at_ns.store(now_ns, .release);
            _ = self.pipeline.metrics.post_read_bytes.fetchAdd(@intCast(self.len), .monotonic);
        }

        fn markSuccessful(self: *RequestContext) void {
            self.successful.store(true, .release);
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
                self.pipeline.metrics.recordProbeCommit(
                    self.pipeline.io,
                    .read,
                    self.read_epoch,
                    self.len,
                );
            }
            self.pipeline.metrics.endRequest(self.read_epoch, self.len);
            self.completed.store(true, .release);
            self.pipeline.request_gate.release(self.pipeline.io);
        }
    };

    const BlockContext = struct {
        pipeline: *VectoredLoadPipeline,
        request: *RequestContext,
        lease: mem.DmaBlockPool.Lease,
        ready_at: std.Io.Timestamp,
        pending_submissions: usize,
        read_epoch: u64,
        len: usize,
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
        read_epoch: u64,
        dma_epoch: u64,
        bytes: usize,
    };

    const ControlSnapshot = struct {
        active_events: usize,
        active_capacity: usize,
        active_slot_ns: u64,
        capacity_slot_ns: u64,
        max_device_active: usize,
        peak_device_active: usize,
        ready_entries: usize,
        ready_oldest_age_ns: u64,
        ready_old_entries: usize,
        post_read_requests: usize,
        post_read_oldest_age_ns: u64,
        any_device_saturated: bool,
        probe_capacity_active: bool,
        completed_device_mask: u64,
    };

    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    pool: *mem.DmaBlockPool,
    worker_gate: *AdaptiveRequestGate,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    block_size: usize,
    metrics: *VectoredLoadMetrics,
    first_error: std.atomic.Value(u16) = .init(0),
    metadata_mutex: std.Io.Mutex = .init,
    requests: std.ArrayListUnmanaged(*RequestContext) = .empty,
    blocks: std.ArrayListUnmanaged(*BlockContext) = .empty,
    ready_queues: []std.ArrayListUnmanaged(ReadyTransfer),
    events: std.ArrayListUnmanaged(*EventContext) = .empty,
    active_by_device: []usize,
    peak_by_device: []usize,
    probe_active_by_device: []usize,
    probe_peak_by_device: []usize,
    dma_limit: std.atomic.Value(usize),
    dma_probe_epoch: u64 = std.math.maxInt(u64),
    dma_probe_required_mask: u64 = 0,
    used_device_mask: u64 = 0,
    expected_device_mask: std.atomic.Value(u64) = .init(0),
    completed_device_mask: u64 = 0,
    next_device: usize = 0,
    pumping: bool = false,
    active_events: usize = 0,
    ready_entries: usize = 0,
    reads_finished: bool = false,
    slot_sample_at: std.Io.Timestamp,
    active_slot_ns: u64 = 0,
    capacity_slot_ns: u64 = 0,
    dma_done: std.Io.Event = .unset,

    fn init(
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const Platform,
        pool: *mem.DmaBlockPool,
        worker_gate: *AdaptiveRequestGate,
        read_gate: *AdaptiveRequestGate,
        request_gate: *AdaptiveRequestGate,
        block_size: usize,
        metrics: *VectoredLoadMetrics,
        initial_dma_limit: usize,
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
        const probe_active_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(probe_active_by_device);
        @memset(probe_active_by_device, 0);
        const probe_peak_by_device = try allocator.alloc(usize, platform.devices.len);
        errdefer allocator.free(probe_peak_by_device);
        @memset(probe_peak_by_device, 0);
        return .{
            .allocator = allocator,
            .io = io,
            .platform = platform,
            .pool = pool,
            .worker_gate = worker_gate,
            .read_gate = read_gate,
            .request_gate = request_gate,
            .block_size = block_size,
            .metrics = metrics,
            .ready_queues = ready_queues,
            .active_by_device = active_by_device,
            .peak_by_device = peak_by_device,
            .probe_active_by_device = probe_active_by_device,
            .probe_peak_by_device = probe_peak_by_device,
            .dma_limit = .init(initial_dma_limit),
            .slot_sample_at = .now(io, .awake),
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
        self.allocator.free(self.probe_active_by_device);
        self.allocator.free(self.probe_peak_by_device);
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
            self.pool.close(self.io);
            self.worker_gate.close(self.io);
            self.read_gate.close(self.io);
            self.request_gate.close(self.io);
        }
    }

    fn registerRequest(self: *VectoredLoadPipeline, read_epoch: u64, len: usize) !*RequestContext {
        const request = try self.allocator.create(RequestContext);
        errdefer self.allocator.destroy(request);
        request.* = .{
            .pipeline = self,
            .started_at = .now(self.io, .awake),
            .read_epoch = read_epoch,
            .len = len,
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.requests.append(self.allocator, request);
        self.metrics.beginRequest(read_epoch, len);
        return request;
    }

    fn expectTensorTargets(self: *VectoredLoadPipeline, tensor: *const VectoredTensorTransfer) void {
        for (tensor.targets) |target| {
            _ = self.expected_device_mask.fetchOr(@as(u64, 1) << @intCast(target.device_index), .release);
        }
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
            .read_epoch = request.read_epoch,
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
        destination_offset: usize,
        len: usize,
    ) !void {
        var immediate: [64]ReadyTransfer = undefined;
        var immediate_count: usize = 0;
        var queued = false;
        const dma_epoch = self.metrics.config_epoch.load(.acquire);
        self.metadata_mutex.lockUncancelable(self.io);
        errdefer self.metadata_mutex.unlock(self.io);
        var reserve_mask = writer_mask;
        while (reserve_mask != 0) {
            const writer_index: usize = @intCast(@ctz(reserve_mask));
            reserve_mask &= reserve_mask - 1;
            const target = &tensor.targets[writer_index];
            try self.ready_queues[target.device_index].ensureUnusedCapacity(self.allocator, 1);
        }
        self.accountSlotsLocked();
        _ = self.metrics.ready_bytes.fetchAdd(len, .monotonic);
        _ = self.metrics.ready_blocks.fetchAdd(1, .monotonic);
        var mask = writer_mask;
        while (mask != 0) {
            const writer_index: usize = @intCast(@ctz(mask));
            mask &= mask - 1;
            const target = &tensor.targets[writer_index];
            const transfer: ReadyTransfer = .{
                .tensor = tensor,
                .target = target,
                .block = block,
                .destination_offset = destination_offset,
                .len = len,
            };
            self.used_device_mask |= @as(u64, 1) << @intCast(target.device_index);
            const queue = &self.ready_queues[target.device_index];
            if (queue.items.len == 0 and self.active_by_device[target.device_index] < self.dma_limit.load(.acquire) and
                self.transferReady(transfer))
            {
                immediate[immediate_count] = transfer;
                immediate_count += 1;
                self.active_by_device[target.device_index] += 1;
                self.peak_by_device[target.device_index] = @max(
                    self.peak_by_device[target.device_index],
                    self.active_by_device[target.device_index],
                );
                self.active_events += 1;
                if (dma_epoch == self.dma_probe_epoch) {
                    self.probe_active_by_device[target.device_index] += 1;
                    self.probe_peak_by_device[target.device_index] = @max(
                        self.probe_peak_by_device[target.device_index],
                        self.probe_active_by_device[target.device_index],
                    );
                }
                std.debug.assert(block.pending_submissions > 0);
                block.pending_submissions -= 1;
            } else {
                queue.appendAssumeCapacity(transfer);
                self.ready_entries += 1;
                queued = true;
            }
        }
        if (block.pending_submissions == 0) {
            _ = self.metrics.ready_bytes.fetchSub(block.len, .monotonic);
            _ = self.metrics.ready_blocks.fetchSub(1, .monotonic);
        }
        self.metadata_mutex.unlock(self.io);
        for (immediate[0..immediate_count]) |transfer| self.submitOne(transfer, dma_epoch);
        if (queued) self.requestPump();
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
            var dma_epoch: u64 = 0;
            self.metadata_mutex.lockUncancelable(self.io);
            self.accountSlotsLocked();
            if (!self.failed()) {
                const limit = self.dma_limit.load(.acquire);
                for (0..self.ready_queues.len) |offset| {
                    const device_index = (self.next_device + offset) % self.ready_queues.len;
                    if (self.active_by_device[device_index] >= limit) continue;
                    const queue = &self.ready_queues[device_index];
                    for (queue.items, 0..) |transfer, i| {
                        if (!self.transferReady(transfer)) continue;
                        // Prefer recently filled blocks: swapRemove keeps the
                        // hot suffix moving toward the front. Controller
                        // pressure uses an aged cohort rather than one oldest
                        // entry so a cold tail cannot cause false backoff.
                        selected = queue.swapRemove(i);
                        self.next_device = (device_index + 1) % self.ready_queues.len;
                        self.active_by_device[device_index] += 1;
                        self.peak_by_device[device_index] = @max(self.peak_by_device[device_index], self.active_by_device[device_index]);
                        self.active_events += 1;
                        self.ready_entries -= 1;
                        dma_epoch = self.metrics.config_epoch.load(.acquire);
                        if (dma_epoch == self.dma_probe_epoch) {
                            self.probe_active_by_device[device_index] += 1;
                            self.probe_peak_by_device[device_index] = @max(
                                self.probe_peak_by_device[device_index],
                                self.probe_active_by_device[device_index],
                            );
                        }
                        std.debug.assert(transfer.block.pending_submissions > 0);
                        transfer.block.pending_submissions -= 1;
                        if (transfer.block.pending_submissions == 0) {
                            _ = self.metrics.ready_bytes.fetchSub(transfer.block.len, .monotonic);
                            _ = self.metrics.ready_blocks.fetchSub(1, .monotonic);
                            const ready_elapsed = transfer.block.ready_at.untilNow(self.io, .awake);
                            const age_us: u64 = @intCast(@max(ready_elapsed.nanoseconds, 0) / std.time.ns_per_us);
                            _ = self.metrics.weighted_ready_age_us.fetchAdd(age_us *| @as(u64, @intCast(transfer.block.len)), .monotonic);
                        }
                        break;
                    }
                    if (selected != null) break;
                }
            }
            if (selected == null) {
                self.pumping = false;
                self.maybeDoneLocked();
                self.metadata_mutex.unlock(self.io);
                return;
            }
            self.metadata_mutex.unlock(self.io);
            self.submitOne(selected.?, dma_epoch);
        }
    }

    fn submitOne(self: *VectoredLoadPipeline, transfer: ReadyTransfer, dma_epoch: u64) void {
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
            self.eventCompleted(transfer.target.device_index, dma_epoch);
            return;
        };
        if (is_last) transfer.target.final_submitted = true;
        _ = transfer.target.submitted_bytes.fetchAdd(transfer.len, .release);

        const ctx = self.allocator.create(EventContext) catch {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            event.deinit(self.platform.pjrt_api);
            self.recordError(error.OutOfMemory);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index, dma_epoch);
            return;
        };
        ctx.* = .{
            .pipeline = self,
            .block = transfer.block,
            .pjrt_event = event,
            .submitted_at = submitted_at,
            .device_index = transfer.target.device_index,
            .read_epoch = transfer.block.read_epoch,
            .dma_epoch = dma_epoch,
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
            self.eventCompleted(transfer.target.device_index, dma_epoch);
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
                    ctx_.pipeline.metrics.recordProbeCommit(ctx_.pipeline.io, .dma, ctx_.dma_epoch, ctx_.bytes);
                }
                ctx_.block.complete();
                ctx_.pipeline.eventCompleted(ctx_.device_index, ctx_.dma_epoch);
            }
        }.call, ctx) catch |err| {
            event.awaitRaw(self.platform.pjrt_api) catch {};
            self.recordError(err);
            transfer.block.complete();
            self.eventCompleted(transfer.target.device_index, dma_epoch);
        };
    }

    fn eventCompleted(self: *VectoredLoadPipeline, device_index: usize, dma_epoch: u64) void {
        self.metadata_mutex.lockUncancelable(self.io);
        self.accountSlotsLocked();
        std.debug.assert(self.active_events > 0);
        std.debug.assert(self.active_by_device[device_index] > 0);
        self.active_events -= 1;
        self.active_by_device[device_index] -= 1;
        self.completed_device_mask |= @as(u64, 1) << @intCast(device_index);
        if (dma_epoch == self.dma_probe_epoch) {
            std.debug.assert(self.probe_active_by_device[device_index] > 0);
            self.probe_active_by_device[device_index] -= 1;
        }
        self.maybeDoneLocked();
        self.metadata_mutex.unlock(self.io);
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

    fn setDmaLimit(self: *VectoredLoadPipeline, limit: usize) void {
        self.metadata_mutex.lockUncancelable(self.io);
        self.accountSlotsLocked();
        self.dma_limit.store(limit, .release);
        self.metadata_mutex.unlock(self.io);
        self.requestPump();
    }

    fn beginDmaProbe(self: *VectoredLoadPipeline, epoch: u64, candidate: usize) bool {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        var mask: u64 = 0;
        for (self.ready_queues, 0..) |queue, device_index| {
            var eligible: usize = 0;
            for (queue.items) |transfer| {
                if (self.transferReady(transfer)) eligible += 1;
            }
            if (self.active_by_device[device_index] + eligible >= candidate) {
                mask |= @as(u64, 1) << @intCast(device_index);
            }
        }
        if (mask == 0) return false;
        self.dma_probe_epoch = epoch;
        self.dma_probe_required_mask = mask;
        @memset(self.probe_active_by_device, 0);
        @memset(self.probe_peak_by_device, 0);
        return true;
    }

    fn clearDmaProbe(self: *VectoredLoadPipeline, epoch: u64) void {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        if (self.dma_probe_epoch != epoch) return;
        self.dma_probe_epoch = std.math.maxInt(u64);
        self.dma_probe_required_mask = 0;
    }

    fn controlSnapshot(self: *VectoredLoadPipeline) ControlSnapshot {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        self.accountSlotsLocked();
        var ready_oldest_age_ns: u64 = 0;
        var ready_old_entries: usize = 0;
        var ready_entries: usize = 0;
        for (self.ready_queues) |queue| {
            for (queue.items) |transfer| {
                // A final transfer is intentionally held until every earlier
                // byte for its transfer manager has been submitted. It is an
                // ordering barrier, not DMA-ready queue pressure.
                if (!self.transferReady(transfer)) continue;
                ready_entries += 1;
                const age: u64 = @intCast(@max(transfer.block.ready_at.untilNow(self.io, .awake).nanoseconds, 0));
                ready_oldest_age_ns = @max(ready_oldest_age_ns, age);
                if (age > 250 * std.time.ns_per_ms) ready_old_entries += 1;
            }
        }
        const now_ns: u64 = @intCast(@max(std.Io.Timestamp.now(self.io, .awake).nanoseconds, 1));
        var post_read_requests: usize = 0;
        var post_read_oldest_age_ns: u64 = 0;
        for (self.requests.items) |request| {
            if (request.completed.load(.acquire)) continue;
            const read_finished_at = request.read_finished_at_ns.load(.acquire);
            if (read_finished_at == 0) continue;
            post_read_requests += 1;
            post_read_oldest_age_ns = @max(post_read_oldest_age_ns, now_ns -| read_finished_at);
        }
        const limit = self.dma_limit.load(.acquire);
        var max_device_active: usize = 0;
        var peak_device_active: usize = 0;
        var any_device_saturated = false;
        for (self.active_by_device, self.peak_by_device, self.ready_queues) |active, peak, queue| {
            max_device_active = @max(max_device_active, active);
            peak_device_active = @max(peak_device_active, peak);
            if (active >= limit) {
                for (queue.items) |transfer| {
                    if (self.transferReady(transfer)) {
                        any_device_saturated = true;
                        break;
                    }
                }
            }
        }
        var active_mask: u64 = 0;
        for (self.probe_peak_by_device, 0..) |peak, device_index| {
            if (peak >= limit) active_mask |= @as(u64, 1) << @intCast(device_index);
        }
        return .{
            .active_events = self.active_events,
            .active_capacity = @popCount(self.used_device_mask) * limit,
            .active_slot_ns = self.active_slot_ns,
            .capacity_slot_ns = self.capacity_slot_ns,
            .max_device_active = max_device_active,
            .peak_device_active = peak_device_active,
            .ready_entries = ready_entries,
            .ready_oldest_age_ns = ready_oldest_age_ns,
            .ready_old_entries = ready_old_entries,
            .post_read_requests = post_read_requests,
            .post_read_oldest_age_ns = post_read_oldest_age_ns,
            .any_device_saturated = any_device_saturated,
            .probe_capacity_active = self.dma_probe_required_mask != 0 and
                (active_mask & self.dma_probe_required_mask) == self.dma_probe_required_mask,
            .completed_device_mask = self.completed_device_mask,
        };
    }

    fn accountSlotsLocked(self: *VectoredLoadPipeline) void {
        const elapsed = self.slot_sample_at.untilNow(self.io, .awake);
        const elapsed_ns: u64 = @intCast(@max(elapsed.nanoseconds, 0));
        self.active_slot_ns +|= elapsed_ns *| @as(u64, @intCast(self.active_events));
        const capacity = @popCount(self.used_device_mask) * self.dma_limit.load(.acquire);
        self.capacity_slot_ns +|= elapsed_ns *| @as(u64, @intCast(capacity));
        self.slot_sample_at = .now(self.io, .awake);
    }

    fn maybeDoneLocked(self: *VectoredLoadPipeline) void {
        if (self.reads_finished and self.ready_entries == 0 and self.active_events == 0) self.dma_done.set(self.io);
    }
};

const VectoredReadRequest = struct {
    fn run(
        request: *VectoredLoadPipeline.RequestContext,
        tensor: *VectoredTensorTransfer,
        pipeline: *VectoredLoadPipeline,
        source_offset: usize,
        request_len: usize,
        configured_request_size: usize,
        has_remote_timing: bool,
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

        const pool_wait_ns = pipeline.pool.acquireMany(pipeline.io, leased) catch |err| {
            pipeline.recordError(err);
            return;
        };
        if (pool_wait_ns > 0) _ = pipeline.metrics.pool_waits.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.pool_wait_ns.fetchAdd(pool_wait_ns, .monotonic);
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

        if (!pipeline.read_gate.acquire(pipeline.io)) return;
        pipeline.metrics.beginRead(pipeline.io, request.read_epoch);
        const read_started: std.Io.Timestamp = .now(pipeline.io, .awake);
        const read_result = tensor.reader.readPositionalAllV(iovecs, source_offset);
        const read_elapsed = read_started.untilNow(pipeline.io, .awake);
        pipeline.metrics.endRead(pipeline.io, request.read_epoch);
        pipeline.read_gate.release(pipeline.io);
        read_result catch |err| {
            pipeline.recordError(err);
            return;
        };
        pipeline.metrics.recordProbeRead(pipeline.io, request.read_epoch, request_len);
        const read_elapsed_ns: u64 = @intCast(@max(read_elapsed.nanoseconds, 0));
        const read_elapsed_us: u64 = read_elapsed_ns / std.time.ns_per_us;
        _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.read_bytes.fetchAdd(request_len, .monotonic);
        _ = pipeline.metrics.read_ns.fetchAdd(read_elapsed_ns, .monotonic);
        _ = pipeline.metrics.weighted_read_latency_us.fetchAdd(read_elapsed_us *| @as(u64, @intCast(request_len)), .monotonic);
        if (!has_remote_timing) {
            pipeline.metrics.recordLocalReadTiming(configured_request_size, request_len, read_elapsed_ns);
        }
        tensor.recordReadProgress(request_len);
        request.markReadFinished();

        if (pipeline.failed()) return;
        for (plan.blocks, 0..) |block_plan, i| {
            const references: usize = @popCount(block_plan.writer_mask);
            const block = pipeline.registerBlock(request, leased[i], references, block_plan.len) catch {
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
                var remaining = references;
                while (remaining > 0) : (remaining -= 1) block.complete();
                pipeline.recordError(err);
                return;
            };
        }
        request.markSuccessful();
    }
};

const VectoredReadScheduler = struct {
    const Job = struct {
        tensor_index: usize,
        source_offset: usize,
        len: usize,
        request_size: usize,
        epoch: u64,
    };

    const Snapshot = struct {
        remaining_bytes: u64,
        request_size: usize,
        epoch: u64,
        has_unscheduled: bool,
    };

    const CandidateCounts = struct {
        total: usize = 0,
        full: usize = 0,
    };

    allocator: std.mem.Allocator,
    tensor_sizes: []const usize,
    offsets: []usize,
    next_tensor: usize = 0,
    remaining_bytes: u64,
    request_size: usize,
    epoch: u64,
    mutex: std.Io.Mutex = .init,

    fn init(
        allocator: std.mem.Allocator,
        tensor_sizes: []const usize,
        request_size: usize,
        epoch: u64,
    ) !VectoredReadScheduler {
        const offsets = try allocator.alloc(usize, tensor_sizes.len);
        @memset(offsets, 0);
        var remaining_bytes: u64 = 0;
        for (tensor_sizes) |size| remaining_bytes +|= @intCast(size);
        return .{
            .allocator = allocator,
            .tensor_sizes = tensor_sizes,
            .offsets = offsets,
            .remaining_bytes = remaining_bytes,
            .request_size = request_size,
            .epoch = epoch,
        };
    }

    fn deinit(self: *VectoredReadScheduler) void {
        self.allocator.free(self.offsets);
        self.* = undefined;
    }

    fn claim(self: *VectoredReadScheduler, io: std.Io) ?Job {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        if (self.remaining_bytes == 0 or self.tensor_sizes.len == 0) return null;

        var visited: usize = 0;
        while (visited < self.tensor_sizes.len) : (visited += 1) {
            const tensor_index = (self.next_tensor + visited) % self.tensor_sizes.len;
            const offset = self.offsets[tensor_index];
            const tensor_size = self.tensor_sizes[tensor_index];
            if (offset >= tensor_size) continue;

            const len = @min(self.request_size, tensor_size - offset);
            self.offsets[tensor_index] += len;
            self.remaining_bytes -= len;
            self.next_tensor = (tensor_index + 1) % self.tensor_sizes.len;
            return .{
                .tensor_index = tensor_index,
                .source_offset = offset,
                .len = len,
                .request_size = self.request_size,
                .epoch = self.epoch,
            };
        }
        std.debug.assert(self.remaining_bytes == 0);
        return null;
    }

    fn setTuple(self: *VectoredReadScheduler, io: std.Io, request_size: usize, epoch: u64) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.request_size = request_size;
        self.epoch = epoch;
    }

    fn trySetCandidateTuple(
        self: *VectoredReadScheduler,
        io: std.Io,
        request_size: usize,
        epoch: u64,
        minimum_total: usize,
        minimum_full: usize,
    ) bool {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        const counts = self.candidateCountsLocked(request_size);
        if (counts.total < minimum_total or counts.full < minimum_full) return false;
        self.request_size = request_size;
        self.epoch = epoch;
        return true;
    }

    fn snapshot(self: *VectoredReadScheduler, io: std.Io) Snapshot {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return .{
            .remaining_bytes = self.remaining_bytes,
            .request_size = self.request_size,
            .epoch = self.epoch,
            .has_unscheduled = self.remaining_bytes != 0,
        };
    }

    fn candidateCounts(self: *VectoredReadScheduler, io: std.Io, candidate_size: usize) CandidateCounts {
        if (candidate_size == 0) return .{};
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        return self.candidateCountsLocked(candidate_size);
    }

    fn candidateCountsLocked(self: *VectoredReadScheduler, candidate_size: usize) CandidateCounts {
        std.debug.assert(candidate_size > 0);
        var counts: CandidateCounts = .{};
        for (self.tensor_sizes, self.offsets) |tensor_size, offset| {
            const remaining = tensor_size -| offset;
            if (remaining == 0) continue;
            counts.total +|= std.math.divCeil(usize, remaining, candidate_size) catch std.math.maxInt(usize);
            counts.full +|= remaining / candidate_size;
        }
        return counts;
    }
};

const adaptive_request_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
    64 * 1024 * 1024,
    128 * 1024 * 1024,
};

fn requestTimingBucketIndex(request_size: usize) ?usize {
    for (adaptive_request_sizes, 0..) |size, index| {
        if (request_size == size) return index;
    }
    return null;
}

fn largestAdaptiveRequestAtMost(bytes: usize) usize {
    var selected = adaptive_request_sizes[0];
    for (adaptive_request_sizes) |size| {
        if (size > bytes) break;
        selected = size;
    }
    return selected;
}

const AdaptiveVectoredController = struct {
    const Mode = enum { startup, steady };
    const Dimension = enum { read, request_size, dma };
    const ProbeKind = enum { increase, reduce_resource };

    const Limits = struct {
        read: usize,
        request_size: usize = 2 * 1024 * 1024,
        dma: usize,
    };

    const Probe = struct {
        dimension: Dimension,
        kind: ProbeKind,
        baseline: Limits,
        candidate: Limits,
        epoch: u64,
        baseline_goodput: f64,
        baseline_starvation: f64,
        activated: bool = false,
        candidate_successes: u64 = 0,
        candidate_bytes: u64 = 0,
    };

    const Sample = struct {
        now_ns: u64,
        committed_goodput: f64 = 0,
        logical_goodput: f64 = 0,
        probe_goodput: f64 = 0,
        probe_committed_bytes: u64 = 0,
        probe_elapsed_ns: u64 = 0,
        probe_read_operations: u64 = 0,
        probe_read_bytes: u64 = 0,
        dma_latency_us: f64 = 0,
        dma_latency_reliable: bool = false,
        read_latency_us: f64 = 0,
        read_latency_reliable: bool = false,
        dma_starvation_ratio: f64 = 1,
        high_latency_source: bool = false,
        read_saturated: bool = false,
        source_stalled: bool = false,
        dma_saturated: bool = false,
        dma_probe_capacity: bool = false,
        ready_pressure: bool = false,
        slow_reads: bool = false,
        source_throttled: bool = false,
        source_failure_ratio: f64 = 0,
        source_failure_reliable: bool = false,
        source_timing_successes: u64 = 0,
        source_timing_bytes: u64 = 0,
        source_ttfb_us: f64 = 0,
        source_body_bandwidth: f64 = 0,
        source_timing_reliable: bool = false,
        source_low_load: bool = false,
        remaining_bytes: u64 = std.math.maxInt(u64),
        remaining_candidate_requests: usize = std.math.maxInt(usize),
        remaining_full_candidate_requests: usize = std.math.maxInt(usize),
        estimated_remaining_ns: f64 = std.math.inf(f64),
        hard_dma_pressure: bool = false,
        allow_probe: bool = true,
        allow_resource_probe: bool = true,
    };

    const Decision = struct {
        const Action = enum {
            none,
            read_bootstrap,
            startup_settle,
            read_probe_start,
            read_probe_keep,
            read_probe_rollback,
            size_probe_start,
            size_probe_keep,
            size_probe_rollback,
            dma_probe_start,
            dma_probe_keep,
            dma_probe_rollback,
            read_reduce_start,
            read_reduce_keep,
            read_reduce_rollback,
            dma_reduce_start,
            dma_reduce_keep,
            dma_reduce_rollback,
            read_backoff,
            dma_backoff,
            probe_timeout,
            probe_tail_rollback,
        };

        const Reason = enum {
            none,
            gain_below_threshold,
            starvation_improved,
            ready_pressure,
            dma_pressure,
            source_throttle,
            source_failures,
            starvation_unresolved,
            capacity_not_exercised,
            finite_tail,
        };

        limits: Limits,
        epoch: u64,
        changed: bool = false,
        action: Action = .none,
        reason: Reason = .none,
        started_probe: ?Dimension = null,
        finished_probe: bool = false,
    };

    mode: Mode = .startup,
    max_read: usize,
    max_dma: usize,
    max_request_size: usize = 2 * 1024 * 1024,
    max_pinned_bytes: usize = std.math.maxInt(usize),
    bootstrap_read_limit: usize = 32,
    read_adaptive: bool = true,
    request_size_adaptive: bool = false,
    dma_adaptive: bool = true,
    limits: Limits,
    probe: ?Probe = null,
    epoch: u64 = 0,
    stable_goodput: f64 = 0,
    peak_goodput: f64 = 0,
    stable_logical_goodput: f64 = 0,
    peak_logical_goodput: f64 = 0,
    stable_dma_latency_us: f64 = 0,
    stable_read_latency_us: f64 = 0,
    dma_started: bool = false,
    slow_source_observed: bool = false,
    startup_read_probes: u8 = 0,
    max_startup_read_probes: u8 = 1,
    representative_windows: u8 = 0,
    dma_fed_windows: u8 = 0,
    hard_dma_windows: u8 = 0,
    source_failure_windows: u8 = 0,
    source_timing_pressure_windows: u8 = 0,
    source_timing_observed: bool = false,
    request_size_settled: bool = true,
    dma_settled: bool = false,
    // A flat width result yields one turn to request-size growth and a bounded
    // retry delay. It must not permanently suppress useful later width growth.
    read_growth_blocked_until_ns: [adaptive_request_sizes.len]u64 = @splat(0),
    prefer_request_size_probe: bool = false,
    last_source_change_ns: u64 = 0,
    ttfb_baseline_us: [7]f64 = @splat(0),
    body_bandwidth_baseline: [7]f64 = @splat(0),
    last_probe_ns: u64 = 0,
    last_startup_change_ns: u64 = 0,
    last_resource_probe_ns: u64 = 0,
    last_dma_starvation_ns: u64 = 0,
    performance_probe_blocked_until_ns: u64 = 0,
    pressure_backoff_blocked_until_ns: u64 = 0,
    resource_probe_blocked_until_ns: u64 = 0,

    const probe_byte_floor: u64 = 64 * 1024 * 1024;
    const probe_early_time_floor_ns: u64 = 100 * std.time.ns_per_ms;
    const probe_time_floor_ns: u64 = 200 * std.time.ns_per_ms;

    fn init(max_read: usize, max_dma: usize) AdaptiveVectoredController {
        return .{
            .max_read = max_read,
            .max_dma = max_dma,
            .limits = .{
                .read = @min(12, max_read),
                .request_size = 2 * 1024 * 1024,
                .dma = @min(8, max_dma),
            },
        };
    }

    fn initConfigured(
        read: Parallelism,
        dma: Parallelism,
        initial_request_size: usize,
        maximum_request_size: usize,
        request_size_adaptive: bool,
        max_pinned_bytes: usize,
    ) AdaptiveVectoredController {
        const max_read = read.maximum();
        const initial_read_capacity = @min(max_read, @max(@as(usize, 1), max_pinned_bytes / initial_request_size));
        const effective_max_request_size = if (request_size_adaptive)
            @min(maximum_request_size, largestAdaptiveRequestAtMost(max_pinned_bytes))
        else
            maximum_request_size;
        return .{
            .max_read = max_read,
            .max_dma = dma.maximum(),
            .max_request_size = effective_max_request_size,
            .max_pinned_bytes = max_pinned_bytes,
            .read_adaptive = read.isAdaptive(),
            .request_size_adaptive = request_size_adaptive,
            .request_size_settled = !request_size_adaptive or initial_request_size >= effective_max_request_size,
            .dma_adaptive = dma.isAdaptive(),
            .dma_settled = !dma.isAdaptive() or dma.initial() >= dma.maximum(),
            .limits = .{
                .read = @min(read.initial(), initial_read_capacity),
                .request_size = initial_request_size,
                .dma = dma.initial(),
            },
        };
    }

    fn readCapacity(self: *const AdaptiveVectoredController, request_size: usize) usize {
        return @min(self.max_read, @max(@as(usize, 1), self.max_pinned_bytes / request_size));
    }

    fn readGrowthCapacity(self: *const AdaptiveVectoredController) usize {
        const capacity = self.readCapacity(self.limits.request_size);
        return if (self.source_timing_observed)
            capacity
        else
            @min(capacity, self.bootstrap_read_limit);
    }

    fn readGrowthIsBlocked(
        self: *const AdaptiveVectoredController,
        request_size: usize,
        now_ns: u64,
    ) bool {
        const index = requestTimingBucketIndex(request_size) orelse return false;
        return now_ns < self.read_growth_blocked_until_ns[index];
    }

    fn blockReadGrowth(
        self: *AdaptiveVectoredController,
        request_size: usize,
        until_ns: u64,
    ) void {
        const index = requestTimingBucketIndex(request_size) orelse return;
        self.read_growth_blocked_until_ns[index] = until_ns;
    }

    fn clearReadGrowthBlock(self: *AdaptiveVectoredController, request_size: usize) void {
        self.blockReadGrowth(request_size, 0);
    }

    fn requestSizeCanGrow(self: *const AdaptiveVectoredController) bool {
        return self.request_size_adaptive and self.limits.request_size < self.max_request_size;
    }

    fn rearmRequestSizeGrowth(self: *AdaptiveVectoredController) void {
        if (self.request_size_adaptive) {
            self.request_size_settled = self.limits.request_size >= self.max_request_size;
        }
    }

    fn modeledReadConcurrency(self: *const AdaptiveVectoredController, sample: Sample, request_size: usize) usize {
        if (!sample.source_timing_reliable or sample.source_body_bandwidth <= 0) return 0;
        const service_seconds = sample.source_ttfb_us / std.time.us_per_s +
            @as(f64, @floatFromInt(request_size)) / sample.source_body_bandwidth;
        const target_bandwidth = @max(
            @max(self.peak_logical_goodput, sample.logical_goodput),
            @min(sample.source_body_bandwidth, @max(self.peak_goodput, sample.committed_goodput)),
        );
        if (target_bandwidth <= 0 or service_seconds <= 0) return 0;
        const modeled: usize = @max(1, @as(usize, @intFromFloat(@ceil(
            1.2 * target_bandwidth * service_seconds / @as(f64, @floatFromInt(request_size)),
        ))));
        return @min(self.readCapacity(request_size), modeled);
    }

    fn sourceTupleSettled(self: *const AdaptiveVectoredController, now_ns: u64) bool {
        if (self.mode != .steady) return false;
        if (!self.request_size_settled) return false;
        return now_ns -| self.last_source_change_ns >= 500 * std.time.ns_per_ms;
    }

    fn observe(self: *AdaptiveVectoredController, sample: Sample) Decision {
        if (sample.slow_reads) self.slow_source_observed = true;
        if (sample.source_timing_successes > 0) self.source_timing_observed = true;
        if (sample.dma_starvation_ratio > 0.10) self.last_dma_starvation_ns = sample.now_ns;
        if (sample.hard_dma_pressure) {
            self.hard_dma_windows = @min(2, self.hard_dma_windows +| 1);
        } else {
            self.hard_dma_windows = 0;
        }
        if (sample.source_failure_reliable and sample.source_failure_ratio > 0.10) {
            self.source_failure_windows = @min(2, self.source_failure_windows +| 1);
        } else if (sample.source_failure_reliable) {
            self.source_failure_windows = 0;
        }

        if (sample.source_timing_reliable) {
            if (requestTimingBucketIndex(self.limits.request_size)) |index| {
                const baseline_ttfb = self.ttfb_baseline_us[index];
                const baseline_body_bandwidth = self.body_bandwidth_baseline[index];
                const timing_inflated = (baseline_ttfb > 0 and sample.source_ttfb_us > 1.5 * baseline_ttfb) or
                    (baseline_body_bandwidth > 0 and sample.source_body_bandwidth < 0.80 * baseline_body_bandwidth);
                const goodput_flat_or_regressing = self.peak_logical_goodput > 0 and
                    sample.logical_goodput <= 1.03 * self.peak_logical_goodput;
                if (timing_inflated and goodput_flat_or_regressing) {
                    self.source_timing_pressure_windows = @min(2, self.source_timing_pressure_windows +| 1);
                } else {
                    self.source_timing_pressure_windows = 0;
                }
                if (self.probe == null and (sample.source_low_load or baseline_body_bandwidth == 0)) {
                    if (sample.source_ttfb_us > 0) {
                        self.ttfb_baseline_us[index] = if (baseline_ttfb == 0)
                            sample.source_ttfb_us
                        else
                            0.90 * baseline_ttfb + 0.10 * sample.source_ttfb_us;
                    }
                    if (sample.source_body_bandwidth > 0) {
                        self.body_bandwidth_baseline[index] = if (baseline_body_bandwidth == 0)
                            sample.source_body_bandwidth
                        else
                            0.90 * baseline_body_bandwidth + 0.10 * sample.source_body_bandwidth;
                    }
                }
            }
        }

        const source_backoff_reason: ?Decision.Reason = if (sample.source_throttled)
            .source_throttle
        else if (self.source_failure_windows >= 2)
            .source_failures
        else
            null;
        if (sample.source_throttled) {
            self.performance_probe_blocked_until_ns = sample.now_ns +| 5 * std.time.ns_per_s;
            self.pressure_backoff_blocked_until_ns = sample.now_ns +| 5 * std.time.ns_per_s;
        }
        if (source_backoff_reason != null and self.read_adaptive and self.limits.read > 1) {
            if (self.probe) |probe| self.limits = probe.baseline;
            self.probe = null;
            self.mode = .steady;
            self.epoch += 1;
            self.limits.read = @max(@as(usize, 1), @as(usize, @intFromFloat(@floor(0.70 * @as(f64, @floatFromInt(self.limits.read))))));
            self.last_probe_ns = sample.now_ns;
            self.last_source_change_ns = sample.now_ns;
            self.clearReadGrowthBlock(self.limits.request_size);
            self.prefer_request_size_probe = false;
            self.rearmRequestSizeGrowth();
            self.performance_probe_blocked_until_ns = sample.now_ns +| 5 * std.time.ns_per_s;
            self.pressure_backoff_blocked_until_ns = sample.now_ns +| 5 * std.time.ns_per_s;
            self.source_failure_windows = 0;
            return self.decision(.read_backoff, true, source_backoff_reason.?, null, true);
        }

        if (self.probe) |*active_probe| {
            if (active_probe.dimension == .request_size) {
                active_probe.candidate_successes +|= sample.source_timing_successes;
                active_probe.candidate_bytes +|= sample.source_timing_bytes;
            }
            const probe = active_probe.*;
            if (!probe.activated) return self.currentDecision();
            const pressure_reason: ?Decision.Reason = if (probe.kind == .increase and
                (probe.dimension == .read or probe.dimension == .request_size) and sample.ready_pressure and
                (!self.slow_source_observed or sample.now_ns -| self.last_dma_starvation_ns >= 2 * std.time.ns_per_s))
                .ready_pressure
            else if ((probe.dimension == .read or probe.dimension == .request_size) and sample.source_throttled)
                .source_throttle
            else if (probe.dimension == .dma and sample.hard_dma_pressure)
                .dma_pressure
            else
                null;
            if (pressure_reason) |reason| {
                self.probe = null;
                self.limits = probe.baseline;
                self.epoch += 1;
                self.mode = .steady;
                self.last_probe_ns = sample.now_ns;
                if (probe.dimension == .request_size) {
                    self.request_size_settled = true;
                    self.prefer_request_size_probe = false;
                    self.last_source_change_ns = sample.now_ns;
                } else if (probe.dimension == .read) {
                    if (reason == .source_throttle) {
                        self.clearReadGrowthBlock(probe.baseline.request_size);
                        self.prefer_request_size_probe = false;
                    } else {
                        self.blockReadGrowth(
                            probe.baseline.request_size,
                            sample.now_ns +| 5 * std.time.ns_per_s,
                        );
                        self.prefer_request_size_probe = self.requestSizeCanGrow();
                    }
                    self.last_source_change_ns = sample.now_ns;
                    self.rearmRequestSizeGrowth();
                } else {
                    self.dma_settled = true;
                }
                const cooldown_ns: u64 = if (reason == .source_throttle)
                    5 * std.time.ns_per_s
                else
                    2 * std.time.ns_per_s;
                self.pressure_backoff_blocked_until_ns = sample.now_ns +| cooldown_ns;
                self.performance_probe_blocked_until_ns = sample.now_ns +| cooldown_ns;
                return self.decision(probeAction(probe.dimension, probe.kind, false), true, reason, null, true);
            }
            const candidate_byte_floor: u64 = if (probe.dimension == .request_size)
                @max(probe_byte_floor, @as(u64, @intCast(4 *| probe.candidate.request_size)))
            else
                probe_byte_floor;
            const candidate_samples_ready = switch (probe.dimension) {
                .read => sample.probe_read_operations >= 8 and sample.probe_read_bytes >= candidate_byte_floor,
                .request_size => probe.candidate_successes >= 8 and probe.candidate_bytes >= candidate_byte_floor,
                .dma => true,
            };
            const early_time_floor_ns: u64 = if (probe.dimension == .request_size)
                50 * std.time.ns_per_ms
            else
                probe_early_time_floor_ns;
            const full_time_floor_ns: u64 = if (probe.dimension == .request_size)
                100 * std.time.ns_per_ms
            else
                probe_time_floor_ns;
            const early_score = candidate_samples_ready and sample.probe_committed_bytes >= candidate_byte_floor and
                sample.probe_elapsed_ns >= early_time_floor_ns and
                self.probeResultIsDecisive(probe, sample);
            if (!early_score and
                (!candidate_samples_ready or sample.probe_committed_bytes < candidate_byte_floor or
                    sample.probe_elapsed_ns < full_time_floor_ns))
            {
                return self.currentDecision();
            }
            return self.scoreProbe(probe, sample);
        }

        if (sample.committed_goodput > 0) {
            self.representative_windows +|= 1;
            self.stable_goodput = if (self.stable_goodput == 0)
                sample.committed_goodput
            else
                0.90 * self.stable_goodput + 0.10 * sample.committed_goodput;
            self.peak_goodput = @max(self.peak_goodput, self.stable_goodput);
            if (sample.dma_latency_reliable and sample.dma_latency_us > 0) {
                self.stable_dma_latency_us = if (self.stable_dma_latency_us == 0)
                    sample.dma_latency_us
                else
                    0.95 * self.stable_dma_latency_us + 0.05 * sample.dma_latency_us;
            }
        }
        if (sample.logical_goodput > 0) {
            self.stable_logical_goodput = if (self.stable_logical_goodput == 0)
                sample.logical_goodput
            else
                0.90 * self.stable_logical_goodput + 0.10 * sample.logical_goodput;
            self.peak_logical_goodput = @max(self.peak_logical_goodput, self.stable_logical_goodput);
        }
        if (sample.read_latency_reliable and sample.read_latency_us > 0) {
            self.stable_read_latency_us = if (self.stable_read_latency_us == 0)
                sample.read_latency_us
            else
                0.95 * self.stable_read_latency_us + 0.05 * sample.read_latency_us;
        }
        if (sample.committed_goodput > 0 and sample.dma_saturated and
            (self.dma_started or self.mode == .steady))
        {
            self.dma_fed_windows = @min(2, self.dma_fed_windows +| 1);
        } else {
            self.dma_fed_windows = 0;
        }

        if (self.dma_adaptive and self.hard_dma_windows >= 2 and self.limits.dma > 1 and sample.now_ns >= self.pressure_backoff_blocked_until_ns) {
            self.mode = .steady;
            self.epoch += 1;
            self.limits.dma = @max(@as(usize, 1), @as(usize, @intFromFloat(@floor(0.70 * @as(f64, @floatFromInt(self.limits.dma))))));
            self.dma_settled = true;
            self.last_probe_ns = sample.now_ns;
            self.performance_probe_blocked_until_ns = sample.now_ns +| 2 * std.time.ns_per_s;
            self.pressure_backoff_blocked_until_ns = sample.now_ns +| 250 * std.time.ns_per_ms;
            return self.decision(.dma_backoff, true, .dma_pressure, null, false);
        }

        const performance_due = sample.now_ns >= self.performance_probe_blocked_until_ns and
            (self.mode == .startup or sample.now_ns -| self.last_probe_ns >= 2 * std.time.ns_per_s);
        const baseline = if (self.stable_goodput > 0) self.stable_goodput else sample.committed_goodput;
        const logical_baseline = if (self.stable_logical_goodput > 0)
            self.stable_logical_goodput
        else
            sample.logical_goodput;

        if (!self.dma_started and sample.allow_probe and self.mode == .startup and sample.source_stalled and
            self.read_adaptive and sample.read_saturated and
            self.limits.read < @min(self.bootstrap_read_limit, self.readCapacity(self.limits.request_size)))
        {
            // There is no representative output to score yet. Double read
            // fanout directly so high-latency sources can fill the pipe before
            // their first response arrives.
            self.epoch += 1;
            self.limits.read = @min(
                @min(self.bootstrap_read_limit, self.readCapacity(self.limits.request_size)),
                @max(self.limits.read + 1, self.limits.read *| 2),
            );
            self.last_probe_ns = sample.now_ns;
            self.last_startup_change_ns = sample.now_ns;
            return self.decision(.read_bootstrap, true, .none, null, false);
        }

        if (self.dma_started and sample.allow_probe and self.mode == .startup and
            self.read_adaptive and
            self.startup_read_probes < self.max_startup_read_probes and
            !(sample.high_latency_source and self.slow_source_observed) and
            sample.dma_starvation_ratio > 0.10 and sample.read_saturated and
            !sample.ready_pressure and self.source_timing_pressure_windows < 2 and
            self.limits.read < self.readGrowthCapacity())
        {
            // Remote sources have already had a chance to bootstrap while no
            // source response existed. Slow sources first establish the
            // bootstrap tuple's baseline; otherwise pipeline fill makes the
            // first scored increase look artificially good. Fast sources may
            // still take one startup step, and multi-device loads one more.
            self.startup_read_probes += 1;
            var candidate = self.limits;
            const modeled = self.modeledReadConcurrency(sample, self.limits.request_size);
            const growth_capacity = self.readGrowthCapacity();
            const startup_step = @min(
                growth_capacity,
                std.math.divCeil(usize, self.limits.read *| 3, 2) catch unreachable,
            );
            candidate.read = @min(
                growth_capacity,
                @max(self.increase(self.limits.read, growth_capacity), @max(modeled, startup_step)),
            );
            return self.startProbe(.read, .increase, candidate, logical_baseline, sample, .read_probe_start);
        }

        if (self.dma_started and self.mode == .startup and self.representative_windows >= 2 and
            sample.now_ns -| self.last_startup_change_ns >= 500 * std.time.ns_per_ms)
        {
            self.mode = .steady;
            self.last_probe_ns = sample.now_ns;
            self.performance_probe_blocked_until_ns = sample.now_ns +| 2 * std.time.ns_per_s;
            return self.decision(.startup_settle, false, .none, null, false);
        }

        if ((self.dma_started or self.mode == .steady) and sample.allow_probe and
            self.read_adaptive and
            self.mode == .steady and performance_due and
            sample.dma_starvation_ratio > 0.10 and sample.read_saturated and
            !sample.ready_pressure and self.source_timing_pressure_windows < 2 and
            !(self.prefer_request_size_probe and self.request_size_adaptive and
                !self.request_size_settled and sample.source_timing_reliable and
                self.limits.request_size < self.max_request_size) and
            !self.readGrowthIsBlocked(self.limits.request_size, sample.now_ns) and
            self.limits.read < self.readGrowthCapacity())
        {
            var candidate = self.limits;
            const growth_capacity = self.readGrowthCapacity();
            candidate.read = @min(
                growth_capacity,
                @max(
                    self.increase(self.limits.read, growth_capacity),
                    self.modeledReadConcurrency(sample, self.limits.request_size),
                ),
            );
            return self.startProbe(.read, .increase, candidate, logical_baseline, sample, .read_probe_start);
        }

        if (self.mode == .steady and self.request_size_adaptive and
            (!self.request_size_settled or self.dma_settled) and sample.allow_probe and performance_due and
            sample.source_timing_reliable and self.source_timing_pressure_windows < 2 and
            self.limits.request_size < self.max_request_size)
        {
            const candidate_size = @min(self.max_request_size, self.limits.request_size *| 2);
            const fallback_total_u64 = std.math.divCeil(
                u64,
                sample.remaining_bytes,
                @intCast(candidate_size),
            ) catch std.math.maxInt(u64);
            const fallback_full_u64 = sample.remaining_bytes / @as(u64, @intCast(candidate_size));
            const fallback_total: usize = @intCast(@min(fallback_total_u64, std.math.maxInt(usize)));
            const fallback_full: usize = @intCast(@min(fallback_full_u64, std.math.maxInt(usize)));
            const candidate_requests = if (sample.remaining_candidate_requests == std.math.maxInt(usize))
                fallback_total
            else
                sample.remaining_candidate_requests;
            const full_candidate_requests = if (sample.remaining_full_candidate_requests == std.math.maxInt(usize))
                fallback_full
            else
                sample.remaining_full_candidate_requests;
            var candidate = self.limits;
            candidate.request_size = candidate_size;
            const capacity = self.readCapacity(candidate_size);
            if (!self.read_adaptive and self.limits.read > capacity) {
                self.request_size_settled = true;
            } else {
                const modeled = self.modeledReadConcurrency(sample, candidate_size);
                candidate.read = if (self.read_adaptive)
                    @min(capacity, @max(@as(usize, 1), modeled))
                else
                    self.limits.read;
                const required_candidate_bytes: f64 = @floatFromInt(@max(
                    probe_byte_floor,
                    @as(u64, @intCast(8 *| candidate_size)),
                ));
                const reference_goodput = @max(
                    @max(self.peak_logical_goodput, self.stable_logical_goodput),
                    sample.logical_goodput,
                );
                const estimated_candidate_ns = if (reference_goodput > 0)
                    required_candidate_bytes / reference_goodput * std.time.ns_per_s
                else
                    std.math.inf(f64);
                const candidate_service_ns = if (sample.source_body_bandwidth > 0)
                    sample.source_ttfb_us * std.time.ns_per_us +
                        @as(f64, @floatFromInt(candidate_size)) / sample.source_body_bandwidth * std.time.ns_per_s
                else
                    std.math.inf(f64);
                if (full_candidate_requests >= 8 and candidate_requests >= candidate.read and
                    @max(estimated_candidate_ns, candidate_service_ns) <= 0.25 * sample.estimated_remaining_ns)
                {
                    return self.startProbe(
                        .request_size,
                        .increase,
                        candidate,
                        logical_baseline,
                        sample,
                        .size_probe_start,
                    );
                }
            }
            self.request_size_settled = true;
            self.prefer_request_size_probe = false;
        }

        // Do not compare startup initialization or a single fed burst with a
        // DMA candidate. Transfer-manager initialization and pipeline fill
        // systematically reward extra event credits unless the baseline has
        // remained both fed and saturated for two representative windows.
        if (self.dma_adaptive and self.sourceTupleSettled(sample.now_ns) and
            (self.dma_started or self.mode == .steady) and sample.allow_probe and performance_due and
            self.representative_windows >= 2 and self.dma_fed_windows >= 2 and baseline > 0 and
            sample.dma_saturated and sample.dma_probe_capacity and
            !sample.hard_dma_pressure and self.limits.dma < self.max_dma)
        {
            var candidate = self.limits;
            candidate.dma = self.increase(self.limits.dma, self.max_dma);
            return self.startProbe(
                .dma,
                .increase,
                candidate,
                @max(baseline, sample.committed_goodput),
                sample,
                .dma_probe_start,
            );
        }

        if (sample.ready_pressure and sample.dma_starvation_ratio <= 0.10 and
            (!self.slow_source_observed or sample.now_ns -| self.last_dma_starvation_ns >= 2 * std.time.ns_per_s) and
            self.read_adaptive and self.limits.read > 1 and sample.now_ns >= self.pressure_backoff_blocked_until_ns)
        {
            self.mode = .steady;
            self.epoch += 1;
            self.limits.read = @max(@as(usize, 1), @as(usize, @intFromFloat(@floor(0.70 * @as(f64, @floatFromInt(self.limits.read))))));
            self.last_probe_ns = sample.now_ns;
            self.last_source_change_ns = sample.now_ns;
            self.clearReadGrowthBlock(self.limits.request_size);
            self.prefer_request_size_probe = false;
            self.rearmRequestSizeGrowth();
            self.performance_probe_blocked_until_ns = sample.now_ns +| 2 * std.time.ns_per_s;
            self.pressure_backoff_blocked_until_ns = sample.now_ns +| 250 * std.time.ns_per_ms;
            return self.decision(.read_backoff, true, .ready_pressure, null, false);
        }

        const resource_due = sample.allow_probe and sample.allow_resource_probe and self.mode == .steady and
            sample.dma_starvation_ratio <= 0.10 and
            sample.now_ns -| self.last_dma_starvation_ns >= 2 * std.time.ns_per_s and
            sample.now_ns >= self.resource_probe_blocked_until_ns and
            sample.now_ns -| self.last_resource_probe_ns >= 2 * std.time.ns_per_s;
        if (resource_due) {
            if (self.read_adaptive and self.limits.read > 1) {
                var candidate = self.limits;
                candidate.read = @max(@as(usize, 1), std.math.divCeil(usize, self.limits.read, 2) catch unreachable);
                const resource_baseline = @max(self.peak_logical_goodput, self.stable_logical_goodput);
                return self.startProbe(.read, .reduce_resource, candidate, resource_baseline, sample, .read_reduce_start);
            }
            if (self.dma_adaptive and self.limits.dma > 1) {
                var candidate = self.limits;
                candidate.dma -= @max(@as(usize, 1), std.math.sqrt(self.limits.dma));
                const resource_baseline = @max(self.peak_goodput, self.stable_goodput);
                return self.startProbe(.dma, .reduce_resource, candidate, resource_baseline, sample, .dma_reduce_start);
            }
        }

        // Startup ends on a scored rollback or an explicit pressure backoff.
        // A quiet initialization window must not force the two-second steady
        // probe cadence before representative reads or transfers exist.
        if (self.mode == .startup) return self.currentDecision();
        return self.currentDecision();
    }

    fn activateProbe(self: *AdaptiveVectoredController, epoch: u64) bool {
        if (self.probe) |*probe| {
            if (probe.epoch != epoch or probe.activated) return false;
            probe.activated = true;
            return true;
        }
        return false;
    }

    fn markDmaStarted(self: *AdaptiveVectoredController, now_ns: u64) void {
        if (self.dma_started) return;
        self.dma_started = true;
        self.representative_windows = 0;
        self.dma_fed_windows = 0;
        self.stable_goodput = 0;
        self.peak_goodput = 0;
        self.stable_logical_goodput = 0;
        self.peak_logical_goodput = 0;
        self.stable_dma_latency_us = 0;
        self.stable_read_latency_us = 0;
        self.last_startup_change_ns = now_ns;
    }

    fn rollbackTimedOutProbe(self: *AdaptiveVectoredController, now_ns: u64) ?Decision {
        return self.rollbackProbe(now_ns, .probe_timeout, .capacity_not_exercised);
    }

    fn rollbackUnfinishedProbe(self: *AdaptiveVectoredController, now_ns: u64) ?Decision {
        return self.rollbackProbe(now_ns, .probe_tail_rollback, .finite_tail);
    }

    fn rollbackProbe(
        self: *AdaptiveVectoredController,
        now_ns: u64,
        action: Decision.Action,
        reason: Decision.Reason,
    ) ?Decision {
        const probe = self.probe orelse return null;
        self.probe = null;
        self.limits = probe.baseline;
        self.epoch += 1;
        self.mode = .steady;
        self.last_probe_ns = now_ns;
        if (probe.dimension == .request_size) {
            self.request_size_settled = true;
            self.prefer_request_size_probe = false;
        } else if (probe.dimension == .read and action == .probe_timeout) {
            self.prefer_request_size_probe = self.requestSizeCanGrow();
            self.rearmRequestSizeGrowth();
        }
        if (probe.dimension == .dma) self.dma_settled = true;
        if (probe.dimension == .read or probe.dimension == .request_size) self.last_source_change_ns = now_ns;
        self.performance_probe_blocked_until_ns = now_ns +| 2 * std.time.ns_per_s;
        return self.decision(action, true, reason, null, true);
    }

    fn probeResultIsDecisive(
        self: *const AdaptiveVectoredController,
        probe: Probe,
        sample: Sample,
    ) bool {
        _ = self;
        if (probe.baseline_goodput <= 0 or sample.probe_goodput <= 0) {
            return false;
        }

        // Only matching-epoch completions participate in probe decisions.
        // Measuring from activation keeps candidate startup latency in the
        // score instead of starting the clock at its first completion.
        const clear_gain = sample.probe_goodput >= 1.10 * probe.baseline_goodput and
            !sample.ready_pressure and !sample.hard_dma_pressure;
        const clear_loss = sample.probe_goodput <= 0.90 * probe.baseline_goodput;
        return clear_gain or clear_loss;
    }

    fn scoreProbe(self: *AdaptiveVectoredController, probe: Probe, sample: Sample) Decision {
        const candidate_goodput = sample.probe_goodput;
        const no_regression = probe.baseline_goodput == 0 or candidate_goodput >= 0.97 * probe.baseline_goodput;
        const starvation_improved = probe.baseline_starvation > 0.10 and
            sample.dma_starvation_ratio <= 0.10;
        const starvation_unresolved = probe.baseline_starvation > 0.10 and
            sample.dma_starvation_ratio > 0.10;
        const read_pressure_ok = !sample.ready_pressure or
            (self.slow_source_observed and sample.now_ns -| self.last_dma_starvation_ns < 2 * std.time.ns_per_s);
        const dimension_peak = switch (probe.dimension) {
            .read, .request_size => self.peak_logical_goodput,
            .dma => self.peak_goodput,
        };
        const keep = switch (probe.kind) {
            .increase => switch (probe.dimension) {
                .read => no_regression and read_pressure_ok and
                    (probe.baseline_goodput == 0 or candidate_goodput >= 1.03 * probe.baseline_goodput or
                        starvation_improved or (self.mode == .startup and starvation_unresolved)),
                .request_size => no_regression and read_pressure_ok and
                    (probe.baseline_goodput == 0 or candidate_goodput >= 1.03 * probe.baseline_goodput),
                .dma => !sample.hard_dma_pressure and
                    (probe.baseline_goodput == 0 or candidate_goodput >= 1.03 * probe.baseline_goodput),
            },
            .reduce_resource => candidate_goodput >= 0.97 * @max(probe.baseline_goodput, dimension_peak) and
                sample.dma_starvation_ratio <= 0.10 and !sample.ready_pressure and !sample.hard_dma_pressure,
        };

        self.probe = null;
        self.last_probe_ns = sample.now_ns;
        const action = probeAction(probe.dimension, probe.kind, keep);
        if (keep) {
            switch (probe.dimension) {
                .read, .request_size => {
                    self.stable_logical_goodput = candidate_goodput;
                    self.peak_logical_goodput = @max(self.peak_logical_goodput, candidate_goodput);
                },
                .dma => {
                    self.stable_goodput = candidate_goodput;
                    self.peak_goodput = @max(self.peak_goodput, candidate_goodput);
                },
            }
            if (probe.kind == .reduce_resource) {
                self.last_resource_probe_ns = sample.now_ns;
                self.mode = .steady;
            }
            if (probe.dimension == .read or probe.dimension == .request_size) {
                self.last_source_change_ns = sample.now_ns;
                self.dma_settled = !self.dma_adaptive or self.limits.dma >= self.max_dma;
                if (probe.dimension == .request_size) {
                    self.clearReadGrowthBlock(self.limits.request_size);
                    self.prefer_request_size_probe = false;
                    self.request_size_settled = self.limits.request_size >= self.max_request_size;
                    self.source_timing_pressure_windows = 0;
                } else {
                    const performance_gain = probe.baseline_goodput == 0 or
                        candidate_goodput >= 1.03 * probe.baseline_goodput;
                    if (performance_gain) {
                        self.clearReadGrowthBlock(self.limits.request_size);
                    } else {
                        self.blockReadGrowth(
                            self.limits.request_size,
                            sample.now_ns +| 5 * std.time.ns_per_s,
                        );
                    }
                    self.prefer_request_size_probe = self.requestSizeCanGrow();
                    self.rearmRequestSizeGrowth();
                }
            } else if (probe.dimension == .dma) {
                self.dma_settled = true;
            }
            const reason: Decision.Reason = if ((probe.dimension == .read or probe.dimension == .request_size) and
                candidate_goodput < 1.03 * probe.baseline_goodput)
                if (starvation_improved) .starvation_improved else if (starvation_unresolved) .starvation_unresolved else .none
            else
                .none;
            return self.decision(action, false, reason, null, true);
        }

        self.limits = probe.baseline;
        self.epoch += 1;
        self.mode = .steady;
        self.pressure_backoff_blocked_until_ns = sample.now_ns +| 250 * std.time.ns_per_ms;
        if (probe.kind == .increase) {
            self.performance_probe_blocked_until_ns = sample.now_ns +| 2 * std.time.ns_per_s;
        } else {
            self.last_resource_probe_ns = sample.now_ns;
            self.resource_probe_blocked_until_ns = sample.now_ns +| 5 * std.time.ns_per_s;
        }
        if (probe.dimension == .request_size) {
            self.request_size_settled = true;
            self.prefer_request_size_probe = false;
            self.last_source_change_ns = sample.now_ns;
            self.source_timing_pressure_windows = 0;
        } else if (probe.dimension == .read) {
            self.blockReadGrowth(
                probe.baseline.request_size,
                sample.now_ns +| 5 * std.time.ns_per_s,
            );
            self.prefer_request_size_probe = self.requestSizeCanGrow();
            self.rearmRequestSizeGrowth();
            self.last_source_change_ns = sample.now_ns;
        } else if (probe.dimension == .dma) {
            self.dma_settled = true;
        }
        const reason: Decision.Reason = if (sample.ready_pressure)
            .ready_pressure
        else if (sample.hard_dma_pressure)
            .dma_pressure
        else
            .gain_below_threshold;
        return self.decision(action, true, reason, null, true);
    }

    fn startProbe(
        self: *AdaptiveVectoredController,
        dimension: Dimension,
        kind: ProbeKind,
        candidate: Limits,
        baseline_goodput: f64,
        sample: Sample,
        action: Decision.Action,
    ) Decision {
        self.epoch += 1;
        self.probe = .{
            .dimension = dimension,
            .kind = kind,
            .baseline = self.limits,
            .candidate = candidate,
            .epoch = self.epoch,
            .baseline_goodput = baseline_goodput,
            .baseline_starvation = sample.dma_starvation_ratio,
        };
        self.limits = candidate;
        self.last_probe_ns = sample.now_ns;
        if (dimension == .read or dimension == .request_size) {
            self.last_source_change_ns = sample.now_ns;
            self.dma_settled = !self.dma_adaptive or candidate.dma >= self.max_dma;
            if (dimension == .request_size) {
                self.request_size_settled = false;
                self.source_timing_pressure_windows = 0;
                if (requestTimingBucketIndex(candidate.request_size)) |index| {
                    self.ttfb_baseline_us[index] = 0;
                    self.body_bandwidth_baseline[index] = 0;
                }
            }
        } else if (dimension == .dma) {
            self.dma_settled = false;
        }
        if (self.mode == .startup) self.last_startup_change_ns = sample.now_ns;
        return self.decision(action, true, .none, dimension, false);
    }

    fn increase(self: *const AdaptiveVectoredController, current: usize, maximum: usize) usize {
        const step = if (self.mode == .startup)
            @max(@as(usize, 4), std.math.sqrt(current))
        else
            @max(@as(usize, 1), std.math.sqrt(current));
        return @min(maximum, current +| step);
    }

    fn decision(
        self: *const AdaptiveVectoredController,
        action: Decision.Action,
        changed: bool,
        reason: Decision.Reason,
        started_probe: ?Dimension,
        finished_probe: bool,
    ) Decision {
        return .{
            .limits = self.limits,
            .epoch = self.epoch,
            .changed = changed,
            .action = action,
            .reason = reason,
            .started_probe = started_probe,
            .finished_probe = finished_probe,
        };
    }

    fn currentDecision(self: *const AdaptiveVectoredController) Decision {
        return self.decision(.none, false, .none, null, false);
    }

    fn probeAction(dimension: Dimension, kind: ProbeKind, keep: bool) Decision.Action {
        return switch (dimension) {
            .read => switch (kind) {
                .increase => if (keep) .read_probe_keep else .read_probe_rollback,
                .reduce_resource => if (keep) .read_reduce_keep else .read_reduce_rollback,
            },
            .request_size => switch (kind) {
                .increase => if (keep) .size_probe_keep else .size_probe_rollback,
                .reduce_resource => if (keep) .size_probe_keep else .size_probe_rollback,
            },
            .dma => switch (kind) {
                .increase => if (keep) .dma_probe_keep else .dma_probe_rollback,
                .reduce_resource => if (keep) .dma_reduce_keep else .dma_reduce_rollback,
            },
        };
    }
};

const VectoredReadStatsSource = struct {
    profile_id: usize,
    name: []const u8,
    provider: VFS.ReadStatsProvider,
    initial: VFS.ReadStats,
    previous: VFS.ReadStats,
};

const SourceTelemetry = struct {
    retries: u64 = 0,
    attempts: u64 = 0,
    timing_successes: u64 = 0,
    timing_bytes: u64 = 0,
    ttfb_ns: u64 = 0,
    body_ns: u64 = 0,
    transient_retries: u64 = 0,
    timeouts: u64 = 0,
    server_failures: u64 = 0,
    timing_transient_retries: u64 = 0,
    timing_timeouts: u64 = 0,
    timing_server_failures: u64 = 0,
    throttles: u64 = 0,
    retry_delay_ns: u64 = 0,

    fn failures(self: SourceTelemetry) u64 {
        return self.timing_transient_retries +| self.timing_timeouts +| self.timing_server_failures;
    }

    fn responseObserved(self: SourceTelemetry) bool {
        return self.timing_successes > 0 or self.transient_retries > 0 or self.timeouts > 0 or
            self.server_failures > 0 or self.throttles > 0;
    }

    fn hasActivity(self: SourceTelemetry) bool {
        return self.attempts > 0 or self.responseObserved();
    }

    fn ttfbMicroseconds(self: SourceTelemetry) f64 {
        if (self.timing_successes == 0) return 0;
        return @as(f64, @floatFromInt(self.ttfb_ns)) /
            @as(f64, @floatFromInt(self.timing_successes)) / std.time.ns_per_us;
    }

    fn bodyBandwidth(self: SourceTelemetry) f64 {
        if (self.body_ns == 0) return 0;
        return @as(f64, @floatFromInt(self.timing_bytes)) /
            (@as(f64, @floatFromInt(self.body_ns)) / std.time.ns_per_s);
    }

    fn addTiming(self: *SourceTelemetry, other: SourceTelemetry) void {
        self.timing_successes +|= other.timing_successes;
        self.timing_bytes +|= other.timing_bytes;
        self.ttfb_ns +|= other.ttfb_ns;
        self.body_ns +|= other.body_ns;
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

const AdaptiveVectoredRuntime = struct {
    controller: AdaptiveVectoredController,
    pipeline: *VectoredLoadPipeline,
    worker_gate: *AdaptiveRequestGate,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    post_read_slack_requests: usize,
    metrics: *VectoredLoadMetrics,
    scheduler: *VectoredReadScheduler,
    max_pinned_bytes: usize,
    total_physical_upper_bound: u64,
    read_stats_sources: []VectoredReadStatsSource,
    source_bootstrap_enabled: bool,
    source_bootstrap_interval_ns: u64,
    last_source_bootstrap_ns: u64 = 0,
    source_response_observed: bool = false,
    deferred_source_timing: SourceTelemetry = .{},
    failure_cohort_request_size: usize = 0,
    failure_cohort_attempts: u64 = 0,
    failure_cohort_failures: u64 = 0,
    done: std.Io.Event = .unset,
    probe_installed_at: std.Io.Timestamp,
    probe_activated_at: std.Io.Timestamp,
    source_pressure_windows: u8 = 0,

    const FailureFeedback = struct {
        ratio: f64 = 0,
        reliable: bool = false,
    };

    fn takeRemoteTelemetry(self: *AdaptiveVectoredRuntime, request_size: usize) SourceTelemetry {
        var result: SourceTelemetry = .{};
        const timing_index = requestTimingBucketIndex(request_size);
        for (self.read_stats_sources) |*source| {
            const current = source.provider.snapshot();
            const stats_delta = current.sub(source.previous);
            source.previous = current;
            result.retries +|= stats_delta.retries;
            result.transient_retries +|= stats_delta.transient_retries;
            result.timeouts +|= stats_delta.timeouts;
            result.server_failures +|= stats_delta.server_failures;
            result.throttles +|= stats_delta.throttles;
            result.retry_delay_ns +|= stats_delta.retry_delay_ns;
            if (timing_index) |index| {
                const timing = stats_delta.timing[index];
                result.attempts +|= timing.attempts;
                result.timing_successes +|= timing.successes;
                result.timing_bytes +|= timing.successful_bytes;
                result.ttfb_ns +|= timing.ttfb_ns;
                result.body_ns +|= timing.body_ns;
                result.timing_transient_retries +|= timing.transient_retries;
                result.timing_timeouts +|= timing.timeouts;
                result.timing_server_failures +|= timing.server_failures;
            }
        }
        return result;
    }

    fn resetFailureCohort(self: *AdaptiveVectoredRuntime, request_size: usize) void {
        self.failure_cohort_request_size = request_size;
        self.failure_cohort_attempts = 0;
        self.failure_cohort_failures = 0;
    }

    fn updateFailureCohort(
        self: *AdaptiveVectoredRuntime,
        request_size: usize,
        attempts: u64,
        failures: u64,
    ) FailureFeedback {
        if (self.failure_cohort_request_size != request_size) {
            self.resetFailureCohort(request_size);
        }
        self.failure_cohort_attempts +|= attempts;
        self.failure_cohort_failures +|= failures;
        const changed = attempts > 0 or failures > 0;
        // Bound the cohort so a long clean prefix cannot permanently hide a
        // later failure regime. Halving preserves the observed ratio while
        // retaining attempts whose outcomes arrive in a later sample.
        if (self.failure_cohort_attempts > 128) {
            self.failure_cohort_attempts = std.math.divCeil(u64, self.failure_cohort_attempts, 2) catch unreachable;
            self.failure_cohort_failures = std.math.divCeil(u64, self.failure_cohort_failures, 2) catch unreachable;
        }
        return .{
            .ratio = if (self.failure_cohort_attempts == 0)
                0
            else
                @as(f64, @floatFromInt(self.failure_cohort_failures)) /
                    @as(f64, @floatFromInt(self.failure_cohort_attempts)),
            // Do not count the same accumulated cohort as two reliable
            // windows when no new telemetry arrived.
            .reliable = changed and self.failure_cohort_attempts >= 8,
        };
    }

    fn run(self: *AdaptiveVectoredRuntime, io: std.Io) std.Io.Cancelable!void {
        const started: std.Io.Timestamp = .now(io, .awake);
        var window_started = started;
        var previous = self.metrics.snapshot();
        var previous_control = self.pipeline.controlSnapshot();
        var previous_post_read_bytes = previous.post_read_bytes;
        var previous_local_timing: [adaptive_request_sizes.len]VectoredLoadMetrics.LocalReadTiming = undefined;
        for (&previous_local_timing, 0..) |*timing, index| timing.* = self.metrics.localReadTiming(index);
        load_log.debug("adaptive controller started: mode={s}, reads={d}/{d}, request_size={Bi:.2}/{Bi:.2}, feasible_width={d}, retained_request_slack={d}, dma_per_device={d}/{d}", .{
            @tagName(self.controller.mode),
            self.controller.limits.read,
            self.controller.max_read,
            self.controller.limits.request_size,
            self.controller.max_request_size,
            self.controller.readCapacity(self.controller.limits.request_size),
            self.post_read_slack_requests,
            self.controller.limits.dma,
            self.controller.max_dma,
        });
        defer load_log.debug("adaptive controller stopped: mode={s}, reads={d}, request_size={Bi:.2}, feasible_width={d}, dma_per_device={d}, peak_goodput={d:.2}MiB/s", .{
            @tagName(self.controller.mode),
            self.controller.limits.read,
            self.controller.limits.request_size,
            self.controller.readCapacity(self.controller.limits.request_size),
            self.controller.limits.dma,
            self.controller.peak_goodput / (1024 * 1024),
        });

        while (!self.done.isSet()) {
            const sample_interval_ms: i64 = if (!self.controller.dma_started)
                @intCast(@min(@as(u64, 25), @max(@as(u64, 1), self.source_bootstrap_interval_ns / std.time.ns_per_ms)))
            else
                25;
            self.done.waitTimeout(io, .{ .duration = .{
                .raw = .fromMilliseconds(sample_interval_ms),
                .clock = .awake,
            } }) catch |err| switch (err) {
                error.Timeout => {},
                error.Canceled => return error.Canceled,
            };
            if (self.done.isSet()) break;

            const now: std.Io.Timestamp = .now(io, .awake);
            const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));
            if (!self.controller.dma_started) {
                const snapshot = self.metrics.snapshot();
                const control = self.pipeline.controlSnapshot();
                const source_telemetry = self.takeRemoteTelemetry(self.controller.limits.request_size);
                self.deferred_source_timing.addTiming(source_telemetry);
                if (snapshot.read_bytes > 0 or source_telemetry.responseObserved()) {
                    self.source_response_observed = true;
                }
                const failure_feedback = self.updateFailureCohort(
                    self.controller.limits.request_size,
                    source_telemetry.attempts,
                    source_telemetry.failures(),
                );
                if (source_telemetry.hasActivity()) {
                    const source_timing_reliable = self.deferred_source_timing.timing_successes >= 2 and
                        self.deferred_source_timing.timing_bytes >= @as(u64, @intCast(2 *| self.controller.limits.request_size));
                    const feedback_decision = self.controller.observe(.{
                        .now_ns = now_ns,
                        .source_throttled = source_telemetry.throttles > 0,
                        .source_failure_ratio = failure_feedback.ratio,
                        .source_failure_reliable = failure_feedback.reliable,
                        .source_timing_successes = self.deferred_source_timing.timing_successes,
                        .source_timing_bytes = self.deferred_source_timing.timing_bytes,
                        .source_ttfb_us = self.deferred_source_timing.ttfbMicroseconds(),
                        .source_body_bandwidth = self.deferred_source_timing.bodyBandwidth(),
                        .source_timing_reliable = source_timing_reliable,
                        .allow_probe = false,
                    });
                    if (feedback_decision.changed or feedback_decision.finished_probe) {
                        _ = self.applyDecision(io, feedback_decision, std.math.maxInt(u64));
                    }
                    if (source_telemetry.retries > 0 or source_telemetry.throttles > 0) {
                        load_log.debug("pre-DMA source retry feedback: retries={d}, transient={d}, timeouts={d}, server_failures={d}, throttles={d}, failure_ratio={d:.1}%, delay={d:.1}ms", .{
                            source_telemetry.retries,
                            source_telemetry.transient_retries,
                            source_telemetry.timeouts,
                            source_telemetry.server_failures,
                            source_telemetry.throttles,
                            failure_feedback.ratio * 100,
                            @as(f64, @floatFromInt(source_telemetry.retry_delay_ns)) / std.time.ns_per_ms,
                        });
                    }
                }
                const required_device_mask = self.pipeline.expected_device_mask.load(.acquire);
                if (snapshot.committed_bytes > 0 and
                    required_device_mask != 0 and
                    (control.completed_device_mask & required_device_mask) == required_device_mask)
                {
                    self.controller.markDmaStarted(now_ns);
                    previous = snapshot;
                    previous_control = control;
                    previous_post_read_bytes = snapshot.post_read_bytes;
                    self.metrics.resetReadPeak();
                    window_started = now;
                    load_log.debug("adaptive baseline started after every destination completed one PJRT event: elapsed={d:.3}s, requests={d}, dma_per_device={d}", .{
                        @as(f64, @floatFromInt(now_ns)) / std.time.ns_per_s,
                        self.controller.limits.read,
                        self.controller.limits.dma,
                    });
                    continue;
                }

                // Before the first source response, opening more requests is
                // the only way to cover a genuinely high-latency backend. Once
                // a read has completed, however, any lack of device progress
                // belongs to PJRT initialization and must not inflate the
                // source window.
                const source_calls_fill_window = shouldBootstrapSource(
                    self.source_bootstrap_enabled,
                    self.source_response_observed,
                    snapshot.read_bytes,
                    snapshot.outstanding_requests,
                    self.controller.limits.read,
                    self.scheduler.snapshot(io).has_unscheduled,
                );
                if (source_calls_fill_window and self.controller.limits.read < @min(self.controller.max_read, self.controller.bootstrap_read_limit) and
                    now_ns -| self.last_source_bootstrap_ns >= self.source_bootstrap_interval_ns)
                {
                    var decision = self.controller.observe(.{
                        .now_ns = now_ns,
                        .read_saturated = true,
                        .source_stalled = true,
                        .allow_probe = true,
                    });
                    if (decision.changed) {
                        decision = self.applyDecision(io, decision, std.math.maxInt(u64));
                    }
                    self.last_source_bootstrap_ns = now_ns;
                    load_log.debug("source no-progress bootstrap: requests={d}/{d}, active_source_calls={d}", .{
                        decision.limits.read,
                        self.controller.max_read,
                        snapshot.active_reads,
                    });
                }
                continue;
            }
            if (self.controller.probe) |probe| {
                if (!probe.activated) {
                    const control = self.pipeline.controlSnapshot();
                    const capacity_active = switch (probe.dimension) {
                        .read, .request_size => if (probe.kind == .increase)
                            self.metrics.probe_peak_reads.load(.acquire) >= probe.candidate.read
                        else
                            self.metrics.active_reads.load(.acquire) <= probe.candidate.read,
                        .dma => if (probe.kind == .increase)
                            control.probe_capacity_active
                        else
                            control.max_device_active <= probe.candidate.dma and control.probe_capacity_active,
                    };
                    if (capacity_active and self.controller.activateProbe(probe.epoch)) {
                        load_log.debug("probe capacity active: epoch={d}, dimension={s}, reads={d}, dma_per_device={d}", .{
                            probe.epoch,
                            @tagName(probe.dimension),
                            probe.candidate.read,
                            probe.candidate.dma,
                        });
                        previous = self.metrics.snapshot();
                        previous_control = control;
                        previous_post_read_bytes = previous.post_read_bytes;
                        window_started = now;
                        continue;
                    }
                    if (self.probe_installed_at.untilNow(io, .awake).nanoseconds >= 5 * std.time.ns_per_s) {
                        const old_probe_epoch = probe.epoch;
                        if (self.controller.rollbackTimedOutProbe(now_ns)) |decision| {
                            const applied = self.applyDecision(io, decision, old_probe_epoch);
                            load_log.debug("probe capacity timeout: epoch={d}, dimension={s}, reads={d}, dma_per_device={d}", .{
                                old_probe_epoch,
                                @tagName(probe.dimension),
                                applied.limits.read,
                                applied.limits.dma,
                            });
                        }
                        previous = self.metrics.snapshot();
                        previous_control = self.pipeline.controlSnapshot();
                        previous_post_read_bytes = previous.post_read_bytes;
                        window_started = now;
                        continue;
                    }
                }
            }

            const elapsed = window_started.untilNow(io, .awake);
            const elapsed_ns: u64 = @intCast(@max(elapsed.nanoseconds, 0));
            const startup = self.controller.mode == .startup;
            const size_probe_active = if (self.controller.probe) |probe|
                probe.dimension == .request_size and probe.activated
            else
                false;
            const min_ns: u64 = if (startup or size_probe_active) 50 * std.time.ns_per_ms else 100 * std.time.ns_per_ms;
            const max_ns: u64 = if (size_probe_active)
                100 * std.time.ns_per_ms
            else if (startup)
                100 * std.time.ns_per_ms
            else
                250 * std.time.ns_per_ms;
            if (elapsed_ns < min_ns) continue;

            const snapshot = self.metrics.snapshot();
            const delta = snapshot.sub(previous);
            const control = self.pipeline.controlSnapshot();
            const progress_bytes = @max(delta.read_bytes, delta.committed_bytes);
            const byte_floor: u64 = if (startup) 32 * 1024 * 1024 else 64 * 1024 * 1024;
            if (progress_bytes < byte_floor and elapsed_ns < max_ns) continue;

            const seconds = @as(f64, @floatFromInt(@max(elapsed_ns, 1))) / std.time.ns_per_s;
            const committed_goodput = @as(f64, @floatFromInt(delta.committed_bytes)) / seconds;
            const logical_goodput = @as(f64, @floatFromInt(delta.retired_bytes)) / seconds;
            const read_latency_us = if (delta.read_bytes == 0)
                0
            else
                @as(f64, @floatFromInt(delta.weighted_read_latency_us)) / @as(f64, @floatFromInt(delta.read_bytes));
            const average_read_bytes = if (delta.read_operations == 0)
                0
            else
                @as(f64, @floatFromInt(delta.read_bytes)) / @as(f64, @floatFromInt(delta.read_operations));
            const read_service_bandwidth = if (read_latency_us > 0)
                average_read_bytes / (read_latency_us / std.time.us_per_s)
            else
                0;
            const slow_reads = average_read_bytes >= 1024 * 1024 and read_service_bandwidth > 0 and
                read_service_bandwidth < 1.5 * 1024 * 1024 * 1024;
            const dma_latency_us = if (delta.committed_bytes == 0)
                0
            else
                @as(f64, @floatFromInt(delta.weighted_dma_latency_us)) / @as(f64, @floatFromInt(delta.committed_bytes));

            const active_slot_delta = control.active_slot_ns -| previous_control.active_slot_ns;
            const capacity_slot_delta = control.capacity_slot_ns -| previous_control.capacity_slot_ns;
            const integrated_dma_starvation = if (capacity_slot_delta == 0)
                1
            else
                1 - @min(1, @as(f64, @floatFromInt(active_slot_delta)) / @as(f64, @floatFromInt(capacity_slot_delta)));
            const instantaneous_dma_starvation = if (!self.controller.slow_source_observed and
                control.ready_entries == 0 and control.active_capacity > 0)
                1 - @min(1, @as(f64, @floatFromInt(control.active_events)) / @as(f64, @floatFromInt(control.active_capacity)))
            else
                0;
            const dma_starvation_ratio = @max(integrated_dma_starvation, instantaneous_dma_starvation);
            const source_pressure_capacity: u64 = @max(@as(u64, 1), snapshot.outstanding_request_bytes);
            const post_read_growth = snapshot.post_read_bytes -| previous_post_read_bytes;
            const post_read_growth_ratio = @as(f64, @floatFromInt(post_read_growth)) /
                @as(f64, @floatFromInt(source_pressure_capacity));
            const post_read_occupancy = @as(f64, @floatFromInt(snapshot.post_read_bytes)) /
                @as(f64, @floatFromInt(source_pressure_capacity));
            const post_read_age_pressure = control.post_read_requests >= 4 and
                control.post_read_oldest_age_ns > 250 * std.time.ns_per_ms;
            const read_latency_reliable = delta.read_bytes >= 32 * 1024 * 1024;
            const read_latency_pressure = read_latency_reliable and self.controller.stable_read_latency_us > 0 and
                read_latency_us > 1.5 * self.controller.stable_read_latency_us;
            const goodput_regressed = self.controller.peak_logical_goodput > 0 and
                logical_goodput < 0.97 * self.controller.peak_logical_goodput;
            const raw_source_pressure = goodput_regressed and
                (post_read_growth_ratio > 0.20 or (post_read_occupancy > 0.75 and post_read_age_pressure) or read_latency_pressure);
            if (raw_source_pressure and dma_starvation_ratio <= 0.10) {
                self.source_pressure_windows = @min(2, self.source_pressure_windows +| 1);
            } else {
                self.source_pressure_windows = 0;
            }
            const ready_pressure = self.source_pressure_windows >= 2;
            const read_saturated = snapshot.peak_reads >= self.controller.limits.read and
                self.scheduler.snapshot(io).has_unscheduled;
            const source_stalled = startup and progress_bytes < byte_floor and read_saturated and elapsed_ns >= max_ns;
            const dma_saturated = control.any_device_saturated and control.ready_entries > 0;
            const dma_latency_reliable = delta.committed_bytes >= 32 * 1024 * 1024;
            const hard_dma_pressure = dma_latency_reliable and self.controller.stable_dma_latency_us > 0 and
                dma_latency_us > 2 * self.controller.stable_dma_latency_us and self.controller.peak_goodput > 0 and
                committed_goodput < 0.95 * self.controller.peak_goodput and dma_saturated;

            var source_telemetry = self.takeRemoteTelemetry(self.controller.limits.request_size);
            source_telemetry.addTiming(self.deferred_source_timing);
            self.deferred_source_timing = .{};
            const source_retries = source_telemetry.retries;
            var source_timing_successes: u64 = 0;
            var source_timing_bytes: u64 = 0;
            var source_ttfb_ns: u64 = 0;
            var source_body_ns: u64 = 0;
            const source_transient_retries = source_telemetry.transient_retries;
            const source_timeouts = source_telemetry.timeouts;
            const source_server_failures = source_telemetry.server_failures;
            const source_throttles = source_telemetry.throttles;
            const source_retry_delay_ns = source_telemetry.retry_delay_ns;
            const timing_index = requestTimingBucketIndex(self.controller.limits.request_size);
            source_timing_successes +|= source_telemetry.timing_successes;
            source_timing_bytes +|= source_telemetry.timing_bytes;
            source_ttfb_ns +|= source_telemetry.ttfb_ns;
            source_body_ns +|= source_telemetry.body_ns;
            var local_timing_delta: [adaptive_request_sizes.len]VectoredLoadMetrics.LocalReadTiming = undefined;
            for (&local_timing_delta, &previous_local_timing, 0..) |*timing_delta, *old, index| {
                const current = self.metrics.localReadTiming(index);
                timing_delta.* = current.sub(old.*);
                old.* = current;
            }
            if (timing_index) |index| {
                const local = local_timing_delta[index];
                source_timing_successes +|= local.successes;
                source_timing_bytes +|= local.successful_bytes;
                source_body_ns +|= local.service_ns;
            }
            const source_ttfb_us = if (source_timing_successes == 0)
                0
            else
                @as(f64, @floatFromInt(source_ttfb_ns)) /
                    @as(f64, @floatFromInt(source_timing_successes)) / std.time.ns_per_us;
            const source_body_bandwidth = if (source_body_ns == 0)
                0
            else
                @as(f64, @floatFromInt(source_timing_bytes)) /
                    (@as(f64, @floatFromInt(source_body_ns)) / std.time.ns_per_s);
            const failure_feedback = self.updateFailureCohort(
                self.controller.limits.request_size,
                source_telemetry.attempts,
                source_telemetry.failures(),
            );
            const source_failure_ratio = failure_feedback.ratio;
            const source_timing_reliable = source_timing_successes >= 2 and
                source_timing_bytes >= @as(u64, @intCast(2 *| self.controller.limits.request_size));

            const remaining = self.total_physical_upper_bound -| snapshot.committed_bytes;
            const reference_goodput = @max(committed_goodput, self.controller.peak_goodput);
            const remaining_ns = if (reference_goodput > 0)
                @as(f64, @floatFromInt(remaining)) / reference_goodput * std.time.ns_per_s
            else
                std.math.inf(f64);
            const estimated_probe_ns = @max(
                @as(f64, 200 * std.time.ns_per_ms),
                @as(f64, @floatFromInt(source_ttfb_ns +| source_body_ns)) /
                    @as(f64, @floatFromInt(@max(source_timing_successes, 1))),
            );
            const scheduler_snapshot = self.scheduler.snapshot(io);
            const current_candidate_counts = self.scheduler.candidateCounts(
                io,
                self.controller.limits.request_size,
            );
            const next_request_size = @min(
                self.controller.max_request_size,
                self.controller.limits.request_size *| 2,
            );
            const next_candidate_counts = self.scheduler.candidateCounts(io, next_request_size);
            const allow_probe = remaining_ns > 250 * std.time.ns_per_ms and
                estimated_probe_ns <= 0.25 * remaining_ns and
                current_candidate_counts.total >= 4;
            const allow_resource_probe = remaining_ns > 2 * std.time.ns_per_s;

            var probe_goodput: f64 = 0;
            var probe_elapsed_ns: u64 = 0;
            if (self.controller.probe) |probe| {
                if (probe.activated) {
                    probe_elapsed_ns = @intCast(@max(
                        self.probe_activated_at.untilNow(io, .awake).nanoseconds,
                        1,
                    ));
                    const probe_seconds = @as(f64, @floatFromInt(probe_elapsed_ns)) / std.time.ns_per_s;
                    probe_goodput = @as(f64, @floatFromInt(snapshot.probe_committed_bytes)) / probe_seconds;
                }
            }

            const probe_baseline_goodput = if (self.controller.probe) |probe| probe.baseline_goodput else 0;
            const old_probe_epoch = if (self.controller.probe) |probe| probe.epoch else std.math.maxInt(u64);
            var decision = self.controller.observe(.{
                .now_ns = now_ns,
                .committed_goodput = committed_goodput,
                .logical_goodput = logical_goodput,
                .probe_goodput = probe_goodput,
                .probe_committed_bytes = snapshot.probe_committed_bytes,
                .probe_elapsed_ns = probe_elapsed_ns,
                .probe_read_operations = snapshot.probe_read_operations,
                .probe_read_bytes = snapshot.probe_read_bytes,
                .dma_latency_us = dma_latency_us,
                .dma_latency_reliable = dma_latency_reliable,
                .read_latency_us = read_latency_us,
                .read_latency_reliable = read_latency_reliable,
                .dma_starvation_ratio = dma_starvation_ratio,
                .high_latency_source = self.source_bootstrap_enabled,
                .read_saturated = read_saturated,
                .source_stalled = source_stalled,
                .dma_saturated = dma_saturated,
                .dma_probe_capacity = control.any_device_saturated,
                .ready_pressure = ready_pressure,
                .slow_reads = slow_reads,
                .source_throttled = source_throttles > 0,
                .source_failure_ratio = source_failure_ratio,
                .source_failure_reliable = failure_feedback.reliable,
                .source_timing_successes = source_timing_successes,
                .source_timing_bytes = source_timing_bytes,
                .source_ttfb_us = source_ttfb_us,
                .source_body_bandwidth = source_body_bandwidth,
                .source_timing_reliable = source_timing_reliable,
                .source_low_load = snapshot.active_reads <= @max(@as(usize, 2), self.controller.limits.read / 2),
                .remaining_bytes = scheduler_snapshot.remaining_bytes,
                .remaining_candidate_requests = next_candidate_counts.total,
                .remaining_full_candidate_requests = next_candidate_counts.full,
                .estimated_remaining_ns = remaining_ns,
                .hard_dma_pressure = hard_dma_pressure,
                .allow_probe = allow_probe,
                .allow_resource_probe = allow_resource_probe,
            });
            if (decision.started_probe == .dma and
                !self.pipeline.beginDmaProbe(decision.epoch, decision.limits.dma))
            {
                decision = self.controller.rollbackTimedOutProbe(now_ns).?;
            }
            if (decision.changed or decision.finished_probe) {
                decision = self.applyDecision(io, decision, old_probe_epoch);
            }

            load_log.debug("adaptive window: action={s}, reason={s}, mode={s}, epoch={d}, requests={d}/{d} in_use={d} reading={d} saturated={}, dma_per_device={d}/{d} active={d}/{d} saturated={}, physical={d:.2}MiB/s logical={d:.2}MiB/s, dma_latency={d:.1}us, starvation={d:.1}%, post_read={Bi:.2}/{Bi:.2} age={d:.1}ms count={d} pressure={}, ready={Bi:.2}/{d}, probe={d:.2}/{d:.2}MiB/s {Bi:.2}/{d:.1}ms", .{
                @tagName(decision.action),
                @tagName(decision.reason),
                @tagName(self.controller.mode),
                decision.epoch,
                decision.limits.read,
                self.controller.max_read,
                snapshot.outstanding_requests,
                snapshot.active_reads,
                read_saturated,
                decision.limits.dma,
                self.controller.max_dma,
                control.active_events,
                control.active_capacity,
                dma_saturated,
                committed_goodput / (1024 * 1024),
                logical_goodput / (1024 * 1024),
                dma_latency_us,
                dma_starvation_ratio * 100,
                snapshot.post_read_bytes,
                source_pressure_capacity,
                @as(f64, @floatFromInt(control.post_read_oldest_age_ns)) / std.time.ns_per_ms,
                control.post_read_requests,
                ready_pressure,
                snapshot.ready_bytes,
                control.ready_entries,
                probe_baseline_goodput / (1024 * 1024),
                probe_goodput / (1024 * 1024),
                snapshot.probe_committed_bytes,
                @as(f64, @floatFromInt(probe_elapsed_ns)) / std.time.ns_per_ms,
            });
            const average_body_us = if (source_timing_successes == 0)
                0
            else
                @as(f64, @floatFromInt(source_body_ns)) /
                    @as(f64, @floatFromInt(source_timing_successes)) / std.time.ns_per_us;
            const overhead_fraction = if (source_ttfb_us + average_body_us == 0)
                0
            else
                source_ttfb_us / (source_ttfb_us + average_body_us);
            load_log.debug("source tuple: reads={d}, request_size={Bi:.2}, feasible_width={d}, ttfb={d:.2}ms, body_bandwidth={d:.2}MiB/s, overhead={d:.1}%, timing_successes={d}, remaining={Bi:.2}", .{
                decision.limits.read,
                decision.limits.request_size,
                self.controller.readCapacity(decision.limits.request_size),
                source_ttfb_us / std.time.us_per_ms,
                source_body_bandwidth / (1024 * 1024),
                overhead_fraction * 100,
                source_timing_successes,
                scheduler_snapshot.remaining_bytes,
            });
            if (source_retries > 0 or source_throttles > 0) {
                load_log.debug("source retry feedback: retries={d}, transient={d}, timeouts={d}, server_failures={d}, throttles={d}, failure_ratio={d:.1}%, delay={d:.1}ms", .{
                    source_retries,
                    source_transient_retries,
                    source_timeouts,
                    source_server_failures,
                    source_throttles,
                    source_failure_ratio * 100,
                    @as(f64, @floatFromInt(source_retry_delay_ns)) / std.time.ns_per_ms,
                });
            }

            previous = snapshot;
            previous_control = control;
            previous_post_read_bytes = snapshot.post_read_bytes;
            self.metrics.resetReadPeak();
            window_started = now;
        }

        if (self.controller.probe) |probe| {
            const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));
            if (self.controller.rollbackUnfinishedProbe(now_ns)) |decision| {
                const applied = self.applyDecision(io, decision, probe.epoch);
                load_log.debug("adaptive tail rollback: epoch={d}, dimension={s}, reads={d}, dma_per_device={d}", .{
                    probe.epoch,
                    @tagName(probe.dimension),
                    applied.limits.read,
                    applied.limits.dma,
                });
            }
        }
    }

    fn applyDecision(
        self: *AdaptiveVectoredRuntime,
        io: std.Io,
        decision: AdaptiveVectoredController.Decision,
        old_probe_epoch: u64,
    ) AdaptiveVectoredController.Decision {
        var applied = decision;
        var scheduler_tuple_installed = false;
        if (decision.finished_probe and old_probe_epoch != std.math.maxInt(u64)) {
            self.metrics.clearProbe(io, old_probe_epoch);
            self.pipeline.clearDmaProbe(old_probe_epoch);
        }
        if (decision.started_probe) |dimension| {
            const metrics_dimension: VectoredLoadMetrics.ProbeDimension = switch (dimension) {
                .read => .read,
                .request_size => .request_size,
                .dma => .dma,
            };
            self.metrics.prepareProbe(io, decision.epoch, metrics_dimension);
            self.probe_installed_at = .now(io, .awake);
            self.probe_activated_at = self.probe_installed_at;
            if (dimension == .request_size) {
                scheduler_tuple_installed = self.scheduler.trySetCandidateTuple(
                    io,
                    decision.limits.request_size,
                    decision.epoch,
                    decision.limits.read,
                    8,
                );
                if (!scheduler_tuple_installed) {
                    self.metrics.clearProbe(io, decision.epoch);
                    applied = self.controller.rollbackUnfinishedProbe(self.controller.last_probe_ns).?;
                    self.metrics.config_epoch.store(applied.epoch, .release);
                }
            }
        } else if (decision.changed) {
            self.metrics.config_epoch.store(decision.epoch, .release);
        }
        if (self.failure_cohort_request_size != 0 and
            self.failure_cohort_request_size != applied.limits.request_size)
        {
            self.resetFailureCohort(applied.limits.request_size);
        } else if (applied.action == .read_backoff and
            (applied.reason == .source_throttle or applied.reason == .source_failures))
        {
            self.resetFailureCohort(applied.limits.request_size);
        }
        if (!scheduler_tuple_installed) {
            self.scheduler.setTuple(io, applied.limits.request_size, applied.epoch);
        }
        const gate_limits: PinnedGateLimits = .init(
            applied.limits.read,
            applied.limits.request_size,
            self.max_pinned_bytes,
            self.post_read_slack_requests,
        );
        self.worker_gate.setLimit(io, gate_limits.read);
        self.read_gate.setLimit(io, gate_limits.read);
        self.request_gate.setLimit(io, gate_limits.lifecycle);
        self.pipeline.setDmaLimit(applied.limits.dma);
        return applied;
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
        profile_id: usize,
        profile_name: []const u8,
        minimum_request_size: usize,
        high_latency: bool,
        read_stats: ?VFS.ReadStatsProvider,
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
            const profile = VFS.readProfileForPath(io, descriptor.file_uri);
            const minimum = if (profile) |p| p.hints.minimum_request_size else 2 * 1024 * 1024;
            switch (opts.read_request_size) {
                .adaptive => {},
                .fixed => |fixed| if (fixed < minimum) {
                    load_log.warn("fixed source request size {Bi:.2} is below the {Bi:.2} minimum advertised by {s}", .{
                        fixed,
                        minimum,
                        if (profile) |p| p.scheme else "local/default",
                    });
                },
            }
            try source_slots.append(allocator, .{
                .uri = descriptor.file_uri,
                .profile_id = if (profile) |p| p.id else 0,
                .profile_name = if (profile) |p| p.scheme else "local/default",
                .minimum_request_size = minimum,
                .high_latency = if (profile) |p| p.hints.high_latency else false,
                .read_stats = if (profile) |p| p.stats else null,
            });
            load_log.debug("source profile: name={s}, minimum_request_size={Bi:.2}, mode={s}, uri={s}", .{
                source_slots.items[index].profile_name,
                minimum,
                @tagName(opts.read_request_size),
                descriptor.file_uri,
            });
            break :blk index;
        };
    }

    var source_minimum: usize = opts.dma_block_size;
    var profile_ids: std.ArrayListUnmanaged(usize) = .empty;
    defer profile_ids.deinit(allocator);
    for (source_slots.items) |slot| {
        source_minimum = @max(source_minimum, slot.minimum_request_size);
        for (profile_ids.items) |profile_id| {
            if (profile_id == slot.profile_id) break;
        } else {
            try profile_ids.append(allocator, slot.profile_id);
        }
    }
    if (profile_ids.items.len > 1) {
        load_log.warn("mixed source profiles use one conservative adaptive tuple: profiles={d}, minimum_request_size={Bi:.2}", .{
            profile_ids.items.len,
            source_minimum,
        });
    }
    const initial_request_size = resolveReadRequestSize(opts.read_request_size, source_minimum, opts.dma_block_size);
    const maximum_request_size = switch (opts.read_request_size) {
        .adaptive => |adaptive| adaptive.maximum,
        .fixed => |fixed| fixed,
    };
    switch (opts.read_request_size) {
        .adaptive => {
            stdx.debug.assert(source_minimum <= max_load_read_request_size, "zml.io.load source minimum exceeds the absolute request-size limit", .{});
            stdx.debug.assert(initial_request_size >= source_minimum, "zml.io.load adaptive request-size initial is below the source minimum", .{});
            stdx.debug.assert(initial_request_size <= maximum_request_size, "zml.io.load source minimum/initial exceeds adaptive request-size maximum", .{});
        },
        .fixed => {},
    }
    stdx.debug.assert(initial_request_size <= opts.max_pinned_bytes, "zml.io.load initial request size exceeds max_pinned_bytes", .{});

    var read_stats_sources: std.ArrayListUnmanaged(VectoredReadStatsSource) = .empty;
    defer read_stats_sources.deinit(allocator);
    for (source_slots.items) |slot| {
        const provider = slot.read_stats orelse continue;
        for (read_stats_sources.items) |source| {
            if (source.profile_id == slot.profile_id) break;
        } else {
            const initial = provider.snapshot();
            try read_stats_sources.append(allocator, .{
                .profile_id = slot.profile_id,
                .name = slot.profile_name,
                .provider = provider,
                .initial = initial,
                .previous = initial,
            });
        }
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
            batch_iovecs_: bool,
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
                        batch_iovecs_,
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

    const high_latency_source = for (source_slots.items) |slot| {
        if (slot.high_latency) break true;
    } else false;
    // Remote backends need their active source-call cap to remain fully
    // occupied while the preceding requests spend a short time queued or in
    // PJRT. Eight retained lifecycles are 128 MiB for S3/GCS and 256 MiB for
    // HF; local reads keep strict lane-style coupling with no slack.
    const post_read_slack_requests: usize = if (high_latency_source) 8 else 0;

    const coordinator_started_at: std.Io.Timestamp = .now(io, .awake);
    load_log.debug("vectored coordinator started: tensors={d}, elapsed={d:.3}s", .{
        tensor_count,
        @as(f64, @floatFromInt(load_started.untilNow(io, .awake).nanoseconds)) / std.time.ns_per_s,
    });

    var metrics: VectoredLoadMetrics = .{};
    var controller: AdaptiveVectoredController = .initConfigured(
        opts.read_parallelism,
        opts.dma_parallelism,
        initial_request_size,
        maximum_request_size,
        opts.read_request_size.isAdaptive(),
        opts.max_pinned_bytes,
    );
    controller.max_startup_read_probes = if (platform.devices.len > 1) 2 else 1;
    const initial_gate_limits: PinnedGateLimits = .init(
        controller.limits.read,
        controller.limits.request_size,
        opts.max_pinned_bytes,
        post_read_slack_requests,
    );
    var worker_gate: AdaptiveRequestGate = .init(initial_gate_limits.read);
    var read_gate: AdaptiveRequestGate = .init(initial_gate_limits.read);
    var request_gate: AdaptiveRequestGate = .init(initial_gate_limits.lifecycle);
    var pipeline = try VectoredLoadPipeline.init(
        allocator,
        io,
        platform,
        &pool,
        &worker_gate,
        &read_gate,
        &request_gate,
        opts.dma_block_size,
        &metrics,
        controller.limits.dma,
    );
    defer pipeline.deinit();

    const tensor_sizes = try allocator.alloc(usize, tensor_count);
    defer allocator.free(tensor_sizes);
    for (tensors, tensor_sizes) |tensor, *size| size.* = tensor.byteSize();
    var scheduler = try VectoredReadScheduler.init(allocator, tensor_sizes, controller.limits.request_size, controller.epoch);
    defer scheduler.deinit();
    var worker_group: std.Io.Group = .init;
    var controller_runtime: AdaptiveVectoredRuntime = .{
        .controller = controller,
        .pipeline = &pipeline,
        .worker_gate = &worker_gate,
        .read_gate = &read_gate,
        .request_gate = &request_gate,
        .post_read_slack_requests = post_read_slack_requests,
        .metrics = &metrics,
        .scheduler = &scheduler,
        .max_pinned_bytes = opts.max_pinned_bytes,
        .total_physical_upper_bound = 0,
        .read_stats_sources = read_stats_sources.items,
        .source_bootstrap_enabled = high_latency_source,
        .source_bootstrap_interval_ns = if (high_latency_source)
            10 * std.time.ns_per_ms
        else
            100 * std.time.ns_per_ms,
        .probe_installed_at = .now(io, .awake),
        .probe_activated_at = .now(io, .awake),
    };
    // Resource/tail probes need the physical transfer total: sharded tensors
    // transfer one logical copy in aggregate, while replicas transfer one copy
    // to each destination.
    var physical_upper_bound: u64 = 0;
    for (tensors) |tensor| {
        const sharding = Sharding.pickSharding(opts.shardings, tensor.shape(), .explicit_axis_binding) orelse
            platform.replicated_sharding;
        const placement = try sharding.placement(tensor.shape());
        physical_upper_bound +|= @as(u64, @intCast(placement.shape.byteSize())) *|
            @as(u64, @intCast(sharding.devicesInCanonicalOrder().len));
    }
    controller_runtime.total_physical_upper_bound = physical_upper_bound;
    var controller_group: std.Io.Group = .init;
    try controller_group.concurrent(io, AdaptiveVectoredRuntime.run, .{ &controller_runtime, io });

    const worker_count = if (controller_runtime.total_physical_upper_bound == 0) 0 else opts.read_parallelism.maximum();
    for (0..worker_count) |worker_index| {
        worker_group.concurrent(io, struct {
            fn run(
                worker_index_: usize,
                worker_gate_: *AdaptiveRequestGate,
                scheduler_: *VectoredReadScheduler,
                request_gate_: *AdaptiveRequestGate,
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
                    if (!worker_gate_.waitUntilEnabled(io_, worker_index_)) return;
                    if (pipeline_.failed()) return;
                    if (!request_gate_.acquire(io_)) return;
                    const job = scheduler_.claim(io_) orelse {
                        request_gate_.release(io_);
                        worker_gate_.close(io_);
                        request_gate_.close(io_);
                        return;
                    };
                    const request = pipeline_.registerRequest(job.epoch, job.len) catch |err| {
                        request_gate_.release(io_);
                        pipeline_.recordError(err);
                        return;
                    };
                    const source_file = source_slots_[source_indices_[job.tensor_index]].ensure(io_) catch |err| {
                        request.finishScheduling();
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
                        !source_slots_[source_indices_[job.tensor_index]].high_latency,
                        shardings_,
                        buffers_[job.tensor_index],
                        progress_,
                    ) catch |err| {
                        request.finishScheduling();
                        pipeline_.recordError(err);
                        return;
                    };
                    pipeline_.expectTensorTargets(tensor);
                    VectoredReadRequest.run(
                        request,
                        tensor,
                        pipeline_,
                        job.source_offset,
                        job.len,
                        job.request_size,
                        source_slots_[source_indices_[job.tensor_index]].read_stats != null,
                    );
                }
            }
        }.run, .{
            worker_index,
            &worker_gate,
            &scheduler,
            &request_gate,
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
    worker_group.await(io) catch |err| pipeline.recordError(err);
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
    controller_runtime.done.set(io);
    controller_group.await(io) catch |err| pipeline.recordError(err);
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
    const average_read_ms = if (metrics.read_bytes.load(.acquire) == 0) 0 else @as(f64, @floatFromInt(metrics.weighted_read_latency_us.load(.acquire))) / @as(f64, @floatFromInt(metrics.read_bytes.load(.acquire))) / std.time.us_per_ms;
    const average_request_ms = if (metrics.retired_bytes.load(.acquire) == 0) 0 else @as(f64, @floatFromInt(metrics.weighted_request_latency_us.load(.acquire))) / @as(f64, @floatFromInt(metrics.retired_bytes.load(.acquire))) / std.time.us_per_ms;
    const average_dma_ms = if (metrics.committed_bytes.load(.acquire) == 0) 0 else @as(f64, @floatFromInt(metrics.weighted_dma_latency_us.load(.acquire))) / @as(f64, @floatFromInt(metrics.committed_bytes.load(.acquire))) / std.time.us_per_ms;
    const average_ready_ms = if (metrics.read_bytes.load(.acquire) == 0) 0 else @as(f64, @floatFromInt(metrics.weighted_ready_age_us.load(.acquire))) / @as(f64, @floatFromInt(metrics.read_bytes.load(.acquire))) / std.time.us_per_ms;
    const final_control = pipeline.controlSnapshot();
    var physical_source_requests: u64 = 0;
    var physical_source_bytes: u64 = 0;
    var source_retries: u64 = 0;
    var source_throttles: u64 = 0;
    var source_retry_delay_ns: u64 = 0;
    for (read_stats_sources.items) |source| {
        const source_stats = source.provider.snapshot().sub(source.initial);
        physical_source_requests +|= source_stats.physical_requests;
        physical_source_bytes +|= source_stats.physical_bytes;
        source_retries +|= source_stats.retries;
        source_throttles +|= source_stats.throttles;
        source_retry_delay_ns +|= source_stats.retry_delay_ns;
    }
    load_log.debug("completed: vectored=true, tensors={d}, logical_bytes={Bi:.2}, elapsed={d:.3}s, logical_goodput={d:.2}MiB/s, reads={d}, peak_requests={d}, final_requests={d}, final_request_size={Bi:.2}, feasible_width={d}, average_read={Bi:.2}, average_read_latency={d:.3}ms, average_request_lifetime={d:.3}ms, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}, source_retry_delay={d:.3}s, dma_submissions={d}, peak_dma_per_device={d}, final_dma_per_device={d}, average_dma={Bi:.2}, average_dma_latency={d:.3}ms, average_ready_age={d:.3}ms, submitted={Bi:.2}, committed={Bi:.2}, pinned_high_water={Bi:.2}, mapped={Bi:.2}, pool_waits={d}, pool_wait={d:.3}s", .{
        tensor_count,
        loaded_bytes,
        elapsed_seconds,
        goodput / (1024 * 1024),
        read_operations,
        metrics.request_high_water.load(.acquire),
        controller_runtime.controller.limits.read,
        controller_runtime.controller.limits.request_size,
        controller_runtime.controller.readCapacity(controller_runtime.controller.limits.request_size),
        average_read,
        average_read_ms,
        average_request_ms,
        physical_source_requests,
        physical_source_bytes,
        source_retries,
        source_throttles,
        @as(f64, @floatFromInt(source_retry_delay_ns)) / std.time.ns_per_s,
        dma_submissions,
        final_control.peak_device_active,
        controller_runtime.controller.limits.dma,
        average_dma,
        average_dma_ms,
        average_ready_ms,
        metrics.submitted_bytes.load(.acquire),
        metrics.committed_bytes.load(.acquire),
        pool.highWaterBytes(),
        pool.mappedBytes(),
        metrics.pool_waits.load(.acquire),
        @as(f64, @floatFromInt(metrics.pool_wait_ns.load(.acquire))) / std.time.ns_per_s,
    });
    return loaded_bytes;
}

pub const max_load_read_parallelism: usize = 128;
pub const max_load_dma_parallelism: usize = 32;
pub const max_load_read_request_size: usize = 128 * 1024 * 1024;

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

pub const ReadRequestSize = union(enum) {
    adaptive: Adaptive,
    fixed: usize,

    pub const Adaptive = struct {
        /// Null selects the conservative source minimum.
        initial: ?usize = null,
        maximum: usize = max_load_read_request_size,
    };

    fn isAdaptive(self: ReadRequestSize) bool {
        return switch (self) {
            .adaptive => true,
            .fixed => false,
        };
    }
};

fn roundUpRequestSize(bytes: usize) usize {
    const mib = 1024 * 1024;
    const whole_mib = std.math.divCeil(usize, @max(bytes, mib), mib) catch unreachable;
    const rounded_mib = std.math.ceilPowerOfTwo(usize, whole_mib) catch max_load_read_request_size / mib;
    return @min(rounded_mib *| mib, max_load_read_request_size);
}

fn resolveReadRequestSize(configured: ReadRequestSize, source_minimum: usize, dma_block_size: usize) usize {
    return switch (configured) {
        .adaptive => |adaptive| roundUpRequestSize(@max(adaptive.initial orelse source_minimum, @max(source_minimum, dma_block_size))),
        .fixed => |fixed| fixed,
    };
}

pub const LoadOpts = struct {
    pub const auto: LoadOpts = .{};

    /// Concurrent positional source requests.
    read_parallelism: Parallelism = .{ .adaptive = .{ .initial = 12, .maximum = max_load_read_parallelism } },
    /// In-flight PJRT transfers on each device.
    dma_parallelism: Parallelism = .{ .adaptive = .{ .initial = 8, .maximum = max_load_dma_parallelism } },
    /// Logical bytes gathered by one positional source request. Automatic
    /// sizing uses the greater of the source VFS minimum and the DMA block
    /// size. A fixed value must be at least one DMA block.
    read_request_size: ReadRequestSize = .{ .adaptive = .{} },
    /// Physical transfer and pool allocation unit.
    dma_block_size: usize = 2 * 1024 * 1024,
    /// Client-wide hard limit for registered host memory.
    max_pinned_bytes: usize = 2 * 1024 * 1024 * 1024,
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
        .group = .init(opts.read_parallelism.initial()),
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
    const read_initial = opts.read_parallelism.initial();
    const read_maximum = opts.read_parallelism.maximum();
    const dma_initial = opts.dma_parallelism.initial();
    const dma_maximum = opts.dma_parallelism.maximum();
    stdx.debug.assert(read_initial > 0, "zml.io.load read_parallelism initial/fixed value must be greater than zero", .{});
    stdx.debug.assert(read_maximum >= read_initial, "zml.io.load read_parallelism maximum must be at least initial", .{});
    stdx.debug.assert(read_maximum <= max_load_read_parallelism, "zml.io.load read_parallelism exceeds the absolute limit", .{});
    stdx.debug.assert(dma_initial > 0, "zml.io.load dma_parallelism initial/fixed value must be greater than zero", .{});
    stdx.debug.assert(dma_maximum >= dma_initial, "zml.io.load dma_parallelism maximum must be at least initial", .{});
    stdx.debug.assert(dma_maximum <= max_load_dma_parallelism, "zml.io.load dma_parallelism exceeds the absolute limit", .{});
    switch (opts.read_request_size) {
        .adaptive => |adaptive| {
            if (adaptive.initial) |initial| {
                stdx.debug.assert(initial > 0, "zml.io.load adaptive read_request_size initial must be greater than zero", .{});
                stdx.debug.assert(initial <= adaptive.maximum, "zml.io.load adaptive read_request_size maximum must be at least initial", .{});
            }
            stdx.debug.assert(adaptive.maximum > 0, "zml.io.load adaptive read_request_size maximum must be greater than zero", .{});
            stdx.debug.assert(adaptive.maximum <= max_load_read_request_size, "zml.io.load adaptive read_request_size exceeds the absolute limit", .{});
            stdx.debug.assert(adaptive.maximum % (1024 * 1024) == 0 and std.math.isPowerOfTwo(adaptive.maximum / (1024 * 1024)), "zml.io.load adaptive read_request_size maximum must be a power-of-two MiB value", .{});
            stdx.debug.assert(adaptive.maximum >= opts.dma_block_size, "zml.io.load adaptive read_request_size maximum must be at least dma_block_size", .{});
        },
        .fixed => |fixed| {
            stdx.debug.assert(fixed > 0, "zml.io.load fixed read_request_size must be greater than zero", .{});
            stdx.debug.assert(fixed >= opts.dma_block_size, "zml.io.load fixed read_request_size must be at least dma_block_size", .{});
            stdx.debug.assert(fixed <= max_load_read_request_size, "zml.io.load fixed read_request_size exceeds the absolute limit", .{});
            stdx.debug.assert(fixed <= opts.max_pinned_bytes, "zml.io.load fixed read_request_size exceeds max_pinned_bytes", .{});
        },
    }
    stdx.debug.assert(opts.dma_block_size > 0, "zml.io.load dma_block_size must be greater than zero", .{});
    stdx.debug.assert(opts.dma_block_size <= max_load_read_request_size, "zml.io.load dma_block_size exceeds the maximum request size", .{});
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
    load_log.debug("configured: target={s}, vectored={}, tensors={d}, max_read_parallelism={d}, max_dma_parallelism_per_device={d}, read_request_size={s}, fixed_read_request_size={Bi:.2}, dma_block_size={Bi:.2}, max_pinned_bytes={Bi:.2}, logical_bytes={Bi:.2}", .{
        @tagName(platform.target),
        direct,
        tensor_count,
        read_maximum,
        dma_maximum,
        @tagName(opts.read_request_size),
        switch (opts.read_request_size) {
            .adaptive => 0,
            .fixed => |fixed| fixed,
        },
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

test "adaptive vectored request size is power-of-two and never smaller than a DMA block" {
    try std.testing.expectEqual(
        @as(usize, 4 * 1024 * 1024),
        resolveReadRequestSize(.{ .adaptive = .{} }, 2 * 1024 * 1024, 4 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 16 * 1024 * 1024),
        resolveReadRequestSize(.{ .adaptive = .{} }, 10 * 1024 * 1024, 4 * 1024 * 1024),
    );
    try std.testing.expectEqual(
        @as(usize, 8 * 1024 * 1024),
        resolveReadRequestSize(.{ .fixed = 8 * 1024 * 1024 }, 16 * 1024 * 1024, 4 * 1024 * 1024),
    );
}

test "source bootstrap requires a high-latency source with no observed response" {
    try std.testing.expect(shouldBootstrapSource(true, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(false, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, true, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, false, 1, 12, 12, true));
}

test "source failure cohort joins attempts and failures across windows" {
    var runtime: AdaptiveVectoredRuntime = undefined;
    runtime.failure_cohort_request_size = 0;
    runtime.failure_cohort_attempts = 0;
    runtime.failure_cohort_failures = 0;

    const attempts = runtime.updateFailureCohort(16 * 1024 * 1024, 12, 0);
    try std.testing.expect(attempts.reliable);
    try std.testing.expectEqual(@as(f64, 0), attempts.ratio);

    const delayed_failures = runtime.updateFailureCohort(16 * 1024 * 1024, 0, 3);
    try std.testing.expect(delayed_failures.reliable);
    try std.testing.expectApproxEqAbs(@as(f64, 0.25), delayed_failures.ratio, 0.0001);

    const unchanged = runtime.updateFailureCohort(16 * 1024 * 1024, 0, 0);
    try std.testing.expect(!unchanged.reliable);

    const resized = runtime.updateFailureCohort(32 * 1024 * 1024, 8, 0);
    try std.testing.expect(resized.reliable);
    try std.testing.expectEqual(@as(f64, 0), resized.ratio);
}

test "vectored read scheduler applies size changes only to unscheduled ranges" {
    const io = std.testing.io;
    const sizes = [_]usize{ 10, 9, 8 };
    var scheduler = try VectoredReadScheduler.init(std.testing.allocator, &sizes, 4, 1);
    defer scheduler.deinit();

    const first = scheduler.claim(io).?;
    const second = scheduler.claim(io).?;
    const third = scheduler.claim(io).?;
    try std.testing.expectEqual(@as(usize, 0), first.tensor_index);
    try std.testing.expectEqual(@as(usize, 1), second.tensor_index);
    try std.testing.expectEqual(@as(usize, 2), third.tensor_index);
    try std.testing.expectEqual(@as(usize, 4), first.len);
    try std.testing.expectEqual(@as(u64, 1), first.epoch);

    scheduler.setTuple(io, 6, 2);
    const resized = scheduler.claim(io).?;
    try std.testing.expectEqual(@as(usize, 0), resized.tensor_index);
    try std.testing.expectEqual(@as(usize, 4), resized.source_offset);
    try std.testing.expectEqual(@as(usize, 6), resized.len);
    try std.testing.expectEqual(@as(usize, 6), resized.request_size);
    try std.testing.expectEqual(@as(u64, 2), resized.epoch);

    var total = first.len + second.len + third.len + resized.len;
    while (scheduler.claim(io)) |job| total += job.len;
    try std.testing.expectEqual(@as(usize, 27), total);
    try std.testing.expect(!scheduler.snapshot(io).has_unscheduled);
}

test "vectored read scheduler counts full candidate ranges per tensor" {
    const io = std.testing.io;
    const candidate_size = 4 * 1024 * 1024;
    const sizes = [_]usize{
        7 * candidate_size + 1,
        candidate_size - 1,
        candidate_size - 1,
    };
    var scheduler = try VectoredReadScheduler.init(std.testing.allocator, &sizes, 2 * 1024 * 1024, 1);
    defer scheduler.deinit();

    const counts = scheduler.candidateCounts(io, candidate_size);
    try std.testing.expectEqual(@as(usize, 10), counts.total);
    try std.testing.expectEqual(@as(usize, 7), counts.full);
}

test "vectored read scheduler checks tail eligibility while installing a candidate" {
    const io = std.testing.io;
    const candidate_size = 4 * 1024 * 1024;
    const sizes = [_]usize{8 * candidate_size};
    var scheduler = try VectoredReadScheduler.init(std.testing.allocator, &sizes, 2, 1);
    defer scheduler.deinit();

    _ = scheduler.claim(io).?;
    try std.testing.expect(!scheduler.trySetCandidateTuple(io, candidate_size, 2, 8, 8));
    const unchanged = scheduler.snapshot(io);
    try std.testing.expectEqual(@as(usize, 2), unchanged.request_size);
    try std.testing.expectEqual(@as(u64, 1), unchanged.epoch);
}

test "probe source capacity counts active reads rather than retained requests" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 7, .read);
    for (0..48) |_| metrics.beginRequest(7, 1);
    for (0..8) |_| metrics.beginRead(io, 7);

    try std.testing.expectEqual(@as(usize, 48), metrics.outstanding_requests.load(.acquire));
    try std.testing.expectEqual(@as(usize, 8), metrics.probe_peak_reads.load(.acquire));

    for (0..8) |_| metrics.endRead(io, 7);
    for (0..48) |_| metrics.endRequest(7, 1);
    metrics.clearProbe(io, 7);
}

test "pinned feasibility clips read width and remote lifecycle slack" {
    const clipped: PinnedGateLimits = .init(128, 128 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 8);
    try std.testing.expectEqual(@as(usize, 16), clipped.feasible_width);
    try std.testing.expectEqual(@as(usize, 16), clipped.read);
    try std.testing.expectEqual(@as(usize, 16), clipped.lifecycle);

    const slack: PinnedGateLimits = .init(12, 128 * 1024 * 1024, 2 * 1024 * 1024 * 1024, 8);
    try std.testing.expectEqual(@as(usize, 12), slack.read);
    try std.testing.expectEqual(@as(usize, 16), slack.lifecycle);
}

test "fixed controller dimensions remain immutable under pressure" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .fixed = 7 },
        .{ .fixed = 5 },
        16 * 1024 * 1024,
        16 * 1024 * 1024,
        false,
        2 * 1024 * 1024 * 1024,
    );
    _ = controller.observe(.{
        .now_ns = std.time.ns_per_s,
        .source_throttled = true,
        .hard_dma_pressure = true,
        .ready_pressure = true,
        .source_stalled = true,
        .read_saturated = true,
        .allow_probe = false,
    });
    const decision = controller.observe(.{
        .now_ns = 2 * std.time.ns_per_s,
        .source_throttled = true,
        .hard_dma_pressure = true,
        .ready_pressure = true,
        .source_stalled = true,
        .read_saturated = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 7), decision.limits.read);
    try std.testing.expectEqual(@as(usize, 5), decision.limits.dma);
    try std.testing.expectEqual(@as(usize, 16 * 1024 * 1024), decision.limits.request_size);
}

test "adaptive controller caps no-response bootstrap at thirty two" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 12, .maximum = 128 } },
        .{ .adaptive = .{ .initial = 8, .maximum = 32 } },
        16 * 1024 * 1024,
        128 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    _ = controller.observe(.{ .now_ns = 10 * std.time.ns_per_ms, .source_stalled = true, .read_saturated = true });
    _ = controller.observe(.{ .now_ns = 20 * std.time.ns_per_ms, .source_stalled = true, .read_saturated = true });
    const capped = controller.observe(.{ .now_ns = 30 * std.time.ns_per_ms, .source_stalled = true, .read_saturated = true });
    try std.testing.expectEqual(.none, capped.action);
    try std.testing.expectEqual(@as(usize, 32), capped.limits.read);
}

test "adaptive controller distinguishes transient failures from throttles" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    const transient = controller.observe(.{
        .now_ns = std.time.ns_per_s,
        .source_failure_ratio = 0.20,
        .source_failure_reliable = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.none, transient.action);
    const repeated = controller.observe(.{
        .now_ns = 2 * std.time.ns_per_s,
        .source_failure_ratio = 0.20,
        .source_failure_reliable = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.read_backoff, repeated.action);
    try std.testing.expectEqual(.source_failures, repeated.reason);

    var throttled: AdaptiveVectoredController = .init(32, 32);
    const immediate = throttled.observe(.{
        .now_ns = std.time.ns_per_s,
        .source_throttled = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.read_backoff, immediate.action);
    try std.testing.expectEqual(.source_throttle, immediate.reason);
}

test "adaptive request-size probes require candidate responses and matching bytes" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        .{ .fixed = 8 },
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .source_timing_reliable = true,
        .source_body_bandwidth = 1024 * 1024 * 1024,
        .remaining_bytes = 1024 * 1024 * 1024,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.size_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 4 * 1024 * 1024), decision.limits.request_size);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 100 * std.time.ns_per_ms,
        .logical_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 100 * std.time.ns_per_ms,
        .source_timing_successes = 7,
        .source_timing_bytes = 56 * 1024 * 1024,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.none, decision.action);
    decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 125 * std.time.ns_per_ms,
        .logical_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 125 * std.time.ns_per_ms,
        .source_timing_successes = 1,
        .source_timing_bytes = 8 * 1024 * 1024,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.size_probe_keep, decision.action);
}

test "adaptive request-size probes require eight remaining candidate requests" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        .{ .fixed = 8 },
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    const decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .source_timing_reliable = true,
        .source_body_bandwidth = 1024 * 1024 * 1024,
        .remaining_bytes = 7 * 4 * 1024 * 1024,
        .dma_starvation_ratio = 0,
        .allow_resource_probe = false,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expect(controller.request_size_settled);
}

test "adaptive request-size probes require eight full per-tensor candidates" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 12, .maximum = 32 } },
        .{ .fixed = 8 },
        2 * 1024 * 1024,
        4 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    const decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .source_timing_reliable = true,
        .source_body_bandwidth = 1024 * 1024 * 1024,
        .remaining_bytes = 8 * 4 * 1024 * 1024,
        .remaining_candidate_requests = 8,
        .remaining_full_candidate_requests = 7,
        .dma_starvation_ratio = 0,
        .allow_resource_probe = false,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expect(controller.request_size_settled);
}

test "adaptive request-size probes include candidate service time in tail cost" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 32, .maximum = 32 } },
        .{ .fixed = 8 },
        32 * 1024 * 1024,
        64 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 948 * 1024 * 1024;
    controller.peak_logical_goodput = 948 * 1024 * 1024;
    const decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 948 * 1024 * 1024,
        .source_timing_reliable = true,
        .source_body_bandwidth = 64 * 1024 * 1024,
        .remaining_bytes = 2 * 1024 * 1024 * 1024,
        .remaining_candidate_requests = 32,
        .remaining_full_candidate_requests = 32,
        .estimated_remaining_ns = 3 * std.time.ns_per_s,
        .dma_starvation_ratio = 0.50,
        .allow_resource_probe = false,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expect(controller.request_size_settled);
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
        .destination_offset = 20,
        .len = 20,
    };
    targets[0].submitted_bytes.store(0, .release);
    try std.testing.expect(pipeline.transferReady(non_final));
}

test "adaptive vectored controller scores one post-warm startup read growth" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.markDmaStarted(0);
    var decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 18), decision.limits.read);
    try std.testing.expectEqual(@as(usize, 8), decision.limits.dma);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 300 * std.time.ns_per_ms,
        .committed_goodput = 104,
        .logical_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_keep, decision.action);
    try std.testing.expectEqual(@as(usize, 18), controller.limits.read);

    decision = controller.observe(.{
        .now_ns = 350 * std.time.ns_per_ms,
        .committed_goodput = 104,
        .logical_goodput = 104,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 18), controller.limits.read);
}

test "adaptive vectored controller permits a second multi-device startup read probe" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.max_startup_read_probes = 2;
    controller.markDmaStarted(0);
    var decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));
    decision = controller.observe(.{
        .now_ns = 300 * std.time.ns_per_ms,
        .committed_goodput = 110,
        .logical_goodput = 110,
        .probe_goodput = 110,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0.50,
    });
    try std.testing.expectEqual(.read_probe_keep, decision.action);
    decision = controller.observe(.{
        .now_ns = 350 * std.time.ns_per_ms,
        .committed_goodput = 110,
        .logical_goodput = 110,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 27), decision.limits.read);
}

test "adaptive vectored controller bootstraps a no-response source to the public cap" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    var decision = controller.observe(.{
        .now_ns = 10 * std.time.ns_per_ms,
        .source_stalled = true,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_bootstrap, decision.action);
    try std.testing.expectEqual(@as(usize, 24), decision.limits.read);

    decision = controller.observe(.{
        .now_ns = 20 * std.time.ns_per_ms,
        .source_stalled = true,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_bootstrap, decision.action);
    try std.testing.expectEqual(@as(usize, 32), decision.limits.read);
    try std.testing.expect(controller.probe == null);
}

test "adaptive vectored controller holds at thirty two until source timing arrives" {
    var controller: AdaptiveVectoredController = .init(128, 32);
    controller.limits.read = 32;
    controller.markDmaStarted(0);

    var decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 32), decision.limits.read);

    decision = controller.observe(.{
        .now_ns = 200 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_successes = 1,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 48), decision.limits.read);
}

test "adaptive vectored controller lets a slow bootstrap tuple settle before probing" {
    var controller: AdaptiveVectoredController = .init(128, 32);
    controller.limits.read = 32;
    controller.markDmaStarted(0);
    const decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .high_latency_source = true,
        .read_saturated = true,
        .slow_reads = true,
        .source_timing_successes = 8,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 32), decision.limits.read);
}

test "adaptive vectored controller retains startup probing for a slow local source" {
    var controller: AdaptiveVectoredController = .init(128, 32);
    controller.limits.read = 32;
    controller.markDmaStarted(0);
    const decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .slow_reads = true,
        .source_timing_successes = 8,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(decision.limits.read > 32);
}

test "adaptive vectored controller advances to size after a flat read probe" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 32, .maximum = 128 } },
        .{ .fixed = 8 },
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_successes = 8,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 4 * std.time.ns_per_s,
        .probe_goodput = 100,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0.50,
    });
    try std.testing.expectEqual(.read_probe_rollback, decision.action);
    try std.testing.expect(controller.readGrowthIsBlocked(
        16 * 1024 * 1024,
        4 * std.time.ns_per_s,
    ));
    try std.testing.expect(controller.prefer_request_size_probe);

    decision = controller.observe(.{
        .now_ns = 7 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_reliable = true,
        .source_body_bandwidth = 32 * 1024 * 1024,
        .source_ttfb_us = 50 * std.time.us_per_ms,
        .remaining_bytes = 8 * 1024 * 1024 * 1024,
        .remaining_candidate_requests = 256,
        .remaining_full_candidate_requests = 256,
    });
    try std.testing.expectEqual(.size_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 32 * 1024 * 1024), decision.limits.request_size);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 8 * std.time.ns_per_s,
        .probe_goodput = 90,
        .probe_committed_bytes = 128 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .source_timing_successes = 8,
        .source_timing_bytes = 256 * 1024 * 1024,
        .dma_starvation_ratio = 0.50,
    });
    try std.testing.expectEqual(.size_probe_rollback, decision.action);
    try std.testing.expect(!controller.prefer_request_size_probe);

    decision = controller.observe(.{
        .now_ns = 10 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_successes = 8,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
}

test "adaptive read results keep an already maximum request size settled" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 32, .maximum = 128 } },
        .{ .fixed = 8 },
        32 * 1024 * 1024,
        32 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_successes = 8,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 4 * std.time.ns_per_s,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0.50,
    });
    try std.testing.expectEqual(.read_probe_keep, decision.action);
    try std.testing.expect(controller.request_size_settled);
    try std.testing.expect(!controller.prefer_request_size_probe);
    try std.testing.expect(controller.sourceTupleSettled(
        4 * std.time.ns_per_s + 500 * std.time.ns_per_ms,
    ));
}

test "adaptive vectored controller scores starvation-driven read growth after startup" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 15), decision.limits.read);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 4 * std.time.ns_per_s,
        .committed_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0.08,
    });
    try std.testing.expectEqual(.read_probe_keep, decision.action);
    try std.testing.expectEqual(@as(usize, 15), controller.limits.read);
}

test "adaptive vectored controller rolls back a flat DMA probe" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.stable_goodput = 100;
    controller.peak_goodput = 100;
    controller.representative_windows = 2;
    controller.dma_fed_windows = 1;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 10), decision.limits.dma);
    try std.testing.expect(controller.activateProbe(decision.epoch));
    const candidate_epoch = decision.epoch;

    decision = controller.observe(.{
        .now_ns = 400 * std.time.ns_per_ms,
        .committed_goodput = 101,
        .probe_goodput = 101,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.dma_probe_rollback, decision.action);
    try std.testing.expectEqual(@as(usize, 8), decision.limits.dma);
    try std.testing.expect(decision.epoch > candidate_epoch);
}

test "adaptive vectored controller scores only decisive probes at 100 ms" {
    var gain: AdaptiveVectoredController = .init(32, 32);
    gain.mode = .steady;
    gain.stable_goodput = 100;
    gain.peak_goodput = 100;
    gain.representative_windows = 2;
    gain.dma_fed_windows = 1;
    var decision = gain.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expect(gain.activateProbe(decision.epoch));
    decision = gain.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 100 * std.time.ns_per_ms,
        .committed_goodput = 112,
        .probe_goodput = 112,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 100 * std.time.ns_per_ms,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.dma_probe_keep, decision.action);

    var ambiguous: AdaptiveVectoredController = .init(32, 32);
    ambiguous.mode = .steady;
    ambiguous.stable_goodput = 100;
    ambiguous.peak_goodput = 100;
    ambiguous.representative_windows = 2;
    ambiguous.dma_fed_windows = 1;
    decision = ambiguous.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expect(ambiguous.activateProbe(decision.epoch));
    decision = ambiguous.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 100 * std.time.ns_per_ms,
        .committed_goodput = 105,
        .probe_goodput = 105,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 100 * std.time.ns_per_ms,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expect(ambiguous.probe != null);

    var loss: AdaptiveVectoredController = .init(32, 32);
    loss.mode = .steady;
    loss.stable_goodput = 100;
    loss.peak_goodput = 100;
    loss.representative_windows = 2;
    loss.dma_fed_windows = 1;
    decision = loss.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expect(loss.activateProbe(decision.epoch));
    decision = loss.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 100 * std.time.ns_per_ms,
        .committed_goodput = 88,
        .probe_goodput = 88,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 100 * std.time.ns_per_ms,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.dma_probe_rollback, decision.action);
    try std.testing.expectEqual(@as(usize, 8), decision.limits.dma);
}

test "adaptive vectored controller qualifies a DMA baseline before probing" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.resource_probe_blocked_until_ns = 10 * std.time.ns_per_s;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 25,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.none, decision.action);

    decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(@as(f64, 100), controller.probe.?.baseline_goodput);
}

test "adaptive vectored controller waits for the source tuple to settle before DMA probing" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    var decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .dma_saturated = true,
        .dma_probe_capacity = true,
        .slow_reads = true,
    });
    try std.testing.expectEqual(.none, decision.action);
    try std.testing.expectEqual(@as(usize, 12), decision.limits.read);
    try std.testing.expectEqual(@as(usize, 8), decision.limits.dma);

    controller.markDmaStarted(100 * std.time.ns_per_ms);
    decision = controller.observe(.{
        .now_ns = 150 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.none, decision.action);
    decision = controller.observe(.{
        .now_ns = 200 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.none, decision.action);
    controller.mode = .steady;
    decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expectEqual(@as(usize, 10), decision.limits.dma);
}

test "adaptive vectored controller settles after startup read fanout is quiet" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.markDmaStarted(0);
    var decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 300 * std.time.ns_per_ms,
        .committed_goodput = 104,
        .logical_goodput = 104,
        .probe_goodput = 104,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.read_probe_keep, decision.action);

    decision = controller.observe(.{
        .now_ns = 700 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.startup_settle, decision.action);
    try std.testing.expectEqual(.steady, controller.mode);
    try std.testing.expectEqual(@as(usize, 18), decision.limits.read);
    try std.testing.expect(controller.probe == null);
}

test "adaptive vectored controller ignores transient DMA saturation between source bursts" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.markDmaStarted(0);
    controller.stable_goodput = 100;
    controller.peak_goodput = 100;
    controller.representative_windows = 2;
    var decision = controller.observe(.{
        .now_ns = 50 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.none, decision.action);

    decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 150 * std.time.ns_per_ms,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.none, decision.action);
}

test "adaptive vectored controller respects caps and bootstraps a stalled source" {
    var capped: AdaptiveVectoredController = .init(4, 2);
    try std.testing.expectEqual(@as(usize, 4), capped.limits.read);
    try std.testing.expectEqual(@as(usize, 2), capped.limits.dma);
    const unchanged = capped.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .dma_starvation_ratio = 1,
        .read_saturated = true,
        .source_stalled = true,
    });
    try std.testing.expectEqual(.none, unchanged.action);

    var controller: AdaptiveVectoredController = .init(32, 32);
    const decision = controller.observe(.{
        .now_ns = 100 * std.time.ns_per_ms,
        .dma_starvation_ratio = 1,
        .read_saturated = true,
        .source_stalled = true,
    });
    try std.testing.expectEqual(.read_bootstrap, decision.action);
    try std.testing.expectEqual(@as(usize, 24), decision.limits.read);
    try std.testing.expect(controller.probe == null);
    const doubled_again = controller.observe(.{
        .now_ns = 200 * std.time.ns_per_ms,
        .dma_starvation_ratio = 1,
        .read_saturated = true,
        .source_stalled = true,
    });
    try std.testing.expectEqual(.read_bootstrap, doubled_again.action);
    try std.testing.expectEqual(@as(usize, 32), doubled_again.limits.read);
}

test "adaptive vectored controller keeps a three percent DMA gain" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.stable_goodput = 100;
    controller.peak_goodput = 100;
    controller.representative_windows = 2;
    controller.dma_fed_windows = 1;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .dma_saturated = true,
        .dma_probe_capacity = true,
    });
    try std.testing.expectEqual(.dma_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));
    decision = controller.observe(.{
        .now_ns = 400 * std.time.ns_per_ms,
        .probe_goodput = 103,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.dma_probe_keep, decision.action);
    try std.testing.expectEqual(@as(usize, 10), decision.limits.dma);
}

test "adaptive vectored controller suppresses probes at the finite tail and during cooldown" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.stable_goodput = 100;
    controller.performance_probe_blocked_until_ns = 3 * std.time.ns_per_s;
    var decision = controller.observe(.{
        .now_ns = 2 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.5,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.none, decision.action);
    decision = controller.observe(.{
        .now_ns = 4 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.5,
        .read_saturated = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.none, decision.action);
}

test "adaptive vectored controller protects a slow bursty source from early read backoff" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.limits.read = 16;
    _ = controller.observe(.{
        .now_ns = std.time.ns_per_s,
        .slow_reads = true,
        .dma_starvation_ratio = 0.5,
        .allow_probe = false,
    });
    const protected = controller.observe(.{
        .now_ns = 2 * std.time.ns_per_s,
        .dma_starvation_ratio = 0,
        .ready_pressure = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.none, protected.action);
    const backed_off = controller.observe(.{
        .now_ns = 4 * std.time.ns_per_s,
        .dma_starvation_ratio = 0,
        .ready_pressure = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.read_backoff, backed_off.action);
}

test "vectored probe metrics retain install-time bytes and ignore wrong epochs" {
    const io = std.testing.io;
    var metrics: VectoredLoadMetrics = .{};
    metrics.prepareProbe(io, 7, .read);
    metrics.recordProbeCommit(io, .read, 6, 16);
    metrics.recordProbeCommit(io, .dma, 7, 16);
    metrics.recordProbeCommit(io, .read, 7, 32);
    try std.testing.expectEqual(@as(u64, 32), metrics.probe_committed_bytes.load(.acquire));
    metrics.recordProbeCommit(io, .read, 6, 64);
    metrics.recordProbeCommit(io, .dma, 7, 64);
    metrics.recordProbeCommit(io, .read, 7, 128);
    try std.testing.expectEqual(@as(u64, 160), metrics.probe_committed_bytes.load(.acquire));
    metrics.clearProbe(io, 7);
}

test "adaptive probes score only matching-epoch committed goodput" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.5,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s + 200 * std.time.ns_per_ms,
        .probe_goodput = 96,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0.5,
    });
    try std.testing.expectEqual(.read_probe_rollback, decision.action);
}

test "adaptive vectored controller backs off sustained DMA pressure" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.stable_goodput = 100;
    controller.peak_goodput = 100;
    _ = controller.observe(.{
        .now_ns = std.time.ns_per_s,
        .committed_goodput = 90,
        .hard_dma_pressure = true,
        .allow_probe = false,
    });
    const decision = controller.observe(.{
        .now_ns = std.time.ns_per_s + 100 * std.time.ns_per_ms,
        .committed_goodput = 90,
        .hard_dma_pressure = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.dma_backoff, decision.action);
    try std.testing.expectEqual(@as(usize, 5), decision.limits.dma);
}

test "adaptive vectored controller backs reads off only on downstream pressure" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.limits.read = 16;
    controller.last_dma_starvation_ns = 0;
    const decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
        .ready_pressure = true,
        .allow_probe = false,
    });
    try std.testing.expectEqual(.read_backoff, decision.action);
    try std.testing.expectEqual(@as(usize, 11), decision.limits.read);
}

test "adaptive vectored controller keeps a lower-resource limit within three percent" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.limits.read = 8;
    controller.stable_goodput = 100;
    controller.peak_goodput = 100;
    var decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.read_reduce_start, decision.action);
    try std.testing.expectEqual(@as(usize, 4), decision.limits.read);
    try std.testing.expect(controller.activateProbe(decision.epoch));

    decision = controller.observe(.{
        .now_ns = 4 * std.time.ns_per_s,
        .committed_goodput = 98,
        .probe_goodput = 98,
        .probe_committed_bytes = 64 * 1024 * 1024,
        .probe_elapsed_ns = 200 * std.time.ns_per_ms,
        .probe_read_operations = 8,
        .probe_read_bytes = 64 * 1024 * 1024,
        .dma_starvation_ratio = 0,
    });
    try std.testing.expectEqual(.read_reduce_keep, decision.action);
    try std.testing.expectEqual(@as(usize, 4), controller.limits.read);
}

test "adaptive vectored controller restores limits when capacity cannot activate" {
    var controller: AdaptiveVectoredController = .init(16, 12);
    controller.mode = .steady;
    const decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.5,
        .read_saturated = true,
    });
    try std.testing.expectEqual(@as(usize, 15), decision.limits.read);
    const rollback = controller.rollbackTimedOutProbe(6 * std.time.ns_per_s).?;
    try std.testing.expectEqual(.probe_timeout, rollback.action);
    try std.testing.expectEqual(@as(usize, 12), rollback.limits.read);
    try std.testing.expectEqual(@as(usize, 8), rollback.limits.dma);
    try std.testing.expect(!controller.readGrowthIsBlocked(
        rollback.limits.request_size,
        6 * std.time.ns_per_s,
    ));
    try std.testing.expect(!controller.prefer_request_size_probe);
}

test "an unexercised read probe yields one turn to request-size discovery" {
    var controller: AdaptiveVectoredController = .initConfigured(
        .{ .adaptive = .{ .initial = 32, .maximum = 128 } },
        .{ .fixed = 8 },
        16 * 1024 * 1024,
        32 * 1024 * 1024,
        true,
        2 * 1024 * 1024 * 1024,
    );
    controller.mode = .steady;
    controller.dma_started = true;
    controller.stable_logical_goodput = 100;
    controller.peak_logical_goodput = 100;
    const read_probe = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_successes = 8,
    });
    try std.testing.expectEqual(.read_probe_start, read_probe.action);

    const rollback = controller.rollbackTimedOutProbe(6 * std.time.ns_per_s).?;
    try std.testing.expectEqual(.probe_timeout, rollback.action);
    try std.testing.expect(controller.prefer_request_size_probe);
    try std.testing.expect(!controller.readGrowthIsBlocked(
        rollback.limits.request_size,
        6 * std.time.ns_per_s,
    ));

    const size_probe = controller.observe(.{
        .now_ns = 8 * std.time.ns_per_s,
        .logical_goodput = 100,
        .dma_starvation_ratio = 0.50,
        .read_saturated = true,
        .source_timing_reliable = true,
        .source_body_bandwidth = 32 * 1024 * 1024,
        .source_ttfb_us = 50 * std.time.us_per_ms,
        .remaining_bytes = 8 * 1024 * 1024 * 1024,
        .remaining_candidate_requests = 256,
        .remaining_full_candidate_requests = 256,
    });
    try std.testing.expectEqual(.size_probe_start, size_probe.action);
}

test "adaptive vectored controller restores the complete tuple at the finite tail" {
    var controller: AdaptiveVectoredController = .init(32, 32);
    controller.mode = .steady;
    controller.limits = .{ .read = 12, .dma = 8 };
    const decision = controller.observe(.{
        .now_ns = 3 * std.time.ns_per_s,
        .committed_goodput = 100,
        .dma_starvation_ratio = 0.5,
        .read_saturated = true,
    });
    try std.testing.expectEqual(.read_probe_start, decision.action);
    const rollback = controller.rollbackUnfinishedProbe(300 * std.time.ns_per_ms).?;
    try std.testing.expectEqual(.probe_tail_rollback, rollback.action);
    try std.testing.expectEqual(.finite_tail, rollback.reason);
    try std.testing.expectEqual(@as(usize, 12), rollback.limits.read);
    try std.testing.expectEqual(@as(usize, 8), rollback.limits.dma);
    try std.testing.expect(!controller.readGrowthIsBlocked(
        rollback.limits.request_size,
        300 * std.time.ns_per_ms,
    ));
    try std.testing.expect(!controller.prefer_request_size_probe);
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
