const std = @import("std");
const builtin = @import("builtin");

const pjrt = @import("pjrt");
const stdx = @import("stdx");
pub const VFS = @import("io").VFS;

const Buffer = @import("buffer.zig").Buffer;
const Bufferized = @import("zml.zig").Bufferized;
const platform_mod = @import("platform.zig");
const CreateOptions = platform_mod.CreateOptions;
const mem = @import("mem.zig");
const meta = @import("meta.zig");
const pjrtx = @import("pjrtx.zig");
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
    ) void {
        self.probe_mutex.lockUncancelable(io);
        defer self.probe_mutex.unlock(io);
        if (epoch != self.probe_epoch.load(.acquire) or
            admission_id < self.probe_admission_start) return;
        if (bytes == load_read_request_size)
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

    fn init(read: usize, feasible_width: usize, requested_slack: usize) PinnedGateLimits {
        std.debug.assert(feasible_width > 0);
        const effective_read = @min(read, feasible_width);
        const slack = @min(requested_slack, feasible_width - effective_read);
        return .{
            .feasible_width = feasible_width,
            .read = effective_read,
            .lifecycle = effective_read + slack,
        };
    }
};

fn dmaAdmissionLessLoaded(
    lhs_active: usize,
    lhs_capacity: usize,
    rhs_active: usize,
    rhs_capacity: usize,
) bool {
    std.debug.assert(lhs_capacity > 0 and rhs_capacity > 0);
    return @as(u128, lhs_active) * @as(u128, rhs_capacity) <
        @as(u128, rhs_active) * @as(u128, lhs_capacity);
}

fn selectLoaderDmaDevice(
    active: []const usize,
    per_device_limit: usize,
    ready_mask: u64,
    next_device: usize,
    weighted: bool,
) ?usize {
    std.debug.assert(active.len > 0 and active.len <= 64);
    std.debug.assert(per_device_limit > 0 and next_device < active.len);
    var selected: ?usize = null;
    for (0..active.len) |offset| {
        const device_index = (next_device + offset) % active.len;
        if (ready_mask & (@as(u64, 1) << @intCast(device_index)) == 0 or
            active[device_index] >= per_device_limit)
        {
            continue;
        }
        if (selected == null or
            (weighted and dmaAdmissionLessLoaded(
                active[device_index],
                per_device_limit,
                active[selected.?],
                per_device_limit,
            )))
        {
            selected = device_index;
        }
    }
    return selected;
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
        }
    };

    const BlockContext = struct {
        pipeline: *VectoredLoadPipeline,
        request: *RequestContext,
        lease: mem.DmaBlockPool.Lease,
        ready_at: std.Io.Timestamp,
        pending_submissions: usize,
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
    global_dma_limit: ?usize,
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
        worker_gate: *AdaptiveRequestGate,
        read_gate: *AdaptiveRequestGate,
        request_gate: *AdaptiveRequestGate,
        block_size: usize,
        device_pool_indices: []const usize,
        numa_explicit: bool,
        metrics: *VectoredLoadMetrics,
        dma_limit: usize,
        global_dma_limit: ?usize,
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
            .device_pool_indices = device_pool_indices,
            .numa_explicit = numa_explicit,
            .metrics = metrics,
            .ready_queues = ready_queues,
            .active_by_device = active_by_device,
            .peak_by_device = peak_by_device,
            .dma_limit = dma_limit,
            .global_dma_limit = global_dma_limit,
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
            self.pool.close(self.io);
            self.worker_gate.close(self.io);
            self.read_gate.close(self.io);
            self.request_gate.close(self.io);
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
        };
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        try self.requests.append(self.allocator, request);
        self.metrics.beginRequest(len);
        return request;
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
            const queue = &self.ready_queues[target.device_index];
            queue.appendAssumeCapacity(transfer);
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
            self.metadata_mutex.lockUncancelable(self.io);
            if (!self.failed()) {
                const limit = self.dma_limit;
                const global_available = self.global_dma_limit == null or
                    self.active_events < self.global_dma_limit.?;
                if (global_available) {
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
                        self.global_dma_limit != null,
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
                        if (self.global_dma_limit) |global_limit| {
                            std.debug.assert(self.active_events <= global_limit);
                        }
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
            }
            if (selected == null) {
                self.pumping = false;
                self.maybeDoneLocked();
                self.metadata_mutex.unlock(self.io);
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
            transfer.block.lease.data[0..transfer.len],
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

    fn peakDeviceActive(self: *VectoredLoadPipeline) usize {
        self.metadata_mutex.lockUncancelable(self.io);
        defer self.metadata_mutex.unlock(self.io);
        var peak_device_active: usize = 0;
        for (self.peak_by_device) |peak| {
            peak_device_active = @max(peak_device_active, peak);
        }
        return peak_device_active;
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
        for (plan.blocks, affinities) |block_plan, *affinity| {
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

        const pool_wait_ns = pipeline.pool.acquireMany(pipeline.io, leased, affinities) catch |err| {
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
        // Generation and admission identity belong to the source-call permit,
        // not to earlier job claim or pinned-block waits.
        request.read_epoch = pipeline.metrics.config_epoch.load(.acquire);
        request.admission_id = pipeline.next_read_admission.fetchAdd(1, .monotonic);
        pipeline.metrics.beginRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
        );
        const read_started: std.Io.Timestamp = .now(pipeline.io, .awake);
        const read_result = tensor.reader.readPositionalAllV(iovecs, source_offset);
        const read_elapsed = read_started.untilNow(pipeline.io, .awake);
        read_result catch |err| {
            pipeline.metrics.endRead(
                pipeline.io,
                request.read_epoch,
                request.admission_id,
            );
            pipeline.read_gate.release(pipeline.io);
            pipeline.recordError(err);
            return;
        };
        pipeline.metrics.recordProbeRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
            request_len,
        );
        const read_elapsed_ns: u64 = @intCast(@max(read_elapsed.nanoseconds, 0));
        const read_elapsed_us: u64 = read_elapsed_ns / std.time.ns_per_us;
        _ = pipeline.metrics.read_operations.fetchAdd(1, .monotonic);
        _ = pipeline.metrics.read_bytes.fetchAdd(request_len, .monotonic);
        _ = pipeline.metrics.read_ns.fetchAdd(read_elapsed_ns, .monotonic);
        _ = pipeline.metrics.weighted_read_latency_us.fetchAdd(read_elapsed_us *| @as(u64, @intCast(request_len)), .monotonic);
        tensor.recordReadProgress(request_len);
        request.markReadFinished();
        pipeline.metrics.endRead(
            pipeline.io,
            request.read_epoch,
            request.admission_id,
        );
        pipeline.read_gate.release(pipeline.io);

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

/// Fixed-size source jobs scheduled by destination-device debt. A replicated
/// job is present in every destination queue but is claimed exactly once and
/// credits every device it serves.
const FairVectoredReadScheduler = struct {
    const Job = struct {
        tensor_index: usize,
        source_offset: usize,
        len: usize,
    };

    const StoredJob = struct {
        tensor_index: usize,
        source_offset: usize,
        len: usize,
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

    allocator: std.mem.Allocator,
    device_count: usize,
    jobs: std.ArrayListUnmanaged(StoredJob) = .empty,
    physical_bytes: std.ArrayListUnmanaged(usize) = .empty,
    queues: []std.ArrayListUnmanaged(usize),
    cursors: []usize,
    claimed: []bool,
    scheduled_physical_bytes: []u64,
    remaining_bytes: u64,
    remaining_jobs: usize,
    remaining_full_jobs: usize,
    maximum_blocks_per_job: usize = 0,
    next_device: usize = 0,
    mutex: std.Io.Mutex = .init,

    fn init(
        allocator: std.mem.Allocator,
        platform: *const Platform,
        tensors: []const *const Tensor,
        shardings: []const Sharding,
        block_size: usize,
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
            .queues = queues,
            .cursors = cursors,
            .claimed = &.{},
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
            const sharding = Sharding.pickSharding(shardings, shape, .explicit_axis_binding) orelse
                platform.replicated_sharding;
            plan.* = .{
                .dispatch_spans = try .init(allocator, shape, sharding),
                .device_indices = &.{},
                .total = shape.byteSize(),
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
            const len = @min(load_read_request_size, tensor_size - source_offset);
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
                .source_offset = source_offset,
                .len = len,
            });
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
            if (len == load_read_request_size) self.remaining_full_jobs += 1;
        }
        self.claimed = try allocator.alloc(bool, self.jobs.items.len);
        @memset(self.claimed, false);
        return self;
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
            .queues = queues,
            .cursors = cursors,
            .claimed = &.{},
            .scheduled_physical_bytes = scheduled,
            .remaining_bytes = 0,
            .remaining_jobs = 0,
            .remaining_full_jobs = 0,
        };
        errdefer self.deinit();
        for (test_jobs, 0..) |job, job_index| {
            if (job.physical_bytes.len != device_count or job.block_count == 0)
                return error.InvalidTestJob;
            try self.jobs.append(allocator, .{
                .tensor_index = job.tensor_index,
                .source_offset = 0,
                .len = job.len,
            });
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
            if (job.len == load_read_request_size) self.remaining_full_jobs += 1;
        }
        self.claimed = try allocator.alloc(bool, test_jobs.len);
        @memset(self.claimed, false);
        return self;
    }

    fn deinit(self: *FairVectoredReadScheduler) void {
        for (self.queues) |*queue| queue.deinit(self.allocator);
        self.allocator.free(self.queues);
        self.allocator.free(self.cursors);
        if (self.claimed.len != 0) self.allocator.free(self.claimed);
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
        for (0..self.device_count) |offset| {
            const device_index = (self.next_device + offset) % self.device_count;
            const queue = &self.queues[device_index];
            while (self.cursors[device_index] < queue.items.len and
                self.claimed[queue.items[self.cursors[device_index]]])
            {
                self.cursors[device_index] += 1;
            }
            if (self.cursors[device_index] == queue.items.len) continue;
            if (selected_device == null or
                self.scheduled_physical_bytes[device_index] <
                    self.scheduled_physical_bytes[selected_device.?])
            {
                selected_device = device_index;
            }
        }
        const device_index = selected_device orelse unreachable;
        const job_index = self.queues[device_index].items[self.cursors[device_index]];
        self.cursors[device_index] += 1;
        std.debug.assert(!self.claimed[job_index]);
        self.claimed[job_index] = true;
        self.remaining_jobs -= 1;
        const stored = self.jobs.items[job_index];
        if (stored.len == load_read_request_size) self.remaining_full_jobs -= 1;
        self.remaining_bytes -= stored.len;
        const row = self.physical_bytes.items[job_index * self.device_count ..][0..self.device_count];
        for (row, self.scheduled_physical_bytes) |bytes, *scheduled| {
            scheduled.* +|= @intCast(bytes);
        }
        self.next_device = (device_index + 1) % self.device_count;
        return .{
            .tensor_index = stored.tensor_index,
            .source_offset = stored.source_offset,
            .len = stored.len,
        };
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
        .{ .tensor_index = 0, .len = load_read_request_size, .physical_bytes = &.{load_read_request_size} },
        .{ .tensor_index = 1, .len = load_read_request_size, .physical_bytes = &.{load_read_request_size} },
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

fn readTimingBucketIndex(request_size: usize) ?usize {
    for (VFS.read_timing_bucket_sizes, 0..) |size, index| {
        if (request_size == size) return index;
    }
    return null;
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
        const initial_index = @min(widthIndexAtMost(12), maximum_index);
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

const VectoredReadStatsSource = struct {
    profile_id: usize,
    name: []const u8,
    provider: VFS.ReadStatsProvider,
    initial: VFS.ReadStats,
    previous: VFS.ReadStats,
};

const SourceTelemetry = struct {
    retries: u64 = 0,
    timing_successes: u64 = 0,
    transient_retries: u64 = 0,
    timeouts: u64 = 0,
    server_failures: u64 = 0,
    timing_transient_retries: u64 = 0,
    timing_timeouts: u64 = 0,
    timing_server_failures: u64 = 0,
    throttles: u64 = 0,

    fn failures(self: SourceTelemetry) u64 {
        return self.timing_transient_retries +| self.timing_timeouts +| self.timing_server_failures;
    }

    fn responseObserved(self: SourceTelemetry) bool {
        return self.timing_successes > 0 or self.transient_retries > 0 or self.timeouts > 0 or
            self.server_failures > 0 or self.throttles > 0;
    }
};

const SourceReadRuntime = struct {
    controller: SourceReadWidthController,
    worker_gate: *AdaptiveRequestGate,
    read_gate: *AdaptiveRequestGate,
    request_gate: *AdaptiveRequestGate,
    post_read_slack_requests: usize,
    metrics: *VectoredLoadMetrics,
    next_read_admission: *std.atomic.Value(u64),
    scheduler: *FairVectoredReadScheduler,
    pinned_feasible_width: usize,
    read_stats_sources: []VectoredReadStatsSource,
    source_bootstrap_enabled: bool,
    source_response_observed: bool = false,
    probe_dirty: bool = false,
    probe_transition_pending: bool = false,
    probe_measuring: bool = false,
    scoring_pending: bool = false,
    blind_admissions: bool = false,
    pending_read_limit: usize = 1,
    pending_evidence: SourceReadWidthController.Evidence = undefined,
    last_blind_growth_ns: u64 = 0,
    done: std.Io.Event = .unset,

    fn takeRemoteTelemetry(self: *SourceReadRuntime) SourceTelemetry {
        var result: SourceTelemetry = .{};
        const timing_index = readTimingBucketIndex(load_read_request_size).?;
        for (self.read_stats_sources) |*source| {
            const current = source.provider.snapshot();
            const delta = current.sub(source.previous);
            source.previous = current;
            result.retries +|= delta.retries;
            result.transient_retries +|= delta.transient_retries;
            result.timeouts +|= delta.timeouts;
            result.server_failures +|= delta.server_failures;
            result.throttles +|= delta.throttles;
            const timing = delta.timing[timing_index];
            result.timing_successes +|= timing.successes;
            result.timing_transient_retries +|= timing.transient_retries;
            result.timing_timeouts +|= timing.timeouts;
            result.timing_server_failures +|= timing.server_failures;
        }
        return result;
    }

    fn telemetryClean(telemetry: SourceTelemetry) bool {
        return telemetry.retries == 0 and telemetry.transient_retries == 0 and
            telemetry.timeouts == 0 and telemetry.server_failures == 0 and
            telemetry.throttles == 0 and telemetry.failures() == 0;
    }

    fn applyDecision(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
        force_probe: bool,
    ) void {
        const gate_limits: PinnedGateLimits = .init(
            decision.width,
            self.pinned_feasible_width,
            self.post_read_slack_requests,
        );
        self.worker_gate.setLimit(io, gate_limits.read);
        self.request_gate.setLimit(io, gate_limits.lifecycle);
        if (!decision.settled and (decision.changed or force_probe)) {
            self.read_gate.setLimit(io, 0);
            self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
            self.pending_read_limit = gate_limits.read;
            self.probe_transition_pending = true;
            self.probe_measuring = false;
            self.scoring_pending = false;
            self.blind_admissions = false;
            self.probe_dirty = false;
            _ = self.activatePendingProbe(io);
        } else if (decision.settled) {
            self.read_gate.setLimit(io, gate_limits.read);
            self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
            self.metrics.config_epoch.store(decision.generation, .release);
            self.probe_transition_pending = false;
            self.probe_measuring = false;
            self.scoring_pending = false;
            self.blind_admissions = false;
            self.probe_dirty = false;
        }
    }

    fn activatePendingProbe(self: *SourceReadRuntime, io: std.Io) bool {
        if (!self.probe_transition_pending or
            self.read_gate.inUse(io) != 0) return false;
        // VFS counters are several atomics updated by the source call before
        // it releases the gate. Snapshot once more after the drain so one
        // logical retry cannot be split across two probe generations.
        _ = self.takeRemoteTelemetry();
        const admission_start = self.next_read_admission.load(.acquire);
        self.metrics.prepareProbe(io, self.controller.generation, admission_start);
        self.probe_transition_pending = false;
        self.probe_measuring = true;
        self.probe_dirty = false;
        self.read_gate.setLimit(io, self.pending_read_limit);
        return true;
    }

    fn applyBlindGrowth(
        self: *SourceReadRuntime,
        io: std.Io,
        decision: SourceReadWidthController.Decision,
    ) void {
        const gate_limits: PinnedGateLimits = .init(
            decision.width,
            self.pinned_feasible_width,
            self.post_read_slack_requests,
        );
        self.worker_gate.setLimit(io, gate_limits.read);
        self.read_gate.setLimit(io, gate_limits.read);
        self.request_gate.setLimit(io, gate_limits.lifecycle);
        self.metrics.clearProbe(io, self.metrics.probe_epoch.load(.acquire));
        self.metrics.config_epoch.store(decision.generation, .release);
        self.probe_transition_pending = false;
        self.probe_measuring = false;
        self.scoring_pending = false;
        self.blind_admissions = true;
    }

    fn restartDirtyProbe(
        self: *SourceReadRuntime,
        io: std.Io,
        remaining_full_jobs: usize,
    ) void {
        if (self.controller.phase == .settled) return;
        if (!self.controller.restartFitsTail(remaining_full_jobs)) {
            self.applyDecision(io, self.controller.rollbackTail(), false);
            return;
        }
        self.controller.generation +|= 1;
        var decision = self.controller.currentDecision();
        decision.changed = true;
        self.applyDecision(io, decision, true);
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
            .clean = !self.probe_dirty,
            .remaining_full_jobs = remaining_full_jobs,
        };
    }

    fn finalize(self: *SourceReadRuntime, io: std.Io) void {
        std.debug.assert(self.read_gate.inUse(io) == 0);
        const telemetry = self.takeRemoteTelemetry();
        if (!telemetryClean(telemetry)) self.probe_dirty = true;
        if (self.controller.phase != .settled and !self.probe_dirty) {
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

    fn run(self: *SourceReadRuntime, io: std.Io) std.Io.Cancelable!void {
        const started: std.Io.Timestamp = .now(io, .awake);
        self.applyDecision(io, self.controller.currentDecision(), self.controller.adaptive);
        while (true) {
            self.done.waitTimeout(io, .{ .duration = .{
                .raw = .fromMilliseconds(if (self.source_response_observed) 25 else 10),
                .clock = .awake,
            } }) catch |err| switch (err) {
                error.Timeout => {},
                error.Canceled => return error.Canceled,
            };
            if (self.done.isSet()) {
                self.finalize(io);
                break;
            }

            const telemetry = self.takeRemoteTelemetry();
            if (telemetry.responseObserved()) self.source_response_observed = true;
            if (self.metrics.read_bytes.load(.acquire) != 0) self.source_response_observed = true;
            const scheduler_snapshot = self.scheduler.snapshot(io);
            const now_ns: u64 = @intCast(@max(started.untilNow(io, .awake).nanoseconds, 0));
            if (!telemetryClean(telemetry)) self.probe_dirty = true;

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

            // A completed score is held until all calls admitted at that
            // width have drained. The final telemetry snapshot above can then
            // invalidate the frozen evidence before it changes the width.
            if (self.scoring_pending) {
                if (self.read_gate.inUse(io) != 0) continue;
                const drained_telemetry = self.takeRemoteTelemetry();
                if (!telemetryClean(drained_telemetry)) self.probe_dirty = true;
                if (self.probe_dirty) {
                    self.scoring_pending = false;
                    self.restartDirtyProbe(io, scheduler_snapshot.remaining_full_jobs);
                    continue;
                }
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
                    self.applyDecision(io, self.controller.rollbackTail(), false);
                    continue;
                }
                _ = self.activatePendingProbe(io);
                continue;
            }

            if (self.probe_measuring) {
                if (self.probe_dirty) {
                    self.restartDirtyProbe(io, scheduler_snapshot.remaining_full_jobs);
                    continue;
                }
                const evidence = self.currentEvidence(
                    io,
                    scheduler_snapshot.remaining_full_jobs,
                );
                if (evidence.scoreable()) {
                    // Freeze a complete clean interval, then drain admissions
                    // that raced with the snapshot. Their bytes are excluded,
                    // but their final VFS telemetry must still be clean.
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
                self.controller.phase != .settled)
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

fn effectiveDmaGlobalCap(
    calibrated: ?usize,
    used_device_count: usize,
    per_device: usize,
) !?usize {
    const uncapped = try std.math.mul(usize, used_device_count, per_device);
    if (calibrated) |cap| {
        if (cap < uncapped) return cap;
    }
    return null;
}

fn loadVectored(
    comptime ModelType: type,
    model: *const ModelType,
    bufferized: *Bufferized(ModelType),
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const Platform,
    store: *const TensorStore,
    opts: LoadOpts,
    dma_resources: *DmaPlatformSettings,
    used_device_ids: []const u32,
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

    var dma_config = dma_resources.config;
    dma_config.global_max_in_flight = try effectiveDmaGlobalCap(
        dma_config.global_max_in_flight,
        used_device_ids.len,
        dma_config.max_in_flight_per_device,
    );
    const node_reserves = try allocator.alloc(usize, dma_resources.workspace.pools.len);
    defer allocator.free(node_reserves);
    @memset(node_reserves, 0);
    for (used_device_ids) |device_id| {
        const device_index = for (platform.devices, 0..) |device, index| {
            if (device.id() == device_id) break index;
        } else return error.DmaDeviceMismatch;
        const node_index = dma_resources.workspace.device_pool_indices[device_index];
        node_reserves[node_index] = try std.math.add(
            usize,
            node_reserves[node_index],
            dma_config.max_in_flight_per_device,
        );
    }
    var pool = try mem.DmaBlockPool.initFromProvider(
        allocator,
        dma_resources.workspace.blockPoolArenaProvider(),
        dma_config.block_size,
        dma_config.max_mapped_bytes,
        node_reserves,
    );
    defer pool.deinit();
    try dma_resources.workspace.ensureLoadBlockReserves(
        dma_config.block_size,
        used_device_ids.len,
        node_reserves,
    );
    try pool.refreshProviderArenas(io);

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
            if (load_read_request_size < minimum) {
                load_log.warn("fixed source request size {Bi:.2} is below the {Bi:.2} minimum advertised by {s}", .{
                    load_read_request_size,
                    minimum,
                    if (profile) |p| p.scheme else "local/default",
                });
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
                "fixed",
                descriptor.file_uri,
            });
            break :blk index;
        };
    }

    var source_minimum: usize = dma_config.block_size;
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

    var scheduler = try FairVectoredReadScheduler.init(
        allocator,
        platform,
        tensors,
        opts.shardings,
        dma_config.block_size,
    );
    defer scheduler.deinit();
    const maximum_blocks_per_job = scheduler.maximumBlocksPerJob();
    const pinned_feasible_width = if (maximum_blocks_per_job == 0)
        @as(usize, 1)
    else
        try pool.aggregatePotentialRequestWidth(maximum_blocks_per_job);
    if (pinned_feasible_width == 0) return error.DmaMappedBudgetExceeded;
    const strict_pinned_feasible_width = if (maximum_blocks_per_job == 0)
        @as(usize, 1)
    else
        try pool.minimumStrictAffinityRequestWidth(maximum_blocks_per_job);
    load_log.debug("DMA workspace feasibility: maximum_blocks_per_job={d}, aggregate_read_width={d}, minimum_strict_read_width={d}", .{
        maximum_blocks_per_job,
        pinned_feasible_width,
        strict_pinned_feasible_width,
    });

    var metrics: VectoredLoadMetrics = .{};
    const controller = SourceReadWidthController.init(
        opts.read_parallelism,
        pinned_feasible_width,
    );
    const initial_gate_limits: PinnedGateLimits = .init(
        controller.width(),
        pinned_feasible_width,
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
        dma_config.block_size,
        dma_resources.workspace.device_pool_indices,
        for (dma_config.device_numa_nodes) |node| {
            if (node != null) break true;
        } else false,
        &metrics,
        dma_config.max_in_flight_per_device,
        dma_config.global_max_in_flight,
    );
    defer pipeline.deinit();

    var worker_group: std.Io.Group = .init;
    var controller_runtime: SourceReadRuntime = .{
        .controller = controller,
        .worker_gate = &worker_gate,
        .read_gate = &read_gate,
        .request_gate = &request_gate,
        .post_read_slack_requests = post_read_slack_requests,
        .metrics = &metrics,
        .next_read_admission = &pipeline.next_read_admission,
        .scheduler = &scheduler,
        .pinned_feasible_width = pinned_feasible_width,
        .read_stats_sources = read_stats_sources.items,
        .source_bootstrap_enabled = high_latency_source,
    };
    var controller_group: std.Io.Group = .init;
    try controller_group.concurrent(io, SourceReadRuntime.run, .{ &controller_runtime, io });

    const worker_count = if (scheduler.snapshot(io).remaining_jobs == 0)
        0
    else
        opts.read_parallelism.maximum();
    for (0..worker_count) |worker_index| {
        worker_group.concurrent(io, struct {
            fn run(
                worker_index_: usize,
                worker_gate_: *AdaptiveRequestGate,
                scheduler_: *FairVectoredReadScheduler,
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
                    pipeline_.reserveSourceJob();
                    const job = scheduler_.claim(io_) orelse {
                        pipeline_.abandonSourceJob();
                        request_gate_.release(io_);
                        worker_gate_.close(io_);
                        request_gate_.close(io_);
                        return;
                    };
                    const request = pipeline_.registerRequest(job.len) catch |err| {
                        pipeline_.abandonSourceJob();
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
                    VectoredReadRequest.run(
                        request,
                        tensor,
                        pipeline_,
                        job.source_offset,
                        job.len,
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
    const peak_device_active = pipeline.peakDeviceActive();
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
    for (0..pool.nodeCount()) |node_index| {
        const stats = pool.nodeStats(node_index);
        if (dma_resources.workspace.pools[node_index].numa_allocator.node) |node| {
            load_log.debug("DMA workspace node: numa_node={d}, retained={Bi:.2}, newly_mapped={Bi:.2}, leased_high_water={Bi:.2}, unused_tail={Bi:.2}", .{
                node,
                stats.retained_mapped_bytes,
                stats.newly_mapped_bytes,
                stats.leased_high_water_bytes,
                stats.unused_tail_bytes,
            });
        } else {
            load_log.debug("DMA workspace node: numa_node=single, retained={Bi:.2}, newly_mapped={Bi:.2}, leased_high_water={Bi:.2}, unused_tail={Bi:.2}", .{
                stats.retained_mapped_bytes,
                stats.newly_mapped_bytes,
                stats.leased_high_water_bytes,
                stats.unused_tail_bytes,
            });
        }
    }
    load_log.debug("completed: vectored=true, tensors={d}, logical_bytes={Bi:.2}, elapsed={d:.3}s, logical_goodput={d:.2}MiB/s, reads={d}, peak_requests={d}, final_requests={d}, final_request_size={Bi:.2}, feasible_width={d}, average_read={Bi:.2}, average_read_latency={d:.3}ms, average_request_lifetime={d:.3}ms, source_requests={d}, source_bytes={Bi:.2}, source_retries={d}, source_throttles={d}, source_retry_delay={d:.3}s, dma_submissions={d}, peak_dma_per_device={d}, final_dma_per_device={d}, average_dma={Bi:.2}, average_dma_latency={d:.3}ms, average_ready_age={d:.3}ms, submitted={Bi:.2}, committed={Bi:.2}, leased_high_water={Bi:.2}, mapped={Bi:.2}, newly_mapped={Bi:.2}, unused_tail={Bi:.2}, pool_waits={d}, pool_wait={d:.3}s", .{
        tensor_count,
        loaded_bytes,
        elapsed_seconds,
        goodput / (1024 * 1024),
        read_operations,
        metrics.request_high_water.load(.acquire),
        controller_runtime.controller.width(),
        load_read_request_size,
        pinned_feasible_width,
        average_read,
        average_read_ms,
        average_request_ms,
        physical_source_requests,
        physical_source_bytes,
        source_retries,
        source_throttles,
        @as(f64, @floatFromInt(source_retry_delay_ns)) / std.time.ns_per_s,
        dma_submissions,
        peak_device_active,
        dma_config.max_in_flight_per_device,
        average_dma,
        average_dma_ms,
        average_ready_ms,
        metrics.submitted_bytes.load(.acquire),
        metrics.committed_bytes.load(.acquire),
        pool.highWaterBytes(),
        pool.mappedBytes(),
        pool.newlyMappedBytes(),
        pool.unusedTailBytes(),
        metrics.pool_waits.load(.acquire),
        @as(f64, @floatFromInt(metrics.pool_wait_ns.load(.acquire))) / std.time.ns_per_s,
    });
    return loaded_bytes;
}

pub const default_dma_benchmark_block_sizes = [_]usize{
    2 * 1024 * 1024,
    4 * 1024 * 1024,
    8 * 1024 * 1024,
    16 * 1024 * 1024,
    32 * 1024 * 1024,
};

pub const default_dma_benchmark_parallelism = [_]usize{ 1, 2, 4, 6, 8, 12, 16, 24, 32 };

const dma_benchmark_repeats = 3;
const dma_benchmark_global_repeats = 3;

const DmaBenchmarkPhase = enum {
    block,
    block_confirmation,
    parallelism,
    parallelism_confirmation,
    aggregate,
    global_limit,
};

const DmaBenchmarkSample = struct {
    phase: DmaBenchmarkPhase,
    device_index: usize,
    block_size: usize,
    parallelism: usize,
    global_parallelism: ?usize = null,
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

const GlobalDmaRecommendation = struct {
    searched: bool = false,
    parallelism: ?usize = null,
    uncapped_bytes_per_second: f64 = 0,
    uncapped_average_latency_ns: f64 = 0,
    recommended_bytes_per_second: ?f64 = null,
    recommended_average_latency_ns: ?f64 = null,
    recommended_min_device_retention: ?f64 = null,
    recommended_normalized_fairness: ?f64 = null,
    windows: usize = 0,
};

const GlobalDmaCandidate = struct {
    parallelism: usize,
    bytes_per_second: f64,
    average_latency_ns: f64,
    min_device_retention: f64,
    normalized_fairness: f64,
};

/// Immutable DMA settings shared by every device participating in one load.
/// The slices are owned by the enclosing platform settings.
const DmaLoadConfig = struct {
    device_kind: []const u8,
    device_ids: []const u32,
    device_numa_nodes: []const ?usize,
    block_size: usize,
    max_in_flight_per_device: usize,
    global_max_in_flight: ?usize,
    max_mapped_bytes: usize,
};

fn requiredDmaWorkspaceBytes(config: DmaLoadConfig) !usize {
    const request_blocks = std.math.divCeil(
        usize,
        load_read_request_size,
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
        config.block_size > load_read_request_size or
        config.max_mapped_bytes < config.block_size)
        return error.InvalidDmaLoadConfig;
    const uncapped = std.math.mul(
        usize,
        config.device_ids.len,
        config.max_in_flight_per_device,
    ) catch return error.InvalidDmaLoadConfig;
    if (config.global_max_in_flight) |limit| {
        if (limit == 0 or limit > uncapped) return error.InvalidDmaLoadConfig;
    }
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
        .global_max_in_flight = config.global_max_in_flight,
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
        .global_max_in_flight = 4,
        .max_mapped_bytes = 64 * 1024 * 1024,
    };
    try validateDmaLoadConfig(valid);

    var invalid = valid;
    invalid.global_max_in_flight = 17;
    try std.testing.expectError(error.InvalidDmaLoadConfig, validateDmaLoadConfig(invalid));
    invalid = valid;
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
            .global_max_in_flight = config.global_max_in_flight,
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
            .global_max_in_flight = defaults.global_max_in_flight,
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
        .global_max_in_flight = settings.config.global_max_in_flight,
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
    global_candidates: []GlobalDmaCandidate,
    global: GlobalDmaRecommendation,
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
        self.allocator.free(self.global_candidates);
        self.* = undefined;
    }
};

pub const BenchTransferOptions = struct {
    block_sizes: []const usize = &default_dma_benchmark_block_sizes,
    parallelism: []const usize = &default_dma_benchmark_parallelism,
    /// Width used while comparing block sizes and the minimum device width
    /// recommended to the loader. Smaller values remain global-cap candidates.
    block_parallelism: usize = 8,
    /// A screen window runs for at least this long and, unless the target is
    /// zero, until every participating device completes the transfer target.
    duration_ns: u64 = 10 * std.time.ns_per_ms,
    minimum_transfers_per_device: u64 = 128,
    global_duration_ns: u64 = 10 * std.time.ns_per_ms,
    global_minimum_transfers_per_device: u64 = 128,
    /// Borderline local decisions receive longer alternating paired windows.
    confirmation_duration_ns: u64 = 25 * std.time.ns_per_ms,
    confirmation_minimum_transfers_per_device: u64 = 256,
    confirmation_margin: f64 = 0.02,
    /// Prefer a smaller transaction once it supplies enough headroom over the
    /// source pipeline instead of maximizing isolated copy-engine throughput.
    block_selection_tolerance: f64 = 0.08,
    parallelism_selection_tolerance: f64 = 0.05,
    /// A global cap must stay much closer to peak throughput because it is a
    /// shared runtime constraint rather than a per-device setting.
    global_parallelism_selection_tolerance: f64 = 0.02,
    /// Prevent aggregate throughput from hiding one under-served shard.
    global_min_device_retention: f64 = 0.95,
    /// Jain fairness over each device's retention relative to the uncapped run.
    global_fairness_floor: f64 = 0.98,
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

const DmaBenchmarkSourcePool = struct {
    numa_allocator: DmaBenchmarkNumaAllocator,
    dma_map_allocator: mem.DmaMapAllocator,
    allocations: std.ArrayListUnmanaged([]u8) = .empty,
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
                pool.dma_map_allocator.allocator().free(allocation);
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
            const arena = pool.allocations.items[index];
            if (arena.len >= minimum_len) return arena;
        }
        unreachable;
    }

    fn verifyNumaPlacement(
        self: *DmaBenchmarkSourcePools,
        pool: *const DmaBenchmarkSourcePool,
        source: []const u8,
    ) !usize {
        const node = pool.numa_allocator.node orelse return 0;
        if (comptime builtin.os.tag != .linux) return error.DmaBenchmarkNumaUnsupported;
        const page_count = std.math.divCeil(usize, source.len, std.heap.page_size_min) catch unreachable;
        const sample_count = @min(page_count, 256);
        const pages = try self.allocator.alloc(*const anyopaque, sample_count);
        defer self.allocator.free(pages);
        const statuses = try self.allocator.alloc(i32, sample_count);
        defer self.allocator.free(statuses);
        for (pages, 0..) |*page, sample_index| {
            const page_index = sample_index * page_count / sample_count;
            page.* = @ptrFromInt(@intFromPtr(source.ptr) + page_index * std.heap.page_size_min);
        }
        const rc = std.os.linux.syscall6(
            .move_pages,
            0,
            sample_count,
            @intFromPtr(pages.ptr),
            0,
            @intFromPtr(statuses.ptr),
            0,
        );
        if (std.os.linux.errno(rc) != .SUCCESS) {
            log.err("unable to query DMA benchmark NUMA placement: {s}", .{
                @tagName(std.os.linux.errno(rc)),
            });
            return error.DmaBenchmarkNumaQueryFailed;
        }
        for (statuses) |status| {
            if (status < 0 or status != node) {
                log.err("DMA benchmark source requested NUMA node {d}, observed status {d}", .{
                    node,
                    status,
                });
                return error.DmaBenchmarkNumaPlacementMismatch;
            }
        }
        return sample_count;
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
        const dma_allocator = pool.dma_map_allocator.allocator();
        const replacement = try dma_allocator.alignedAlloc(
            u8,
            .fromByteUnits(std.heap.page_size_min),
            required_bytes,
        );
        errdefer dma_allocator.free(replacement);
        @memset(replacement, 0xa5);
        const verified_pages = try self.verifyNumaPlacement(pool, replacement);
        try pool.allocations.append(self.allocator, replacement);
        _ = self.allocated_bytes.fetchAdd(replacement.len, .release);
        pool.source = replacement;
        if (pool.numa_allocator.node) |node| {
            log.info("DMA benchmark source pool numa_node={d} address=0x{x} size={Bi:.2} verified_pages={d}", .{
                node,
                @intFromPtr(pool.source.ptr),
                pool.source.len,
                verified_pages,
            });
        } else {
            log.info("DMA benchmark source pool numa_node=single address=0x{x} size={Bi:.2}", .{
                @intFromPtr(pool.source.ptr),
                pool.source.len,
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
            load_read_request_size,
            block_size,
        ) catch return error.InvalidDmaLoadConfig;
        const request_blocks = std.math.add(
            usize,
            base_request_blocks,
            maximum_writer_groups - 1,
        ) catch return error.InvalidDmaLoadConfig;
        const missing_bytes = try self.allocator.alloc(usize, self.pools.len);
        defer self.allocator.free(missing_bytes);
        var missing_total: usize = 0;
        for (self.pools, calibrated_reserves, missing_bytes) |pool, reserve, *missing| {
            var usable_blocks: usize = 0;
            for (pool.allocations.items) |arena| {
                usable_blocks = std.math.add(
                    usize,
                    usable_blocks,
                    arena.len / block_size,
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
                    return pools.pools[node_index].allocations.items[arena_index];
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
                    ];
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

const DmaBenchmarkRunSpec = struct {
    device_index: usize,
    block_size: usize,
    parallelism: usize,
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

const DmaBenchmarkFairGate = struct {
    allocator: std.mem.Allocator,
    limit: usize,
    capacities: []usize,
    active: []usize,
    waiting: []usize,
    grants: []usize,
    conditions: []std.Io.Condition,
    active_total: usize = 0,
    next_device: usize = 0,
    mutex: std.Io.Mutex = .init,

    fn init(
        allocator: std.mem.Allocator,
        specs: []const DmaBenchmarkRunSpec,
        limit: usize,
    ) !DmaBenchmarkFairGate {
        std.debug.assert(specs.len > 0 and limit > 0);
        const capacities = try allocator.alloc(usize, specs.len);
        errdefer allocator.free(capacities);
        const active = try allocator.alloc(usize, specs.len);
        errdefer allocator.free(active);
        const waiting = try allocator.alloc(usize, specs.len);
        errdefer allocator.free(waiting);
        const grants = try allocator.alloc(usize, specs.len);
        errdefer allocator.free(grants);
        const conditions = try allocator.alloc(std.Io.Condition, specs.len);
        errdefer allocator.free(conditions);
        for (specs, capacities) |spec, *capacity| capacity.* = spec.parallelism;
        @memset(active, 0);
        @memset(waiting, 0);
        @memset(grants, 0);
        for (conditions) |*condition| condition.* = .init;
        return .{
            .allocator = allocator,
            .limit = limit,
            .capacities = capacities,
            .active = active,
            .waiting = waiting,
            .grants = grants,
            .conditions = conditions,
        };
    }

    fn deinit(self: *DmaBenchmarkFairGate) void {
        std.debug.assert(self.active_total == 0);
        for (self.active, self.waiting, self.grants) |active, waiting, grants| {
            std.debug.assert(active == 0 and waiting == 0 and grants == 0);
        }
        self.allocator.free(self.capacities);
        self.allocator.free(self.active);
        self.allocator.free(self.waiting);
        self.allocator.free(self.grants);
        self.allocator.free(self.conditions);
        self.* = undefined;
    }

    fn lessLoaded(self: *const DmaBenchmarkFairGate, lhs: usize, rhs: usize) bool {
        return dmaAdmissionLessLoaded(
            self.active[lhs],
            self.capacities[lhs],
            self.active[rhs],
            self.capacities[rhs],
        );
    }

    fn grantNextLocked(self: *DmaBenchmarkFairGate) ?usize {
        if (self.active_total >= self.limit) return null;
        var selected: ?usize = null;
        for (0..self.capacities.len) |offset| {
            const device_index = (self.next_device + offset) % self.capacities.len;
            if (self.waiting[device_index] <= self.grants[device_index] or
                self.active[device_index] >= self.capacities[device_index]) continue;
            if (selected == null or self.lessLoaded(device_index, selected.?)) {
                selected = device_index;
            }
        }
        const device_index = selected orelse return null;
        self.grants[device_index] += 1;
        self.active[device_index] += 1;
        self.active_total += 1;
        self.next_device = (device_index + 1) % self.capacities.len;
        return device_index;
    }

    fn dispatchLocked(self: *DmaBenchmarkFairGate, io: std.Io) void {
        while (self.grantNextLocked()) |device_index| {
            self.conditions[device_index].signal(io);
        }
    }

    fn acquire(self: *DmaBenchmarkFairGate, io: std.Io, device_index: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        self.waiting[device_index] += 1;
        self.dispatchLocked(io);
        while (self.grants[device_index] == 0) {
            self.conditions[device_index].waitUncancelable(io, &self.mutex);
        }
        self.grants[device_index] -= 1;
        self.waiting[device_index] -= 1;
    }

    fn release(self: *DmaBenchmarkFairGate, io: std.Io, device_index: usize) void {
        self.mutex.lockUncancelable(io);
        defer self.mutex.unlock(io);
        std.debug.assert(self.active_total > 0 and self.active[device_index] > 0);
        self.active_total -= 1;
        self.active[device_index] -= 1;
        self.dispatchLocked(io);
    }
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
    global_parallelism: ?usize,
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
        lane_index: usize,
        slot: usize,
        metrics: *DmaBenchmarkAtomicMetrics,
        ready: *std.atomic.Value(usize),
        start: *std.Io.Event,
        stop: *std.atomic.Value(bool),
        gate: ?*DmaBenchmarkFairGate,

        fn transfer(self: @This()) void {
            if (self.gate) |gate| gate.acquire(self.lane.cohort.io, self.lane_index);
            self.lane.cohort.transfer(self.lane.source, self.slot, self.metrics);
            if (self.gate) |gate| gate.release(self.lane.cohort.io, self.lane_index);
        }

        fn run(self: @This()) void {
            _ = self.ready.fetchAdd(1, .release);
            self.start.waitUncancelable(self.lane.cohort.io);
            while (!self.stop.load(.acquire)) {
                self.transfer();
                if (self.lane.cohort.first_error.load(.acquire) != 0) return;
            }
        }
    };

    var ready: std.atomic.Value(usize) = .init(0);
    var start: std.Io.Event = .unset;
    var stop: std.atomic.Value(bool) = .init(false);
    var gate_storage: DmaBenchmarkFairGate = undefined;
    const gate: ?*DmaBenchmarkFairGate = if (global_parallelism) |limit| gate: {
        const specs = try allocator.alloc(DmaBenchmarkRunSpec, lanes.len);
        defer allocator.free(specs);
        for (lanes, specs) |lane, *spec| spec.* = .{
            .device_index = lane.cohort.device_index,
            .block_size = lane.cohort.block_size,
            .parallelism = lane.parallelism,
        };
        gate_storage = try .init(allocator, specs, limit);
        break :gate &gate_storage;
    } else null;
    defer if (gate != null) gate_storage.deinit();
    var group: std.Io.Group = .init;
    var worker_count: usize = 0;
    for (lanes, atomic_metrics, 0..) |lane, *metrics, lane_index| {
        for (0..lane.parallelism) |slot| {
            try group.concurrent(io, Worker.run, .{Worker{
                .lane = lane,
                .lane_index = lane_index,
                .slot = slot,
                .metrics = metrics,
                .ready = &ready,
                .start = &start,
                .stop = &stop,
                .gate = gate,
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
        global_parallelism: ?usize,
        repeat: usize,
    ) ![]DmaBenchmarkRunMetrics {
        const metrics = try runReusableDmaBenchmarkWindow(
            self.allocator,
            self.io,
            lanes,
            duration_ns,
            minimum_transfers_per_device,
            global_parallelism,
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
                .global_parallelism = global_parallelism,
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
pub const load_read_request_size: usize = 32 * 1024 * 1024;

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

pub const LoadOpts = struct {
    pub const auto: LoadOpts = .{};

    /// Concurrent positional source requests.
    read_parallelism: Parallelism = .{ .adaptive = .{ .initial = 12, .maximum = max_load_read_parallelism } },
    shardings: []const Sharding = &.{},
    progress: ?*std.Progress.Node = null,
    total_bytes: ?*usize = null,
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
                    null,
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

fn appendUniqueUsize(
    values: *std.ArrayListUnmanaged(usize),
    allocator: std.mem.Allocator,
    value: usize,
) !void {
    for (values.items) |existing| if (existing == value) return;
    try values.append(allocator, value);
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
                null,
                repeat,
            );
            defer session.allocator.free(metrics);
            try candidate.metrics.append(session.allocator, metrics[0]);
        }
    }
}

fn updateDmaBenchmarkPlateau(
    selected: *usize,
    unchanged_candidates: *usize,
    next_selected: usize,
) bool {
    if (next_selected == selected.*) {
        unchanged_candidates.* += 1;
    } else {
        selected.* = next_selected;
        unchanged_candidates.* = 0;
    }
    return unchanged_candidates.* == 2;
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

    var widths: std.ArrayListUnmanaged(usize) = .empty;
    defer widths.deinit(session.allocator);
    try appendUniqueUsize(&widths, session.allocator, opts.block_parallelism);
    for (opts.parallelism) |parallelism| {
        if (parallelism <= opts.block_parallelism or
            !dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, selected_cohort.block_size, parallelism))
            continue;
        try appendUniqueUsize(&widths, session.allocator, parallelism);
    }
    std.mem.sort(usize, widths.items, {}, std.sort.asc(usize));

    var width_candidates: std.ArrayListUnmanaged(DmaBenchmarkCandidate) = .empty;
    defer {
        for (width_candidates.items) |*candidate| candidate.deinit(session.allocator);
        width_candidates.deinit(session.allocator);
    }
    // The block screen already supplied three measurements at the minimum
    // loader width, so do not measure that exact tuple again.
    try width_candidates.append(session.allocator, .{
        .value = opts.block_parallelism,
        .cohort = selected_cohort,
    });
    try width_candidates.items[0].metrics.appendSlice(
        session.allocator,
        block_candidates[block_decision.index].metrics.items,
    );

    var selected_width = opts.block_parallelism;
    var unchanged_candidates: usize = 0;
    for (widths.items[1..]) |parallelism| {
        try source_pools.growPool(
            pool_index,
            try std.math.mul(usize, selected_cohort.block_size, parallelism),
        );
        try width_candidates.append(session.allocator, .{
            .value = parallelism,
            .cohort = selected_cohort,
        });
        try measureDmaBenchmarkCandidates(
            session,
            .parallelism,
            width_candidates.items[width_candidates.items.len - 1 ..],
            source_pools.pools[pool_index].source,
            null,
            opts.duration_ns,
            opts.minimum_transfers_per_device,
            dma_benchmark_repeats,
        );
        const decision = try selectDmaBenchmarkCandidate(
            session.allocator,
            width_candidates.items,
            opts.parallelism_selection_tolerance,
        );
        const next_selected_width = width_candidates.items[decision.index].value;
        if (updateDmaBenchmarkPlateau(
            &selected_width,
            &unchanged_candidates,
            next_selected_width,
        )) break;
    }
    const width_decision = try confirmAndSelectDmaBenchmarkCandidate(
        session,
        opts,
        .parallelism_confirmation,
        width_candidates.items,
        source_pools.pools[pool_index].source,
        null,
        opts.parallelism_selection_tolerance,
    );

    return .{
        .recommendation = .{
            .device_index = device_index,
            .device_id = session.platform.devices[device_index].id(),
            .dma_block_size = selected_cohort.block_size,
            .dma_parallelism = width_candidates.items[width_decision.index].value,
            .measured_bytes_per_second = width_decision.metrics.bytesPerSecond(),
            .average_latency_ns = width_decision.metrics.averageLatencyNs(),
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

fn combinedDmaMetrics(metrics: []const DmaBenchmarkRunMetrics) DmaBenchmarkRunMetrics {
    var combined: DmaBenchmarkRunMetrics = .{ .bytes = 0, .transfers = 0, .total_latency_ns = 0, .elapsed_ns = 0 };
    for (metrics) |metric| {
        combined.bytes +|= metric.bytes;
        combined.transfers +|= metric.transfers;
        combined.total_latency_ns +|= metric.total_latency_ns;
        combined.elapsed_ns = @max(combined.elapsed_ns, metric.elapsed_ns);
    }
    return combined;
}

fn medianCombinedDmaRunIndex(
    allocator: std.mem.Allocator,
    runs: []const []DmaBenchmarkRunMetrics,
) !usize {
    std.debug.assert(runs.len > 0);
    const order = try allocator.alloc(usize, runs.len);
    defer allocator.free(order);
    for (order, 0..) |*index, i| index.* = i;
    std.mem.sort(usize, order, runs, struct {
        fn lessThan(all_runs: []const []DmaBenchmarkRunMetrics, lhs: usize, rhs: usize) bool {
            return combinedDmaMetrics(all_runs[lhs]).bytesPerSecond() <
                combinedDmaMetrics(all_runs[rhs]).bytesPerSecond();
        }
    }.lessThan);
    return order[order.len / 2];
}

fn medianPairedDmaRunIndex(
    allocator: std.mem.Allocator,
    candidates: []const []DmaBenchmarkRunMetrics,
    baselines: []const []DmaBenchmarkRunMetrics,
) !usize {
    std.debug.assert(candidates.len == baselines.len and candidates.len > 0);
    const order = try allocator.alloc(usize, candidates.len);
    defer allocator.free(order);
    for (order, 0..) |*index, i| index.* = i;
    const Context = struct {
        candidates: []const []DmaBenchmarkRunMetrics,
        baselines: []const []DmaBenchmarkRunMetrics,
    };
    std.mem.sort(usize, order, Context{
        .candidates = candidates,
        .baselines = baselines,
    }, struct {
        fn lessThan(context: Context, lhs: usize, rhs: usize) bool {
            const lhs_baseline = combinedDmaMetrics(context.baselines[lhs]).bytesPerSecond();
            const rhs_baseline = combinedDmaMetrics(context.baselines[rhs]).bytesPerSecond();
            const lhs_ratio = if (lhs_baseline == 0)
                0
            else
                combinedDmaMetrics(context.candidates[lhs]).bytesPerSecond() / lhs_baseline;
            const rhs_ratio = if (rhs_baseline == 0)
                0
            else
                combinedDmaMetrics(context.candidates[rhs]).bytesPerSecond() / rhs_baseline;
            return lhs_ratio < rhs_ratio;
        }
    }.lessThan);
    return order[order.len / 2];
}

fn measureGlobalDmaRounds(
    session: *ReusableDmaBenchmarkSession,
    lanes: []const ReusableDmaBenchmarkLane,
    caps: []const usize,
    candidate_indices: []const usize,
    runs: [][]DmaBenchmarkRunMetrics,
    total_parallelism: usize,
    duration_ns: u64,
    minimum_transfers_per_device: u64,
    total_repeats: usize,
    first_repeat: usize,
    repeat_count: usize,
) !void {
    if (candidate_indices.len == 0 or repeat_count == 0) return;
    for (first_repeat..first_repeat + repeat_count) |repeat| {
        for (0..candidate_indices.len) |offset| {
            const candidate_index = candidate_indices[(offset + repeat) % candidate_indices.len];
            const cap = caps[candidate_index];
            const run = &runs[candidate_index * total_repeats + repeat];
            std.debug.assert(run.len == 0);
            run.* = try session.measure(
                if (cap == total_parallelism) .aggregate else .global_limit,
                lanes,
                duration_ns,
                minimum_transfers_per_device,
                if (cap == total_parallelism) null else cap,
                repeat,
            );
        }
    }
}

fn representativeGlobalDmaRun(
    allocator: std.mem.Allocator,
    runs: []const []DmaBenchmarkRunMetrics,
) ![]DmaBenchmarkRunMetrics {
    return runs[try medianCombinedDmaRunIndex(allocator, runs)];
}

fn globalDmaCandidate(
    parallelism: usize,
    metrics: []const DmaBenchmarkRunMetrics,
    uncapped_metrics: []const DmaBenchmarkRunMetrics,
) GlobalDmaCandidate {
    std.debug.assert(metrics.len == uncapped_metrics.len);
    var min_device_retention: f64 = std.math.inf(f64);
    var utilization_sum: f64 = 0;
    var utilization_squared_sum: f64 = 0;
    for (metrics, uncapped_metrics) |device, baseline| {
        const baseline_rate = baseline.bytesPerSecond();
        const retention = if (baseline_rate == 0) 0 else device.bytesPerSecond() / baseline_rate;
        min_device_retention = @min(min_device_retention, retention);
        utilization_sum += retention;
        utilization_squared_sum += retention * retention;
    }
    const normalized_fairness = if (utilization_squared_sum == 0)
        0
    else
        utilization_sum * utilization_sum /
            (@as(f64, @floatFromInt(metrics.len)) * utilization_squared_sum);
    const aggregate = combinedDmaMetrics(metrics);
    return .{
        .parallelism = parallelism,
        .bytes_per_second = aggregate.bytesPerSecond(),
        .average_latency_ns = aggregate.averageLatencyNs(),
        .min_device_retention = min_device_retention,
        .normalized_fairness = normalized_fairness,
    };
}

fn selectGlobalDmaCandidate(
    candidates: []const GlobalDmaCandidate,
    opts: DmaBenchmarkOpts,
) ?GlobalDmaCandidate {
    if (candidates.len == 0) return null;
    var peak_rate: f64 = 0;
    for (candidates) |candidate| peak_rate = @max(peak_rate, candidate.bytes_per_second);
    const throughput_floor = peak_rate * (1.0 - opts.global_parallelism_selection_tolerance);
    var uncapped = candidates[0];
    for (candidates[1..]) |candidate| {
        if (candidate.parallelism > uncapped.parallelism) uncapped = candidate;
    }
    var selected: ?GlobalDmaCandidate = null;
    for (candidates) |candidate| {
        if (candidate.bytes_per_second < throughput_floor or
            candidate.min_device_retention < opts.global_min_device_retention or
            candidate.normalized_fairness < opts.global_fairness_floor) continue;
        if (selected == null or candidate.parallelism < selected.?.parallelism) selected = candidate;
    }
    return selected;
}

fn shouldRecommendGlobalDmaLimit(
    uncapped: GlobalDmaCandidate,
    candidate: GlobalDmaCandidate,
    uncapped_parallelism: usize,
    opts: DmaBenchmarkOpts,
) bool {
    if (candidate.parallelism >= uncapped_parallelism) return false;
    const uncapped_rate = uncapped.bytes_per_second;
    const candidate_rate = candidate.bytes_per_second;
    const throughput_gain = if (uncapped_rate == 0) 0 else candidate_rate / uncapped_rate;
    // Keep hysteresis around the two-times boundary so measurement noise does
    // not turn an almost-half latency into a machine-wide runtime constraint.
    const latency_improvement = candidate.average_latency_ns * 2 <=
        uncapped.average_latency_ns *
            (1.0 - 2.0 * opts.global_parallelism_selection_tolerance);
    const keeps_throughput = candidate_rate >=
        uncapped_rate * (1.0 - opts.global_parallelism_selection_tolerance);
    return throughput_gain >= 1.0 + opts.global_parallelism_selection_tolerance or
        (latency_improvement and keeps_throughput);
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
    if (platform.devices.len == 0 or opts.block_sizes.len == 0 or opts.parallelism.len == 0)
        return error.NoFeasibleDmaBenchmarkTuple;
    if (opts.duration_ns == 0 or opts.global_duration_ns == 0 or
        opts.confirmation_duration_ns == 0)
        return error.InvalidDmaBenchmarkOptions;
    if (opts.block_parallelism == 0 or opts.block_parallelism > max_load_dma_parallelism)
        return error.InvalidDmaBenchmarkOptions;
    if (!(opts.block_selection_tolerance >= 0 and opts.block_selection_tolerance < 1) or
        !(opts.parallelism_selection_tolerance >= 0 and opts.parallelism_selection_tolerance < 1) or
        !(opts.global_parallelism_selection_tolerance >= 0 and opts.global_parallelism_selection_tolerance < 1) or
        !(opts.confirmation_margin >= 0 and opts.confirmation_margin < 1) or
        !(opts.global_min_device_retention > 0 and opts.global_min_device_retention <= 1) or
        !(opts.global_fairness_floor > 0 and opts.global_fairness_floor <= 1))
        return error.InvalidDmaBenchmarkOptions;
    for (opts.parallelism) |parallelism| {
        if (parallelism == 0 or parallelism > max_load_dma_parallelism)
            return error.InvalidDmaBenchmarkOptions;
    }
    var has_feasible_block = false;
    for (opts.block_sizes) |block_size| {
        if (block_size == 0) return error.InvalidDmaBenchmarkOptions;
        if (dmaBenchmarkTupleFeasible(opts.max_mapped_bytes, block_size, opts.block_parallelism))
            has_feasible_block = true;
    }
    if (!has_feasible_block) return error.NoFeasibleDmaBenchmarkTuple;
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

    var total_parallelism: usize = 0;
    var aggregate_source_bytes: usize = 0;
    for (recommendations.items) |recommendation| {
        total_parallelism = try std.math.add(
            usize,
            total_parallelism,
            recommendation.dma_parallelism,
        );
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

    if (tuned.items.len == 1) {
        const recommendation = recommendations.items[0];
        try source_pools.ensureLoadBlockReserves(
            uniform_block_size,
            used_devices.items.len,
            calibrated_node_reserves,
        );
        const owned_devices = try recommendations.toOwnedSlice(allocator);
        errdefer allocator.free(owned_devices);
        const owned_samples = try samples.toOwnedSlice(allocator);
        errdefer allocator.free(owned_samples);
        const owned_global_candidates = try allocator.alloc(GlobalDmaCandidate, 0);
        errdefer allocator.free(owned_global_candidates);
        const resources = try DmaPlatformSettings.adopt(
            allocator,
            platform,
            .{
                .device_kind = representative_kind,
                .device_ids = platform_device_ids,
                .device_numa_nodes = config_numa_nodes,
                .block_size = recommendation.dma_block_size,
                .max_in_flight_per_device = recommendation.dma_parallelism,
                .global_max_in_flight = null,
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
            .global_candidates = owned_global_candidates,
            .global = .{
                .uncapped_bytes_per_second = recommendation.measured_bytes_per_second,
                .uncapped_average_latency_ns = recommendation.average_latency_ns,
            },
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

    var caps: std.ArrayListUnmanaged(usize) = .empty;
    defer caps.deinit(allocator);
    try appendUniqueUsize(&caps, allocator, tuned.items.len);
    for (opts.parallelism) |per_device| {
        const cap = std.math.mul(usize, per_device, tuned.items.len) catch continue;
        if (cap <= total_parallelism)
            try appendUniqueUsize(&caps, allocator, cap);
    }
    try appendUniqueUsize(&caps, allocator, total_parallelism);
    std.mem.sort(usize, caps.items, {}, std.sort.asc(usize));

    const runs = try allocator.alloc([]DmaBenchmarkRunMetrics, caps.items.len * dma_benchmark_global_repeats);
    @memset(runs, &.{});
    defer {
        for (runs) |metrics| {
            if (metrics.len != 0) allocator.free(metrics);
        }
        allocator.free(runs);
    }
    const global_started_windows = session.windows;
    const uncapped_index = caps.items.len - 1;
    std.debug.assert(caps.items[uncapped_index] == total_parallelism);
    const candidate_indices = try allocator.alloc(usize, caps.items.len);
    defer allocator.free(candidate_indices);
    for (candidate_indices, 0..) |*candidate_index, index| candidate_index.* = index;
    try measureGlobalDmaRounds(
        &session,
        lanes,
        caps.items,
        candidate_indices,
        runs,
        total_parallelism,
        opts.global_duration_ns,
        opts.global_minimum_transfers_per_device,
        dma_benchmark_global_repeats,
        0,
        dma_benchmark_global_repeats,
    );

    const uncapped_runs = runs[uncapped_index * dma_benchmark_global_repeats .. (uncapped_index + 1) * dma_benchmark_global_repeats];
    const uncapped_metrics = try representativeGlobalDmaRun(allocator, uncapped_runs);
    var global_candidates: std.ArrayListUnmanaged(GlobalDmaCandidate) = .empty;
    errdefer global_candidates.deinit(allocator);
    for (caps.items, 0..) |cap, candidate_index| {
        const candidate_runs = runs[candidate_index * dma_benchmark_global_repeats .. (candidate_index + 1) * dma_benchmark_global_repeats];
        const metrics = try representativeGlobalDmaRun(allocator, candidate_runs);
        try global_candidates.append(allocator, globalDmaCandidate(
            cap,
            metrics,
            uncapped_metrics,
        ));
    }
    var final_uncapped_metrics = uncapped_metrics;
    var uncapped_verification_windows: usize = dma_benchmark_global_repeats;
    var uncapped_candidate = global_candidates.items[global_candidates.items.len - 1];
    const selected_candidate = selectGlobalDmaCandidate(global_candidates.items, opts);
    var recommended_candidate: ?GlobalDmaCandidate = if (selected_candidate) |candidate|
        if (shouldRecommendGlobalDmaLimit(
            uncapped_candidate,
            candidate,
            total_parallelism,
            opts,
        )) candidate else null
    else
        null;

    var confirmation_runs: [][]DmaBenchmarkRunMetrics = &.{};
    defer {
        for (confirmation_runs) |metrics| {
            if (metrics.len != 0) allocator.free(metrics);
        }
        if (confirmation_runs.len != 0) allocator.free(confirmation_runs);
    }
    // A global cap outlives calibration. Confirm it against alternating
    // uncapped windows and compare the same repeat before emitting it.
    if (recommended_candidate) |candidate| {
        const confirmation_caps = [_]usize{ candidate.parallelism, total_parallelism };
        const confirmation_indices = [_]usize{ 0, 1 };
        confirmation_runs = try allocator.alloc(
            []DmaBenchmarkRunMetrics,
            confirmation_caps.len * dma_benchmark_repeats,
        );
        @memset(confirmation_runs, &.{});
        try measureGlobalDmaRounds(
            &session,
            lanes,
            &confirmation_caps,
            &confirmation_indices,
            confirmation_runs,
            total_parallelism,
            opts.confirmation_duration_ns,
            opts.confirmation_minimum_transfers_per_device,
            dma_benchmark_repeats,
            0,
            dma_benchmark_repeats,
        );
        const candidate_runs = confirmation_runs[0..dma_benchmark_repeats];
        const baseline_runs = confirmation_runs[dma_benchmark_repeats .. 2 * dma_benchmark_repeats];
        const paired_index = try medianPairedDmaRunIndex(
            allocator,
            candidate_runs,
            baseline_runs,
        );
        final_uncapped_metrics = baseline_runs[paired_index];
        const confirmed_candidate = globalDmaCandidate(
            candidate.parallelism,
            candidate_runs[paired_index],
            final_uncapped_metrics,
        );
        uncapped_candidate = globalDmaCandidate(
            total_parallelism,
            final_uncapped_metrics,
            final_uncapped_metrics,
        );
        const confirmed_selection = selectGlobalDmaCandidate(
            &.{ confirmed_candidate, uncapped_candidate },
            opts,
        );
        recommended_candidate = if (confirmed_selection) |confirmed|
            if (confirmed.parallelism == candidate.parallelism and
                shouldRecommendGlobalDmaLimit(
                    uncapped_candidate,
                    confirmed,
                    total_parallelism,
                    opts,
                )) confirmed else null
        else
            null;
        global_candidates.clearRetainingCapacity();
        try global_candidates.append(allocator, confirmed_candidate);
        try global_candidates.append(allocator, uncapped_candidate);
        uncapped_verification_windows += dma_benchmark_repeats;
    }

    for (recommendations.items, final_uncapped_metrics) |*recommendation, metrics| {
        recommendation.measured_bytes_per_second = metrics.bytesPerSecond();
        recommendation.average_latency_ns = metrics.averageLatencyNs();
        recommendation.windows += uncapped_verification_windows;
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
    const owned_global_candidates = try global_candidates.toOwnedSlice(allocator);
    errdefer allocator.free(owned_global_candidates);
    const resources = try DmaPlatformSettings.adopt(
        allocator,
        platform,
        .{
            .device_kind = representative_kind,
            .device_ids = platform_device_ids,
            .device_numa_nodes = config_numa_nodes,
            .block_size = uniform_block_size,
            .max_in_flight_per_device = uniform_parallelism,
            .global_max_in_flight = if (recommended_candidate) |candidate|
                candidate.parallelism
            else
                null,
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
        .global_candidates = owned_global_candidates,
        .global = .{
            .searched = true,
            .parallelism = if (recommended_candidate) |candidate| candidate.parallelism else null,
            .uncapped_bytes_per_second = uncapped_candidate.bytes_per_second,
            .uncapped_average_latency_ns = uncapped_candidate.average_latency_ns,
            .recommended_bytes_per_second = if (recommended_candidate) |candidate|
                candidate.bytes_per_second
            else
                null,
            .recommended_average_latency_ns = if (recommended_candidate) |candidate|
                candidate.average_latency_ns
            else
                null,
            .recommended_min_device_retention = if (recommended_candidate) |candidate|
                candidate.min_device_retention
            else
                null,
            .recommended_normalized_fairness = if (recommended_candidate) |candidate|
                candidate.normalized_fairness
            else
                null,
            .windows = session.windows - global_started_windows,
        },
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
    log.info("dma_bench version=7 synthetic=true numa_pools={d} retained_mapped_bytes={d} platform={s} devices={d} elapsed_ms={d:.3} calibration_ms={d:.3} allocator_warmup_ms={d:.3} source_registration_ms={d:.3} benchmark_setup_ms={d:.3} sampling_ms={d:.3} benchmark_overhead_ms={d:.3} windows={d}", .{
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
        log.info("dma_bench_sample phase={s} device_index={d} device_id={d} block_bytes={d} parallelism={d} global_parallelism={?d} repeat={d} bytes={d} transfers={d} elapsed_ns={d} gib_s={d:.3} average_latency_ms={d:.3}", .{
            @tagName(sample.phase),
            sample.device_index,
            device.id(),
            sample.block_size,
            sample.parallelism,
            sample.global_parallelism,
            sample.repeat,
            sample.bytes,
            sample.transfers,
            sample.elapsed_ns,
            sample.bytesPerSecond() / (1024 * 1024 * 1024),
            sample.averageLatencyNs() / std.time.ns_per_ms,
        });
    }
    var uncapped_parallelism: usize = 0;
    for (result.devices) |recommendation| {
        uncapped_parallelism += recommendation.dma_parallelism;
    }
    for (result.global_candidates) |candidate| {
        log.info("dma_bench_global_candidate global_parallelism={d} uncapped={} gib_s={d:.3} average_latency_ms={d:.3} min_device_retention={d:.4} normalized_fairness={d:.4}", .{
            candidate.parallelism,
            candidate.parallelism == uncapped_parallelism,
            candidate.bytes_per_second / (1024 * 1024 * 1024),
            candidate.average_latency_ns / std.time.ns_per_ms,
            candidate.min_device_retention,
            candidate.normalized_fairness,
        });
    }
    log.info("dma_bench_global searched={} uncapped_gib_s={d:.3} uncapped_latency_ms={d:.3} recommended_global_parallelism={?d} windows={d}", .{
        result.global.searched,
        result.global.uncapped_bytes_per_second / (1024 * 1024 * 1024),
        result.global.uncapped_average_latency_ns / std.time.ns_per_ms,
        result.global.parallelism,
        result.global.windows,
    });
}

/// Calibrates every addressable device and atomically replaces the platform's
/// private settings. The previous/default settings remain active on failure.
pub fn benchTransfer(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *Platform,
    opts: BenchTransferOptions,
) !void {
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

test "DMA benchmark width plateau stops after two unchanged decisions" {
    var selected: usize = 8;
    var unchanged: usize = 0;
    try std.testing.expect(!updateDmaBenchmarkPlateau(&selected, &unchanged, 8));
    try std.testing.expect(updateDmaBenchmarkPlateau(&selected, &unchanged, 8));
    try std.testing.expectEqual(@as(usize, 2), unchanged);

    try std.testing.expect(!updateDmaBenchmarkPlateau(&selected, &unchanged, 12));
    try std.testing.expectEqual(@as(usize, 12), selected);
    try std.testing.expectEqual(@as(usize, 0), unchanged);
}

test "DMA benchmark paired confirmation compares the same repeat" {
    var candidate_0 = [_]DmaBenchmarkRunMetrics{.{ .bytes = 100, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s }};
    var candidate_1 = [_]DmaBenchmarkRunMetrics{.{ .bytes = 90, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s }};
    var candidate_2 = [_]DmaBenchmarkRunMetrics{.{ .bytes = 120, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s }};
    var baseline_0 = [_]DmaBenchmarkRunMetrics{.{ .bytes = 100, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s }};
    var baseline_1 = [_]DmaBenchmarkRunMetrics{.{ .bytes = 60, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s }};
    var baseline_2 = [_]DmaBenchmarkRunMetrics{.{ .bytes = 150, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s }};
    const candidates = [_][]DmaBenchmarkRunMetrics{
        candidate_0[0..],
        candidate_1[0..],
        candidate_2[0..],
    };
    const baselines = [_][]DmaBenchmarkRunMetrics{
        baseline_0[0..],
        baseline_1[0..],
        baseline_2[0..],
    };
    try std.testing.expectEqual(
        @as(usize, 0),
        try medianPairedDmaRunIndex(std.testing.allocator, &candidates, &baselines),
    );
}

test "DMA benchmark global fairness compares per-device uncapped retention" {
    const uncapped = [_]DmaBenchmarkRunMetrics{
        .{ .bytes = 100, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s },
        .{ .bytes = 50, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s },
    };
    const capped = [_]DmaBenchmarkRunMetrics{
        .{ .bytes = 50, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s },
        .{ .bytes = 40, .transfers = 1, .total_latency_ns = 1, .elapsed_ns = std.time.ns_per_s },
    };
    const candidate = globalDmaCandidate(2, &capped, &uncapped);
    try std.testing.expectApproxEqAbs(@as(f64, 0.5), candidate.min_device_retention, 0.0001);
    try std.testing.expectApproxEqAbs(@as(f64, 0.9494), candidate.normalized_fairness, 0.0001);
    const baseline = globalDmaCandidate(4, &uncapped, &uncapped);
    try std.testing.expectApproxEqAbs(@as(f64, 1), baseline.normalized_fairness, 0.0001);
}

test "loader DMA admission rotates below device count and lends idle slots" {
    const all_ready: u64 = 0b1111;
    var active = [_]usize{ 0, 0, 0, 0 };
    var next_device: usize = 0;

    // A global cap of one still reaches every continuously-ready device.
    for ([_]usize{ 0, 1, 2, 3, 0, 1, 2, 3 }) |expected| {
        const selected = selectLoaderDmaDevice(
            &active,
            8,
            all_ready,
            next_device,
            true,
        ).?;
        try std.testing.expectEqual(expected, selected);
        active[selected] += 1;
        try std.testing.expectEqual(@as(usize, 1), active[selected]);
        next_device = (selected + 1) % active.len;
        active[selected] -= 1;
    }

    // When other devices are idle, their share is lent to the ready device.
    for (0..8) |_| {
        const selected = selectLoaderDmaDevice(
            &active,
            8,
            @as(u64, 1) << 2,
            next_device,
            true,
        ).?;
        try std.testing.expectEqual(@as(usize, 2), selected);
        active[selected] += 1;
        next_device = (selected + 1) % active.len;
    }
    try std.testing.expectEqual(
        @as(?usize, null),
        selectLoaderDmaDevice(&active, 8, @as(u64, 1) << 2, next_device, true),
    );
    active[2] -= 1;
    try std.testing.expectEqual(
        @as(?usize, 3),
        selectLoaderDmaDevice(&active, 8, all_ready, next_device, true),
    );

    // With one long-lived event under a global cap of two, the lending slot
    // continues rotating among every less-loaded ready peer.
    active = .{ 0, 1, 0, 0 };
    next_device = 2;
    for ([_]usize{ 2, 3, 0, 2, 3, 0 }) |expected| {
        const selected = selectLoaderDmaDevice(
            &active,
            8,
            all_ready,
            next_device,
            true,
        ).?;
        try std.testing.expectEqual(expected, selected);
        active[selected] += 1;
        try std.testing.expectEqual(@as(usize, 2), active[0] + active[1] + active[2] + active[3]);
        next_device = (selected + 1) % active.len;
        active[selected] -= 1;
    }
}

test "loader DMA admission bypasses weighting without a global cap" {
    const active = [_]usize{ 7, 0 };
    try std.testing.expectEqual(
        @as(?usize, 0),
        selectLoaderDmaDevice(&active, 8, 0b11, 0, false),
    );
    try std.testing.expectEqual(
        @as(?usize, 1),
        selectLoaderDmaDevice(&active, 8, 0b11, 0, true),
    );
    const saturated = [_]usize{ 8, 0 };
    try std.testing.expectEqual(
        @as(?usize, 1),
        selectLoaderDmaDevice(&saturated, 8, 0b11, 0, false),
    );
}

test "platform DMA global cap is disabled when a device subset cannot reach it" {
    try std.testing.expectEqual(
        @as(?usize, null),
        try effectiveDmaGlobalCap(4, 1, 4),
    );
    try std.testing.expectEqual(
        @as(?usize, 4),
        try effectiveDmaGlobalCap(4, 2, 4),
    );
    try std.testing.expectEqual(
        @as(?usize, null),
        try effectiveDmaGlobalCap(null, 4, 8),
    );
}

test "DMA benchmark fair gate balances, rotates, and lends slots" {
    const allocator = std.testing.allocator;
    const balanced_specs = [_]DmaBenchmarkRunSpec{
        .{ .device_index = 0, .block_size = 4, .parallelism = 8 },
        .{ .device_index = 1, .block_size = 4, .parallelism = 8 },
        .{ .device_index = 2, .block_size = 4, .parallelism = 8 },
        .{ .device_index = 3, .block_size = 4, .parallelism = 8 },
    };
    var balanced = try DmaBenchmarkFairGate.init(allocator, &balanced_specs, 8);
    defer balanced.deinit();
    @memset(balanced.waiting, 8);
    for (0..8) |_| try std.testing.expect(balanced.grantNextLocked() != null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 2, 2, 2 }, balanced.active);
    @memset(balanced.active, 0);
    @memset(balanced.waiting, 0);
    @memset(balanced.grants, 0);
    balanced.active_total = 0;

    var rotating = try DmaBenchmarkFairGate.init(allocator, &balanced_specs, 2);
    defer rotating.deinit();
    @memset(rotating.waiting, 1);
    try std.testing.expectEqual(@as(?usize, 0), rotating.grantNextLocked());
    try std.testing.expectEqual(@as(?usize, 1), rotating.grantNextLocked());
    rotating.grants[0] -= 1;
    rotating.waiting[0] -= 1;
    rotating.grants[1] -= 1;
    rotating.waiting[1] -= 1;
    rotating.active[0] -= 1;
    rotating.active_total -= 1;
    rotating.waiting[0] += 1;
    try std.testing.expectEqual(@as(?usize, 2), rotating.grantNextLocked());
    rotating.active[1] -= 1;
    rotating.active_total -= 1;
    rotating.waiting[1] += 1;
    try std.testing.expectEqual(@as(?usize, 3), rotating.grantNextLocked());
    @memset(rotating.active, 0);
    @memset(rotating.waiting, 0);
    @memset(rotating.grants, 0);
    rotating.active_total = 0;

    var lending = try DmaBenchmarkFairGate.init(allocator, &balanced_specs, 4);
    defer lending.deinit();
    lending.waiting[2] = 4;
    for (0..4) |_| try std.testing.expectEqual(@as(?usize, 2), lending.grantNextLocked());
    try std.testing.expectEqual(@as(usize, 4), lending.active[2]);
    @memset(lending.active, 0);
    @memset(lending.waiting, 0);
    @memset(lending.grants, 0);
    lending.active_total = 0;

    const weighted_specs = [_]DmaBenchmarkRunSpec{
        .{ .device_index = 0, .block_size = 4, .parallelism = 4 },
        .{ .device_index = 1, .block_size = 4, .parallelism = 8 },
    };
    var weighted = try DmaBenchmarkFairGate.init(allocator, &weighted_specs, 6);
    defer weighted.deinit();
    @memset(weighted.waiting, 8);
    for (0..6) |_| try std.testing.expect(weighted.grantNextLocked() != null);
    try std.testing.expectEqualSlices(usize, &.{ 2, 4 }, weighted.active);
    @memset(weighted.active, 0);
    @memset(weighted.waiting, 0);
    @memset(weighted.grants, 0);
    weighted.active_total = 0;
}

test "DMA benchmark global limit policy requires measured contention and benefit" {
    const opts: DmaBenchmarkOpts = .{};
    const candidates = [_]GlobalDmaCandidate{
        .{ .parallelism = 32, .bytes_per_second = 100, .average_latency_ns = 1_000, .min_device_retention = 1, .normalized_fairness = 0.999 },
        .{ .parallelism = 4, .bytes_per_second = 96, .average_latency_ns = 300, .min_device_retention = 0.95, .normalized_fairness = 0.99 },
        .{ .parallelism = 8, .bytes_per_second = 99, .average_latency_ns = 400, .min_device_retention = 0.97, .normalized_fairness = 0.995 },
    };
    const selected = selectGlobalDmaCandidate(&candidates, opts).?;
    try std.testing.expectEqual(@as(usize, 8), selected.parallelism);
    try std.testing.expect(shouldRecommendGlobalDmaLimit(candidates[0], selected, 32, opts));

    const starved = [_]GlobalDmaCandidate{
        candidates[0],
        .{ .parallelism = 8, .bytes_per_second = 101, .average_latency_ns = 300, .min_device_retention = 0.80, .normalized_fairness = 0.90 },
    };
    try std.testing.expectEqual(@as(usize, 32), selectGlobalDmaCandidate(&starved, opts).?.parallelism);

    const topology_asymmetric = [_]GlobalDmaCandidate{
        .{ .parallelism = 16, .bytes_per_second = 90, .average_latency_ns = 600, .min_device_retention = 0.90, .normalized_fairness = 0.95 },
        .{ .parallelism = 32, .bytes_per_second = 101, .average_latency_ns = 700, .min_device_retention = 1.02, .normalized_fairness = 0.995 },
        .{ .parallelism = 64, .bytes_per_second = 80, .average_latency_ns = 1_000, .min_device_retention = 1, .normalized_fairness = 1 },
    };
    try std.testing.expectEqual(
        @as(usize, 32),
        selectGlobalDmaCandidate(&topology_asymmetric, opts).?.parallelism,
    );

    const no_benefit: GlobalDmaCandidate = .{
        .parallelism = 16,
        .bytes_per_second = 100,
        .average_latency_ns = 900,
        .min_device_retention = 0.99,
        .normalized_fairness = 0.999,
    };
    const borderline_latency: GlobalDmaCandidate = .{
        .parallelism = 16,
        .bytes_per_second = 99,
        .average_latency_ns = 490,
        .min_device_retention = 0.99,
        .normalized_fairness = 0.999,
    };
    try std.testing.expect(!shouldRecommendGlobalDmaLimit(candidates[0], no_benefit, 32, opts));
    try std.testing.expect(!shouldRecommendGlobalDmaLimit(candidates[0], borderline_latency, 32, opts));
    try std.testing.expect(!shouldRecommendGlobalDmaLimit(candidates[0], candidates[0], 32, opts));
}

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
    stdx.debug.assert(read_initial > 0, "zml.io.load read_parallelism initial/fixed value must be greater than zero", .{});
    stdx.debug.assert(read_maximum >= read_initial, "zml.io.load read_parallelism maximum must be at least initial", .{});
    stdx.debug.assert(read_maximum <= max_load_read_parallelism, "zml.io.load read_parallelism exceeds the absolute limit", .{});

    const direct = platform.target == .cuda or platform.target == .rocm or platform.target == .oneapi;
    var used_devices: ?DmaUsedDevices = null;
    defer if (used_devices) |*devices| devices.deinit();
    var dma_resources: ?*DmaPlatformSettings = null;
    defer if (dma_resources != null) releasePlatformDmaSettings(platform);
    if (direct) {
        const placement_tensors = try allocator.alloc(*const Tensor, meta.count(Tensor, model));
        defer allocator.free(placement_tensors);
        meta.forEachVisit(model, *const Tensor, struct {
            fn call(i: usize, tensor: *const Tensor, output: []*const Tensor) void {
                output[i] = tensor;
            }
        }.call, .{placement_tensors});
        used_devices = try dmaUsedDevicesForTensors(
            allocator,
            platform,
            placement_tensors,
            opts.shardings,
        );
        const resources = try acquirePlatformDmaSettings(
            platform,
            used_devices.?.device_ids,
        );
        dma_resources = resources;
        if (resources.config.block_size > load_read_request_size or
            resources.config.max_mapped_bytes < load_read_request_size)
            return error.InvalidDmaLoadConfig;
    }

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

    const dma_config: DmaLoadConfig = if (dma_resources) |resources| resources.config else .{
        .device_kind = "buffered",
        .device_ids = &.{},
        .device_numa_nodes = &.{},
        .block_size = 0,
        .max_in_flight_per_device = 0,
        .global_max_in_flight = null,
        .max_mapped_bytes = 0,
    };
    load_log.debug("configured: target={s}, vectored={}, tensors={d}, max_read_parallelism={d}, dma_parallelism_per_device={d}, global_dma_parallelism={?d}, read_request_size={Bi:.2}, dma_block_size={Bi:.2}, max_mapped_bytes={Bi:.2}, logical_bytes={Bi:.2}", .{
        @tagName(platform.target),
        direct,
        tensor_count,
        read_maximum,
        dma_config.max_in_flight_per_device,
        dma_config.global_max_in_flight,
        if (direct) load_read_request_size else 0,
        dma_config.block_size,
        dma_config.max_mapped_bytes,
        total_logical_bytes,
    });

    const loaded_bytes = if (direct)
        try loadVectored(
            ModelType,
            model,
            &bufferized,
            allocator,
            io,
            platform,
            store,
            opts,
            dma_resources.?,
            used_devices.?.device_ids,
            load_started,
        )
    else
        try loadBuffered(ModelType, model, &bufferized, allocator, io, platform, store, opts);
    if (opts.total_bytes) |total_bytes| total_bytes.* = loaded_bytes;
    return bufferized;
}

test "source bootstrap requires a high-latency source with no observed response" {
    try std.testing.expect(shouldBootstrapSource(true, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(false, false, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, true, 0, 12, 12, true));
    try std.testing.expect(!shouldBootstrapSource(true, false, 1, 12, 12, true));
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
    metrics.recordProbeRead(io, 7, 40, load_read_request_size);
    metrics.beginRead(io, 7, 41);
    metrics.recordProbeRead(io, 7, 41, load_read_request_size);
    const admitted = metrics.snapshot(io);
    try std.testing.expectEqual(@as(usize, 3), admitted.active_reads);
    try std.testing.expect(admitted.probe_first_read_ns != 0);
    try std.testing.expectEqual(@as(usize, 1), admitted.probe_active_reads);
    try std.testing.expectEqual(@as(u64, 1), admitted.probe_full_read_operations);
    try std.testing.expectEqual(@as(u64, load_read_request_size), admitted.probe_read_bytes);
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

test "pinned feasibility clips read width and remote lifecycle slack" {
    const clipped: PinnedGateLimits = .init(128, 16, 8);
    try std.testing.expectEqual(@as(usize, 16), clipped.feasible_width);
    try std.testing.expectEqual(@as(usize, 16), clipped.read);
    try std.testing.expectEqual(@as(usize, 16), clipped.lifecycle);

    const slack: PinnedGateLimits = .init(12, 16, 8);
    try std.testing.expectEqual(@as(usize, 12), slack.read);
    try std.testing.expectEqual(@as(usize, 16), slack.lifecycle);
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
