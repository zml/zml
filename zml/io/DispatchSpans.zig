//! Converts a resolved sharding into source byte ranges and packed destination
//! offsets. Mirrored ranges share one span and a mask of their destinations.

const std = @import("std");

const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Placement = Sharding.Placement;

pub const InitError = std.mem.Allocator.Error || Sharding.Error;

const DispatchSpans = @This();

const Span = struct {
    start: usize,
    end: usize,
    writer_offset: usize,
    writer_mask: u64,
};

const PlacementSpan = struct {
    writer_index: usize,
    start: usize,
    len: usize,
};

spans: []Span,

pub fn init(allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) InitError!DispatchSpans {
    const placement = try sharding.placement(shape);
    const ordered_devices = sharding.devicesInCanonicalOrder();
    std.debug.assert(ordered_devices.len <= 64);

    var placement_spans: std.ArrayList(PlacementSpan) = .empty;
    defer placement_spans.deinit(allocator);

    const byte_strides = shape.computeByteStrides();

    for (ordered_devices, 0..) |device, writer_index| {
        try appendShardPlacementSpans(allocator, &placement_spans, shape, placement.slices(device.coords).constSlice(), byte_strides.constSlice(), writer_index);
    }

    var spans: std.ArrayList(Span) = try .initCapacity(allocator, placement_spans.items.len);
    errdefer spans.deinit(allocator);

    const writer_offsets = try allocator.alloc(usize, ordered_devices.len);
    defer allocator.free(writer_offsets);
    @memset(writer_offsets, 0);
    try deduplicateByRange(allocator, placement_spans.items, shape.byteSize(), &spans, writer_offsets);

    return .{ .spans = try spans.toOwnedSlice(allocator) };
}

pub fn deinit(self: DispatchSpans, allocator: std.mem.Allocator) void {
    allocator.free(self.spans);
}

pub fn spanIndexAt(self: DispatchSpans, offset: usize) ?usize {
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
    spans: *std.ArrayList(Span),
    writer_offsets: []usize,
) std.mem.Allocator.Error!void {
    const SortContext = struct {
        fn lessThan(_: void, lhs: PlacementSpan, rhs: PlacementSpan) bool {
            if (lhs.start != rhs.start) return lhs.start < rhs.start;
            if (lhs.len != rhs.len) return lhs.len < rhs.len;
            return lhs.writer_index < rhs.writer_index;
        }
    };

    std.mem.sort(PlacementSpan, placement_spans, {}, SortContext.lessThan);

    var i: usize = 0;
    var cursor: usize = 0;
    while (i < placement_spans.len) {
        const span = placement_spans[i];
        // The spans tile `[0, byteSize)`: `Placement.init` divides every
        // sharded axis with `@divExact` (`zml/Sharding.zig`), so the shards
        // of one axis cover it exactly, and `Planner.appendTransfers`
        // already relies on that tiling when it maps a job's bytes to
        // destinations.
        std.debug.assert(span.start == cursor);

        // Record packed offsets in file order so request tasks can finish
        // out of order without mutating writer cursors.
        const writer_offset = writer_offsets[span.writer_index];
        var writer_mask: u64 = 0;
        var j = i;
        while (j < placement_spans.len) : (j += 1) {
            const member = placement_spans[j];
            if (member.start != span.start or member.len != span.len) break;
            std.debug.assert(writer_offsets[member.writer_index] == writer_offset);
            writer_offsets[member.writer_index] += span.len;
            writer_mask |= @as(u64, 1) << @intCast(member.writer_index);
        }

        try spans.append(allocator, .{
            .start = span.start,
            .end = span.start + span.len,
            .writer_offset = writer_offset,
            .writer_mask = writer_mask,
        });
        cursor += span.len;
        i = j;
    }

    std.debug.assert(cursor == total_bytes);
}

fn appendShardPlacementSpans(
    allocator: std.mem.Allocator,
    placement_spans: *std.ArrayList(PlacementSpan),
    shape: Shape,
    slices: []const Placement.Slice1d,
    byte_strides: []const i64,
    writer_index: usize,
) std.mem.Allocator.Error!void {
    if (shape.rank() == 0) {
        try placement_spans.append(allocator, .{ .writer_index = writer_index, .start = 0, .len = shape.byteSize() });
        return;
    }

    try appendShardAxisPlacementSpans(allocator, placement_spans, slices, byte_strides, writer_index, 0, contiguousSliceAxis(shape, slices), 0);
}

fn appendShardAxisPlacementSpans(
    allocator: std.mem.Allocator,
    placement_spans: *std.ArrayList(PlacementSpan),
    slices: []const Placement.Slice1d,
    byte_strides: []const i64,
    writer_index: usize,
    axis: usize,
    contiguous_axis: usize,
    base_start: i64,
) std.mem.Allocator.Error!void {
    const slice = slices[axis];
    if (slice.size == 0) return;

    if (axis == contiguous_axis) {
        try placement_spans.append(allocator, .{
            .writer_index = writer_index,
            .start = @intCast(base_start + slice.start * byte_strides[axis]),
            .len = @intCast(slice.size * byte_strides[axis]),
        });
        return;
    }

    var i: i64 = 0;
    while (i < slice.size) : (i += 1) {
        const child_start = base_start + (slice.start + i) * byte_strides[axis];
        try appendShardAxisPlacementSpans(allocator, placement_spans, slices, byte_strides, writer_index, axis + 1, contiguous_axis, child_start);
    }
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

test "dispatch spans preserve mirrored ranges and packed writer offsets" {
    const allocator = std.testing.allocator;
    var placements = [_]DispatchSpans.PlacementSpan{
        .{ .writer_index = 63, .start = 8, .len = 4 },
        .{ .writer_index = 2, .start = 12, .len = 4 },
        .{ .writer_index = 0, .start = 0, .len = 4 },
        .{ .writer_index = 1, .start = 4, .len = 4 },
        .{ .writer_index = 63, .start = 0, .len = 4 },
        .{ .writer_index = 0, .start = 8, .len = 4 },
        .{ .writer_index = 2, .start = 4, .len = 4 },
        .{ .writer_index = 1, .start = 12, .len = 4 },
    };
    var spans: std.ArrayList(DispatchSpans.Span) = .empty;
    defer spans.deinit(allocator);
    var offsets: [64]usize = @splat(0);
    try DispatchSpans.deduplicateByRange(allocator, &placements, 16, &spans, &offsets);
    const expected = [_]DispatchSpans.Span{
        .{ .start = 0, .end = 4, .writer_offset = 0, .writer_mask = 0x8000000000000001 },
        .{ .start = 4, .end = 8, .writer_offset = 0, .writer_mask = 0b110 },
        .{ .start = 8, .end = 12, .writer_offset = 4, .writer_mask = 0x8000000000000001 },
        .{ .start = 12, .end = 16, .writer_offset = 4, .writer_mask = 0b110 },
    };
    try std.testing.expectEqualDeep(expected[0..], spans.items);
    for ([_]usize{ 0, 1, 2, 63 }) |writer| try std.testing.expectEqual(8, offsets[writer]);
    for (offsets[3..63]) |offset| try std.testing.expectEqual(0, offset);

    const dispatch: DispatchSpans = .{ .spans = spans.items };
    for (0..16) |offset| try std.testing.expectEqual(@as(?usize, offset / 4), dispatch.spanIndexAt(offset));
    try std.testing.expectEqual(@as(?usize, null), dispatch.spanIndexAt(16));
}
