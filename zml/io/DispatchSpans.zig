//! Converts a resolved sharding into source byte ranges and packed destination
//! offsets. Mirrored ranges share one span and a mask of their destinations.

const std = @import("std");

const Shape = @import("../shape.zig").Shape;
const Sharding = @import("../Sharding.zig");
const Placement = Sharding.Placement;

pub const InitError = std.mem.Allocator.Error || Sharding.Error;

const DispatchSpans = @This();

const DispatchSpan = struct {
    start: usize,
    end: usize,
    // in which devices should this span be written in.
    device_mask: u64,
    // where to write within the device's buffer.
    device_offset: usize,
};

const DeviceSpan = struct {
    device_index: usize,
    start: usize,
    len: usize,
};

spans: []DispatchSpan,

pub fn init(allocator: std.mem.Allocator, shape: Shape, sharding: Sharding) InitError!DispatchSpans {
    const placement = try sharding.placement(shape);
    const ordered_devices = sharding.devicesInCanonicalOrder();
    std.debug.assert(ordered_devices.len <= 64);

    var device_spans: std.ArrayList(DeviceSpan) = .empty;
    defer device_spans.deinit(allocator);
    try collectDeviceSpans(
        allocator,
        &device_spans,
        shape,
        placement,
        ordered_devices,
    );

    var spans: std.ArrayList(DispatchSpan) = try .initCapacity(allocator, device_spans.items.len);
    errdefer spans.deinit(allocator);
    try mergeDeviceSpans(
        allocator,
        &spans,
        device_spans.items,
        shape.byteSize(),
        ordered_devices.len,
    );

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

/// Convert each device's rectangular tensor slice into contiguous source byte
/// ranges. Placement supplies { start, size } in elements for each tensor axis;
/// byte strides convert those coordinates into offsets in the source tensor.
///
/// For a [4, 8]f32 tensor stored row by row, each letter is one element:
///
///     Source tensor             Device's first two columns
///     a b c d e f g h           a b
///     i j k l m n o p           i j
///     q r s t u v w x           q r
///     y z A B C D E F           y z
///
/// Placement returns one { start, size } per axis:
///
///     {
///         {0, 4}, // rows
///         {0, 2}, // columns
///     }
///
/// Byte strides describe how far to move in the source for one step per axis:
///
///     {
///         32, // next row: 8 f32 elements
///          4, // next column: 1 f32 element
///     }
///
/// We emit four source byte ranges:
///
///     a b -> [ 0,   8)
///     i j -> [32,  40)
///     q r -> [64,  72)
///     y z -> [96, 104)
///
/// Between a b and i j, we skip c d e f g h.
///
/// Replicated devices produce duplicate ranges here. mergeDeviceSpans merges
/// them into destination masks and assigns packed offsets within each device.
fn collectDeviceSpans(
    allocator: std.mem.Allocator,
    device_spans: *std.ArrayList(DeviceSpan),
    shape: Shape,
    placement: Placement,
    ordered_devices: []const Sharding.Device,
) std.mem.Allocator.Error!void {
    const byte_strides = shape.computeByteStrides();

    if (shape.rank() == 0) { // scalar
        for (0..ordered_devices.len) |device_index| {
            try device_spans.append(allocator, .{
                .device_index = device_index,
                .start = 0,
                .len = shape.byteSize(),
            });
        }
        return;
    }

    const Context = struct {
        allocator: std.mem.Allocator,
        device_spans: *std.ArrayList(DeviceSpan),
        slices: []const Placement.Slice1d,
        byte_strides: []const i64,
        device_index: usize,
        contiguous_axis: usize,

        fn appendSpans(self: @This(), axis: usize, base_start: i64) std.mem.Allocator.Error!void {
            const slice = self.slices[axis];
            if (slice.size == 0) return;

            // All later axes are selected in full, so this axis's selected
            // values and their trailing dimensions form one contiguous span.
            if (axis == self.contiguous_axis) {
                try self.device_spans.append(self.allocator, .{
                    .device_index = self.device_index,
                    .start = @intCast(base_start + slice.start * self.byte_strides[axis]),
                    .len = @intCast(slice.size * self.byte_strides[axis]),
                });
                return;
            }

            // Earlier axes need separate visits: selecting the first two
            // columns of each row requires one span per row.
            var i: i64 = 0;
            while (i < slice.size) : (i += 1) {
                const child_start = base_start + (slice.start + i) * self.byte_strides[axis];
                try self.appendSpans(axis + 1, child_start);
            }
        }
    };

    for (ordered_devices, 0..) |device, device_index| {
        // One { start, size } per tensor axis, measured in elements.
        // For the first two columns of a [4, 8] tensor: {0, 4}, {0, 2}.
        const slices = placement.slices(device.coords);
        const context: Context = .{
            .allocator = allocator,
            .device_spans = device_spans,
            .slices = slices.constSlice(),
            .byte_strides = byte_strides.constSlice(),
            .device_index = device_index,
            .contiguous_axis = firstContiguousAxis(shape, slices.constSlice()),
        };
        try context.appendSpans(0, 0);
    }
}

/// Find the outermost axis where traversal can emit one contiguous span.
/// In this [4, 8] tensor, brackets mark the values selected for a device:
///
///     First two columns           First two complete rows
///     [a b] c d e f g h           [a b c d e f g h
///     [i j] k l m n o p            i j k l m n o p]
///     [q r] s t u v w x            q r s t u v w x
///     [y z] A B C D E F            y z A B C D E F
///
/// Left: axis 1 (columns). ab, ij, qr, yz need four separate spans because
/// unselected values lie between them in memory.
/// Right: axis 0 (rows). a through p form one span: h is followed directly by i
/// in memory, so both complete rows can be emitted together.
///
/// Move outward only while the current axis is selected in full.
fn firstContiguousAxis(shape: Shape, slices: []const Placement.Slice1d) usize {
    var axis = shape.rank() - 1;
    while (axis > 0) {
        const slice = slices[axis];
        if (slice.start != 0 or slice.size != shape.dim(axis)) break;
        axis -= 1;
    }
    return axis;
}

fn mergeDeviceSpans(
    allocator: std.mem.Allocator,
    spans: *std.ArrayList(DispatchSpan),
    device_spans: []DeviceSpan,
    total_bytes: usize,
    device_count: usize,
) std.mem.Allocator.Error!void {
    const device_offsets = try allocator.alloc(usize, device_count);
    defer allocator.free(device_offsets);
    @memset(device_offsets, 0);

    const SortContext = struct {
        fn lessThan(_: void, lhs: DeviceSpan, rhs: DeviceSpan) bool {
            if (lhs.start != rhs.start) return lhs.start < rhs.start;
            if (lhs.len != rhs.len) return lhs.len < rhs.len;
            return lhs.device_index < rhs.device_index;
        }
    };

    std.mem.sort(DeviceSpan, device_spans, {}, SortContext.lessThan);

    var i: usize = 0;
    var cursor: usize = 0;
    while (i < device_spans.len) {
        const span = device_spans[i];
        std.debug.assert(span.start == cursor);

        const device_offset = device_offsets[span.device_index];
        var device_mask: u64 = 0;
        var j = i;
        while (j < device_spans.len) : (j += 1) {
            const member = device_spans[j];
            if (member.start != span.start or member.len != span.len) break;
            std.debug.assert(device_offsets[member.device_index] == device_offset);
            device_offsets[member.device_index] += span.len;
            device_mask |= @as(u64, 1) << @intCast(member.device_index);
        }

        try spans.append(allocator, .{
            .start = span.start,
            .end = span.start + span.len,
            .device_offset = device_offset,
            .device_mask = device_mask,
        });
        cursor += span.len;
        i = j;
    }

    std.debug.assert(cursor == total_bytes);
}

test "dispatch spans preserve mirrored ranges and packed device offsets" {
    const allocator = std.testing.allocator;
    var device_spans = [_]DispatchSpans.DeviceSpan{
        .{ .device_index = 63, .start = 8, .len = 4 },
        .{ .device_index = 2, .start = 12, .len = 4 },
        .{ .device_index = 0, .start = 0, .len = 4 },
        .{ .device_index = 1, .start = 4, .len = 4 },
        .{ .device_index = 63, .start = 0, .len = 4 },
        .{ .device_index = 0, .start = 8, .len = 4 },
        .{ .device_index = 2, .start = 4, .len = 4 },
        .{ .device_index = 1, .start = 12, .len = 4 },
    };
    var spans: std.ArrayList(DispatchSpans.DispatchSpan) = .empty;
    defer spans.deinit(allocator);
    try DispatchSpans.mergeDeviceSpans(allocator, &spans, &device_spans, 16, 64);
    const expected = [_]DispatchSpans.DispatchSpan{
        .{ .start = 0, .end = 4, .device_offset = 0, .device_mask = 0x8000000000000001 },
        .{ .start = 4, .end = 8, .device_offset = 0, .device_mask = 0b110 },
        .{ .start = 8, .end = 12, .device_offset = 4, .device_mask = 0x8000000000000001 },
        .{ .start = 12, .end = 16, .device_offset = 4, .device_mask = 0b110 },
    };
    try std.testing.expectEqualDeep(expected[0..], spans.items);

    const dispatch: DispatchSpans = .{ .spans = spans.items };
    for (0..16) |offset| try std.testing.expectEqual(@as(?usize, offset / 4), dispatch.spanIndexAt(offset));
    try std.testing.expectEqual(@as(?usize, null), dispatch.spanIndexAt(16));
}
