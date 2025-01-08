const std = @import("std");
const xplane_proto = @import("//tsl:xplane_proto");
const xplane_schema = @import("xplane_schema.zig");

pub const XPlaneVisitor = struct {
    plane: *const xplane_proto.XPlane,
    event_metadata_by_id: std.AutoHashMapUnmanaged(i64, *const xplane_proto.XEventMetadata) = .{},
    stat_metadata_by_id: std.AutoHashMapUnmanaged(i64, *const xplane_proto.XStatMetadata) = .{},

    pub fn init(
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
    ) !XPlaneVisitor {
        var res: XPlaneVisitor = .{ .plane = plane };

        try res.event_metadata_by_id.ensureUnusedCapacity(allocator, @intCast(plane.event_metadata.items.len));
        // build event metadata map
        for (plane.event_metadata.items) |*event_metadata| {
            res.event_metadata_by_id.putAssumeCapacity(event_metadata.key, &event_metadata.value.?);
        }

        // build stat metadata map
        try res.stat_metadata_by_id.ensureUnusedCapacity(allocator, @intCast(plane.stat_metadata.items.len));
        for (plane.stat_metadata.items) |*stat_metadata| {
            res.stat_metadata_by_id.putAssumeCapacity(stat_metadata.key, &stat_metadata.value.?);
        }

        return res;
    }

    pub fn deinit(self: *XPlaneVisitor, allocator: std.mem.Allocator) void {
        self.stat_metadata_by_id.deinit(allocator);
        self.event_metadata_by_id.deinit(allocator);
    }

    pub fn name(self: XPlaneVisitor) []const u8 {
        return self.plane.name.getSlice();
    }

    pub fn getEventType(self: XPlaneVisitor, event_metadata_id: i64) xplane_schema.HostEventType {
        if (self.event_metadata_by_id.get(event_metadata_id)) |v| {
            return xplane_schema.HostEventType.fromString(v.name.getSlice());
        } else return .unknown;
    }

    pub fn getStatMetadataName(self: XPlaneVisitor, stat_metadata_id: i64) []const u8 {
        if (self.stat_metadata_by_id.get(stat_metadata_id)) |v| {
            return v.name.getSlice();
        } else return &[_]u8{};
    }

    pub fn getStatType(self: XPlaneVisitor, stat_metadata_id: i64) xplane_schema.StatType {
        if (self.stat_metadata_by_id.get(stat_metadata_id)) |v| {
            return xplane_schema.StatType.fromString(v.name.getSlice());
        } else return .unknown;
    }

    pub fn xstatToString(self: XPlaneVisitor, stat: xplane_proto.XStat, writer: anytype) !void {
        if (stat.value == null) return;

        switch (stat.value.?) {
            inline .int64_value, .uint64_value, .double_value => |v| try writer.print("{d}", .{v}),
            .str_value => |*v| try writer.writeAll(v.getSlice()),
            .bytes_value => try writer.writeAll("<opaque bytes>"),
            .ref_value => |v| try writer.writeAll(self.getStatMetadataName(@intCast(v))),
        }
    }
};
