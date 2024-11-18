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

        // build event metadata map
        for (plane.event_metadata.items) |*event_metadata| {
            try res.event_metadata_by_id.put(allocator, event_metadata.key, &event_metadata.value.?);
        }

        // build stat metadata map
        for (plane.stat_metadata.items) |*stat_metadata| {
            try res.stat_metadata_by_id.put(allocator, stat_metadata.key, &stat_metadata.value.?);
        }

        return res;
    }

    pub fn getEventType(self: *const XPlaneVisitor, event_metadata_id: i64) xplane_schema.HostEventType {
        if (self.event_metadata_by_id.get(event_metadata_id)) |v| {
            return xplane_schema.HostEventType.fromString(v.name.getSlice());
        } else return .unknown;
    }

    pub fn name(self: *const XPlaneVisitor) []const u8 {
        return self.plane.name.getSlice();
    }

    pub fn getStatMetadataName(self: *const XPlaneVisitor, stat_metadata_id: i64) []const u8 {
        if (self.stat_metadata_by_id.get(stat_metadata_id)) |v| {
            return v.name.getSlice();
        } else return &[_]u8{};
    }

    pub fn getStatType(self: *const XPlaneVisitor, stat_metadata_id: i64) xplane_schema.StatType {
        if (self.stat_metadata_by_id.get(stat_metadata_id)) |v| {
            return xplane_schema.StatType.fromString(v.name.getSlice());
        } else return .unknown;
    }
};
