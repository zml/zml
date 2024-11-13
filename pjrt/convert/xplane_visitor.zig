const std = @import("std");
const xplane_proto = @import("//tsl:xplane_proto");
const xplane_schema = @import("xplane_schema.zig");

const XEventMetadataInstance = xplane_proto.XEventMetadata.init();
const XStatMetadataInstance = xplane_proto.XStatMetadata.init();

pub const XPlaneVisitor = struct {
    plane: *const xplane_proto.XPlane,
    event_type_by_id: std.AutoHashMapUnmanaged(i64, xplane_schema.HostEventType) = .{},
    stat_type_by_id: std.AutoHashMapUnmanaged(i64, xplane_schema.StatType) = .{},
    stat_metadata_by_type: std.AutoHashMapUnmanaged(xplane_schema.StatType, *const xplane_proto.XStatMetadata) = .{},

    const EventTypeGetter = *const fn ([]const u8) xplane_schema.HostEventType;
    const StatTypeGetter = *const fn ([]const u8) xplane_schema.StatType;

    pub fn init(
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
        event_type_getter: ?EventTypeGetter,
        stat_type_getter: ?StatTypeGetter,
    ) !XPlaneVisitor {
        var res: XPlaneVisitor = .{ .plane = plane };
        try res.buildEventTypeMap(allocator, plane, event_type_getter);
        try res.buildStatTypeMap(allocator, plane, stat_type_getter);
        return res;
    }

    pub fn buildEventTypeMap(
        self: *XPlaneVisitor,
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
        event_type_getter: ?EventTypeGetter,
    ) !void {
        if (event_type_getter) |getter| {
            for (plane.event_metadata.items) |event_metadata| {
                const metadata_id = event_metadata.key;
                const metadata = event_metadata.value.?;
                try self.event_type_by_id.put(allocator, metadata_id, getter(metadata.name.getSlice()));
            }
        } else return;
    }

    pub fn buildStatTypeMap(
        self: *XPlaneVisitor,
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
        stat_type_getter: ?StatTypeGetter,
    ) !void {
        if (stat_type_getter) |getter| {
            for (plane.stat_metadata.items) |stat_metadata| {
                const metadata_id = stat_metadata.key;
                const metadata = stat_metadata.value.?;
                const stat_type = getter(metadata.name.getSlice());
                try self.stat_type_by_id.put(allocator, metadata_id, stat_type);
                try self.stat_metadata_by_type.put(allocator, stat_type, &stat_metadata.value.?);
            }
        } else return;
    }

    pub fn getEventType(self: *const XPlaneVisitor, event_metadata_id: i64) xplane_schema.HostEventType {
        return self.event_type_by_id.get(event_metadata_id) orelse .unknown;
    }

    pub fn name(self: *const XPlaneVisitor) []const u8 {
        return self.plane.name.getSlice();
    }

    pub fn getEventMetadata(self: *const XPlaneVisitor, event_metadata_id: i64) *const xplane_proto.XEventMetadata {
        for (self.plane.event_metadata.items) |*event_metadata| {
            if (event_metadata.value) |*v| {
                if (v.id == event_metadata_id) return v;
            }
        }

        return &XEventMetadataInstance;
    }

    pub fn getStatMetadata(self: *const XPlaneVisitor, stat_metadata_id: i64) *const xplane_proto.XStatMetadata {
        for (self.plane.stat_metadata.items) |*stat_metadata| {
            if (stat_metadata.value) |*v| {
                if (v.id == stat_metadata_id) return v;
            }
        }
        return &XStatMetadataInstance;
    }

    pub fn getStatType(self: *const XPlaneVisitor, stat_metadata_id: i64) xplane_schema.StatType {
        return self.stat_type_by_id.get(stat_metadata_id) orelse .unknown;
    }
};
