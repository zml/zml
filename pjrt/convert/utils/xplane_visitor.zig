const math_utils = @import("math_utils.zig");
const std = @import("std");
const xplane_proto = @import("//tsl:xplane_proto");
const xplane_schema = @import("xplane_schema.zig");

pub const XEventMetadataVisitor = struct {
    plane: *const XPlaneVisitor,
    stats_owner: *const xplane_proto.XEventMetadata,

    pub fn init(plane: *const XPlaneVisitor, metadata: *const xplane_proto.XEventMetadata) XEventMetadataVisitor {
        return .{
            .plane = plane,
            .stats_owner = metadata,
        };
    }

    pub fn forEachStat(
        self: *const XEventMetadataVisitor,
        allocator: std.mem.Allocator,
        cb: fn (allocator: std.mem.Allocator, xstat: XStatVisitor, ctx: ?*anyopaque) std.mem.Allocator.Error!void,
        ctx: ?*anyopaque,
    ) !void {
        for (self.stats_owner.stats.items) |*stat| {
            try cb(allocator, XStatVisitor.init(self.plane, stat), ctx);
        }
    }
};

pub const XStatVisitor = struct {
    stat: *const xplane_proto.XStat,
    metadata: *const xplane_proto.XStatMetadata,
    plane: *const XPlaneVisitor,
    type_: ?xplane_schema.StatType = null,

    pub fn init(plane: *const XPlaneVisitor, stat: *const xplane_proto.XStat) XStatVisitor {
        return XStatVisitor.internalInit(
            plane,
            stat,
            plane.getStatMetadata(stat.metadata_id),
            plane.getStatType(stat.metadata_id),
        );
    }

    pub fn internalInit(
        plane: *const XPlaneVisitor,
        stat: *const xplane_proto.XStat,
        metadata: *const xplane_proto.XStatMetadata,
        type_: ?xplane_schema.StatType,
    ) XStatVisitor {
        return .{
            .stat = stat,
            .metadata = metadata,
            .plane = plane,
            .type_ = type_,
        };
    }

    pub fn @"type"(self: *const XStatVisitor) ?xplane_schema.StatType {
        return self.type_;
    }

    pub fn name(self: *const XStatVisitor) []const u8 {
        return self.metadata.name.getSlice();
    }

    pub fn toString(self: *const XStatVisitor, allocator: std.mem.Allocator) ![]const u8 {
        var out = std.ArrayList(u8).init(allocator);
        var writer = out.writer();
        if (self.stat.value) |vc| {
            switch (vc) {
                inline .int64_value, .uint64_value, .double_value => |v| try writer.print("{d}", .{v}),
                .str_value => |*v| try writer.writeAll(v.getSlice()),
                .bytes_value => try writer.writeAll("<opaque bytes>"),
                .ref_value => |v| try writer.writeAll(self.plane.getStatMetadata(@intCast(v)).name.getSlice()),
            }
        }
        return out.toOwnedSlice();
    }
};

pub const XEventVisitor = struct {
    plane: *const XPlaneVisitor,
    line: *const xplane_proto.XLine,
    event: *const xplane_proto.XEvent,
    metadata: *const xplane_proto.XEventMetadata,
    type_: ?xplane_schema.HostEventType,

    pub fn init(
        plane: *const XPlaneVisitor,
        line: *const xplane_proto.XLine,
        event: *const xplane_proto.XEvent,
    ) XEventVisitor {
        return .{
            .plane = plane,
            .line = line,
            .event = event,
            .metadata = plane.getEventMetadata(event.metadata_id),
            .type_ = plane.getEventType(event.metadata_id),
        };
    }

    pub fn hasDisplayName(self: *const XEventVisitor) bool {
        return self.metadata.display_name != .Empty;
    }

    pub fn displayName(self: *const XEventVisitor) []const u8 {
        return self.metadata.display_name.getSlice();
    }

    pub fn name(self: *const XEventVisitor) []const u8 {
        return self.metadata.name.getSlice();
    }

    pub fn timestampPs(self: *const XEventVisitor) u128 {
        return math_utils.nanoToPico(@intCast(self.line.timestamp_ns)) + @as(u128, @intCast(self.event.data.?.offset_ps));
    }

    pub fn durationPs(self: *const XEventVisitor) i64 {
        return self.event.duration_ps;
    }

    pub fn metadataVisitor(self: *const XEventVisitor) XEventMetadataVisitor {
        return XEventMetadataVisitor.init(self.plane, self.metadata);
    }

    pub fn forEachStat(
        self: *const XEventVisitor,
        allocator: std.mem.Allocator,
        cb: fn (allocator: std.mem.Allocator, xstat: XStatVisitor, ctx: ?*anyopaque) std.mem.Allocator.Error!void,
        ctx: ?*anyopaque,
    ) !void {
        for (self.event.stats.items) |*stat| {
            try cb(allocator, XStatVisitor.init(self.plane, stat), ctx);
        }
    }
};

pub const XLineVisitor = struct {
    plane: *const XPlaneVisitor,
    line: *const xplane_proto.XLine,

    pub fn init(plane: *const XPlaneVisitor, line: *const xplane_proto.XLine) XLineVisitor {
        return .{
            .plane = plane,
            .line = line,
        };
    }

    pub fn id(self: *const XLineVisitor) i64 {
        return self.line.display_id;
    }

    pub fn displayId(self: *const XLineVisitor) i64 {
        return if (self.line.display_id != 0) self.line.display_id else self.line.id;
    }

    pub fn name(self: *const XLineVisitor) []const u8 {
        return self.line.name.getSlice();
    }

    pub fn displayName(self: *const XLineVisitor) []const u8 {
        return if (self.line.display_name != .Empty) self.line.display_name.getSlice() else self.name();
    }

    pub fn timestampNs(self: *const XLineVisitor) i64 {
        return self.line.timestamp_ns;
    }

    pub fn durationPs(self: *const XLineVisitor) i64 {
        return self.line.duration_ps;
    }

    pub fn numEvents(self: *const XLineVisitor) usize {
        return self.line.events.len;
    }

    pub fn forEachEvent(
        self: *const XLineVisitor,
        allocator: std.mem.Allocator,
        cb: fn (allocator: std.mem.Allocator, event: XEventVisitor, ctx: ?*anyopaque) std.mem.Allocator.Error!void,
        ctx: ?*anyopaque,
    ) !void {
        for (self.line.events.items) |*event| {
            try cb(allocator, XEventVisitor.init(self.plane, self.line, event), ctx);
        }
    }
};

const XEventMetadataInstance = xplane_proto.XEventMetadata.init();
const XStatMetadataInstance = xplane_proto.XStatMetadata.init();

pub const XPlaneVisitor = struct {
    plane: *const xplane_proto.XPlane,
    event_type_by_id: std.AutoHashMapUnmanaged(i64, xplane_schema.HostEventType) = .{},
    stat_type_by_id: std.AutoHashMapUnmanaged(i64, xplane_schema.StatType) = .{},
    stat_metadata_by_type: std.AutoHashMapUnmanaged(xplane_schema.StatType, *const xplane_proto.XStatMetadata) = .{},

    const EventTypeGetter = *const fn ([]const u8) ?xplane_schema.HostEventType;
    const StatTypeGetter = *const fn ([]const u8) ?xplane_schema.StatType;

    pub fn init(
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
        event_type_getter_list: []const EventTypeGetter,
        stat_type_getter_list: []const StatTypeGetter,
    ) !XPlaneVisitor {
        var res: XPlaneVisitor = .{ .plane = plane };
        try res.buildEventTypeMap(allocator, plane, event_type_getter_list);
        try res.buildStatTypeMap(allocator, plane, stat_type_getter_list);
        return res;
    }

    pub fn buildEventTypeMap(
        self: *XPlaneVisitor,
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
        event_type_getter_list: []const EventTypeGetter,
    ) !void {
        if (event_type_getter_list.len == 0) return;
        for (plane.event_metadata.items) |event_metadata| {
            const metadata_id = event_metadata.key;
            const metadata = event_metadata.value.?;
            for (event_type_getter_list) |event_type_getter| {
                if (event_type_getter(metadata.name.getSlice())) |event_type| {
                    try self.event_type_by_id.put(allocator, metadata_id, event_type);
                    break;
                }
            }
        }
    }

    pub fn buildStatTypeMap(
        self: *XPlaneVisitor,
        allocator: std.mem.Allocator,
        plane: *const xplane_proto.XPlane,
        stat_type_getter_list: []const StatTypeGetter,
    ) !void {
        if (stat_type_getter_list.len == 0) return;
        for (plane.stat_metadata.items) |stat_metadata| {
            const metadata_id = stat_metadata.key;
            const metadata = stat_metadata.value.?;
            for (stat_type_getter_list) |stat_type_getter| {
                if (stat_type_getter(metadata.name.getSlice())) |stat_type| {
                    try self.stat_type_by_id.put(allocator, metadata_id, stat_type);
                    try self.stat_metadata_by_type.put(allocator, stat_type, &stat_metadata.value.?);
                    break;
                }
            }
        }
    }

    pub fn getEventType(self: *const XPlaneVisitor, event_metadata_id: i64) ?xplane_schema.HostEventType {
        return self.event_type_by_id.get(event_metadata_id);
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

    pub fn getStatType(self: *const XPlaneVisitor, stat_metadata_id: i64) ?xplane_schema.StatType {
        return self.stat_type_by_id.get(stat_metadata_id);
    }

    pub fn forEachLine(
        self: *const XPlaneVisitor,
        allocator: std.mem.Allocator,
        cb: fn (allocator: std.mem.Allocator, xline: XLineVisitor, ctx: ?*anyopaque) std.mem.Allocator.Error!void,
        ctx: ?*anyopaque,
    ) !void {
        for (self.plane.lines.items) |*line| {
            try cb(allocator, XLineVisitor.init(self, line), ctx);
        }
    }
};
