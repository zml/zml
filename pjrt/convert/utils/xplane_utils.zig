const std = @import("std");
const xplane_proto = @import("//tsl:xplane_proto");

pub fn findPlaneWithName(space: *const xplane_proto.XSpace, name: []const u8) ?*xplane_proto.XPlane {
    for (space.planes.items) |*v| {
        if (std.mem.eql(u8, v.name.getSlice(), name)) return v;
    }
    return null;
}

pub fn findPlanesWithPrefix(
    allocator: std.mem.Allocator,
    space: *const xplane_proto.XSpace,
    prefix: []const u8,
) ![]*const xplane_proto.XPlane {
    var res = std.ArrayList(*const xplane_proto.XPlane).init(allocator);
    for (space.planes.items) |*p| {
        if (std.mem.startsWith(u8, p.name.getSlice(), prefix)) try res.append(p);
    }
    return res.toOwnedSlice();
}
