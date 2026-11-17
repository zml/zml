const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

// ── SurfaceBuilder ─────────────────────────────────────────────────

pub const SurfaceBuilder = struct {
    arena: std.mem.Allocator,
    children: std.ArrayList(vxfw.SubSurface),

    pub fn add(self: *SurfaceBuilder, row: i17, col: i17, surface: vxfw.Surface) !void {
        try self.addZ(row, col, surface, 0);
    }

    pub fn addZ(self: *SurfaceBuilder, row: i17, col: i17, surface: vxfw.Surface, z_index: u8) !void {
        try self.children.append(self.arena, .{ .origin = .{ .row = row, .col = col }, .surface = surface, .z_index = z_index });
    }

    pub fn finish(self: *SurfaceBuilder, size: vxfw.Size, wgt: vxfw.Widget) vxfw.Surface {
        return .{ .size = size, .widget = wgt, .buffer = &.{}, .children = self.children.items };
    }
};

pub fn surfaceBuilder(arena: std.mem.Allocator) SurfaceBuilder {
    return .{ .arena = arena, .children = .empty };
}
