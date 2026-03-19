const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

// ── SurfaceBuilder ─────────────────────────────────────────────────

pub const SurfaceBuilder = struct {
    arena: std.mem.Allocator,
    children: std.ArrayList(vxfw.SubSurface),

    pub fn add(self: *SurfaceBuilder, row: i17, col: i17, surface: vxfw.Surface) !void {
        try self.children.append(self.arena, .{ .origin = .{ .row = row, .col = col }, .surface = surface });
    }

    pub fn finish(self: *SurfaceBuilder, size: vxfw.Size, wgt: vxfw.Widget) vxfw.Surface {
        return .{ .size = size, .widget = wgt, .buffer = &.{}, .children = self.children.items };
    }
};

pub fn surfaceBuilder(arena: std.mem.Allocator) SurfaceBuilder {
    return .{ .arena = arena, .children = .empty };
}

// ── Fluent widget wrappers ─────────────────────────────────────────

pub fn center(arena: std.mem.Allocator, child: vxfw.Widget) std.mem.Allocator.Error!vxfw.Widget {
    const c = try arena.create(vxfw.Center);
    c.* = .{ .child = child };
    return c.widget();
}

pub fn sized(arena: std.mem.Allocator, child: vxfw.Widget, size: vxfw.Size) std.mem.Allocator.Error!vxfw.Widget {
    const s = try arena.create(vxfw.SizedBox);
    s.* = .{ .child = child, .size = size };
    return s.widget();
}

pub fn pad(arena: std.mem.Allocator, child: vxfw.Widget, padding: vxfw.Padding.PadValues) std.mem.Allocator.Error!vxfw.Widget {
    const p = try arena.create(vxfw.Padding);
    p.* = .{ .child = child, .padding = padding };
    return p.widget();
}
