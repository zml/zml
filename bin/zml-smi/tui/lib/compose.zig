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

pub const PadValues = struct {
    left: u16 = 0,
    right: u16 = 0,
    top: u16 = 0,
    bottom: u16 = 0,
};

/// Padding wrapper matching vxfw.Padding without its non-null ctx.max asserts,
/// so unbounded-height layouts (e.g. fixedWidth) work cleanly.
pub const Padding = struct {
    child: vxfw.Widget,
    padding: PadValues = .{},

    pub fn widget(self: *const Padding) vxfw.Widget {
        return .{
            .userdata = @constCast(self),
            .drawFn = typeErasedDrawFn,
        };
    }

    fn typeErasedDrawFn(ptr: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const self: *const Padding = @ptrCast(@alignCast(ptr));
        return self.draw(ctx);
    }

    pub fn draw(self: *const Padding, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const p = self.padding;
        const inner_min: vxfw.Size = .{
            .width = ctx.min.width -| (p.right + p.left),
            .height = ctx.min.height -| (p.top + p.bottom),
        };
        const inner_max: vxfw.MaxSize = .{
            .width = if (ctx.max.width) |max| max -| (p.right + p.left) else null,
            .height = if (ctx.max.height) |max| max -| (p.top + p.bottom) else null,
        };

        const child_surface = try self.child.draw(ctx.withConstraints(inner_min, inner_max));

        const children = try ctx.arena.alloc(vxfw.SubSurface, 1);
        children[0] = .{
            .surface = child_surface,
            .z_index = 0,
            .origin = .{ .row = p.top, .col = p.left },
        };

        return .{
            .size = .{
                .width = child_surface.size.width + (p.right + p.left),
                .height = child_surface.size.height + (p.top + p.bottom),
            },
            .widget = self.widget(),
            .buffer = &.{},
            .children = children,
        };
    }
};

pub fn pad(arena: std.mem.Allocator, child: vxfw.Widget, padding: PadValues) std.mem.Allocator.Error!vxfw.Widget {
    const p = try arena.create(Padding);
    p.* = .{ .child = child, .padding = padding };
    return p.widget();
}
