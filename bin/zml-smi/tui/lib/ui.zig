const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

/// Creates a vxfw.Widget from any pointer whose type has a `draw` method
/// (and optionally a `handleEvent` method).
pub fn widget(ptr: anytype) vxfw.Widget {
    const Ptr = @TypeOf(ptr);
    const T = @typeInfo(Ptr).pointer.child;
    return .{
        .userdata = @ptrCast(@constCast(ptr)),
        .drawFn = struct {
            fn f(p: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
                const self: *T = @ptrCast(@alignCast(p));
                return self.draw(ctx);
            }
        }.f,
        .eventHandler = if (@hasDecl(T, "handleEvent")) struct {
            fn f(p: *anyopaque, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
                const self: *T = @ptrCast(@alignCast(p));
                return self.handleEvent(ctx, event);
            }
        }.f else null,
    };
}

/// Creates a vxfw.Widget that calls a specific draw function (for non-standard widget patterns
/// like MetricCard.drawContent or Chart.drawContent).
pub fn drawWidget(ptr: anytype, comptime drawFn: anytype) vxfw.Widget {
    const Ptr = @TypeOf(ptr);
    const T = @typeInfo(Ptr).pointer.child;
    return .{
        .userdata = @ptrCast(@constCast(ptr)),
        .drawFn = struct {
            fn f(p: *anyopaque, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
                const self: *T = @ptrCast(@alignCast(p));
                return @call(.auto, drawFn, .{ self, ctx });
            }
        }.f,
    };
}

// ── DrawContext constraint helpers ──────────────────────────────────

/// min = max = {w, h} — exact size.
pub fn fixedSize(ctx: vxfw.DrawContext, w: u16, h: u16) vxfw.DrawContext {
    return ctx.withConstraints(.{ .width = w, .height = h }, .{ .width = w, .height = h });
}

/// Width fixed (min = max = w), height unbounded.
pub fn fixedWidth(ctx: vxfw.DrawContext, w: u16) vxfw.DrawContext {
    return ctx.withConstraints(.{ .width = w }, .{ .width = w, .height = null });
}

/// max = {w, h}, min = default — bounded but unconstrained from below.
pub fn maxSize(ctx: vxfw.DrawContext, w: u16, h: u16) vxfw.DrawContext {
    return ctx.withConstraints(.{}, .{ .width = w, .height = h });
}

// ── Drawing helpers ────────────────────────────────────────────────

/// Draw a single-line RichText with softwrap=false, overflow=clip, max width.
pub fn drawRichLine(ctx: vxfw.DrawContext, segments: []const vaxis.Cell.Segment, max_w: u16) std.mem.Allocator.Error!vxfw.Surface {
    const rich: vxfw.RichText = .{
        .text = segments,
        .softwrap = false,
        .overflow = .clip,
    };
    return rich.draw(ctx.withConstraints(.{}, .{ .width = max_w, .height = 1 }));
}

pub const ImageCellSize = struct {
    cols: u16,
    rows: u16,
};

/// Compute the cell dimensions for an image scaled to a target row count,
/// maintaining aspect ratio based on the terminal's cell pixel size.
/// The terminal stretches the image over exactly these cells, so the width
/// is rounded to the nearest column: rounding up would widen the image.
pub fn imageCellSize(img: vaxis.Image, target_rows: u16, cell_size: vxfw.Size) ImageCellSize {
    if (img.width == 0 or img.height == 0 or target_rows == 0) return .{ .cols = 0, .rows = 0 };

    const has_cell = cell_size.width > 0 and cell_size.height > 0;
    const cell_h: f64 = @floatFromInt(if (has_cell) cell_size.height else 20);
    const cell_w: f64 = @floatFromInt(if (has_cell) cell_size.width else 10);
    const height_px = @as(f64, @floatFromInt(target_rows)) * cell_h;
    const width_px = @as(f64, @floatFromInt(img.width)) * (height_px / @as(f64, @floatFromInt(img.height)));
    const cols_f = std.math.clamp(@round(width_px / cell_w), 1.0, @as(f64, @floatFromInt(std.math.maxInt(u16))));
    return .{ .cols = @intFromFloat(cols_f), .rows = target_rows };
}
