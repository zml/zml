const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

/// Creates a vxfw.Widget from any pointer whose type has a `draw` method
/// (and optionally a `handleEvent` method).
pub fn widget(ptr: anytype) vxfw.Widget {
    const T = @typeInfo(@TypeOf(ptr)).pointer.child;
    var result = drawWidget(ptr, T.draw);
    result.eventHandler = if (@hasDecl(T, "handleEvent")) struct {
        fn handle(pointer: *anyopaque, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
            const self: *T = @ptrCast(@alignCast(pointer));
            return self.handleEvent(ctx, event);
        }
    }.handle else null;
    return result;
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
