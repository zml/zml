const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const data = @import("data.zig");
const ui = @import("lib/ui.zig");
const compose = @import("lib/compose.zig");
const image_cache = @import("image_cache.zig");
const Logo = @import("widgets/logo.zig");
const ProcessTable = @import("widgets/process_table.zig");
const Overview = @import("pages/overview.zig");
const Detail = @import("pages/detail.zig");
const StatusLine = @import("widgets/status_line.zig");
const Selection = @import("selection.zig").Selection;

const min_width: u16 = 45;
const min_height: u16 = 24;

const ScrollState = struct {
    row: i17 = 0,
    col: i17 = 0,

    pub fn reset(self: *ScrollState) void {
        self.row = 0;
        self.col = 0;
    }

    pub fn scrollBy(self: *ScrollState, dr: i17, dc: i17) void {
        self.row = @max(0, self.row + dr);
        self.col = @max(0, self.col + dc);
    }

    pub fn clamp(self: *ScrollState, content_h: u16, viewport_h: u16, content_w: u16, viewport_w: u16) void {
        const max_row: i17 = @max(0, @as(i17, content_h) - @as(i17, viewport_h));
        const max_col: i17 = @max(0, @as(i17, content_w) - @as(i17, viewport_w));
        self.row = std.math.clamp(self.row, 0, max_row);
        self.col = std.math.clamp(self.col, 0, max_col);
    }
};

const Model = struct {
    allocator: std.mem.Allocator,
    state: *data.SystemState,
    viewing_device: ?u16 = null,
    prev_viewing_device: ?u16 = null,
    scroll: ScrollState = .{},
    vx: *vaxis.Vaxis,
    tty: *vaxis.Tty,
    process_table: ProcessTable = .{},
    overview: Overview = undefined,
    selection: Selection = .{},
    /// One-shot: scroll the page to reveal the current selection on next draw.
    pending_scroll_to_sel: bool = false,

    fn deinit(self: *Model) void {
        self.process_table.deinit(self.allocator);
        self.overview.deinit(self.allocator);
        image_cache.global.deinit(self.allocator);
    }

    pub fn handleEvent(self: *Model, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        switch (event) {
            .init => {
                self.process_table.selection = &self.selection;
                self.overview = try Overview.init(self.allocator, self.state, &self.process_table, &self.viewing_device, &self.selection);
                self.overview.use_braille = true;
                image_cache.global.loadAll(self.vx, self.allocator, self.tty.writer());
                try ctx.tick(1000, ui.widget(self));
            },
            .tick => {
                self.state.recordHistory();
                try ctx.tick(self.state.tui_refresh_rate, ui.widget(self));
                ctx.redraw = true;
            },
            .key_press => |key| {
                if (key.matches('q', .{}) or key.matches('c', .{ .ctrl = true })) {
                    ctx.quit = true;
                    return;
                }
                if (self.viewing_device != null) {
                    // ── Detail page ──
                    if (key.matches(vaxis.Key.escape, .{}) or key.matches(vaxis.Key.backspace, .{})) {
                        self.viewing_device = null;
                        return ctx.consumeAndRedraw();
                    }
                } else {
                    // ── Overview page ──
                    if (key.matches('v', .{})) {
                        self.overview.use_braille = !self.overview.use_braille;
                        return ctx.consumeAndRedraw();
                    }
                    if (key.matches(vaxis.Key.enter, .{})) {
                        // Enter on a selected device opens it, just like a click.
                        if (self.selection.kind == .device) {
                            self.viewing_device = self.selection.device;
                            return ctx.consumeAndRedraw();
                        }
                    }
                }
                // ── Shared selection / kill / horizontal-scroll keys ──
                if (key.matches('x', .{})) {
                    self.process_table.killSelected(); // no-op unless a process is selected
                    return ctx.consumeAndRedraw();
                }
                if (key.matches(vaxis.Key.down, .{}) or key.matches('j', .{}) or key.matches('n', .{ .ctrl = true })) {
                    return self.moveSelection(ctx, 1);
                }
                if (key.matches(vaxis.Key.up, .{}) or key.matches('k', .{}) or key.matches('p', .{ .ctrl = true })) {
                    return self.moveSelection(ctx, -1);
                }

                if (key.matches(vaxis.Key.right, .{})) {
                    if (self.process_table.scrollCommand(1)) {
                        ctx.consumeAndRedraw();
                    }
                } else if (key.matches(vaxis.Key.left, .{})) {
                    if (self.process_table.scrollCommand(-1)) {
                        ctx.consumeAndRedraw();
                    }
                }

                return;
            },
            .mouse => |mouse| {
                // remove selection
                if (mouse.type == .press and mouse.button == .left) {
                    if (self.selection.kind != .none) {
                        self.selection.clear();
                        return ctx.consumeAndRedraw();
                    }
                    return;
                }
                switch (mouse.button) {
                    .wheel_up => self.scroll.scrollBy(-3, 0),
                    .wheel_down => self.scroll.scrollBy(3, 0),
                    .wheel_left => if (!self.process_table.scrollCommand(1)) return,
                    .wheel_right => if (!self.process_table.scrollCommand(-1)) return,
                    else => return,
                }
                return ctx.consumeAndRedraw();
            },
            else => {},
        }
    }

    fn currentSelIndex(self: *const Model, dev_n: usize) ?usize {
        switch (self.selection.kind) {
            .device => if (self.viewing_device == null and self.selection.device < dev_n) return self.selection.device,
            .process => if (self.process_table.rowIndexOfPid(self.selection.pid)) |r| return dev_n + r,
            .none => {},
        }
        return null;
    }

    fn moveSelection(self: *Model, ctx: *vxfw.EventContext, delta: i32) void {
        const dev_n = if (self.viewing_device == null) self.state.deviceCount() else 0;
        const item_count = dev_n + self.process_table.merged.items.len;

        if (item_count == 0) {
            return ctx.consumeEvent();
        }

        // Pick the target index. With nothing selected yet, the first press
        // selects the top item; otherwise step by delta, clamped to the list.
        var next: usize = 0;
        if (self.currentSelIndex(dev_n)) |cur| {
            const last: i32 = @intCast(item_count - 1);
            const moved: i32 = @as(i32, @intCast(cur)) + delta;

            next = @intCast(std.math.clamp(moved, 0, last));
        }

        if (next < dev_n) { // A device card.
            _ = self.selection.setDevice(@intCast(next));
        } else { // A process row
            const row = next - dev_n;

            _ = self.selection.setProcess(self.process_table.merged.items[row].pid);
            self.process_table.scroll_bars.scroll_view.cursor = @intCast(row);
            self.process_table.scroll_bars.scroll_view.ensureScroll();
        }

        self.pending_scroll_to_sel = true;
        return ctx.consumeAndRedraw();
    }

    /// Pointer to the widget the current selection lives on, for auto-scroll.
    fn selectionTargetPtr(self: *Model) ?*anyopaque {
        switch (self.selection.kind) {
            .device => if (self.viewing_device == null and self.selection.device < self.overview.device_cards.len)
                return @ptrCast(&self.overview.device_cards[self.selection.device]),
            .process => if (self.process_table.rowIndexOfPid(self.selection.pid)) |r|
                return @ptrCast(&self.process_table.rows.items[r]),
            .none => {},
        }
        return null;
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const screen = ctx.max.size();
        const content_w = @max(screen.width, min_width);

        if (self.viewing_device != self.prev_viewing_device) {
            self.scroll.reset();
            self.process_table.resetScroll();
            self.selection.clear();
            self.prev_viewing_device = self.viewing_device;
        }

        self.process_table.prepare(self.state, self.viewing_device);

        var content_surf: vxfw.Surface = undefined;

        if (self.viewing_device) |dev_id| {
            const detail: Detail = .{
                .state = self.state,
                .device_id = dev_id,
                .process_table = &self.process_table,
            };
            content_surf = try detail.draw(ui.fixedWidth(ctx, content_w));
        } else {
            content_surf = try self.overview.draw(ui.fixedWidth(ctx, content_w));
        }

        content_surf.size.height = @max(content_surf.size.height, min_height);

        const content_h = screen.height -| 1;

        // Bring the keyboard selection into view
        // TODO: maybe find another way of doing that, see `findWidgetY` func below
        if (self.pending_scroll_to_sel) {
            self.pending_scroll_to_sel = false;

            if (self.selectionTargetPtr()) |target| {
                if (findWidgetY(content_surf, target, 0)) |found| {
                    const view_h: i17 = @intCast(content_h);
                    const bot = found.y + @as(i17, @intCast(found.h));

                    if (found.y < self.scroll.row) {
                        self.scroll.row = found.y;
                    } else if (bot > self.scroll.row + view_h) {
                        self.scroll.row = bot - view_h;
                    }
                }
            }
        }

        self.scroll.clamp(content_surf.size.height, content_h, content_surf.size.width, screen.width);

        const status_line: StatusLine = .{
            .viewing_device = self.viewing_device,
            .use_braille = self.overview.use_braille,
            .can_kill = self.process_table.killableSelected(),
        };
        const status_surf = try status_line.draw(ui.fixedSize(ctx, screen.width, 1));

        var sb = compose.surfaceBuilder(ctx.arena);
        try sb.add(-self.scroll.row, -self.scroll.col, content_surf);
        try sb.addZ(@intCast(content_h), 0, status_surf, 1);
        return sb.finish(screen, ui.widget(self));
    }
};

const FoundWidget = struct { y: i17, h: u16 };

// TODO: probably not optimal but we'll see once we bump libvaxis
fn findWidgetY(surface: vxfw.Surface, target: *anyopaque, base: i17) ?FoundWidget {
    if (surface.widget.userdata == target) return .{ .y = base, .h = surface.size.height };
    for (surface.children) |child| {
        if (findWidgetY(child.surface, target, base + child.origin.row)) |found| return found;
    }
    return null;
}

pub fn run(allocator: std.mem.Allocator, io: std.Io, state: *data.SystemState) !void {
    var app = try vxfw.App.init(allocator, io);
    defer app.deinit();

    var model: Model = .{
        .allocator = allocator,
        .state = state,
        .vx = &app.vx,
        .tty = &app.tty,
    };
    defer model.deinit();

    try app.run(ui.widget(&model), .{});
}
