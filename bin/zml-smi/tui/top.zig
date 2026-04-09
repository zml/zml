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

    fn deinit(self: *Model) void {
        self.overview.deinit(self.allocator);
        image_cache.global.deinit(self.allocator);
    }

    pub fn handleEvent(self: *Model, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        switch (event) {
            .init => {
                self.overview = try Overview.init(self.allocator, self.state, &self.process_table, &self.viewing_device);
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
                    if (key.matches(vaxis.Key.escape, .{}) or key.matches(vaxis.Key.backspace, .{})) {
                        self.viewing_device = null;
                        return ctx.consumeAndRedraw();
                    }
                } else {
                    if (key.matches('v', .{})) {
                        self.overview.use_braille = !self.overview.use_braille;
                        return ctx.consumeAndRedraw();
                    }
                }
                if (key.matches(vaxis.Key.down, .{})) {
                    self.scroll.scrollBy(1, 0);
                } else if (key.matches(vaxis.Key.up, .{})) {
                    self.scroll.scrollBy(-1, 0);
                } else if (key.matches(vaxis.Key.right, .{})) {
                    self.scroll.scrollBy(0, 1);
                } else if (key.matches(vaxis.Key.left, .{})) {
                    self.scroll.scrollBy(0, -1);
                } else return;
                return ctx.consumeAndRedraw();
            },
            .mouse => |mouse| {
                switch (mouse.button) {
                    .wheel_up => self.scroll.scrollBy(-3, 0),
                    .wheel_down => self.scroll.scrollBy(3, 0),
                    .wheel_left => self.scroll.scrollBy(0, 3),
                    .wheel_right => self.scroll.scrollBy(0, -3),
                    else => return,
                }
                return ctx.consumeAndRedraw();
            },
            else => {},
        }
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const screen = ctx.max.size();
        const content_w = @max(screen.width, min_width);

        if (self.viewing_device != self.prev_viewing_device) {
            self.scroll.reset();
            self.process_table.resetScroll();
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
        self.scroll.clamp(content_surf.size.height, content_h, content_surf.size.width, screen.width);

        const status_line: StatusLine = .{
            .viewing_device = self.viewing_device,
            .use_braille = self.overview.use_braille,
        };
        const status_surf = try status_line.draw(ui.fixedSize(ctx, screen.width, 1));

        var sb = compose.surfaceBuilder(ctx.arena);
        try sb.add(-self.scroll.row, -self.scroll.col, content_surf);
        try sb.addZ(@intCast(content_h), 0, status_surf, 1);
        return sb.finish(screen, ui.widget(self));
    }
};

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
