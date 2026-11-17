const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;

const data = @import("data.zig");
const ui = @import("lib/ui.zig");
const compose = @import("lib/compose.zig");
const image_cache = @import("image_cache.zig");
const ProcessTable = @import("widgets/process_table.zig");
const Overview = @import("pages/overview.zig");
const detail = @import("pages/detail.zig");
const StatusLine = @import("widgets/status_line.zig");

const min_width: u16 = 45;
const min_height: u16 = 24;

const Model = struct {
    terminal: *vxfw.App,
    prev_viewing_device: ?u16 = null,
    scroll: vxfw.RelativePoint = .{ .row = 0, .col = 0 },
    process_table: ProcessTable = .{},
    overview: Overview,

    fn scrollBy(self: *Model, rows: i17, columns: i17) void {
        self.scroll.row = @max(0, self.scroll.row + rows);
        self.scroll.col = @max(0, self.scroll.col + columns);
    }

    pub fn handleEvent(self: *Model, ctx: *vxfw.EventContext, event: vxfw.Event) anyerror!void {
        switch (event) {
            .init, .tick => {
                const state = self.overview.state;
                if (event == .init) {
                    const app = self.terminal;
                    image_cache.loadAll(&app.vx, app.allocator, app.tty.writer(), state.logos);
                }
                try ctx.tick(if (event == .init) 1000 else state.tui_refresh_rate, ui.widget(self));
                if (event == .tick) ctx.redraw = true;
            },
            .key_press => |key| {
                if (key.matches('q', .{}) or key.matches('c', .{ .ctrl = true })) {
                    ctx.quit = true;
                    return;
                }
                const back = key.matchesAny(&.{ vaxis.Key.escape, vaxis.Key.backspace }, .{});
                if (self.overview.viewing_device != null and back) {
                    self.overview.viewing_device = null;
                } else if (self.overview.viewing_device == null and key.matches('v', .{})) {
                    self.overview.use_braille = !self.overview.use_braille;
                } else if (key.matches(vaxis.Key.down, .{})) {
                    self.scrollBy(1, 0);
                } else if (key.matches(vaxis.Key.up, .{})) {
                    self.scrollBy(-1, 0);
                } else if (key.matches(vaxis.Key.right, .{})) {
                    self.scrollBy(0, 1);
                } else if (key.matches(vaxis.Key.left, .{})) {
                    self.scrollBy(0, -1);
                } else return;
                return ctx.consumeAndRedraw();
            },
            .mouse => |mouse| {
                switch (mouse.button) {
                    .wheel_up => self.scrollBy(-3, 0),
                    .wheel_down => self.scrollBy(3, 0),
                    .wheel_left => self.scrollBy(0, 3),
                    .wheel_right => self.scrollBy(0, -3),
                    else => return,
                }
                return ctx.consumeAndRedraw();
            },
            else => {},
        }
    }

    pub fn draw(self: *Model, ctx: vxfw.DrawContext) std.mem.Allocator.Error!vxfw.Surface {
        const state = self.overview.state;
        const viewing_device = self.overview.viewing_device;
        const screen = ctx.max.size();
        const content_w = @max(screen.width, min_width);
        state.history_len = @max(state.history_len, 2 * @as(usize, content_w));

        if (viewing_device != self.prev_viewing_device) {
            self.scroll = .{ .row = 0, .col = 0 };
            self.process_table.resetScroll();
            self.prev_viewing_device = viewing_device;
        }

        try state.capture(ctx.arena);
        try self.process_table.prepare(ctx.arena, state.frame.processes, viewing_device);

        const content_ctx = ui.fixedWidth(ctx, content_w);
        var content_surf = if (viewing_device) |index|
            try detail.draw(content_ctx, state, index, &self.process_table)
        else
            try self.overview.draw(content_ctx);

        content_surf.size.height = @max(content_surf.size.height, min_height);

        const content_h = screen.height -| 1;
        self.scroll.row = std.math.clamp(self.scroll.row, 0, content_surf.size.height -| content_h);
        self.scroll.col = std.math.clamp(self.scroll.col, 0, content_surf.size.width -| screen.width);

        const status_line: StatusLine = .{
            .viewing_device = viewing_device,
            .use_braille = self.overview.use_braille,
        };
        const status_surf = try status_line.draw(ui.fixedSize(ctx, screen.width, 1));

        var sb = compose.surfaceBuilder(ctx.arena);
        try sb.add(-self.scroll.row, -self.scroll.col, content_surf);
        try sb.addZ(@intCast(content_h), 0, status_surf, 1);
        return sb.finish(screen, ui.widget(self));
    }
};

pub fn run(allocator: std.mem.Allocator, io: std.Io, env_map: *std.process.Environ.Map, state: *data.SystemState) !void {
    var tty_buffer: [4096]u8 = undefined;
    var app = try vxfw.App.init(io, allocator, env_map, &tty_buffer);
    defer app.deinit();

    var model: Model = .{
        .terminal = &app,
        .overview = .{ .state = state, .use_braille = true },
    };
    model.overview.process_table = &model.process_table;
    try model.overview.init(allocator);
    defer model.overview.deinit(allocator);
    defer image_cache.deinit(allocator);

    try app.run(ui.widget(&model), .{});
}
