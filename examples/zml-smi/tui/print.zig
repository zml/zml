const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("data.zig");
const image_cache = @import("image_cache.zig");
const Overview = @import("pages/overview.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, state: *data.SystemState) !void {
    var tty = try vaxis.Tty.init(io);
    defer tty.deinit();

    var vx = try vaxis.init(allocator, .{});
    defer {
        // Reset terminal modes without clearing screen content.
        vx.resetModes(tty.writer()) catch {};
        var buf: [256]u8 = undefined;
        var discarding: std.Io.Writer.Discarding = .init(&buf);
        vx.deinit(allocator, &discarding.writer);
    }

    const ws = try vaxis.Tty.getWinsize(tty.fd);
    try vx.resize(allocator, tty.writer(), ws);

    {
        const EventLoop = vaxis.Loop(vxfw.Event);
        var loop: EventLoop = .{ .tty = &tty, .vaxis = &vx, .io = io, .queue = .{ .io = io } };
        try loop.start();
        defer loop.stop();

        try vx.queryTerminal(tty.writer(), io, 1 * std.time.ns_per_s);
    }

    image_cache.global.loadAll(&vx, allocator, tty.writer());
    defer image_cache.global.deinit(allocator);

    vxfw.DrawContext.init(vx.screen.width_method);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const content_w: u16 = @intCast(vx.screen.width);

    var viewing_device: ?u8 = null;
    var overview: Overview = .{
        .state = state,
        .viewing_device = &viewing_device,
    };
    try overview.init(allocator);
    defer overview.deinit(allocator);

    const draw_ctx: vxfw.DrawContext = .{
        .arena = arena.allocator(),
        .min = .{ .width = content_w },
        .max = .{ .width = content_w, .height = null },
        .cell_size = .{
            .width = if (vx.screen.width > 0) vx.screen.width_pix / vx.screen.width else 0,
            .height = if (vx.screen.height > 0) vx.screen.height_pix / vx.screen.height else 0,
        },
    };

    const surface = try overview.draw(draw_ctx);

    try vx.resize(allocator, tty.writer(), .{
        .rows = surface.size.height,
        .cols = ws.cols,
        .x_pixel = ws.x_pixel,
        .y_pixel = ws.y_pixel,
    });

    const win = vx.window();
    win.clear();
    const root_win = win.child(.{
        .width = surface.size.width,
        .height = surface.size.height,
    });
    surface.render(root_win, overview.widget());

    // Scroll terminal to make room for content
    const writer = tty.writer();
    for (0..surface.size.height) |_| {
        try writer.writeByte('\n');
    }
    try writer.flush();

    vx.state.cursor.row = surface.size.height -| 1;

    // Force full render
    vx.queueRefresh();
    try vx.render(tty.writer());
}
