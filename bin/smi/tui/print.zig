const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("data.zig");
const ui = @import("lib/ui.zig");
const image_cache = @import("image_cache.zig");
const Overview = @import("pages/overview.zig");

fn silentResize(vx: *vaxis.Vaxis, allocator: std.mem.Allocator, winsize: vaxis.Winsize) !void {
    vx.screen.deinit(allocator);
    vx.screen = try vaxis.Screen.init(allocator, winsize);
    vx.screen.width_method = vx.caps.unicode;
    vx.screen_last.deinit(allocator);
    vx.screen_last = try vaxis.AllocatingScreen.init(allocator, winsize.cols, winsize.rows);
}

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
    try silentResize(&vx, allocator, ws);

    {
        const EventLoop = vaxis.Loop(vxfw.Event);
        var loop: EventLoop = .{ .tty = &tty, .vaxis = &vx, .io = io, .queue = .{ .io = io } };
        try loop.start();
        defer loop.stop();

        // queryTerminal sends cursor home (\x1b[H) for capability detection.
        // Save/restore cursor so we stay at the current position.
        const writer = tty.writer();
        try writer.writeAll("\x1b7"); // DECSC: save cursor
        try writer.flush();
        try vx.queryTerminal(writer, io, 1 * std.time.ns_per_s);
        try writer.writeAll("\x1b8"); // DECRC: restore cursor
        try writer.flush();
    }

    image_cache.global.loadAll(&vx, allocator, tty.writer());
    defer image_cache.global.deinit(allocator);

    vxfw.DrawContext.init(vx.screen.width_method);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const content_w: u16 = @min(@as(u16, @intCast(vx.screen.width)), Overview.host_line_width + 2);

    // Neuron util% is delta-based; wait for 2 poll cycles to get a real sample.
    if (state.targets.contains(.neuron)) {
        io.sleep(.fromSeconds(2), .awake) catch {};
    }

    var viewing_device: ?u8 = null;
    var overview = try Overview.init(allocator, state, null, &viewing_device);
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

    try silentResize(&vx, allocator, .{
        .rows = surface.size.height,
        .cols = ws.cols,
        .x_pixel = ws.x_pixel,
        .y_pixel = ws.y_pixel,
    });

    const win = vx.window();
    const root_win = win.child(.{
        .width = surface.size.width,
        .height = surface.size.height,
    });
    surface.render(root_win, ui.widget(&overview));

    try vx.prettyPrint(tty.writer());
}
