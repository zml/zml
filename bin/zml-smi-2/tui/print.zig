const std = @import("std");
const vaxis = @import("vaxis");
const vxfw = vaxis.vxfw;
const data = @import("data.zig");
const image_cache = @import("image_cache.zig");
const Overview = @import("pages/overview.zig");

pub fn run(allocator: std.mem.Allocator, io: std.Io, env_map: *std.process.Environ.Map, state: *data.SystemState) !void {
    var tty_buffer: [4096]u8 = undefined;
    var tty = try vaxis.Tty.init(io, &tty_buffer);
    defer tty.deinit();
    const writer = tty.writer();
    var discarding: std.Io.Writer.Discarding = .init(&.{});

    var vx = try vaxis.init(io, allocator, env_map, .{});
    defer {
        // Reset terminal modes without clearing screen content.
        vx.resetModes(writer) catch {};
        vx.deinit(allocator, &discarding.writer);
    }

    const ws = try tty.getWinsize();
    try vx.resize(allocator, &discarding.writer, ws);

    {
        var loop: vaxis.Loop(vxfw.Event) = .init(io, &tty, &vx);
        try loop.start();
        defer loop.stop();

        // queryTerminal sends cursor home (\x1b[H) for capability detection.
        // Save/restore cursor so we stay at the current position.
        try writer.writeAll("\x1b7"); // DECSC: save cursor
        try writer.flush();
        try vx.queryTerminal(writer, .fromSeconds(1));
        try writer.writeAll("\x1b8"); // DECRC: restore cursor
        try writer.flush();
    }

    image_cache.loadAll(&vx, allocator, writer, state.logos);
    defer image_cache.deinit(allocator);

    vxfw.DrawContext.init(vx.screen.width_method);

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const content_w: u16 = @min(@as(u16, @intCast(vx.screen.width)), Overview.host_line_width + 2);

    const draw_ctx: vxfw.DrawContext = .{
        .arena = arena.allocator(),
        .min = .{ .width = content_w },
        .max = .{ .width = content_w, .height = null },
        .cell_size = .{
            .width = if (vx.screen.width > 0) vx.screen.width_pix / vx.screen.width else 0,
            .height = if (vx.screen.height > 0) vx.screen.height_pix / vx.screen.height else 0,
        },
    };

    try state.capture(draw_ctx.arena);
    var overview: Overview = .{ .state = state };
    try overview.init(draw_ctx.arena);
    const surface = try overview.draw(draw_ctx);

    try vx.resize(allocator, &discarding.writer, .{
        .rows = surface.size.height,
        .cols = content_w,
        .x_pixel = ws.x_pixel,
        .y_pixel = ws.y_pixel,
    });

    surface.render(vx.window(), surface.widget);
    try vx.prettyPrint(writer);
}
