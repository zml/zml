const std = @import("std");
const vaxis = @import("vaxis");

var logos: []?vaxis.Image = &.{};

pub fn deinit(allocator: std.mem.Allocator) void {
    allocator.free(logos);
    logos = &.{};
}

/// The supplied order defines the logo indices. Reuse the allocator for deinit.
pub fn loadAll(
    terminal: *vaxis.Vaxis,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    assets: []const []const u8,
) void {
    deinit(allocator);
    logos = allocator.alloc(?vaxis.Image, assets.len) catch return;
    for (assets, logos) |png, *image| {
        image.* = terminal.loadImage(allocator, writer, .{ .mem = png }) catch null;
    }
}

pub fn get(logo: ?usize) ?vaxis.Image {
    const index = logo orelse return null;
    return if (index < logos.len) logos[index] else null;
}
