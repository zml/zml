const std = @import("std");

const Svg = @This();

handle: *anyopaque,
pixels: []const u8,
width: u16,
height: u16,

extern fn zml_smi_render_svg(data: [*]const u8, length: usize, width: c_int, height: *c_int, pixels: *[*]const u8) ?*anyopaque;
extern fn zml_smi_free_svg(bitmap: *anyopaque) void;

pub fn render(data: []const u8, width: u16) error{InvalidSvg}!Svg {
    if (width == 0) return error.InvalidSvg;
    var height: c_int = undefined;
    var pixels: [*]const u8 = undefined;
    const handle = zml_smi_render_svg(data.ptr, data.len, width, &height, &pixels) orelse return error.InvalidSvg;
    errdefer zml_smi_free_svg(handle);
    const image_height = std.math.cast(u16, height) orelse return error.InvalidSvg;
    return .{
        .handle = handle,
        .pixels = pixels[0 .. @as(usize, width) * @as(usize, @intCast(height)) * 4],
        .width = width,
        .height = image_height,
    };
}

pub fn deinit(self: Svg) void {
    zml_smi_free_svg(self.handle);
}

test "renders SVG as straight RGBA with preserved aspect ratio and transparency" {
    const svg = try render(
        \\<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 1">
        \\  <rect width="1" height="1" fill="#ff0000" fill-opacity="0.5"/>
        \\</svg>
    , 4);
    defer svg.deinit();
    try std.testing.expectEqual(@as(u16, 4), svg.width);
    try std.testing.expectEqual(@as(u16, 2), svg.height);
    try std.testing.expectEqual(@as(usize, 32), svg.pixels.len);
    try std.testing.expectEqualSlices(u8, &.{ 255, 0, 0 }, svg.pixels[0..3]);
    try std.testing.expect(svg.pixels[3] >= 127 and svg.pixels[3] <= 128);
    try std.testing.expectEqual(@as(u8, 0), svg.pixels[15]);
}

test "rejects invalid SVG and zero render width" {
    try std.testing.expectError(error.InvalidSvg, render("not an SVG", 4));
    try std.testing.expectError(error.InvalidSvg, render("<svg/>", 0));
}
