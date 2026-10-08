const std = @import("std");
const vaxis = @import("vaxis");
const Svg = @import("lib/svg.zig");

const ImageCache = @This();

pub var global: ImageCache = .{};

map: std.StringHashMapUnmanaged(vaxis.Image) = .empty,

pub fn deinit(self: *ImageCache, allocator: std.mem.Allocator) void {
    self.map.deinit(allocator);
}

pub fn load(
    self: *ImageCache,
    vx: *vaxis.Vaxis,
    allocator: std.mem.Allocator,
    writer: *std.Io.Writer,
    key: []const u8,
    data: []const u8,
) void {
    const image = vx.loadImage(allocator, undefined, writer, .{ .mem = data }) catch return;
    self.map.put(allocator, key, image) catch return;
}

pub fn loadAll(self: *ImageCache, vx: *vaxis.Vaxis, allocator: std.mem.Allocator, writer: *std.Io.Writer) void {
    self.loadSvg(vx, allocator, writer, "logo", @embedFile("assets/zml-logo-glow.svg"));
    self.load(vx, allocator, writer, "gpu_cuda", @embedFile("assets/nvidia.png"));
    self.load(vx, allocator, writer, "gpu_rocm", @embedFile("assets/amd.png"));
    self.load(vx, allocator, writer, "gpu_oneapi", @embedFile("assets/intel.png"));
    self.load(vx, allocator, writer, "gpu_neuron", @embedFile("assets/neuron.png"));
    self.load(vx, allocator, writer, "gpu_tpu", @embedFile("assets/tpu.png"));
}

fn loadSvg(self: *ImageCache, vx: *vaxis.Vaxis, allocator: std.mem.Allocator, writer: *std.Io.Writer, key: []const u8, data: []const u8) void {
    if (!vx.caps.kitty_graphics) return;
    const svg = Svg.render(data, 512) catch return;
    defer svg.deinit();

    const encoder = std.base64.standard.Encoder;
    const encoded = allocator.alloc(u8, encoder.calcSize(svg.pixels.len)) catch return;
    defer allocator.free(encoded);
    const image = vx.transmitPreEncodedImage(writer, encoder.encode(encoded, svg.pixels), svg.width, svg.height, .rgba) catch return;
    self.map.put(allocator, key, image) catch {
        vx.freeImage(writer, image.id);
    };
}

pub fn get(self: *const ImageCache, key: []const u8) ?vaxis.Image {
    return self.map.get(key);
}

test "renders the embedded SVG logo" {
    const svg = try Svg.render(@embedFile("assets/zml-logo-glow.svg"), 512);
    defer svg.deinit();
    try std.testing.expectEqual(@as(u16, 512), svg.width);
    try std.testing.expectEqual(@as(u16, 495), svg.height);
    try std.testing.expectEqual(@as(u8, 0), svg.pixels[3]);
    const center = (@as(usize, svg.height) / 2 * svg.width + svg.width / 2) * 4;
    try std.testing.expect(svg.pixels[center + 3] > 0);
}
