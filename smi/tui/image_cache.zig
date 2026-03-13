const std = @import("std");
const vaxis = @import("vaxis");

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
    self.load(vx, allocator, writer, "logo", @embedFile("assets/logo.png"));
    self.load(vx, allocator, writer, "gpu_cuda", @embedFile("assets/nvidia.png"));
    self.load(vx, allocator, writer, "gpu_rocm", @embedFile("assets/amd.png"));
    self.load(vx, allocator, writer, "gpu_neuron", @embedFile("assets/neuron.png"));
    self.load(vx, allocator, writer, "gpu_tpu", @embedFile("assets/tpu.png"));
}

pub fn get(self: *const ImageCache, key: []const u8) ?vaxis.Image {
    return self.map.get(key);
}
