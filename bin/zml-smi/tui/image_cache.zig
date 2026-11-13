const std = @import("std");
const vaxis = @import("vaxis");
const smi = @import("zml-smi");

const ImageCache = @This();

pub var global: ImageCache = .{};

const logos: std.EnumMap(smi.Target, []const u8) = .init(.{
    .cuda = @embedFile("assets/nvidia.png"),
    .rocm = @embedFile("assets/amd.png"),
    .oneapi = @embedFile("assets/intel.png"),
    .neuron = @embedFile("assets/neuron.png"),
    .tpu = @embedFile("assets/tpu.png"),
});

zml: ?vaxis.Image = null,
targets: std.EnumMap(smi.Target, vaxis.Image) = .{},

pub fn deinit(self: *ImageCache) void {
    self.* = .{};
}

pub fn loadAll(self: *ImageCache, vx: *vaxis.Vaxis, allocator: std.mem.Allocator, writer: *std.Io.Writer) void {
    self.zml = vx.loadImage(allocator, writer, .{ .mem = @embedFile("assets/logo.png") }) catch null;
    for (std.enums.values(smi.Target)) |target| {
        const png = logos.get(target) orelse continue;
        self.targets.put(target, vx.loadImage(allocator, writer, .{ .mem = png }) catch continue);
    }
}

pub fn get(self: *const ImageCache, target: smi.Target) ?vaxis.Image {
    return self.targets.get(target);
}
