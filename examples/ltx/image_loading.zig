/// Image loading and preprocessing for image conditioning.
///
/// Loads a JPEG/PNG image from disk, resizes with center-crop to the target
/// resolution, normalizes to [-1, 1] bf16, and uploads to a zml.Buffer with
/// shape [1, 3, 1, H, W].
///
/// Uses stb_image for decoding and stb_image_resize2 for bilinear resize.
const std = @import("std");
const zml = @import("zml");

const c = @import("c");

const BFloat16 = zml.floats.BFloat16;

/// Load an image from `path`, resize+center-crop to (target_h, target_w),
/// normalize to [-1, 1] bf16, and return a device Buffer [1, 3, 1, H, W].
pub fn loadAndPreprocess(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *zml.Platform,
    sharding: zml.sharding.Sharding,
    path: []const u8,
    target_h: u32,
    target_w: u32,
) !zml.Buffer {
    // -- Load image from disk --
    var src_w: c_int = 0;
    var src_h: c_int = 0;
    var src_channels: c_int = 0;

    // stbi_load requires a null-terminated string.
    const c_path = try allocator.dupeZ(u8, path);
    defer allocator.free(c_path);

    const raw_ptr: ?[*]u8 = c.stbi_load(c_path.ptr, &src_w, &src_h, &src_channels, 3);
    if (raw_ptr == null) {
        std.log.err("Failed to load image: {s}", .{path});
        return error.ImageLoadFailed;
    }
    defer c.stbi_image_free(raw_ptr);

    const sw: u32 = @intCast(src_w);
    const sh: u32 = @intCast(src_h);
    const src_pixels: [*]const u8 = raw_ptr.?;
    std.log.info("  Loaded image {s}: {d}x{d} ({d} channels forced to 3)", .{ path, sw, sh, src_channels });

    // -- Resize with fill strategy (scale to cover, then center crop) --
    // Compute scale: fill = max of the two axis ratios.
    const scale_h: f64 = @as(f64, @floatFromInt(target_h)) / @as(f64, @floatFromInt(sh));
    const scale_w: f64 = @as(f64, @floatFromInt(target_w)) / @as(f64, @floatFromInt(sw));
    const scale: f64 = @max(scale_h, scale_w);

    const inter_h: u32 = @intFromFloat(@ceil(@as(f64, @floatFromInt(sh)) * scale));
    const inter_w: u32 = @intFromFloat(@ceil(@as(f64, @floatFromInt(sw)) * scale));

    // Resize to intermediate size (bilinear).
    const inter_buf = try allocator.alloc(u8, inter_h * inter_w * 3);
    defer allocator.free(inter_buf);

    const resize_ok = c.stbir_resize_uint8_linear(
        src_pixels,
        @intCast(sw),
        @intCast(sh),
        0, // stride_in_bytes = 0 means tightly packed
        inter_buf.ptr,
        @intCast(inter_w),
        @intCast(inter_h),
        0,
        c.STBIR_RGB,
    );
    if (resize_ok == null) {
        std.log.err("stbir_resize failed", .{});
        return error.ResizeFailed;
    }

    // Center crop from (inter_h, inter_w) to (target_h, target_w).
    const crop_y = (inter_h - target_h) / 2;
    const crop_x = (inter_w - target_w) / 2;

    // -- Convert to bf16 [1, 3, 1, H, W] layout, normalized to [-1, 1] --
    // Layout: batch=0, channel C, frame=0, row Y, col X.
    // Pixel index in the bf16 buffer: ((c * target_h) + y) * target_w + x
    const n_bf16 = 1 * 3 * 1 * target_h * target_w;
    const bf16_buf = try allocator.alloc(BFloat16, n_bf16);
    defer allocator.free(bf16_buf);

    for (0..target_h) |y| {
        const src_row = (crop_y + @as(u32, @intCast(y))) * inter_w * 3;
        for (0..target_w) |x| {
            const src_idx = src_row + (crop_x + @as(u32, @intCast(x))) * 3;
            for (0..3) |ch| {
                const pixel_u8 = inter_buf[src_idx + ch];
                // Normalize: pixel / 127.5 - 1.0  →  [-1, 1]
                const pixel_f32 = @as(f32, @floatFromInt(pixel_u8)) / 127.5 - 1.0;
                // bf16 buffer layout: [1, 3, 1, H, W] row-major
                // Index: ch * (target_h * target_w) + y * target_w + x
                const dst_idx = @as(u32, @intCast(ch)) * (target_h * target_w) +
                    @as(u32, @intCast(y)) * target_w +
                    @as(u32, @intCast(x));
                bf16_buf[dst_idx] = BFloat16.fromF32(pixel_f32);
            }
        }
    }

    std.log.info("  Preprocessed to [{d}, 3, 1, {d}, {d}] bf16", .{ 1, target_h, target_w });

    // -- Upload to device --
    const shape = zml.Shape.init(.{
        1,
        3,
        1,
        @as(i64, @intCast(target_h)),
        @as(i64, @intCast(target_w)),
    }, .bf16);
    return zml.Buffer.fromBytes(io, platform, shape, sharding, std.mem.sliceAsBytes(bf16_buf));
}
