const std = @import("std");
const stdx = @import("stdx");
const zml = @import("zml");

const QwenImage21Decoder3d = @import("./QwenImage21Decoder3d.zig");
const QwenImage21CausalConv3d = @import("./QwenImage21CausalConv3d.zig");

const CompileArgs = @import("../../compiler.zig").CompileArgs;

pub const AutoencoderKLQwenImage21 = @This();

decoder: QwenImage21Decoder3d,
post_quant_conv: QwenImage21CausalConv3d,

pub fn load(
    self: *const AutoencoderKLQwenImage21,
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    store: *const zml.io.TensorStore,
) !zml.Bufferized(AutoencoderKLQwenImage21) {
    var buffers = try zml.mem.bufferize(allocator, AutoencoderKLQwenImage21, self);
    errdefer unloadBuffers(&buffers);

    var loader: zml.io.Loader = try .init(allocator, platform, .{
        .dma_chunks = 32,
        .dma_chunk_size = 256 * zml.MiB,
        .parallelism = 16,
    });
    defer loader.deinit();

    try loader.load(io, AutoencoderKLQwenImage21, self, &buffers, store, platform.shardings.values(), .{});
    try loader.await(io);

    return buffers;
}

pub fn unloadBuffers(self: *zml.Bufferized(AutoencoderKLQwenImage21)) void {
    QwenImage21Decoder3d.unloadBuffers(&self.decoder);
    QwenImage21CausalConv3d.unloadBuffers(&self.post_quant_conv);
}

// The minimal tile height and width for spatial tiling to be used
const tile_sample_min_height: i64 = 256;
const tile_sample_min_width: i64 = 256;

// The minimal distance between two spatial tiles
const tile_sample_stride_height: i64 = 192;
const tile_sample_stride_width: i64 = 192;

const spatial_compression_ratio: i64 = 16;

const decoder_base_dim = 144;

const max_tiles = 16;
const Tiles = [max_tiles][max_tiles]zml.Tensor;

pub fn forward(self: AutoencoderKLQwenImage21, input_ids: zml.Tensor, args: CompileArgs) zml.Tensor {
    _ = args; // autofix
    const height = input_ids.dim(.kx);
    const width = input_ids.dim(.ky);

    const tile_latent_min_height = tile_sample_min_height / spatial_compression_ratio;
    const tile_latent_min_width = tile_sample_min_width / spatial_compression_ratio;
    const tile_latent_stride_height = tile_sample_stride_height / spatial_compression_ratio;
    const tile_latent_stride_width = tile_sample_stride_width / spatial_compression_ratio;

    // range(0, size, stride)
    const rows_count: usize = @intCast(@divTrunc(height + tile_latent_stride_height - 1, tile_latent_stride_height));
    const per_row: usize = @intCast(@divTrunc(width + tile_latent_stride_width - 1, tile_latent_stride_width));

    // 11 for a 2048x2048 image
    stdx.debug.assert(rows_count <= max_tiles, "Found that rows_count ({}) is greater than {}", .{ rows_count, max_tiles });
    stdx.debug.assert(per_row <= max_tiles, "Found that per_row ({}) is greater than {}", .{ per_row, max_tiles });
    var tiles: Tiles = undefined;

    for (0..rows_count) |x| {
        const i: i64 = @as(i64, @intCast(x)) * tile_latent_stride_height;
        for (0..per_row) |y| {
            const j: i64 = @as(i64, @intCast(y)) * tile_latent_stride_width;
            // Like PyTorch's slicing, tiles on the bottom/right edges are cut short
            const tile = input_ids
                .slice(.kx, .{ .start = i, .end = @min(i + tile_latent_min_height, height) })
                .slice(.ky, .{ .start = j, .end = @min(j + tile_latent_min_width, width) });
            // const post_quant = self.post_quant_conv.forward(tile);
            tiles[x][y] = self.decoder.forward(tile);
        }
    }

    return stitch(&tiles, rows_count, per_row, .{
        .blend_height = tile_sample_min_height - tile_sample_stride_height,
        .blend_width = tile_sample_min_width - tile_sample_stride_width,
        .stride_height = tile_sample_stride_height,
        .stride_width = tile_sample_stride_width,
        .sample_height = height * spatial_compression_ratio,
        .sample_width = width * spatial_compression_ratio,
    });
    // return dec.clamp(
    //     zml.Tensor.scalar(-1, dec.dtype()).broad(dec.shape()),
    //     zml.Tensor.scalar(1, dec.dtype()).broad(dec.shape()),
    // );
}

const StitchArgs = struct {
    blend_height: i64,
    blend_width: i64,
    stride_height: i64,
    stride_width: i64,
    sample_height: i64,
    sample_width: i64,
};

/// Second half of `tiled_decode`: blends every decoded tile with its upper and left neighbours,
/// keeps the first `stride` rows/columns of each, and stitches them back into one image.
/// `tiles[i][j]` is {bs, c, t, kx, ky}.
fn stitch(tiles: *Tiles, rows_count: usize, per_row: usize, args: StitchArgs) zml.Tensor {
    var result_rows: [max_tiles]zml.Tensor = undefined;
    for (0..rows_count) |i| {
        var result_row: [max_tiles]zml.Tensor = undefined;
        for (0..per_row) |j| {
            var tile = tiles[i][j];
            // blend the above tile and the left tile to the current tile
            if (i > 0) {
                tile = blend_v(tiles[i - 1][j], tile, args.blend_height);
            }
            if (j > 0) {
                tile = blend_h(tiles[i][j - 1], tile, args.blend_width);
            }
            // PyTorch blends in place, so the neighbours of the next tiles are the blended ones
            tiles[i][j] = tile;

            // tile[:, :, :, :stride_height, :stride_width]
            result_row[j] = crop(crop(tile, .kx, args.stride_height), .ky, args.stride_width);
        }
        result_rows[i] = zml.Tensor.concatenate(result_row[0..per_row], .ky);
    }

    // torch.cat(result_rows, dim=3)[:, :, :, :sample_height, :sample_width]
    const dec = zml.Tensor.concatenate(result_rows[0..rows_count], .kx);
    return crop(crop(dec, .kx, args.sample_height), .ky, args.sample_width);
}

/// x[..., :len] along `axis`, like PyTorch slicing it doesn't fail when `len` is larger than the axis.
fn crop(x: zml.Tensor, comptime axis: @EnumLiteral(), len: i64) zml.Tensor {
    return x.slice(axis, .{ .start = 0, .end = @min(len, x.dim(axis)) });
}

/// Cross-fades the first `extent` rows of `b` with the last `extent` rows of `a` (vertical seam).
fn blend_v(a: zml.Tensor, b: zml.Tensor, extent: i64) zml.Tensor {
    return blend(a, b, extent, .kx);
}

/// Cross-fades the first `extent` columns of `b` with the last `extent` columns of `a` (horizontal seam).
fn blend_h(a: zml.Tensor, b: zml.Tensor, extent: i64) zml.Tensor {
    return blend(a, b, extent, .ky);
}

// For i in [0, n): b[i] = a[-n + i] * (1 - i / n) + b[i] * (i / n), the rest of `b` is untouched.
// Same rounding points as PyTorch on a half-precision tensor: each product is computed in f32 and
// rounded to the tensor dtype, then the sum is rounded again.
fn blend(a: zml.Tensor, b: zml.Tensor, extent: i64, comptime axis: @EnumLiteral()) zml.Tensor {
    const n = @min(a.dim(axis), b.dim(axis), extent);
    if (n <= 0) {
        return b;
    }

    const a_tail = a.slice(axis, .{ .start = a.dim(axis) - n });
    const b_head = b.slice(axis, .{ .start = 0, .end = n });

    // weight_b[i] = i / n and weight_a[i] = (n - i) / n, so weight_a is 1 - weight_b without a second rounding.
    const i = zml.Tensor.arange(.{ .end = n }, .f32).withTags(.{axis});
    const weight_b = i.divByConst(n).broad(b_head.shape().withDtype(.f32));
    const weight_a = i.scale(-1).addConstant(n).divByConst(n).broad(b_head.shape().withDtype(.f32));

    const blended = a_tail.convert(.f32).mul(weight_a).convert(b.dtype())
        .add(b_head.convert(.f32).mul(weight_b).convert(b.dtype()));

    if (n == b.dim(axis)) {
        return blended;
    }
    return zml.Tensor.concatenate(&.{ blended, b.slice(axis, .{ .start = n }) }, axis);
}

test "blend_v and blend_h match the PyTorch reference" {
    const platform = zml.testing.env();

    const Local = struct {
        fn blendV(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
            return blend_v(a.withTags(.{ .bs, .c, .t, .kx, .ky }), b.withTags(.{ .bs, .c, .t, .kx, .ky }), 3);
        }
        fn blendH(a: zml.Tensor, b: zml.Tensor) zml.Tensor {
            return blend_h(a.withTags(.{ .bs, .c, .t, .kx, .ky }), b.withTags(.{ .bs, .c, .t, .kx, .ky }), 3);
        }
    };

    // a, b: [1, 1, 1, 4, 5]. The extent (3) is smaller than both sizes, so the untouched tail is exercised too.
    var a_data: [20]f32 = undefined;
    var b_data: [20]f32 = undefined;
    for (&a_data, &b_data, 0..) |*x, *y, k| {
        x.* = @floatFromInt(k + 1);
        y.* = @floatFromInt(100 + 2 * k);
    }
    const shape = zml.Shape.init(.{ 1, 1, 1, 4, 5 }, .f32);
    const a: zml.Tensor = .fromShape(shape);
    const b: zml.Tensor = .fromShape(shape);

    // Reference computed with the PyTorch loops: rows 0..2 of b blended with rows 1..3 of a (V), columns 0..2 with columns 2..4 (H).
    var expected_v = b_data;
    var expected_h = b_data;
    for (0..3) |y| {
        const w: f32 = @as(f32, @floatFromInt(y)) / 3;
        for (0..5) |x| expected_v[y * 5 + x] = a_data[(1 + y) * 5 + x] * (1 - w) + b_data[y * 5 + x] * w;
    }
    for (0..3) |x| {
        const w: f32 = @as(f32, @floatFromInt(x)) / 3;
        for (0..4) |y| expected_h[y * 5 + x] = a_data[y * 5 + 2 + x] * (1 - w) + b_data[y * 5 + x] * w;
    }

    inline for (.{ .{ Local.blendV, expected_v }, .{ Local.blendH, expected_h } }) |case| {
        const exe = try zml.module.compile(std.testing.allocator, std.testing.io, case[0], .{ a, b }, platform, .{});
        defer exe.deinit();

        var a_buffer = try zml.Buffer.fromBytes(std.testing.io, platform, shape, .replicated, std.mem.sliceAsBytes(&a_data));
        defer a_buffer.deinit();
        var b_buffer = try zml.Buffer.fromBytes(std.testing.io, platform, shape, .replicated, std.mem.sliceAsBytes(&b_data));
        defer b_buffer.deinit();

        const output = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, case[0], .{ a_buffer, b_buffer });
        try zml.testing.expectClose(std.testing.io, zml.Slice.init(shape, std.mem.sliceAsBytes(&case[1])), output, .{});
    }
}

test "stitch matches the PyTorch reference" {
    const platform = zml.testing.env();

    const Local = struct {
        // Row-major values, same formula as the PyTorch reference
        fn fill(comptime n: usize, t: usize) [n]f32 {
            var out: [n]f32 = undefined;
            for (&out, 0..) |*v, p| v.* = @as(f32, @floatFromInt((p * 7 + t * 13) % 17)) - 8.0;
            return out;
        }

        fn run(t00: zml.Tensor, t01: zml.Tensor, t10: zml.Tensor, t11: zml.Tensor) zml.Tensor {
            var tiles: Tiles = undefined;
            tiles[0][0] = t00.withTags(.{ .bs, .c, .t, .kx, .ky });
            tiles[0][1] = t01.withTags(.{ .bs, .c, .t, .kx, .ky });
            tiles[1][0] = t10.withTags(.{ .bs, .c, .t, .kx, .ky });
            tiles[1][1] = t11.withTags(.{ .bs, .c, .t, .kx, .ky });
            return stitch(&tiles, 2, 2, .{
                .blend_height = 2,
                .blend_width = 2,
                .stride_height = 3,
                .stride_width = 3,
                .sample_height = 5,
                .sample_width = 5,
            });
        }
    };

    // Bottom/right tiles are smaller than the stride-sized crop, and the output is cropped to 5x5.
    const shapes = [4]zml.Shape{
        zml.Shape.init(.{ 1, 1, 1, 5, 5 }, .f32),
        zml.Shape.init(.{ 1, 1, 1, 5, 3 }, .f32),
        zml.Shape.init(.{ 1, 1, 1, 3, 5 }, .f32),
        zml.Shape.init(.{ 1, 1, 1, 3, 3 }, .f32),
    };
    const data = .{ Local.fill(25, 0), Local.fill(15, 1), Local.fill(15, 2), Local.fill(9, 3) };

    const exe = try zml.module.compile(
        std.testing.allocator,
        std.testing.io,
        Local.run,
        .{ zml.Tensor.fromShape(shapes[0]), zml.Tensor.fromShape(shapes[1]), zml.Tensor.fromShape(shapes[2]), zml.Tensor.fromShape(shapes[3]) },
        platform,
        .{},
    );
    defer exe.deinit();

    var buffers: [4]zml.Buffer = undefined;
    inline for (&buffers, shapes, data, 0..) |*buffer, shape, values, k| {
        _ = k;
        buffer.* = try zml.Buffer.fromBytes(std.testing.io, platform, shape, .replicated, std.mem.sliceAsBytes(&values));
    }
    defer for (&buffers) |*buffer| buffer.deinit();

    const output = try zml.testing.autoCall(std.testing.allocator, std.testing.io, &exe, Local.run, .{ buffers[0], buffers[1], buffers[2], buffers[3] });

    // Computed with the diffusers loops (tiled_decode's blend_v / blend_h / crop / cat) on the same inputs
    const expected = [25]f32{ -8, -1, 6, -4, -1, -7, 0, 7, -3, 1.5, -6, 1, 8, -2, 4, -5, 2, -8, -1, 6.25, -1, -2.5, -4, 3, 2.875 };
    try zml.testing.expectClose(std.testing.io, zml.Slice.init(zml.Shape.init(.{ 1, 1, 1, 5, 5 }, .f32), std.mem.sliceAsBytes(&expected)), output, .{});
}
