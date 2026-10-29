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

    try loader.load(io, AutoencoderKLQwenImage21, self, &buffers, store, .{});
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

pub fn forward(self: AutoencoderKLQwenImage21, input_ids: zml.Tensor) zml.Tensor {
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
    // Like PyTorch's slicing, tiles on the bottom/right edges are cut short, so the tiles only come in
    // a handful of shapes. Each group of same-shaped tiles is decoded by a `stablehlo.while` that traces
    // the decoder once, instead of unrolling one decoder per tile (121 for a 2048x2048 image).
    var row_runs: [max_tiles]TileRun = undefined;
    var col_runs: [max_tiles]TileRun = undefined;
    const row_runs_count = tileRuns(height, tile_latent_stride_height, tile_latent_min_height, &row_runs);
    const col_runs_count = tileRuns(width, tile_latent_stride_width, tile_latent_min_width, &col_runs);

    var tiles: Tiles = undefined;
    for (row_runs[0..row_runs_count]) |rows| {
        for (col_runs[0..col_runs_count]) |cols| {
            const decoded = self.decodeTiles(input_ids, rows, cols, .{
                .height = tile_latent_stride_height,
                .width = tile_latent_stride_width,
            });
            for (0..rows.count) |r| {
                for (0..cols.count) |c| {
                    tiles[rows.first + r][cols.first + c] = decoded.slice(.tile, .single(@intCast(r * cols.count + c)));
                }
            }
        }
    }

    const dec = stitch(&tiles, rows_count, per_row, .{
        .blend_height = tile_sample_min_height - tile_sample_stride_height,
        .blend_width = tile_sample_min_width - tile_sample_stride_width,
        .stride_height = tile_sample_stride_height,
        .stride_width = tile_sample_stride_width,
        .sample_height = height * spatial_compression_ratio,
        .sample_width = width * spatial_compression_ratio,
    });
    return dec.clamp(
        zml.Tensor.scalar(-1, dec.dtype()).broad(dec.shape()),
        zml.Tensor.scalar(1, dec.dtype()).broad(dec.shape()),
    );
}

/// Consecutive tiles along one axis sharing the same size: tiles `first..first + count` all span `size` latents.
const TileRun = struct {
    first: usize,
    count: usize,
    size: i64,
};

/// Splits the tiles starting at range(0, dim, stride) into runs of tiles of the same size.
/// Tile sizes are min(tile_size, dim - start), so they never increase and same-sized tiles are contiguous.
fn tileRuns(dim: i64, stride: i64, tile_size: i64, runs: *[max_tiles]TileRun) usize {
    var runs_count: usize = 0;
    var start: i64 = 0;
    var index: usize = 0;
    while (start < dim) : ({
        start += stride;
        index += 1;
    }) {
        const size = @min(tile_size, dim - start);
        if (runs_count > 0 and runs[runs_count - 1].size == size) {
            runs[runs_count - 1].count += 1;
        } else {
            runs[runs_count] = .{ .first = index, .count = 1, .size = size };
            runs_count += 1;
        }
    }
    return runs_count;
}

/// Decodes the `rows.count * cols.count` latent tiles of a group with a single traced decoder.
/// Returns {tile, bs, c, t, kx, ky}, tiles in row-major order.
fn decodeTiles(self: AutoencoderKLQwenImage21, latents: zml.Tensor, rows: TileRun, cols: TileRun, stride: struct { height: i64, width: i64 }) zml.Tensor {
    const tiles_count: i64 = @intCast(rows.count * cols.count);
    const out_channels = self.decoder.conv_out.conv.weight.dim(.bs);
    const decoded_shape = latents.shape()
        .setDim(.c, out_channels)
        .setDim(.kx, rows.size * spatial_compression_ratio)
        .setDim(.ky, cols.size * spatial_compression_ratio)
        .insertTag(0, tiles_count, .tile);

    const result = zml.ops.@"while"(DecodeTiles, .{
        .model = self,
        .latents = latents,
        .rows = rows,
        .cols = cols,
        .stride_height = stride.height,
        .stride_width = stride.width,
        .tiles_count = tiles_count,
    }, .{
        .k = .scalar(0, .i32),
        .decoded = .zeroes(decoded_shape),
    });
    return result.decoded;
}

const DecodeTiles = struct {
    model: AutoencoderKLQwenImage21,
    latents: zml.Tensor,
    rows: TileRun,
    cols: TileRun,
    stride_height: i64,
    stride_width: i64,
    tiles_count: i64,

    pub const State = struct {
        k: zml.Tensor,
        decoded: zml.Tensor,
    };

    pub fn cond(self: DecodeTiles, state: State) zml.Tensor {
        return state.k.cmp(.LT, zml.Tensor.scalar(self.tiles_count, .i32));
    }

    pub fn body(self: DecodeTiles, state: State) State {
        const per_row: i64 = @intCast(self.cols.count);
        const row = state.k.divByConst(per_row).addConstant(self.rows.first);
        const col = state.k.remainderConst(per_row).addConstant(self.cols.first);
        const tile = self.latents
            .slice(.kx, .dyn(row.scale(self.stride_height), self.rows.size))
            .slice(.ky, .dyn(col.scale(self.stride_width), self.cols.size));
        const decoded_tile = self.model.decoder.forward(self.model.post_quant_conv.forward(tile));
        return .{
            .k = state.k.addConstant(1),
            .decoded = state.decoded.dynamicUpdateSlice(.{ .tile = state.k }, decoded_tile),
        };
    }
};

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

test "tileRuns groups same-sized tiles" {
    var runs: [max_tiles]TileRun = undefined;

    // 2048x2048 image: starts 0, 12, .., 120, only the last tile is cut short (128 - 120 = 8).
    try std.testing.expectEqualSlices(TileRun, &.{
        .{ .first = 0, .count = 10, .size = 16 },
        .{ .first = 10, .count = 1, .size = 8 },
    }, runs[0..tileRuns(128, 12, 16, &runs)]);

    // Two edge tiles are cut short: starts 108 and 120 give 123 - 108 = 15 and 123 - 120 = 3.
    try std.testing.expectEqualSlices(TileRun, &.{
        .{ .first = 0, .count = 9, .size = 16 },
        .{ .first = 9, .count = 1, .size = 15 },
        .{ .first = 10, .count = 1, .size = 3 },
    }, runs[0..tileRuns(123, 12, 16, &runs)]);

    // Single tile, smaller than the tile size.
    try std.testing.expectEqualSlices(TileRun, &.{
        .{ .first = 0, .count = 1, .size = 10 },
    }, runs[0..tileRuns(10, 12, 16, &runs)]);
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

        var a_buffer = try zml.Buffer.fromBytes(std.testing.io, platform, shape, std.mem.sliceAsBytes(&a_data));
        defer a_buffer.deinit();
        var b_buffer = try zml.Buffer.fromBytes(std.testing.io, platform, shape, std.mem.sliceAsBytes(&b_data));
        defer b_buffer.deinit();

        var output = try exe.eval(std.testing.allocator, std.testing.io, .{ a_buffer, b_buffer });
        defer output.deinit();
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
        buffer.* = try zml.Buffer.fromBytes(std.testing.io, platform, shape, std.mem.sliceAsBytes(&values));
    }
    defer for (&buffers) |*buffer| buffer.deinit();

    var output = try exe.eval(std.testing.allocator, std.testing.io, .{ buffers[0], buffers[1], buffers[2], buffers[3] });
    defer output.deinit();

    // Computed with the diffusers loops (tiled_decode's blend_v / blend_h / crop / cat) on the same inputs
    const expected = [25]f32{ -8, -1, 6, -4, -1, -7, 0, 7, -3, 1.5, -6, 1, 8, -2, 4, -5, 2, -8, -1, 6.25, -1, -2.5, -4, 3, 2.875 };
    try zml.testing.expectClose(std.testing.io, zml.Slice.init(zml.Shape.init(.{ 1, 1, 1, 5, 5 }, .f32), std.mem.sliceAsBytes(&expected)), output, .{});
}
