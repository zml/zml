const std = @import("std");

const zml = @import("zml");

const config = @import("config.zig");
const ops = @import("ops.zig");

const log = std.log.scoped(.minimax_h3_visual_enc);

const spatial_downsample = [_]i64{ 2, 2, 2, 2, 1, 1 };
const temporal_downsample = [_]i64{ 1, 2, 2, 1, 1, 1 };
const norm_groups: i64 = 32;
const norm_eps: f32 = 1e-6;

fn convWeight(store: zml.io.TensorStore.View, name: []const u8) zml.Tensor {
    return store.createTensor(name, .{ .co, .ci, .kt, .kh, .kw }, .replicated);
}

fn reflectPadBoth(x: zml.Tensor, axis: anytype, pad: i64) zml.Tensor {
    if (pad <= 0) return x;
    const n = x.dim(axis);
    if (n <= 1) {
        const first = x.slice(axis, .{ .start = 0, .end = 1 });
        const extra = first.broad(first.shape().setDim(axis, pad));
        return zml.Tensor.concatenate(&.{ extra, x, extra }, axis);
    }
    const left = x.slice(axis, .{ .start = 1, .end = 1 + pad }).reverse(.{axis});
    const right = x.slice(axis, .{ .start = n - 1 - pad, .end = n - 1 }).reverse(.{axis});
    return zml.Tensor.concatenate(&.{ left, x, right }, axis);
}

fn reflectPadHigh(x: zml.Tensor, axis: anytype, pad: i64) zml.Tensor {
    if (pad <= 0) return x;
    const n = x.dim(axis);
    if (n <= 1) {
        const last = x.slice(axis, .{ .start = n - 1, .end = n });
        return zml.Tensor.concatenate(&.{ x, last.broad(last.shape().setDim(axis, pad)) }, axis);
    }
    const tail = x.slice(axis, .{ .start = n - 1 - pad, .end = n - 1 }).reverse(.{axis});
    return zml.Tensor.concatenate(&.{ x, tail }, axis);
}

fn causalPadT(x: zml.Tensor, pad: i64) zml.Tensor {
    if (pad <= 0) return x;
    const zeros = zml.Tensor.zeroes(x.shape().setDim(.t, pad));
    return zml.Tensor.concatenate(&.{ zeros, x }, .t);
}

fn isolatedGroupNorm(x: zml.Tensor, weight: zml.Tensor, bias: zml.Tensor, groups: i64, eps: f32) zml.Tensor {
    const xf = x.convert(.f32).withPartialTags(.{ .b, .c, .t, .h, .w });
    const b = xf.dim(.b);
    const c = xf.dim(.c);
    const t = xf.dim(.t);
    const h = xf.dim(.h);
    const w = xf.dim(.w);
    const cg = @divExact(c, groups);
    var y = xf.transpose(.{ .b, .t, .c, .h, .w });
    y = y.merge(.{ .bt = .{ .b, .t } }).splitAxis(.c, .{ .g = groups, .cg = cg });
    y = y.merge(.{ .n = .{ .cg, .h, .w } });
    const mean = y.mean(.n);
    const centered = y.sub(mean.broad(y.shape()));
    const variance = centered.mul(centered).mean(.n);
    y = centered.mul(variance.addConstant(eps).rsqrt().broad(y.shape()));
    y = y.splitAxis(.n, .{ .cg = cg, .h = h, .w = w });
    y = y.merge(.{ .c = .{ .g, .cg } }).splitAxis(.bt, .{ .b = b, .t = t });
    y = y.transpose(.{ .b, .c, .t, .h, .w });
    const scale = weight.convert(.f32).withTags(.{.c}).broad(y.shape());
    const shift = bias.convert(.f32).withTags(.{.c}).broad(y.shape());
    return y.mul(scale).add(shift).convert(x.dtype());
}

const CausalConv3d = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor,
    stride_t: i64,
    stride_hw: i64,
    spatial_pad: i64,
    temporal_pad: i64,

    pub fn init(store: zml.io.TensorStore.View, stride_t: i64, stride_hw: i64, spatial_pad: i64, temporal_pad: i64) CausalConv3d {
        return .{
            .weight = convWeight(store, "weight"),
            .bias = store.maybeCreateTensor("bias", .{.co}, .replicated),
            .stride_t = stride_t,
            .stride_hw = stride_hw,
            .spatial_pad = spatial_pad,
            .temporal_pad = temporal_pad,
        };
    }

    pub fn forward(self: CausalConv3d, x: zml.Tensor) zml.Tensor {
        var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t, .h, .w });
        y = reflectPadBoth(y, .h, self.spatial_pad);
        y = reflectPadBoth(y, .w, self.spatial_pad);
        y = causalPadT(y, self.temporal_pad);
        const w = self.weight.convert(.f32).withPartialTags(.{ .co, .ci, .kt, .kh, .kw });
        y = y.conv3d(w, .{
            .window_strides = &.{ self.stride_t, self.stride_hw, self.stride_hw },
        });
        if (self.bias) |bias| y = y.add(bias.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
        return y.convert(x.dtype());
    }
};

const GroupNorm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) GroupNorm {
        return .{
            .weight = store.createTensor("weight", .{.c}, .replicated),
            .bias = store.createTensor("bias", .{.c}, .replicated),
        };
    }

    pub fn forward(self: GroupNorm, x: zml.Tensor) zml.Tensor {
        return isolatedGroupNorm(x, self.weight, self.bias, norm_groups, norm_eps);
    }
};

const Resnet = struct {
    norm1: GroupNorm,
    conv1: CausalConv3d,
    norm2: GroupNorm,
    conv2: CausalConv3d,
    shortcut: ?CausalConv3d,

    pub fn init(store: zml.io.TensorStore.View) Resnet {
        return .{
            .norm1 = .init(store.withPrefix("norm1")),
            .conv1 = .init(store.withPrefix("conv1"), 1, 1, 1, 2),
            .norm2 = .init(store.withPrefix("norm2")),
            .conv2 = .init(store.withPrefix("conv2"), 1, 1, 1, 2),
            .shortcut = if (store.hasKey("conv_shortcut.weight"))
                .init(store.withPrefix("conv_shortcut"), 1, 1, 0, 0)
            else
                null,
        };
    }

    pub fn forward(self: Resnet, x: zml.Tensor) zml.Tensor {
        var h = self.conv1.forward(self.norm1.forward(x).silu());
        h = self.conv2.forward(self.norm2.forward(h).silu());
        var residual = x;
        if (self.shortcut) |s| residual = s.forward(residual);
        return residual.add(h);
    }
};

const Downsample = struct {
    conv: CausalConv3d,
    spatial_stride: i64,

    pub fn init(store: zml.io.TensorStore.View, temporal_stride: i64, spatial_stride: i64) Downsample {
        const inner = store.withPrefix("conv");
        return .{
            .conv = .init(inner, temporal_stride, spatial_stride, 0, 2),
            .spatial_stride = spatial_stride,
        };
    }

    pub fn forward(self: Downsample, x: zml.Tensor) zml.Tensor {
        var y = x.withPartialTags(.{ .b, .c, .t, .h, .w });
        if (self.spatial_stride == 2) {
            y = reflectPadHigh(y, .h, 1);
            y = reflectPadHigh(y, .w, 1);
        }
        return self.conv.forward(y);
    }
};

const DownBlock = struct {
    block0: Resnet,
    block1: Resnet,
    downsample: ?Downsample,

    pub fn init(store: zml.io.TensorStore.View, temporal_factor: i64, spatial_factor: i64) DownBlock {
        const blocks = store.withPrefix("resnets");
        return .{
            .block0 = .init(blocks.withLayer(0)),
            .block1 = .init(blocks.withLayer(1)),
            .downsample = if (temporal_factor * spatial_factor > 1)
                .init(store.withPrefix("downsamplers").withLayer(0), temporal_factor, spatial_factor)
            else
                null,
        };
    }

    pub fn forward(self: DownBlock, x: zml.Tensor) zml.Tensor {
        var h = self.block1.forward(self.block0.forward(x));
        if (self.downsample) |d| h = d.forward(h);
        return h;
    }
};

fn encoderView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    return store.withPrefix("encoder");
}

pub const Model = struct {
    conv_in: CausalConv3d,
    downs: [6]DownBlock,
    norm_out: GroupNorm,
    conv_out: CausalConv3d,
    quant_conv: CausalConv3d,

    pub fn init(store: zml.io.TensorStore.View) Model {
        const root = store;
        const enc = encoderView(root);
        const down_root = enc.withPrefix("down_blocks");
        var downs: [6]DownBlock = undefined;
        for (&downs, 0..) |*block, i| {
            block.* = .init(down_root.withLayer(i), temporal_downsample[i], spatial_downsample[i]);
        }
        const quant = root.withPrefix("quant_conv");
        return .{
            .conv_in = .init(enc.withPrefix("conv_in"), 1, 1, 1, 2),
            .downs = downs,
            .norm_out = .init(enc.withPrefix("norm_out")),
            .conv_out = .init(enc.withPrefix("conv_out"), 1, 1, 1, 2),
            .quant_conv = .init(quant, 1, 1, 0, 0),
        };
    }
};

pub const EncodeInput = struct {
    model: Model,
    pixels: zml.Tensor,
};

pub const EncodeOutput = struct {
    moments: zml.Tensor,
};

pub fn encode(input: EncodeInput) EncodeOutput {
    const self = input.model;
    var h = self.conv_in.forward(input.pixels);
    for (self.downs) |block| h = block.forward(h);
    h = self.conv_out.forward(self.norm_out.forward(h).silu());
    return .{ .moments = self.quant_conv.forward(h) };
}

pub const LoadedModel = struct {
    inner: Model,
    cfg: config.VisualConfig,

    pub fn init(store: zml.io.TensorStore.View, cfg: config.VisualConfig) LoadedModel {
        return .{ .inner = .init(store), .cfg = cfg };
    }

    pub fn loadBuffers(self: *const LoadedModel, run: *const ops.Run, store: *zml.io.TensorStore) !zml.Bufferized(Model) {
        return ops.load(run, store, Model, &self.inner, null);
    }
};

pub const Compiled = struct {
    encode: zml.FnExe(encode),
    encode_clip: ?zml.FnExe(encode) = null,
    tile: u32,
    clip_t: u32 = 17,

    pub fn deinit(self: *Compiled) void {
        self.encode.deinit();
        if (self.encode_clip) |*c| c.deinit();
    }
};

pub fn compile(run: *const ops.Run, model: Model) !Compiled {
    const tile: u32 = config.vae_tile_px;
    const exe = try zml.FnExe(encode).compile(run.allocator, run.io, run.platform, .{
        .shardings = &run.mesh,
        .program_name = "minimax_h3_visual_encode",
    }, .{.{
        .model = model,
        .pixels = .init(.{ .b = 1, .c = 3, .t = 1, .h = tile, .w = tile }, .f32),
    }});
    return .{ .encode = exe, .tile = tile };
}

pub fn compileClip(run: *const ops.Run, compiled: *Compiled, model: Model, vae: config.VisualConfig) !void {
    if (compiled.encode_clip != null) return;
    const tile = compiled.tile;
    const clip_t: u32 = @intCast(vae.clip_length);
    compiled.encode_clip = try zml.FnExe(encode).compile(run.allocator, run.io, run.platform, .{
        .shardings = &run.mesh,
        .program_name = "minimax_h3_visual_encode_clip",
    }, .{.{
        .model = model,
        .pixels = .init(.{ .b = 1, .c = 3, .t = clip_t, .h = tile, .w = tile }, .f32),
    }});
    compiled.clip_t = clip_t;
}

fn rgbToNchw(allocator: std.mem.Allocator, rgb: []const u8, h: u32, w: u32) ![]f32 {
    return rgbVideoToNchw(allocator, rgb, 1, h, w);
}

pub fn rgbVideoToNchw(allocator: std.mem.Allocator, rgb: []const u8, frames: u32, h: u32, w: u32) ![]f32 {
    const plane = @as(usize, h) * w;
    const out = try allocator.alloc(f32, 3 * frames * plane);
    var f: u32 = 0;
    while (f < frames) : (f += 1) {
        var i: usize = 0;
        while (i < plane) : (i += 1) {
            inline for (0..3) |c| {
                const v = @as(f32, @floatFromInt(rgb[(f * plane + i) * 3 + c])) / 255.0;
                out[(c * frames + f) * plane + i] = (v - config.imagenet_mean[c]) / config.imagenet_std[c];
            }
        }
    }
    return out;
}

fn copyTile(src: []const f32, src_h: u32, src_w: u32, y0: u32, x0: u32, tile: u32, dst: []f32) void {
    @memset(dst, 0);
    const copy_h = @min(tile, src_h -| y0);
    const copy_w = @min(tile, src_w -| x0);
    var c: u32 = 0;
    while (c < 3) : (c += 1) {
        var y: u32 = 0;
        while (y < copy_h) : (y += 1) {
            const s = (c * src_h + (y0 + y)) * src_w + x0;
            const d = (c * tile + y) * tile;
            @memcpy(dst[d..][0..copy_w], src[s..][0..copy_w]);
        }
    }
}

fn nchwIdx(c: usize, y: usize, x: usize, h: usize, w: usize) usize {
    return ((c * h) + y) * w + x;
}

fn blendW(a: []const f32, b: []f32, h: u32, w: u32, extent: u32) void {
    const e = @min(w, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    for (0..48) |c| {
        for (0..h) |y| {
            for (0..e) |x| {
                const wb = @as(f32, @floatFromInt(x)) / ef;
                const ai = nchwIdx(c, y, w - e + x, h, w);
                const bi = nchwIdx(c, y, x, h, w);
                b[bi] = a[ai] * (1.0 - wb) + b[bi] * wb;
            }
        }
    }
}

fn blendH(a: []const f32, b: []f32, h: u32, w: u32, extent: u32) void {
    const e = @min(h, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    for (0..48) |c| {
        for (0..e) |y| {
            for (0..w) |x| {
                const wb = @as(f32, @floatFromInt(y)) / ef;
                const ai = nchwIdx(c, h - e + y, x, h, w);
                const bi = nchwIdx(c, y, x, h, w);
                b[bi] = a[ai] * (1.0 - wb) + b[bi] * wb;
            }
        }
    }
}

fn copyCrop(dst: []f32, dst_h: u32, dst_w: u32, oy: u32, ox: u32, src: []const f32, src_h: u32, src_w: u32, use_h: u32, use_w: u32) void {
    for (0..48) |c| {
        for (0..use_h) |y| {
            @memcpy(
                dst[nchwIdx(c, oy + y, ox, dst_h, dst_w)..][0..use_w],
                src[nchwIdx(c, y, 0, src_h, src_w)..][0..use_w],
            );
        }
    }
}

fn samplePosterior(allocator: std.mem.Allocator, moments: []const f32, h: u32, w: u32) ![]f32 {
    const spatial = @as(usize, h) * w;
    const out = try allocator.alloc(f32, spatial * 24);
    var rng = std.Random.DefaultPrng.init(42);
    const r = rng.random();
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        const mean = moments[i];
        const logvar = std.math.clamp(moments[spatial * 24 + i], -30.0, 20.0);
        out[i] = mean + @exp(0.5 * logvar) * r.floatNorm(f32);
    }
    return out;
}

fn nchwToThwc(allocator: std.mem.Allocator, nchw: []const f32, h: u32, w: u32) ![]f32 {
    const out = try allocator.alloc(f32, nchw.len);
    var c: u32 = 0;
    while (c < 24) : (c += 1) {
        var y: u32 = 0;
        while (y < h) : (y += 1) {
            var x: u32 = 0;
            while (x < w) : (x += 1) {
                out[((y * w + x) * 24) + c] = nchw[((c * h) + y) * w + x];
            }
        }
    }
    return out;
}

fn patchify(allocator: std.mem.Allocator, thwc: []const f32, h: u32, w: u32) ![]f32 {
    const rows = (h / 2) * (w / 2);
    const out = try allocator.alloc(f32, rows * 96);
    var row: usize = 0;
    var hh: u32 = 0;
    while (hh < h) : (hh += 2) {
        var ww: u32 = 0;
        while (ww < w) : (ww += 2) {
            var dst: usize = 0;
            var c: u32 = 0;
            while (c < 24) : (c += 1) {
                var dh: u32 = 0;
                while (dh < 2) : (dh += 1) {
                    var dw: u32 = 0;
                    while (dw < 2) : (dw += 1) {
                        out[row * 96 + dst] = thwc[((((hh + dh) * w + (ww + dw)) * 24) + c)];
                        dst += 1;
                    }
                }
            }
            row += 1;
        }
    }
    return out;
}

/// VAE-encode one still to DiT video patches `{s, 96}`.
pub fn encodeStill(
    run: *const ops.Run,
    compiled: *const Compiled,
    bufs: *const zml.Bufferized(Model),
    cfg: config.VisualConfig,
    rgb: []const u8,
    height: u32,
    width: u32,
) ![]f32 {
    const allocator = run.allocator;
    const spatial = cfg.spatial();
    const tile = compiled.tile;
    const pad_h = @max(height, tile);
    const pad_w = @max(width, tile);
    var nchw = try rgbToNchw(allocator, rgb, height, width);
    defer allocator.free(nchw);
    if (pad_h != height or pad_w != width) {
        const padded = try allocator.alloc(f32, 3 * pad_h * pad_w);
        @memset(padded, 0);
        var c: u32 = 0;
        while (c < 3) : (c += 1) {
            var y: u32 = 0;
            while (y < height) : (y += 1) {
                @memcpy(
                    padded[(c * pad_h + y) * pad_w ..][0..width],
                    nchw[(c * height + y) * width ..][0..width],
                );
            }
        }
        allocator.free(nchw);
        nchw = padded;
    }

    const y_plan = try ops.splitTiles(allocator, pad_h, tile, config.vae_tile_overlap_px, spatial);
    defer y_plan.deinit(allocator);
    const x_plan = try ops.splitTiles(allocator, pad_w, tile, config.vae_tile_overlap_px, spatial);
    defer x_plan.deinit(allocator);

    const lat_h = pad_h / spatial;
    const lat_w = pad_w / spatial;
    const tile_lat = tile / spatial;
    const moments = try allocator.alloc(f32, 48 * lat_h * lat_w);
    defer allocator.free(moments);
    @memset(moments, 0);

    var runner = try zml.FnExe(encode).Runner(.{.model}).init(&compiled.encode, allocator, .{ .model = bufs.* });
    defer runner.deinit(allocator);
    const pix = try allocator.alloc(f32, 3 * tile * tile);
    defer allocator.free(pix);
    const tile_mom = try allocator.alloc(f32, 48 * tile_lat * tile_lat);
    defer allocator.free(tile_mom);

    const n_y: u32 = @intCast(y_plan.starts.len);
    const n_x: u32 = @intCast(x_plan.starts.len);
    const tile_n = 48 * tile_lat * tile_lat;
    const prev_row = try allocator.alloc(f32, n_x * tile_n);
    defer allocator.free(prev_row);
    const curr_row = try allocator.alloc(f32, n_x * tile_n);
    defer allocator.free(curr_row);
    const work = try allocator.alloc(f32, tile_n);
    defer allocator.free(work);
    var out_y: u32 = 0;
    for (y_plan.starts, 0..) |y0, yi| {
        var out_x: u32 = 0;
        for (x_plan.starts, 0..) |x0, xi| {
            copyTile(nchw, pad_h, pad_w, y0, x0, tile, pix);
            var pix_buf = try zml.Buffer.fromBytes(run.io, run.platform, .init(.{ .b = 1, .c = 3, .t = 1, .h = tile, .w = tile }, .f32), .replicated, std.mem.sliceAsBytes(pix));
            defer pix_buf.deinit();
            var mom_buf: zml.Buffer = undefined;
            runner.run(run.io, .{
                .inputs = .{ .pixels = pix_buf },
                .outputs = .{ .moments = &mom_buf },
                .opts = .{ .wait = true },
            });
            defer mom_buf.deinit();
            const slice = try mom_buf.toSliceAlloc(allocator, run.io);
            defer slice.free(allocator);
            @memcpy(tile_mom, slice.items(f32)[0..tile_mom.len]);

            @memcpy(curr_row[xi * tile_n ..][0..tile_n], tile_mom);
            @memcpy(work, tile_mom);
            if (yi > 0) blendH(prev_row[xi * tile_n ..][0..tile_n], work, tile_lat, tile_lat, y_plan.overlaps[yi - 1] / spatial);
            if (xi > 0) blendW(curr_row[(xi - 1) * tile_n ..][0..tile_n], work, tile_lat, tile_lat, x_plan.overlaps[xi - 1] / spatial);
            const use_h = if (yi + 1 < n_y) tile_lat - y_plan.overlaps[yi] / spatial else lat_h - out_y;
            const use_w = if (xi + 1 < n_x) tile_lat - x_plan.overlaps[xi] / spatial else lat_w - out_x;
            copyCrop(moments, lat_h, lat_w, out_y, out_x, work, tile_lat, tile_lat, use_h, use_w);
            out_x += use_w;
        }
        @memcpy(prev_row, curr_row);
        out_y += if (yi + 1 < n_y) tile_lat - y_plan.overlaps[yi] / spatial else lat_h - out_y;
    }

    const crop_h = height / spatial;
    const crop_w = width / spatial;
    const cropped = try allocator.alloc(f32, 48 * crop_h * crop_w);
    defer allocator.free(cropped);
    for (0..48) |c| {
        for (0..crop_h) |y| {
            @memcpy(
                cropped[nchwIdx(c, y, 0, crop_h, crop_w)..][0..crop_w],
                moments[nchwIdx(c, y, 0, lat_h, lat_w)..][0..crop_w],
            );
        }
    }

    const sampled = try samplePosterior(allocator, cropped, crop_h, crop_w);
    defer allocator.free(sampled);
    var thwc = try nchwToThwc(allocator, sampled, crop_h, crop_w);
    defer allocator.free(thwc);
    var i: usize = 0;
    while (i < thwc.len) : (i += 1) {
        const c = i % 24;
        thwc[i] = (thwc[i] - cfg.latents_mean[c]) / cfg.latents_std[c];
    }
    return patchify(allocator, thwc, crop_h, crop_w);
}

fn ncthw(c: usize, t: usize, y: usize, x: usize, T: usize, H: usize, W: usize) usize {
    return (((c * T + t) * H + y) * W) + x;
}

fn copyTileT(src: []const f32, src_t: u32, src_h: u32, src_w: u32, y0: u32, x0: u32, tile: u32, dst: []f32) void {
    @memset(dst, 0);
    const copy_h = @min(tile, src_h -| y0);
    const copy_w = @min(tile, src_w -| x0);
    var c: u32 = 0;
    while (c < 3) : (c += 1) {
        var t: u32 = 0;
        while (t < src_t) : (t += 1) {
            var y: u32 = 0;
            while (y < copy_h) : (y += 1) {
                const s = ncthw(c, t, y0 + y, x0, src_t, src_h, src_w);
                const d = ncthw(c, t, y, 0, src_t, tile, tile);
                @memcpy(dst[d..][0..copy_w], src[s..][0..copy_w]);
            }
        }
    }
}

fn blendWT(a: []const f32, b: []f32, t: u32, h: u32, w: u32, extent: u32) void {
    const e = @min(w, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    for (0..48) |c| {
        for (0..t) |ti| {
            for (0..h) |y| {
                for (0..e) |x| {
                    const wb = @as(f32, @floatFromInt(x)) / ef;
                    const ai = ncthw(c, ti, y, w - e + x, t, h, w);
                    const bi = ncthw(c, ti, y, x, t, h, w);
                    b[bi] = a[ai] * (1.0 - wb) + b[bi] * wb;
                }
            }
        }
    }
}

fn blendHT(a: []const f32, b: []f32, t: u32, h: u32, w: u32, extent: u32) void {
    const e = @min(h, extent);
    if (e == 0) return;
    const ef: f32 = @floatFromInt(e);
    for (0..48) |c| {
        for (0..t) |ti| {
            for (0..e) |y| {
                for (0..w) |x| {
                    const wb = @as(f32, @floatFromInt(y)) / ef;
                    const ai = ncthw(c, ti, h - e + y, x, t, h, w);
                    const bi = ncthw(c, ti, y, x, t, h, w);
                    b[bi] = a[ai] * (1.0 - wb) + b[bi] * wb;
                }
            }
        }
    }
}

fn copyCropT(dst: []f32, dst_t: u32, dst_h: u32, dst_w: u32, oy: u32, ox: u32, src: []const f32, src_t: u32, src_h: u32, src_w: u32, use_h: u32, use_w: u32) void {
    for (0..48) |c| {
        for (0..src_t) |ti| {
            for (0..use_h) |y| {
                @memcpy(
                    dst[ncthw(c, ti, oy + y, ox, dst_t, dst_h, dst_w)..][0..use_w],
                    src[ncthw(c, ti, y, 0, src_t, src_h, src_w)..][0..use_w],
                );
            }
        }
    }
}

fn samplePosteriorT(allocator: std.mem.Allocator, moments: []const f32, t: u32, h: u32, w: u32) ![]f32 {
    const spatial = @as(usize, t) * h * w;
    const out = try allocator.alloc(f32, spatial * 24);
    var rng = std.Random.DefaultPrng.init(42);
    const r = rng.random();
    var i: usize = 0;
    while (i < out.len) : (i += 1) {
        const mean = moments[i];
        const logvar = std.math.clamp(moments[spatial * 24 + i], -30.0, 20.0);
        out[i] = mean + @exp(0.5 * logvar) * r.floatNorm(f32);
    }
    return out;
}

fn nchwToThwcT(allocator: std.mem.Allocator, nchw: []const f32, t: u32, h: u32, w: u32) ![]f32 {
    const out = try allocator.alloc(f32, nchw.len);
    var c: u32 = 0;
    while (c < 24) : (c += 1) {
        var ti: u32 = 0;
        while (ti < t) : (ti += 1) {
            var y: u32 = 0;
            while (y < h) : (y += 1) {
                var x: u32 = 0;
                while (x < w) : (x += 1) {
                    out[(((ti * h + y) * w + x) * 24) + c] = nchw[ncthw(c, ti, y, x, t, h, w)];
                }
            }
        }
    }
    return out;
}

fn patchifyT(allocator: std.mem.Allocator, thwc: []const f32, t: u32, h: u32, w: u32) ![]f32 {
    const rows = t * (h / 2) * (w / 2);
    const out = try allocator.alloc(f32, rows * 96);
    var row: usize = 0;
    var tt: u32 = 0;
    while (tt < t) : (tt += 1) {
        var hh: u32 = 0;
        while (hh < h) : (hh += 2) {
            var ww: u32 = 0;
            while (ww < w) : (ww += 2) {
                var dst: usize = 0;
                var c: u32 = 0;
                while (c < 24) : (c += 1) {
                    var dh: u32 = 0;
                    while (dh < 2) : (dh += 1) {
                        var dw: u32 = 0;
                        while (dw < 2) : (dw += 1) {
                            out[row * 96 + dst] = thwc[((((tt * h + (hh + dh)) * w) + (ww + dw)) * 24) + c];
                            dst += 1;
                        }
                    }
                }
                row += 1;
            }
        }
    }
    return out;
}

fn padTimeNchw(dst: []f32, src: []const f32, src_t: u32, dst_t: u32, h: u32, w: u32) void {
    const plane = @as(usize, h) * w;
    var c: u32 = 0;
    while (c < 3) : (c += 1) {
        @memcpy(dst[(c * dst_t) * plane ..][0 .. src_t * plane], src[(c * src_t) * plane ..][0 .. src_t * plane]);
        if (src_t == 0) continue;
        const last = src[(c * src_t + (src_t - 1)) * plane ..][0..plane];
        var t: u32 = src_t;
        while (t < dst_t) : (t += 1) {
            @memcpy(dst[(c * dst_t + t) * plane ..][0..plane], last);
        }
    }
}

/// VAE-encode a reference clip to DiT video patches `{s, 96}`.
pub fn encodeVideo(
    run: *const ops.Run,
    compiled: *const Compiled,
    bufs: *const zml.Bufferized(Model),
    cfg: config.VisualConfig,
    nchw: []const f32,
    frames: u32,
    height: u32,
    width: u32,
) ![]f32 {
    const clip_exe = if (compiled.encode_clip) |*c| c else return error.VisualClipMissing;
    const allocator = run.allocator;
    const spatial = cfg.spatial();
    const tile = compiled.tile;
    const clip_t = compiled.clip_t;
    const chunk = config.visual_latents_per_chunk;
    const pad = (clip_t - (frames % clip_t)) % clip_t;
    const padded_t = frames + pad;
    const plane = @as(usize, height) * width;
    const padded = try allocator.alloc(f32, 3 * padded_t * plane);
    defer allocator.free(padded);
    padTimeNchw(padded, nchw, frames, padded_t, height, width);

    const clips = padded_t / clip_t;
    const pad_h = @max(height, tile);
    const pad_w = @max(width, tile);
    const y_plan = try ops.splitTiles(allocator, pad_h, tile, config.vae_tile_overlap_px, spatial);
    defer y_plan.deinit(allocator);
    const x_plan = try ops.splitTiles(allocator, pad_w, tile, config.vae_tile_overlap_px, spatial);
    defer x_plan.deinit(allocator);

    const lat_h = pad_h / spatial;
    const lat_w = pad_w / spatial;
    const tile_lat = tile / spatial;
    const crop_h = height / spatial;
    const crop_w = width / spatial;
    const all = try allocator.alloc(f32, 48 * clips * chunk * crop_h * crop_w);
    defer allocator.free(all);
    @memset(all, 0);

    var runner = try zml.FnExe(encode).Runner(.{.model}).init(clip_exe, allocator, .{ .model = bufs.* });
    defer runner.deinit(allocator);
    const pix = try allocator.alloc(f32, 3 * clip_t * tile * tile);
    defer allocator.free(pix);
    const tile_mom = try allocator.alloc(f32, 48 * chunk * tile_lat * tile_lat);
    defer allocator.free(tile_mom);
    const n_y: u32 = @intCast(y_plan.starts.len);
    const n_x: u32 = @intCast(x_plan.starts.len);
    const tile_n = 48 * chunk * tile_lat * tile_lat;
    const prev_row = try allocator.alloc(f32, n_x * tile_n);
    defer allocator.free(prev_row);
    const curr_row = try allocator.alloc(f32, n_x * tile_n);
    defer allocator.free(curr_row);
    const work = try allocator.alloc(f32, tile_n);
    defer allocator.free(work);
    const clip_px = try allocator.alloc(f32, 3 * clip_t * pad_h * pad_w);
    defer allocator.free(clip_px);

    var clip_i: u32 = 0;
    while (clip_i < clips) : (clip_i += 1) {
        @memset(clip_px, 0);
        var ch: u32 = 0;
        while (ch < 3) : (ch += 1) {
            var t: u32 = 0;
            while (t < clip_t) : (t += 1) {
                var y: u32 = 0;
                while (y < height) : (y += 1) {
                    const src = ncthw(ch, clip_i * clip_t + t, y, 0, padded_t, height, width);
                    const dst = ncthw(ch, t, y, 0, clip_t, pad_h, pad_w);
                    @memcpy(clip_px[dst..][0..width], padded[src..][0..width]);
                }
            }
        }

        const moments = try allocator.alloc(f32, 48 * chunk * lat_h * lat_w);
        defer allocator.free(moments);
        @memset(moments, 0);
        var out_y: u32 = 0;
        for (y_plan.starts, 0..) |y0, yi| {
            var out_x: u32 = 0;
            for (x_plan.starts, 0..) |x0, xi| {
                copyTileT(clip_px, clip_t, pad_h, pad_w, y0, x0, tile, pix);
                var pix_buf = try zml.Buffer.fromBytes(run.io, run.platform, .init(.{ .b = 1, .c = 3, .t = clip_t, .h = tile, .w = tile }, .f32), .replicated, std.mem.sliceAsBytes(pix));
                defer pix_buf.deinit();
                var mom_buf: zml.Buffer = undefined;
                runner.run(run.io, .{
                    .inputs = .{ .pixels = pix_buf },
                    .outputs = .{ .moments = &mom_buf },
                    .opts = .{ .wait = true },
                });
                defer mom_buf.deinit();
                const slice = try mom_buf.toSliceAlloc(allocator, run.io);
                defer slice.free(allocator);
                @memcpy(tile_mom, slice.items(f32)[0..tile_mom.len]);
                @memcpy(curr_row[xi * tile_n ..][0..tile_n], tile_mom);
                @memcpy(work, tile_mom);
                if (yi > 0) blendHT(prev_row[xi * tile_n ..][0..tile_n], work, chunk, tile_lat, tile_lat, y_plan.overlaps[yi - 1] / spatial);
                if (xi > 0) blendWT(curr_row[(xi - 1) * tile_n ..][0..tile_n], work, chunk, tile_lat, tile_lat, x_plan.overlaps[xi - 1] / spatial);
                const use_h = if (yi + 1 < n_y) tile_lat - y_plan.overlaps[yi] / spatial else lat_h - out_y;
                const use_w = if (xi + 1 < n_x) tile_lat - x_plan.overlaps[xi] / spatial else lat_w - out_x;
                copyCropT(moments, chunk, lat_h, lat_w, out_y, out_x, work, chunk, tile_lat, tile_lat, use_h, use_w);
                out_x += use_w;
            }
            @memcpy(prev_row, curr_row);
            out_y += if (yi + 1 < n_y) tile_lat - y_plan.overlaps[yi] / spatial else lat_h - out_y;
        }

        for (0..48) |c| {
            for (0..chunk) |t| {
                for (0..crop_h) |y| {
                    const dst = ncthw(c, clip_i * chunk + t, y, 0, clips * chunk, crop_h, crop_w);
                    const src = ncthw(c, t, y, 0, chunk, lat_h, lat_w);
                    @memcpy(all[dst..][0..crop_w], moments[src..][0..crop_w]);
                }
            }
        }
        log.info("visual encode clip {d}/{d}", .{ clip_i + 1, clips });
    }

    const acc_t = clips * chunk;
    const keep_t = if (@as(u32, @intCast(cfg.token_drop)) < acc_t) acc_t - @as(u32, @intCast(cfg.token_drop)) else acc_t;
    const kept = try allocator.alloc(f32, 48 * keep_t * crop_h * crop_w);
    defer allocator.free(kept);
    var c: u32 = 0;
    while (c < 48) : (c += 1) {
        @memcpy(kept[(c * keep_t) * crop_h * crop_w ..][0 .. keep_t * crop_h * crop_w], all[(c * acc_t) * crop_h * crop_w ..][0 .. keep_t * crop_h * crop_w]);
    }
    const sampled = try samplePosteriorT(allocator, kept, keep_t, crop_h, crop_w);
    defer allocator.free(sampled);
    var thwc = try nchwToThwcT(allocator, sampled, keep_t, crop_h, crop_w);
    defer allocator.free(thwc);
    var i: usize = 0;
    while (i < thwc.len) : (i += 1) {
        const ch = i % 24;
        thwc[i] = (thwc[i] - cfg.latents_mean[ch]) / cfg.latents_std[ch];
    }
    log.info("visual encode video {d}x{d}x{d} -> {d}x{d}x{d}", .{ frames, height, width, keep_t, crop_h, crop_w });
    return patchifyT(allocator, thwc, keep_t, crop_h, crop_w);
}
