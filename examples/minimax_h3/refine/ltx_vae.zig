const std = @import("std");

const zml = @import("zml");

const sku = @import("../recipe/sku.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3_stage2);

// =============================================================================
// refine/ltx_vae.zig — LTX video VAE encoder
//
// Stage 2 encodes handoff RGB at half the target size.
// =============================================================================

pub const default_path = "/var/models/super-accel/ltx/vae/ltx-2.5-video-vae-conv-bf16.safetensors";
pub const weight_paths = [_][]const u8{
    default_path,
    "output/ltx-bf16/vae/ltx-2.5-video-vae-conv-bf16.safetensors",
    sku.hf_ltx_vae,
};

pub const Conv3 = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
};

pub const Res = struct {
    conv1: Conv3,
    conv2: Conv3,
};

pub const Encoder = struct {
    conv_in: Conv3,
    r0: [4]Res,
    down1: Conv3,
    r2: [6]Res,
    down3: Conv3,
    r4: [4]Res,
    down5: Conv3,
    r6: [2]Res,
    down7: Conv3,
    r8: [2]Res,
    conv_out: Conv3,
    mean: zml.Tensor,
    std: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Encoder {
        const e = store.withPrefix("encoder");
        return .{
            .conv_in = conv(e.withPrefix("conv_in.conv")),
            .r0 = res4(e.withPrefix("down_blocks.0.res_blocks")),
            .down1 = conv(e.withPrefix("down_blocks.1.conv.conv")),
            .r2 = res6(e.withPrefix("down_blocks.2.res_blocks")),
            .down3 = conv(e.withPrefix("down_blocks.3.conv.conv")),
            .r4 = res4(e.withPrefix("down_blocks.4.res_blocks")),
            .down5 = conv(e.withPrefix("down_blocks.5.conv.conv")),
            .r6 = res2(e.withPrefix("down_blocks.6.res_blocks")),
            .down7 = conv(e.withPrefix("down_blocks.7.conv.conv")),
            .r8 = res2(e.withPrefix("down_blocks.8.res_blocks")),
            .conv_out = conv(e.withPrefix("conv_out.conv")),
            .mean = store.createTensor("per_channel_statistics.mean-of-means", .{.c}, .replicated),
            .std = store.createTensor("per_channel_statistics.std-of-means", .{.c}, .replicated),
        };
    }
};

fn conv(store: zml.io.TensorStore.View) Conv3 {
    return .{
        .weight = store.createTensor("weight", .{ .co, .ci, .kt, .kh, .kw }, .replicated),
        .bias = store.maybeCreateTensor("bias", .{.co}, .replicated),
    };
}

fn oneRes(store: zml.io.TensorStore.View) Res {
    return .{
        .conv1 = conv(store.withPrefix("conv1.conv")),
        .conv2 = conv(store.withPrefix("conv2.conv")),
    };
}

fn res2(store: zml.io.TensorStore.View) [2]Res {
    return .{ oneRes(store.withLayer(0)), oneRes(store.withLayer(1)) };
}

fn res4(store: zml.io.TensorStore.View) [4]Res {
    return .{
        oneRes(store.withLayer(0)),
        oneRes(store.withLayer(1)),
        oneRes(store.withLayer(2)),
        oneRes(store.withLayer(3)),
    };
}

fn res6(store: zml.io.TensorStore.View) [6]Res {
    return .{
        oneRes(store.withLayer(0)),
        oneRes(store.withLayer(1)),
        oneRes(store.withLayer(2)),
        oneRes(store.withLayer(3)),
        oneRes(store.withLayer(4)),
        oneRes(store.withLayer(5)),
    };
}

fn asNcthw(x: zml.Tensor) zml.Tensor {
    return x.withTags(.{ .n, .c, .t, .h, .w });
}

fn asDt(x: zml.Tensor, t: zml.Tensor) zml.Tensor {
    return if (x.dtype() == t.dtype()) x else x.convert(t.dtype());
}

fn pixelNorm(x: zml.Tensor) zml.Tensor {
    const xf = x.convert(.f32);
    const mean_sq = xf.mul(xf).mean(.c);
    return xf.div(mean_sq.addConstant(1e-8).sqrt().broad(xf.shape())).convert(x.dtype());
}

fn causalConv(x: zml.Tensor, c: Conv3) zml.Tensor {
    const xf = asNcthw(x);
    const first = xf.slice(.t, .{ .start = 0, .end = 1 });
    const pad_t = first.broad(first.shape().setDim(.t, 2));
    var y = zml.Tensor.concatenate(&.{ pad_t, xf }, .t);
    y = y.pad(0, .{
        .h = zml.Tensor.Pad{ .low = 1, .high = 1 },
        .w = zml.Tensor.Pad{ .low = 1, .high = 1 },
    });
    y = asNcthw(y.conv3d(asDt(c.weight, y), .{}));
    if (c.bias) |b| y = y.add(asDt(b, y).rename(.{ .co = .c }).broad(y.shape()));
    return y;
}

fn applyRes(r: Res, x: zml.Tensor) zml.Tensor {
    var h = causalConv(pixelNorm(x).silu(), r.conv1);
    h = causalConv(pixelNorm(h).silu(), r.conv2);
    return h.add(x);
}

fn spaceToDepth(x: zml.Tensor, pt: i64, ph: i64, pw: i64) zml.Tensor {
    const n = x.dim(.n);
    const c = x.dim(.c);
    const t = x.dim(.t);
    const h = x.dim(.h);
    const w = x.dim(.w);
    const y = x.reshape(.{
        .n = n,
        .c = c,
        .t = @divExact(t, pt),
        .pt = pt,
        .h = @divExact(h, ph),
        .ph = ph,
        .w = @divExact(w, pw),
        .pw = pw,
    });
    const z = y.transpose(.{ .n, .c, .pt, .ph, .pw, .t, .h, .w });
    return asNcthw(z.reshape(.{
        .n = n,
        .c = c * pt * ph * pw,
        .t = @divExact(t, pt),
        .h = @divExact(h, ph),
        .w = @divExact(w, pw),
    }));
}

fn spaceDown(x: zml.Tensor, c: Conv3, pt: i64, ph: i64, pw: i64, out_c: i64) zml.Tensor {
    var src = x;
    if (pt == 2) {
        const first = x.slice(.t, .{ .start = 0, .end = 1 });
        src = zml.Tensor.concatenate(&.{ first, x }, .t);
    }
    const skip_full = spaceToDepth(src, pt, ph, pw);
    const groups = @divExact(skip_full.dim(.c), out_c);
    const skip = skip_full.reshape(.{
        .n = skip_full.dim(.n),
        .c = out_c,
        .g = groups,
        .t = skip_full.dim(.t),
        .h = skip_full.dim(.h),
        .w = skip_full.dim(.w),
    }).mean(.g).squeeze(.g);
    var y = spaceToDepth(causalConv(src, c), pt, ph, pw);
    return y.add(asDt(skip, y));
}

/// Official `patchify`: `b c (f p) (h q) (w r) -> b (c p r q) f h w`
/// (time, then width inner, then height inner). SpaceToDepth downsamples use `c,pt,ph,pw`.
fn patchify4(x: zml.Tensor) zml.Tensor {
    const n = x.dim(.n);
    const c = x.dim(.c);
    const t = x.dim(.t);
    const h = x.dim(.h);
    const w = x.dim(.w);
    const y = x.reshape(.{
        .n = n,
        .c = c,
        .t = t,
        .h = @divExact(h, 4),
        .q = @as(i64, 4),
        .w = @divExact(w, 4),
        .r = @as(i64, 4),
    });
    const z = y.transpose(.{ .n, .c, .r, .q, .t, .h, .w });
    return asNcthw(z.reshape(.{
        .n = n,
        .c = c * 16,
        .t = t,
        .h = @divExact(h, 4),
        .w = @divExact(w, 4),
    }));
}

pub fn nhwcToNcthwMinus1(nhwc: zml.Tensor) zml.Tensor {
    const n = nhwc.dim(.n);
    const t = nhwc.dim(.t);
    const h = nhwc.dim(.h);
    const w = nhwc.dim(.w);
    const y = nhwc.withTags(.{ .n, .t, .h, .w, .c }).transpose(.{ .n, .c, .t, .h, .w });
    const two = zml.Tensor.scalar(2.0, y.dtype());
    const one = zml.Tensor.scalar(1.0, y.dtype());
    _ = n;
    _ = t;
    _ = h;
    _ = w;
    return asNcthw(y.mul(two).sub(one));
}

pub const EncodeInput = struct {
    model: Encoder,
    pixels: zml.Tensor,
};

pub const EncodeOutput = struct {
    latent: zml.Tensor,
};

pub fn encode(input: EncodeInput) EncodeOutput {
    const m = input.model;
    var x = patchify4(nhwcToNcthwMinus1(input.pixels.convert(.f32))).convert(.bf16);
    x = causalConv(x, m.conv_in);
    for (m.r0) |r| x = applyRes(r, x);
    x = spaceDown(x, m.down1, 1, 2, 2, 256);
    for (m.r2) |r| x = applyRes(r, x);
    x = spaceDown(x, m.down3, 2, 1, 1, 512);
    for (m.r4) |r| x = applyRes(r, x);
    x = spaceDown(x, m.down5, 2, 2, 2, 1024);
    for (m.r6) |r| x = applyRes(r, x);
    x = spaceDown(x, m.down7, 2, 2, 2, 1024);
    for (m.r8) |r| x = applyRes(r, x);
    x = causalConv(pixelNorm(x).silu(), m.conv_out);
    x = x.slice(.c, .{ .start = 0, .end = 128 });
    const mean = asDt(m.mean, x).withTags(.{.c}).broad(x.shape());
    const stdv = asDt(m.std, x).withTags(.{.c}).broad(x.shape());
    return .{ .latent = x.sub(mean).div(stdv).convert(.f32) };
}

pub fn latentSize(frames: u32, height: u32, width: u32) struct { t: u32, h: u32, w: u32 } {
    return .{
        .t = @max(@as(u32, 1), (frames + 7) / 8),
        .h = height / 32,
        .w = width / 32,
    };
}

pub const Compiled = struct {
    encode: zml.FnExe(encode),
    bufs: zml.Bufferized(Encoder),
    time: u32,
    height: u32,
    width: u32,
    owns_bufs: bool = true,

    pub fn deinit(self: *Compiled) void {
        if (self.owns_bufs) zml.Buffer.deinitAll(Encoder, &self.bufs);
        self.encode.deinit();
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: Encoder,
    frames: u32,
    height: u32,
    width: u32,
    shardings: []const zml.Sharding,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    reuse: ?*const Compiled,
) !Compiled {
    progress.increaseEstimatedTotalItems(1);
    const exe = try zml.FnExe(encode).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_ltx_vae",
    }, .{.{
        .model = model,
        .pixels = .init(.{ .n = 1, .t = frames, .h = height, .w = width, .c = 3 }, .f32),
    }});
    const bufs = if (reuse) |src| src.bufs else try weights.load(allocator, io, platform, store, shardings, Encoder, &model, progress, null);
    log.info("compile LTX VAE encode {d}x{d}x{d}{s}", .{
        width,
        height,
        frames,
        if (reuse != null) " reuse weights" else "",
    });
    return .{
        .encode = exe,
        .bufs = bufs,
        .time = frames,
        .height = height,
        .width = width,
        .owns_bufs = reuse == null,
    };
}
