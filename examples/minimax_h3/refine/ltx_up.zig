const std = @import("std");

const zml = @import("zml");

const sku = @import("../recipe/sku.zig");
const weights = @import("../recipe/weights.zig");

const log = std.log.scoped(.minimax_h3_stage2);

// =============================================================================
// refine/ltx_up.zig — ×2 latent spatial upsampler
//
// Half-res VAE latent → refine canvas.
// =============================================================================

pub const default_path = "/var/models/super-accel/ltx/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors";
pub const weight_paths = [_][]const u8{
    default_path,
    "output/ltx-bf16/latent_upscale_models/ltx-2.5-latent-spatial-upscaler-x2-bf16-1.0.safetensors",
    sku.hf_ltx_up,
};
pub const spatial_factor: u32 = 2;

pub const Conv3 = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
};

pub const Conv2 = struct {
    weight: zml.Tensor,
    bias: ?zml.Tensor = null,
};

pub const Norm = struct {
    weight: zml.Tensor,
    bias: zml.Tensor,
};

pub const Res = struct {
    conv1: Conv3,
    norm1: Norm,
    conv2: Conv3,
    norm2: Norm,
};

pub const Model = struct {
    initial_conv: Conv3,
    initial_norm: Norm,
    res_blocks: [4]Res,
    up_conv: Conv2,
    post: [4]Res,
    final_conv: Conv3,

    pub fn init(store: zml.io.TensorStore.View) Model {
        return .{
            .initial_conv = conv3(store, "initial_conv"),
            .initial_norm = norm(store, "initial_norm"),
            .res_blocks = .{
                res(store, "res_blocks.0"),
                res(store, "res_blocks.1"),
                res(store, "res_blocks.2"),
                res(store, "res_blocks.3"),
            },
            .up_conv = conv2(store, "upsampler.0"),
            .post = .{
                res(store, "post_upsample_res_blocks.0"),
                res(store, "post_upsample_res_blocks.1"),
                res(store, "post_upsample_res_blocks.2"),
                res(store, "post_upsample_res_blocks.3"),
            },
            .final_conv = conv3(store, "final_conv"),
        };
    }
};

fn conv3(store: zml.io.TensorStore.View, name: []const u8) Conv3 {
    const s = store.withPrefix(name);
    return .{
        .weight = s.createTensor("weight", .{ .co, .ci, .kt, .kh, .kw }, .replicated),
        .bias = s.maybeCreateTensor("bias", .{.co}, .replicated),
    };
}

fn conv2(store: zml.io.TensorStore.View, name: []const u8) Conv2 {
    const s = store.withPrefix(name);
    return .{
        .weight = s.createTensor("weight", .{ .co, .ci, .kh, .kw }, .replicated),
        .bias = s.maybeCreateTensor("bias", .{.co}, .replicated),
    };
}

fn norm(store: zml.io.TensorStore.View, name: []const u8) Norm {
    const s = store.withPrefix(name);
    return .{
        .weight = s.createTensor("weight", .{.c}, .replicated),
        .bias = s.createTensor("bias", .{.c}, .replicated),
    };
}

fn res(store: zml.io.TensorStore.View, name: []const u8) Res {
    const s = store.withPrefix(name);
    return .{
        .conv1 = conv3(s, "conv1"),
        .norm1 = norm(s, "norm1"),
        .conv2 = conv3(s, "conv2"),
        .norm2 = norm(s, "norm2"),
    };
}

fn asNcthw(x: zml.Tensor) zml.Tensor {
    return x.withTags(.{ .n, .c, .t, .h, .w });
}

fn asNchw(x: zml.Tensor) zml.Tensor {
    return x.withTags(.{ .n, .c, .h, .w });
}

fn asDt(x: zml.Tensor, t: zml.Tensor) zml.Tensor {
    return if (x.dtype() == t.dtype()) x else x.convert(t.dtype());
}

fn applyConv3(x: zml.Tensor, c: Conv3) zml.Tensor {
    const xf = asNcthw(x).convert(.f32);
    var y = asNcthw(xf.conv3d(c.weight.convert(.f32), .{ .padding = &.{ 1, 1, 1, 1, 1, 1 } }));
    if (c.bias) |b| y = y.add(b.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
    return y;
}

fn applyConv2(x: zml.Tensor, c: Conv2) zml.Tensor {
    const xf = asNchw(x).convert(.f32);
    var y = asNchw(xf.conv2d(c.weight.convert(.f32), .{ .padding = &.{ 1, 1, 1, 1 } }));
    if (c.bias) |b| y = y.add(b.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
    return y;
}

fn applyNorm(x: zml.Tensor, n: Norm) zml.Tensor {
    const xf = asNcthw(x).convert(.f32);
    const groups: i64 = 32;
    const cg = @divExact(xf.dim(.c), groups);
    var y = xf.splitAxis(.c, .{ .g = groups, .cg = cg });
    y = y.merge(.{ .m = .{ .cg, .t, .h, .w } });
    const mean = y.mean(.m);
    const centered = y.sub(mean.broad(y.shape()));
    const variance = centered.mul(centered).mean(.m);
    y = centered.mul(variance.addConstant(1e-5).rsqrt().broad(y.shape()));
    y = y.splitAxis(.m, .{ .cg = cg, .t = xf.dim(.t), .h = xf.dim(.h), .w = xf.dim(.w) });
    y = asNcthw(y.merge(.{ .c = .{ .g, .cg } }));
    const scale = n.weight.convert(.f32).withTags(.{.c}).broad(y.shape());
    const shift = n.bias.convert(.f32).withTags(.{.c}).broad(y.shape());
    return y.mul(scale).add(shift);
}

fn applyRes(r: Res, x: zml.Tensor) zml.Tensor {
    var y = applyConv3(x, r.conv1);
    y = applyNorm(y, r.norm1).silu();
    y = applyConv3(y, r.conv2);
    y = applyNorm(y, r.norm2);
    return y.add(x).silu();
}

fn pixelShuffle2(x: zml.Tensor) zml.Tensor {
    const y = asNchw(x).splitAxis(.c, .{ .c = @divExact(x.dim(.c), 4), .rh = 2, .rw = 2 });
    const z = y.transpose(.{ .n, .c, .h, .rh, .w, .rw });
    return asNchw(z.merge(.{ .h = .{ .h, .rh }, .w = .{ .w, .rw } }));
}

/// Official `per_channel_statistics.un_normalize` before the ×2 conv.
fn unnorm(x: zml.Tensor, mean: zml.Tensor, stdv: zml.Tensor) zml.Tensor {
    const m = asDt(mean, x).withTags(.{.c}).broad(x.shape());
    const s = asDt(stdv, x).withTags(.{.c}).broad(x.shape());
    return x.mul(s).add(m);
}

/// Official `per_channel_statistics.normalize` after the ×2 conv.
fn renorm(x: zml.Tensor, mean: zml.Tensor, stdv: zml.Tensor) zml.Tensor {
    const m = asDt(mean, x).withTags(.{.c}).broad(x.shape());
    const s = asDt(stdv, x).withTags(.{.c}).broad(x.shape());
    return x.sub(m).div(s);
}

pub const Input = struct {
    model: Model,
    latent: zml.Tensor,
    mean: zml.Tensor,
    std: zml.Tensor,
};

pub const Output = struct {
    after_initial: zml.Tensor,
    after_pre: zml.Tensor,
    after_up: zml.Tensor,
    after_post: zml.Tensor,
    latent: zml.Tensor,
};

pub fn forward(input: Input) Output {
    const m = input.model;
    var x = unnorm(asNcthw(input.latent).convert(.bf16), input.mean, input.std).convert(.f32);
    x = applyNorm(applyConv3(x, m.initial_conv), m.initial_norm).silu();
    const after_initial = x.convert(.f32);
    for (m.res_blocks) |r| x = applyRes(r, x);
    const after_pre = x.convert(.f32);

    const n = x.dim(.n);
    const t = x.dim(.t);
    var y = x.transpose(.{ .n, .t, .c, .h, .w }).merge(.{ .n = .{ .n, .t } });
    y = pixelShuffle2(applyConv2(y, m.up_conv));
    x = asNcthw(y.splitAxis(.n, .{ .n = n, .t = t }).transpose(.{ .n, .c, .t, .h, .w }));
    const after_up = x.convert(.f32);

    for (m.post) |r| x = applyRes(r, x);
    const after_post = x.convert(.f32);
    x = applyConv3(x, m.final_conv).convert(.bf16);
    return .{
        .after_initial = after_initial,
        .after_pre = after_pre,
        .after_up = after_up,
        .after_post = after_post,
        .latent = renorm(x, input.mean, input.std).convert(.f32),
    };
}

pub const Compiled = struct {
    forward: zml.FnExe(forward),
    bufs: zml.Bufferized(Model),
    time: u32,
    height: u32,
    width: u32,
    owns_bufs: bool = true,

    pub fn deinit(self: *Compiled) void {
        if (self.owns_bufs) zml.Buffer.deinitAll(Model, &self.bufs);
        self.forward.deinit();
    }
};

pub fn compile(
    allocator: std.mem.Allocator,
    io: std.Io,
    platform: *const zml.Platform,
    model: Model,
    time: u32,
    height: u32,
    width: u32,
    shardings: []const zml.Sharding,
    store: *zml.io.TensorStore,
    progress: *std.Progress.Node,
    reuse: ?*const Compiled,
) !Compiled {
    progress.increaseEstimatedTotalItems(1);
    const exe = try zml.FnExe(forward).compile(allocator, io, platform, .{
        .shardings = shardings,
        .program_name = "minimax_h3_ltx_up",
    }, .{.{
        .model = model,
        .latent = .init(.{ .n = 1, .c = 128, .t = time, .h = height, .w = width }, .f32),
        .mean = .init(.{ .c = 128 }, .bf16),
        .std = .init(.{ .c = 128 }, .bf16),
    }});
    const bufs = if (reuse) |src| src.bufs else try weights.load(allocator, io, platform, store, shardings, Model, &model, progress, null);
    log.info("compile LTX upsampler {d}x{d}x{d}{s}", .{
        width,
        height,
        time,
        if (reuse != null) " reuse weights" else "",
    });
    return .{
        .forward = exe,
        .bufs = bufs,
        .time = time,
        .height = height,
        .width = width,
        .owns_bufs = reuse == null,
    };
}
