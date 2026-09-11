//! Audio VAE: decoder plus the encoder used for `--refs` audio.
//!
//!   1. reshape packed `(2·T, C)` left/right rows → `(2, C, T)`
//!   2. denormalize latents (`x * std + mean`)
//!   3. proj → conv_pre → 7× (upsample + AMP residual average) → conv_post
//!   4. clamp to `[-1, 1]`, interleave stereo

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");

const AudioConfig = config.AudioConfig;
const Run = ops.Run;

const log = std.log.scoped(.minimax_h3);

const WnLayout = enum { conv, conv_transpose };

fn padRepeatT(x: zml.Tensor, low: i64, high: i64) zml.Tensor {
    var y = x;
    if (low > 0) {
        const first = x.slice(.t, .{ .start = 0, .end = 1 });
        y = zml.Tensor.concatenate(&.{ first.broad(first.shape().setDim(.t, low)), y }, .t);
    }
    if (high > 0) {
        const last = x.slice(.t, .{ .start = x.dim(.t) - 1, .end = x.dim(.t) });
        y = zml.Tensor.concatenate(&.{ y, last.broad(last.shape().setDim(.t, high)) }, .t);
    }
    return y;
}

fn unloadOpt(t: *?zml.Buffer) void {
    if (t.*) |*buf| buf.deinit();
}

fn loadWn(store: zml.io.TensorStore.View, layout: WnLayout) struct { v: zml.Tensor, g: zml.Tensor } {
    return switch (layout) {
        .conv => .{
            .v = store.createTensor("weight_v", .{ .co, .ci, .k }, .replicated),
            .g = store.createTensor("weight_g", .{ .co, .ci, .k }, .replicated),
        },
        .conv_transpose => .{
            .v = store.createTensor("weight_v", .{ .ci, .co, .k }, .replicated),
            .g = store.createTensor("weight_g", .{ .ci, .co, .k }, .replicated),
        },
    };
}

/// `v * g / ||v||`. ZML `sum` keeps reduced axes at 1, matching `g`.
fn weightNorm(v: zml.Tensor, g: zml.Tensor, comptime ax0: anytype, comptime ax1: anytype) zml.Tensor {
    const vf = v.convert(.f32);
    const sq = vf.mul(vf).sum(ax0).sum(ax1).addConstant(1e-9);
    return vf.mul(g.convert(.f32).mul(sq.rsqrt()));
}

const WNConv1d = struct {
    weight_v: zml.Tensor,
    weight_g: zml.Tensor,
    bias: ?zml.Tensor,
    stride: i64,
    dilation: i64,
    padding: i64,

    pub fn init(store: zml.io.TensorStore.View, stride: i64, dilation: i64, padding: i64) WNConv1d {
        const wn = loadWn(store, .conv);
        return .{
            .weight_v = wn.v,
            .weight_g = wn.g,
            .bias = store.maybeCreateTensor("bias", .{.co}, .replicated),
            .stride = stride,
            .dilation = dilation,
            .padding = padding,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(WNConv1d)) void {
        self.weight_v.deinit();
        self.weight_g.deinit();
        unloadOpt(&self.bias);
    }

    pub fn forward(self: WNConv1d, x: zml.Tensor) zml.Tensor {
        const fused = weightNorm(self.weight_v.withPartialTags(.{ .co, .ci, .k }), self.weight_g, .k, .ci);
        var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t }).conv1d(fused, .{
            .window_strides = self.stride,
            .rhs_dilation = self.dilation,
            .padding = &.{ self.padding, self.padding },
        });
        if (self.bias) |bias| y = y.add(bias.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
        return y.convert(x.dtype());
    }
};

const TransposeConv = struct {
    weight_v: zml.Tensor,
    weight_g: zml.Tensor,
    bias: ?zml.Tensor,
    stride: i64,
    kernel: i64,

    pub fn init(store: zml.io.TensorStore.View, stride: i64, kernel: i64) TransposeConv {
        const inner = store.withPrefix("0");
        const wn = loadWn(inner, .conv_transpose);
        return .{
            .weight_v = wn.v,
            .weight_g = wn.g,
            .bias = inner.maybeCreateTensor("bias", .{.co}, .replicated),
            .stride = stride,
            .kernel = kernel,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransposeConv)) void {
        self.weight_v.deinit();
        self.weight_g.deinit();
        unloadOpt(&self.bias);
    }

    pub fn forward(self: TransposeConv, x: zml.Tensor) zml.Tensor {
        const fused = weightNorm(self.weight_v.withPartialTags(.{ .ci, .co, .k }), self.weight_g, .k, .co).reverse(.{.k});
        // conv_transpose1d: reverse the kernel, then conv1d with lhs dilation = stride.
        const conv_pad = @divFloor(self.kernel - self.stride, 2);
        const xla_pad = self.kernel - 1 - conv_pad;
        var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t }).conv1d(fused, .{
            .window_strides = 1,
            .lhs_dilation = self.stride,
            .padding = &.{ xla_pad, xla_pad },
            .kernel_input_feature_dimension = 0,
            .kernel_output_feature_dimension = 1,
            .kernel_spatial_dimensions = 2,
        });
        if (self.bias) |bias| y = y.add(bias.convert(.f32).rename(.{ .co = .c }).broad(y.shape()));
        return y.convert(x.dtype());
    }
};

const SnakeBeta = struct {
    alpha: zml.Tensor,
    beta: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) SnakeBeta {
        const act = store.withPrefix("act");
        return .{
            .alpha = act.createTensor("alpha", .{.c}, .replicated),
            .beta = act.createTensor("beta", .{.c}, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SnakeBeta)) void {
        self.alpha.deinit();
        self.beta.deinit();
    }

    pub fn forward(self: SnakeBeta, x: zml.Tensor) zml.Tensor {
        const alpha = self.alpha.convert(.f32).exp();
        const beta = self.beta.convert(.f32).exp();
        const xf = x.convert(.f32);
        const shaped = alpha.broad(xf.shape());
        const mag = zml.Tensor.scalar(1.0, .f32).div(beta.addConstant(1e-9)).broad(xf.shape());
        const s = xf.mul(shaped).sin();
        return xf.add(mag.mul(s.mul(s))).convert(x.dtype());
    }
};

const Activation1d = struct {
    act: SnakeBeta,
    up_filter: zml.Tensor,
    down_filter: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Activation1d {
        return .{
            .act = .init(store),
            .up_filter = store.createTensor("upsample.filter", .{ .co, .ci, .k }, .replicated),
            .down_filter = store.createTensor("downsample.lowpass.filter", .{ .co, .ci, .k }, .replicated),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Activation1d)) void {
        SnakeBeta.unloadBuffers(&self.act);
        self.up_filter.deinit();
        self.down_filter.deinit();
    }

    pub fn forward(self: Activation1d, x: zml.Tensor) zml.Tensor {
        const xt = x.withPartialTags(.{ .b, .c, .t });
        const channels = xt.dim(.c);
        const up = self.up_filter.convert(.f32).broad(zml.Shape.init(.{
            .co = channels,
            .ci = 1,
            .k = self.up_filter.dim(.k),
        }, .f32));
        const down = self.down_filter.convert(.f32).broad(zml.Shape.init(.{
            .co = channels,
            .ci = 1,
            .k = self.down_filter.dim(.k),
        }, .f32));
        const pad = @divFloor(config.audio_activation_kernel, config.audio_activation_ratio) - 1;
        const crop_left = pad * config.audio_activation_ratio + @divFloor(config.audio_activation_kernel - config.audio_activation_ratio, 2);
        const crop_right = pad * config.audio_activation_ratio + @divFloor(config.audio_activation_kernel - config.audio_activation_ratio + 1, 2);
        var y = padRepeatT(xt.convert(.f32), pad, pad);
        y = y.conv1d(up, .{
            .window_strides = 1,
            .lhs_dilation = config.audio_activation_ratio,
            .feature_group_count = channels,
            .padding = &.{ config.audio_activation_kernel - 1, config.audio_activation_kernel - 1 },
        }).scale(@as(f32, @floatFromInt(config.audio_activation_ratio)));
        y = y.slice(.t, .{ .start = crop_left, .end = y.dim(.t) - crop_right });
        y = self.act.forward(y.convert(x.dtype())).convert(.f32);
        const pad_left = @divFloor(config.audio_activation_kernel, 2) - 1;
        const pad_right = @divFloor(config.audio_activation_kernel, 2);
        y = padRepeatT(y, pad_left, pad_right);
        return y.conv1d(down, .{
            .window_strides = config.audio_activation_ratio,
            .feature_group_count = channels,
            .padding = &.{ 0, 0 },
        }).convert(x.dtype());
    }
};

const AMPBlock = struct {
    convs1: [3]WNConv1d,
    convs2: [3]WNConv1d,
    acts: [6]Activation1d,

    pub fn init(store: zml.io.TensorStore.View, kernel: i64, dilations: [3]i64) AMPBlock {
        var convs1: [3]WNConv1d = undefined;
        var convs2: [3]WNConv1d = undefined;
        var acts: [6]Activation1d = undefined;
        for (dilations, 0..) |d, i| {
            const pad1 = @divFloor(kernel * d - d, 2);
            const pad2 = @divFloor(kernel - 1, 2);
            convs1[i] = .init(store.withPrefix("convs1").withLayer(i), 1, d, pad1);
            convs2[i] = .init(store.withPrefix("convs2").withLayer(i), 1, 1, pad2);
            acts[i * 2] = .init(store.withPrefix("activations").withLayer(i * 2));
            acts[i * 2 + 1] = .init(store.withPrefix("activations").withLayer(i * 2 + 1));
        }
        return .{ .convs1 = convs1, .convs2 = convs2, .acts = acts };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AMPBlock)) void {
        for (&self.convs1) |*c| WNConv1d.unloadBuffers(c);
        for (&self.convs2) |*c| WNConv1d.unloadBuffers(c);
        for (&self.acts) |*a| Activation1d.unloadBuffers(a);
    }

    pub fn forward(self: AMPBlock, x: zml.Tensor) zml.Tensor {
        var hidden = x;
        for (0..self.convs1.len) |i| {
            var residual = self.acts[i * 2].forward(hidden);
            residual = self.convs1[i].forward(residual);
            residual = self.acts[i * 2 + 1].forward(residual);
            residual = self.convs2[i].forward(residual);
            hidden = hidden.add(residual);
        }
        return hidden;
    }
};

fn conv1x1(store: zml.io.TensorStore.View) zml.nn.Linear {
    const weight = store.createTensor("weight", .{ .dout, .d, .k }, .replicated);
    return .init(weight, store.maybeCreateTensor("bias", .{.dout}, .replicated), .d);
}

const Decoder = struct {
    dec_in_proj: zml.nn.Linear,
    conv_pre: WNConv1d,
    ups: []TransposeConv,
    resblocks: []AMPBlock,
    activation_post: Activation1d,
    conv_post: WNConv1d,
    cfg: AudioConfig,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: AudioConfig) !Decoder {
        const dec = store.withPrefix("decoder");
        const ups = try allocator.alloc(TransposeConv, cfg.decoder_rates.len);
        errdefer allocator.free(ups);
        for (ups, 0..) |*up, i| {
            up.* = .init(dec.withPrefix("ups").withLayer(i), cfg.decoder_rates[i], cfg.decoder_kernel_sizes[i]);
        }
        const n_res = cfg.decoder_rates.len * cfg.resblock_kernel_sizes.len;
        const resblocks = try allocator.alloc(AMPBlock, n_res);
        errdefer allocator.free(resblocks);
        for (0..cfg.decoder_rates.len) |i| {
            for (0..cfg.resblock_kernel_sizes.len) |j| {
                resblocks[i * cfg.resblock_kernel_sizes.len + j] = .init(
                    dec.withPrefix("resblocks").withLayer(i * cfg.resblock_kernel_sizes.len + j),
                    cfg.resblock_kernel_sizes[j],
                    cfg.resblock_dilation_sizes[j],
                );
            }
        }
        return .{
            .dec_in_proj = conv1x1(store.withPrefix("dec_in_proj")),
            .conv_pre = .init(dec.withPrefix("conv_pre"), 1, 1, 3),
            .ups = ups,
            .resblocks = resblocks,
            .activation_post = .init(dec.withPrefix("activation_post")),
            .conv_post = .init(dec.withPrefix("conv_post"), 1, 1, 3),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Decoder, allocator: std.mem.Allocator) void {
        allocator.free(self.ups);
        allocator.free(self.resblocks);
    }

    /// Nested slices (`ups`, `resblocks`) are not freed by `Buffer.deinitAll`.
    pub fn unloadBuffers(self: *zml.Bufferized(Decoder), allocator: std.mem.Allocator) void {
        self.dec_in_proj.weight.deinit();
        if (self.dec_in_proj.bias) |*bias| bias.deinit();
        WNConv1d.unloadBuffers(&self.conv_pre);
        for (self.ups) |*up| TransposeConv.unloadBuffers(up);
        allocator.free(self.ups);
        for (self.resblocks) |*block| AMPBlock.unloadBuffers(block);
        allocator.free(self.resblocks);
        Activation1d.unloadBuffers(&self.activation_post);
        WNConv1d.unloadBuffers(&self.conv_post);
    }
};

const DecodeInput = struct { model: Decoder, latents: zml.Tensor };
const DecodeOutput = struct { wav: zml.Tensor };

fn packedToBct(latents: zml.Tensor) zml.Tensor {
    const x = latents.withPartialTags(.{ .b, .s, .d }).squeeze(.b);
    const t = @divExact(x.dim(.s), 2);
    return x.splitAxis(.s, .{ .b = 2, .t = t }).rename(.{ .d = .c }).transpose(.{ .b, .c, .t });
}

fn projectIn(self: Decoder, latents: zml.Tensor) zml.Tensor {
    const x = latents.withPartialTags(.{ .b, .c, .t }).convert(.f32);
    const weight = self.dec_in_proj.weight.squeeze(.k);
    return (zml.nn.Linear.init(weight, self.dec_in_proj.bias, .d))
        .forward(x.rename(.{ .c = .d }))
        .rename(.{ .dout = .c })
        .transpose(.{ .b, .c, .t });
}

fn decode(input: DecodeInput) DecodeOutput {
    const self = input.model;
    var x = packedToBct(input.latents);
    x = ops.denorm(x, &self.cfg.latents_mean, &self.cfg.latents_std);
    x = projectIn(self, x);
    x = self.conv_pre.forward(x);
    const n_up = self.ups.len;
    const n_k = self.cfg.resblock_kernel_sizes.len;
    for (0..n_up) |i| {
        x = self.ups[i].forward(x);
        const blocks = self.resblocks[i * n_k ..][0..n_k];
        var acc = blocks[0].forward(x);
        for (blocks[1..]) |block| acc = acc.add(block.forward(x));
        x = acc.scale(1.0 / @as(f32, @floatFromInt(n_k)));
    }
    x = self.activation_post.forward(x);
    x = self.conv_post.forward(x);
    const one = zml.Tensor.scalar(1.0, x.dtype());
    const neg = zml.Tensor.scalar(-1.0, x.dtype());
    return .{ .wav = x.minimum(one).maximum(neg) };
}

fn interleaveStereo(allocator: std.mem.Allocator, left: []const f32, right: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, left.len * 2);
    for (left, right, 0..) |l, r, i| {
        out[i * 2] = l;
        out[i * 2 + 1] = r;
    }
    return out;
}

fn tensorRank(store: zml.io.TensorStore.View, name: []const u8) u8 {
    var buffer: [256]u8 = undefined;
    const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", name }) catch return 2;
    return if (store.store.getShape(key)) |shape| shape.rank() else 2;
}

fn pickChannel(store: zml.io.TensorStore.View, name: []const u8) zml.Tensor {
    return switch (tensorRank(store, name)) {
        3 => store.createTensor(name, .{ .unused_a, .c, .unused_b }, .replicated),
        2 => store.createTensor(name, .{ .unused_a, .c }, .replicated),
        else => store.createTensor(name, .{.c}, .replicated),
    };
}

fn squeezeToTag(t: zml.Tensor, comptime tag: anytype) zml.Tensor {
    var out = t.convert(.f32);
    var changed = true;
    while (changed and out.rank() > 1) {
        changed = false;
        var ax: i8 = 0;
        while (ax < @as(i8, @intCast(out.rank()))) : (ax += 1) {
            if (out.dim(ax) == 1) {
                out = out.squeeze(ax);
                changed = true;
                break;
            }
        }
    }
    return out.withTags(.{tag});
}

fn encLinear(store: zml.io.TensorStore.View) zml.nn.Linear {
    const weight = switch (tensorRank(store, "weight")) {
        3 => store.createTensor("weight", .{ .dout, .d, .k }, .replicated),
        else => store.createTensor("weight", .{ .dout, .d }, .replicated),
    };
    return .init(weight, store.maybeCreateTensor("bias", .{.dout}, .replicated), .d);
}

const Snake1d = struct {
    alpha: zml.Tensor,

    pub fn init(store: zml.io.TensorStore.View) Snake1d {
        return .{ .alpha = pickChannel(store, "alpha") };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Snake1d)) void {
        self.alpha.deinit();
    }

    pub fn forward(self: Snake1d, x: zml.Tensor) zml.Tensor {
        const xf = x.convert(.f32).withPartialTags(.{ .b, .c, .t });
        const a = squeezeToTag(self.alpha.convert(.f32), .c).broad(xf.shape());
        const s = xf.mul(a).sin();
        return xf.add(s.mul(s).div(a.addConstant(1e-9))).convert(x.dtype());
    }
};

const ResidualUnit = struct {
    snake0: Snake1d,
    conv0: WNConv1d,
    snake1: Snake1d,
    conv1: WNConv1d,

    pub fn init(store: zml.io.TensorStore.View, dilation: i64) ResidualUnit {
        const inner = store.withPrefix("block");
        const pad = @divFloor(6 * dilation, 2);
        return .{
            .snake0 = .init(inner.withLayer(0)),
            .conv0 = .init(inner.withLayer(1), 1, dilation, pad),
            .snake1 = .init(inner.withLayer(2)),
            .conv1 = .init(inner.withLayer(3), 1, 1, 0),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(ResidualUnit)) void {
        Snake1d.unloadBuffers(&self.snake0);
        WNConv1d.unloadBuffers(&self.conv0);
        Snake1d.unloadBuffers(&self.snake1);
        WNConv1d.unloadBuffers(&self.conv1);
    }

    pub fn forward(self: ResidualUnit, x: zml.Tensor) zml.Tensor {
        var y = self.conv1.forward(self.snake1.forward(self.conv0.forward(self.snake0.forward(x))));
        const xt = x.withPartialTags(.{ .b, .c, .t });
        const yt = y.withPartialTags(.{ .b, .c, .t });
        if (xt.dim(.t) != yt.dim(.t)) {
            const pad = @divFloor(xt.dim(.t) - yt.dim(.t), 2);
            return yt.add(xt.slice(.t, .{ .start = pad, .end = xt.dim(.t) - pad }));
        }
        return yt.add(xt);
    }
};

const EncoderBlock = struct {
    unit0: ResidualUnit,
    unit1: ResidualUnit,
    unit2: ResidualUnit,
    snake: Snake1d,
    conv: WNConv1d,

    pub fn init(store: zml.io.TensorStore.View, stride: i64) EncoderBlock {
        const inner = store.withPrefix("block");
        const pad = std.math.divCeil(i64, stride, 2) catch stride;
        return .{
            .unit0 = .init(inner.withLayer(0), 1),
            .unit1 = .init(inner.withLayer(1), 3),
            .unit2 = .init(inner.withLayer(2), 9),
            .snake = .init(inner.withLayer(3)),
            .conv = .init(inner.withLayer(4), stride, 1, pad),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(EncoderBlock)) void {
        ResidualUnit.unloadBuffers(&self.unit0);
        ResidualUnit.unloadBuffers(&self.unit1);
        ResidualUnit.unloadBuffers(&self.unit2);
        Snake1d.unloadBuffers(&self.snake);
        WNConv1d.unloadBuffers(&self.conv);
    }

    pub fn forward(self: EncoderBlock, x: zml.Tensor) zml.Tensor {
        return self.conv.forward(self.snake.forward(self.unit2.forward(self.unit1.forward(self.unit0.forward(x)))));
    }
};

const GeGluMlp = struct {
    norm: zml.nn.LayerNorm,
    w0: zml.nn.Linear,
    w1: zml.nn.Linear,
    w2: zml.nn.Linear,

    pub fn init(store: zml.io.TensorStore.View) GeGluMlp {
        return .{
            .norm = ops.ln(store.withPrefix("norm"), 1e-5),
            .w0 = encLinear(store.withPrefix("w0")),
            .w1 = encLinear(store.withPrefix("w1")),
            .w2 = encLinear(store.withPrefix("w2")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GeGluMlp)) void {
        self.norm.weight.deinit();
        if (self.norm.bias) |*b| b.deinit();
        zml.nn.Linear.unloadBuffers(&self.w0);
        zml.nn.Linear.unloadBuffers(&self.w1);
        zml.nn.Linear.unloadBuffers(&self.w2);
    }

    pub fn forward(self: GeGluMlp, x: zml.Tensor) zml.Tensor {
        const n = self.norm.forward(x);
        return self.w2.forward(self.w0.forward(n).gelu().mul(self.w1.forward(n)).rename(.{ .dout = .d })).rename(.{ .dout = .d });
    }
};

const CausalAttn = struct {
    qkv: zml.nn.Linear,
    q_bias: zml.Tensor,
    v_bias: zml.Tensor,
    k_bias: zml.Tensor,
    proj: zml.nn.Linear,
    num_heads: i64,
    head_dim: i64,
    out_dim: i64,

    pub fn init(store: zml.io.TensorStore.View, in_dim: i64, out_dim: i64, num_heads: i64) CausalAttn {
        return .{
            .qkv = encLinear(store.withPrefix("qkv")),
            .q_bias = store.createTensor("q_bias", .{.d}, .replicated),
            .v_bias = store.createTensor("v_bias", .{.d}, .replicated),
            .k_bias = store.createTensor("zero_k_bias", .{.d}, .replicated),
            .proj = encLinear(store.withPrefix("proj")),
            .num_heads = num_heads,
            .head_dim = @divExact(in_dim, num_heads),
            .out_dim = out_dim,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(CausalAttn)) void {
        zml.nn.Linear.unloadBuffers(&self.qkv);
        self.q_bias.deinit();
        self.v_bias.deinit();
        self.k_bias.deinit();
        zml.nn.Linear.unloadBuffers(&self.proj);
    }

    pub fn forward(self: CausalAttn, x: zml.Tensor) zml.Tensor {
        const xt = x.withPartialTags(.{ .b, .s, .d });
        const seq = xt.dim(.s);
        var qkv = self.qkv.forward(xt);
        const bias = zml.Tensor.concatenate(&.{
            self.q_bias.convert(xt.dtype()).withTags(.{.dout}),
            self.k_bias.convert(xt.dtype()).withTags(.{.dout}),
            self.v_bias.convert(xt.dtype()).withTags(.{.dout}),
        }, .dout);
        qkv = qkv.add(bias.broad(qkv.shape()));
        const parts = qkv.chunkExact(.dout, 3);
        const q = parts[0].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim }).rename(.{ .s = .q });
        const k = parts[1].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim }).rename(.{ .s = .k });
        const v = parts[2].rename(.{ .dout = .d }).splitAxis(.d, .{ .h = self.num_heads, .hd = self.head_dim }).rename(.{ .s = .k });
        const mask = zml.nn.causalAttnMask(.{ .q = seq, .k = seq }, .f32, null);
        var attn = zml.nn.sdpa(q, k, v, .{ .attn_mask = mask }).rename(.{ .q = .s });
        attn = attn.mean(.h).squeeze(.h);
        const pool = @divExact(self.head_dim, self.out_dim);
        attn = attn.splitAxis(.hd, .{ .d = self.out_dim, .k = pool }).mean(.k).squeeze(.k);
        return self.proj.forward(attn).rename(.{ .dout = .d });
    }
};

const AttnProjection = struct {
    norm1: zml.nn.LayerNorm,
    attn: CausalAttn,
    proj: zml.nn.Linear,
    norm3: zml.nn.LayerNorm,
    norm2: zml.nn.LayerNorm,
    mlp: GeGluMlp,

    pub fn init(store: zml.io.TensorStore.View, in_dim: i64, out_dim: i64) AttnProjection {
        return .{
            .norm1 = ops.ln(store.withPrefix("norm1"), 1e-5),
            .attn = .init(store.withPrefix("attn"), in_dim, out_dim, 8),
            .proj = encLinear(store.withPrefix("proj")),
            .norm3 = ops.ln(store.withPrefix("norm3"), 1e-5),
            .norm2 = ops.ln(store.withPrefix("norm2"), 1e-5),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttnProjection)) void {
        self.norm1.weight.deinit();
        if (self.norm1.bias) |*b| b.deinit();
        CausalAttn.unloadBuffers(&self.attn);
        zml.nn.Linear.unloadBuffers(&self.proj);
        self.norm3.weight.deinit();
        if (self.norm3.bias) |*b| b.deinit();
        self.norm2.weight.deinit();
        if (self.norm2.bias) |*b| b.deinit();
        GeGluMlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: AttnProjection, x: zml.Tensor) zml.Tensor {
        const xt = x.withPartialTags(.{ .b, .s, .d });
        var y = self.proj.forward(self.norm3.forward(xt)).rename(.{ .dout = .d });
        y = y.add(self.attn.forward(self.norm1.forward(xt)));
        return y.add(self.mlp.forward(self.norm2.forward(y)));
    }
};

pub const EncoderModel = struct {
    conv_in: WNConv1d,
    blocks: [5]EncoderBlock,
    snake: Snake1d,
    conv_out: WNConv1d,
    pre_block: AttnProjection,
    mean_proj: zml.nn.Linear,
    cfg: AudioConfig,

    pub fn init(store: zml.io.TensorStore.View, cfg: AudioConfig) EncoderModel {
        const enc = store.withPrefix("encoder.block");
        const latent_dim: i64 = 2048;
        return .{
            .conv_in = .init(enc.withLayer(0), 1, 1, 3),
            .blocks = .{
                .init(enc.withLayer(1), cfg.encoder_rates[0]),
                .init(enc.withLayer(2), cfg.encoder_rates[1]),
                .init(enc.withLayer(3), cfg.encoder_rates[2]),
                .init(enc.withLayer(4), cfg.encoder_rates[3]),
                .init(enc.withLayer(5), cfg.encoder_rates[4]),
            },
            .snake = .init(enc.withLayer(6)),
            .conv_out = .init(enc.withLayer(7), 1, 1, 1),
            .pre_block = .init(store.withPrefix("pre_block"), latent_dim, cfg.latent_channels),
            .mean_proj = encLinear(store.withPrefix("mean_proj")),
            .cfg = cfg,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(EncoderModel)) void {
        WNConv1d.unloadBuffers(&self.conv_in);
        for (&self.blocks) |*block| EncoderBlock.unloadBuffers(block);
        Snake1d.unloadBuffers(&self.snake);
        WNConv1d.unloadBuffers(&self.conv_out);
        AttnProjection.unloadBuffers(&self.pre_block);
        zml.nn.Linear.unloadBuffers(&self.mean_proj);
    }
};

const EncodeInput = struct { model: EncoderModel, wav: zml.Tensor };
const EncodeOutput = struct { latents: zml.Tensor };

pub fn encode(input: EncodeInput) EncodeOutput {
    const self = input.model;
    var x = input.wav.withPartialTags(.{ .b, .c, .t }).convert(.f32);
    x = self.conv_in.forward(x);
    for (self.blocks) |block| x = block.forward(x);
    x = self.conv_out.forward(self.snake.forward(x));
    x = x.transpose(.{ .b, .t, .c }).rename(.{ .c = .d, .t = .s });
    x = self.pre_block.forward(x);
    x = self.mean_proj.forward(x).rename(.{ .dout = .c });
    if (x.shape().hasTag(.k) != null) x = x.squeeze(.k);
    return .{ .latents = x.transpose(.{ .b, .c, .s }).rename(.{ .s = .t }) };
}

fn audioBctToRows(dst: []f32, bct: []const f32, channels: u32, t: u32) void {
    var ear: usize = 0;
    while (ear < 2) : (ear += 1) {
        const src = bct[ear * channels * t ..][0 .. channels * t];
        const out = dst[ear * t * channels ..][0 .. t * channels];
        var c: usize = 0;
        while (c < channels) : (c += 1) {
            var ti: usize = 0;
            while (ti < t) : (ti += 1) {
                out[ti * channels + c] = src[c * t + ti];
            }
        }
    }
}

pub const AudioEncoded = struct {
    values: []f32,
    latent_t: u32,
};

pub fn encodeAudio(
    run: *const Run,
    exe: *const zml.FnExe(encode),
    bufs: *const zml.Bufferized(EncoderModel),
    cfg: AudioConfig,
    stereo: []const f32,
) !AudioEncoded {
    const allocator = run.allocator;
    const hop = cfg.hop();
    const frames: u32 = @intCast(stereo.len / 2);
    const pad = (hop - (frames % hop)) % hop;
    const samples = frames + pad;
    const batch = try allocator.alloc(f32, 2 * samples);
    defer allocator.free(batch);
    @memset(batch, 0);
    var i: usize = 0;
    while (i < frames) : (i += 1) {
        batch[i] = stereo[i * 2];
        batch[samples + i] = stereo[i * 2 + 1];
    }
    var runner = try zml.FnExe(encode).Runner(.{.model}).init(exe, allocator, .{ .model = bufs.* });
    defer runner.deinit(allocator);
    var wav = try zml.Buffer.fromBytes(run.io, run.platform, .init(.{ .b = 2, .c = 1, .t = samples }, .f32), .replicated, std.mem.sliceAsBytes(batch));
    defer wav.deinit();
    var latents: zml.Buffer = undefined;
    runner.run(run.io, .{
        .inputs = .{ .wav = wav },
        .outputs = .{ .latents = &latents },
        .opts = .{ .wait = true },
    });
    defer latents.deinit();
    const latent_t = samples / hop;
    const channels: usize = @intCast(cfg.latent_channels);
    const host = try allocator.alloc(f32, 2 * channels * latent_t);
    defer allocator.free(host);
    try latents.toSlice(run.io, .init(zml.Shape.init(.{ .b = 2, .c = cfg.latent_channels, .t = latent_t }, .f32), std.mem.sliceAsBytes(host)));
    const packed_latents = try allocator.alloc(f32, host.len);
    audioBctToRows(packed_latents, host, @intCast(channels), latent_t);
    var n: usize = 0;
    while (n < packed_latents.len) : (n += 1) {
        const c = n % channels;
        packed_latents[n] = (packed_latents[n] - cfg.latents_mean[c]) / cfg.latents_std[c];
    }
    log.info("audio encode samples={d} latent_t={d}", .{ samples, latent_t });
    return .{ .values = packed_latents, .latent_t = latent_t };
}

pub fn compileAudioEncode(run: *const Run, model: EncoderModel, samples: u32) !zml.FnExe(encode) {
    return zml.FnExe(encode).compile(run.allocator, run.io, run.platform, .{
        .shardings = &run.mesh,
        .program_name = "minimax_h3_audio_encode",
    }, .{.{
        .model = model,
        .wav = .init(.{ .b = 2, .c = 1, .t = samples }, .f32),
    }});
}

pub const Loaded = struct {
    bufs: zml.Bufferized(Decoder),
    loader: ?zml.io.Loader = null,

    pub fn wait(self: *Loaded, io: std.Io) !void {
        if (self.loader) |*loader| {
            try loader.await(io);
            loader.deinit();
            self.loader = null;
        }
    }

    pub fn deinit(self: *Loaded, allocator: std.mem.Allocator, io: std.Io) void {
        self.wait(io) catch {};
        Decoder.unloadBuffers(&self.bufs, allocator);
        allocator.destroy(self);
    }
};

pub const AudioVae = struct {
    inner: Decoder,
    compiled: ?zml.FnExe(decode) = null,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: AudioConfig) !AudioVae {
        return .{ .inner = try .init(allocator, store, cfg) };
    }

    pub fn deinit(self: *AudioVae, allocator: std.mem.Allocator) void {
        if (self.compiled) |*c| c.deinit();
        self.inner.deinit(allocator);
    }

    pub fn compile(self: *AudioVae, run: *const Run, geo: config.Geometry) !void {
        var node = run.progress.start("Compiling MiniMax-H3 audio VAE", 1);
        defer node.end();
        self.compiled = try zml.FnExe(decode).compile(run.allocator, run.io, run.platform, .{
            .shardings = &run.mesh,
            .program_name = "minimax_h3_audio_decode",
        }, .{.{
            .model = self.inner,
            .latents = .init(.{ .b = 1, .s = geo.audio_tokens, .d = self.inner.cfg.latent_channels }, .f32),
        }});
    }

    pub fn startLoad(self: *const AudioVae, run: *const Run, store: *zml.io.TensorStore) !*Loaded {
        const loaded = try run.allocator.create(Loaded);
        errdefer run.allocator.destroy(loaded);
        loaded.* = .{
            .bufs = try zml.mem.bufferize(run.allocator, Decoder, &self.inner),
            .loader = try .init(run.allocator, run.platform, ops.loader_opts),
        };
        errdefer Decoder.unloadBuffers(&loaded.bufs, run.allocator);
        errdefer loaded.loader.?.deinit();
        if (loaded.loader) |*loader| {
            try loader.load(run.io, Decoder, &self.inner, &loaded.bufs, store, &run.mesh, .{ .progress = run.progress });
        }
        return loaded;
    }

    /// Denoised audio tokens → interleaved stereo f32 in `[-1, 1]`.
    pub fn decodeAudio(
        self: *const AudioVae,
        run: *const Run,
        packed_audio: zml.Buffer,
        loaded: *Loaded,
    ) ![]f32 {
        const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
        try loaded.wait(run.io);
        var runner = try zml.FnExe(decode).Runner(.{.model}).init(compiled, run.allocator, .{ .model = loaded.bufs });
        defer runner.deinit(run.allocator);

        const decode_start: std.Io.Timestamp = .now(run.io, .awake);
        var wav: zml.Buffer = undefined;
        runner.run(run.io, .{
            .inputs = .{ .latents = packed_audio },
            .outputs = .{ .wav = &wav },
            .opts = .{ .wait = true },
        });
        defer wav.deinit();

        const samples: usize = @intCast(wav.shape().dim(.t));
        const host_pcm = try run.allocator.alloc(f32, 2 * samples);
        errdefer run.allocator.free(host_pcm);
        try wav.toSlice(run.io, .init(wav.shape(), std.mem.sliceAsBytes(host_pcm)));
        const interleaved = try interleaveStereo(run.allocator, host_pcm[0..samples], host_pcm[samples..]);
        run.allocator.free(host_pcm);
        log.info("decode audio: ok [{f}]", .{decode_start.untilNow(run.io, .awake)});
        return interleaved;
    }
};
