const std = @import("std");

const zml = @import("zml");

const config = @import("../core/config.zig");
const vae = @import("geometry.zig");
const weights = @import("../core/weights.zig");

const log = std.log.scoped(.minimax_h3_audio_vae);

/// Per-channel latent moments from the released `audio_vae/config.json`.
pub const official_latents_mean = [32]f32{
    -0.020211687488382354, 0.3876466479950502,   -0.04398279799186767, -0.28591514936373,
    0.08179686214561671,   -0.35782641352446604, 0.040623809960919084, -0.01552534501956604,
    -0.223362481667332,    0.1821006842509091,   0.2941778783780663,   -0.07901167601970885,
    -0.056815072777201,    -0.3699028221860095,  -0.31616315591624855, 0.5905951377425391,
    -0.052139568068853864, 0.013673160263486295, -0.03691647864630577, 0.09732660653298163,
    -0.3394662328788498,   -0.30685677538541667, -0.24504598907458763, -0.034698524462007344,
    0.02868032184767538,   -0.21217779266454084, -0.1678263169941987,  0.3221287889040614,
    -0.1223055851554907,   0.4356604928128464,   -0.0502599202236253,  0.3979258376211797,
};

pub const official_latents_std = [32]f32{
    1.6895524230479284, 2.76263727217653,   1.7945344281264435, 1.6801681847309828,
    1.6390226546605453, 2.7788298348882177, 1.7659090095747236, 1.6199757612137327,
    2.6336525640336896, 1.8539356672817833, 2.5056497896915633, 1.811019237886178,
    1.9579657790720237, 1.6685498243529284, 1.4922469314453364, 3.298670198067373,
    1.9491804496832168, 1.8720003270431442, 1.8334080103291832, 1.6488070416529093,
    1.6176957696319716, 1.9131449234774398, 1.5695245398428617, 1.6943659940415912,
    1.8318420762504692, 1.5540637421583379, 1.9344930328968526, 1.599198216109855,
    1.718045989838149,  1.6307219190837705, 1.8661226051202384, 1.5613768203168363,
};

pub const Config = struct {
    latent_channels: i64 = 32,
    latent_dim: i64 = 2048,
    encoder_dim: i64 = 64,
    decoder_dim: i64 = 1024,
    sample_rate: u32 = 32_000,
    hop: u32 = 800,
    upsample_rates: [7]i64 = .{ 5, 5, 2, 2, 2, 2, 2 },
    upsample_kernels: [7]i64 = .{ 9, 9, 4, 4, 4, 4, 4 },
    encoder_rates: [5]i64 = .{ 2, 4, 4, 5, 5 },
    resblock_kernels: [3]i64 = .{ 3, 7, 11 },
    resblock_dilations: [3][3]i64 = .{ .{ 1, 3, 5 }, .{ 1, 3, 5 }, .{ 1, 3, 5 } },
    latents_mean: [32]f32 = official_latents_mean,
    latents_std: [32]f32 = official_latents_std,

    pub fn official() Config {
        return .{};
    }

    pub fn spec(self: Config) vae.AudioSpec {
        return .{
            .channels = @intCast(self.latent_channels),
            .sample_rate = self.sample_rate,
            .hop = self.hop,
        };
    }
};

const FileConfig = struct {
    latent_channels: ?i64 = null,
    latent_dim: ?i64 = null,
    encoder_dim: ?i64 = null,
    decoder_dim: ?i64 = null,
    sampling_rate: ?u32 = null,
    latents_mean: ?[]const f32 = null,
    latents_std: ?[]const f32 = null,

    fn resolve(self: FileConfig) Config {
        var out = Config.official();
        if (self.latent_channels) |v| out.latent_channels = v;
        if (self.latent_dim) |v| out.latent_dim = v;
        if (self.encoder_dim) |v| out.encoder_dim = v;
        if (self.decoder_dim) |v| out.decoder_dim = v;
        if (self.sampling_rate) |v| out.sample_rate = v;
        if (self.latents_mean) |mean| {
            for (0..@min(mean.len, out.latents_mean.len)) |i| out.latents_mean[i] = mean[i];
        }
        if (self.latents_std) |stddev| {
            for (0..@min(stddev.len, out.latents_std.len)) |i| out.latents_std[i] = stddev[i];
        }
        return out;
    }
};

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

fn loadWn(store: zml.io.TensorStore.View, comptime transpose: bool) struct { v: zml.Tensor, g: zml.Tensor } {
    return .{
        .v = if (transpose)
            store.createTensor("weight_v", .{ .ci, .co, .k }, .replicated)
        else
            store.createTensor("weight_v", .{ .co, .ci, .k }, .replicated),
        .g = store.createTensor("weight_g", .{ .co, .ci, .k }, .replicated),
    };
}

const WNConv1d = struct {
    weight_v: zml.Tensor,
    weight_g: zml.Tensor,
    bias: ?zml.Tensor,
    stride: i64,
    dilation: i64,
    padding: i64,

    pub fn init(store: zml.io.TensorStore.View, stride: i64, dilation: i64, padding: i64) WNConv1d {
        const wn = loadWn(store, false);
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
        const v = self.weight_v.convert(.f32).withPartialTags(.{ .co, .ci, .k });
        const gs = squeezeToTag(self.weight_g.convert(.f32), .co);
        const sq = squeezeToTag(v.mul(v).sum(.k).sum(.ci), .co).addConstant(1e-9);
        const fused = v.mul(gs.mul(sq.rsqrt()).broad(v.shape()));
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
        const wn = loadWn(inner, true);
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
        const v = self.weight_v.convert(.f32).withPartialTags(.{ .ci, .co, .k });
        const gs = squeezeToTag(self.weight_g.convert(.f32), .ci);
        const sq = squeezeToTag(v.mul(v).sum(.k).sum(.co), .ci).addConstant(1e-9);
        const kernel = v.mul(gs.mul(sq.rsqrt()).broad(v.shape()));
        // Reverse along `k`: this conv is conv_transpose1d.
        const fused = kernel.reverse(.{.k});
        const official_pad = @divFloor(self.kernel - self.stride, 2);
        const xla_pad = self.kernel - 1 - official_pad;
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
    logscale: bool = true,

    pub fn init(store: zml.io.TensorStore.View) SnakeBeta {
        const act = store.withPrefix("act");
        return .{
            .alpha = pickChannel(act, "alpha"),
            .beta = pickChannel(act, "beta"),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(SnakeBeta)) void {
        self.alpha.deinit();
        self.beta.deinit();
    }

    pub fn forward(self: SnakeBeta, x: zml.Tensor) zml.Tensor {
        var alpha = self.alpha.convert(.f32);
        var beta = self.beta.convert(.f32);
        if (self.logscale) {
            alpha = alpha.exp();
            beta = beta.exp();
        }
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
    ratio: i64 = 2,
    kernel: i64 = 12,

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
            .k = self.up_filter.dim(-1),
        }, .f32));
        const down = self.down_filter.convert(.f32).broad(zml.Shape.init(.{
            .co = channels,
            .ci = 1,
            .k = self.down_filter.dim(-1),
        }, .f32));
        const pad = @divFloor(self.kernel, self.ratio) - 1;
        const crop_left = pad * self.ratio + @divFloor(self.kernel - self.ratio, 2);
        const crop_right = pad * self.ratio + @divFloor(self.kernel - self.ratio + 1, 2);
        var y = padRepeatT(xt.convert(.f32), pad, pad);
        y = y.conv1d(up, .{
            .window_strides = 1,
            .lhs_dilation = self.ratio,
            .feature_group_count = channels,
            .padding = &.{ self.kernel - 1, self.kernel - 1 },
        }).scale(@as(f32, @floatFromInt(self.ratio)));
        y = y.slice(.t, .{ .start = crop_left, .end = y.dim(.t) - crop_right });
        y = self.act.forward(y.convert(x.dtype())).convert(.f32);
        const even = @mod(self.kernel, 2) == 0;
        const pad_left = @divFloor(self.kernel, 2) - @intFromBool(even);
        const pad_right = @divFloor(self.kernel, 2);
        y = padRepeatT(y, pad_left, pad_right);
        return y.conv1d(down, .{
            .window_strides = self.ratio,
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
        for (0..3) |i| {
            var residual = self.acts[i * 2].forward(hidden);
            residual = self.convs1[i].forward(residual);
            residual = self.acts[i * 2 + 1].forward(residual);
            residual = self.convs2[i].forward(residual);
            hidden = hidden.add(residual);
        }
        return hidden;
    }
};

const layerNorm = weights.layerNorm;

fn conv1x1(store: zml.io.TensorStore.View) zml.nn.Linear {
    const weight = switch (tensorRank(store, "weight")) {
        3 => store.createTensor("weight", .{ .dout, .d, .k }, .replicated),
        else => store.createTensor("weight", .{ .dout, .d }, .replicated),
    };
    return .init(weight, store.maybeCreateTensor("bias", .{.dout}, .replicated), .d);
}

pub const Model = struct {
    dec_in_proj: zml.nn.Linear,
    conv_pre: WNConv1d,
    ups: []TransposeConv,
    resblocks: []AMPBlock,
    activation_post: Activation1d,
    conv_post: WNConv1d,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, store: zml.io.TensorStore.View, cfg: Config) !Model {
        const dec = store.withPrefix("decoder");

        const ups = try allocator.alloc(TransposeConv, cfg.upsample_rates.len);
        errdefer allocator.free(ups);
        for (ups, 0..) |*up, i| {
            up.* = .init(dec.withPrefix("ups").withLayer(i), cfg.upsample_rates[i], cfg.upsample_kernels[i]);
        }

        const n_res = cfg.upsample_rates.len * cfg.resblock_kernels.len;
        const resblocks = try allocator.alloc(AMPBlock, n_res);
        errdefer allocator.free(resblocks);
        for (0..cfg.upsample_rates.len) |i| {
            for (0..cfg.resblock_kernels.len) |j| {
                resblocks[i * cfg.resblock_kernels.len + j] = .init(
                    dec.withPrefix("resblocks").withLayer(i * cfg.resblock_kernels.len + j),
                    cfg.resblock_kernels[j],
                    cfg.resblock_dilations[j],
                );
            }
        }

        const proj_store = store.withPrefix("dec_in_proj");
        return .{
            .dec_in_proj = conv1x1(proj_store),
            .conv_pre = .init(dec.withPrefix("conv_pre"), 1, 1, 3),
            .ups = ups,
            .resblocks = resblocks,
            .activation_post = .init(dec.withPrefix("activation_post")),
            .conv_post = .init(dec.withPrefix("conv_post"), 1, 1, 3),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: Model, allocator: std.mem.Allocator) void {
        allocator.free(self.ups);
        allocator.free(self.resblocks);
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model), allocator: std.mem.Allocator) void {
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

pub const DecodeInput = struct {
    model: Model,
    latents: zml.Tensor,
};

pub const DecodeOutput = struct {
    wav: zml.Tensor,
};

fn projectIn(self: Model, latents: zml.Tensor) zml.Tensor {
    const x = latents.withPartialTags(.{ .b, .c, .t }).convert(.f32);
    var weight = self.dec_in_proj.weight;
    while (weight.rank() > 2) weight = weight.squeeze(-1);
    return (zml.nn.Linear.init(weight.withTags(.{ .dout, .d }), self.dec_in_proj.bias, .d))
        .forward(x.rename(.{ .c = .d }))
        .rename(.{ .dout = .c })
        .transpose(.{ .b, .c, .t });
}

pub fn decode(input: DecodeInput) DecodeOutput {
    const self = input.model;
    var x = projectIn(self, input.latents);
    x = self.conv_pre.forward(x);
    const n_up = self.ups.len;
    const n_k: usize = 3;
    for (0..n_up) |i| {
        x = self.ups[i].forward(x);
        var acc = self.resblocks[i * n_k].forward(x);
        var j: usize = 1;
        while (j < n_k) : (j += 1) {
            acc = acc.add(self.resblocks[i * n_k + j].forward(x));
        }
        x = acc.scale(1.0 / @as(f32, @floatFromInt(n_k)));
    }
    x = self.activation_post.forward(x);
    x = self.conv_post.forward(x);
    const one = zml.Tensor.scalar(1.0, x.dtype());
    const neg = zml.Tensor.scalar(-1.0, x.dtype());
    return .{ .wav = x.minimum(one).maximum(neg) };
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
            .norm = layerNorm(store.withPrefix("norm"), 1e-5),
            .w0 = conv1x1(store.withPrefix("w0")),
            .w1 = conv1x1(store.withPrefix("w1")),
            .w2 = conv1x1(store.withPrefix("w2")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(GeGluMlp)) void {
        zml.nn.LayerNorm.unloadBuffers(&self.norm);
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
            .qkv = conv1x1(store.withPrefix("qkv")),
            .q_bias = store.createTensor("q_bias", .{.d}, .replicated),
            .v_bias = store.createTensor("v_bias", .{.d}, .replicated),
            .k_bias = store.createTensor("zero_k_bias", .{.d}, .replicated),
            .proj = conv1x1(store.withPrefix("proj")),
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
            .norm1 = layerNorm(store.withPrefix("norm1"), 1e-5),
            .attn = .init(store.withPrefix("attn"), in_dim, out_dim, 8),
            .proj = conv1x1(store.withPrefix("proj")),
            .norm3 = layerNorm(store.withPrefix("norm3"), 1e-5),
            .norm2 = layerNorm(store.withPrefix("norm2"), 1e-5),
            .mlp = .init(store.withPrefix("mlp")),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(AttnProjection)) void {
        zml.nn.LayerNorm.unloadBuffers(&self.norm1);
        CausalAttn.unloadBuffers(&self.attn);
        zml.nn.Linear.unloadBuffers(&self.proj);
        zml.nn.LayerNorm.unloadBuffers(&self.norm3);
        zml.nn.LayerNorm.unloadBuffers(&self.norm2);
        GeGluMlp.unloadBuffers(&self.mlp);
    }

    pub fn forward(self: AttnProjection, x: zml.Tensor) zml.Tensor {
        const xt = x.withPartialTags(.{ .b, .s, .d });
        var y = self.proj.forward(self.norm3.forward(xt)).rename(.{ .dout = .d });
        y = y.add(self.attn.forward(self.norm1.forward(xt)));
        return y.add(self.mlp.forward(self.norm2.forward(y)));
    }
};

pub fn decodeReady(store: zml.io.TensorStore.View) bool {
    return store.hasKey("dec_in_proj.weight") and store.hasKey("decoder.conv_pre.weight_v");
}

pub fn encodeReady(store: zml.io.TensorStore.View) bool {
    return store.hasKey("encoder.block.0.weight_v") and store.hasKey("mean_proj.weight");
}

pub const EncoderModel = struct {
    conv_in: WNConv1d,
    blocks: [5]EncoderBlock,
    snake: Snake1d,
    conv_out: WNConv1d,
    pre_block: AttnProjection,
    mean_proj: zml.nn.Linear,
    cfg: Config,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) EncoderModel {
        const enc = store.withPrefix("encoder.block");
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
            .pre_block = .init(store.withPrefix("pre_block"), cfg.latent_dim, cfg.latent_channels),
            .mean_proj = conv1x1(store.withPrefix("mean_proj")),
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

pub const EncodeInput = struct {
    model: EncoderModel,
    wav: zml.Tensor,
};

pub const EncodeOutput = struct {
    latents: zml.Tensor,
};

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

pub const LoadedEncoder = struct {
    inner: EncoderModel,
    cfg: Config,

    pub fn init(store: zml.io.TensorStore.View, cfg: Config) LoadedEncoder {
        return .{ .inner = .init(store, cfg), .cfg = cfg };
    }

    pub fn loadBuffers(
        self: *const LoadedEncoder,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(EncoderModel) {
        var buffers = try zml.mem.bufferize(allocator, EncoderModel, &self.inner);
        errdefer EncoderModel.unloadBuffers(&buffers);
        var loader = try weights.initLoader(allocator, platform);
        defer loader.deinit();
        const now: std.Io.Timestamp = .now(io, .awake);
        try weights.populate(&loader, io, store, shardings, EncoderModel, &self.inner, &buffers, progress);
        log.info("loaded audio VAE encoder [{f}]", .{now.untilNow(io, .awake)});
        return buffers;
    }
};

pub const LoadedModel = struct {
    inner: Model,
    cfg: Config,

    pub fn init(allocator: std.mem.Allocator, io: std.Io, repo: std.Io.Dir, store: zml.io.TensorStore.View) !LoadedModel {
        var cfg = Config.official();
        if (try config.parseOptional(FileConfig, allocator, io, repo, "config.json")) |parsed| {
            defer parsed.deinit();
            cfg = parsed.value.resolve();
        }
        log.info("audio vae: hop={d} latent_c={d} mean0={d:.4} std0={d:.4}", .{
            cfg.hop,
            cfg.latent_channels,
            cfg.latents_mean[0],
            cfg.latents_std[0],
        });
        return .{
            .inner = try .init(allocator, store, cfg),
            .cfg = cfg,
        };
    }

    pub fn deinit(self: *LoadedModel, allocator: std.mem.Allocator) void {
        self.inner.deinit(allocator);
    }

    pub fn loadBuffers(
        self: *const LoadedModel,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *zml.io.TensorStore,
        shardings: []const zml.Sharding,
        progress: *std.Progress.Node,
    ) !zml.Bufferized(Model) {
        var buffers = try zml.mem.bufferize(allocator, Model, &self.inner);
        errdefer Model.unloadBuffers(&buffers, allocator);
        var loader = try weights.initLoader(allocator, platform);
        defer loader.deinit();
        const now: std.Io.Timestamp = .now(io, .awake);
        try weights.populate(&loader, io, store, shardings, Model, &self.inner, &buffers, progress);
        log.info("loaded audio VAE [{f}]", .{now.untilNow(io, .awake)});
        return buffers;
    }
};
