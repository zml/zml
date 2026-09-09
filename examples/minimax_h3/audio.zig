//! Audio VAE decoder.
//!
//!   1. denormalize latents (`v ← v * std + mean`)
//!   2. reshape packed `(2·T, C)` left/right rows → `(2, C, T)`
//!   3. proj → conv_pre → 7× (upsample + AMP residual average) → conv_post
//!   4. clamp to `[-1, 1]`, interleave stereo

const std = @import("std");
const zml = @import("zml");
const config = @import("config.zig");
const ops = @import("ops.zig");

const AudioConfig = config.AudioConfig;
const load = ops.load;
const applyLatentNorm = ops.applyLatentNorm;
const Run = ops.Run;

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
    var x = projectIn(self, input.latents);
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

/// Packed DiT audio is `(2 * T, C)` left then right. VAE wants `(2, C, T)`.
fn audioRowsToBct(dst: []f32, rows: []const f32, channels: u32, t: u32) void {
    const ch: usize = channels;
    const tt: usize = t;
    for (0..2) |ear| {
        const src = rows[ear * tt * ch ..][0 .. tt * ch];
        const out = dst[ear * ch * tt ..][0 .. ch * tt];
        for (0..tt) |ti| {
            for (0..ch) |c| {
                out[c * tt + ti] = src[ti * ch + c];
            }
        }
    }
}

fn interleaveStereo(allocator: std.mem.Allocator, left: []const f32, right: []const f32) ![]f32 {
    const out = try allocator.alloc(f32, left.len * 2);
    for (left, right, 0..) |l, r, i| {
        out[i * 2] = l;
        out[i * 2 + 1] = r;
    }
    return out;
}

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
            .latents = .init(.{ .b = 2, .c = self.inner.cfg.latent_channels, .t = geo.audio_t }, .f32),
        }});
    }

    /// Denoised audio tokens → interleaved stereo f32 in `[-1, 1]`.
    pub fn decodeAudio(
        self: *const AudioVae,
        run: *const Run,
        store: *zml.io.TensorStore,
        geo: config.Geometry,
        packed_audio: []f32,
    ) ![]f32 {
        const compiled = if (self.compiled) |*c| c else return error.NotCompiled;
        const cfg = self.inner.cfg;
        const channels: u32 = @intCast(cfg.latent_channels);
        applyLatentNorm(packed_audio, &cfg.latents_mean, &cfg.latents_std);
        const t = geo.audio_t;
        const batch = try run.allocator.alloc(f32, 2 * @as(usize, channels) * t);
        defer run.allocator.free(batch);
        audioRowsToBct(batch, packed_audio, channels, t);

        var bufs = try load(run, store, Decoder, &self.inner, null);
        defer Decoder.unloadBuffers(&bufs, run.allocator);
        var runner = try zml.FnExe(decode).Runner(.{.model}).init(compiled, run.allocator, .{ .model = bufs });
        defer runner.deinit(run.allocator);

        var latent_buf = try zml.Buffer.fromBytes(
            run.io,
            run.platform,
            .init(.{ .b = 2, .c = cfg.latent_channels, .t = t }, .f32),
            .replicated,
            std.mem.sliceAsBytes(batch),
        );
        defer latent_buf.deinit();

        var wav: zml.Buffer = undefined;
        runner.run(run.io, .{
            .inputs = .{ .latents = latent_buf },
            .outputs = .{ .wav = &wav },
            .opts = .{ .wait = true },
        });
        defer wav.deinit();

        const samples = t * cfg.hop();
        const host_pcm = try run.allocator.alloc(f32, 2 * samples);
        errdefer run.allocator.free(host_pcm);
        try wav.toSlice(run.io, .init(zml.Shape.init(.{ .b = 2, .c = 1, .t = samples }, .f32), std.mem.sliceAsBytes(host_pcm)));
        const interleaved = try interleaveStereo(run.allocator, host_pcm[0..samples], host_pcm[samples..]);
        run.allocator.free(host_pcm);
        return interleaved;
    }
};
