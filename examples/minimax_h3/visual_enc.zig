const std = @import("std");

const zml = @import("zml");

const visual_vae = @import("visual_vae.zig");

const log = std.log.scoped(.minimax_h3_visual_enc);

pub const block_out_channels = [_]i64{ 128, 256, 256, 512, 512, 1024 };
pub const spatial_downsample = [_]i64{ 2, 2, 2, 2, 1, 1 };
pub const temporal_downsample = [_]i64{ 1, 2, 2, 1, 1, 1 };
pub const layers_per_block: usize = 2;
pub const norm_groups: i64 = 32;
pub const norm_eps: f32 = 1e-6;

fn tensorRank(store: zml.io.TensorStore.View, name: []const u8) u8 {
    var buffer: [256]u8 = undefined;
    const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ store.prefix() orelse "", name }) catch return 5;
    return if (store.store.getShape(key)) |shape| shape.rank() else 5;
}

fn convWeight(store: zml.io.TensorStore.View, name: []const u8) zml.Tensor {
    return switch (tensorRank(store, name)) {
        5 => store.createTensor(name, .{ .co, .ci, .kt, .kh, .kw }, .replicated),
        4 => store.createTensor(name, .{ .co, .ci, .kh, .kw }, .replicated),
        else => store.createTensor(name, .{ .co, .ci, .k }, .replicated),
    };
}

fn unloadConv(weight: *zml.Buffer, bias: *?zml.Buffer) void {
    weight.deinit();
    if (bias.*) |*b| b.deinit();
}

fn reflectPadBoth(x: zml.Tensor, axis: anytype, pad: i64) zml.Tensor {
    if (pad <= 0) return x;
    const n = x.dim(axis);
    if (n <= 1) {
        const first = x.slice1d(axis, .{ .start = 0, .end = 1 });
        const extra = first.broad(first.shape().setDim(axis, pad));
        return zml.Tensor.concatenate(&.{ extra, x, extra }, axis);
    }
    const left = x.slice1d(axis, .{ .start = 1, .end = 1 + pad }).reverse(.{axis});
    const right = x.slice1d(axis, .{ .start = n - 1 - pad, .end = n - 1 }).reverse(.{axis});
    return zml.Tensor.concatenate(&.{ left, x, right }, axis);
}

fn reflectPadHigh(x: zml.Tensor, axis: anytype, pad: i64) zml.Tensor {
    if (pad <= 0) return x;
    const n = x.dim(axis);
    if (n <= 1) {
        const last = x.slice1d(axis, .{ .start = n - 1, .end = n });
        return zml.Tensor.concatenate(&.{ x, last.broad(last.shape().setDim(axis, pad)) }, axis);
    }
    const tail = x.slice1d(axis, .{ .start = n - 1 - pad, .end = n - 1 }).reverse(.{axis});
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

    pub fn unloadBuffers(self: *zml.Bufferized(CausalConv3d)) void {
        unloadConv(&self.weight, &self.bias);
    }

    pub fn forward(self: CausalConv3d, x: zml.Tensor) zml.Tensor {
        var y = x.convert(.f32).withPartialTags(.{ .b, .c, .t, .h, .w });
        y = reflectPadBoth(y, .h, self.spatial_pad);
        y = reflectPadBoth(y, .w, self.spatial_pad);
        y = causalPadT(y, self.temporal_pad);
        var w = self.weight.convert(.f32);
        if (w.rank() < 5) {
            while (w.rank() < 5) w = w.appendAxes(.{.kt});
        }
        w = w.withPartialTags(.{ .co, .ci, .kt, .kh, .kw });
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

    pub fn unloadBuffers(self: *zml.Bufferized(GroupNorm)) void {
        self.weight.deinit();
        self.bias.deinit();
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
            .shortcut = if (store.hasKey("nin_shortcut.weight"))
                .init(store.withPrefix("nin_shortcut"), 1, 1, 0, 0)
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Resnet)) void {
        GroupNorm.unloadBuffers(&self.norm1);
        CausalConv3d.unloadBuffers(&self.conv1);
        GroupNorm.unloadBuffers(&self.norm2);
        CausalConv3d.unloadBuffers(&self.conv2);
        if (self.shortcut) |*s| CausalConv3d.unloadBuffers(s);
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
        const inner = if (store.hasKey("conv.weight")) store.withPrefix("conv") else store;
        return .{
            .conv = .init(inner, temporal_stride, spatial_stride, 0, 2),
            .spatial_stride = spatial_stride,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Downsample)) void {
        CausalConv3d.unloadBuffers(&self.conv);
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
        const blocks = if (store.hasKey("block.0.conv1.weight")) store.withPrefix("block") else store.withPrefix("resnets");
        return .{
            .block0 = .init(blocks.withLayer(0)),
            .block1 = .init(blocks.withLayer(1)),
            .downsample = if (temporal_factor * spatial_factor > 1)
                .init(if (store.hasKey("downsample.conv.weight") or store.hasKey("downsample.weight")) store.withPrefix("downsample") else store, temporal_factor, spatial_factor)
            else
                null,
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(DownBlock)) void {
        Resnet.unloadBuffers(&self.block0);
        Resnet.unloadBuffers(&self.block1);
        if (self.downsample) |*d| Downsample.unloadBuffers(d);
    }

    pub fn forward(self: DownBlock, x: zml.Tensor) zml.Tensor {
        var h = self.block1.forward(self.block0.forward(x));
        if (self.downsample) |d| h = d.forward(h);
        return h;
    }
};

fn encoderView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("encoder.conv_in.weight")) return store.withPrefix("encoder");
    if (store.hasKey("model.encoder.conv_in.weight")) return store.withPrefix("model.encoder");
    return store;
}

fn rootView(store: zml.io.TensorStore.View) zml.io.TensorStore.View {
    if (store.hasKey("encoder.conv_in.weight") or store.hasKey("quant_conv.weight")) return store;
    if (store.hasKey("model.encoder.conv_in.weight")) return store.withPrefix("model");
    return store;
}

pub fn ready(store: zml.io.TensorStore.View) bool {
    return store.hasKey("encoder.conv_in.weight") or
        store.hasKey("model.encoder.conv_in.weight") or
        store.hasKey("conv_in.weight");
}

pub const Model = struct {
    conv_in: CausalConv3d,
    downs: [6]DownBlock,
    norm_out: GroupNorm,
    conv_out: CausalConv3d,
    quant_conv: CausalConv3d,

    pub fn init(store_: zml.io.TensorStore.View) Model {
        const root = rootView(store_);
        const enc = encoderView(root);
        const down_root = if (enc.hasKey("down.0.block.0.conv1.weight") or enc.hasKey("down.0.resnets.0.conv1.weight"))
            enc.withPrefix("down")
        else
            enc.withPrefix("down_blocks");
        var downs: [6]DownBlock = undefined;
        for (&downs, 0..) |*block, i| {
            block.* = .init(down_root.withLayer(i), temporal_downsample[i], spatial_downsample[i]);
        }
        const quant = if (root.hasKey("quant_conv.weight")) root.withPrefix("quant_conv") else enc.withPrefix("quant_conv");
        return .{
            .conv_in = .init(enc.withPrefix("conv_in"), 1, 1, 1, 2),
            .downs = downs,
            .norm_out = .init(enc.withPrefix("norm_out")),
            .conv_out = .init(enc.withPrefix("conv_out"), 1, 1, 1, 2),
            .quant_conv = .init(quant, 1, 1, 0, 0),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(Model)) void {
        CausalConv3d.unloadBuffers(&self.conv_in);
        for (&self.downs) |*block| DownBlock.unloadBuffers(block);
        GroupNorm.unloadBuffers(&self.norm_out);
        CausalConv3d.unloadBuffers(&self.conv_out);
        CausalConv3d.unloadBuffers(&self.quant_conv);
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
    cfg: visual_vae.Config,

    pub fn init(store: zml.io.TensorStore.View, cfg: visual_vae.Config) LoadedModel {
        return .{ .inner = .init(store), .cfg = cfg };
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
        errdefer Model.unloadBuffers(&buffers);
        var loader: zml.io.Loader = try .init(allocator, platform, .{
            .dma_chunks = 32,
            .dma_chunk_size = 256 * zml.MiB,
            .parallelism = 16,
        });
        defer loader.deinit();
        const now: std.Io.Timestamp = .now(io, .awake);
        loader.load(io, Model, &self.inner, &buffers, store, shardings, .{ .progress = progress });
        try loader.await(io);
        log.info("loaded visual VAE encoder [{f}]", .{now.untilNow(io, .awake)});
        return buffers;
    }
};
