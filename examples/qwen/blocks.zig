const std = @import("std");
const stdx = @import("stdx");
const log = std.log;

const zml = @import("zml");

pub const TransformerBlock = struct {
    img_mlp: Mlp,
    attn: Attn,

    pub fn load(
        self: *const TransformerBlock,
        allocator: std.mem.Allocator,
        io: std.Io,
        platform: *const zml.Platform,
        store: *const zml.io.TensorStore,
    ) !zml.Bufferized(TransformerBlock) {
        var buffers = try zml.mem.bufferize(allocator, TransformerBlock, self);
        errdefer unloadBuffers(&buffers);

        var loader: zml.io.Loader = try .init(allocator, platform, .default);
        errdefer loader.deinit();

        try loader.load(io, TransformerBlock, self, &buffers, store, &.{}, .{});
        try loader.await(io);

        return buffers;
    }

    pub fn unloadBuffers(self: *zml.Bufferized(TransformerBlock)) void {
        Mlp.unloadBuffers(&self.img_mlp);
        Attn.unloadBuffers(&self.attn);
    }

    pub fn forward(self: TransformerBlock, x: zml.Tensor) zml.Tensor {
        const y = self.attn.forward(x);
        return self.img_mlp.forward(y);
    }
};

pub const Mlp = struct {
    gate_layer: zml.nn.Linear,
    out: zml.nn.Linear,
    proj: zml.nn.Linear,

    pub fn unloadBuffers(self: *zml.Bufferized(Mlp)) void {
        zml.nn.Linear.unloadBuffers(&self.gate_layer);
        zml.nn.Linear.unloadBuffers(&self.gate_layer);
        zml.nn.Linear.unloadBuffers(&self.gate_layer);
    }

    pub fn forward(self: Mlp, x: zml.Tensor) zml.Tensor {
        // It seems like I need to tag again after the silu(?)
        const left = self.gate_layer.forward(x, x.dtype()).silu().withTags(.{ .bs, .d, .dup });
        const right = self.proj.forward(x, x.dtype());
        const dot = left.mul(right).withTags(.{ .bs, .dup, .d });
        return self.out.forward(dot, x.dtype());
    }
};

pub const Attn = struct {
    norm_k: zml.nn.Linear,
    norm_q: zml.nn.Linear,
    to_k: zml.nn.Linear,
    to_out: zml.nn.Linear,
    to_q: zml.nn.Linear,
    to_v: zml.nn.Linear,

    pub fn unloadBuffers(self: *zml.Bufferized(Attn)) void {
        zml.nn.Linear.unloadBuffers(&self.norm_k);
        zml.nn.Linear.unloadBuffers(&self.norm_q);
        zml.nn.Linear.unloadBuffers(&self.to_k);
        zml.nn.Linear.unloadBuffers(&self.to_out);
        zml.nn.Linear.unloadBuffers(&self.to_q);
        zml.nn.Linear.unloadBuffers(&self.to_v);
    }

    pub fn forward(self: Attn, x: zml.Tensor) zml.Tensor {
        _ = self; // autofix
        return x;
    }
};
