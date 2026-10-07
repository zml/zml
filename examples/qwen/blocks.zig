const std = @import("std");
const stdx = @import("stdx");
const log = std.log;

const zml = @import("zml");

// pos_embed -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21Rope'>
// time_text_embed -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21TimestepProjEmbeddings'>
// txt_in -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21TextProjection'>
// img_in -> <class 'torch.nn.modules.linear.Linear'>
// modulation -> <class 'torch.nn.modules.container.Sequential'>
// transformer_blocks -> <class 'torch.nn.modules.container.ModuleList'>
// norm_out -> <class 'diffusers.models.transformers.transformer_qwenimage21.QwenImage21AdaLayerNormContinuous'>
// proj_out -> <class 'torch.nn.modules.linear.Linear'>

// QwenImage21TransformerBlock(
//   (img_norm1): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//   (attn): QwenImage21Attention(
//     (to_q): Linear(in_features=4096, out_features=4096, bias=False)
//     (to_k): Linear(in_features=4096, out_features=4096, bias=False)
//     (to_v): Linear(in_features=4096, out_features=4096, bias=False)
//     (to_out): ModuleList(
//       (0): Linear(in_features=4096, out_features=4096, bias=False)
//       (1): Dropout(p=0.0, inplace=False)
//     )
//     (norm_q): RMSNorm()
//     (norm_k): RMSNorm()
//   )
//   (img_norm2): LayerNorm((4096,), eps=1e-06, elementwise_affine=False, bias=False)
//   (img_mlp): QwenImage21SwiGLUFeedForward(
//     (proj): Linear(in_features=4096, out_features=12288, bias=False)
//     (out): Linear(in_features=12288, out_features=4096, bias=False)
//     (gate_layer): Linear(in_features=4096, out_features=12288, bias=False)
//     (activation_fn): SiLU()
//   )
// )

pub const TransformerBlock = struct {
    img_norm1: zml.nn.LayerNorm,
    img_norm2: zml.nn.LayerNorm,
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
    norm_k: RMSNorm,
    norm_q: RMSNorm,
    to_k: zml.nn.Linear,
    to_out: zml.nn.Linear,
    to_q: zml.nn.Linear,
    to_v: zml.nn.Linear,

    pub fn unloadBuffers(self: *zml.Bufferized(Attn)) void {
        RMSNorm.unloadBuffers(&self.norm_k);
        RMSNorm.unloadBuffers(&self.norm_q);
        zml.nn.Linear.unloadBuffers(&self.to_k);
        zml.nn.Linear.unloadBuffers(&self.to_out);
        zml.nn.Linear.unloadBuffers(&self.to_q);
        zml.nn.Linear.unloadBuffers(&self.to_v);
    }

    // Implemented following QwenImage21AttnProcessor (which is the default Processor)
    // and following the prefill part (not decode)
    pub fn forward(self: Attn, x: zml.Tensor) zml.Tensor {
        _ = self; // autofix
        return x;
    }
};

// From diffusers/models/normalization.py, with is_torch_npu_available() == False
pub const RMSNorm = struct {
    // Removed the optional
    weight: zml.Tensor,
    /// Defaults to 1e-6
    eps: f32,
    tag: zml.Shape.Tag,

    pub fn init(weight: zml.Tensor, eps: ?f32, tag: anytype) RMSNorm {
        return .{
            .weight = weight,
            .eps = eps orelse 1e-6,
            .tag = zml.Shape.toTag(tag),
        };
    }

    pub fn unloadBuffers(self: *zml.Bufferized(RMSNorm)) void {
        self.weight.deinit();
    }

    // The basic nn.LayerNorm
    pub fn forward(self: RMSNorm, x: zml.Tensor) zml.Tensor {
        var normalized = zml.nn.rmsNorm(x, self.tag, self.eps);
        return normalized.mul(self.weight.broad(normalized.shape()));
    }
};
