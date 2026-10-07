const std = @import("std");
const stdx = @import("stdx");
const log = std.log;

const zml = @import("zml");
const CompileArgs = @import("./compiler.zig").CompileArgs;

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
    /// TODO: remove seq_len once it's computed
    seq_len: i64,
    num_attention_heads: i64,
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
    pub fn forward(
        self: Attn,
        x: zml.Tensor,
        rotary_emb: zml.Tensor,
        args: CompileArgs,
    ) zml.Tensor {
        const prepared = self.prepare(x, rotary_emb);
        const query = prepared.q;
        const key = prepared.k;
        const value = prepared.v;

        const pos_idx = zml.Tensor.scalar(0, .i32);
        const att = zml.attention.attention(
            query.rename(.{ .bs = .b, .seq_len = .q, .d = .h }),
            key.rename(.{ .bs = .b, .seq_len = .k, .d = .h }),
            value.rename(.{ .bs = .b, .seq_len = .k, .d = .h }),
            pos_idx.broad(.init(.{ .b = 1 }, .i32)),
            args.attention_metadata,
            args.attention_parameters,
        );
        return att.merge(.{ .d = .{ .h, .hd } });
    }

    // apply_rotary_emb_qwen from diffusers/models/transformers/transformer_qwenimage21.py in the use_real=False path
    fn apply_rotary_emb_qwen(x: zml.Tensor, freqs_cis: zml.Tensor) zml.Tensor {
        return zml.nn.rope(x, .{ .inv_freq_pos = freqs_cis.rename(.{ .d = .hd }) });
    }

    const PreparedQKV = struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, seq_len_q: i64 };
    // _qwenimage21_prepare_qkv from diffusers/models/transformers/transformer_qwenimage21.py
    fn prepare(self: Attn, x: zml.Tensor, rotary_emb: zml.Tensor) PreparedQKV {
        const num_attention_heads: usize = 32;

        const query_flat = self.to_q.forward(x, x.dtype());
        const key_flat = self.to_k.forward(x, x.dtype());
        const value_flat = self.to_v.forward(x, x.dtype());

        const query_unflat = query_flat.unflatten(2, num_attention_heads).withTags(.{ .bs, .seq_len, .d, .hd });
        const key_unflat = key_flat.unflatten(2, num_attention_heads).withTags(.{ .bs, .seq_len, .d, .hd });
        const value_unflat = value_flat.unflatten(2, num_attention_heads).withTags(.{ .bs, .seq_len, .d, .hd });

        const query = self.norm_q.forward(query_unflat).convert(value_unflat.dtype());
        const key = self.norm_k.forward(key_unflat).convert(value_unflat.dtype());

        const rotated_query = apply_rotary_emb_qwen(query, rotary_emb);
        const rotated_key = apply_rotary_emb_qwen(key, rotary_emb);

        const seq_len_q = query.shape().dim(2); // Check this 2
        return PreparedQKV{ .q = rotated_query, .k = rotated_key, .v = value_unflat, .seq_len_q = seq_len_q };
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
        return normalized.mul(self.weight.withTags(.{.hd}).broad(normalized.shape()));
    }
};
