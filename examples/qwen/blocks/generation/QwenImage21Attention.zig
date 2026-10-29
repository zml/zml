const std = @import("std");
const zml = @import("zml");

const RMSNorm = @import("./RMSNorm.zig");
const CompileArgs = @import("../../compiler.zig").CompileArgs;

// QwenImage21Attention(
//   (to_q): Linear(in_features=4096, out_features=4096, bias=False)
//   (to_k): Linear(in_features=4096, out_features=4096, bias=False)
//   (to_v): Linear(in_features=4096, out_features=4096, bias=False)
//   (to_out): ModuleList(
//     (0): Linear(in_features=4096, out_features=4096, bias=False)
//     (1): Dropout(p=0.0, inplace=False)
//   )
//   (norm_q): RMSNorm()
//   (norm_k): RMSNorm()
// )

pub const QwenImage21Attention = @This();

num_attention_heads: i64,
norm_k: RMSNorm,
norm_q: RMSNorm,
to_k: zml.nn.Linear,
to_out: zml.nn.Linear,
to_q: zml.nn.Linear,
to_v: zml.nn.Linear,

pub fn unloadBuffers(self: *zml.Bufferized(QwenImage21Attention)) void {
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
    self: QwenImage21Attention,
    x: zml.Tensor,
    rotary_emb: zml.Tensor,
    text_len: i64,
    args: CompileArgs,
) zml.Tensor {
    const prepared = self.prepare(x, rotary_emb);
    const q = prepared.q.rename(.{ .bs = .b, .seq_len = .q });
    const k = prepared.k.rename(.{ .bs = .b, .seq_len = .k });
    const v = prepared.v.rename(.{ .bs = .b, .seq_len = .k });

    const text_k = k.slice(.k, .{ .end = text_len });
    const text_v = v.slice(.k, .{ .end = text_len });
    const causal_mask = zml.nn.causalAttnMask(.{ .q = text_len, .k = text_len }, q.dtype(), null);
    const text_out = zml.nn.sdpa(q.slice(.q, .{ .end = text_len }), text_k, text_v, .{ .attn_mask = causal_mask });

    const image_out = full_attention(q.slice(.q, .{ .start = text_len }), k, v, args);

    const out = zml.Tensor.concatenate(&.{
        text_out.transpose(.{ .b, .q, .h, .hd }),
        image_out.transpose(.{ .b, .q, .h, .hd }),
    }, .q).merge(.{ .d = .{ .h, .hd } }).rename(.{ .b = .bs, .q = .seq_len });

    return self.to_out.forward(out, x.dtype()).withTags(.{ .bs, .seq_len, .d });
}

fn full_attention(q: zml.Tensor, k: zml.Tensor, v: zml.Tensor, args: CompileArgs) zml.Tensor {
    const token_index = zml.Tensor.scalar(0, .i32).broad(.init(.{ .b = q.dim(.b) }, .i32));
    return zml.attention.attentionOpts(q, k, v, token_index, args.attention_metadata, args.attention_parameters, .{ .causal = false });
}

fn apply_rotary_emb(x: zml.Tensor, freqs_cis: zml.Tensor) zml.Tensor {
    const x_real, const x_imag = zml.nn.splitRealImg(x.convert(.f32), .interleaved);
    const freqs = freqs_cis.withTags(.{ .seq_len, .hd });
    const cos = freqs.real().broad(x_real.shape());
    const sin = freqs.imag().broad(x_real.shape());

    const y_real = x_real.mul(cos).sub(x_imag.mul(sin));
    const y_imag = x_real.mul(sin).add(x_imag.mul(cos));
    return zml.nn.mergeRealImg(y_real, y_imag, .interleaved).convert(x.dtype());
}

const PreparedQKV = struct { q: zml.Tensor, k: zml.Tensor, v: zml.Tensor };

// _qwenimage21_prepare_qkv from diffusers/models/transformers/transformer_qwenimage21.py
fn prepare(self: QwenImage21Attention, x: zml.Tensor, rotary_emb: zml.Tensor) PreparedQKV {
    const query = self.split_heads(self.to_q.forward(x, x.dtype()));
    const key = self.split_heads(self.to_k.forward(x, x.dtype()));
    const value = self.split_heads(self.to_v.forward(x, x.dtype()));

    const normed_query = self.norm_q.forward(query).convert(value.dtype());
    const normed_key = self.norm_k.forward(key).convert(value.dtype());

    return .{
        .q = apply_rotary_emb(normed_query, rotary_emb),
        .k = apply_rotary_emb(normed_key, rotary_emb),
        .v = value,
    };
}

fn split_heads(self: QwenImage21Attention, x: zml.Tensor) zml.Tensor {
    return x.withTags(.{ .bs, .seq_len, .d }).splitAxis(.d, .{ .h = self.num_attention_heads, .hd = .auto });
}
