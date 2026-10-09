const zml = @import("zml");

const RMSNorm = @import("./RMSNorm.zig");
const CompileArgs = @import("../compiler.zig").CompileArgs;

pub const Attn = @This();

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
