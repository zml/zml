const std = @import("std");
const zml = @import("zml");

const CompileArgs = @import("../../compiler.zig").CompileArgs;

const Qwen3VLTextRMSNorm = @import("./Qwen3VLTextRMSNorm.zig");

const Qwen3VLTextAttention = @This();

head_dim: i64,
q_proj: zml.nn.Linear,
k_proj: zml.nn.Linear,
v_proj: zml.nn.Linear,
o_proj: zml.nn.Linear,
q_norm: Qwen3VLTextRMSNorm,
k_norm: Qwen3VLTextRMSNorm,

pub fn unloadBuffers(self: *zml.Bufferized(Qwen3VLTextAttention)) void {
    zml.nn.Linear.unloadBuffers(&self.q_proj);
    zml.nn.Linear.unloadBuffers(&self.k_proj);
    zml.nn.Linear.unloadBuffers(&self.v_proj);
    zml.nn.Linear.unloadBuffers(&self.o_proj);
    Qwen3VLTextRMSNorm.unloadBuffers(&self.q_norm);
    Qwen3VLTextRMSNorm.unloadBuffers(&self.k_norm);
}

pub fn forward(
    self: Qwen3VLTextAttention,
    x: zml.Tensor,
    position_ids: zml.Tensor,
    cos: zml.Tensor,
    sin: zml.Tensor,
    args: CompileArgs,
) zml.Tensor {
    const input_shape = x.shape().drop(-1);
    const hidden_shape = input_shape.append(.{ .h = -1, .hd = self.head_dim });

    // std.debug.print("self.q_proj.forward(x, x.dtype()): {f}\n", .{self.q_proj.forward(x, x.dtype()).shape()}); // self.q_proj.forward(x, x.dtype()): {bs=1,seq_len=45,dup=4096,bf16}
    // std.debug.print("self.k_proj.forward(x, x.dtype()): {f}\n", .{self.k_proj.forward(x, x.dtype()).shape()}); // self.k_proj.forward(x, x.dtype()): {bs=1,seq_len=45,dup=1024,bf16}

    const q_proj = self.q_proj.forward(x, x.dtype()).reshape(hidden_shape);
    const k_proj = self.k_proj.forward(x, x.dtype()).reshape(hidden_shape);

    // std.debug.print("q_proj: {f}\n", .{q_proj.shape()}); // q_proj: {bs=1,seq_len=45,h=32,hd=128,bf16}
    // std.debug.print("k_proj: {f}\n", .{k_proj.shape()}); // k_proj: {bs=1,seq_len=45,h=8,hd=128,bf16}

    const query_states = self.q_norm.forward(q_proj); // swapAxes(1, 2);
    const key_states = self.k_norm.forward(k_proj); // swapAxes(1, 2);
    const values_states = self.v_proj.forward(x, x.dtype()).reshape(hidden_shape); // swapAxes(1, 2);

    // std.debug.print("query_states: {f}\n", .{query_states.shape()}); // query_states: {bs=1,h=32,seq_len=45,hd=128,bf16}
    // std.debug.print("key_states: {f}\n", .{key_states.shape()}); // key_states: {bs=1,h=8,seq_len=45,hd=128,bf16}
    // std.debug.print("values_states: {f}\n", .{values_states.shape()}); // values_states: {bs=1,h=8,seq_len=45,hd=128,bf16}

    // std.debug.print("cos: {f}\n", .{cos.shape()}); // cos: {1/replicated,45/replicated,128/replicated,bf16}
    // std.debug.print("sin: {f}\n", .{sin.shape()}); // sin: {1/replicated,45/replicated,128/replicated,bf16}

    const rotated_query = apply_rotary_pos_emb(query_states, cos, sin);
    const rotated_key = apply_rotary_pos_emb(key_states, cos, sin);

    // std.debug.print("rotated_query: {f}\n", .{rotated_query.shape()}); // rotated_query: {bs=1,h=32,seq_len=45,hd=128,bf16}
    // std.debug.print("rotated_key: {f}\n", .{rotated_key.shape()}); // rotated_key: {bs=1,h=8,seq_len=45,hd=128,bf16}

    const pos_idx = zml.Tensor.scalar(position_ids.dim(-1), .i32).broad(.init(.{ .b = 1 }, .i32));
    const att = zml.attention.attention(
        rotated_query.rename(.{ .bs = .b, .seq_len = .q }), // .swapAxes(1, 2),
        rotated_key.rename(.{ .bs = .b, .seq_len = .k }), // .swapAxes(1, 2),
        values_states.rename(.{ .bs = .b, .seq_len = .k }), // .swapAxes(1, 2),
        pos_idx,
        args.attention_metadata,
        args.attention_parameters,
    );

    const attn_output = att.merge(.{ .d = .{ .h, .hd } });
    return self.o_proj.forward(attn_output, x.dtype());
}

pub fn apply_rotary_pos_emb(x: zml.Tensor, cos: zml.Tensor, sin: zml.Tensor) zml.Tensor {
    // cos / sin shape: [B, 1, S, D] -> broadcast to match x: [B, H, S, D]
    const cos_x = cos.withTags(.{ .bs, .seq_len, .hd }).broad(x.shape());
    const sin_x = sin.withTags(.{ .bs, .seq_len, .hd }).broad(x.shape());

    return x.mul(cos_x).add(rotate_half(x).mul(sin_x));
}

fn rotate_half(x: zml.Tensor) zml.Tensor {
    const half_dim = @divExact(x.dim(-1), 2);
    const x1 = x.slice(-1, .{ .start = 0, .end = half_dim });
    const x2 = x.slice(-1, .{ .start = half_dim, .end = x.dim(-1) });

    // Concatenate [-x2, x1] along the last dimension
    return zml.Tensor.concatenate(&.{ x2.negate(), x1 }, -1);
}
