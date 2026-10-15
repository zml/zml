const zml = @import("zml");

pub const CompileArgs = struct {
    seq_len: usize,
    num_attention_heads: usize,
    head_dim: usize,
    rope_theta: usize,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,
};
