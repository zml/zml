const zml = @import("zml");

pub const CompileArgs = struct {
    width: usize,
    height: usize,
    vae_scale_factor: usize,

    // Transformer
    in_channels: usize,
    num_inference_steps: i64,

    // Attention
    num_attention_heads: usize,
    head_dim: usize,
    rope_theta: usize,
    attention_metadata: zml.attention.Metadata,
    attention_parameters: zml.attention.Parameters,

    base_image_seq_len: ?i64 = null,
    max_image_seq_len: ?i64 = null,
    base_shift: ?f32 = null,
    max_shift: ?f32 = null,
};
