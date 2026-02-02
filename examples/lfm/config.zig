/// Copied from https://github.com/huggingface/transformers/blob/0997c2f2ab08c32c8e2f90aaad06e29a7108535b/src/transformers/models/lfm2/configuration_lfm2.py
///
/// Configuration parsed from LFM2-style `config.json`.
/// Notes:
/// - `layer_types` is expected to be fully specified (HF can derive it from `full_attn_idxs`).
/// - HF may use `block_ff_dim` as the effective MLP size; keep `intermediate_size` and `block_ff_dim`
///   in sync in the JSON if you rely on both.
/// - HF uses `tie_embedding` as an alias for `tie_word_embeddings`.
/// - `eos_token_id` can differ from the HF default (e.g. 7 in LFM2 configs).
pub const Config = struct {
    /// Model class names in the source config.
    architectures: []const []const u8,
    /// Whether to auto-adjust the FFN dimension.
    block_auto_adjust_ff_dim: bool,
    /// Base block hidden dimension.
    block_dim: u32,
    /// Block FFN dimension (HF uses this as the effective `intermediate_size` when present).
    block_ff_dim: u32,
    /// Multiplier for FFN dimension.
    block_ffn_dim_multiplier: f32,
    /// MLP init scale in block config.
    block_mlp_init_scale: f32,
    /// Multiple for intermediate size rounding.
    block_multiple_of: u32,
    /// RMS norm epsilon within blocks.
    block_norm_eps: f32,
    /// Output init scale in block config.
    block_out_init_scale: f32,
    /// Whether to use SwiGLU in blocks.
    block_use_swiglu: bool,
    /// Whether to use Xavier init in blocks.
    block_use_xavier_init: bool,
    /// BOS token id.
    bos_token_id: u32,
    /// Convolution cache length.
    conv_L_cache: u32,
    /// Whether conv layers use bias.
    conv_bias: bool,
    /// Convolution dimension.
    conv_dim: u32,
    /// Whether to use Xavier init in conv layers.
    conv_use_xavier_init: bool,
    /// Serialized dtype (e.g. "bfloat16").
    dtype: []const u8,
    /// EOS token id.
    eos_token_id: u32,
    /// Hidden size of the model.
    hidden_size: u32,
    /// Weight init stddev.
    initializer_range: f32,
    /// FFN hidden dimension (HF may override with `block_ff_dim`).
    intermediate_size: u32,
    /// Per-layer operator types (e.g. "conv" or "full_attention").
    layer_types: []const []const u8,
    /// Max sequence length supported by RoPE.
    max_position_embeddings: u32,
    /// Model type tag, expected "lfm2".
    model_type: []const u8,
    /// RMS norm epsilon.
    norm_eps: f32,
    /// Number of attention heads.
    num_attention_heads: u32,
    /// Total number of heads (may duplicate `num_attention_heads`).
    num_heads: u32,
    /// Number of transformer layers.
    num_hidden_layers: u32,
    /// Number of key/value heads for GQA/MQA.
    num_key_value_heads: u32,
    /// PAD token id.
    pad_token_id: u32,
    /// RoPE theta.
    rope_theta: f32,
    /// Whether to tie embeddings (aka `tie_word_embeddings`).
    tie_embedding: bool,
    /// Transformers version string.
    transformers_version: []const u8,
    /// Whether to use KV cache.
    use_cache: bool,
    /// Whether to use positional encodings.
    use_pos_enc: bool,
    /// Vocabulary size.
    vocab_size: u32,
};
