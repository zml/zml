pub const TransformerBlock = @import("./blocks/TransformerBlock.zig");
pub const Attn = @import("./blocks/Attn.zig");
pub const Mlp = @import("./blocks/Mlp.zig");

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
