const std = @import("std");

const llama = @import("llama.zig");

const c = zml.c;
const asynk = @import("async");
const dialects = @import("mlir/dialects");
const mlir = @import("mlir");
const stdx = @import("stdx");
const upb = zml.upb;
const zml = @import("zml");
const pjrt = zml.pjrt;
const Shape = zml.Shape;
const EnumLiteral = @TypeOf(.enum_literal);
const Platform = zml.Platform;

const Dims = stdx.BoundedArray(i64, zml.Shape.MAX_RANK);
const log = std.log.scoped(.main);

const safetensor =
    \\{
    \\  "metadata": {
    \\    "total_size": 16060522496
    \\  },
    \\  "weight_map": {
    \\    "lm_head.weight": "model-00004-of-00004.safetensors",
    \\    "model.embed_tokens.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.0.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.1.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.10.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.10.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.11.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.12.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.13.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.14.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.15.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.16.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.17.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.18.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.19.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.2.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.2.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.20.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.20.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.20.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.20.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.20.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.20.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.20.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.20.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.20.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.21.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.21.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.22.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.23.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.24.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.25.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.26.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.27.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.28.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.29.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.3.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.3.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.30.input_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.mlp.down_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.post_attention_layernorm.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.30.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.31.input_layernorm.weight": "model-00004-of-00004.safetensors",
    \\    "model.layers.31.mlp.down_proj.weight": "model-00004-of-00004.safetensors",
    \\    "model.layers.31.mlp.gate_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.31.mlp.up_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.31.post_attention_layernorm.weight": "model-00004-of-00004.safetensors",
    \\    "model.layers.31.self_attn.k_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.31.self_attn.o_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.31.self_attn.q_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.31.self_attn.v_proj.weight": "model-00003-of-00004.safetensors",
    \\    "model.layers.4.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.4.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.5.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.6.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.7.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.input_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.mlp.down_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.mlp.gate_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.mlp.up_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.post_attention_layernorm.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.self_attn.k_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.self_attn.o_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.self_attn.q_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.8.self_attn.v_proj.weight": "model-00001-of-00004.safetensors",
    \\    "model.layers.9.input_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.mlp.down_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.mlp.gate_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.mlp.up_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.post_attention_layernorm.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.self_attn.k_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.self_attn.o_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.self_attn.q_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.layers.9.self_attn.v_proj.weight": "model-00002-of-00004.safetensors",
    \\    "model.norm.weight": "model-00004-of-00004.safetensors"
    \\  }
    \\}
;

pub const std_options: std.Options = .{
    .log_level = .debug,
    .log_scope_levels = &.{
        .{ .scope = .@"zml/async", .level = .info },
    },
};

var global_compilation_context: ?*CompilationContext = null;

const AttributeList = stdx.BoundedArray(mlir.NamedAttribute, 3);

pub const TensorOriginType = enum {
    argument,
    value,
};

pub const TensorOrigin = union(TensorOriginType) {
    argument: void,
    value: *const mlir.Value,
};

pub const Tensor = struct {
    var current_id: std.atomic.Value(usize) = .{ .raw = 1 };
    pub const MAX_RANK = Shape.MAX_RANK;

    id: usize,
    auto_broadcast: bool = false,
    _shape: zml.Shape,
    tensor_origin: TensorOrigin = .{ .argument = {} },

    fn mlirType(self: Tensor, mlir_ctx: *mlir.Context) *const mlir.Type {
        return mlir.rankedTensorType(
            self.dims(),
            zml.mlir.Type.fromDType(mlir_ctx, self.dtype()),
        );
    }

    fn mlirCtx() *mlir.Context {
        return CompilationContext.current().mlir_ctx;
    }

    fn currentBlock() *mlir.Block {
        return CompilationContext.current().currentScope().block;
    }

    pub fn init(shape_: zml.Shape) Tensor {
        return .{ .id = Tensor.current_id.fetchAdd(1, .seq_cst), ._shape = shape_ };
    }

    /// Returns the shape of a Tensor.
    pub fn shape(self: Tensor) Shape {
        return self._shape;
    }

    /// Returns the datatype of a Tensor.
    pub fn dtype(self: Tensor) zml.DataType {
        return self._shape.dtype();
    }

    /// Returns the rank of a Tensor.
    pub fn rank(self: Tensor) u4 {
        return self._shape.rank();
    }

    /// Returns the number of element of a Tensor.
    pub fn count(self: Tensor) usize {
        return self._shape.count();
    }

    /// Returns the size in bytes of a Tensor.
    pub fn byteSize(self: Tensor) usize {
        return self._shape.byteSize();
    }

    /// Returns the dimension of axis 'axis_'.
    ///
    /// 'axis_' can be an integer or a tag.
    pub fn dim(self: Tensor, axis_: anytype) i64 {
        return self._shape.dim(axis_);
    }

    /// Returns the dimensions of a Tensor as a slice.
    pub fn dims(self: *const Tensor) []const i64 {
        return self._shape.dims();
    }

    /// Returns the index of axis 'axis_'.
    ///
    /// 'axis_' can be an integer or a tag.
    pub fn axis(self: Tensor, axis_: anytype) u3 {
        return self._shape.axis(axis_);
    }

    /// Returns the indices of each of the given axes.
    ///
    /// 'axis_' can be an integer or a tag.
    pub fn axes(self: Tensor, axes_: anytype) stdx.BoundedArray(u3, Tensor.MAX_RANK) {
        return self._shape.axes(axes_);
    }

    /// Returns a Tensor tagged with the tags in 'tagz'.
    pub fn withTags(self: Tensor, tagz: anytype) Tensor {
        var res = self;
        res._shape = self._shape.withTags(tagz);
        return res;
    }

    /// Returns a Tensor tagged partially with the tags in 'tagz'.
    ///
    /// If 'tagz' is of length n, the n last dimensions of the Tensor will be tagged.
    pub fn withPartialTags(self: Tensor, tagz: anytype) Tensor {
        var res = self;
        res._shape = self._shape.withPartialTags(tagz);
        return res;
    }

    /// Returns a Tensor with new tag names.
    pub fn rename(self: Tensor, renames: anytype) Tensor {
        var res = self;
        res._shape = self._shape.rename(renames);
        return res;
    }

    pub fn renameAxis(self: Tensor, ax: i8, name: EnumLiteral) Tensor {
        var res = self;
        res._shape._tags.set(self.axis(ax), @tagName(name).ptr);
        return res;
    }

    pub fn autoBroadcast(self: Tensor) Tensor {
        var ret = self;
        ret.auto_broadcast = true;
        return ret;
    }

    pub fn matmul(lhs: Tensor, rhs: Tensor) Tensor {
        //stdx.debug.assert(lhs.rank() >= 1 and rhs.rank() >= 1, "Can't matmul({f}, {f}) ! The two tensors need to have at least rank 1.", .{ lhs, rhs });

        const contracting = [_][2]i8{.{ -1, if (rhs.rank() >= 2) rhs.rank() - 2 else 0 }};
        if (lhs.rank() == 1 or rhs.rank() <= 2) {
            // When lhs is a vector or rhs is small the torch semantics match the dot_general semantics and life is easy.
            return lhs.dotGeneral(rhs, &contracting, &.{});
        }

        //stdx.debug.assert(lhs.rank() == 2, "Can't matmul({f}, {f}) ! One of the two tensors need to have a rank less than 2.", .{ lhs, rhs });

        // Pytorch treats the extra dimensions of rhs has batching dimensions,
        // and implicitly broadcast lhs along those.
        // We make this broadcasting explicit.
        var left_shape = rhs.shape();
        left_shape._dims.set(left_shape.axis(-2), lhs.dim(-2));
        left_shape._tags.set(left_shape.axis(-2), lhs.shape().tag(-2));
        left_shape._dims.set(left_shape.axis(-1), lhs.dim(-1));
        left_shape._tags.set(left_shape.axis(-1), lhs.shape().tag(-1));
        const lhs_broad = lhs.broadcastLeft(left_shape);

        const n_batching_axes = rhs.rank() - lhs.rank();
        var batching: [MAX_RANK][2]i8 = undefined;
        for (0..n_batching_axes) |i| {
            batching[i] = .{ @intCast(i), @intCast(i) };
        }
        return lhs_broad.dotGeneral(rhs, &contracting, batching[0..n_batching_axes]);
    }

    const ArgsKind = enum {
        simple,
        contracting_only,
        full,
    };

    fn isFullArgsKind(comptime T: type) bool {
        return std.meta.fieldIndex(T, "contracting") != null and std.meta.fieldIndex(T, "batching") != null;
    }

    fn getArgsKind(comptime T: type) ArgsKind {
        const type_info = @typeInfo(T);
        return switch (type_info) {
            .enum_literal => .simple,
            .@"struct" => if (isFullArgsKind(T)) .full else .contracting_only,
            else => unreachable,
        };
    }

    pub fn dot(lhs: Tensor, rhs: Tensor, args: anytype) Tensor {
        //const args_kind = getArgsKind(@TypeOf(args));
        //std.debug.print("{}\n", .{args_kind});
        const lhs_contracting_dim: i8 = @intCast(lhs.shape().hasTag(args).?);
        const rhs_contracting_dim: i8 = @intCast(rhs.shape().hasTag(args).?);

        var batching_axes: stdx.BoundedArray([2]i8, Shape.MAX_RANK) = .{};
        for (0..lhs.rank()) |lhs_tag_index| {
            const lhs_tag = lhs.shape().tag(lhs_tag_index);
            if (lhs_tag == Shape.toTag(args)) continue;
            if (rhs.shape().hasTag(lhs_tag)) |rhs_tag_index| {
                batching_axes.appendAssumeCapacity(.{ @intCast(lhs_tag_index), @intCast(rhs_tag_index) });
            }
        }
        return lhs.dotGeneral(rhs, &.{.{ lhs_contracting_dim, rhs_contracting_dim }}, batching_axes.slice());
    }

    pub fn dotGeneral(
        lhs: Tensor,
        rhs: Tensor,
        contracting_axes: []const [2]i8,
        batching_axes: []const [2]i8,
    ) Tensor {
        stdx.debug.assert(lhs.dtype() == rhs.dtype(), "dotGeneral expects tensors to be of the same type, got {} and {}", .{ lhs.dtype(), rhs.dtype() });

        const Axes = stdx.BoundedArray(i64, MAX_RANK);

        var res_shape: Shape = .{ ._dtype = lhs.dtype() };
        // Validate batching axes
        var lhs_batching_axes: Axes = .{};
        var rhs_batching_axes: Axes = .{};
        for (batching_axes) |b_axes| {
            const l, const r = b_axes;
            stdx.debug.assert(lhs._shape.dim(l) == rhs._shape.dim(r), "dotGeneral expects batching dimensions to be equal, got {} and {} in {f} and {f}", .{ l, r, lhs, rhs });
            var t = lhs._shape.tag(l);
            if (t == Shape.TagUnknown) t = rhs._shape.tag(r);
            res_shape = res_shape.appendDim(lhs._shape.dim(l), t);
            lhs_batching_axes.appendAssumeCapacity(lhs._shape.axis(l));
            rhs_batching_axes.appendAssumeCapacity(rhs._shape.axis(r));
        }

        // Validate contracting axes
        var lhs_contracting_axes: Axes = .{};
        var rhs_contracting_axes: Axes = .{};
        for (contracting_axes) |c_axes| {
            const l, const r = c_axes;
            stdx.debug.assert(lhs._shape.dim(l) == rhs._shape.dim(r), "dotGeneral expects contracting dimensions to be equal, got {} and {} in {f} and {f}", .{ l, r, lhs, rhs });
            lhs_contracting_axes.appendAssumeCapacity(lhs._shape.axis(l));
            rhs_contracting_axes.appendAssumeCapacity(rhs._shape.axis(r));
        }

        // Result shape is obtained by concatenating batching dimensions, (already done)
        // then dimensions from lhs axes that aren't contracting nor batching,
        // then dimensions from rhs axes that aren't contracting nor batching.
        for (0..lhs.rank()) |l| {
            if (std.mem.indexOfScalar(i64, lhs_contracting_axes.constSlice(), @intCast(l))) |_| {
                continue;
            }
            if (std.mem.indexOfScalar(i64, lhs_batching_axes.constSlice(), @intCast(l))) |_| {
                continue;
            }
            res_shape = res_shape.appendDim(lhs._shape.dim(l), lhs._shape.tag(l));
        }
        for (0..rhs.rank()) |r| {
            if (std.mem.indexOfScalar(i64, rhs_contracting_axes.constSlice(), @intCast(r))) |_| {
                continue;
            }
            if (std.mem.indexOfScalar(i64, rhs_batching_axes.constSlice(), @intCast(r))) |_| {
                continue;
            }
            res_shape = res_shape.appendDim(rhs._shape.dim(r), rhs._shape.tag(r));
        }

        const mlir_ctx = lhs.getContext().mlir_ctx;
        //const loc = lhs.getContext().location(@src(), "dot({f},{f},contracting={any},batching={any}", .{ lhs, rhs, contracting_axes, batching_axes });
        const loc = mlir.Location.unknown(mlirCtx());
        const op = dialects.stablehlo.dot_general(
            mlir_ctx,
            lhs.value(),
            rhs.value(),
            mlir.rankedTensorType(res_shape.dims(), zml.mlir.Type.fromDType(mlir_ctx, res_shape.dtype())),
            .{
                .lhs_batching_dimensions = lhs_batching_axes.constSlice(),
                .rhs_batching_dimensions = rhs_batching_axes.constSlice(),
                .lhs_contracting_dimensions = lhs_contracting_axes.constSlice(),
                .rhs_contracting_dimensions = rhs_contracting_axes.constSlice(),
                .dot_precision = .fast,
            },
            loc,
        ).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    pub fn _result(sh: Shape, val: *const mlir.Value) Tensor {
        const res: Tensor = .{
            ._shape = sh,
            .tensor_origin = .{ .value = val },
            .id = Tensor.current_id.fetchAdd(1, .seq_cst),
        };

        //if (builtin.mode == .Debug) {
        //    // Check that the MLIR value actually have the same shape.
        //    const other = fromMlirValue(val);
        //    stdx.debug.internalAssert(sh.eql(other._shape), "Created a {f} from Mlir value but expected {f}", .{ other._shape, res._shape });
        //}

        return res;
    }

    pub fn constant(val: zml.DataType.Value) Tensor {
        const op = dialects.stablehlo.constant(
            mlirCtx(),
            &.{},
            zml.mlir.Type.fromDType(mlirCtx(), val.dtype()),
            val.constSlice(),
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(.init(&.{}, val.dtype()), op.result(0)).autoBroadcast();
    }

    pub fn add(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "add", dialects.stablehlo.add)(self, other);
    }

    pub fn sub(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "sub", dialects.stablehlo.subtract)(self, other);
    }

    pub fn mul(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "mul", dialects.stablehlo.multiply)(self, other);
    }

    /// Returns a Tensor containing the element-wise division of the input Tensors.
    pub fn div(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "div", dialects.stablehlo.divide)(self, other);
    }

    pub fn maximum(self: Tensor, other: Tensor) Tensor {
        return binaryOp(@src(), "maximum", dialects.stablehlo.maximum)(self, other);
    }

    /// Returns a Tensor containing the element-wise cosine of the input Tensor.
    pub fn cos(self: Tensor) Tensor {
        const cosine_op = dialects.stablehlo.cosine(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, cosine_op.result(0));
    }

    /// Returns a Tensor containing the element-wise sine of the input Tensor.
    pub fn sin(self: Tensor) Tensor {
        const sine_op = dialects.stablehlo.sine(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, sine_op.result(0));
    }

    pub fn relu(self: Tensor) Tensor {
        return self.maximum(.constant(self.dtype().zero()));
    }

    /// Flattens the given axis and the next one, into one new axis.
    pub fn flatten(self: Tensor, axis_: anytype) Tensor {
        // TODO: move to torch.zig, this is equivalent to merge
        const old_shape = self._shape;
        const a = self.axis(axis_);
        // stdx.debug.assert(a + 1 < self.rank(), "Can't flatten {} on the last axis {}.", .{ self, axis });
        const new_shape = old_shape.remove(a + 1).set(a, old_shape.dim(a) * old_shape.dim(a + 1));

        const tensor_type = mlir.rankedTensorType(new_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), new_shape.dtype()));
        const reshaped_val = dialects.stablehlo.reshape(
            mlirCtx(),
            self.value(),
            tensor_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        // log.debug("flatten({d}, {d}) -> {d}", .{ self.dims(), axis_, new_shape[0 .. self.rank() - 1] });
        return _result(new_shape, reshaped_val.result(0));
    }

    pub fn flattenAll(self: Tensor) Tensor {
        // TODO: rename to just flatten, once flatten is moved to torch
        return self.reshape(.{self.count()});
    }

    pub fn broadcast(self: Tensor, output_shape: Shape, axes_: []const i64) Tensor {
        stdx.debug.assert(axes_.len == self.rank(), "broadcast expects axes_ to map all axes from self to axes of the output shape, got broadcast({f}, {f}, {any})", .{ self, output_shape, axes_ });
        for (0.., axes_) |self_ax, other_ax| {
            const d = self.dim(self_ax);
            stdx.debug.assert(d == 1 or d == output_shape.dim(other_ax), "broadcast expects shape axes to either be 1-sized or to match the target size. got broadcast({f}, {f}, {any}), error on self axis {d} mapping to other axis {d}", .{ self, output_shape, axes_, self_ax, other_ax });
        }

        const res_shape = output_shape.withDtype(self.dtype());
        if (std.mem.eql(i64, self.dims(), output_shape.dims())) {
            // No broadcast needed. We don't emit a new stablehlo value
            // but we propagate output_shape tags.
            return _result(res_shape, self.value());
        }
        const ctx = self.getContext();
        const result_type = mlir.rankedTensorType(
            res_shape.dims(),
            zml.mlir.Type.fromDType(ctx.mlir_ctx, res_shape.dtype()),
        );
        const broadcast_op = dialects.stablehlo.broadcast_in_dim(ctx.mlir_ctx, self.value(), axes_, result_type, .unknown(ctx.mlir_ctx)).appendTo(currentBlock());
        return _result(res_shape, broadcast_op.result(0));
    }

    pub fn broad(self: Tensor, other: Shape) Tensor {
        // TODO: broad is too restrictive because sometime you only want to specify one specific axis
        // Note: if you code below, make sure to update Shape.canBroadcastTo.
        stdx.debug.assert(self._shape.canBroadcastTo(other), "Can't broadcast {f} to {f}", .{ self, other });

        // Already the right shape
        if (std.mem.eql(i64, self.dims(), other.dims())) return self;

        // Non ambiguous broadcasting
        // TODO: broad is error prone because of this:
        // it will happily broadcast .{ .a = 10, .b = 1 } to .{ .b = 10, .a = 5 }
        if (self._shape.rank() == 0 or self._shape.rank() == other.rank()) {
            const all_axes = [MAX_RANK]i64{ 0, 1, 2, 3, 4, 5, 6, 7 };
            return self.broadcast(other, all_axes[0..self.rank()]);
        }

        // check that each axis of self maps to an axis of other
        var axes_: stdx.BoundedArray(i64, MAX_RANK) = .{};
        for (self._shape.tags()) |t| {
            axes_.appendAssumeCapacity(@intCast(other.axis(t)));
        }
        return self.broadcast(other, axes_.constSlice());
    }

    pub fn broadcastLeft(self: Tensor, output_shape: Shape) Tensor {
        stdx.debug.assert(self.rank() <= output_shape.rank(), "broadcastLeft expects tensor rank to be less than output tensor rank, got {d} and {d}", .{ self.rank(), output_shape.rank() });

        const a = output_shape.rank() - self.rank();
        if (self.rank() == output_shape.rank() and std.mem.eql(i64, self.dims(), output_shape.dims())) {
            return self;
        }

        return self.broadcast(output_shape, Shape.range(output_shape.rank(), output_shape.dtype()).dims()[a..]);
    }

    pub const ArangeArgs = struct {
        start: i64 = 0,
        end: i64,
        step: i64 = 1,
    };

    pub fn arange(args: ArangeArgs, dt: zml.DataType) Tensor {
        stdx.debug.assert(args.start <= args.end, "arange expects 'args.start' to be less than 'args.end', got {} and {}", .{ args.start, args.end });
        stdx.debug.assert(args.step > 0, "arange expects 'args.step' to be positive, got {}", .{args.step});

        //const loc = ctx.location(@src(), "arange({}, dtype={})", .{ args, dt });
        const loc = mlir.Location.unknown(mlirCtx());

        const n_steps = std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable;
        const sh = Shape.init(.{n_steps}, dt);
        var op = dialects.stablehlo.iota(mlirCtx(), 0, mlir.rankedTensorType(sh.dims(), zml.mlir.Type.fromDType(mlirCtx(), sh.dtype())), loc).appendTo(currentBlock());
        var res = _result(sh, op.result(0));
        _ = &res;

        if (args.step != 1) {
            res = res.scale(args.step);
        }

        if (args.start != 0) {
            res = res.addConstant(args.start);
        }

        return res;
    }

    pub fn reshape(self: Tensor, output_shape_: anytype) Tensor {
        const output_shape = self._shape.reshape(output_shape_);
        const tensor_type = mlir.rankedTensorType(output_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), output_shape.dtype()));
        //const loc = self.getContext().location(@src(), "reshape({f})", .{output_shape});
        const reshape_op = dialects.stablehlo.reshape(
            mlirCtx(),
            self.value(),
            tensor_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(output_shape, reshape_op.result(0));
    }

    pub fn convert(self: Tensor, to: zml.DataType) Tensor {
        if (to == self.dtype()) {
            return self;
        }
        //const loc = self.getContext().location(@src(), "convert({f},to={s})", .{ self, @tagName(to) });

        const res_type = mlir.rankedTensorType(self.dims(), zml.mlir.Type.fromDType(mlirCtx(), to));
        const op = dialects.stablehlo.convert(
            mlirCtx(),
            self.value(),
            res_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(self._shape.withDtype(to), op.result(0));
    }

    pub fn select(bool_tensor: Tensor, on_true: Tensor, on_false: Tensor) Tensor {
        stdx.debug.assert(bool_tensor.dtype() == .bool, "select expects input tensor type to be a boolean, got {}", .{bool_tensor.dtype()});
        stdx.debug.assert(on_true.dtype() == on_false.dtype(), "select expects 'on_true' and 'on_false' tensor types to be equal, got {} and {}", .{ on_true.dtype(), on_false.dtype() });

        if (bool_tensor.rank() != 0 and on_true.rank() == 0) {
            return bool_tensor.select(on_true.broad(bool_tensor.shape()), on_false);
        }
        if (bool_tensor.rank() != 0 and on_false.rank() == 0) {
            return bool_tensor.select(on_true, on_false.broad(bool_tensor.shape()));
        }

        stdx.debug.assert(bool_tensor._shape.eqlDims(on_true._shape), "select expects input tensor and 'on_true' tensor dimensions to match, got {f} and {f}", .{ bool_tensor._shape, on_true._shape });
        stdx.debug.assert(bool_tensor._shape.eqlDims(on_false._shape), "select expects input tensor and 'on_false' tensor dimensions to match, got {f} and {f}", .{ bool_tensor._shape, on_false._shape });

        //const loc = bool_tensor.mlirCtx().location(@src());
        const loc = mlir.Location.unknown(mlirCtx());
        const op = dialects.stablehlo.select(
            mlirCtx(),
            bool_tensor.value(),
            on_true.value(),
            on_false.value(),
            loc,
        ).appendTo(currentBlock());

        return _result(on_true._shape, op.result(0));
    }

    pub const ArgMaxRes = struct {
        values: Tensor,
        indices: Tensor,
    };

    pub const LogicalOp = enum { OR, XOR, AND };
    pub fn logical(self: Tensor, comptime logical_op: LogicalOp, other: Tensor) Tensor {
        return switch (logical_op) {
            .OR => binaryOp(@src(), "or", dialects.stablehlo.or_)(self, other),
            .XOR => binaryOp(@src(), "xor", dialects.stablehlo.xor)(self, other),
            .AND => binaryOp(@src(), "and", dialects.stablehlo.and_)(self, other),
        };
    }

    fn getComparisonType(dt: zml.DataType) dialects.stablehlo.CompareType.Type {
        return switch (dt) {
            .i4, .i8, .i16, .i32, .i64 => .SIGNED,
            .bool, .u4, .u8, .u16, .u32, .u64 => .UNSIGNED,
            .f8e4m3b11fnuz, .f8e4m3fn, .f8e4m3fnuz, .f8e5m2, .f8e5m2fnuz, .bf16, .f16, .f32, .f64 => .FLOAT,
            .c64, .c128 => @panic("Can't compare complex numbers"),
        };
    }

    pub fn cmp(self: Tensor, direction: dialects.stablehlo.ComparisonDirection.Direction, other: Tensor) Tensor {
        stdx.debug.assert(self.dtype() == other.dtype(), "cmp expects input tensors to be of the same type, got {t} and {t}", .{ self.dtype(), other.dtype() });

        if (self.rank() == 0 and other.rank() != 0) return self.broadcast(other._shape, &.{}).cmp(direction, other);
        if (self.rank() != 0 and other.rank() == 0) return self.cmp(direction, other.broadcast(self._shape, &.{}));

        stdx.debug.assert(self._shape.eql(other._shape), "cmp expects input tensor shapes to match, got {f} and {f}", .{ self._shape, other._shape });

        //const loc = self.getContext().location(@src(), "cmp(.{s})", .{@tagName(direction)});
        const loc = mlir.Location.unknown(mlirCtx());
        const op = dialects.stablehlo.compare(
            mlirCtx(),
            self.value(),
            other.value(),
            direction,
            getComparisonType(self.dtype()),
            loc,
        ).appendTo(currentBlock());

        return _result(self._shape.withDtype(.bool), op.result(0));
    }

    pub fn argMax(x: Tensor, axis_: anytype) ArgMaxRes {
        const a = x.axis(axis_);
        const dt: zml.DataType = if (x.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;

        const values, const indices = reduce(
            .{ x, Tensor.arange(.{ .end = x.dim(axis_) }, dt).broadcast(x.shape(), &.{a}) },
            .{ Tensor.constant(x.dtype().minValue()), Tensor.constant(dt.zero()) },
            &.{a},
            struct {
                fn cmp(values: ReduceArgs, indices: ReduceArgs) struct { Tensor, Tensor } {
                    const left_gt_right = values.left.cmp(.GT, values.right);
                    const is_nan = values.left.cmp(.NE, values.left);
                    const left_gt_or_nan = left_gt_right.logical(.OR, is_nan);
                    // we are bubbling up Nan.
                    const max_val = left_gt_or_nan.select(values.left, values.right);

                    // If values.left == values.right: keep the smallest idx.
                    const is_same = values.left.cmp(.EQ, values.right);
                    const is_first = indices.left.cmp(.LT, indices.right);
                    const is_same_but_first = is_same.logical(.AND, is_first);
                    const keep_left_idx = left_gt_or_nan.logical(.OR, is_same_but_first);
                    const max_idx = keep_left_idx.select(indices.left, indices.right);

                    return .{ max_val, max_idx };
                }
            }.cmp,
            .{},
        );
        return .{ .values = values, .indices = indices };
    }

    pub const ReduceArgs = struct {
        left: Tensor,
        right: Tensor,
    };

    fn reduce(inputs: anytype, inits: anytype, axes_: []const i64, comptime func: anytype, context: anytype) stdx.meta.FnResult(func) {
        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const reduce_block, var result = b: {
            const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inits.len);
            var args: ArgsType = undefined;
            var block_types: [2 * inits.len]*const mlir.Type = undefined;

            inline for (0..inits.len) |i| {
                args[i].left = Tensor.init(inits[i].shape());
                args[i].right = Tensor.init(inits[i].shape());

                block_types[i] = mlir.rankedTensorType(args[i].left.dims(), zml.mlir.Type.fromDType(mlirCtx(), args[i].left.dtype()));
                block_types[i + inits.len] = mlir.rankedTensorType(args[i].right.dims(), zml.mlir.Type.fromDType(mlirCtx(), args[i].right.dtype()));
            }

            const block_locs: [2 * inits.len]*const mlir.Location = @splat(mlir.Location.unknown(mlirCtx()));
            const reduce_block = mlir.Block.init(&block_types, &block_locs);
            errdefer reduce_block.deinit();

            CompilationContext.current().pushBlock(reduce_block);
            defer CompilationContext.current().popBlock();

            const scope = CompilationContext.current().currentScope();
            inline for (0..inits.len) |i| {
                scope.id_to_argument.put(scope.arena.allocator(), args[i].left.id, i) catch unreachable;
                scope.id_to_argument.put(scope.arena.allocator(), args[i].right.id, i + inits.len) catch unreachable;
            }

            var result = @call(.auto, func, args ++ context);

            var result_values: [inits.len]*const mlir.Value = undefined;
            inline for (0..inits.len) |i| {
                result_values[i] = result[i].value();
            }

            _ = dialects.stablehlo.returns(mlirCtx(), &result_values, .unknown(mlirCtx())).appendTo(reduce_block);
            break :b .{ reduce_block, result };
        };
        var input_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..inputs.len) |i| {
            input_values[i] = inputs[i].value();
        }

        var init_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..inits.len) |i| init_values[i] = inits[i].value();

        const reduce_op = mlir.Operation.make(mlirCtx(), "stablehlo.reduce", .{
            .operands = .{ .variadic = &.{ &input_values, &init_values } },
            .result_type_inference = true,
            .blocks = &.{reduce_block},
            .attributes = &.{
                .named(mlirCtx(), "dimensions", mlir.denseArrayAttribute(mlirCtx(), .i64, axes_)),
            },
            .verify = true,
            .location = .unknown(mlirCtx()),
        }).appendTo(currentBlock());

        // `stablehlo.reduce` drops axes. We want to avoid that to propagate tags.
        // So we need to broadcast the output of `stablehlo.reduce` to the input shapes.
        // To that order, we initialize `result` to `inputs`, then we use stdx.meta.visit,
        // to find the correct mlir.Value, but we first broadcast before creating the final
        // Tensor struct.
        var broadcasting_axes: stdx.BoundedArray(i64, Tensor.MAX_RANK) = .{};
        for (0..Tensor.MAX_RANK) |i| {
            if (std.mem.indexOfScalar(i64, axes_, @intCast(i)) == null) {
                broadcasting_axes.append(@intCast(i)) catch unreachable;
            }
        }

        inline for (0..result.len) |i| {
            var reduced_shape: Shape = inputs[i].shape();
            for (axes_) |a| {
                reduced_shape = reduced_shape.setDim(a, 1);
            }

            const tensor_type = mlir.rankedTensorType(reduced_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), reduced_shape.dtype()));
            const broad_op = dialects.stablehlo.broadcast_in_dim(
                mlirCtx(),
                reduce_op.result(i),
                broadcasting_axes.slice()[0 .. reduced_shape.rank() - axes_.len],
                tensor_type,
                .unknown(mlirCtx()),
            ).appendTo(currentBlock());

            result[i] = Tensor._result(reduced_shape, broad_op.result(0));
        }

        return result;
    }

    fn binaryOp(
        src: std.builtin.SourceLocation,
        op_name: []const u8,
        comptime op_fn: fn (*mlir.Context, *const mlir.Value, *const mlir.Value, *const mlir.Location) *mlir.Operation,
    ) fn (Tensor, Tensor) Tensor {
        _ = op_name; // autofix
        _ = src; // autofix
        return struct {
            pub fn binaryOpHelper(self: Tensor, other: Tensor) Tensor {
                const other_ = if (other.auto_broadcast) other.broad(self.shape()) else other;
                //stdx.debug.assert(self.dtype() == other.dtype(), "{s} expects tensor to be of same type, got {f} and {f}", .{ op_name, self, other });

                //if (self.rank() == 0 and other.rank() != 0) {
                //    return binaryOpHelper(self.broad(other._shape), other);
                //}

                //if (self.rank() != 0 and other.rank() == 0) {
                //    return binaryOpHelper(self, other.broad(self._shape));
                //}

                //stdx.debug.assert(self._shape.eql(other._shape), "{s} expects tensor shapes to match, got {f} and {f}", .{ op_name, self._shape, other._shape });

                const ctx = self.getContext();
                //const location = ctx.location(src, "{s}({f}, {f})", .{ op_name, self, other });
                const ret = op_fn(mlirCtx(), self.value(), other_.value(), .unknown(ctx.mlir_ctx)).appendTo(currentBlock());
                return _result(self._shape, ret.result(0));
            }
        }.binaryOpHelper;
    }

    pub fn concatenate(tensors: []const Tensor, axis_: anytype) Tensor {
        if (tensors.len == 1) return tensors[0];
        stdx.debug.assert(tensors.len <= 32, "concatenate only supports up to 32 tensors, got {}", .{tensors.len});
        var buffer: [32]*const mlir.Value = undefined;
        std.debug.assert(tensors.len <= buffer.len);
        std.debug.assert(tensors.len > 0);
        const a = tensors[0].axis(axis_);
        // TODO(Corendos): Check that tensor axes match.

        var concatenated_dim: i64 = 0;
        for (tensors, 0..) |t, i| {
            buffer[i] = t.value();
            concatenated_dim += t.dim(a);
        }

        const res_shape = tensors[0]._shape.set(a, concatenated_dim);
        const op = dialects.stablehlo.concatenate(mlirCtx(), buffer[0..tensors.len], a, .unknown(mlirCtx())).appendTo(currentBlock());
        // log.debug("concatenate({}, {}, {d}) -> {d}", .{ tensors[0], tensors[1], a, res_shape });
        return _result(res_shape, op.result(0));
    }

    pub fn value(self: Tensor) *const mlir.Value {
        return switch (self.tensor_origin) {
            .argument => b: {
                const argument_index = self.getContext().currentScope().id_to_argument.get(self.id).?;
                break :b self.getContext().currentScope().block.argument(argument_index);
            },
            .value => |v| v,
        };
    }

    /// Returns the donation data of the tensor.
    pub fn donation(self: Tensor) ?usize {
        return CompilationContext.current().currentScope().id_to_donation.get(self.id);
    }

    pub const Rng = struct {
        _state: Tensor,
        algorithm: dialects.stablehlo.RngAlgorithm.Type = .DEFAULT,

        pub fn init() Rng {
            return .{ ._state = .init(zml.Shape.init(.{2}, .u64)) };
        }

        pub fn initBuffer(platform: Platform, seed: u128) !Bufferized(Rng) {
            return .{
                ._state = try zml.Buffer.fromBytes(platform, Shape.init(.{2}, .u64), std.mem.asBytes(&seed)),
            };
        }

        ///// Returns a Tensor of the given shape, filled with uniform random bits, and a new Rng state.
        /////
        ///// The given Rng state should not be used anymore (or you'll get the same numbers again).
        ///// The output is guaranteed to be deterministic function of `self` Rng state,
        ///// but it is not guaranteed to be deterministic between implementations.
        //pub fn bitGenerator(self: Rng, sh: Shape) struct { Rng, Tensor } {
        //    const ctx = CompilationContext.current();
        //    const loc = ctx.location(@src(), "rand.bitGen({f})", .{sh});
        //    const op = dialects.stablehlo.rng_bit_generator(
        //        ctx.mlirCtx(),
        //        self.algorithm,
        //        self._state.value(),
        //        mlirx.tensorType(ctx.mlirCtx(), self._state._shape),
        //        mlirx.tensorType(ctx.mlirCtx(), sh),
        //        loc,
        //    );
        //    return .{ self.update(op.result(0)), _result(sh, op.result(1)) };
        //}

        fn update(self: Rng, new_state: mlir.Value) Rng {
            return .{
                ._state = _result(self._state._shape, new_state).reuseBuffer(self._state),
                .algorithm = self.algorithm,
            };
        }

        ///// Returns a Tensor of the given shape, filled with uniformly sampled floating point numbers from an interval,
        ///// and a new Rng state.
        /////
        ///// https://en.wikipedia.org/wiki/Continuous_uniform_distribution
        //pub fn uniform(
        //    self: Rng,
        //    shape_: Shape,
        //    opts: struct { min: f64 = 0, max: f64 = 1 },
        //) struct { Rng, Tensor } {
        //    const dt = if (shape_.dtype().isFloat()) shape_.dtype() else .f32;

        //    const mantissa_bit_count = @import("dtype.zig").mantissaSize(dt);
        //    const bit_count: usize = dt.bitSizeOf();
        //    const rng_bit_count = if (mantissa_bit_count < 8) 8 else bit_count;
        //    const uint_dtype: DataType = switch (bit_count) {
        //        8 => .u8,
        //        16 => .u16,
        //        32 => .u32,
        //        64 => .u64,
        //        else => stdx.debug.panic("uniform don't support non-byte aligned dtype. Got: {f}", .{shape_}),
        //    };

        //    const rng, const bits = self.bitGenerator(shape_.withDtype(uint_dtype));

        //    // Erase bits outside of mantissa.
        //    var float_bits = bits.shiftRightLogical(scalar(rng_bit_count - mantissa_bit_count, uint_dtype));

        //    // Set exponent bits to represent e^0 (eg 127 for f32).
        //    float_bits = float_bits.logical(.OR, scalar(1, dt).bitCast(uint_dtype));

        //    // float_bits now uniformly represents number in [1, 2[ range.
        //    // Let's convert to floats, and subtract one to go to [0, 1[ range.
        //    var floats = float_bits.bitCast(dt).sub(scalar(1, dt));
        //    floats = floats.mul(scalar(opts.max - opts.min, dt)).addConstant(opts.min);

        //    // Convert back to integer if needed.
        //    return .{ rng, floats.convert(shape_.dtype()) };
        //}

        //test uniform {
        //    const zml = @import("zml.zig");
        //    const Stats = struct {
        //        const Stats = @This();

        //        mean: Tensor,
        //        variance: Tensor,
        //        min: Tensor,
        //        max: Tensor,

        //        pub fn uniformStats(
        //            rand: Rng,
        //            shape_: Shape,
        //            opts: struct { min: f64, max: f64 },
        //        ) struct { Rng, Stats } {
        //            const rng, const data = rand.uniform(shape_, .{ .min = opts.min, .max = opts.max });
        //            const mean_ = data.mean(0);
        //            const variance = data.sub(mean_.broad(data.shape())).pow(Tensor.scalar(2, .f32)).mean(0);
        //            return .{ rng, .{
        //                .mean = mean_,
        //                .variance = variance,
        //                .min = data.min(0),
        //                .max = data.max(0),
        //            } };
        //        }
        //    };

        //    const platform = zml.testing.env();
        //    // Compute stats over a uniform distribution on [-2, 10].
        //    const rand, const stats = try zml.testing.compileAndCallWithTensors(
        //        platform,
        //        Stats.uniformStats,
        //        .{ Rng.shape(), zml.Shape.init(.{1024}, .f32), .{ .min = -2, .max = 10 } },
        //        .{try Rng.init(platform, 1234)},
        //    );

        //    // Check the Rng state has been modified.
        //    try std.testing.expect(try rand._state.getValue(u128) != 1234);

        //    // Check the mean and variance are close to theoritical values.
        //    const mean_ = try stats.mean.getValue(f32);
        //    try std.testing.expectApproxEqAbs(4, mean_, 0.03);

        //    const variance = try stats.variance.getValue(f32);
        //    try std.testing.expectApproxEqAbs(12.0 * 12.0 / 12.0, variance, 0.01);

        //    // Check that no value is outside of the interval
        //    // and we have samples close to the edges.
        //    const min_ = try stats.min.getValue(f32);
        //    try std.testing.expect(min_ >= -2);
        //    try std.testing.expectApproxEqAbs(-2, min_, 0.05);

        //    const max_ = try stats.max.getValue(f32);
        //    try std.testing.expect(max_ < 10);
        //    try std.testing.expectApproxEqAbs(10, max_, 0.05);
        //}

        ///// Returns a Tensor of the given shape, filled with floating point numbers sampled from a normal distribution.
        /////
        ///// Note: this uses stablehlo.rng which is deprecated.
        ///// https://github.com/openxla/stablehlo/blob/main/rfcs/20240503-opset-deprecations.md
        //pub fn normal(sh: Shape, opts: struct { mean: f64 = 0, stddev: f64 = 1 }) Tensor {
        //    stdx.debug.assert(sh.dtype().isFloat(), "normal expects tensor type to be a float, got {}", .{sh.dtype()});

        //    const ctx = CompilationContext.current();
        //    const loc = ctx.location(@src(), "rand.normal({f}, mean={},stddev={})", .{ sh, opts.mean, opts.stddev });
        //    const a = Tensor.constant(.{}, Data.init(sh.dtype(), opts.mean));
        //    const b = Tensor.constant(.{}, Data.init(sh.dtype(), opts.stddev));
        //    const res_shape = Tensor.constantTensor(HostBuffer.fromSlice(.{sh.rank()}, sh.dims()));
        //    const op = dialect.stablehlo.rng(ctx.mlirCtx(), a.value(), b.value(), res_shape.value(), .NORMAL, loc);
        //    return _result(sh, op.result(0));
        //}

        ///// Returns a Tensor of the given shape, filled with floating point numbers sampled from a Gumbel distribution, and a new Rng state.
        /////
        ///// Often used in ML because of the reparametrization tricks.
        ///// Sampling from a gumbel distribution is equivalent to sample
        ///// from a softmax distribution, but doesn't require to compute the sum of exponentials.
        ///// https://en.wikipedia.org/wiki/Gumbel_distribution#Gumbel_reparametrization_tricks
        ///// See `sampleTokens` for a practical use case.
        ///// Note: we only implement the μ=0, β=1 version.
        //pub fn gumbel(self: Rng, shape_: Shape) struct { Rng, Tensor } {
        //    const rand, const u = self.uniform(
        //        // Always use .f32 to have a big enough mantissa.
        //        shape_.withDtype(.f32),
        //        // We don't want 0 to be sampled otherwise `log` will return -inf.
        //        .{ .min = std.math.floatEps(f32), .max = 1 },
        //    );
        //    return .{ rand, u.log().scale(-1).log().scale(-1).convert(shape_.dtype()) };
        //}

        //test gumbel {
        //    const zml = @import("zml.zig");
        //    const Stats = struct {
        //        const Stats = @This();

        //        mean: Tensor,
        //        variance: Tensor,
        //        actual_dist: Tensor,

        //        pub fn gumbelStats(rand: Rng, target_dist: Tensor) struct { Rng, Stats } {
        //            const s = Shape.init(.{ .n = 1024, .d = 4 }, .f32);
        //            const rng, const data = rand.gumbel(s);
        //            const flat = data.flattenAll();
        //            const mean_ = flat.mean(0);
        //            const variance = flat.sub(mean_.broad(flat.shape())).pow(Tensor.scalar(2, .f32)).mean(0);

        //            // Test out the gumbel reparametrization trick
        //            var x = target_dist.log().withTags(.{.d}).broad(s);
        //            x = x.add(data);
        //            const samples = x.argMax(.d).indices.squeeze(.d);

        //            // count 0, 1, 2 and 3 in samples:
        //            // - map 0 to 1, 1 to 2**16, 2 to 2**32, 3 to N**58
        //            // - sum in u64
        //            // - split to [4]u16
        //            const powers = blk: {
        //                var powers: [4]u64 = undefined;
        //                for (&powers, 0..) |*p, i| p.* = std.math.pow(u64, 2, i * 16);
        //                break :blk powers;
        //            };
        //            const values = Tensor.constantTensor(HostBuffer.fromArray(&powers)).withTags(.{.d});
        //            const counts = values.gatherValues(.d, samples, .{}).sum(.n).bitCast(.u16);
        //            const actual_dist = counts.reshape(target_dist.shape()).convert(target_dist.dtype()).divByConst(s.dim(.n));
        //            return .{ rng, .{ .mean = mean_, .variance = variance, .actual_dist = actual_dist } };
        //        }
        //    };

        //    const platform = zml.testing.env();
        //    const tgt_dist = [_]f32{ 2.0, 1.0, 4.0, 3.0 };
        //    const rand, const stats = try zml.testing.compileAndCallWithTensors(
        //        platform,
        //        Stats.gumbelStats,
        //        .{ Rng.shape(), zml.Shape.init(.{tgt_dist.len}, .f32) },
        //        .{ try Rng.init(platform, 1234), try .fromArray(platform, tgt_dist) },
        //    );
        //    // Check the Rng state has been modified.
        //    try std.testing.expect(try rand._state.getValue(i128) != 1234);

        //    // Check the mean and variance are close to theoritical values.
        //    const mean_ = try stats.mean.getValue(f32);
        //    try std.testing.expectApproxEqAbs(0.5772, mean_, 0.02);

        //    const variance = try stats.variance.getValue(f32);
        //    const pi = std.math.pi;
        //    try std.testing.expectApproxEqAbs(pi * pi / 6.0, variance, 0.03);

        //    // Check the distribution obtained with the gumbel trick matches the target distribution.
        //    const actual_dist = try stats.actual_dist.getValue([4]f32);
        //    scoped_log.debug("tgt_dist: {d}, actual_dist: {d}", .{ tgt_dist, actual_dist });
        //    for (tgt_dist, actual_dist) |tgt, actual| {
        //        // We normalize tgt_dist to make it a well formed distribution.
        //        // We didn't do it before calling gumbel, because the gumbel trick
        //        // doesn't require normalized distributions as input.
        //        try std.testing.expectApproxEqAbs(tgt / 10.0, actual, 0.05);
        //    }
        //}
    };

    /// Tell PJRT compiler that memory should be reuse between the two tensors.
    /// The compiler is already aggressively reusing tensors for intermediate results,
    /// but this API allows to reuse buffer between input and output arguments
    /// of a given function.
    /// Note this is visible from the outside. The caller of a function with donations
    /// is not allowed to reuse the donated input buffer after the call.
    /// For `reuseBuffer` to be effective, it needs to propagate all the way through the output.
    pub fn reuseBuffer(self: Tensor, origin: Tensor) Tensor {
        const compilation_context = CompilationContext.current();
        const scope = compilation_context.currentScope();
        switch (origin.tensor_origin) {
            .argument => {
                const argument_index = scope.id_to_argument.get(origin.id).?;

                const gop = scope.id_to_donation.getOrPut(scope.arena.allocator(), self.id) catch unreachable;
                gop.value_ptr.* = argument_index;
            },
            .value => {
                if (scope.id_to_donation.get(origin.id)) |origin_donation| {
                    const gop = scope.id_to_donation.getOrPut(scope.arena.allocator(), self.id) catch unreachable;
                    gop.value_ptr.* = origin_donation;
                }
            },
        }
        return self;
    }

    pub const GatherOpts = struct { indices_are_sorted: bool = false };

    /// For each coordinate in `indices`,
    /// `gatherValues` extracts a single value of the given tensor.
    ///
    /// * axes_ is a single axis, or a tuple of axis: .b, or .{ .b, .c }
    /// * indices is an integer tensor
    /// * result is a tensor whose shape is similar to the input shape
    /// where the gathered axes have been replaced by axes from 'indices'.
    ///
    /// Some example input for the base case where we work on one axis:
    /// - gatherValues(f:[a]->float, .a, ind:[n]->int)[n] == f[ind[n]]
    /// - gatherValues(f:[a, b], .a, ind:[n])[n, b] == f[ind[n], b]
    /// - gatherValues(f: [a,b,c], .{.b}, ind: [n,m])[a, n, m, c] == f[a, ind[n, m], c]
    ///
    /// If an axis in common between `self` and `indices`,
    /// it is treated as a "batching" axis, meaning that semantically
    /// the operator is doing a gatherValues one time per dimension of this axis:
    /// - gatherValues(f: [a,b,c], .{.b}, ind: [a,n])[a, n] == f[a, ind[a, n]]
    ///
    /// It is an error to have an axis present in `self`, `axes_` and `indices`.
    ///
    /// If several axes are passed, then the last axis of indices is treated as coordinates:
    /// - gatherValues(f: [a,b,c], .{.b, .c}, ind: [n,2])[a, n] == f[a, ind[n][0], ind[n][1]]
    /// - gatherValues(f: [a,b,c,d], .{.b, .c}, ind: [a, n,2])[a, n, d] == f[a, ind[a, n][0], ind[a, n][1], d]
    ///
    /// It is possible to use gatherValues without tags, but batching won't be available.
    pub fn gatherValues(self: Tensor, coord_axes: anytype, indices: Tensor, opts: GatherOpts) Tensor {
        // scoped_log.debug("gatherValues({}, {any}, {})", .{ self, coord_axes, indices });
        const single_coord, const coord_axes_ = _parseGatherCoord(self, coord_axes);

        stdx.debug.assert(coord_axes_.len > 0, "gatherValues expects 1 or more axes to operate one, received none. Example: `x.gatherValues(.a, indices, .{{}})`", .{});
        for (coord_axes_.constSlice(), 0..) |a, i| {
            if (i > 0) {
                stdx.debug.assert(a == coord_axes_.get(i - 1) + 1, "gatherValues expects 'coord_axes' to be sequential. But {any} aren't sequential in {f}", .{ coord_axes, self });
            }
        }

        const LocalAxisKind = enum { batching, offset, collapsed, indices };
        var self_kind: stdx.BoundedArray(LocalAxisKind, MAX_RANK) = .{};
        var indices_batch_axes: Shape.DimsArray = .{};
        for (self._shape.tags(), 0..self.rank()) |t, self_ax| {
            const maybe_coord_ax = std.mem.indexOfScalar(u3, coord_axes_.constSlice(), @intCast(self_ax));
            if (indices._shape.hasTag(t)) |id_ax| {
                // tag is both in self and indices -> it's a batching dim
                // Note: tags are required for batching.
                self_kind.appendAssumeCapacity(.batching);
                indices_batch_axes.appendAssumeCapacity(id_ax);
                stdx.debug.assert(maybe_coord_ax == null, "gatherValues expects axes to appear at most twice. Axis {s} has been found both in 'self={f}', in 'coord_axes_={any}' and in 'indices={f}'", .{ self._shape._tags.get(self_ax), self, coord_axes, indices });
            } else if (maybe_coord_ax) |_| {
                // for gatherValues we collapsed all gathered axes
                // (contrary to gatherSlices where we collapse none)
                self_kind.appendAssumeCapacity(.collapsed);
            } else {
                self_kind.appendAssumeCapacity(.offset);
            }
        }

        // When we receive several coord_axes we need an extra dimension to store
        // one index per axis, which makes the coordinates of one value.
        // Otherwi se stablehlo uses the "indices.rank()" default value.
        const index_coord_axis = if (single_coord)
            indices.rank()
        else blk: {
            const ax = indices._shape.hasTag(.coord) orelse indices._shape.axis(-1);
            stdx.debug.assert(indices.dim(ax) == coord_axes_.len, "gatherValues with axes={any}, expects indices to be of shape [..., {}], got: {f}", .{ coord_axes, coord_axes_.len, indices });
            break :blk ax;
        };

        // compute res shape
        var res_shape = Shape.init(.{}, self.dtype());
        var res_kind: stdx.BoundedArray(LocalAxisKind, MAX_RANK) = .{};
        for (self_kind.constSlice(), 0..) |kind, ax_usize| {
            const ax: u3 = @intCast(ax_usize);
            if (ax == coord_axes_.get(0)) {
                // The first val_ax is special cause this is the place where we insert indices axes.
                for (indices._shape.tags(), 0..indices.rank()) |t, id_ax| {
                    if (id_ax == index_coord_axis) continue;
                    if (std.mem.indexOfScalar(i64, indices_batch_axes.constSlice(), @intCast(id_ax))) |_| {
                        // batching dim are already in res
                        continue;
                    }

                    res_shape = res_shape.appendDim(indices.dim(id_ax), t);
                    res_kind.appendAssumeCapacity(.indices);
                }
            }
            switch (kind) {
                .collapsed => continue,
                else => {
                    res_shape = res_shape.appendDim(self.dim(ax), self._shape.tag(ax));
                    res_kind.appendAssumeCapacity(kind);
                },
            }
        }

        // This is not a gather, but a dynamicSlice.
        // Sometimes the backend recognize this pattern, but not always.
        // So let us handle that.
        if (indices.count() == 1) {
            return self.dynamicSlice1d(coord_axes_.get(0), .{ .start = indices.flattenAll().squeeze(0), .len = 1 }).reshape(res_shape);
        }

        var slice_dims: Shape.DimsArray = .{};
        for (self_kind.constSlice(), self.dims()) |k, d| {
            slice_dims.appendAssumeCapacity(switch (k) {
                .batching, .collapsed => 1,
                .offset => d,
                .indices => unreachable,
            });
        }

        // scoped_log.debug("gatherValues --> {} {any}", .{ res_shape, res_kind.constSlice() });
        const gather_op = dialects.stablehlo.gather(
            mlirCtx(),
            self.value(),
            indices.value(),
            slice_dims.constSlice(),
            .{
                .offset_dims = _collectAxes(LocalAxisKind, res_kind, .offset).constSlice(),
                .collapsed_slice_dims = _collectAxes(LocalAxisKind, self_kind, .collapsed).constSlice(),
                .operand_batching_dims = _collectAxes(LocalAxisKind, self_kind, .batching).constSlice(),
                .start_indices_batching_dims = indices_batch_axes.constSlice(),
                .start_index_map = _collectAxes(LocalAxisKind, self_kind, .collapsed).constSlice(),
                .index_vector_dim = index_coord_axis,
                .indices_are_sorted = opts.indices_are_sorted,
            },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        //const mlir_shape = fromMlirValue(gather_op.result(0)).shape();
        //stdx.debug.assert(mlir_shape.eql(res_shape), "gatherValues expects that batching indices appear in the same order in 'self' and 'indices', got: self={f}, indices={f}. You should transpose one or the other.", .{ self, indices });
        return _result(res_shape, gather_op.result(0));
    }

    /// Returns a Tensor containing the element-wise reverse square-root of the input Tensor.
    pub fn rsqrt(self: Tensor) Tensor {
        const rsqrt_op = dialects.stablehlo.rsqrt(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, rsqrt_op.result(0));
    }

    /// Drops a 1-dim axis at the given index
    pub fn squeeze(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        stdx.debug.assert(self.dim(a) == 1, "squeeze expects axis to be squeezed to have a dimension of 1, got {}", .{self.dim(a)});

        const new_shape = self._shape.remove(a);
        // log.debug("squeeze({}, {d}={d}) -> ({})", .{ self, axis, a, new_shape });

        return _result(new_shape, self.reshape(new_shape).value());
    }

    /// Returns a Tensor containing the sum of elements over the given axis.
    /// Output shape is the input shape with the axis_ dim set to 1.
    pub fn sum(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        const result = reduce(.{self}, .{Tensor.scalar(0, self.dtype())}, &.{a}, struct {
            pub fn acc(args: ReduceArgs) struct { Tensor } {
                const x = args.left;
                const res = args.right;
                const result = res.add(x.convert(res.dtype()));
                return .{result};
            }
        }.acc, .{});
        return result.@"0";
    }

    inline fn wrapIndex(self: Tensor, axis_: usize, idx: i64) i64 {
        return if (idx < 0) self.dim(axis_) + idx else idx;
    }

    /// Returns a Tensor containing the mean of elements over the given axis.
    /// Output shape is the input shape with the axis_ dim set to 1.
    pub fn mean(self: Tensor, axis_: anytype) Tensor {
        return self.sum(axis_).divByConst(self.dim(axis_));
    }

    /// Returns a Tensor containing the element-wise division of the input Tensor by a constant.
    pub fn divByConst(self: Tensor, b: anytype) Tensor {
        return self.div(Tensor.scalar(b, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise multiplication of the input Tensor by a constant.
    pub inline fn scale(self: Tensor, val: anytype) Tensor {
        return self.mul(Tensor.scalar(val, self.dtype()));
    }

    /// Returns a Tensor containing the element-wise addition of the input Tensor with a constant.
    pub fn addConstant(self: Tensor, b: anytype) Tensor {
        return self.add(Tensor.scalar(b, self.dtype()));
    }

    /// Returns a 0-rank Tensor with the given value.
    pub fn scalar(val: anytype, dt: zml.DataType) Tensor {
        const data = zml.Data.init(dt, val);
        switch (dt.class()) {
            .float => stdx.debug.assert(!std.math.isNan(val), "scalar(NaN) is probably due to compiling a model with an uninitialized field", .{}),
            else => {},
        }
        return Tensor.constant(data);
    }

    /// Slices the input Tensor along a specific axis, with a start offset known at runtime.
    /// Note: this doesn't support tagging, if you have tags,
    /// you should use `dynamicSlice` directly.
    pub fn dynamicSlice1d(self: Tensor, axis_: i8, slice_: DynSlice) Tensor {
        stdx.debug.assert(slice_.start.rank() == 0, "dynamicSlice1d expects 'slice_.start' tensor rank to be a scalar, got {f}", .{slice_.start});

        const a = self.axis(axis_);
        const new_shape = self._shape.set(a, slice_.len);

        var start_indices = [_]*const mlir.Value{constant(slice_.start.dtype().zero()).value()} ** MAX_RANK;
        start_indices[a] = slice_.start.value();

        const op = dialects.stablehlo.dynamic_slice(
            mlirCtx(),
            self.value(),
            new_shape.dims(),
            start_indices[0..self.rank()],
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        return _result(new_shape, op.result(0));
    }

    pub const DynSlice = struct { start: Tensor, len: i64 };

    pub const Slice = struct {
        start: i64 = 0,
        end: i64 = to_the_end,
        step: i32 = 1,
        singleton: bool = false,

        pub fn single(offset: i64) Slice {
            return .{ .start = offset, .end = offset + 1, .singleton = true };
        }

        const to_the_end = std.math.maxInt(i64);

        pub fn format(
            self: Slice,
            comptime fmt: []const u8,
            options: std.fmt.FormatOptions,
            writer: anytype,
        ) !void {
            _ = fmt;
            _ = options;
            if (self.singleton) {
                try writer.print("[{}]", .{self.start});
            } else if (self.end == to_the_end and self.step == 1) {
                try writer.print("[{}..]", .{self.start});
            } else if (self.step == 1) {
                try writer.print("[{}..{}]", .{ self.start, self.end });
            } else {
                try writer.print("[{}..{}:{}]", .{ self.start, self.end, self.step });
            }
        }
    };

    /// Slices the input Tensor over the given axis using the given parameters.
    pub fn slice1d(self: Tensor, axis_: anytype, s: Slice) Tensor {
        var slices = [_]Slice{.{}} ** MAX_RANK;
        slices[self.axis(axis_)] = s;
        return self.slice(slices[0..self.rank()]);
    }

    /// Slices the input Tensor using the given parameters.
    pub fn slice(self: Tensor, slices: []const Slice) Tensor {
        var start_indices: [MAX_RANK]i64 = undefined;
        var strides: [MAX_RANK]i64 = undefined;
        var limit_indices: [MAX_RANK]i64 = undefined;
        var res_shape: Shape = self._shape;

        for (slices, 0..) |s, a| {
            stdx.debug.assert(s.step > 0, "slice expects 'step' to be positive, got {} at index {}", .{ s.step, a });

            const args: Slice = .{
                .start = self.wrapIndex(a, s.start),
                .end = if (s.end == Slice.to_the_end) self.dim(a) else self.wrapIndex(a, s.end),
                .step = s.step,
            };
            start_indices[a] = args.start;
            limit_indices[a] = args.end;
            strides[a] = args.step;
            res_shape = res_shape.setDim(a, std.math.divCeil(i64, args.end - args.start, args.step) catch unreachable);
        }

        const result_type = mlir.rankedTensorType(res_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), res_shape.dtype()));
        const slice_op = dialects.stablehlo.slice(
            mlirCtx(),
            self.value(),
            start_indices[0..self.rank()],
            limit_indices[0..self.rank()],
            strides[0..self.rank()],
            result_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        var res = _result(res_shape, slice_op.result(0));
        var to_remove: Shape.AxesArray = .{};
        for (slices, 0..) |s, a| {
            if (s.singleton) to_remove.appendAssumeCapacity(@intCast(a));
        }
        return res.reshape(res_shape.removeMany(to_remove.constSlice()));
    }

    /// Returns a Tensor containing the Sigmoid Linear Unit (SiLU) activation function applied to each element of the input Tensor.
    ///
    /// silu(x) = x σ(x)
    /// https://paperswithcode.com/method/silu
    pub fn silu(x: Tensor) Tensor {
        return x.mul(x.sigmoid());
    }

    /// Returns a Tensor containing the sigmoid function applied to each element of the input Tensor.
    pub fn sigmoid(self: Tensor) Tensor {
        const op = dialects.stablehlo.logistic(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    pub const logistic = sigmoid;

    /// Splits the given axis in several axes.
    /// eg: `Tensor.init(.{ .a = 10, .b = 3 }).split(.a, .{.a1 = 5, .a2 = 2});`
    /// The number of elements in the split shape must match the number of element
    /// in the target axis.
    pub fn splitAxis(self: Tensor, ax: anytype, split_shape: anytype) Tensor {
        const new_shape = self._shape.splitAxis(ax, split_shape);

        const tensor_type = mlir.rankedTensorType(new_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), new_shape.dtype()));
        const reshaped_val = dialects.stablehlo.reshape(
            mlirCtx(),
            self.value(),
            tensor_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(new_shape, reshaped_val.result(0));
    }

    /// Gathers slices along the given axes with runtime indices.
    /// * slice_shape represents the shape of the slices to extract,
    ///   it must be smaller than original shape.
    ///   It must use a subset of self axes.
    ///   If slice_shape is **not** tagged, then it must have the same rank than self.
    /// * `indices` represents a set of coordinates.
    ///   The coordinates are read from the `.coord` axis, or last axis if `.coord` is not found.
    ///   The coordinate axis must have `slice_shape.rank()` dims.
    ///   The coordinates represent the "top-left" corner of the slice to extract.
    /// * the output tensor starts with axes from `indices`.
    /// * if the input tensor has tagged axes, matching `indices` axes,
    ///    they will be considered "batching" axes.
    ///
    /// Sample input/output shapes:
    /// * gatherSlices([A, B, C, D], .{.b=B', .c=C'}, [N, 2]) -> [N, A, B', C', D]
    /// * gatherSlices(x(a,b,c,d), .{.b=B', .c=C'}, g(n,m)) = z(n, a, b', c', d) = x(a, g(n, 0) + b', g(n, 1) + c', d)
    ///
    /// Note: the axis order of the result is different from gatherValues.
    /// This is because gatherSlices, favorizes contiguous copy of the extracted slices,
    /// while gatherValues, always copy values one by one, and as such don't have the same issues.
    /// In our example the contiguous dimension .d is not sliced
    /// and gatherSlices can copy data by group of C'*D elements.
    pub fn gatherSlices(self: Tensor, slice_shape_: anytype, indices: Tensor, opts: GatherOpts) Tensor {
        const slice_shape = if (@TypeOf(slice_shape_) == Shape) slice_shape_ else Shape.init(slice_shape_, .i32);
        // scoped_log.debug("gatherSlice({}, {f}, {})", .{ self, slice_shape, indices });

        const tagged_api = slice_shape.isFullyTagged();
        if (tagged_api) {
            for (slice_shape.tags()) |t| {
                stdx.debug.assert(self._shape.hasTag(t) != null, "gatherSlices expects `slices_shape` to only use tags from `self`. But {s} wasn't found in {f}", .{ t, self });
            }
        } else {
            // For untagged api, we require all slices to be specified.
            // Note: we could relax this and right align the slice.
            stdx.debug.assert(slice_shape.rank() == self.rank(), "gatherSlices expects `slice_shape.rank()` to match `self.rank()`. Got: gatherSlices({f}, slice={f}). To avoid specifying all axes in `slice_shape`, you can use tags.", .{ self, slice_shape });
        }

        const index_coord_axis = indices._shape.hasTag(.coord) orelse indices._shape.axis(-1);
        stdx.debug.assert(indices.dim(index_coord_axis) == slice_shape.rank(), "gatherSlices({f}, slice={f}, indices) expects 'indices' to be a tensor [..., {}], got {f}", .{ self, slice_shape, slice_shape.rank(), indices });

        // Compute result shape
        var res_shape = indices._shape.remove(index_coord_axis).withDtype(self.dtype());
        var slice_dims = self._shape._dims;
        var self_batch_axes: stdx.BoundedArray(i64, MAX_RANK) = .{};
        var indices_batch_axes: stdx.BoundedArray(i64, MAX_RANK) = .{};
        var start_index_map: stdx.BoundedArray(i64, MAX_RANK) = .{};
        var self_offset_axes: stdx.BoundedArray(i64, MAX_RANK) = .{};
        for (self._shape.tags(), 0..self.rank()) |t, self_ax| {
            const maybe_slice_ax: ?u3 = if (tagged_api) slice_shape.hasTag(t) else @intCast(self_ax);

            if (tagged_api and indices._shape.hasTag(t) != null) {
                // tag is both in self and indices -> it's a batching dim
                // Note: tags are required for batching.
                self_batch_axes.appendAssumeCapacity(@intCast(self_ax));
                indices_batch_axes.appendAssumeCapacity(indices._shape.axis(t));
                slice_dims.set(self_ax, 1);
                stdx.debug.assert(slice_shape.hasTag(t) == null, "gatherSlices expect axes to be either batches or slices axes. Axis {s} has been found both in `slices={f}` and `indices={f}`", .{ t, slice_shape, indices });
            } else if (maybe_slice_ax) |slice_ax| {
                // Specified axes contains the start offset of the slices,
                // and are collected in `start_index_map`.
                const slice_dim = slice_shape.dim(slice_ax);
                stdx.debug.assert(slice_dim <= self._shape.dim(self_ax), "gatherSlices expects `slice_shape` to be smaller than `self.shape()`. On axis {s}, got {f} > {f}.", .{ t, slice_shape, self._shape });
                slice_dims.set(self_ax, slice_dim);
                res_shape = res_shape.appendDim(slice_dim, t);
                start_index_map.appendAssumeCapacity(@intCast(self_ax));
                self_offset_axes.appendAssumeCapacity(res_shape.rank() - 1);
            } else {
                // non-batching, non-indexed axes
                res_shape = res_shape.appendDim(self.dim(self_ax), t);
                self_offset_axes.appendAssumeCapacity(res_shape.rank() - 1);
            }
        }

        const gather_op = dialects.stablehlo.gather(
            mlirCtx(),
            self.value(),
            indices.value(),
            slice_dims.constSlice(),
            .{
                .offset_dims = self_offset_axes.constSlice(),
                .collapsed_slice_dims = &.{},
                .operand_batching_dims = self_batch_axes.constSlice(),
                .start_indices_batching_dims = indices_batch_axes.constSlice(),
                .start_index_map = start_index_map.constSlice(),
                .index_vector_dim = index_coord_axis,
                .indices_are_sorted = opts.indices_are_sorted,
            },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, gather_op.result(0));
    }

    /// Merges two or more contiguous axes into one axis.
    pub fn merge(self: Tensor, merges_: anytype) Tensor {
        return self.reshape(self._shape.mergeAxes(merges_));
    }

    /// Slices a Tensor across many axes, with runtime known offsets.
    ///
    /// Due to the nature of stablehlo, the length of the slices need to be known when compiling the IR.
    /// When using the tagged API it is allowed to not specify some axes.
    /// But with the non-tagged API all slices need to be specified.
    /// Examples:
    /// ```
    /// Tensor(.{.a=20,.b=30,.c=40 }).dynamicSlice(.{ .a = .{ .start = a_off, .len = 11});
    /// Tensor(.{.a=20,.b=30,.c=40 }).dynamicSlice(.{
    ///     .a = .{ .start = a_off, .len = 11 },
    ///     .b = .{ .start = b_off, .len = 12 },
    ///   });
    /// Tensor(.{ 20,30,40}).dynamicSlice(.{.{ .start = scalar(0, .i32), .len = 20 }, .{ .start = b_off, .len = 12 }, .{ .start = scalar(0, .i32), .len = 40 }});
    /// ```
    pub fn dynamicSlice(self: Tensor, slices_: anytype) Tensor {
        // TODO: the untagged api is a bit verbose. Should I allow: `Tensor(.{ 20,30,40}).dynamicSlice(.{.{}, .{ .start = b_off, .len = 12 }, .{}});` ??
        //
        const slices, const slices_tags = Shape.parseStruct(DynSlice, slices_);

        // TODO use slices and slices_tags for the format.
        // Currently this prints: "dynSlice(struct{q: struct{start: tensor.Tensor, comptime len: comptime_int = 1}}{ .q = struct{start: tensor.Tensor, comptime len: comptime_int = 1}{ .start = Tensor({1,10}, dtype=.i64), .len = 1 } })"
        // which is kinda ugly.

        const idx_dtype = if (slices.len > 0) slices.get(0).start.dtype() else .i32;
        const zero = Tensor.scalar(0, idx_dtype).value();
        var offset_values = [_]*const mlir.Value{zero} ** MAX_RANK;
        var res_shape = self._shape;
        for (slices.constSlice(), 0..) |slice_, i| {
            const offset = slice_.start;
            const len = slice_.len;
            if (slices_tags.len == 0) {
                stdx.debug.assert(self.rank() == slices.len, "dynamicSlice expects tensor rank and 'slices_' length to be equal, got {d} and {d}", .{ self.rank(), slices.len });

                offset_values[i] = offset.value();
                res_shape._dims.set(i, len);

                stdx.debug.assert(len <= self.dim(i), "dynamicSlice expects slices 'len' to be less than or equal to their corresponding dimension in input tensor, got {d} and {d} for index {d}", .{ len, self.dim(i), i });
            } else {
                const t = slices_tags.get(i);
                const a = res_shape.hasTag(t) orelse stdx.debug.panic("dynamicSlice expects input tensor to have tags used in 'slices_' but {s} is missing (input shape is {f})", .{ t, self._shape });

                stdx.debug.assert(len <= self.dim(a), "dynamicSlice expects slices 'len' to be less than their corresponding dimension in input tensor, got {d} and {d} for axis {s}", .{ len, self.dim(a), t });

                offset_values[a] = offset.value();
                res_shape._dims.set(a, len);
            }
        }
        const op = dialects.stablehlo.dynamic_slice(mlirCtx(), self.value(), res_shape.dims(), offset_values[0..self.rank()], .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    /// Returns a Tensor containing values in increasing order starting from 0 along the given axis.
    ///
    /// The output dtype will be `.i32`, unless the given axis has a too big dimension, in that case we use `.i64`.
    /// In most program this shouldn't matter, because typically this will be used in a comparison,
    /// or explicitly converted by the user to do floating point arithmetic.
    pub fn iota(sh: Shape, axis_: anytype) Tensor {
        const a = sh.axis(axis_);
        const dt: zml.DataType = if (sh.dim(a) <= std.math.maxInt(i32)) .i32 else .i64;
        const res_shape = sh.withDtype(dt);

        const tensor_type = mlir.rankedTensorType(res_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), res_shape.dtype()));
        var op = dialects.stablehlo.iota(
            mlirCtx(),
            a,
            tensor_type,
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    /// Returns a Tensor containing the result of the outer product between the input Tensors.
    pub fn outer(self: Tensor, other: Tensor) Tensor {
        if (self.rank() + other.rank() == 1) {
            return self.mul(other);
        }

        const res_shape = self.shape().outer(other.shape());
        return self.broad(res_shape).mul(other.broad(res_shape));
    }

    /// Appends a 1-dim axis, with the given tag.
    pub fn appendAxes(self: Tensor, t: anytype) Tensor {
        // stdx.debug.assert(self.rank() < Tensor.MAX_RANK - t.len, "appendAxis expects tensor rank to be small enough in order to extend it, got {} and {} (max is {})", .{ self.rank(), t.len, Tensor.MAX_RANK });

        return self.insertAxes(.last, t);
    }

    /// Repeats a Tensor several times along the given axis.
    ///
    /// * repeat1d(x, axis, 4) = concat(&.{x, x, x, x}, axis);
    /// * repeat1d([0, 1, 2, 3], 0, 2) = [0, 1, 2, 3, 0, 1, 2, 3]
    pub fn repeat1d(self: Tensor, axis_: anytype, n_rep: u63) Tensor {
        if (n_rep == 1) {
            return self;
        }

        const a = self.axis(axis_);
        const broadshape = self._shape.insert(a + 1, .{n_rep});
        const repeat_dims = Shape.range(self.rank() + 1, self.dtype()).remove(a + 1);

        var res = self.broadcast(broadshape, repeat_dims.dims()).flatten(a);
        // Restor the tag that has been lost by flatten.
        res._shape._tags.set(a, self._shape.tag(a));

        return res;
    }

    /// Returns a Tensor containing the softmax function applied to each element of the input Tensor.
    pub fn softmax(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        const max_val = self.max(a);
        const row_mask = max_val.cmp(.GT, Tensor.scalar(-std.math.inf(f64), self.dtype()));

        const exp_diff_max = self.sub(self.max(a).broad(self._shape)).exp();
        const res = exp_diff_max.div(exp_diff_max.sum(a).broad(self._shape));

        // If a row is full -inf return full 0 instead of full nan,
        // this fix attention when mask hides a full row.
        return row_mask.broad(self.shape()).select(res, Tensor.scalar(0, self.dtype()));
    }

    inline fn toI64(values: anytype) []i64 {
        var res: [Tensor.MAX_RANK]i64 = undefined;
        for (values, 0..) |val, i| res[i] = @intCast(val);
        return res[0..values.len];
    }

    fn transposeIsJustAReshape(x: Shape, permutation: []const i64) bool {
        var perm: stdx.BoundedArray(struct { u8, bool }, Tensor.MAX_RANK) = .{};
        // Don't rewrite on invalid inputs.
        if (permutation.len > x.rank()) return false;
        for (permutation) |ax| {
            const squeezable = x.dim(ax) == 1;
            perm.appendAssumeCapacity(.{ @intCast(ax), squeezable });
        }

        var effective_ax: u8 = 0;
        for (0..perm.len) |i| {
            const ax, const squeezable = perm.get(i);
            if (squeezable) {
                // Effectively squeeze this axis by decrementing axes coming after by 1.
                for (i..perm.len) |j| {
                    if (perm.buffer[j][0] > ax) {
                        perm.buffer[j][0] -= 1;
                    }
                }
                continue;
            }

            if (ax != effective_ax) return false;
            effective_ax += 1;
        }

        return true;
    }

    /// Returns a transposed Tensor computed using the given axes.
    pub fn transpose(self: Tensor, axes_: anytype) Tensor {
        const axes__ = self.axes(axes_).constSlice();
        const default_perm = [MAX_RANK]i64{ 7, 6, 5, 4, 3, 2, 1, 0 };
        const no_op = [MAX_RANK]i64{ 0, 1, 2, 3, 4, 5, 6, 7 };

        const permutation: []const i64 = if (axes__.len == 0)
            default_perm[MAX_RANK - self.rank() ..]
        else
            toI64(axes__);

        stdx.debug.assert(permutation.len == self.rank(), "transpose expects input tensor rank and 'axes_' length to be equal, got {f} and {any}", .{ self, permutation[0..@min(permutation.len, MAX_RANK + 2)] });

        if (std.mem.eql(i64, permutation, no_op[0..self.rank()])) {
            return self;
        }

        const res_shape = self._shape.transpose(permutation);
        if (transposeIsJustAReshape(self.shape(), permutation)) {
            return self.reshape(res_shape);
        }

        const tensor_type = mlir.rankedTensorType(res_shape.dims(), zml.mlir.Type.fromDType(mlirCtx(), res_shape.dtype()));
        const op = dialects.stablehlo.transpose(
            mlirCtx(),
            self.value(),
            tensor_type,
            .{ .permutation = toI64(permutation) },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(res_shape, op.result(0));
    }

    /// Embeds a buffer with concrete values into an Mlir program.
    pub fn constantTensor(bytes: []const u8, shape_: zml.Shape) Tensor {
        const elem_type = zml.mlir.Type.fromDType(mlirCtx(), shape_.dtype());
        //const elem_type = zml.mlir.denseElementAttrType(shape.dtype()) orelse std.debug.panic("constantTensor expects a dtype that can be serialized to MLIR, like f32 or i32, got {f}", .{shape.dtype()});
        const constant_op = dialects.stablehlo.constant(mlirCtx(), shape_.dims(), elem_type, bytes, .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(shape_, constant_op.result(0));
    }

    /// Inserts 1-dim axes at the given position, with the given tags.
    /// `.{.a = 5, .b = 4}.insert(.b, .{ .c, .d }) -> .{ .a = 5, .c = 1, .d = 1, .b = 4 }`
    pub fn insertAxes(self: Tensor, axis_: anytype, tags: anytype) Tensor {
        const tags_ = Shape.parseTags(tags);
        const ax = if (@TypeOf(axis_) == EnumLiteral and axis_ == .last)
            self.rank()
        else
            self.axis(axis_);

        var res_shape = self._shape;
        const ones = [_]i64{1} ** MAX_RANK;
        res_shape._dims.insertSlice(ax, ones[0..tags_.len]) catch unreachable;
        res_shape._tags.insertSlice(ax, tags_.constSlice()) catch unreachable;

        return self.reshape(res_shape);
    }

    /// Returns a Tensor containing the maximum over a given axis.
    pub fn max(self: Tensor, axis_: anytype) Tensor {
        const a = self.axis(axis_);
        const result = reduce(.{self}, .{Tensor.constant(self.dtype().minValue())}, &.{a}, struct {
            pub fn cmp(values: ReduceArgs) struct { Tensor } {
                return .{values.left.maximum(values.right.convert(values.left.dtype()))};
            }
        }.cmp, .{});
        return result[0];
    }

    /// Returns a Tensor containing the element-wise exponential operation of the input Tensor.
    pub fn exp(self: Tensor) Tensor {
        const op = dialects.stablehlo.exponential(mlirCtx(), self.value(), .unknown(mlirCtx())).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    /// Generalized version of scatter to many inputs.
    /// See `zml.Tensor.scatterSlices` for documentation on scatter.
    ///
    /// This allows to use the same indices to update several tensors at once,
    /// and where the update function is allow to look at elements from the different tensors
    /// to compute the final value.
    ///
    /// This sounds nice but in practice XLA doesn't support this well on GPU,
    /// and will generate slow code. In practice stick with `zml.Tensor.scatterSlices`.
    pub fn scatter(
        inputs: anytype,
        index_tensors: anytype,
        updates: anytype,
        comptime func: anytype,
        context: anytype,
        opts: Tensor.ScatterOpts,
    ) stdx.meta.FnResult(func) {
        var arena = std.heap.ArenaAllocator.init(CompilationContext.current().allocator);
        defer arena.deinit();

        const update_block, var result = b: {
            const ArgsType = std.meta.Tuple(&[1]type{ReduceArgs} ** inputs.len);
            var args: ArgsType = undefined;
            var block_types: [2 * inputs.len]*const mlir.Type = undefined;

            inline for (0..inputs.len) |i| {
                args[i].left = Tensor.init(zml.Shape.init(.{}, inputs[i].dtype()));
                args[i].right = Tensor.init(zml.Shape.init(.{}, inputs[i].dtype()));

                block_types[i] = mlir.rankedTensorType(args[i].left.dims(), zml.mlir.Type.fromDType(mlirCtx(), args[i].left.dtype()));
                block_types[i + inputs.len] = mlir.rankedTensorType(args[i].right.dims(), zml.mlir.Type.fromDType(mlirCtx(), args[i].right.dtype()));
            }

            const block_locs: [2 * inputs.len]*const mlir.Location = @splat(mlir.Location.unknown(mlirCtx()));
            const update_block = mlir.Block.init(&block_types, &block_locs);
            errdefer update_block.deinit();

            CompilationContext.current().pushBlock(update_block);
            defer CompilationContext.current().popBlock();

            const scope = CompilationContext.current().currentScope();
            inline for (0..inputs.len) |i| {
                scope.id_to_argument.put(scope.arena.allocator(), args[i].left.id, i) catch unreachable;
                scope.id_to_argument.put(scope.arena.allocator(), args[i].right.id, i + inputs.len) catch unreachable;
            }

            var result = @call(.auto, func, args ++ context);

            var result_values: [inputs.len]*const mlir.Value = undefined;
            inline for (0..inputs.len) |i| {
                result_values[i] = result[i].value();
            }

            _ = dialects.stablehlo.returns(mlirCtx(), &result_values, .unknown(mlirCtx())).appendTo(update_block);
            break :b .{ update_block, result };
        };

        // Note: I was a bit lazy here, and I only look at tags on the first tensor.
        // we probably should check all of them.
        const self = zml.meta.first(Tensor, inputs);
        const update = zml.meta.first(Tensor, updates);
        var indices_per_axis, var indices_axes = Shape.parseStruct(Tensor, index_tensors);

        if (indices_per_axis.len == 0) return inputs;

        // validate coord axes: all coord_axes should exist inside self
        for (indices_axes.constSlice()) |t| {
            stdx.debug.assert(self._shape.hasTag(t) != null, "zml.ops.scatter expects axes of indices to be axes of inputs, got input={f} and indices={any}", .{ self, indices_axes.constSlice() });
        }

        // Handle scalar indices by broadcasting them to the indices with the highest rank.
        const indices_shape = blk: {
            var higher_rank = indices_per_axis.get(0).shape();
            for (indices_per_axis.constSlice()[1..]) |indices| {
                if (indices.rank() > higher_rank.rank()) {
                    higher_rank = indices.shape();
                }
            }
            break :blk higher_rank;
        };
        for (indices_per_axis.slice()) |*idx| {
            stdx.debug.assert(idx.shape().canBroadcastTo(indices_shape), "zml.ops.scatter expects all indices tensor to have the same shape, got {any}", .{indices_per_axis.slice()});
            stdx.debug.assert(idx.dtype() == indices_shape.dtype(), "zml.ops.scatter expects all indices tensor to have the same dtype, got {any}", .{indices_per_axis.slice()});
            idx.* = idx.broad(indices_shape);
        }

        // rewrite simple scatters to dynamicUpdateSlice.
        if (@TypeOf(inputs) == struct { Tensor } and indices_shape.rank() == 0) {
            return .{self.dynamicUpdateSlice(index_tensors, update)};
        }

        // TODO: ideally we should catch all possible scatter errors and provide nice error messages.
        var config = scatterConfig(self.shape(), update.shape(), indices_per_axis, indices_axes);
        const indices = scatterPrepareIndices(&config, self.shape(), update.shape(), &indices_per_axis, &indices_axes);
        // const n_indices_axes = update.rank() - _collectAxes(AxisKind, up_kind, .update_window).len;
        // stdx.debug.assert(n_indices_axe == indices_axes.len, "scatter({f}, {any}) expects 'updates' to contain all axes from 'indices', got indices={s}, updates={f}", .{ self, index_tensors, indices_axes.constSlice(), update });

        var input_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..inputs.len) |i| {
            input_values[i] = inputs[i].value();
        }

        var updates_values: [inputs.len]*const mlir.Value = undefined;
        inline for (0..updates.len) |i| updates_values[i] = updates[i].value();

        const op = dialects.stablehlo.scatter(
            mlirCtx(),
            &input_values,
            &.{indices.value()},
            &updates_values,
            update_block,
            .{
                .update_window_dims = _collectAxes(AxisKind, config.up_kind, .update_window).constSlice(),
                .inserted_window_dims = _collectAxes(AxisKind, config.op_kind, .inserted_window).constSlice(),
                .input_batching_dims = _collectAxes(AxisKind, config.op_kind, .batching).constSlice(),
                .scatter_indices_batching_dims = config.indices_batch_axes.constSlice(),
                .scatter_dims_to_operand_dims = config.scatter_to_operand_axes.constSlice(),
                .index_vector_dim = indices.rank() - 1,
                .indices_are_sorted = opts.indices_are_sorted,
                .unique_indices = opts.indices_are_unique,
            },
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());

        inline for (0..result.len) |i| {
            result[i] = Tensor._result(inputs[i].shape(), op.result(i));
        }

        return result;
    }

    const ScatterConfig = struct {
        op_kind: stdx.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{},
        up_kind: stdx.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{},
        indices_batch_axes: Shape.DimsArray = .{},
        scatter_to_operand_axes: Shape.DimsArray = .{},
        updates_transpose: Shape.AxesArray = .{},
    };

    const AxisKind = enum { batching, update_window, inserted_window, window_id };

    fn scatterConfig(
        op: Shape,
        update: Shape,
        indices_per_axis: stdx.BoundedArray(Tensor, Tensor.MAX_RANK),
        indices_axes: Shape.TagsArray,
    ) ScatterConfig {
        var op_kind: stdx.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{};
        var up_kind: stdx.BoundedArray(AxisKind, Tensor.MAX_RANK) = .{};
        var indices_batch_axes: Shape.DimsArray = .{};
        var scatter_to_operand_axes: Shape.DimsArray = .{};
        var updates_transpose: Shape.AxesArray = .{};

        const tagged_api = indices_axes.len > 0;
        const indices = indices_per_axis.get(0).shape();

        if (tagged_api) {
            for (indices_axes.constSlice()) |t| {
                scatter_to_operand_axes.appendAssumeCapacity(op.axis(t));
            }
            for (indices.tags()) |t| {
                stdx.debug.assert(update.hasTag(t) != null, "scatter expects 'updates' to have all axes of 'indices', got self={f}, updates={f} and indices={f}", .{ op, update, indices });
                updates_transpose.appendAssumeCapacity(update.axis(t));
            }

            for (op.tags()) |t| {
                if (update.hasTag(t)) |up_ax| {
                    updates_transpose.appendAssumeCapacity(up_ax);

                    if (indices.hasTag(t)) |id_ax| {
                        if (std.mem.indexOfScalar(Shape.Tag, indices_axes.constSlice(), t) != null) {
                            // tag is in indices AND in coords -> it's a batching dim that has been rewritten to a regular insertion dim
                            op_kind.appendAssumeCapacity(.inserted_window);
                        } else {
                            // tag is in op, indices and updates -> it's a batching dim
                            op_kind.appendAssumeCapacity(.batching);
                            indices_batch_axes.appendAssumeCapacity(@intCast(id_ax));
                        }
                    } else {
                        op_kind.appendAssumeCapacity(.update_window);
                    }
                } else {
                    op_kind.appendAssumeCapacity(.inserted_window);
                }
            }

            for (update.tags(), 0..) |t, up_ax| {
                // Handle batch axes right away.
                if (op.hasTag(t)) |self_ax| {
                    if (op_kind.get(self_ax) == .batching) {
                        up_kind.appendAssumeCapacity(.batching);
                        continue;
                    }
                }
                if (indices.hasTag(t) != null) {
                    up_kind.appendAssumeCapacity(.window_id);
                } else if (op.hasTag(t)) |self_ax| {
                    stdx.debug.assert(update.dim(up_ax) <= op.dim(self_ax), "scatter expects the slices described in 'updates' to fit inside 'op', but along axis .{s} it doesn't. Got op={f}, updates={f}.", .{ t, op, update });
                    up_kind.appendAssumeCapacity(.update_window);
                } else {
                    // TODO: consider accepting untagged update here.
                    std.debug.panic("scatter expects 'updates' to be made of axes from op={f} and from indices={any}, got unknown tag {s} in {f}", .{ op, indices_axes.constSlice(), std.mem.sliceTo(t, 0), update });
                }
            }
        } else {
            for (0..indices_per_axis.len) |i| {
                op_kind.appendAssumeCapacity(.inserted_window);
                scatter_to_operand_axes.appendAssumeCapacity(@intCast(i));
                up_kind.appendAssumeCapacity(.window_id);
            }
            for (indices_per_axis.len..op.rank()) |_| {
                op_kind.appendAssumeCapacity(.update_window);
            }
            for (indices_per_axis.len..update.rank()) |_| {
                up_kind.appendAssumeCapacity(.update_window);
            }
            for (0..update.rank()) |i| {
                updates_transpose.appendAssumeCapacity(@intCast(i));
            }
        }

        return .{
            .op_kind = op_kind,
            .up_kind = up_kind,
            .indices_batch_axes = indices_batch_axes,
            .scatter_to_operand_axes = scatter_to_operand_axes,
            .updates_transpose = updates_transpose,
        };
    }

    //test scatterConfig {
    //    const zml = @import("zml.zig");
    //    const platform = zml.testing.env();

    //    var comp = try zml.module.CompilationContext.init(std.testing.allocator, "test", platform);
    //    defer comp.deinit();
    //    comp.activate();
    //    defer comp.deactivate();

    //    const Local = struct {
    //        pub fn _idx(idx_shape: anytype) Tensor {
    //            return Tensor.constant(idx_shape, .{ .i32 = 0 });
    //        }
    //    };

    //    const idx = Local._idx;
    //    const op = Shape.init(.{ .a = 10, .b = 20 }, .f32);

    //    // Use .a as a batching axis with .a=10 x .n=8 updates of 2 elements of .b
    //    {
    //        const indices, const coords_tags = Shape.parseStruct(Tensor, .{ .b = idx(.{ .a = 10, .n = 8 }) });
    //        const update = Shape.init(.{ .a = 10, .n = 8, .b = 2 }, .f32);

    //        const cfg = scatterConfig(op, update, indices, coords_tags);
    //        try std.testing.expectEqualSlices(AxisKind, &.{ .batching, .update_window }, cfg.op_kind.constSlice());
    //        try std.testing.expectEqualSlices(AxisKind, &.{ .batching, .window_id, .update_window }, cfg.up_kind.constSlice());
    //    }

    //    // similar, but use the normalized form where .a is no longer an explicit batching axis.
    //    {
    //        const indices, const coords_tags = Shape.parseStruct(Tensor, .{ .a = idx(.{ .a = 10, .n = 8 }), .b = idx(.{ .a = 10, .n = 8 }) });
    //        const update = Shape.init(.{ .a = 10, .n = 8, .b = 2 }, .f32);

    //        const cfg = scatterConfig(op, update, indices, coords_tags);
    //        try std.testing.expectEqualSlices(AxisKind, &.{ .inserted_window, .update_window }, cfg.op_kind.constSlice());
    //        try std.testing.expectEqualSlices(AxisKind, &.{ .window_id, .window_id, .update_window }, cfg.up_kind.constSlice());
    //    }
    //}

    /// Concatenate all indices tensor in one tensor.
    ///
    /// Is allowed to reorder stuff to simplify the job of the backend,
    /// and to expand the batching dims.
    fn scatterPrepareIndices(
        cfg: *ScatterConfig,
        op: Shape,
        update: Shape,
        indices_per_axis: *stdx.BoundedArray(Tensor, Tensor.MAX_RANK),
        indices_axes: *Shape.TagsArray,
    ) Tensor {
        var old_scatter_to_op_axes = cfg.scatter_to_operand_axes;
        const batching = _collectAxes(AxisKind, cfg.op_kind, .batching);
        for (batching.constSlice()) |batch_ax| {
            const id_shape = indices_per_axis.get(0).shape();
            // batching requires tagging, so we're sure to have a tag here.
            const batch_tag = op.tag(batch_ax);
            indices_axes.appendAssumeCapacity(batch_tag);
            const batch_id = Tensor.iota(id_shape, batch_tag).convert(id_shape.dtype());
            indices_per_axis.appendAssumeCapacity(batch_id);
            cfg.op_kind.buffer[@intCast(batch_ax)] = .inserted_window;
            cfg.up_kind.buffer[update.axis(batch_tag)] = .window_id;
            old_scatter_to_op_axes.appendAssumeCapacity(batch_ax);
        }
        cfg.indices_batch_axes = .{};

        // Reorder the axes so that in indices_per_axis is ordered like in op if possible.
        // TODO: transpose updates if needed
        var indices: stdx.BoundedArray(Tensor, Tensor.MAX_RANK) = .{};
        var scatter_to_op_axes: Shape.DimsArray = .{};

        while (old_scatter_to_op_axes.len > 0) {
            const scatter_ax = std.sort.argMin(i64, old_scatter_to_op_axes.constSlice(), {}, std.sort.asc(i64)).?;
            const op_ax = old_scatter_to_op_axes.orderedRemove(scatter_ax);
            const scatter_idx = indices_per_axis.orderedRemove(scatter_ax);

            scatter_to_op_axes.appendAssumeCapacity(op_ax);
            indices.appendAssumeCapacity(scatter_idx);
        }
        cfg.scatter_to_operand_axes = scatter_to_op_axes;

        for (scatter_to_op_axes.constSlice(), 0..) |sc_ax, i| {
            if (i != sc_ax) {
                //log.warn("Found a slow scatter pattern, which is going to generate a while loop: scatter({f}, {any}, {f}). Because the index axes aren't the major ones in the input tensor.", .{ op, scatter_to_op_axes.constSlice(), update });
                break;
            }
        }
        return Tensor.stack(indices.constSlice(), .last, .coord);
    }

    pub const ScatterOpts = struct {
        /// Promise scatter that all coordinates in `indices` are sorted, wrt to the final offset in `self`
        /// Result is undefined if the promise is violated.
        indices_are_sorted: bool = false,

        /// Promise scatter that slices don't overlap.
        /// Result is undefined if the promise is violated.
        /// This allows for better code generation, because it means that updates can be applied in parallel.
        indices_are_unique: bool = false,

        /// Function used to update previous value in `self` with values from `updates`.
        /// If `update_fn` is not associative (ie the order of execution matters),
        /// then you should make sure the slices don't overlap,
        /// otherwise the result will depend on the runtime scheduling
        /// of the operator which is backend specific.
        update_fn: *const fn (ReduceArgs) struct { Tensor } = increment,

        pub fn increment(values: ReduceArgs) struct { Tensor } {
            return .{values.left.add(values.right)};
        }

        pub fn override(values: ReduceArgs) struct { Tensor } {
            return .{values.right};
        }
    };

    /// Update the given tensor, by copying `values` into slice by slice into `self`.
    /// The slices are chosen at runtime by interpreting indices as coordinates into `self`.
    /// This is a generalized version of `dynamicUpdateSlice` where more than one offset can be specified at a time.
    ///
    /// ### Arguments
    ///
    /// - Return a tensor with same shape than `self`, with updated content.
    /// - `indices` is a set of Tensor (typically rank 1), representing coordinates into `self`.
    ///   all indices must have the same shape, but scalars are accepted.
    /// - each `indices` entry contains offset along an axes into `self`.
    /// Typically axes are identified by their tags, but in the absence of tags on `indices`,
    /// The entry in indices will be assigned to axes of `self` from major to minor axis.
    /// It is recommended to have indices referencing only major axes of `self` for better performance.
    /// - `values` shape is obtained by concatenating the shape of `indices` with the shape of the slices to be extracted.
    /// - `opts`: `zml.Tensor.ScatterOpts` des
    ///
    /// ### Sample input/output shapes with corresponding pseudo-code.
    ///
    /// Basic `scatterSlices` with the first two axes (.a, .b) being indexed, and full (.c, .d) slice copies:
    ///
    /// ```
    /// fn scatterSlices(x[A, B, C, D], .{.a=off_a[N], .b=off_b[N]}, y[N, C, D]) [A, B, C, D] {
    ///     var z = x;
    ///     for (0..N) |n| {
    ///         for (0..C) |c| for (0..D) |d| {{
    ///             z[off_a[n],off_b[n],c,d] += y[n, c, d];
    ///         }}
    ///     }
    ///     return z;
    /// }
    /// ```
    ///
    /// `scatterSlices` with the first three axes (.a, .b, .c) being indexed, and a partial copy of (.c, .d).
    /// Note that .c axis is present both in the indices and updates, and `updates.dim(.c) < self.dim(.c)`.
    ///
    /// ```
    /// fn scatterSlices(x[A, B, C, D], .{.a=off_a[N], .b=off_b[N], .c=off_c[N]}, y[N, C', D]) [A, B, C, D] {
    ///     var z = x;
    ///     for (0..N) |n| {
    ///        for (0..C') |c| for (0..D) |d| {{
    ///           z[off_a[n],off_b[n],off_c[n]+c,d] += y[n, c, d];
    ///        }}
    ///     }
    ///     return z;
    /// }
    /// ```
    ///
    /// `scatterSlices` with the first axis .a being indexed, and where .b is used as a batching axis.
    /// Note that here .b axis is present in `self`, `off_a`, and `updates`,
    /// and is not mentionned in the axes of indices.
    ///
    /// ```
    /// fn scatterSlices(x[A, B, C, D], .{.a=off_a[B,N]}, y[N, B, C, D]) [A, B, C, D] {
    ///     var z = x;
    ///     for (0..B) |b| {
    ///         for (0..N) |n| {
    ///             for (0..C) |c| for (0..D) |d| {{
    ///                 z[off_a[b,n],b,c,d] += y[n, b, c, d];
    ///             }}
    ///         }
    ///     }
    ///     return z;
    /// }
    /// ```
    ///
    /// ### Warnings
    ///
    /// - if `opts.update_fn` is not associative not all calls to `scatterSlices` are sound.
    /// In particular if you scatter overlapping slices, with `zml.Tensor.ScatterOpts.override`,
    /// then the result will depend on the execution order that you don't control.
    /// - `scatterSlices` is a very expressive operator, and can lead to complicated code generation
    /// that requires host<->device synchronization.
    /// ZML tries to generate the easiest to optimize IR, and will warn you if it generates known problematic IR.
    pub fn scatterSlices(self: Tensor, indices: anytype, updates: Tensor, opts: ScatterOpts) Tensor {
        //scoped_log.debug("scatterSlices({}, {any}, {})", .{ self, indices, updates });

        const UpdateType = @TypeOf(ScatterOpts.increment);

        const Custom = struct {
            pub fn inc(values: ReduceArgs, custom: *const UpdateType) struct { Tensor } {
                return @call(.auto, custom, .{values});
            }
        };

        return scatter(.{self}, indices, .{updates}, Custom.inc, .{opts.update_fn}, opts)[0];
    }

    /// Concatenates the input Tensors along a new axis. The Tensors must have the same shape.
    /// For x, y, z of shape .{ .a = 10, .b = 11, .c = 12 }:
    /// - Tensor.stack(&.{x, y, z}, .b, .layers) -> .{ .a, .layers, .b, .c }
    /// - Tensor.stack(&.{x, y, z}, 1, .layers) -> .{ .a, .layers, .b, .c }
    /// - Tensor.stack(&.{x, y, z}, .last, .layers) -> .{ .a, .b, .c, .layers }
    pub fn stack(tensors: []const Tensor, axis_: anytype, tag: anytype) Tensor {
        // Note: we could ask the compilation context for some memory instead of stack allocating
        stdx.debug.assert(tensors.len <= 32, "stack only supports up to 32 tensors, got {}", .{tensors.len});

        const shape0 = tensors[0]._shape;
        const res_shape = shape0.insertTag(axis_, 1, tag);

        for (tensors[1..]) |tensor| {
            stdx.debug.assert(shape0.eqlWithTags(tensor._shape), "stack expects tensor shapes to match, got {f} and {f}", .{ shape0, tensor._shape });
        }

        var reshaped: [32]Tensor = undefined;
        for (tensors, 0..) |tensor, i| {
            reshaped[i] = tensor.reshape(res_shape);
        }

        // Be careful here: we need to resolve ax before calling concatenate,
        // because we added an axis, so all
        const ax = if (@TypeOf(axis_) == EnumLiteral and axis_ == .last)
            shape0.rank()
        else
            shape0.axis(axis_);

        return Tensor.concatenate(reshaped[0..tensors.len], ax);
    }

    /// Updates a part of the input Tensor using the given 'update' Tensor, with runtime known offsets.
    ///
    /// The offsets are specified similarly to the dynamicSlice api.
    /// It's semantically equivalent to:
    /// self.dynamicSlice(offsets_) := update
    /// Examples:
    /// ```
    /// Tensor(.{ .a = 2, .b = 5 }).dynamicUpdateSlice(.{ .a = scalar(1, .i32) }, Tensor(.{ .b = 5 }));
    /// ```
    pub fn dynamicUpdateSlice(self: Tensor, offset_: anytype, update_: Tensor) Tensor {
        // TODO: add updateSlice for when the offset isn't dynamic
        stdx.debug.assert(self.dtype() == update_.dtype(), "dynamicUpdateSlice expects input and 'update_' tensors to be of the same type, got {} and {}", .{ self.dtype(), update_.dtype() });

        const offset, const offset_tags = Shape.parseStruct(Tensor, offset_);
        // log.debug("offset: {any}, offset_tags: {any}", .{ offset, offset_tags });
        for (offset.constSlice(), 0..) |start_idx, i| {
            stdx.debug.assert(start_idx.rank() == 0, "dynamicUpdateSlice expects 'offset_' tensor ranks to be equal to 0, got {} at index {}", .{ start_idx.rank(), i });
        }

        const tagged_api = update_._shape.isFullyTagged() and self._shape.isFullyTagged() and offset_tags.len > 0;
        // When using tags, we can safely insert axis with a 1-dim.
        // the offset into the inserted axis will need to be specified through indices.
        var update = update_;
        if (tagged_api) {
            // Check that all update tags are known.
            for (update._shape._tags.constSlice()) |t| {
                stdx.debug.assert(self._shape.hasTag(t) != null, "dynamicUpdateSlice expects 'update_' tensor tags to be a subset of input tensor tags but {s} is missing (input shape is {f})", .{ t, self._shape });
            }

            var update_shape = self._shape;
            var prev_ax: i8 = -1;
            for (self._shape.tags(), 0..) |t, self_ax| {
                if (update._shape.hasTag(t)) |up_ax| {
                    stdx.debug.assert(up_ax == prev_ax + 1, "dynamicUpdateSlice expects 'update_' and input tensor axis to have the same order, got {f} and {f}. (hint: you need to explicitly transpose 'update_')", .{ update_, self });

                    update_shape._dims.set(self_ax, update.dim(up_ax));
                    prev_ax = up_ax;
                } else {
                    update_shape._dims.set(self_ax, 1);
                }
            }
            update = update.reshape(update_shape);
        }

        stdx.debug.assert(self.rank() == update.rank(), "dynamicUpdateSlice expects input and computed update tensors to have the same rank, got {f} and {f}", .{ self, update });

        for (self.dims(), update.dims(), 0..) |self_d, up_d, ax| {
            const t = self._shape.debugTag(ax);
            stdx.debug.assert(up_d <= self_d, "dynamicUpdateSlice expects 'update_' dimensions to be less than or equal to their corresponding dimension in input tensor, got {} and {} for axis .{s}", .{ up_d, self_d, t });

            if (tagged_api and up_d < self_d) {
                const axis_has_offset = std.mem.indexOfScalar(Shape.Tag, offset_tags.constSlice(), self._shape._tags.get(ax)) != null;

                stdx.debug.assert(axis_has_offset, "dynamicUpdateSlice expects 'update_' dimensions to be equal to their corresponding dimension in input tensor, got {} and {} for axis .{s} (hint: you need to provide an offset)", .{ up_d, self_d, t });
            }
        }

        const idx_dtype = if (offset.len > 0) offset.get(0).dtype() else .i32;
        const zero = Tensor.scalar(0, idx_dtype).value();
        var offset_values: [MAX_RANK]*const mlir.Value = undefined;
        if (offset_tags.len == 0) {
            // Without offset tags we need the same number of offset than rank.
            stdx.debug.assert(self.rank() == offset.len, "dynamicUpdateSlice expects input tensor rank and 'offset_' length to be equal, got {} and {}", .{ self.rank(), offset.len });

            for (offset.constSlice(), 0..) |idx, i| {
                offset_values[i] = idx.value();
            }
        } else {
            // If an axis isn't specified, update the full slice.
            // This is only allowed when using tagged sliced.
            offset_values = .{zero} ** MAX_RANK;
            for (offset.constSlice(), offset_tags.constSlice()) |start, t| {
                const a = self._shape.hasTag(t) orelse stdx.debug.panic("dynamicUpdateSlice expects input tensor to have tags used in 'offset_' but {s} is missing (input shape is {f})", .{ t, self._shape });
                offset_values[a] = start.value();
            }
        }

        const op = dialects.stablehlo.dynamic_update_slice(
            mlirCtx(),
            self.value(),
            update.value(),
            offset_values[0..self.rank()],
            .unknown(mlirCtx()),
        ).appendTo(currentBlock());
        return _result(self._shape, op.result(0));
    }

    pub fn getContext(self: Tensor) *CompilationContext {
        _ = self; // autofix
        return global_compilation_context.?;
    }

    pub fn format(
        self: Tensor,
        writer: *std.Io.Writer,
    ) !void {
        // TODO(0.15.0) handle format
        // const bare_fmt = fmt.len == 1 and fmt[0] == '_';
        try writer.print("{s}({f})", .{ @typeName(Tensor), self._shape });
    }
};

fn _parseGatherCoord(self: Tensor, axes_: anytype) struct { bool, stdx.BoundedArray(u3, Tensor.MAX_RANK) } {
    const AxesT = @TypeOf(axes_);
    const axes_is_scalar = AxesT == EnumLiteral or AxesT == comptime_int or @typeInfo(AxesT) == .int;

    const coord_axes = if (axes_is_scalar)
        stdx.BoundedArray(u3, Tensor.MAX_RANK).fromSlice(&.{self.axis(axes_)}) catch unreachable
    else
        self.axes(axes_);

    return .{ axes_is_scalar, coord_axes };
}

pub fn _collectAxes(T: type, bounded_array: stdx.BoundedArray(T, Tensor.MAX_RANK), value: T) stdx.BoundedArray(i64, Tensor.MAX_RANK) {
    var res: stdx.BoundedArray(i64, Tensor.MAX_RANK) = .{};
    for (bounded_array.constSlice(), 0..) |v, ax| {
        if (v == value) {
            res.appendAssumeCapacity(@intCast(ax));
        }
    }
    return res;
}

pub const TransferEntry = struct {
    buffer: zml.Buffer,
    writer: *Transfer.Writer,
};

pub fn singleTransfer(allocator: std.mem.Allocator, shape: Shape, platform: Platform) !struct { Transfer, TransferEntry } {
    var transfer = try Transfer.init(allocator, &.{shape}, platform);
    errdefer transfer.deinit(allocator, platform);

    const entry: TransferEntry = .{
        .buffer = zml.Buffer.fromPjrtBuffers(platform, shape, &.{transfer.get(0).buffer}),
        .writer = &transfer.get(0).writer,
    };

    return .{ transfer, entry };
}

pub fn multiTransfer(allocator: std.mem.Allocator, shapes: anytype, platform: Platform) !struct { Transfer, [shapes.len]TransferEntry } {
    var shapes_array: [shapes.len]zml.Shape = undefined;
    inline for (shapes, 0..) |shape, index| shapes_array[index] = shape;

    var transfer = try Transfer.init(allocator, &shapes_array, platform);
    errdefer transfer.deinit(allocator, platform);

    var entries: [shapes.len]TransferEntry = undefined;
    inline for (shapes, 0..) |shape, index| {
        entries[index].buffer = zml.Buffer.fromPjrtBuffers(platform, shape, &.{transfer.get(index).buffer});
        entries[index].writer = &transfer.get(index).writer;
    }

    return .{ transfer, entries };
}

pub fn bufferTransfer(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, name: []const u8, platform: Platform) !zml.Buffer {
    const shape = buffer_store.getShape(name).?;
    var transfer, const entry = try singleTransfer(allocator, shape, platform);
    defer transfer.deinit(platform);

    const tensor_desc = buffer_store.store.getPtrFromKey(name).?;

    var tensor_resource: Resource = .{
        .http = try .init(buffer_store.store.http_client, allocator, tensor_desc.resource_uri, .{ .offset = tensor_desc.offset, .size = tensor_desc.shape.byteSize() }),
    };
    defer tensor_resource.deinit();

    const buffer_reader = try allocator.alloc(u8, BUF_16_MB);
    defer allocator.free(buffer_reader);

    var reader = tensor_resource.reader(buffer_reader);

    _ = try reader.interface().streamRemaining(&entry.writer.interface);
    try entry.writer.interface.flush();
    return entry.buffer;
}

pub fn autoLoad(allocator: std.mem.Allocator, model: anytype, buffer_store: BufferStore5.View, platform: Platform) !Bufferized(@TypeOf(model)) {
    const Model = @TypeOf(model);
    var result: Bufferized(Model) = undefined;

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const shapes = try collectShapes(arena.allocator(), &model);

    var transfer: Transfer = try .init(arena.allocator(), shapes, platform);
    defer transfer.deinit(arena.allocator(), platform);

    const readers = try arena.allocator().alloc(BufferStore5.Reader, shapes.len);

    {
        var index: usize = 0;
        const type_info = @typeInfo(Model);
        switch (type_info) {
            .@"struct" => |struct_type_info| {
                inline for (struct_type_info.fields) |field| {
                    const reader = buffer_store.getReader(field.name);
                    readers[index] = reader;
                    index += 1;
                }
            },
            else => unreachable,
        }
    }

    const LocalContext = struct {
        readers: []BufferStore5.Reader,
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
    };
    var context: LocalContext = .{ .readers = readers, .shapes = shapes, .platform = platform, .transfer = &transfer };
    zml.meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
            const writer = context_.transfer.get(context_.index).writer();
            context_.readers[context_.index].stream(writer);

            buffer.* = zml.Buffer.fromPjrtBuffers(context_.platform, context_.shapes[context_.index], &.{context_.transfer.get(context_.index).buffer});
            context_.index += 1;
        }
    }.cb, &context, &result);

    return result;
}

pub fn loadBuffersFromId(allocator: std.mem.Allocator, model: anytype, buffer_store: BufferStore5.View, platform: Platform) !Bufferized(@TypeOf(model)) {
    //const id = zml.tools.tracer.tracer().frameStart("loadBuffersFromId");
    //defer zml.tools.tracer.tracer().frameEnd(id, "loadBuffersFromId");

    const Model = @TypeOf(model);
    var result: Bufferized(Model) = undefined;
    initBufferizedFrom(model, &result);

    var arena: std.heap.ArenaAllocator = .init(allocator);
    defer arena.deinit();

    const shapes = try collectShapes(arena.allocator(), &model);
    std.debug.print("Found {d} shapes\n", .{shapes.len});

    var transfer: Transfer = try .init(arena.allocator(), shapes, platform);
    defer transfer.deinit(platform);

    const tensor_descs = try collectTensorDesc(arena.allocator(), buffer_store, &model);

    const buffer_reader = try allocator.alloc(u8, BUF_16_MB);
    defer allocator.free(buffer_reader);

    const LocalContext = struct {
        tensor_descs: []TensorDesc,
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
        buffer_reader: []u8,
        buffer_store_view: BufferStore5.View,
        allocator: std.mem.Allocator,
    };
    var context: LocalContext = .{
        .tensor_descs = tensor_descs,
        .shapes = shapes,
        .platform = platform,
        .transfer = &transfer,
        .buffer_reader = buffer_reader,
        .buffer_store_view = buffer_store,
        .allocator = allocator,
    };
    zml.meta.visit(struct {
        fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
            const tensor_desc = context_.tensor_descs[context_.index];

            var tensor_resource: Resource = .{
                .http = HttpResource.init(context_.buffer_store_view.store.http_client, context_.allocator, tensor_desc.resource_uri, .{ .offset = tensor_desc.offset, .size = tensor_desc.shape.byteSize() }) catch unreachable,
            };
            defer tensor_resource.deinit();

            var reader = tensor_resource.reader(context_.buffer_reader);
            const entry = context_.transfer.get(context_.index);

            _ = reader.interface().streamRemaining(&entry.writer.interface) catch unreachable;
            entry.writer.interface.flush() catch unreachable;

            buffer.* = zml.Buffer.fromPjrtBuffers(context_.platform, context_.shapes[context_.index], &.{context_.transfer.get(context_.index).buffer});
            context_.index += 1;
        }
    }.cb, &context, &result);

    return result;
}

pub fn initBufferizedFrom(model: anytype, bufferized_: *Bufferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(Bufferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (Bufferized(Model) == zml.Buffer) return;
            inline for (struct_type_info.fields) |field| {
                initBufferizedFrom(@field(model, field.name), &@field(bufferized_, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    bufferized_.* = @unionInit(Bufferized(Model), @tagName(tag), undefined);
                    initBufferizedFrom(v, @field(bufferized_, @tagName(tag)));
                },
            }
        },
        .optional => {
            if (model == null) {
                bufferized_.* = null;
            } else {
                bufferized_.* = undefined;
                initBufferizedFrom(model.?, &bufferized_.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

pub fn structTransfer(allocator: std.mem.Allocator, model: anytype, platform: Platform) !struct { Transfer, Transferized(@TypeOf(model)) } {
    const Model = @TypeOf(model);
    var result: Transferized(Model) = undefined;
    initTransferizedFrom(model, &result);

    const shapes = try collectShapes(allocator, &model);
    defer allocator.free(shapes);
    std.debug.print("Found {d} shapes\n", .{shapes.len});

    var transfer: Transfer = try .init(allocator, shapes, platform);
    errdefer transfer.deinit(allocator, platform);

    const LocalContext = struct {
        shapes: []const Shape,
        platform: Platform,
        transfer: *Transfer,
        index: usize = 0,
    };
    var context: LocalContext = .{ .shapes = shapes, .platform = platform, .transfer = &transfer };
    zml.meta.visit(struct {
        fn cb(context_: *LocalContext, transfer_entry: *TransferEntry) void {
            transfer_entry.buffer = zml.Buffer.fromPjrtBuffers(context_.platform, context_.shapes[context_.index], &.{context_.transfer.get(context_.index).buffer});
            transfer_entry.writer = context_.transfer.get(context_.index).writer();
            context_.index += 1;
        }
    }.cb, &context, &result);

    return .{ transfer, result };
}

pub fn Transferized(comptime T: type) type {
    return zml.meta.MapRestrict(Tensor, TransferEntry).map(T);
}

fn initTransferizedFrom(model: anytype, transferized: *Transferized(@TypeOf(model))) void {
    const Model = @TypeOf(model);
    const type_info = @typeInfo(Transferized(Model));
    switch (type_info) {
        .@"struct" => |struct_type_info| {
            if (Transferized(Model) == TransferEntry) return;
            inline for (struct_type_info.fields) |field| {
                initTransferizedFrom(@field(model, field.name), &@field(transferized, field.name));
            }
        },
        .@"union" => {
            switch (model) {
                inline else => |v, tag| {
                    transferized.* = @unionInit(Bufferized(Model), @tagName(tag), undefined);
                    initTransferizedFrom(v, @field(transferized, @tagName(tag)));
                },
            }
        },
        .optional => {
            if (model == null) {
                transferized.* = null;
            } else {
                transferized.* = undefined;
                initTransferizedFrom(model.?, &transferized.*.?);
            }
        },
        .void, .int, .@"enum", .bool, .enum_literal, .float, .pointer, .vector => {},
        else => unreachable,
    }
}

pub fn bufferTypeFromDtype(dt: zml.DataType) pjrt.BufferType {
    return switch (dt) {
        inline else => |tag| @field(pjrt.BufferType, @tagName(tag)),
    };
}

const Transfer = struct {
    entries: []Entry,
    transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
    arena: std.heap.ArenaAllocator,

    pub const Writer = struct {
        offset: usize = 0,
        interface: std.Io.Writer,

        pub fn init(buffer: []u8) Writer {
            return .{
                .interface = .{
                    .buffer = buffer,
                    .end = 0,
                    .vtable = &.{
                        .drain = drain,
                    },
                },
            };
        }

        pub fn drain(w: *std.Io.Writer, data: []const []const u8, splat: usize) std.Io.Writer.Error!usize {
            _ = data; // autofix
            _ = splat; // autofix
            const writer: *Writer = @alignCast(@fieldParentPtr("interface", w));
            const entry: *Entry = @alignCast(@fieldParentPtr("writer", writer));
            stdx.debug.assert(writer.offset + w.end <= entry.shape.byteSize(), "Can't write more data than required", .{});
            const is_last_transfer = writer.offset + w.end >= entry.shape.byteSize();
            log.debug("Writing {} bytes", .{w.end});
            _ = entry.transfer_manager.transferData(entry.platform.pjrt_api, entry.buffer_index, w.buffer[0..w.end], @intCast(writer.offset), is_last_transfer) catch return error.WriteFailed;
            const written = w.end;
            writer.offset += written;
            w.end = 0;
            return 0;
        }
    };

    pub const Entry = struct {
        shape: Shape,
        transfer_manager: *pjrt.AsyncHostToDeviceTransferManager,
        buffer_index: usize,
        buffer: *pjrt.Buffer,
        platform: Platform,
        writer: Writer,
    };

    pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, platform: Platform) !Transfer {
        const shape_specs = try allocator.alloc(pjrt.ShapeSpec, shapes.len);
        defer allocator.free(shape_specs);

        var temp_arena = std.heap.ArenaAllocator.init(allocator);
        defer temp_arena.deinit();

        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        for (shape_specs, shapes) |*spec, shape| {
            const dims = try temp_arena.allocator().dupe(i64, shape.dims());
            spec.* = pjrt.ShapeSpec.init(dims, bufferTypeFromDtype(shape.dtype()));
        }

        const memory = platform.pjrt_client.memoryByKind(platform.pjrt_api, .unpinned_host).?;

        const transfer_manager = try platform.pjrt_client.createBuffersForAsyncHostToDevice(platform.pjrt_api, .{ .shape_specs = shape_specs, .memory = memory });
        errdefer transfer_manager.deinit(platform.pjrt_api);

        const entries = try arena.allocator().alloc(Entry, shapes.len);
        for (entries, shapes, 0..) |*e, shape, index| {
            const buffer = try transfer_manager.retrieveBuffer(platform.pjrt_api, index);
            const writer_buffer = try arena.allocator().alloc(u8, BUF_16_MB);
            e.* = .{
                .shape = shape,
                .transfer_manager = transfer_manager,
                .buffer_index = index,
                .buffer = buffer,
                .platform = platform,
                .writer = .init(writer_buffer),
            };
        }

        return .{ .entries = entries, .transfer_manager = transfer_manager, .arena = arena };
    }

    pub fn deinit(self: Transfer, platform: Platform) void {
        self.arena.deinit();
        self.transfer_manager.deinit(platform.pjrt_api);
    }

    pub fn get(self: *const Transfer, index: usize) *Entry {
        return &self.entries[index];
    }
};

pub fn main() !void {
    try asynk.AsyncThread.main(std.heap.c_allocator, asyncMain);
}

pub fn tokenizePrompt(allocator: std.mem.Allocator, tokenizer: zml.tokenizer.Tokenizer, config: llama.LlamaLM.Config, prompt: []const u8, skip_llama3_encoding: bool) ![]u32 {
    var encoder = try tokenizer.encoder();
    defer encoder.deinit();

    if (skip_llama3_encoding) {
        // Copy so the ownership is the same in both branches.
        return try allocator.dupe(u32, try encoder.encode(prompt));
    }

    const start_header = tokenizer.tokenToId("<|start_header_id|>") orelse return error.NoSuchToken;
    const end_header = tokenizer.tokenToId("<|end_header_id|>") orelse return error.NoSuchToken;
    const user = tokenizer.tokenToId("user") orelse return error.NoSuchToken;
    const assistant = tokenizer.tokenToId("assistant") orelse return error.NoSuchToken;
    const eot = tokenizer.tokenToId("<|eot_id|>") orelse return error.NoSuchToken;
    const newline = (try encoder.encode("\n"))[0];

    var tokens: std.ArrayList(u32) = try .initCapacity(allocator, prompt.len);
    try tokens.appendSlice(allocator, &.{ config.bos_token_id, start_header, user, end_header, newline });

    try tokens.appendSlice(allocator, try encoder.encode(prompt));
    try tokens.appendSlice(allocator, &.{ eot, newline });

    try tokens.appendSlice(allocator, &.{ start_header, assistant, end_header, newline });

    return tokens.toOwnedSlice(allocator);
}

pub fn asyncMain() !void {
    var gpa: std.heap.GeneralPurposeAllocator(.{}) = .init;
    defer _ = gpa.deinit();
    //const allocator = std.heap.c_allocator;
    const allocator = gpa.allocator();

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const process_args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, process_args);

    const model_config_path = try std.fs.path.join(allocator, &.{ process_args[1], "config.json" });
    defer allocator.free(model_config_path);

    const model_weights_path = b: {
        const simple_path = try std.fs.path.join(allocator, &.{ process_args[1], "model.safetensors" });
        if (asynk.File.access(simple_path, .{})) {
            break :b simple_path;
        } else |_| {
            allocator.free(simple_path);
        }

        const sharded_path = try std.fs.path.join(allocator, &.{ process_args[1], "model.safetensors.index.json" });
        break :b sharded_path;
    };
    defer allocator.free(model_weights_path);

    const model_tokenizer_path = try std.fs.path.join(allocator, &.{ process_args[1], "tokenizer.json" });
    defer allocator.free(model_tokenizer_path);

    var tokenizer = blk: {
        std.log.info("Loading tokenizer from {s}", .{model_tokenizer_path});
        var timer = try stdx.time.Timer.start();
        defer std.log.info("Loaded tokenizer from {s} [{D}]", .{ model_tokenizer_path, timer.read() });

        break :blk try zml.tokenizer.Tokenizer.fromFile(allocator, model_tokenizer_path);
    };
    errdefer tokenizer.deinit();

    const config = blk: {
        var config_json_file = try asynk.File.open(model_config_path, .{ .mode = .read_only });
        defer config_json_file.close() catch unreachable;
        var config_json_buffer: [256]u8 = undefined;
        var config_reader = config_json_file.reader(&config_json_buffer);
        var reader = std.json.Reader.init(allocator, &config_reader.interface);
        defer reader.deinit();
        const config_obj = try std.json.parseFromTokenSourceLeaky(llama.LlamaLM.Config, arena.allocator(), &reader, .{ .ignore_unknown_fields = true });
        break :blk config_obj;
    };

    const options: llama.LlamaLM.Options = .{
        .sampling_strategy = null,
        .max_seq_len = 256,
        .qkv_type = .merged,
    };

    var client: std.http.Client = .{ .allocator = allocator, .write_buffer_size = 8192 };
    try client.initDefaultProxies(arena.allocator());
    defer client.deinit();

    const buffer_reader = try allocator.alloc(u8, BUF_4_MB);
    defer allocator.free(buffer_reader);

    //var resource: Resource = .{ .http = try .init(&client, allocator, std.Uri.parse("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/model.safetensors.index.json") catch unreachable, null) };
    var resource: Resource = .{ .http = try .init(&client, allocator, std.Uri.parse("http://localhost:8000/model.safetensors.index.json") catch unreachable, null) };
    defer resource.deinit();

    var reader = resource.reader(buffer_reader);

    var resource_index = try parseSafetensorsIndex(arena.allocator(), &resource, reader.interface());
    defer resource_index.deinit();

    var registry: TensorRegistry = try .initWithMetadata(allocator, resource_index.metadata);
    defer registry.deinit();

    //var registry: TensorRegistry = .init(allocator);
    //defer registry.deinit();

    //var resource: Resource = .{ .http = try .init(&client, allocator, std.Uri.parse("https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/resolve/main/model.safetensors") catch unreachable, null) };
    //defer resource.deinit();

    //var reader = resource.reader(buffer_reader);

    //try parseSafetensors(&registry, resource.uri(), reader.interface());

    var seen: std.ArrayHashMapUnmanaged(std.Uri, void, ResourceURIContext, false) = .empty;

    var it = resource_index.map.iterator();
    while (it.next()) |entry| {
        if (seen.get(entry.value_ptr.*)) |_| {} else {
            try seen.put(arena.allocator(), entry.value_ptr.*, {});

            std.debug.print("subresource uri: {f}\n", .{entry.value_ptr.fmt(.all)});

            var subresource: Resource = .{ .http = try .init(&client, allocator, entry.value_ptr.*, null) };
            defer subresource.deinit();

            std.log.debug("Opened subresource", .{});

            var reader2 = subresource.reader(buffer_reader);

            try parseSafetensors(&registry, entry.value_ptr.*, reader2.interface());
        }
    }

    var buffer_store: BufferStore5 = try .init(allocator, &registry, &client);
    defer buffer_store.deinit();

    var context = try zml.Context.init();
    defer context.deinit();

    //zml.tools.tracer.initTracer();
    //defer zml.tools.tracer.deinitTracer();

    const platform = context.autoPlatform(.{});
    context.printAvailablePlatforms(platform);

    var llama_model = try llama.LlamaLM.init(allocator, buffer_store.view(), config, options);
    defer llama_model.deinit(allocator);

    const tokens = Tensor.init(zml.Shape.init(.{ .s = 1 }, .u32));
    const token_index = Tensor.init(zml.Shape.init(.{}, .u32));
    const kv_cache = llama.KvCache.init(zml.Shape.init(.{ .layer = config.num_hidden_layers, .k = options.max_seq_len, .h = config.num_key_value_heads, .hd = 128 }, .bf16));
    const rng = Tensor.Rng.init();

    const exe = try compileModel(allocator, llama.LlamaLM.forward, llama_model, .{ tokens, token_index, kv_cache, rng }, platform);
    defer exe.deinit();

    // Compile executables used when loading buffers
    const merge_qkv_exe = try compileMergeQkvExe(allocator, buffer_store.view(), platform);
    defer merge_qkv_exe.deinit();

    const precompute_qkv_exe = try compilePrecomputeQkvExe(allocator, buffer_store.view(), platform);
    defer precompute_qkv_exe.deinit();

    const llama_buffers = try llama.LlamaLM.loadBuffers(allocator, llama_model, buffer_store.view(), platform, .{ .merge_qkv_exe = merge_qkv_exe, .precompute_qkv_exe = precompute_qkv_exe });
    defer llama_buffers.deinit(allocator);

    var tokenizer_decoder = try tokenizer.decoder();
    defer tokenizer_decoder.deinit();

    const prompt = "What is the capital of France ?";
    const tokenized_prompt = try tokenizePrompt(allocator, tokenizer, config, prompt, false);
    defer allocator.free(tokenized_prompt);

    var rng_buffer = try Tensor.Rng.initBuffer(platform, 0);
    defer rng_buffer._state.deinit();

    var kv_cache_buffer = try llama.KvCache.initBuffer(kv_cache.k.shape(), platform);
    defer kv_cache_buffer.layer_index.deinit();
    defer kv_cache_buffer.k.deinit();
    defer kv_cache_buffer.v.deinit();

    var new_token_id: u32 = undefined;
    for (tokenized_prompt[0..], 0..) |token_id, i| {
        std.debug.print("Running iteration {} token id: {}\n", .{ i, token_id });
        var token_index_buffer = try zml.Buffer.fromBytes(platform, token_index.shape(), std.mem.sliceAsBytes(&[1]u32{@intCast(i)}));
        defer token_index_buffer.deinit();

        var token_buffer = try zml.Buffer.fromBytes(platform, tokens.shape(), std.mem.sliceAsBytes(&[1]u32{token_id}));
        defer token_buffer.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);

        var results = try exe.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ llama_buffers, token_buffer, token_index_buffer, kv_cache_buffer, rng_buffer });
        std.log.debug("Calling exe", .{});
        exe.call(args, &results);

        results.fill(&.{ &token_buffer, &kv_cache_buffer, &rng_buffer._state });

        const host = try token_buffer.toHostAlloc(allocator);
        defer host.deinit(allocator);

        const predicted_token = host.items(u32)[0];
        const predicted_token_str = try tokenizer_decoder.decode(&.{predicted_token});
        std.debug.print("Predicted: {} {s}\n", .{ predicted_token, predicted_token_str });
        new_token_id = predicted_token;
    }

    var token_buffer = try zml.Buffer.fromBytes(platform, tokens.shape(), std.mem.sliceAsBytes(&[1]u32{new_token_id}));
    defer token_buffer.deinit();

    for (0..10) |i| {
        std.debug.print("Running iteration {}\n", .{i});
        var token_index_buffer = try zml.Buffer.fromBytes(platform, token_index.shape(), std.mem.sliceAsBytes(&[1]u32{@intCast(i + tokenized_prompt.len)}));
        defer token_index_buffer.deinit();

        var args = try exe.args(allocator);
        defer args.deinit(allocator);

        var results = try exe.results(allocator);
        defer results.deinit(allocator);

        args.set(.{ llama_buffers, token_buffer, token_index_buffer, kv_cache_buffer, rng_buffer });
        std.log.debug("Calling exe", .{});
        exe.call(args, &results);

        results.fill(&.{ &token_buffer, &kv_cache_buffer, &rng_buffer._state });

        const host = try token_buffer.toHostAlloc(allocator);
        defer host.deinit(allocator);

        const predicted_token = host.items(u32)[0];
        const predicted_token_str = try tokenizer_decoder.decode(&.{predicted_token});
        std.debug.print("Predicted: {} {s}\n", .{ predicted_token, predicted_token_str });
    }
}

fn compileMergeQkvExe(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, platform: Platform) !Exe {
    const proj: llama.SeparateQkv = .init(buffer_store.withPrefix("model.layers.0.self_attn"));
    const exe = try compile(allocator, llama.mergeQkv3, .{proj}, platform);
    errdefer exe.deinit();

    return exe;
}

fn compilePrecomputeQkvExe(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, platform: Platform) !Exe {
    const proj: llama.SeparateQkv = .init(buffer_store.withPrefix("model.layers.0.self_attn"));
    const embedding = Tensor.init(buffer_store.getShape("model.embed_tokens.weight").?).withTags(.{ .voc, .d });
    const layer_norm: llama.RmsNorm = .init(buffer_store.withPrefix("model.layers.0.input_layernorm"), 1e-5);

    const exe = try compile(allocator, llama.precomputeQkv0, .{ proj, layer_norm, embedding }, platform);
    errdefer exe.deinit();

    return exe;
}

pub const CompilationContext = struct {
    pub const Scope = struct {
        block: *mlir.Block,
        id_to_argument: std.AutoArrayHashMapUnmanaged(usize, usize),
        id_to_donation: std.AutoArrayHashMapUnmanaged(usize, usize),
        arena: std.heap.ArenaAllocator,

        pub fn initFromBlock(allocator: std.mem.Allocator, block: *mlir.Block) Scope {
            const arena: std.heap.ArenaAllocator = .init(allocator);
            return .{
                .block = block,
                .id_to_argument = .empty,
                .id_to_donation = .empty,
                .arena = arena,
            };
        }

        pub fn deinit(self: *Scope) void {
            self.arena.deinit();
        }
    };

    allocator: std.mem.Allocator,

    mlir_registry: *mlir.DialectRegistry,
    mlir_ctx: *mlir.Context,
    mlir_pass_manager: *mlir.PassManager,
    //mlir_op_pass_manager: *mlir.OpPassManager,
    module: *mlir.Module,

    scopes: stdx.BoundedArray(Scope, 16) = .{},

    pub fn init(allocator: std.mem.Allocator) CompilationContext {
        mlir.registerPasses("Transforms");
        const mlir_registry = mlir.DialectRegistry.init() catch unreachable;
        inline for (.{ "func", "stablehlo" }) |d| {
            mlir.DialectHandle.fromString(d).insertDialect(mlir_registry);
        }
        var mlir_ctx = mlir.Context.init(.{ .registry = mlir_registry, .threading = false }) catch unreachable;
        mlir_ctx.loadAllAvailableDialects();

        //const loc = mlir.Location.fromSrc(mlir_ctx, @src()).named(mlir_ctx, "main");
        const module = mlir.Module.init(.unknown(mlir_ctx));
        module.operation().setAttributeByName("sym_name", mlir.stringAttribute(mlir_ctx, "zml"));

        const pass_manager = mlir.PassManager.init(mlir_ctx);
        pass_manager.enableIRPrinting(.{
            .printBeforeAll = true,
        });
        {
            var opm = pass_manager.asOpPassManager();
            const passes: []const []const u8 = &.{
                "canonicalize",
                "cse",
                "canonicalize",
            };
            for (passes) |pass| {
                opm.addPipeline(pass) catch unreachable;
            }
        }

        return .{
            .allocator = allocator,
            .mlir_registry = mlir_registry,
            .mlir_ctx = mlir_ctx,
            .mlir_pass_manager = pass_manager,
            .module = module,
        };
    }

    pub fn deinit(self: *CompilationContext) void {
        self.mlir_pass_manager.deinit();
        self.module.deinit();
        self.mlir_ctx.deinit();
        self.mlir_registry.deinit();
    }

    pub fn currentScope(self: *CompilationContext) *Scope {
        return &self.scopes.slice()[self.scopes.len - 1];
    }

    pub fn pushBlock(self: *CompilationContext, block: *mlir.Block) void {
        const scope = Scope.initFromBlock(self.allocator, block);
        self.scopes.appendAssumeCapacity(scope);
    }

    pub fn popBlock(self: *CompilationContext) void {
        var maybe_popped_scope = self.scopes.pop();
        if (maybe_popped_scope) |*popped| {
            popped.deinit();
        }
    }

    pub fn current() *CompilationContext {
        return global_compilation_context.?;
    }
};

fn compileModel(allocator: std.mem.Allocator, comptime func: anytype, model: stdx.meta.Head(stdx.meta.FnArgs(func)), args: stdx.meta.Tail(stdx.meta.FnArgs(func)), platform: Platform) !Exe {
    return compile(allocator, func, .{model} ++ args, platform);
}

pub fn compile(allocator: std.mem.Allocator, comptime func: anytype, args: stdx.meta.FnArgs(func), platform: Platform) !Exe {
    var compilation_context: CompilationContext = .init(allocator);
    defer compilation_context.deinit();

    const result = emitMlir(&compilation_context, func, args) catch unreachable;
    defer result.output_info.deinit(compilation_context.allocator);
    defer compilation_context.allocator.free(result.input_shapes);

    _ = result.func.appendTo(compilation_context.module.body());

    var arena = std.heap.ArenaAllocator.init(allocator);
    defer arena.deinit();

    const loaded_executable = compileModuleToPjrtExecutable(arena.allocator(), platform, compilation_context.module, null) catch unreachable;

    const num_devices = platform.sharding().num_replicas * platform.sharding().num_partitions;
    const exe = try Exe.init(allocator, platform, loaded_executable, result.input_shapes, result.output_info.shapes, num_devices);
    errdefer exe.deinit();

    return exe;
}

fn collectShapes(allocator: std.mem.Allocator, v: anytype) ![]Shape {
    const LocalContext = struct {
        list: *std.array_list.Managed(Shape),
    };
    var list = std.array_list.Managed(Shape).init(allocator);
    errdefer list.deinit();

    var context: LocalContext = .{ .list = &list };
    try zml.meta.visit2(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.list.append(tensor.shape());
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

fn collectTensorDesc(allocator: std.mem.Allocator, buffer_store: BufferStore5.View, v: anytype) ![]TensorDesc {
    const LocalContext = struct {
        list: *std.array_list.Managed(TensorDesc),
        buffer_store: BufferStore5.View,
    };
    var list = std.array_list.Managed(TensorDesc).init(allocator);
    var context: LocalContext = .{ .list = &list, .buffer_store = buffer_store };
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            const tensor_desc = ctx_.buffer_store.store.getPtrFromId(tensor.id).?.*;
            ctx_.list.append(tensor_desc) catch unreachable;
        }
    }.cb, &context, v);

    return try list.toOwnedSlice();
}

pub const OutputInfo = struct {
    shapes: []Shape,
    values: []*const mlir.Value,
    donations: []?usize,

    pub fn deinit(self: OutputInfo, allocator: std.mem.Allocator) void {
        allocator.free(self.shapes);
        allocator.free(self.values);
        allocator.free(self.donations);
    }
};

fn collectOutputInfo(allocator: std.mem.Allocator, v: anytype) !OutputInfo {
    const LocalContext = struct {
        shape_list: *std.array_list.Managed(Shape),
        value_list: *std.array_list.Managed(*const mlir.Value),
        donation_list: *std.array_list.Managed(?usize),
    };

    var shape_list = std.array_list.Managed(Shape).init(allocator);
    errdefer shape_list.deinit();
    var value_list = std.array_list.Managed(*const mlir.Value).init(allocator);
    errdefer value_list.deinit();
    var donation_list = std.array_list.Managed(?usize).init(allocator);
    errdefer donation_list.deinit();

    var context: LocalContext = .{ .shape_list = &shape_list, .value_list = &value_list, .donation_list = &donation_list };

    try zml.meta.visit2(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) !void {
            try ctx_.shape_list.append(tensor.shape());
            try ctx_.value_list.append(tensor.value());
            try ctx_.donation_list.append(tensor.donation());
        }
    }.cb, &context, v);

    return .{
        .shapes = try shape_list.toOwnedSlice(),
        .values = try value_list.toOwnedSlice(),
        .donations = try donation_list.toOwnedSlice(),
    };
}

const EmitMlirResult = struct {
    func: *mlir.Operation,
    input_shapes: []const Shape,
    output_info: OutputInfo,
};

fn finalizeAttributeList(allocator_: std.mem.Allocator, mlir_ctx: *mlir.Context, attributes: []AttributeList) ![]*const mlir.Attribute {
    const res = try allocator_.alloc(*const mlir.Attribute, attributes.len);
    for (res, attributes) |*r, attr| {
        r.* = mlir.dictionaryAttribute(mlir_ctx, attr.constSlice());
    }
    return res;
}

fn emitMlir(compilation_context: *CompilationContext, comptime func: anytype, args: stdx.meta.FnArgs(func)) !EmitMlirResult {
    var arena = std.heap.ArenaAllocator.init(compilation_context.allocator);
    defer arena.deinit();

    const module = mlir.Module.init(.unknown(compilation_context.mlir_ctx));
    errdefer module.deinit();

    const block = mlir.Block.init(&.{}, &.{});
    errdefer block.deinit();

    compilation_context.pushBlock(block);
    defer compilation_context.popBlock();

    const LocalContext = struct {
        compilation_context: *CompilationContext,
        current_argument_id: usize = 0,
    };
    var context: LocalContext = .{
        .compilation_context = compilation_context,
    };
    zml.meta.visit(struct {
        fn cb(ctx_: *LocalContext, tensor: *const Tensor) void {
            std.log.debug("Argument {} - shape: {f}", .{ ctx_.current_argument_id, tensor.shape() });
            _ = ctx_.compilation_context.currentScope().block.addArgument(
                tensor.mlirType(ctx_.compilation_context.mlir_ctx),
                .unknown(ctx_.compilation_context.mlir_ctx),
            );
            ctx_.compilation_context.currentScope().id_to_argument.put(ctx_.compilation_context.currentScope().arena.allocator(), tensor.id, ctx_.current_argument_id) catch unreachable;
            ctx_.current_argument_id += 1;
        }
    }.cb, &context, &args);

    const input_shapes = try collectShapes(compilation_context.allocator, &args);
    errdefer compilation_context.allocator.free(input_shapes);

    const input_attributes = try arena.allocator().alloc(AttributeList, input_shapes.len);
    @memset(input_attributes, .{});

    const output_info = b: {
        global_compilation_context = compilation_context;
        defer global_compilation_context = null;

        const result = @call(.auto, func, args);

        const output_info = try collectOutputInfo(compilation_context.allocator, &result);
        errdefer output_info.deinit(compilation_context.allocator);

        break :b output_info;
    };
    errdefer output_info.deinit(compilation_context.allocator);

    for (output_info.donations, 0..) |donation, index| if (donation) |argument_index| {
        input_attributes[argument_index].appendAssumeCapacity(.named(compilation_context.mlir_ctx, "tf.aliasing_output", mlir.integerAttribute(compilation_context.mlir_ctx, .i32, index)));
    };
    _ = dialects.func.returns(compilation_context.mlir_ctx, output_info.values, .unknown(compilation_context.mlir_ctx)).appendTo(compilation_context.currentScope().block);

    const mlir_func = dialects.func.func(compilation_context.mlir_ctx, .{
        .name = "main",
        .block = compilation_context.currentScope().block,
        .location = .unknown(compilation_context.mlir_ctx),
        .args_attributes = try finalizeAttributeList(arena.allocator(), compilation_context.mlir_ctx, input_attributes),
    });

    compilation_context.mlir_pass_manager.runOnOp(mlir_func) catch |err| switch (err) {
        error.MlirUnexpected => {
            std.log.err("Failed to canonicalize invalid mlir: {f}", .{mlir_func});
            // user errors should have triggered a panic before we reach this.
            @panic("ZML generated invalid mlir. Please open a bug report");
        },
    };

    return .{
        .func = mlir_func,
        .input_shapes = input_shapes,
        .output_info = output_info,
    };
}

fn setXlaOverrideFlag(map: *c.upb_Map, flag: []const u8, value: anytype, upb_arena: *c.upb_Arena) !void {
    const result = c.upb_Map_Set(
        map,
        .{ .str_val = upb.stringView(flag) },
        .{ .msg_val = blk: {
            const field = try upb.new(c.xla_OptionOverrideProto, upb_arena);
            switch (@typeInfo(@TypeOf(value))) {
                .bool => c.xla_OptionOverrideProto_set_bool_field(field, value),
                .comptime_int, .int => c.xla_OptionOverrideProto_set_int_field(field, @intCast(value)),
                .comptime_float, .float => c.xla_OptionOverrideProto_set_double_field(field, @floatCast(value)),
                else => c.xla_OptionOverrideProto_set_string_field(field, upb.stringView(value)),
            }
            break :blk @ptrCast(field);
        } },
        upb_arena,
    );

    if (result == false) {
        return std.mem.Allocator.Error.OutOfMemory;
    }
}

fn compileModuleToPjrtExecutable(arena: std.mem.Allocator, platform: Platform, module: *const mlir.Module, xla_dump_to_: ?[]const u8) !*pjrt.LoadedExecutable {
    //const tracer = Tracer.init("ai.zml.compilation");
    //const compile_frame = tracer.frameStart("pjrt compilation");
    //defer tracer.frameEnd(compile_frame, "pjrt compilation");

    const sharding = platform.sharding();

    var upb_alloc: upb.Allocator = .init(arena);
    const upb_arena = c.upb_Arena_Init(null, 0, upb_alloc.inner());
    defer c.upb_Arena_Free(upb_arena);

    const options = blk: {
        const options = try upb.new(c.xla_CompileOptionsProto, upb_arena);
        c.xla_CompileOptionsProto_set_executable_build_options(options, executable_build_options_blk: {
            const exec_build_options = try upb.new(c.xla_ExecutableBuildOptionsProto, upb_arena);

            c.xla_ExecutableBuildOptionsProto_set_device_ordinal(exec_build_options, -1);
            c.xla_ExecutableBuildOptionsProto_set_num_replicas(exec_build_options, sharding.num_replicas);
            c.xla_ExecutableBuildOptionsProto_set_num_partitions(exec_build_options, sharding.num_partitions);
            c.xla_ExecutableBuildOptionsProto_set_use_spmd_partitioning(exec_build_options, sharding.num_partitions > 1 or sharding.num_replicas > 1);

            c.xla_ExecutableBuildOptionsProto_set_device_assignment(exec_build_options, device_assignment_blk: {
                const device_assignment = try upb.new(c.xla_DeviceAssignmentProto, upb_arena);

                c.xla_DeviceAssignmentProto_set_replica_count(device_assignment, sharding.num_replicas);
                c.xla_DeviceAssignmentProto_set_computation_count(device_assignment, sharding.num_partitions);

                const computation_devices = c.xla_DeviceAssignmentProto_resize_computation_devices(device_assignment, sharding.num_partitions, upb_arena);
                for (computation_devices[0..sharding.num_partitions], 0..) |*computation_device, i| {
                    computation_device.* = try upb.new(c.xla_DeviceAssignmentProto_ComputationDevice, upb_arena);
                    _ = c.xla_DeviceAssignmentProto_ComputationDevice_add_replica_device_ids(computation_device.*, @intCast(i), upb_arena);
                }
                break :device_assignment_blk device_assignment;
            });

            break :executable_build_options_blk exec_build_options;
        });

        const overrides_map = c._xla_CompileOptionsProto_env_option_overrides_mutable_upb_map(options, upb_arena);
        switch (platform.target) {
            .cuda => {
                // NVIDIA recommends these settings
                // https://github.com/NVIDIA/JAX-Toolbox?tab=readme-ov-file#environment-variables
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_triton_gemm", false, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_latency_hiding_scheduler", true, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_llvm_module_compilation_parallelism", true, upb_arena);
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_libnvptxcompiler", true, upb_arena);
            },
            .rocm => {
                // Disable Triton GEMM on ROCM. For some reason it's much, much slower when
                // enabled on CDNA and it's used on RDNA. Disable it altogether.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_enable_triton_gemm", false, upb_arena);
                // Use lld from libllvm instead of invoking the ld.lld binary.
                // This saves us from having to sandbox it.
                try setXlaOverrideFlag(overrides_map, "xla_gpu_use_inprocess_lld", true, upb_arena);
            },
            else => {},
        }

        if (xla_dump_to_ orelse platform.compilation_options.xla_dump_to) |xla_dump_to| {
            try setXlaOverrideFlag(overrides_map, "xla_dump_to", xla_dump_to, upb_arena);
            try setXlaOverrideFlag(overrides_map, "xla_dump_hlo_as_proto", true, upb_arena);
            if (platform.compilation_options.xla_dump_fusion_visualization) {
                try setXlaOverrideFlag(overrides_map, "xla_dump_fusion_visualization", true, upb_arena);
            }
            if (platform.compilation_options.xla_dump_hlo_pass_re) |re| {
                try setXlaOverrideFlag(overrides_map, "xla_dump_hlo_pass_re", re, upb_arena);
            }
        }

        break :blk options;
    };

    const loaded_executable = try platform.pjrt_client.compile(
        platform.pjrt_api,
        arena,
        module,
        try upb.serialize(options, upb_arena),
    );
    errdefer loaded_executable.deinit();

    return loaded_executable;
}

pub const Exe = struct {
    platform: Platform,
    exe: *pjrt.LoadedExecutable,

    context: ?*pjrt.ExecuteContext = null,

    input_shapes: []const Shape,
    output_shapes: []const Shape,

    num_devices: u8,

    arena: std.heap.ArenaAllocator,

    pub fn init(
        allocator: std.mem.Allocator,
        platform: Platform,
        exe: *pjrt.LoadedExecutable,
        input_shapes: []const Shape,
        output_shapes: []const Shape,
        num_devices: u8,
    ) !Exe {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        const input_shapes_copy = try arena.allocator().dupe(Shape, input_shapes);
        const output_shapes_copy = try arena.allocator().dupe(Shape, output_shapes);

        return .{
            .platform = platform,
            .exe = exe,
            .input_shapes = input_shapes_copy,
            .output_shapes = output_shapes_copy,
            .num_devices = num_devices,
            .arena = arena,
        };
    }

    pub fn deinit(self: *const Exe) void {
        self.arena.deinit();
    }

    pub fn args(self: *const Exe, allocator: std.mem.Allocator) !Arguments {
        return Arguments.init(allocator, self.input_shapes, self.num_devices);
    }

    pub fn results(self: *const Exe, allocator: std.mem.Allocator) !Results {
        return Results.init(allocator, self.output_shapes, self.num_devices, self.platform);
    }

    pub const FlatBuffers = struct {
        buffers: []const [*]*pjrt.Buffer,
        raw_buffers: []const *pjrt.Buffer,

        num_devices: usize,

        pub fn init(allocator: std.mem.Allocator, count: usize, num_devices: usize) !FlatBuffers {
            const raw_buffers = try allocator.alloc(*pjrt.Buffer, num_devices * count);
            errdefer allocator.free(raw_buffers);

            const buffers = try allocator.alloc([*]*pjrt.Buffer, num_devices);
            errdefer allocator.free(buffers);

            for (0..num_devices) |i| {
                buffers[i] = raw_buffers[i * count ..].ptr;
            }

            return .{
                .buffers = buffers,
                .raw_buffers = raw_buffers,
                .num_devices = num_devices,
            };
        }

        pub fn deinit(self: *const FlatBuffers, allocator: std.mem.Allocator) void {
            allocator.free(self.buffers);
            allocator.free(self.raw_buffers);
        }
    };

    pub const Arguments = struct {
        flat_buffers: FlatBuffers,
        expected_shapes: []const Shape,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, num_devices: usize) !Arguments {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        pub fn deinit(self: *const Arguments, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        pub fn set(self: *Arguments, v: anytype) void {
            return self.setPartial(v, 0);
        }

        pub fn setPartial(self: *Arguments, v: anytype, offset: usize) void {
            const LocalContext = struct {
                self: *Arguments,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = offset };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *const zml.Buffer) void {
                    stdx.debug.assert(context_.self.expected_shapes[context_.current_index].eql(buffer.shape()), "Expected argument {} to have shape {f}, got {f}", .{ context_.current_index, context_.self.expected_shapes[context_.current_index], buffer.shape() });
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        context_.self.flat_buffers.buffers[device_index][context_.current_index] = buffer._shards.get(device_index);
                    }

                    context_.current_index += 1;
                }
            }.cb, &context, &v);
        }
    };

    pub const Results = struct {
        platform: Platform,
        flat_buffers: FlatBuffers,

        expected_shapes: []const Shape,

        pub fn init(allocator: std.mem.Allocator, shapes: []const Shape, num_devices: usize, platform: Platform) !Results {
            const flat_buffers = try FlatBuffers.init(allocator, shapes.len, num_devices);
            errdefer flat_buffers.deinit(allocator);

            const expected_shapes = try allocator.dupe(Shape, shapes);
            errdefer allocator.free(expected_shapes);

            return .{
                .platform = platform,
                .flat_buffers = flat_buffers,
                .expected_shapes = expected_shapes,
            };
        }

        pub fn deinit(self: *const Results, allocator: std.mem.Allocator) void {
            allocator.free(self.expected_shapes);
            self.flat_buffers.deinit(allocator);
        }

        pub fn get(self: *Results, comptime T: type) T {
            var result: T = undefined;
            const LocalContext = struct {
                self: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = 0 };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
                    var shards: zml.Buffer.Shards = .{};
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(context_.self.flat_buffers.buffers[device_index][context_.current_index]);
                    }
                    buffer.* = zml.Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], shards.constSlice());
                    context_.current_index += 1;
                }
            }.cb, &context, &result);
            return result;
        }

        pub fn fill(self: *Results, v: anytype) void {
            const LocalContext = struct {
                self: *Results,
                current_index: usize = 0,
            };
            var context: LocalContext = .{ .self = self, .current_index = 0 };
            zml.meta.visit(struct {
                fn cb(context_: *LocalContext, buffer: *zml.Buffer) void {
                    //stdx.debug.assert(context_.self.expected_shapes[context_.current_index].eql(buffer.shape()), "Expected result {} to have shape {f}, got {f}", .{ context_.current_index, context_.self.expected_shapes[context_.current_index], buffer.shape() });
                    var shards: zml.Buffer.Shards = .{};
                    for (0..context_.self.flat_buffers.num_devices) |device_index| {
                        shards.appendAssumeCapacity(context_.self.flat_buffers.buffers[device_index][context_.current_index]);
                    }
                    buffer.* = zml.Buffer.fromPjrtBuffers(context_.self.platform, context_.self.expected_shapes[context_.current_index], shards.constSlice());
                    context_.current_index += 1;
                }
            }.cb, &context, v);
        }
    };

    pub fn call(self: *const Exe, arguments: Arguments, results_: *Results) void {
        var events = [_]?*pjrt.Event{null} ** Platform.MAX_NUM_DEVICES;
        const sharding = self.platform.sharding();

        self.exe.execute(self.platform.pjrt_api, .{
            .arguments = arguments.flat_buffers.buffers,
            .num_args = arguments.expected_shapes.len,
            .results = results_.flat_buffers.buffers,
            .events = events[0..sharding.num_partitions],
            // this allows to tell a specific buffer shouldn't be donated,
            // even if it has been marked as "can be donated" during compilation.
            // TODO: expose it ?
            .non_donatable_input_indices = &.{},
            .context = self.context,
        }) catch |err| {
            std.debug.panic("PJRT_LoadedExecutable_Execute failed with: {}", .{err});
        };

        for (events[0..sharding.num_partitions]) |e| {
            if (e) |ev| {
                ev.await_(self.platform.pjrt_api) catch unreachable;
            }
        }
    }
};

pub fn Bufferized(comptime T: type) type {
    // TODO: we should strip out the non-buffer fields.
    // Currently it's confusing cause the Bufferized struct contains field that are never read.
    // Also it will simplify the layout of the Bufferized struct.
    // accelerating the calls to execute.
    return zml.meta.MapRestrict(Tensor, zml.Buffer).map(T);
}

pub const BufferStore5 = struct {
    registry: *TensorRegistry,
    id_map: std.AutoHashMapUnmanaged(usize, *TensorDesc),
    allocator: std.mem.Allocator,
    // Temporary
    http_client: *std.http.Client,

    pub fn init(allocator: std.mem.Allocator, registry: *TensorRegistry, http_client: *std.http.Client) !BufferStore5 {
        return .{
            .registry = registry,
            .id_map = .empty,
            .allocator = allocator,
            .http_client = http_client,
        };
    }

    pub fn deinit(self: *BufferStore5) void {
        self.id_map.deinit(self.allocator);
    }

    fn bindIdToKey(self: *BufferStore5, key: []const u8, id: usize) !void {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key).?;

        const gop = try self.id_map.getOrPut(self.allocator, id);
        if (gop.found_existing) {
            stdx.debug.panic("Key {s} already has an associated tensor (id: {})", .{ key, gop.value_ptr.* });
        }
        errdefer self.id_map.removeByPtr(gop.key_ptr);

        gop.value_ptr.* = tensor_desc_ptr;
    }

    fn getPtrFromKey(self: *const BufferStore5, key: []const u8) ?*TensorDesc {
        const tensor_desc_ptr = self.registry.tensors.getPtr(key) orelse return null;
        return tensor_desc_ptr;
    }

    fn getPtrFromId(self: *const BufferStore5, id: usize) ?*TensorDesc {
        const tensor_desc_ptr = self.id_map.get(id) orelse return null;
        return tensor_desc_ptr;
    }

    pub fn view(self: *BufferStore5) View {
        return .{ .store = self };
    }

    pub const View = struct {
        store: *BufferStore5,

        prefix_buffer: [256]u8 = undefined,
        prefix_length: usize = 0,

        pub fn root(self: *const View) View {
            return .{
                .store = self.store,
            };
        }

        pub fn parent(self: *const View) View {
            const slice = self.prefix() orelse unreachable;
            const index = std.mem.lastIndexOfScalar(u8, slice[0 .. slice.len - 1], '.') orelse return self.root();
            var buffer: [256]u8 = undefined;
            @memcpy(buffer[0 .. index + 1], slice[0 .. index + 1]);
            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = index + 1,
            };
        }

        pub fn withPrefix(self: *const View, prefix_: []const u8) View {
            var buffer: [256]u8 = undefined;
            const new_prefix = std.fmt.bufPrint(&buffer, "{s}{s}.", .{ self.prefix() orelse "", prefix_ }) catch unreachable;

            return .{
                .store = self.store,
                .prefix_buffer = buffer,
                .prefix_length = new_prefix.len,
            };
        }

        fn prefix(self: *const View) ?[]const u8 {
            return if (self.prefix_length == 0) null else self.prefix_buffer[0..self.prefix_length];
        }

        pub fn maybeCreateTensor(self: View, subkey: []const u8) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;

            const tensor = Tensor.init(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensor(self: View, subkey: []const u8) Tensor {
            return self.maybeCreateTensor(subkey).?;
        }

        pub fn maybeCreateTensorWithTags(self: View, subkey: []const u8, tagz: anytype) ?Tensor {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;

            const ptr = self.store.getPtrFromKey(key) orelse return null;
            ptr.shape = ptr.shape.withTags(tagz).withSharding(.{0});

            const tensor = Tensor.init(ptr.shape);
            self.store.bindIdToKey(key, tensor.id) catch unreachable;

            return tensor;
        }

        pub fn createTensorWithTags(self: View, subkey: []const u8, tagz: anytype) Tensor {
            return self.maybeCreateTensorWithTags(subkey, tagz).?;
        }

        pub fn getShape(self: View, subkey: []const u8) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }

        pub fn getShapeOpts(self: View, subkey: []const u8, opts: struct { no_prefix: bool = false }) ?Shape {
            var buffer: [256]u8 = undefined;
            const key = if (opts.no_prefix)
                subkey
            else b: {
                break :b std.fmt.bufPrint(&buffer, "{s}{s}", .{ self.prefix() orelse "", subkey }) catch unreachable;
            };
            const entry_ptr = self.store.getPtrFromKey(key) orelse return null;
            return entry_ptr.shape;
        }
    };
};

const KB = 1024;
const MB = 1024 * KB;

const BUF_1_KB = 1 * KB;
const BUF_4_KB = 4 * KB;
const BUF_8_KB = 8 * KB;
const BUF_16_KB = 16 * KB;
const BUF_32_KB = 32 * KB;
const BUF_64_KB = 64 * KB;

const BUF_1_MB = 1 * MB;
const BUF_4_MB = 4 * MB;
const BUF_8_MB = 8 * MB;
const BUF_16_MB = 16 * MB;
const BUF_32_MB = 32 * MB;
const BUF_64_MB = 64 * MB;
const BUF_128_MB = 128 * MB;
const BUF_256_MB = 256 * MB;

pub const MemoryResource = struct {
    uri: ResourceURI,
    data: []const u8,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, data: []const u8) !MemoryResource {
        return .{
            .uri = try std.Uri.parse("memory://"),
            .data = data,
            .allocator = allocator,
        };
    }

    pub fn deinit(_: *MemoryResource) void {}
};

pub const FileResource = struct {
    uri: ResourceURI,
    uri_storage: []const u8,
    file: std.fs.File,
    allocator: std.mem.Allocator,

    pub fn init(allocator: std.mem.Allocator, path: []const u8) !FileResource {
        const uri_storage = try std.fmt.allocPrint(allocator, "file://{s}", .{path});
        errdefer allocator.free(uri_storage);

        const uri = std.Uri.parse(uri_storage);

        const file = try std.fs.openFileAbsolute(path, .{ .mode = .read_only });
        errdefer file.close();

        return .{
            .uri = uri,
            .uri_storage = uri_storage,
            .file = file,
            .allocator = allocator,
        };
    }

    // check ownership
    pub fn deinit(self: *FileResource) void {
        self.allocator.free(self.uri_storage);
        self.file.close();
    }
};

const Range = struct {
    offset: usize,
    size: usize,
};

pub const HttpResource = struct {
    uri: ResourceURI,

    request: std.http.Client.Request,
    arena: std.heap.ArenaAllocator,

    pub fn init(client: *std.http.Client, allocator: std.mem.Allocator, uri: std.Uri, maybe_range: ?Range) !HttpResource {
        var arena = std.heap.ArenaAllocator.init(allocator);
        errdefer arena.deinit();

        var request = try client.request(.GET, uri, .{});
        errdefer request.deinit();

        request.headers.authorization = .{ .override = "Bearer <HUGGINGFACE_TOKEN>" };

        const extra_headers = if (maybe_range) |range| b: {
            var list: std.ArrayList(std.http.Header) = .empty;
            const name = try arena.allocator().dupe(u8, "Range");
            const value = try std.fmt.allocPrint(arena.allocator(), "bytes={}-{}", .{ range.offset, range.offset + range.size - 1 });
            try list.append(arena.allocator(), .{ .name = name, .value = value });

            break :b try list.toOwnedSlice(arena.allocator());
        } else &.{};
        request.extra_headers = extra_headers;
        try request.sendBodiless();

        const redirect_buffer = try arena.allocator().alloc(u8, 1024 * 1024);
        const response = try request.receiveHead(redirect_buffer);
        _ = response;

        return .{
            .uri = uri,
            .request = request,
            .arena = arena,
        };
    }

    // check ownership
    pub fn deinit(self: *HttpResource) void {
        self.request.deinit();
        self.arena.deinit();
    }
};

pub const Resource = union(enum) {
    memory: MemoryResource,
    file: FileResource,
    http: HttpResource,

    pub fn deinit(self: *Resource) void {
        switch (self.*) {
            .memory => |*m| m.deinit(),
            .file => |*f| f.deinit(),
            .http => |*h| h.deinit(),
        }
    }

    pub fn reader(self: *Resource, buffer: []u8) IoReader {
        return .init(self, buffer);
    }

    pub fn uri(self: *Resource) std.Uri {
        return switch (self.*) {
            inline else => |*r| r.uri,
        };
    }
};

pub const IoReader = struct {
    reader: union(enum) {
        memory: std.io.Reader,
        file: std.fs.File.Reader,
        http: *std.io.Reader,
    },
    resource: *Resource,

    pub fn init(resource: *Resource, buffer: []u8) IoReader {
        return .{
            .reader = switch (resource.*) {
                .memory => |*m| .{ .memory = std.Io.Reader.fixed(m.data) },
                .file => |*f| .{ .file = f.file.reader(buffer) },
                .http => |*h| .{ .http = h.request.reader.bodyReader(buffer, h.request.response_transfer_encoding, h.request.response_content_length) },
            },
            .resource = resource,
        };
    }

    pub fn interface(self: *IoReader) *std.io.Reader {
        return switch (self.reader) {
            .memory => |*m| m,
            .file => |*f| &f.interface,
            .http => |h| h,
        };
    }
};

pub const Metadata = union(enum) {
    null: void,
    int: i64,
    float: f64,
    bool: bool,
    string: []const u8,

    array_bool: []const bool,
    array_int: []const i64,
    array_float: []const f64,
    array_string: []const []const u8,

    pub const ItemType = enum {
        int,
        float,
        bool,
        string,

        pub fn toZigType(comptime kind: ItemType) type {
            return switch (kind) {
                .int => i64,
                .float => f64,
                .bool => bool,
                .string => []const u8,
            };
        }
    };

    pub fn wrap(x: anytype) Metadata {
        return switch (@TypeOf(x)) {
            inline u8, i8, u16, i16, u32, i32, u64, i64 => .{ .int = @intCast(x) },
            inline f16, f32, f64 => .{ .float = @floatCast(x) },
            bool => .{ .bool = x },
            []const u8 => .{ .string = x },
            else => @panic("Unsupported type for Value: " ++ @typeName(@TypeOf(x))),
        };
    }

    pub fn copySlice(allocator: std.mem.Allocator, any_slice: anytype) !Metadata {
        return switch (@TypeOf(any_slice[0])) {
            inline u8, i8, u16, i16, u32, i32, u64, i64 => {
                const res = try allocator.alloc(i64, any_slice.len);
                for (res, any_slice) |*r, val| r.* = @intCast(val);
                return .{ .array_int = res };
            },
            inline f16, f32, f64 => {
                const res = try allocator.alloc(f64, any_slice.len);
                for (res, any_slice) |*r, val| r.* = @floatCast(val);
                return .{ .array_float = res };
            },
            bool => .{ .array_bool = try allocator.dupe(bool, any_slice) },
            []const u8 => .{ .array_string = try allocator.dupe([]const u8, @alignCast(any_slice)) },
            else => @panic("Unsupported type for Value: " ++ @typeName(@TypeOf(any_slice))),
        };
    }

    pub fn format(
        self: Metadata,
        writer: *std.Io.Writer,
    ) !void {
        switch (self) {
            .null => _ = try writer.write("null"),
            .string => |s| try writer.print("{s}", .{s}),
            .bool => |b| try writer.print("{}", .{b}),
            .int => |i| try writer.print("{d}", .{i}),
            .float => |f| try writer.print("{d}", .{f}),
            .array_bool => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
                    try writer.print("{}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_int => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
                    try writer.print("{d}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_float => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
                    try writer.print("{d}", .{v});
                }
                try writer.writeByte(']');
            },
            .array_string => |arr| {
                try writer.writeByte('[');
                for (arr, 0..) |v, i| {
                    if (i > 0) try writer.write(", ");
                    try writer.print("\"{s}\"", .{v});
                }
                try writer.writeByte(']');
            },
        }
    }

    pub fn deinit(self: Metadata, allocator: std.mem.Allocator) void {
        switch (self) {
            .string => |s| allocator.free(s),
            .array_bool => |s| allocator.free(s),
            .array_int => |s| allocator.free(s),
            .array_float => |s| allocator.free(s),
            .array_string => |s| {
                for (s) |str| allocator.free(str);
                allocator.free(s);
            },
            else => {},
        }
    }

    pub fn clone(self: Metadata, allocator: std.mem.Allocator) !Metadata {
        return switch (self) {
            .null => .null,
            .int => |v| .{ .int = v },
            .float => |v| .{ .float = v },
            .bool => |v| .{ .bool = v },
            .string => |s| .{ .string = try allocator.dupe(u8, s) },
            .array_bool => |s| .{ .array_bool = try allocator.dupe(bool, s) },
            .array_int => |s| .{ .array_int = try allocator.dupe(i64, s) },
            .array_float => |s| .{ .array_float = try allocator.dupe(f64, s) },
            .array_string => |s| blk: {
                const new_slice = try allocator.alloc([]const u8, s.len);
                errdefer allocator.free(new_slice);
                for (s, 0..) |str, i| {
                    new_slice[i] = try allocator.dupe(u8, str);
                }
                break :blk .{ .array_string = new_slice };
            },
        };
    }
};

pub const ResourceURI = std.Uri;
pub const Resources = std.ArrayHashMapUnmanaged(ResourceURI, Resource, ResourceURIContext, false);
pub const Tensors = std.StringArrayHashMapUnmanaged(TensorDesc);
pub const Metadatas = std.StringArrayHashMapUnmanaged(Metadata);

const ResourceURIContext = struct {
    // todo: implement full uri
    pub fn hash(_: ResourceURIContext, uri: ResourceURI) u32 {
        return std.array_hash_map.hashString(uri.path.percent_encoded);
    }

    pub fn eql(_: ResourceURIContext, a: ResourceURI, b: ResourceURI, _: usize) bool {
        return std.mem.eql(u8, a.path.percent_encoded, b.path.percent_encoded);
    }
};

pub const ResourceType = enum {
    index,
    safetensors,
    unknown,
};

const TensorDesc = struct {
    resource_uri: ResourceURI,
    name: []const u8,
    shape: Shape,
    offset: u64,

    pub fn byteSize(self: Tensor) u64 {
        return self.shape.byteSize();
    }
};

pub const TensorRegistry = struct {
    arena: std.heap.ArenaAllocator,

    tensors: Tensors,
    metadata: Metadatas,

    pub fn init(allocator: std.mem.Allocator) TensorRegistry {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .tensors = .{},
            .metadata = .{},
        };
    }

    pub fn initWithMetadata(
        allocator: std.mem.Allocator,
        metadata: Metadatas,
    ) !TensorRegistry {
        var arena = std.heap.ArenaAllocator.init(allocator);

        return .{
            .arena = arena,
            .tensors = .{},
            .metadata = blk: {
                var arena_allocator = arena.allocator();
                var new_metadata: Metadatas = .{};

                var it = metadata.iterator();
                while (it.next()) |entry| {
                    const key = try arena_allocator.dupe(u8, entry.key_ptr.*);
                    const value = try entry.value_ptr.*.clone(arena_allocator);
                    try new_metadata.put(arena_allocator, key, value);
                }
                break :blk new_metadata;
            },
        };
    }

    pub fn deinit(self: *TensorRegistry) void {
        const allocator = self.arena.allocator();
        self.tensors.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }

    pub fn totalBytes(self: *TensorRegistry) u64 {
        var total: u64 = 0;

        var it = self.tensors.iterator();
        while (it.next()) |entry| {
            total += entry.value_ptr.byteSize();
        }

        return total;
    }
};

const ResourceIndex = struct {
    arena: std.heap.ArenaAllocator,

    map: std.StringArrayHashMapUnmanaged(ResourceURI),
    metadata: Metadatas,

    pub fn init(allocator: std.mem.Allocator) ResourceIndex {
        return .{
            .arena = std.heap.ArenaAllocator.init(allocator),
            .map = .{},
            .metadata = .{},
        };
    }

    pub fn deinit(self: *ResourceIndex) void {
        const allocator = self.arena.allocator();
        self.map.deinit(allocator);
        self.metadata.deinit(allocator);
        self.arena.deinit();
    }
};

fn parseSafetensorsIndex(
    allocator: std.mem.Allocator,
    resource: *Resource,
    reader: *std.io.Reader,
) !ResourceIndex {
    const resource_uri = resource.uri();
    const path = resource_uri.path.percent_encoded;
    const basename = path[0 .. std.mem.lastIndexOfScalar(u8, path, '/').? + 1];

    var resource_index: ResourceIndex = .init(allocator);
    errdefer resource_index.deinit();

    const arena_allocator = resource_index.arena.allocator();

    var json_reader: std.json.Reader = .init(arena_allocator, reader);

    const index = try std.json.parseFromTokenSourceLeaky(
        std.json.Value,
        arena_allocator,
        &json_reader,
        .{ .allocate = .alloc_if_needed },
    );

    const weight_map = index.object.get("weight_map");

    if (weight_map) |wm| {
        var it = wm.object.iterator();

        while (it.next()) |entry| {
            const weight_name = entry.key_ptr.*;
            const filename = entry.value_ptr.string;

            const resource_path = try arena_allocator.alloc(u8, basename.len + filename.len);

            @memcpy(resource_path[0..basename.len], basename);
            @memcpy(resource_path[basename.len..], filename);

            const sibling_uri: ResourceURI = .{
                .scheme = resource_uri.scheme,
                .user = resource_uri.user,
                .password = resource_uri.password,
                .host = resource_uri.host,
                .port = resource_uri.port,
                .path = .{ .percent_encoded = resource_path },
            };

            _ = try resource_index.map.put(arena_allocator, try arena_allocator.dupe(u8, weight_name), sibling_uri);
        }
    } else {
        log.warn("No weight_map attribute found in index", .{});
    }

    if (index.object.get("__metadata__")) |metadata_val| {
        resource_index.metadata = try parseMetadata(arena_allocator, metadata_val);
    }

    return resource_index;
}

fn parseSafetensors(
    registry: *TensorRegistry,
    resource_uri: ResourceURI,
    reader: *std.io.Reader,
) !void {
    var arena_allocator = registry.arena.allocator();

    const json_header_length: usize = @intCast(try reader.takeInt(u64, .little));
    log.debug("json_header_length: {}", .{json_header_length});
    const json_data = try arena_allocator.alloc(u8, json_header_length);
    defer arena_allocator.free(json_data);

    try reader.readSliceAll(json_data);

    const data_start_offset = 8 + json_header_length;
    const metadata_val = try std.json.parseFromSliceLeaky(std.json.Value, arena_allocator, json_data, .{});

    var it = metadata_val.object.iterator();
    while (it.next()) |entry| {
        const key = entry.key_ptr.*;
        const value = entry.value_ptr.*;

        if (std.mem.eql(u8, key, "__metadata__")) {
            registry.metadata = try parseMetadata(arena_allocator, value);
            continue;
        }

        const shape_field = value.object.get("shape").?.array;

        if (shape_field.items.len > Shape.MAX_RANK) {
            log.warn("Can't load tensor {s}, too many dims: {}", .{ key, shape_field.items.len });
            continue;
        }

        const offset_field = value.object.get("data_offsets").?;
        const start: u64 = @intCast(offset_field.array.items[0].integer);
        const end: u64 = @intCast(offset_field.array.items[1].integer);
        const dtype = try stringToDtype(value.object.get("dtype").?.string);

        var dims: Dims = .{};
        for (shape_field.items) |d| {
            dims.appendAssumeCapacity(d.integer);
        }

        const shape: Shape = .init(dims.constSlice(), dtype);
        const size_in_bytes = end - start;
        std.debug.assert(size_in_bytes == shape.byteSize());

        const tensor_name = try arena_allocator.dupe(u8, key);
        const tensor: TensorDesc = .{
            .resource_uri = resource_uri,
            .name = tensor_name,
            .shape = shape,
            .offset = data_start_offset + start,
        };

        try registry.tensors.put(arena_allocator, tensor_name, tensor);
    }
}

fn stringToDtype(safetensor_type: []const u8) !zml.DataType {
    const map = std.StaticStringMap(zml.DataType).initComptime(.{
        .{ "F64", .f64 },
        .{ "F32", .f32 },
        .{ "F16", .f16 },
        .{ "BF16", .bf16 },
        .{ "F8_E4M3", .f8e4m3fn },
        .{ "I64", .i64 },
        .{ "I32", .i32 },
        .{ "I16", .i16 },
        .{ "I8", .i8 },
        .{ "U64", .u64 },
        .{ "U32", .u32 },
        .{ "U16", .u16 },
        .{ "U8", .u8 },
        .{ "BOOL", .bool },
    });

    return map.get(safetensor_type) orelse {
        log.err("Unsupported safetensor data type: {s}", .{safetensor_type});
        return error.UnsupportedDataType;
    };
}

fn populateMetadata(allocator: std.mem.Allocator, prefix: *StringBuilder, val: std.json.Value, metadatas: *Metadatas) !void {
    const key = prefix.items;
    return switch (val) {
        .null => try metadatas.put(allocator, try allocator.dupe(u8, key), .null),
        .bool => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .bool = v }),
        .integer => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .int = v }),
        .float => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .float = v }),
        .number_string, .string => |v| try metadatas.put(allocator, try allocator.dupe(u8, key), .{ .string = try allocator.dupe(u8, v) }),
        .array => |v| {
            if (v.items.len == 0) return;
            if (validSlice(v)) |item_type| {
                const data: Metadata = switch (item_type) {
                    .bool => blk: {
                        const values = try allocator.alloc(bool, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.bool;
                        break :blk .{ .array_bool = values };
                    },
                    .integer => blk: {
                        const values = try allocator.alloc(i64, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.integer;
                        break :blk .{ .array_int = values };
                    },
                    .float => blk: {
                        const values = try allocator.alloc(f64, v.items.len);
                        for (v.items, 0..) |item, i| values[i] = item.float;
                        break :blk .{ .array_float = values };
                    },
                    inline .string, .number_string => |tag| blk: {
                        const values = try allocator.alloc([]const u8, v.items.len);
                        for (v.items, 0..) |item, i| {
                            values[i] = try allocator.dupe(u8, @field(item, @tagName(tag)));
                        }
                        break :blk .{ .array_string = values };
                    },
                    .null, .array, .object => unreachable,
                };
                try metadatas.put(allocator, try allocator.dupe(u8, key), data);
            } else {
                for (v.items, 0..) |item, i| {
                    const old_len = prefix.items.len;
                    if (prefix.items.len > 0) {
                        prefix.appendAssumeCapacity('.');
                    }
                    prefix.items.len += std.fmt.printInt(prefix.unusedCapacitySlice(), i, 10, .lower, .{});
                    try populateMetadata(allocator, prefix, item, metadatas);
                    prefix.items.len = old_len;
                }
            }
        },
        .object => |v| {
            var obj_iter = v.iterator();
            while (obj_iter.next()) |entry| {
                const old_len = prefix.items.len;
                if (prefix.items.len > 0) {
                    prefix.appendAssumeCapacity('.');
                }
                prefix.appendSliceAssumeCapacity(entry.key_ptr.*);
                try populateMetadata(allocator, prefix, entry.value_ptr.*, metadatas);
                prefix.items.len = old_len;
            }
        },
    };
}

const StringBuilder = std.ArrayListUnmanaged(u8);

pub fn parseMetadata(allocator: std.mem.Allocator, val: std.json.Value) !Metadatas {
    var metadatas: Metadatas = .{};
    var prefix_buf: [BUF_1_KB]u8 = undefined;
    var prefix = StringBuilder.initBuffer(&prefix_buf);

    try populateMetadata(allocator, &prefix, val, &metadatas);

    return metadatas;
}

fn validSlice(v: std.json.Array) ?std.meta.Tag(std.json.Value) {
    if (v.items.len == 0) return null;

    const item_type: std.meta.Tag(std.json.Value) = v.items[0];
    switch (item_type) {
        .null, .array, .object => return null,
        else => {},
    }

    for (v.items[1..]) |item| {
        if (item != item_type)
            return null;
    }

    return item_type;
}
