
import torch
from transformers import AutoConfig, AutoModel
import sys

model_path = "/Users/kevin/FLUX.2-klein-4B/text_encoder"
try:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    print(f"Config: {config}")
    print(f"hidden_size: {config.hidden_size}")
    print(f"num_attention_heads: {config.num_attention_heads}")
    if hasattr(config, "head_dim"):
        print(f"head_dim: {config.head_dim}")
except Exception as e:
    print(f"Error loading config: {e}")

try:
    from safetensors import safe_open
    import os
    st_path = os.path.join(model_path, "model.safetensors")
    if not os.path.exists(st_path):
        # try index
        import json
        idx_path = os.path.join(model_path, "model.safetensors.index.json")
        if os.path.exists(idx_path):
            with open(idx_path) as f:
                idx = json.load(f)
            # just pick one file
            first_file = list(idx["weight_map"].values())[0]
            st_path = os.path.join(model_path, first_file)
            
    with safe_open(st_path, framework="pt", device="cpu") as f:
        keys = f.keys()
        if "model.embed_tokens.weight" in keys:
            emb = f.get_tensor("model.embed_tokens.weight")
            print(f"Embedding table shape: {emb.shape}")
            # Token 151644
            if 151644 < emb.shape[0]:
                print(f"Embedding[151644] first 5: {emb[151644][:5].tolist()}")
            else:
                print("Token 151644 out of bounds for non-sharded check (might be in another shard)")
        
        # Check first layer weights
        if "model.layers.0.self_attn.q_proj.weight" in keys:
             print(f"L0 Q_proj shape: {f.get_tensor('model.layers.0.self_attn.q_proj.weight').shape}")
except Exception as e:
    print(f"Error inspecting weights: {e}")
