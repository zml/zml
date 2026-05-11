import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import zml_utils
import os
import inspect
import copy
import json
from huggingface_hub import hf_hub_download

MODEL_PATH = "stepfun-ai/Step-3.5-Flash"
PATCHED_DIR = "./patched_config"

# --- AUTO-RESTORE AND FIX THE CONFIG FILE ---
# Grabs the pristine config.json from cache to undo our previous 45-layer truncation,
# fixes the actual typo (num_hidden_layers: 48), and saves it locally.
os.makedirs(PATCHED_DIR, exist_ok=True)
clean_config_path = hf_hub_download(repo_id=MODEL_PATH, filename="config.json")
with open(clean_config_path, "r") as f:
    cfg = json.load(f)

if "layer_types" in cfg:
    cfg["num_hidden_layers"] = len(cfg["layer_types"]) # Forces exactly 48

with open(os.path.join(PATCHED_DIR, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)

# --- THE "CLEAN ROOM" CONFIG PATCH ---
import transformers.modeling_rope_utils
original_compute = transformers.modeling_rope_utils._compute_llama3_parameters

def patched_compute_llama3(config, device=None, seq_len=None, **rope_kwargs):
    temp_config = copy.copy(config)
    base_list = None
    
    if hasattr(temp_config, "rope_theta") and isinstance(temp_config.rope_theta, list):
        base_list = temp_config.rope_theta
    if hasattr(temp_config, "rope_scaling") and isinstance(temp_config.rope_scaling, dict):
        if "rope_theta" in temp_config.rope_scaling and isinstance(temp_config.rope_scaling["rope_theta"], list):
            base_list = temp_config.rope_scaling["rope_theta"]
    if "base" in rope_kwargs and isinstance(rope_kwargs["base"], list):
        base_list = rope_kwargs["base"]

    if base_list is not None:
        selected_base = float(base_list[0])
        for frame_info in inspect.stack():
            locals_dict = frame_info.frame.f_locals
            if 'layer_idx' in locals_dict and locals_dict['layer_idx'] is not None:
                idx = locals_dict['layer_idx']
                if idx < len(base_list):
                    selected_base = float(base_list[idx])
                break
                
        temp_config.rope_theta = selected_base
        if "base" in rope_kwargs:
            rope_kwargs["base"] = selected_base

    if getattr(temp_config, "rope_scaling", None) is None:
        temp_config.rope_scaling = {}
    else:
        temp_config.rope_scaling = temp_config.rope_scaling.copy()
        
    if base_list is not None:
        temp_config.rope_scaling["rope_theta"] = selected_base

    temp_config.rope_scaling.setdefault("factor", 8.0)
    temp_config.rope_scaling.setdefault("low_freq_factor", 1.0)
    temp_config.rope_scaling.setdefault("high_freq_factor", 4.0)
    temp_config.rope_scaling.setdefault("original_max_position_embeddings", 8192)

    temp_config.head_dim = 64
    temp_config.rotary_dim = 64
    if hasattr(temp_config, "num_attention_heads"):
        temp_config.hidden_size = temp_config.num_attention_heads * 64

    rope_kwargs.pop("dim", None)

    result = original_compute(temp_config, device=device, seq_len=seq_len, **rope_kwargs)
    
    for frame_info in inspect.stack():
        if frame_info.function == '_init_weights':
            module = frame_info.frame.f_locals.get('module')
            if module is not None and hasattr(module, 'inv_freq'):
                target_size = module.inv_freq.shape[0]
                if isinstance(result, tuple):
                    return (result[0][:target_size], result[1])
                else:
                    return result[:target_size]
            break
            
    return result

transformers.modeling_rope_utils._compute_llama3_parameters = patched_compute_llama3

if hasattr(transformers.modeling_rope_utils, "ROPE_INIT_FUNCTIONS"):
    transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["llama3"] = patched_compute_llama3
    transformers.modeling_rope_utils.ROPE_INIT_FUNCTIONS["default"] = patched_compute_llama3

# ---> NEW FIX: Inject the missing API method during Hugging Face's initialization sweep <---
original_init_weights = transformers.modeling_utils.PreTrainedModel._init_weights

def patched_init_weights(self, module):
    # If HF encounters StepFun's custom RoPE, we dynamically attach the missing method
    if module.__class__.__name__ == 'Step3p5RotaryEmbedding':
        if not hasattr(module, 'compute_default_rope_parameters'):
            module.compute_default_rope_parameters = lambda *args, **kwargs: (module.inv_freq, getattr(module, "attention_scaling", 1.0))
    return original_init_weights(self, module)

transformers.modeling_utils.PreTrainedModel._init_weights = patched_init_weights
# ------------------------------------------------

# 1. Load the patched config
config = AutoConfig.from_pretrained(PATCHED_DIR, trust_remote_code=True)

# 2. Point the tokenizer to the PATCHED directory
tokenizer = AutoTokenizer.from_pretrained(PATCHED_DIR, trust_remote_code=True, device_map="auto")

if isinstance(tokenizer.eos_token_id, list):
    config.pad_token_id = tokenizer.eos_token_id[0]
else:
    config.pad_token_id = tokenizer.eos_token_id

# 3. Stream the heavy weights from the Hub
model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

messages = [{"role": "user", "content": "Explain the significance of the number 42."}]

# replace model with zml_utils tracked model
model = zml_utils.ActivationCollector(model, max_layers=1000, stop_after_first_step=True)
inputs = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)

# perform forward pass to collect activations
outputs, activations = model(inputs)

output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

print(output_text)
print(activations.keys())
