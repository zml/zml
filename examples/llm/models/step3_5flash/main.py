import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import inspect
import copy
import json
from huggingface_hub import hf_hub_download
import transformers.modeling_rope_utils
from safetensors.torch import save_file

MODEL_PATH = "stepfun-ai/Step-3.5-Flash"
PATCHED_DIR = "./patched_config"

# --- 1. PATCH CONFIG ---
os.makedirs(PATCHED_DIR, exist_ok=True)
clean_config_path = hf_hub_download(repo_id=MODEL_PATH, filename="config.json")
with open(clean_config_path, "r") as f:
    cfg = json.load(f)

if "layer_types" in cfg:
    cfg["num_hidden_layers"] = len(cfg["layer_types"]) 

with open(os.path.join(PATCHED_DIR, "config.json"), "w") as f:
    json.dump(cfg, f, indent=2)


# --- 2. ROPE & INIT PATCHING (Fixes 'list' vs 'tensor' math error) ---
original_compute = transformers.modeling_rope_utils._compute_llama3_parameters

def patched_compute_llama3(config, device=None, seq_len=None, **rope_kwargs):
    temp_config = copy.copy(config)
    base_list = None
    
    # 1. Find the base list hiding in the config or kwargs
    if hasattr(temp_config, "rope_theta") and isinstance(temp_config.rope_theta, list):
        base_list = temp_config.rope_theta
    if hasattr(temp_config, "rope_scaling") and isinstance(temp_config.rope_scaling, dict):
        if "rope_theta" in temp_config.rope_scaling and isinstance(temp_config.rope_scaling["rope_theta"], list):
            base_list = temp_config.rope_scaling["rope_theta"]
    if "base" in rope_kwargs and isinstance(rope_kwargs["base"], list):
        base_list = rope_kwargs["base"]

    # 2. Extract the float for this specific layer
    selected_base = float(base_list[0]) if base_list else 10000.0
    for frame_info in inspect.stack():
        locals_dict = frame_info.frame.f_locals
        if 'layer_idx' in locals_dict and locals_dict['layer_idx'] is not None:
            idx = locals_dict['layer_idx']
            if base_list and idx < len(base_list):
                selected_base = float(base_list[idx])
            break
                
    # 3. Aggressively overwrite the list with the float everywhere
    temp_config.rope_theta = selected_base
    
    if getattr(temp_config, "rope_scaling", None) is not None:
        temp_config.rope_scaling = temp_config.rope_scaling.copy()
    else:
        temp_config.rope_scaling = {}
        
    temp_config.rope_scaling["rope_theta"] = selected_base
    temp_config.rope_scaling.setdefault("factor", 8.0)
    temp_config.rope_scaling.setdefault("low_freq_factor", 1.0)
    temp_config.rope_scaling.setdefault("high_freq_factor", 4.0)
    temp_config.rope_scaling.setdefault("original_max_position_embeddings", 8192)
        
    temp_config.head_dim = 64
    temp_config.rotary_dim = 64
    if hasattr(temp_config, "num_attention_heads"):
        temp_config.hidden_size = temp_config.num_attention_heads * 64
    
    if "base" in rope_kwargs:
        rope_kwargs["base"] = selected_base
    rope_kwargs.pop("dim", None)

    # 4. Compute
    result = original_compute(temp_config, device=device, seq_len=seq_len, **rope_kwargs)
    
    # 5. Fix shape if Step 3.5 requires specific sizing
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

original_init_weights = transformers.modeling_utils.PreTrainedModel._init_weights

def patched_init_weights(self, module):
    if module.__class__.__name__ == 'Step3p5RotaryEmbedding':
        if not hasattr(module, 'compute_default_rope_parameters'):
            module.compute_default_rope_parameters = lambda *args, **kwargs: (module.inv_freq, getattr(module, "attention_scaling", 1.0))
    return original_init_weights(self, module)

transformers.modeling_utils.PreTrainedModel._init_weights = patched_init_weights


# --- 3. LOAD MODEL & TOKENIZER ---
print("Loading Model and Tokenizer...")
config = AutoConfig.from_pretrained(PATCHED_DIR, trust_remote_code=True)
config.use_cache = False

tokenizer = AutoTokenizer.from_pretrained(PATCHED_DIR, trust_remote_code=True, fix_mistral_regex=True)
if isinstance(tokenizer.eos_token_id, list):
    config.pad_token_id = tokenizer.eos_token_id[0]
else:
    config.pad_token_id = tokenizer.eos_token_id

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    config=config, 
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)


# --- 4. FULL GRAPH HOOKING LOGIC FOR ZML ---
activations = {}

def get_activation(name):
    def hook(module, hook_inputs, hook_output):
        # Capture Inputs
        for i, inp in enumerate(hook_inputs):
            if isinstance(inp, torch.Tensor):
                # ADDED .clone() to prevent shared memory errors in safetensors
                activations[f"{name}.in.{i}"] = inp.detach().cpu().clone()
        
        # Capture Outputs
        if isinstance(hook_output, torch.Tensor):
            activations[f"{name}.out.0"] = hook_output.detach().cpu().clone()
        elif isinstance(hook_output, tuple):
            for i, out in enumerate(hook_output):
                if isinstance(out, torch.Tensor):
                    activations[f"{name}.out.{i}"] = out.detach().cpu().clone()
    return hook

print("Registering hooks for full ZML reconstruction...")
for name, module in model.named_modules():
    # Capture every primitive operation (leaf nodes) + routing gates
    if len(list(module.children())) == 0 or "router" in name.lower() or "gate" in name.lower():
        module.register_forward_hook(get_activation(name))


# --- 5. RUN INFERENCE ---
message = "The capital of France is"
inputs = tokenizer(message, return_tensors="pt").to(model.device)

print(f"Running forward pass on {model.device}...")
with torch.no_grad():
    outputs = model(**inputs)

next_token_logits = outputs.logits[:, -1, :]
next_token_id = torch.argmax(next_token_logits, dim=-1)
print(f"Verified Prediction: '{tokenizer.decode(next_token_id)}'")


# --- 6. SAVE DATA ---
print(f"Captured {len(activations)} tensors. Saving to safetensors...")
save_file(activations, "step3_5flash_full_zml.safetensors")
print("Done! File saved as step3_5flash_full_zml.safetensors")
