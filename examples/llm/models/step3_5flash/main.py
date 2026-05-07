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
import functools
import torch.nn as nn

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

# Hooking graph

activations = {}

def _save(key, value):
    """Store a tensor under `key`, suffixing if the key already exists so that
    repeated calls (e.g. MoE expert loop) do not clobber prior values."""
    if not isinstance(value, torch.Tensor):
        return
    base = key
    i = 0
    while key in activations:
        i += 1
        key = f"{base}.call{i}"
    activations[key] = value.detach().cpu().clone().contiguous()

def make_hook(name):
    call_counts = {"n": 0}
    def hook(module, hook_inputs, hook_output):
        n = call_counts["n"]
        call_counts["n"] += 1
        prefix = name if n == 0 else f"{name}.call{n}"

        for i, inp in enumerate(hook_inputs):
            if isinstance(inp, torch.Tensor):
                _save(f"{prefix}.in.{i}", inp)

        outs = hook_output if isinstance(hook_output, tuple) else (hook_output,)
        for i, out in enumerate(outs):
            if isinstance(out, torch.Tensor):
                _save(f"{prefix}.out.{i}", out)
    return hook

print("Registering forward hooks on every submodule...")
for name, module in model.named_modules():
    if name == "":
        continue  # skip root, it's redundant with lm_head/model
    module.register_forward_hook(make_hook(name))


# --- 4b. INSTRUMENT FUNCTIONAL OPS (no nn.Module → no hook possible) ---
# Resolve the Step3p5* classes from the dynamically-loaded modeling module.
modeling_mod = type(model).__module__
modeling = __import__(modeling_mod, fromlist=["*"])

def _patch_method(cls, method_name, wrapper_factory):
    original = getattr(cls, method_name)
    setattr(cls, method_name, wrapper_factory(original))

# --- rotate_half / apply_rotary_pos_emb (module-level functions) ---
_orig_rotate_half = modeling.rotate_half
_orig_apply_rope = modeling.apply_rotary_pos_emb

# We need a "current layer index" so we can name these activations.
_ctx = {"layer_idx": None}

@functools.wraps(_orig_rotate_half)
def rotate_half_wrapped(x):
    out = _orig_rotate_half(x)
    li = _ctx["layer_idx"]
    if li is not None:
        _save(f"model.layers.{li}.self_attn.rotate_half.out", out)
    return out

@functools.wraps(_orig_apply_rope)
def apply_rope_wrapped(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    q_embed, k_embed = _orig_apply_rope(q, k, cos, sin, position_ids, unsqueeze_dim)
    li = _ctx["layer_idx"]
    tag = f"model.layers.{li}.self_attn" if li is not None else "rope"
    _save(f"{tag}.rope.q_in", q)
    _save(f"{tag}.rope.k_in", k)
    _save(f"{tag}.rope.cos", cos)
    _save(f"{tag}.rope.sin", sin)
    _save(f"{tag}.rope.q_embed", q_embed)
    _save(f"{tag}.rope.k_embed", k_embed)
    return q_embed, k_embed

modeling.rotate_half = rotate_half_wrapped
modeling.apply_rotary_pos_emb = apply_rope_wrapped

# --- repeat_kv ---
_orig_repeat_kv = modeling.repeat_kv
@functools.wraps(_orig_repeat_kv)
def repeat_kv_wrapped(hidden_states, n_rep):
    out = _orig_repeat_kv(hidden_states, n_rep)
    li = _ctx["layer_idx"]
    if li is not None:
        _save(f"model.layers.{li}.self_attn.repeat_kv.in", hidden_states)
        _save(f"model.layers.{li}.self_attn.repeat_kv.out", out)
    return out
modeling.repeat_kv = repeat_kv_wrapped

# --- eager_attention_forward (q@k^T, mask add, softmax, attn@v) ---
_orig_eager = modeling.eager_attention_forward

def eager_attention_forward_wrapped(module, query, key, value, attention_mask,
                                    scaling, dropout=0.0, **kwargs):
    li = getattr(module, "layer_idx", None)
    tag = f"model.layers.{li}.self_attn.attn_fn" if li is not None else "attn_fn"

    key_states = modeling.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling.repeat_kv(value, module.num_key_value_groups)
    _save(f"{tag}.key_repeated", key_states)
    _save(f"{tag}.value_repeated", value_states)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    _save(f"{tag}.qk_scaled", attn_weights)

    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, :key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
        _save(f"{tag}.qk_masked", attn_weights)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1)
    _save(f"{tag}.attn_probs", attn_weights)

    attn_weights = nn.functional.dropout(attn_weights, p=dropout,
                                         training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    _save(f"{tag}.attn_out_premerge", attn_output)

    attn_output = attn_output.transpose(1, 2).contiguous()
    _save(f"{tag}.attn_out", attn_output)
    return attn_output, attn_weights

modeling.eager_attention_forward = eager_attention_forward_wrapped
# ALL_ATTENTION_FUNCTIONS dispatch may also hold the eager fn — re-bind both.
try:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    if "eager" in ALL_ATTENTION_FUNCTIONS:
        ALL_ATTENTION_FUNCTIONS["eager"] = eager_attention_forward_wrapped
except Exception:
    pass

# --- Track "current layer index" via Attention.forward wrapping ---
def _attn_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        prev = _ctx["layer_idx"]
        _ctx["layer_idx"] = self.layer_idx
        try:
            return orig(self, *args, **kwargs)
        finally:
            _ctx["layer_idx"] = prev
    return wrapped
_patch_method(modeling.Step3p5Attention, "forward", _attn_forward_factory)

# --- Step3p5MLP.forward (capture gate, up, gate*up before down_proj) ---
def _mlp_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, x):
        # Walk through manually so we can record the intermediate.
        up = self.up_proj(x)
        gate_pre = self.gate_proj(x)
        gate = self.act_fn(gate_pre)
        if self.limit is not None:
            gate = gate.clamp(min=None, max=self.limit)
            up = up.clamp(min=-self.limit, max=self.limit)
        prod = gate * up
        out = self.down_proj(prod)
        # Try to tag with owning layer if attached via parent traversal.
        li = _ctx["layer_idx"]
        tag = f"model.layers.{li}.mlp" if li is not None else "mlp"
        _save(f"{tag}.gate_pre", gate_pre)
        _save(f"{tag}.gate_act", gate)
        _save(f"{tag}.up", up)
        _save(f"{tag}.gate_times_up", prod)
        _save(f"{tag}.down_out", out)
        return out
    return wrapped
_patch_method(modeling.Step3p5MLP, "forward", _mlp_forward_factory)

# --- Step3p5MoEMLP.forward (routing + per-expert + weighted sum) ---
def _moe_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, hidden_states):
        li = _ctx["layer_idx"]
        tag = f"model.layers.{li}.mlp" if li is not None else "moe"

        bsz, seqlen, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _save(f"{tag}.flat_in", flat)

        gate_in = flat.float() if self.need_fp32_gate else flat
        gating_output = self.gate(gate_in)
        _save(f"{tag}.gating_output", gating_output)

        if self.custom_routing_function:
            routing_weights, selected_experts = self.custom_routing_function(
                gating_output, self.top_k, renormalize=True)
        else:
            # Fallback: replicate softmax routing.
            routing_weights, selected_experts = modeling.softmax_routing_function(
                gating_output, self.top_k, renormalize=True)
        _save(f"{tag}.routing_weights", routing_weights)
        _save(f"{tag}.selected_experts", selected_experts.to(torch.int32))

        routing_weights = routing_weights * self.routed_scaling_factor
        _save(f"{tag}.routing_weights_scaled", routing_weights)

        final_hidden_states = torch.zeros(
            (bsz * seqlen, hidden_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device)

        expert_mask = torch.nn.functional.one_hot(
            selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        _save(f"{tag}.expert_mask", expert_mask)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])
            if top_x.numel() == 0:
                continue
            current_state = flat[top_x]
            expert_out = self.get_expert_output(current_state, expert_idx)
            weighted = expert_out * routing_weights[top_x, idx, None]
            _save(f"{tag}.expert{expert_idx}.tokens", top_x)
            _save(f"{tag}.expert{expert_idx}.input", current_state)
            _save(f"{tag}.expert{expert_idx}.output", expert_out)
            _save(f"{tag}.expert{expert_idx}.weighted", weighted)
            final_hidden_states.index_add_(0, top_x, weighted.to(hidden_states.dtype))

        out = final_hidden_states.reshape(bsz, seqlen, hidden_dim)
        _save(f"{tag}.out", out)
        return out
    return wrapped
_patch_method(modeling.Step3p5MoEMLP, "forward", _moe_forward_factory)

# --- RMSNorm internals (variance, rsqrt-normalized, weight-scaled) ---
def _rms_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, x):
        dtype = x.dtype
        xf = x.float()
        variance = xf.pow(2).mean(dim=-1, keepdim=True)
        normed = xf * torch.rsqrt(variance + self.variance_epsilon)
        scaled = normed * (self.weight.float() + 1)
        out = scaled.to(dtype)
        # Stash under a generic key; module hook will also fire with a proper name.
        _save("_rmsnorm_last.variance", variance)
        _save("_rmsnorm_last.normed", normed)
        _save("_rmsnorm_last.scaled", scaled)
        return out
    return wrapped
_patch_method(modeling.Step3p5RMSNorm, "forward", _rms_forward_factory)

# --- DecoderLayer residuals ---
def _decoder_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, hidden_states, **kwargs):
        li = self.layer_idx
        tag = f"model.layers.{li}"
        _save(f"{tag}.residual_pre_attn", hidden_states)
        normed = self.input_layernorm(hidden_states)
        _save(f"{tag}.input_layernorm.out", normed)
        attn_out, _ = self.self_attn(hidden_states=normed, **kwargs)
        _save(f"{tag}.self_attn.block_out", attn_out)
        hidden_states = hidden_states + attn_out
        _save(f"{tag}.residual_post_attn", hidden_states)

        residual2 = hidden_states
        normed2 = self.post_attention_layernorm(hidden_states)
        _save(f"{tag}.post_attention_layernorm.out", normed2)
        if self.use_moe:
            ffn_output = self.moe_mlp(normed2)
            if hasattr(self, "shared_mlp") and self.shared_mlp is not None:
                shared = self.shared_mlp(normed2)
                _save(f"{tag}.shared_mlp.out", shared)
                ffn_output = ffn_output + shared
        else:
            ffn_output = self.mlp(normed2)
        if isinstance(ffn_output, tuple):
            ffn_output = ffn_output[0]
        _save(f"{tag}.ffn_out", ffn_output)
        hidden_states = residual2 + ffn_output
        _save(f"{tag}.residual_post_ffn", hidden_states)
        return hidden_states
    return wrapped
_patch_method(modeling.Step3p5DecoderLayer, "forward", _decoder_forward_factory)


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
