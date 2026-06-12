import os
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
import json
from huggingface_hub import hf_hub_download
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


# --- 2. ROPE & INIT PATCHING ---
# Removed: the upstream Step3p5RotaryEmbedding.__init__ already handles
# `rope_theta` being a list (it picks per-layer via self.layer_idx). The
# previous patch forced head_dim=64 and bogus llama3 scaling factors, which
# produced near-zero inv_freq and made cos/sin in the dump effectively the
# identity, corrupting every downstream activation. Run the real rope path.


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
COLLECT_ALL_MODULE_HOOKS = False
COLLECT_ATTENTION = False
COLLECT_RMSNORM = False
COLLECT_MOE = True
OUTPUT_PATH = "moe_activations.safetensors"

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

if COLLECT_ALL_MODULE_HOOKS:
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

# We need a "current layer index" so we can name these activations.
_ctx = {"layer_idx": None, "mlp_tag": None}

def _layer_tag(suffix):
    li = _ctx["layer_idx"]
    return f"model.layers.{li}.{suffix}" if li is not None else suffix

def _current_mlp_tag(default_suffix):
    return _ctx["mlp_tag"] or _layer_tag(default_suffix)

# --- rotate_half / apply_rotary_pos_emb (module-level functions) ---
if COLLECT_ATTENTION:
    _orig_rotate_half = modeling.rotate_half
    _orig_apply_rope = modeling.apply_rotary_pos_emb

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
if COLLECT_ATTENTION:
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
if COLLECT_ATTENTION:
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
if COLLECT_ATTENTION:
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
        tag = _current_mlp_tag("mlp")
        if COLLECT_MOE and not COLLECT_ALL_MODULE_HOOKS and ".share_expert" not in tag:
            return out
        _save(f"{tag}.gate_pre", gate_pre)
        _save(f"{tag}.gate_act", gate)
        _save(f"{tag}.up", up)
        _save(f"{tag}.gate_times_up", prod)
        _save(f"{tag}.down_out", out)
        return out
    return wrapped
if COLLECT_ALL_MODULE_HOOKS or COLLECT_MOE:
    _patch_method(modeling.Step3p5MLP, "forward", _mlp_forward_factory)

def _set_context(layer_idx=None, mlp_tag=None):
    prev_layer_idx = _ctx["layer_idx"]
    prev_mlp_tag = _ctx["mlp_tag"]
    _ctx["layer_idx"] = layer_idx
    _ctx["mlp_tag"] = mlp_tag
    return prev_layer_idx, prev_mlp_tag

def _restore_context(prev):
    _ctx["layer_idx"], _ctx["mlp_tag"] = prev

def _route_tag():
    return f"{_current_mlp_tag('moe')}.router"

def _save_normalized_topk(tag, topk_prob, renormalize, eps=None):
    _save(f"{tag}.topk_prob", topk_prob)
    expert_topk_weight = topk_prob
    if renormalize:
        topk_sum = torch.sum(expert_topk_weight, dim=-1, keepdim=True)
        denom = topk_sum
        if eps is not None:
            denom = denom + eps
        _save(f"{tag}.topk_sum", topk_sum)
        _save(f"{tag}.normalize_denom", denom)
        expert_topk_weight = expert_topk_weight / denom
    _save(f"{tag}.expert_topk_weight", expert_topk_weight)
    return expert_topk_weight

if COLLECT_MOE:
    _orig_sigmoid_routing_function = modeling.sigmoid_routing_function
    _orig_softmax_routing_function = modeling.softmax_routing_function

    @functools.wraps(_orig_sigmoid_routing_function)
    def sigmoid_routing_function_wrapped(gating_output, topk, renormalize):
        tag = _route_tag()
        gating_output = gating_output.float()
        gate_prob = torch.sigmoid(gating_output)
        gate_prob_normalized = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
        topk_prob, indices = torch.topk(gate_prob_normalized, k=topk, dim=1)
        expert_topk_weight = _save_normalized_topk(tag, topk_prob, renormalize)
        _save(f"{tag}.gating_output_float", gating_output)
        _save(f"{tag}.gate_prob", gate_prob)
        _save(f"{tag}.gate_prob_normalized", gate_prob_normalized)
        _save(f"{tag}.topk_indices", indices.to(torch.int32))
        return expert_topk_weight, indices

    @functools.wraps(_orig_softmax_routing_function)
    def softmax_routing_function_wrapped(gating_output, top_k, renormalize):
        tag = _route_tag()
        gating_output = gating_output.float()
        gate_prob = torch.softmax(gating_output, dim=-1)
        gate_prob_normalized = gate_prob / gate_prob.sum(dim=-1, keepdim=True)
        topk_prob, indices = torch.topk(gate_prob_normalized, k=top_k, dim=1)
        expert_topk_weight = _save_normalized_topk(tag, topk_prob, renormalize)
        _save(f"{tag}.gating_output_float", gating_output)
        _save(f"{tag}.gate_prob", gate_prob)
        _save(f"{tag}.gate_prob_normalized", gate_prob_normalized)
        _save(f"{tag}.topk_indices", indices.to(torch.int32))
        return expert_topk_weight, indices.to(torch.int32)

    modeling.sigmoid_routing_function = sigmoid_routing_function_wrapped
    modeling.softmax_routing_function = softmax_routing_function_wrapped

    def _router_bias_func_factory(orig):
        @functools.wraps(orig)
        def wrapped(self, gating_output, topk, renormalize):
            tag = _route_tag()
            gating_output_float = gating_output.float()
            gate_prob = torch.sigmoid(gating_output_float)
            router_bias = self.router_bias.unsqueeze(0)
            gate_prob_with_bias = gate_prob + router_bias
            topk_biased_scores, indices = torch.topk(
                gate_prob_with_bias, k=topk, dim=1)
            topk_prob = torch.gather(gate_prob, 1, indices)
            expert_topk_weight = _save_normalized_topk(
                tag, topk_prob, renormalize, eps=1e-20)
            _save(f"{tag}.gating_output_float", gating_output_float)
            _save(f"{tag}.gate_prob", gate_prob)
            _save(f"{tag}.router_bias", router_bias)
            _save(f"{tag}.gate_prob_with_bias", gate_prob_with_bias)
            _save(f"{tag}.topk_biased_scores", topk_biased_scores)
            _save(f"{tag}.topk_indices", indices.to(torch.int32))
            return expert_topk_weight, indices
        return wrapped
    _patch_method(modeling.Step3p5MoEMLP, "router_bias_func", _router_bias_func_factory)

    def _get_expert_output_factory(orig):
        @functools.wraps(orig)
        def wrapped(self, inputs, expert_id):
            tag = f"{_current_mlp_tag('moe')}.expert{expert_id}"
            up = self.up_proj(inputs, expert_id)
            gate_pre = self.gate_proj(inputs, expert_id)
            gate = self.act_fn(gate_pre)
            _save(f"{tag}.input", inputs)
            _save(f"{tag}.up", up)
            _save(f"{tag}.gate_pre", gate_pre)
            _save(f"{tag}.gate_act", gate)
            if self.limit is not None:
                gate = gate.clamp(min=None, max=self.limit)
                up = up.clamp(min=-self.limit, max=self.limit)
                _save(f"{tag}.gate_clamped", gate)
                _save(f"{tag}.up_clamped", up)
            prod = gate * up
            out = self.down_proj(prod, expert_id)
            _save(f"{tag}.gate_times_up", prod)
            _save(f"{tag}.down_out", out)
            return out
        return wrapped
    _patch_method(modeling.Step3p5MoEMLP, "get_expert_output", _get_expert_output_factory)

# --- Step3p5MoEMLP.forward (routing + per-expert + weighted sum) ---
def _moe_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, hidden_states):
        tag = _current_mlp_tag("moe")

        bsz, seqlen, hidden_dim = hidden_states.shape
        flat = hidden_states.view(-1, hidden_dim)
        _save(f"{tag}.flat_in", flat)

        if self.need_fp32_gate:
            gate_in = flat.to(torch.float32)
            gating_output = torch.matmul(
                gate_in, self.gate.weight.t().to(torch.float32))
        else:
            gate_in = flat
            gating_output = self.gate(gate_in)
        _save(f"{tag}.gate_in", gate_in)
        _save(f"{tag}.router_logits", gating_output)

        if getattr(self, "use_moe_router_bias", False):
            routing_weights, selected_experts = self.router_bias_func(
                gating_output, self.top_k, renormalize=True)
        elif self.custom_routing_function:
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
            expert_routing_weights = routing_weights[top_x, idx, None]
            weighted = expert_out * expert_routing_weights
            _save(f"{tag}.expert{expert_idx}.tokens", top_x)
            _save(f"{tag}.expert{expert_idx}.topk_slots", idx)
            _save(f"{tag}.expert{expert_idx}.routing_weights", expert_routing_weights)
            _save(f"{tag}.expert{expert_idx}.input", current_state)
            _save(f"{tag}.expert{expert_idx}.output", expert_out)
            _save(f"{tag}.expert{expert_idx}.weighted", weighted)
            final_hidden_states.index_add_(0, top_x, weighted.to(hidden_states.dtype))

        out = final_hidden_states.reshape(bsz, seqlen, hidden_dim)
        _save(f"{tag}.out", out)
        return out
    return wrapped
if COLLECT_MOE:
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
if COLLECT_RMSNORM:
    _patch_method(modeling.Step3p5RMSNorm, "forward", _rms_forward_factory)

# --- DecoderLayer residuals ---
def _decoder_forward_factory(orig):
    @functools.wraps(orig)
    def wrapped(self, hidden_states, **kwargs):
        li = self.layer_idx
        tag = f"model.layers.{li}"
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.residual_pre_attn", hidden_states)
        normed = self.input_layernorm(hidden_states)
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.input_layernorm.out", normed)
        prev = _set_context(li, None)
        try:
            attn_out, _ = self.self_attn(hidden_states=normed, **kwargs)
        finally:
            _restore_context(prev)
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.self_attn.block_out", attn_out)
        hidden_states = hidden_states + attn_out
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.residual_post_attn", hidden_states)

        residual2 = hidden_states
        normed2 = self.post_attention_layernorm(hidden_states)
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.post_attention_layernorm.out", normed2)
        if self.use_moe:
            prev = _set_context(li, f"{tag}.share_expert")
            try:
                share_output = self.share_expert(normed2)
            finally:
                _restore_context(prev)
            prev = _set_context(li, f"{tag}.moe")
            try:
                moe_output = self.moe(normed2)
            finally:
                _restore_context(prev)
            if COLLECT_MOE:
                _save(f"{tag}.share_expert.out", share_output)
                _save(f"{tag}.moe.out_with_share", moe_output + share_output)
            ffn_output = moe_output + share_output
        else:
            prev = _set_context(li, f"{tag}.mlp")
            try:
                ffn_output = self.mlp(normed2)
            finally:
                _restore_context(prev)
        if isinstance(ffn_output, tuple):
            ffn_output = ffn_output[0]
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.ffn_out", ffn_output)
        hidden_states = residual2 + ffn_output
        if COLLECT_ALL_MODULE_HOOKS:
            _save(f"{tag}.residual_post_ffn", hidden_states)
        return hidden_states
    return wrapped
if COLLECT_ALL_MODULE_HOOKS or COLLECT_ATTENTION or COLLECT_MOE:
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
save_file(activations, OUTPUT_PATH)
print(f"Done! File saved as {OUTPUT_PATH}")
