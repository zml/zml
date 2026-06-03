"""Generate one safetensors activation fixture using canonical model.layers.<n>... keys.

Captures, for every applicable layer:
  - layer in/out (model.layers.<n>.in/out)
  - input_layernorm / post_attention_layernorm in/out
  - self_attn in/out + k_in, v_in, rope.q_in / rope.k_in / rope.q_embed / rope.k_embed
  - mlp in/out (non-MoE layers) or moe in/out (MoE layers)
  - moe.router in/out (full ranking), moe.dispatch.{selected_experts, routing_weights,
    expert_mask, assignment_counts}, moe.expert.<e>.{token_indices, slot_indices,
    weighted_outputs, unscaled_sums}
  - model in/out, model.norm in/out, lm_head in/out

The output file is `activations.safetensors` next to this script. It uses canonical
`model.layers.<n>...` keys throughout — no old-schema `moe.expertN.callK` style keys.

Usage:
    python3 activations.py [--model PATH] [--prompt TEXT] [--out FILE] [--max-len N]
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent

CAPTURES: dict[str, torch.Tensor] = {}


def _to_cpu(t: torch.Tensor) -> torch.Tensor:
    t = t.detach()
    if t.is_floating_point():
        return t.to("cpu").contiguous()
    return t.to("cpu", dtype=torch.int32).contiguous()


def _store(key: str, value: torch.Tensor) -> None:
    if not isinstance(value, torch.Tensor):
        return
    if key in CAPTURES:
        return
    CAPTURES[key] = _to_cpu(value)


def _flatten_tensors(x):
    if x is None:
        return []
    if isinstance(x, torch.Tensor):
        return [x]
    if isinstance(x, (list, tuple)):
        out = []
        for v in x:
            out.extend(_flatten_tensors(v))
        return out
    if isinstance(x, dict):
        out = []
        for v in x.values():
            out.extend(_flatten_tensors(v))
        return out
    return []


def _module_hook(module, args, kwargs, output):
    name = getattr(module, "_capture_prefix", None)
    if name is None or name == "":
        return
    in_tensors = _flatten_tensors(list(args)) + _flatten_tensors(list(kwargs.values()))
    out_tensors = _flatten_tensors(output)
    for i, t in enumerate(in_tensors):
        _store(f"{name}.in.{i}", t)
    for i, t in enumerate(out_tensors):
        _store(f"{name}.out.{i}", t)


def make_patched_moe_forward():
    def patched_moe_forward(self, hidden_states: torch.Tensor):
        prefix = getattr(self, "_capture_prefix", None)
        bsz, seqlen, hdim = hidden_states.shape
        hs_2d = hidden_states.view(-1, hdim)

        if self.need_fp32_gate:
            router_logits = torch.matmul(
                hs_2d.to(torch.float32), self.gate.weight.t().to(torch.float32)
            )
        else:
            router_logits = self.gate(hs_2d)

        if self.use_moe_router_bias:
            gate_prob = torch.sigmoid(router_logits.float())
            gate_score = gate_prob + self.router_bias.unsqueeze(0)
            full_ranking = torch.argsort(gate_score, dim=-1, descending=True)
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
        elif self.custom_routing_function is not None:
            routing_weights, selected_experts = self.custom_routing_function(
                router_logits, self.top_k, renormalize=True
            )
            full_ranking = torch.argsort(router_logits.float(), dim=-1, descending=True)
        else:
            sm = F.softmax(router_logits, dim=1, dtype=torch.float)
            full_ranking = torch.argsort(sm, dim=-1, descending=True)
            routing_weights, selected_experts = torch.topk(sm, self.top_k, dim=-1)

        routing_weights = routing_weights * self.routed_scaling_factor

        if prefix is not None:
            _store(f"{prefix}.router.in.0", hs_2d)
            _store(f"{prefix}.router.out.0", routing_weights.to(torch.float32))
            _store(f"{prefix}.router.out.1", full_ranking)

        final_hidden_states = torch.zeros(
            (bsz * seqlen, hdim), dtype=hs_2d.dtype, device=hs_2d.device
        )
        expert_mask = F.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
        assignment_counts = expert_mask.reshape(self.num_experts, -1).sum(dim=-1)

        if prefix is not None:
            _store(f"{prefix}.dispatch.selected_experts", selected_experts)
            _store(f"{prefix}.dispatch.routing_weights", routing_weights.to(torch.float32))
            _store(f"{prefix}.dispatch.expert_mask", expert_mask)
            _store(f"{prefix}.dispatch.assignment_counts", assignment_counts)

        for expert_idx in range(self.num_experts):
            idx, top_x = torch.where(expert_mask[expert_idx])

            current_state = hs_2d[None, top_x].reshape(-1, hdim)
            unscaled = self.get_expert_output(current_state, expert_idx)
            scales = routing_weights[top_x, idx, None]
            weighted = unscaled * scales

            if prefix is not None:
                ep = f"{prefix}.expert.{expert_idx}"
                _store(f"{ep}.token_indices", top_x)
                _store(f"{ep}.slot_indices", idx)
                _store(f"{ep}.unscaled_sums", unscaled.to(torch.float32))
                _store(f"{ep}.weighted_outputs", weighted.to(torch.float32))

            final_hidden_states.index_add_(0, top_x, weighted.to(hs_2d.dtype))

        return final_hidden_states.reshape(bsz, seqlen, hdim)

    return patched_moe_forward


def make_patched_attn_forward(modeling_module):
    apply_rotary_pos_emb = modeling_module.apply_rotary_pos_emb
    eager_attention_forward = getattr(modeling_module, "eager_attention_forward", None)

    def _fallback_eager(module, q, k, v, attention_mask, scaling, dropout=0.0, **kwargs):
        n_rep = module.num_key_value_groups
        if n_rep > 1:
            k = k.repeat_interleave(n_rep, dim=1)
            v = v.repeat_interleave(n_rep, dim=1)
        attn = torch.matmul(q, k.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn = attn + attention_mask[:, :, :, : k.shape[-2]]
        attn = F.softmax(attn, dim=-1, dtype=torch.float32).to(q.dtype)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous()
        return out, attn

    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    def patched_attn_forward(
        self,
        hidden_states,
        attention_mask=None,
        past_key_value=None,
        cache_position=None,
        position_ids=None,
        **kwargs,
    ):
        prefix = getattr(self, "_capture_prefix", None)
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(
            self.q_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        key_states = self.k_norm(
            self.k_proj(hidden_states).view(hidden_shape)
        ).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        if prefix is not None:
            _store(f"{prefix}.k_in", key_states)
            _store(f"{prefix}.v_in", value_states)
            _store(f"{prefix}.rope.q_in", query_states)
            _store(f"{prefix}.rope.k_in", key_states)

        if self.use_head_wise_attn_gate:
            gate_states = self.g_proj(hidden_states)

        cos, sin = self.rotary_emb(hidden_states, position_ids)
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if prefix is not None:
            _store(f"{prefix}.rope.q_embed", query_states)
            _store(f"{prefix}.rope.k_embed", key_states)
            _store(f"{prefix}.rope.cos", cos)
            _store(f"{prefix}.rope.sin", sin)

        if past_key_value is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

        attention_interface = eager_attention_forward or _fallback_eager
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        if prefix is not None:
            _store(f"{prefix}.attn", attn_output)
        attn_output = attn_output.reshape(*input_shape, -1)
        if self.use_head_wise_attn_gate:
            gate_sig = gate_states.unsqueeze(-1).sigmoid()
            head_view = attn_output.view(
                *attn_output.shape[:-1], self.num_attention_heads, self.head_dim
            )
            out = head_view * gate_sig
            if prefix is not None:
                _store(f"{prefix}.gate_sig", gate_sig)
                _store(f"{prefix}.gated", out)
            attn_output = out.view(*attn_output.shape)
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    return patched_attn_forward


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=os.environ.get("STEP3P5_MODEL", "stepfun-ai/Step-3.5-Flash"),
        help="HF repo dir or hub id (uses trust_remote_code)",
    )
    ap.add_argument("--prompt", default="The capital of France is")
    ap.add_argument("--out", default=str(HERE / "activations.safetensors"))
    ap.add_argument("--max-len", type=int, default=16)
    ap.add_argument(
        "--dtype",
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
    )
    args = ap.parse_args()

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    # The patched modeling.py uses rope_type="default" for sliding_attention
    # layers, but this version of transformers dropped the "default" entry from
    # ROPE_INIT_FUNCTIONS. Register a minimal unscaled implementation.
    if "default" not in ROPE_INIT_FUNCTIONS:
        def _compute_default_rope_parameters(config, device=None, **_):
            config.standardize_rope_params()
            params = config.rope_parameters
            base = params["rope_theta"]
            partial_rotary_factor = params.get("partial_rotary_factor", 1.0)
            head_dim = getattr(
                config, "head_dim",
                config.hidden_size // config.num_attention_heads,
            )
            dim = int(head_dim * partial_rotary_factor)
            inv_freq = 1.0 / (
                base
                ** (torch.arange(0, dim, 2, dtype=torch.int64).float().to(device) / dim)
            )
            return inv_freq, 1.0
        ROPE_INIT_FUNCTIONS["default"] = _compute_default_rope_parameters

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]

    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    if getattr(cfg, "pad_token_id", None) is None:
        eos = getattr(cfg, "eos_token_id", None)
        cfg.pad_token_id = eos[0] if isinstance(eos, (list, tuple)) else eos

    # The checkpoint stores 45 standard transformer layers (0..44) plus 3 MTP
    # heads (45..47) under non-standard parameter names that this modeling
    # script doesn't implement. If we keep num_hidden_layers=48 the loader
    # treats layers 45..47 as missing and re-runs weight init (which crashes
    # in this transformers version due to per-layer rope_parameters mutation).
    # We only need MoE layers 3..44 for the fixture, so drop the MTP layers.
    cfg.num_hidden_layers = 45
    if isinstance(getattr(cfg, "layer_types", None), list):
        cfg.layer_types = cfg.layer_types[:45]
    if isinstance(getattr(cfg, "rope_theta", None), list):
        cfg.rope_theta = cfg.rope_theta[:45]
    if hasattr(cfg, "num_nextn_predict_layers"):
        cfg.num_nextn_predict_layers = 0
    # Defaults the modeling code reads but Step3p5Config doesn't declare.
    for attr, default in [
        ("use_cache", False),
        ("output_attentions", False),
        ("output_hidden_states", False),
        ("output_router_logits", False),
    ]:
        if not hasattr(cfg, attr):
            setattr(cfg, attr, default)

    # The patched modeling.py keeps `config.rope_theta` as a 48-element list and
    # overwrites it to a per-layer scalar inside Step3p5RotaryEmbedding.__init__
    # before calling `rope_init_fn`. But HF's PreTrainedConfig.__init__ has
    # already copied the list into `config.rope_parameters["rope_theta"]`, which
    # is what `_compute_llama3_parameters` actually reads. Drop the stale list
    # key here, and patch RotaryEmbedding.__init__ so each layer refreshes the
    # dict with its own scalar before rope_init_fn runs.
    if isinstance(cfg.rope_parameters.get("rope_theta"), list):
        cfg.rope_parameters.pop("rope_theta", None)
    if getattr(cfg, "rope_scaling", None) is not None:
        cfg.rope_scaling.pop("rope_theta", None)

    RotaryCls = get_class_from_dynamic_module(
        "modeling_step3p5.Step3p5RotaryEmbedding", args.model
    )
    _orig_rotary_init = RotaryCls.__init__

    def _patched_rotary_init(self, config, device=None, layer_idx=None):
        # Pre-resolve the per-layer scalar so rope_init_fn sees a scalar.
        theta = config.rope_theta
        if isinstance(theta, list):
            scalar = theta[layer_idx] if layer_idx is not None else theta[0]
        else:
            scalar = theta
        # Overwrite any stale value in the shared rope_parameters / rope_scaling
        # dicts so `setdefault` inside standardize_rope_params doesn't keep the
        # previous layer's value (or the original list).
        if getattr(config, "rope_parameters", None) is not None:
            config.rope_parameters["rope_theta"] = scalar
        if getattr(config, "rope_scaling", None) is not None:
            config.rope_scaling["rope_theta"] = scalar
        _orig_rotary_init(self, config, device=device, layer_idx=layer_idx)

    RotaryCls.__init__ = _patched_rotary_init

    # Bypass HF's missing-keys re-init pass. It always runs to recreate
    # non-persistent buffers (inv_freq), and on this model it crashes inside
    # _compute_llama3_parameters because the patched per-layer rotary init
    # mutates the shared config.rope_parameters dict (eventually dropping
    # 'factor'). All real weights are loaded from the checkpoint, and rotary
    # inv_freq buffers are already populated during module __init__.
    from transformers.modeling_utils import PreTrainedModel
    PreTrainedModel._initialize_missing_keys = lambda self, *a, **k: None

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    print(f"[activations] loading {args.model} ({args.dtype})...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=cfg,
        torch_dtype=dtype,
        device_map="auto",
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        attn_implementation="eager",
    ).eval()

    decoder_layers = model.model.layers
    AttnCls = type(decoder_layers[0].self_attn)
    modeling_module = sys.modules[AttnCls.__module__]

    # low_cpu_mem_usage=True initialized modules on the `meta` device, so the
    # non-persistent `inv_freq` buffer in every Step3p5RotaryEmbedding was
    # materialized as zeros. Since we also disabled _initialize_missing_keys,
    # HF never re-ran the rotary init. Recompute inv_freq per-layer now.
    for layer_idx, layer in enumerate(decoder_layers):
        rot = getattr(layer.self_attn, "rotary_emb", None)
        if rot is None:
            continue
        scalar = cfg.rope_theta[layer_idx] if isinstance(cfg.rope_theta, list) else cfg.rope_theta
        if cfg.rope_parameters is not None:
            cfg.rope_parameters["rope_theta"] = scalar
        if getattr(cfg, "rope_scaling", None) is not None:
            cfg.rope_scaling["rope_theta"] = scalar
        prf = getattr(cfg, "partial_rotary_factors", None)
        if prf is not None:
            cfg.partial_rotary_factor = prf[layer_idx]
        device = rot.inv_freq.device if rot.inv_freq.device.type != "meta" else "cpu"
        inv_freq, scaling = rot.rope_init_fn(cfg, device)
        rot.inv_freq = inv_freq.to(device)
        rot.original_inv_freq = rot.inv_freq
        rot.attention_scaling = scaling

    moe_cls = None
    for layer in decoder_layers:
        if hasattr(layer, "moe"):
            moe_cls = type(layer.moe)
            break
    if moe_cls is None:
        print("[activations] WARNING: no MoE layer found in model", file=sys.stderr)

    for name, mod in model.named_modules():
        mod._capture_prefix = name

    patched_attn_fwd = make_patched_attn_forward(modeling_module)
    AttnCls.forward = patched_attn_fwd
    if moe_cls is not None:
        patched_moe_fwd = make_patched_moe_forward()
        moe_cls.forward = patched_moe_fwd

    # accelerate's device_map hooks snapshot each instance's original forward
    # as `_old_forward` and call that, ignoring later class-level overrides.
    # Rebind our patched functions onto each affected instance.
    for mod in model.modules():
        if hasattr(mod, "_old_forward"):
            cls = type(mod)
            if cls is AttnCls:
                mod._old_forward = patched_attn_fwd.__get__(mod, cls)
            elif moe_cls is not None and cls is moe_cls:
                mod._old_forward = patched_moe_fwd.__get__(mod, cls)

    handle = torch.nn.modules.module.register_module_forward_hook(
        _module_hook, with_kwargs=True
    )

    enc = tok(args.prompt, return_tensors="pt", truncation=True, max_length=args.max_len)
    first_device = next(model.parameters()).device
    enc = {k: v.to(first_device) for k, v in enc.items()}

    print(
        f"[activations] running forward on T={enc['input_ids'].shape[-1]}...",
        flush=True,
    )
    with torch.no_grad():
        _ = model(**enc)

    handle.remove()

    CAPTURES.pop(".in.0", None)
    CAPTURES.pop(".out.0", None)

    required_per_moe_layer = [
        "moe.in.0",
        "moe.router.in.0",
        "moe.router.out.1",
        "moe.dispatch.selected_experts",
        "moe.dispatch.routing_weights",
        "moe.dispatch.assignment_counts",
    ]
    moe_layer = 3
    missing = [
        f"model.layers.{moe_layer}.{k}"
        for k in required_per_moe_layer
        if f"model.layers.{moe_layer}.{k}" not in CAPTURES
    ]
    if missing:
        print(
            "[activations] ERROR: missing required canonical keys:",
            missing,
            file=sys.stderr,
        )
        return 1

    print(
        f"[activations] captured {len(CAPTURES)} tensors; writing {args.out}",
        flush=True,
    )
    payload = {k: v.clone() for k, v in CAPTURES.items()}
    save_file(payload, args.out)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(
        f"[activations] wrote {args.out} ({size_mb:.1f} MiB, {len(payload)} tensors)"
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
