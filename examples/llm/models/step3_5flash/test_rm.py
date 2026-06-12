from pathlib import Path

import torch
from accelerate import init_empty_weights

from step3_5flash.configuration_step3p5 import Step3p5Config
from step3_5flash.non_redundant_modeling_step3p5 import Step3p5ForCausalLM


MODEL_DIR = Path(__file__).resolve().parent


def load_config():
  cfg = Step3p5Config.from_pretrained(MODEL_DIR)

  if getattr(cfg, "pad_token_id", None) is None:
      eos = cfg.eos_token_id
      cfg.pad_token_id = eos[0] if isinstance(eos, list) else eos

  return cfg


def test_definition_constructs():
  cfg = load_config()

  with init_empty_weights():
      model = Step3p5ForCausalLM(cfg)

  print("definition construction ok")
  print("layers:", len(model.model.layers))
  print("hidden_size:", model.config.hidden_size)
  print("vocab_size:", model.config.vocab_size)


def test_full_forward():
  from transformers import AutoTokenizer

  cfg = load_config()
  tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)

  model = Step3p5ForCausalLM.from_pretrained(
      MODEL_DIR,
      config=cfg,
      torch_dtype=torch.bfloat16,
      device_map="auto",
      low_cpu_mem_usage=True,
  ).eval()

  inputs = tok("The capital of France is", return_tensors="pt")
  first_device = next(model.parameters()).device
  inputs = {k: v.to(first_device) for k, v in inputs.items()}

  with torch.no_grad():
      out = model(**inputs)

  next_id = out.logits[:, -1].argmax(dim=-1)
  print("full forward ok")
  print(tok.decode(next_id))


if __name__ == "__main__":
  test_definition_constructs()

  # Uncomment only when ready to load real weights.
  test_full_forward()
