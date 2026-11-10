import torch
import sounddevice as sd
from unittest.mock import patch
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch.nn import LayerNorm
from transformers import AutoProcessor, GradiumPhononTTSForConditionalGeneration
from transformers.models.gradium_phonon_tts.modeling_gradium_phonon_tts import GradiumPhononTTSMLP
import zml_utils

from transformers.conversion_mapping import get_checkpoint_conversion_mapping
from transformers.core_model_loading import WeightConverter, WeightRenaming, rename_source_key



def leaf_modules(module, prefix=""):
  children = list(module.named_children())

  if not children:
      yield prefix, module
      return

  for name, child in children:
      path = f"{prefix}.{name}" if prefix else name
      yield from leaf_modules(child, path)



model_id = "gradium/phonon-31m-en"
# voice_id = "LMDIYE0Oa377Qh6Z"
voice_id = "Tek4tJXiX6_yvXq7"


def print_model_tree(module, show_parameters=True, prefix=""):
  def visit(node, indent, path):
      entries = [
          (name, child)
          for name, child in node.named_children()
      ]

      if show_parameters:
          entries += [
              (name, parameter)
              for name, parameter in node.named_parameters(recurse=False)
          ]

      for index, (name, value) in enumerate(entries):
          last = index == len(entries) - 1
          branch = "└── " if last else "├── "
          full_name = f"{path}.{name}" if path else name
          next_indent = indent + ("    " if last else "│   ")

          if isinstance(value, torch.nn.Module):
              print(f"{indent}{branch}{name}: {type(value).__name__}")
              visit(value, next_indent, full_name)
          else:
              print(
                  f"{indent}{branch}{name}: "
                  f"shape={tuple(value.shape)}, dtype={value.dtype}"
              )
              checkpoint_name = original_name(full_name)
              if checkpoint_name not in checkpoint_names:
                  checkpoint_name = "(not present in checkpoint)"
              print(f"{next_indent}├── loaded: {full_name}")
              print(f"{next_indent}└── checkpoint: {checkpoint_name}")

  print(type(module).__name__)
  visit(module, "", prefix)

# processor = AutoProcessor.from_pretrained(model_id, device_map="auto")
# model = GradiumPhononTTSForConditionalGeneration.from_pretrained(model_id, device_map="auto")


processor = AutoProcessor.from_pretrained(model_id)

model = GradiumPhononTTSForConditionalGeneration.from_pretrained( model_id, dtype=torch.float32).to("cpu")
model.eval()

mapping = get_checkpoint_conversion_mapping(model.config.model_type)

reverse_mapping = [rule.reverse_transform() for rule in reversed(mapping)]
reverse_renamings = [
  rule for rule in reverse_mapping
  if isinstance(rule, WeightRenaming)
]
reverse_converters = [
  rule for rule in reverse_mapping if isinstance(rule, WeightConverter)
]

def original_name(python_name):
  checkpoint_name, _ = rename_source_key(
      python_name,
      reverse_renamings,
      reverse_converters,
      reverse=True,
  )
  return checkpoint_name

checkpoint_path = hf_hub_download(model_id, "model.safetensors", local_files_only=True)
with safe_open(checkpoint_path, framework="pt", device="cpu") as checkpoint:
  checkpoint_names = set(checkpoint.keys())

print_model_tree(model)

# if True:
#     for name, module in model.named_modules():
#         print(name, "->", type(module).__name__)
# else:
#     for name, module in model.named_modules():
#       if isinstance(module, GradiumPhononTTSMLP):
#           print(name, "->", type(module).__name__)


mlp_name = "model.layers.0.self_attn"
mlp = model.get_submodule(mlp_name)
print(mlp)
collector = zml_utils.ActivationCollector(mlp)



captured = {}

# def print_tensor_metadata(name, tensor):
#   print(f"{name}:")
#   print(f"    shape:         {tuple(tensor.shape)}")
#   print(f"    dtype:         {tensor.dtype}")
#   print(f"    device:        {tensor.device}")
#   print(f"    stride:        {tensor.stride()}")
#   print(f"    contiguous:    {tensor.is_contiguous()}")
#   print(f"    elements:      {tensor.numel()}")
#   print(f"    bytes/element: {tensor.element_size()}")
#   print(f"    logical bytes: {tensor.numel() * tensor.element_size()}")
#   print(f"    requires_grad: {tensor.requires_grad}")
#
# def capture_first_call(module, args, output):
#   if captured:
#       return
#
#   print(f"-v-v-v-v-v- Module: {mlp_name} ({type(module).__name__}) -v-v-v-v-v-")
#
#   for i, argument in enumerate(args):
#       print_tensor_metadata(f"input[{i}]", argument)
#
#   print_tensor_metadata("output", output)
#
#   for name, parameter in module.named_parameters():
#       full_name = f"{mlp_name}.{name}"
#       print(f"\n{full_name} -> {original_name(full_name)}")
#       print_tensor_metadata("parameter", parameter)
#
#   # captured[f"{mlp_name}.in"] = (
#   captured["in"] = (
#       args[0].detach().cpu().clone().contiguous()
#   )
#   # captured[f"{mlp_name}.out"] = (
#   captured["out"] = (
#       output.detach().cpu().clone().contiguous()
#   )
#
#
#   print(f"-^-^-^-^-^- Module: {mlp_name} ({type(module).__name__}) -^-^-^-^-^-")
#
# hook = mlp.register_forward_hook(capture_first_call)



def snapshot(tensor):
  return tensor.detach().cpu().clone().contiguous()


# Four complete, uncached model-body invocations: prompt, then 1/2/3 generated frames.
# Missing optional tensor keys mean Python passed None, not an empty tensor.
transformer_calls = {"fixtures": [], "current": None}

def capture_transformer_call_input(module, args, kwargs):
  transformer_calls["current"] = None
  if len(transformer_calls["fixtures"]) >= 4:
      return
  if kwargs.get("use_cache") is not False or kwargs.get("past_key_values") is not None:
      raise RuntimeError("Transformer-call fixtures require use_cache=False and no KV cache")
  fixture = {}
  for name in ("input_ids", "audio_prompt_latents", "audio_prompt_latents_mask",
               "audio_latents", "audio_latents_mask", "attention_mask", "position_ids",
               "inputs_embeds"):
      value = kwargs.get(name)
      if name == "input_ids" and value is None and args:
          value = args[0]
      if isinstance(value, torch.Tensor):
          fixture[name] = snapshot(value)
  transformer_calls["fixtures"].append(fixture)
  transformer_calls["current"] = fixture


def capture_transformer_call_embeddings(module, args, output):
  fixture = transformer_calls["current"]
  if fixture is not None:
      fixture["embeddings"] = snapshot(output)


def capture_transformer_call_prepared(module, args, kwargs):
  fixture = transformer_calls["current"]
  if fixture is not None:
      fixture["prepared_embeddings"] = snapshot(args[0] if args else kwargs["hidden_states"])
      fixture["position_ids"] = snapshot(kwargs["position_ids"])
      cos, sin = kwargs["position_embeddings"]
      fixture["cos"] = snapshot(cos)
      fixture["sin"] = snapshot(sin)
      if kwargs.get("attention_mask") is not None:
          fixture["causal_mask"] = snapshot(kwargs["attention_mask"])


def capture_transformer_call_output(module, args, output):
  fixture = transformer_calls["current"]
  if fixture is not None:
      fixture["out"] = snapshot(output.last_hidden_state)
  transformer_calls["current"] = None


def register_transformer_call_hooks(model):
  return [
      model.model.register_forward_pre_hook(capture_transformer_call_input, with_kwargs=True),
      model.model.embed_tokens.register_forward_hook(capture_transformer_call_embeddings),
      model.model.layers[0].register_forward_pre_hook(capture_transformer_call_prepared, with_kwargs=True),
      model.model.register_forward_hook(capture_transformer_call_output),
  ]


def save_transformer_call_fixtures():
  if len(transformer_calls["fixtures"]) != 4:
      raise RuntimeError("Expected four uncached transformer calls")
  for index, fixture in enumerate(transformer_calls["fixtures"]):
      if "out" not in fixture:
          raise RuntimeError(f"Transformer call {index} did not finish")
      path = f"model.call.{index}.activations.safetensors"
      save_file(fixture, path, metadata={
          "call_index": str(index), "use_cache": "false",
          "audio_latents_space": "normalized TTS latents, before decoder rescaling",
          "optional_inputs": "absent keys mean None",
          "embeddings": "raw token lookup before replacements",
          "prepared_embeddings": "after replacements, before layer 0",
          "out": "after all six layers and final normalization",
      })
      # Projection-only fixture: absent history becomes a shape-explicit empty
      # tensor for Zig's full-history input contract. Other captured data is unchanged.
      projection = {
          "in": fixture["embeddings"], "input_ids": fixture["input_ids"],
          "audio_prompt_latents": fixture["audio_prompt_latents"],
          "audio_latents": fixture.get("audio_latents", torch.empty(
              (fixture["input_ids"].shape[0], 0, 32), dtype=torch.float32)),
          "out": fixture["prepared_embeddings"],
      }
      save_file(projection, f"model.call.{index}.prepare_inputs.activations.safetensors")
      save_file({"input_ids": fixture["input_ids"],
                 "audio_prompt_latents": fixture["audio_prompt_latents"],
                 "audio_latents": projection["audio_latents"], "out": fixture["out"]},
                f"model.call.{index}.forward.activations.safetensors")
      print(f"Saved {path}")
      for name, tensor in fixture.items():
          print(f"  {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")


decoder_layers_captured = [{} for _ in range(6)]

def capture_first_decoder_layer(index):
  """Capture each layer's own first prefill input/output, before generation repeats."""
  def capture(module, args, kwargs, output):
      fixture = decoder_layers_captured[index]
      if fixture:
          return
      fixture["in"] = snapshot(args[0] if args else kwargs["hidden_states"])
      fixture["out"] = snapshot(output)
  return capture


def save_decoder_layer_fixtures():
  for index, fixture in enumerate(decoder_layers_captured):
      save_file(fixture, f"model.layers.{index}.activations.safetensors")


transformer_captured = {}
model_captured = {}
eos_captured = {}
timestep_captured = {}
input_projection_captured = {}
condition_projection_captured = {}

def save_prediction_fixture(token_ids, inputs):
  """One initial prediction with explicit candidate noise and flow times."""
  save_file({
      "input_ids": snapshot(token_ids),
      "voice": snapshot(inputs["audio_prompt_latents"]),
      "audio_latents": torch.empty((token_ids.shape[0], 0, 32), dtype=torch.float32),
      "candidate": snapshot(flow_head_captured["in"]),
      "start_time": snapshot(flow_head_captured["start_time"]),
      "end_time": snapshot(flow_head_captured["end_time"]),
      "eos_score": snapshot(eos_captured["out"][:, -1]),
      "velocity": snapshot(flow_head_captured["out"]),
  }, "prediction.activations.safetensors")


# Record actual sampler inputs/results, so Zig need not match PyTorch's RNG.
generation_captured = []

def capture_sample_latent(condition, noise, num_flow_steps):
  result = original_sample_latent(condition, noise, num_flow_steps)
  if len(generation_captured) < 4:
      generation_captured.append({"noise": snapshot(noise), "frame": snapshot(result),
                                  "eos": snapshot(model.eos_head(condition))})
  return result


def save_generation_fixture(token_ids, inputs):
  if len(generation_captured) != 4:
      raise RuntimeError("Expected four sampled frames")
  save_file({"input_ids": snapshot(token_ids),
             "voice": snapshot(inputs["audio_prompt_latents"]),
             "noise": torch.cat([x["noise"] for x in generation_captured]),
             "frames": torch.cat([x["frame"] for x in generation_captured]),
             "eos": torch.cat([x["eos"] for x in generation_captured])},
            "generation.activations.safetensors",
            metadata={"latent_space": "normalized TTS", "num_flow_steps": "1", "use_cache": "false"})


flow_head_captured = {}

def capture_first_flow_head(module, args, kwargs, output):
  if flow_head_captured:
      return
  times = kwargs["times"]
  flow_head_captured["in"] = snapshot(args[0])
  flow_head_captured["condition"] = snapshot(args[1])
  flow_head_captured["start_time"] = snapshot(times[0])
  flow_head_captured["end_time"] = snapshot(times[1])
  flow_head_captured["out"] = snapshot(output)


def save_flow_head_fixtures(model):
  save_file(flow_head_captured, "flow_head.activations.safetensors")
  generator = torch.Generator(device="cpu").manual_seed(42)
  candidate = torch.randn((4, model.flow_head.input_proj.in_features), generator=generator)
  condition = torch.randn((4, model.flow_head.cond_proj.in_features), generator=generator)
  start = torch.tensor([[0.0], [0.125], [0.5], [0.75]])
  end = torch.tensor([[0.25], [0.375], [0.75], [1.0]])
  save_file({"in": snapshot(candidate), "condition": snapshot(condition),
             "start_time": snapshot(start), "end_time": snapshot(end),
             "out": snapshot(model.flow_head(candidate, condition, times=[start, end]))},
            "flow_head.batch.activations.safetensors")


flow_final_layer_captured = {}

def capture_first_flow_final_layer(module, args, output):
  if flow_final_layer_captured:
      return
  flow_final_layer_captured["in"] = snapshot(args[0])
  flow_final_layer_captured["condition"] = snapshot(args[1])
  flow_final_layer_captured["out"] = snapshot(output)


def save_flow_final_layer_fixtures(model):
  save_file(flow_final_layer_captured, "flow_head.final_layer.activations.safetensors")
  generator = torch.Generator(device="cpu").manual_seed(42)
  sample = torch.randn((4, model.flow_head.final_layer.proj.in_features), generator=generator)
  condition = torch.randn(sample.shape, generator=generator)
  save_file({"in": snapshot(sample), "condition": snapshot(condition),
             "out": snapshot(model.flow_head.final_layer(sample, condition))},
            "flow_head.final_layer.batch.activations.safetensors")


flow_res_blocks_captured = [{} for _ in range(6)]

def capture_first_flow_res_block(index):
  def capture(module, args, output):
      fixture = flow_res_blocks_captured[index]
      if fixture:
          return
      fixture["in"] = snapshot(args[0])
      fixture["condition"] = snapshot(args[1])
      fixture["out"] = snapshot(output)
  return capture


def save_flow_res_block_fixtures(model):
  generator = torch.Generator(device="cpu").manual_seed(42)
  for index, block in enumerate(model.flow_head.layers):
      save_file(flow_res_blocks_captured[index], f"flow_head.layers.{index}.activations.safetensors")
      sample = torch.randn((4, block.fc1.in_features), generator=generator)
      condition = torch.randn(sample.shape, generator=generator)
      save_file({"in": snapshot(sample), "condition": snapshot(condition),
                 "out": snapshot(block(sample, condition))},
                f"flow_head.layers.{index}.batch.activations.safetensors")


def capture_first_input_projection(module, args, output):
  """Keep the first candidate audio latent projection call."""
  if input_projection_captured:
      return
  input_projection_captured["in"] = snapshot(args[0])
  input_projection_captured["out"] = snapshot(output)

def capture_first_condition_projection(module, args, output):
  """Keep the first transformer conditioning vector projection call."""
  if condition_projection_captured:
      return
  condition_projection_captured["in"] = snapshot(args[0])
  condition_projection_captured["out"] = snapshot(output)

def capture_first_timestep(module, args, output):
  """Keep the first call to flow_head.timestep_embeddings.0."""
  if timestep_captured:
      return
  timestep_captured["in"] = snapshot(args[0])
  timestep_captured["out"] = snapshot(output)

def capture_first_eos(module, args, output):
  """Keep the first EOS-head call: hidden states in, raw end-of-speech scores out."""
  if eos_captured:
      return
  eos_captured["in"] = snapshot(args[0])
  eos_captured["out"] = snapshot(output)

def capture_model_output(module, args, output):
  """Keep the first model-body output, after its final layer normalization."""
  if "out" not in model_captured:
      model_captured["out"] = snapshot(output.last_hidden_state)

def capture_transformer_input(module, args, kwargs):
  """Keep the first prefill's embeddings after replacements, before layer 0."""
  if "in" in transformer_captured:
      return

  hidden_states = args[0] if args else kwargs["hidden_states"]
  transformer_captured["in"] = snapshot(hidden_states)
  transformer_captured["position_ids"] = snapshot(kwargs["position_ids"])
  cos, sin = kwargs["position_embeddings"]
  transformer_captured["cos"] = snapshot(cos)
  transformer_captured["sin"] = snapshot(sin)
  mask = kwargs.get("attention_mask")
  if mask is not None:
      transformer_captured["attention_mask"] = snapshot(mask)


def capture_transformer_output(module, args, output):
  """Keep the last layer's output, before the model's final normalization."""
  if "out" in transformer_captured:
      return

  transformer_captured["out"] = snapshot(output)


def capture_first_attention(module, args, kwargs, output):
  if captured:
      return

  cos, sin = kwargs["position_embeddings"]

  captured["in"] = snapshot(kwargs["hidden_states"])
  captured["out"] = snapshot(output[0])
  captured["position_ids"] = snapshot(kwargs["position_ids"])
  captured["cos"] = snapshot(cos)
  captured["sin"] = snapshot(sin)

  mask = kwargs.get("attention_mask")
  if mask is not None:
      captured["attention_mask"] = snapshot(mask)

def inspect_first_call(module, args, kwargs, output):

  global captured

  if captured:
      return
  captured = True

  print(f"-v-v-v-v-v- Module: {mlp_name} ({type(module).__name__}) -v-v-v-v-v-")

  def describe(name, value):
      if isinstance(value, torch.Tensor):
          print(name, tuple(value.shape), value.dtype)
      elif isinstance(value, (tuple, list)):
          for i, item in enumerate(value):
              describe(f"{name}[{i}]", item)
      elif isinstance(value, dict):
          for key, item in value.items():
              describe(f"{name}.{key}", item)
      else:
          print(name, type(value).__name__)

  describe("args", args)
  describe("kwargs", kwargs)
  describe("output", output)

  print(f"-^-^-^-^-^- Module: {mlp_name} ({type(module).__name__}) -^-^-^-^-^-")

# hook = mlp.register_forward_hook( inspect_first_call, with_kwargs=True)
hook = mlp.register_forward_hook( capture_first_attention, with_kwargs=True)


def load_voice(voice_id):
    """A voice of gradium/phonon-voices, as latents of shape (num_frames, 512)."""
    path = hf_hub_download("gradium/phonon-voices", f"voices/{voice_id}.safetensors")
    return load_file(path)["speaker_wavs"][0].T.float()


# Easton (en-us); the checkpoint's own voice: hf_hub_download(model_id, "default-voice.safetensors")
voice = load_voice(voice_id)
# an empty `audio` item marks the voice
conversation = [
    {"role": "user", "content": [{"type": "audio"}, {"type": "text", "text": "Bonjour Corentin dit Cerisier"}]}
]

tokenizer_captured = {}
original_tokenizer_call = type(processor.tokenizer).__call__

def capture_tokenizer_call(tokenizer, *args, **kwargs):
  """Capture the actual tokenizer boundary, after the processor expands placeholders."""
  text = kwargs.get("text", args[0] if args else None)
  output = original_tokenizer_call(tokenizer, *args, **kwargs)
  if tokenizer is not processor.tokenizer or tokenizer_captured:
      return output

  texts = [text] if isinstance(text, str) else text
  if not isinstance(texts, (list, tuple)) or not all(isinstance(item, str) for item in texts):
      raise TypeError("Tokenizer capture expects a string or a batch of strings")

  # Safetensors has no string tensors. Store exact UTF-8 bytes and batch boundaries.
  encoded = [item.encode("utf-8") for item in texts]
  offsets = [0]
  for item in encoded:
      offsets.append(offsets[-1] + len(item))
  tokenizer_captured["in"] = torch.tensor(list(b"".join(encoded)), dtype=torch.uint8)
  tokenizer_captured["in_offsets"] = torch.tensor(offsets, dtype=torch.int64)
  tokenizer_captured["out"] = snapshot(torch.as_tensor(output["input_ids"]))
  if "attention_mask" in output:
      tokenizer_captured["attention_mask"] = snapshot(torch.as_tensor(output["attention_mask"]))
  return output


# Tokenizers are not nn.Modules, so forward hooks do not apply. Python looks up
# __call__ on the class; patch it only while preparing this conversation.
with patch.object(type(processor.tokenizer), "__call__", capture_tokenizer_call):
  inputs = processor.apply_chat_template(
      conversation,
      tokenize=True,
      return_dict=True,
      add_generation_prompt=True,
      processor_kwargs={"audio_prompt_latents": [voice]},
  )

print("Token IDs:", inputs["input_ids"].tolist())

if not tokenizer_captured:
    raise RuntimeError("The processor did not call the tokenizer during input preparation")
save_file(tokenizer_captured, "tokenizer.activations.safetensors", metadata={
    "model_id": model_id,
    "input_encoding": "UTF-8",
    "input_layout": "Text i is in[in_offsets[i]:in_offsets[i+1]]; out contains input_ids",
})
for name, tensor in tokenizer_captured.items():
    print(f"tokenizer {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

inputs = inputs.to(model.device)

# Capture the lookup alone, before the model replaces voice/audio-start positions.
with torch.inference_mode():
    token_ids = inputs["input_ids"]
    embeddings = model.model.embed_tokens(token_ids)
    save_file({
        "in": snapshot(token_ids),
        "out": snapshot(embeddings),
    }, "model.embed_tokens.activations.safetensors")
    print(f"embedding in: shape={tuple(token_ids.shape)}, dtype={token_ids.dtype}")
    print(f"embedding out: shape={tuple(embeddings.shape)}, dtype={embeddings.dtype}")

# ModuleList is a container, so hook the first/last layers around its loop.
# The final model.norm is deliberately outside these capture boundaries.
transformer_call_hooks = register_transformer_call_hooks(model)
decoder_layer_hooks = [
    layer.register_forward_hook(capture_first_decoder_layer(index), with_kwargs=True)
    for index, layer in enumerate(model.model.layers)
]
transformer_hooks = [
    model.model.register_forward_hook(capture_model_output),
    model.model.layers[0].register_forward_pre_hook(capture_transformer_input, with_kwargs=True),
    model.model.layers[-1].register_forward_hook(capture_transformer_output),
]
eos_hook = model.eos_head.register_forward_hook(capture_first_eos)
timestep_hook = model.flow_head.timestep_embeddings[0].register_forward_hook(capture_first_timestep)

flow_head_hook = model.flow_head.register_forward_hook(capture_first_flow_head, with_kwargs=True)
flow_final_layer_hook = model.flow_head.final_layer.register_forward_hook(capture_first_flow_final_layer)
flow_res_block_hooks = [
    block.register_forward_hook(capture_first_flow_res_block(index))
    for index, block in enumerate(model.flow_head.layers)
]
input_projection_hook = model.flow_head.input_proj.register_forward_hook(capture_first_input_projection)
condition_projection_hook = model.flow_head.cond_proj.register_forward_hook(capture_first_condition_projection)

original_sample_latent = model.sample_latent
model.sample_latent = capture_sample_latent

try:
  with torch.inference_mode():
     outputs = model.generate(**inputs, use_cache=False)
     audio = processor.decode(
         outputs.audio_latents,
         outputs.audio_latents_mask,
     )

     from safetensors.torch import save_file
     save_file(captured, f"{mlp_name}.activations.safetensors")
     save_transformer_call_fixtures()
     save_decoder_layer_fixtures()
     save_file(transformer_captured, "model.layers.activations.safetensors")
     save_file(eos_captured, "eos_head.activations.safetensors")
     save_prediction_fixture(token_ids, inputs)
     save_generation_fixture(token_ids, inputs)
     save_flow_res_block_fixtures(model)
     save_flow_final_layer_fixtures(model)
     save_flow_head_fixtures(model)
     save_file(input_projection_captured, "flow_head.input_proj.activations.safetensors")
     save_file(condition_projection_captured, "flow_head.cond_proj.activations.safetensors")
     # Nonzero, batched candidates also exercise the input projection's weights.
     generator = torch.Generator(device="cpu").manual_seed(42)
     for name, projection, width in (
         ("input_proj", model.flow_head.input_proj, 32),
         ("cond_proj", model.flow_head.cond_proj, 512),
     ):
         sample = torch.randn((4, width), generator=generator, device="cpu")
         save_file({"in": snapshot(sample), "out": snapshot(projection(sample))},
                   f"flow_head.{name}.batch.activations.safetensors")
     for label, fixture in (("input projection", input_projection_captured),
                            ("condition projection", condition_projection_captured)):
         for name, tensor in fixture.items():
             print(f"{label} {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
     save_file(timestep_captured, "flow_head.timestep_embeddings.0.activations.safetensors")
     for name, tensor in timestep_captured.items():
         print(f"timestep embedding 0 {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
     for name, tensor in eos_captured.items():
         print(f"EOS head {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")

     model_inputs = {"in": snapshot(token_ids), "out": model_captured["out"]}
     for name in ("audio_prompt_latents", "audio_prompt_latents_mask", "attention_mask"):
         if name in inputs:
             model_inputs[name] = snapshot(inputs[name])
     save_file(model_inputs, "model.activations.safetensors")

     # No hidden-state transformation occurs between the final replacement and
     # layer 0; its input is exactly the post-replacement tensor we want.
     prepared_inputs = {
         "in": snapshot(embeddings),
         "out": transformer_captured["in"],
         "input_ids": snapshot(token_ids),
     }
     for name in ("audio_prompt_latents", "audio_prompt_latents_mask", "audio_latents", "audio_latents_mask"):
         if name in inputs:
             prepared_inputs[name] = snapshot(inputs[name])
     save_file(prepared_inputs, "model.prepare_inputs.activations.safetensors")
     print(f"after replacements: shape={tuple(prepared_inputs['out'].shape)}, dtype={prepared_inputs['out'].dtype}")

     for name, tensor in transformer_captured.items():
         print(f"transformer loop {name}: shape={tuple(tensor.shape)}, dtype={tensor.dtype}")
finally:
  model.sample_latent = original_sample_latent
  hook.remove()
  eos_hook.remove()
  timestep_hook.remove()
  flow_head_hook.remove()
  flow_final_layer_hook.remove()
  for block_hook in flow_res_block_hooks:
      block_hook.remove()
  input_projection_hook.remove()
  condition_projection_hook.remove()
  for transformer_call_hook in transformer_call_hooks:
      transformer_call_hook.remove()
  for decoder_layer_hook in decoder_layer_hooks:
      decoder_layer_hook.remove()
  for transformer_hook in transformer_hooks:
      transformer_hook.remove()


# print("Playing generated audio...")
# waveforms = audio if isinstance(audio, (list, tuple)) else [audio]
# for waveform in waveforms:
#     sd.play(waveform.detach().cpu().float().numpy(), samplerate=processor.feature_extractor.sampling_rate)
#     sd.wait()
# print("Playback finished.")
