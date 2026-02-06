#%%
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch
import numpy as np
import sys
from PIL import Image

from diffusers import Flux2KleinPipeline
from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps

from transformers.models.qwen2 import Qwen2TokenizerFast
from transformers.models.qwen3 import Qwen3ForCausalLM

from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from safetensors.torch import save_file

def unpatchify_latents(latents):
    batch_size, num_channels_latents, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), 2, 2, height, width)
    latents = latents.permute(0, 1, 4, 2, 5, 3)
    latents = latents.reshape(batch_size, num_channels_latents // (2 * 2), height * 2, width * 2)
    return latents


def unpack_latents_with_ids(x: torch.Tensor, x_ids: torch.Tensor) -> list[torch.Tensor]:
    """
    using position ids to scatter tokens into place
    """
    x_list = []
    for data, pos in zip(x, x_ids):
        _, ch = data.shape  # noqa: F841
        h_ids = pos[:, 1].to(torch.int64)
        w_ids = pos[:, 2].to(torch.int64)

        h = torch.max(h_ids) + 1
        w = torch.max(w_ids) + 1

        flat_ids = h_ids * w + w_ids

        out = torch.zeros((h * w, ch), device=data.device, dtype=data.dtype)
        out.scatter_(0, flat_ids.unsqueeze(1).expand(-1, ch), data)

        # reshape from (H * W, C) to (H, W, C) and permute to (C, H, W)

        out = out.view(h, w, ch).permute(2, 0, 1)
        x_list.append(out)

    return torch.stack(x_list, dim=0)

# Load Model
repo_id = "black-forest-labs/FLUX.2-klein-4B"
# 1. Check Inputs & Define Parameters
# ------------------------------------------------------------------------------
prompt = "A flying surperman style cat"
height = 128
width = 128
num_inference_steps = 1
guidance_scale = 1.0 # can be 3.5 Default guidance scale for Flux-Klein
num_images_per_prompt = 1
seed = 0
# device = "mps" if torch.backends.mps.is_available() else "cpu"
device = "cpu"
# dtype = torch.bfloat16
dtype = torch.bfloat16


print(f"Device: {device}, Dtype: {dtype}")
print(f"Prompt: {prompt}")


# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_text_ids
def prepare_text_ids(
    x: torch.Tensor,  # (B, L, D) or (L, D)
    t_coord: Optional[torch.Tensor] = None,
):
    B, L, _ = x.shape
    out_ids = []

    for i in range(B):
        t = torch.arange(1) if t_coord is None else t_coord[i]
        h = torch.arange(1)
        w = torch.arange(1)
        l = torch.arange(L)

        coords = torch.cartesian_prod(t, h, w, l)
        out_ids.append(coords)

    return torch.stack(out_ids)


# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._pack_latents
def pack_latents(latents):
    """
    pack latents: (batch_size, num_channels, height, width) -> (batch_size, height * width, num_channels)
    """
    batch_size, num_channels, height, width = latents.shape
    latents = latents.reshape(batch_size, num_channels, height * width).permute(0, 2, 1)
    return latents

# Copied from diffusers.pipelines.flux2.pipeline_flux2.Flux2Pipeline._prepare_latent_ids
def prepare_latent_ids(latents):
    batch_size, _, height, width = latents.shape

    t = torch.arange(1, device=latents.device)  # [0] - time dimension
    h = torch.arange(height, device=latents.device)
    w = torch.arange(width, device=latents.device)
    l = torch.arange(1, device=latents.device)  # [0] - layer dimension

    # Create position IDs: (H*W, 4)
    # Note: cartesian_prod works best on same device
    latent_ids = torch.cartesian_prod(t, h, w, l)

    # Expand to batch: (B, H*W, 4)
    latent_ids = latent_ids.unsqueeze(0).expand(batch_size, -1, -1)

    return latent_ids


pipe = Flux2KleinPipeline.from_pretrained(repo_id, dtype=dtype)
pipe.to(device)
hidden_states_layers: list[int] = [9, 18, 27]

pipeline = pipe # Alias for clarity

# 2. Define call parameters matching pipeline.__call__
# ------------------------------------------------------------------------------
if isinstance(prompt, str):
    batch_size = 1
else:
    batch_size = len(prompt)

# tokenizer: Qwen2TokenizerFast
# type of tokenizer: <class 'transformers.models.qwen2.tokenization_qwen2_fast.Qwen2TokenizerFast'>
messages = [{"role": "user", "content": prompt}]
text = pipeline.tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
print(f"apply_chat_template: text {text}")

inputs = pipeline.tokenizer(
    text,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt"
)

all_input_ids = [inputs["input_ids"]]
all_attention_masks = [inputs["attention_mask"]]
# print(f"tokenizer: inputs[input_ids] {inputs["input_ids"]}")
# print(f"tokenizer: inputs[attention_mask] {inputs["attention_mask"]}")
input_ids = torch.cat(all_input_ids, dim=0).to(device)
attention_mask = torch.cat(all_attention_masks, dim=0).to(device)

print(f"input_ids: {input_ids.flatten()}")
print(f"attention_mask: {attention_mask.flatten()}")

# =================================================
# tokenizer = AutoTokenizer.from_pretrained(repo_id, subfolder="tokenizer") in zml ??
# Same as above but using tokenizer directly
hand_tokenizer = Qwen2TokenizerFast.from_pretrained(
    repo_id, 
    subfolder="tokenizer"
)
hand_text = hand_tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False, # Specific to Qwen2-based flows
)


hand_inputs = hand_tokenizer(
    hand_text,
    padding="max_length",
    max_length=20,
    truncation=True,
    return_tensors="pt"
)

assert hand_inputs["input_ids"].equal(inputs["input_ids"])
assert hand_inputs["attention_mask"].equal(inputs["attention_mask"])


# =================================================


# text_encoder: Qwen3ForCausalLM,
# Forward pass through the model
output = pipeline.text_encoder(
    input_ids=input_ids,
    attention_mask=attention_mask,
    output_hidden_states=True,
    use_cache=False,
)

# Only use outputs from intermediate layers and stack them
out = torch.stack([output.hidden_states[k] for k in hidden_states_layers], dim=1)
print(f"out: {out.flatten()[:10]}")
print(f"out before: {dtype} : {device}")
out = out.to(dtype=torch.float32, device=device)
print(f"out2: {out.flatten()[:10]}")
batch_size, num_channels, seq_len, hidden_dim = out.shape
prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)


print(f"prompt_embeds 2: {prompt_embeds.flatten()[:10]}")


# batch_size, seq_len, _ = prompt_embeds.shape
# prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
# prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

text_ids = prepare_text_ids(prompt_embeds)
text_ids = text_ids.to(device)

# print(f"prompt_embeds: {prompt_embeds.flatten().tolist()[:10]}  text_ids: {text_ids.flatten()[:10]}")

# with open("prompt_embeds_original.txt", "w") as f:
    # f.write(str(prompt_embeds.flatten().tolist()[:20]))

# =================================================

hand_text_encoder = Qwen3ForCausalLM.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        dtype=torch.float32
    )
hand_output = hand_text_encoder(
    input_ids=hand_inputs["input_ids"].to(device),
    attention_mask=hand_inputs["attention_mask"].to(device),
    output_hidden_states=True,
    use_cache=False,
)

hand_out = torch.stack([hand_output.hidden_states[k] for k in hidden_states_layers], dim=1)
hand_out = hand_out.to(dtype=torch.float32, device=device)
assert hand_out.equal(out), "Mismatch in text encoder outputs"
print("Text encoder outputs match between pipeline and manual implementation.")

print(f"Prompt Embeds (first 20 vals): {hand_out.flatten().tolist()[:20]}")

hand_prompt_embeds = hand_out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
hand_text_ids = prepare_text_ids(hand_prompt_embeds)
hand_text_ids = hand_text_ids.to(device)

print(f"Prompt Embeds (first 20 vals): {hand_prompt_embeds.flatten().tolist()[:20]}")

assert hand_prompt_embeds.equal(prompt_embeds), "Mismatch in prompt embeds"
assert hand_text_ids.equal(text_ids), "Mismatch in text ids"
print("Prompt embeds and text ids match between pipeline and manual implementation.")

# =================================================

# 4. Prepare Latents
# ------------------------------------------------------------------------------
print("\n>>> [Step 4] Preparing Latents...")
print(f"pipeline.transformer.config.in_channels : {pipeline.transformer.config.in_channels}")
num_channels_latents = pipeline.transformer.config.in_channels // 4
print(f"   Num Channels Latents: {num_channels_latents}")
print(f"prompt_embeds.dtype : {prompt_embeds.dtype}")

# Let pipeline handle shape and packing
latents, latent_ids = pipeline.prepare_latents(
    batch_size=batch_size * num_images_per_prompt,
    num_latents_channels=num_channels_latents,
    height=height,
    width=width,
    dtype=torch.float32,
    device=device,
    generator=torch.Generator(device=device).manual_seed(seed)
)

print(f"    Latents : {latents.flatten().tolist()[:10]}")
print(f"    Latent_ids (first 10): {latent_ids.flatten().tolist()[:10]}")


# ===============================================================================

# Flux2Transformer2DModel


print("\n>>> [Step 5] Manual Latent Preparation...")

# 1. Load the Transformer Model manually
hand_transformer = Flux2Transformer2DModel.from_pretrained(
    repo_id,
    subfolder="transformer"
)
hand_transformer.to(device)

print(f"hand_transformer.dtype : {hand_transformer.dtype}")
# 2. Get Config params
# Flux packs 2x2 patches into the channel dimension, so latent channels = transformer inputs / 4
patch_size = hand_transformer.config.patch_size # Usually 1
num_channels_latents = hand_transformer.config.in_channels // 4

vae_scale_factor = 8 
adjusted_height = 2 * (int(height) // (vae_scale_factor * 2))
adjusted_width = 2 * (int(width) // (vae_scale_factor * 2))

num_channels_latents = pipeline.transformer.config.in_channels // 4

shape = (
    batch_size * num_images_per_prompt, 
    num_channels_latents * 4,
    adjusted_height // 2,
    adjusted_width // 2
)

hand_latents_raw = torch.randn(shape, generator=torch.Generator(device=device).manual_seed(seed), device=device, dtype=torch.float32).to(device)

print(f"    Hand Latents Raw (first 20): {hand_latents_raw.flatten().tolist()[:20]}")

# save_file({"latents": hand_latents_raw}, "flux2_latents.safetensors")
# print(f">>> Saved latents to 'flux2_latents.safetensors'")

hand_latent_ids = prepare_latent_ids(hand_latents_raw)
hand_latents = pack_latents(hand_latents_raw)
# Assert latents and latent_ids

# assert  latents, latent_ids to hand_latents, hand_latent_ids

print(f"    Hand Latents : {hand_latents.flatten().tolist()[:20]}")
print(f"    Hand Latent IDs (first 20): {hand_latent_ids.flatten().tolist()[:20]}")
assert hand_latents.equal(latents), "Mismatch in latents"
assert hand_latent_ids.equal(latent_ids), "Mismatch in latent_ids"
print("Latents and Latent IDs match between pipeline and manual implementation.")

# ===============================================================================



# ===============================================================================
# FlowMatchEulerDiscreteScheduler



# AutoencoderKLFlux2
# 1. Load VAE manually

# timesteps
# ===============================================================================


print("\n>>>FlowMatchEulerDiscreteScheduler...")


hand_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
    repo_id,
    subfolder="scheduler"
)
assert hand_scheduler.config == pipeline.scheduler.config, "Mismatch in scheduler config"

print(f"hand_scheduler.config : {hand_scheduler.config}")

# 5. Prepare Timesteps
# ------------------------------------------------------------------------------
print("\n>>> [Step 5] Preparing Timesteps...")
sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
image_seq_len = latents.shape[1]
mu = compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
print(f"   Latents (first 20): {latents.flatten().tolist()[:20]}")

# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
timesteps, num_inference_steps = retrieve_timesteps(
    pipeline.scheduler, # hand_scheduler
    num_inference_steps,
    device,
    sigmas=sigmas,
    mu=mu,
)

print(f"   Sigmas: {sigmas}")
print(f"   Empirical Mu: {mu:.4f}")
print(f"Scheduler Timesteps (first 20): {timesteps.flatten().tolist()[:20]}")
print(f"Scheduler Sigmas (first 20): {pipeline.scheduler.sigmas.flatten().tolist()[:20]}")
print(f"   Num Inference Steps: {num_inference_steps}")

# 6. Denoising Loop
# ------------------------------------------------------------------------------
print("\n>>> [Step 6] Starting Denoising Loop...")
pipeline.scheduler.set_begin_index(0)

with torch.no_grad():
    for i, timestep in enumerate(timesteps):
        print(f"   Step {i+1}/{num_inference_steps} (t={timestep:.2f})")
    
        # Broadcast timestep
        timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        
        latent_model_input = latents.to(hand_transformer.dtype)
        latent_image_ids = latent_ids
        
        # Transformer Forward Pass
        # We need to handle CFG if enabled
        
        # 1. Conditional Pass
        noise_pred = hand_transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]

        # Scheduler Step
        latents_dtype = latents.dtype
        # print first 10 latents before step
        print(f"   Latents before step: {latents.flatten()[:10]}")
        latents = pipeline.scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
             latents = latents.to(latents_dtype)
             
        print(f"   Latents after step (first 20): {latents.flatten()[:20]}")


# Stop here for debugging


# 7. Decoding
# ------------------------------------------------------------------------------
print("\n>>> [Step 7] Decoding...")
# Unpack
latents = unpack_latents_with_ids(latents, latent_ids)
print(f"   Unpacked Latents Shape: {latents.shape}")
print(f"   Latents after step (first 20): {latents.flatten()[:20]}")


# VAE Normalization (Flux2 Klein Specific)
latents_bn_mean = pipeline.vae.bn.running_mean.view(1, -1, 1, 1).to(latents.device, latents.dtype)
print(f"   VAE BN Running Mean (first 20): {latents_bn_mean.flatten()[:20]}")
latents_bn_std = torch.sqrt(pipeline.vae.bn.running_var.view(1, -1, 1, 1) + pipeline.vae.config.batch_norm_eps).to(
    latents.device, latents.dtype
)
print(f"   VAE BN Running Std (first 20): {latents_bn_std.flatten()[:20]}")
# Apply Denormalization
latents = latents * latents_bn_std + latents_bn_mean
print(f"   Latents after VAE Denormalization (first 20): {latents.flatten()[:20]}")

# Unpatchify
latents = unpatchify_latents(latents)
print(f"   Unpatched Latents Shape: {latents.shape}")
print(f"   Latents after Unpatchify (first 20): {latents.flatten()[:20]}")

hand_vae = AutoencoderKLFlux2.from_pretrained(
    repo_id,
    subfolder="vae"
)
hand_vae.to(device)

print(f"hand_vae.dtype : {hand_vae.config}")


# Calculate stats
min_val = latents.min()
max_val = latents.max()
mean_val = latents.mean()
non_zero = torch.count_nonzero(latents)
print(f"   VAE Input Latents: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, NonZero={non_zero}/{latents.numel()}")

print(f"latents 20 : {latents.flatten()[:20]}")

with torch.no_grad():
    # image = pipeline.vae.decode(latents, return_dict=False)[0]
    image = hand_vae.decode(latents, return_dict=False)[0]
print(f"   Decoded Image Shape: {image.shape}")
print(f"   Decoded Image (first 20): {image.flatten()[:20]}")
print(f"image dtype {image.dtype}")

# Pipeline Save
# image_pipe = pipeline.image_processor.postprocess(image, output_type="pil")[0]
# output_path = "flux_klein_result_pipe.png"
# image_pipe.save(output_path)
# print(f">>> Image saved to '{output_path}'")

# Manual Postprocess with PIL
image_hand = (image / 2 + 0.5).clamp(0, 1)
image_hand = image_hand.cpu().permute(0, 2, 3, 1).float().numpy()
image_hand = (image_hand * 255).round().astype("uint8")
print(f"image_hand 20 : {image_hand.flatten()[:20]}")
first_image = image_hand[0]
print(f"image_hand[0] 20 : {first_image.flatten()[:20]}")

np.save("first_image.npy", first_image)
print(f">>> Numpy array saved to 'first_image.npy'")

loaded_image = first_image
print(f"loaded_image 20 : {loaded_image.flatten()[:20]}")
first_image_pil = Image.fromarray(loaded_image)
output_path = "flux_klein_result_hand.png"
first_image_pil.save(output_path)
print(f">>> Image saved to '{output_path}'")

