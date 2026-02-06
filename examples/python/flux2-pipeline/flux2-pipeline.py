#%%
from typing import Dict, Tuple
import os
import sys
import torch
import numpy as np
from PIL import Image

from transformers.models.qwen2 import Qwen2TokenizerFast
from transformers.models.qwen3 import Qwen3ForCausalLM

# from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel
# from diffusers.models.autoencoders.autoencoder_kl_flux2 import AutoencoderKLFlux2
# from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from transformer_flux2 import Flux2Transformer2DModel
from autoencoder_kl_flux2 import AutoencoderKLFlux2
from scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

import flux2_tools
import utils

def get_tokens_from_prompt(repo_id: str, prompt: str, max_length=20) -> Dict[str, torch.Tensor]:
    hand_tokenizer = Qwen2TokenizerFast.from_pretrained(
        repo_id,
        subfolder="tokenizer"
    )
    text = hand_tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False, # Specific to Qwen2-based flows
    )
    print(f"text_templated: from {prompt} to {text}")

    token = hand_tokenizer(
        text,
        padding="max_length",
        max_length=max_length,
        truncation=True,
        return_tensors="pt",
    )
    return token


def get_encoding_embeded_from_tokens(repo_id: str, 
                                     tokens: Dict[str, torch.Tensor],
                                     hidden_states_layers: list[int] = [9, 18, 27],
                                     device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    
    token_encoder = Qwen3ForCausalLM.from_pretrained(
        repo_id,
        subfolder="text_encoder",
        dtype=torch.float32,
    )
    token_encoder.to(device)
    text_outputs = token_encoder(
        input_ids=tokens["input_ids"].to(device),
        attention_mask=tokens["attention_mask"].to(device),
        output_hidden_states=True,
        use_cache=False,
    )
    out = torch.stack([text_outputs.hidden_states[k] for k in hidden_states_layers], dim=1)
    out = out.to(dtype=torch.float32, device=device)
    batch_size, num_channels, seq_len, hidden_dim = out.shape
    assert batch_size == 1, "Only batch size 1 is supported currently."
    prompt_embeds = out.permute(0, 2, 1, 3).reshape(batch_size, seq_len, num_channels * hidden_dim)
    text_ids = flux2_tools.prepare_text_ids(prompt_embeds)
    text_ids = text_ids.to(device)
    return prompt_embeds, text_ids


def get_latents(transformer: Flux2Transformer2DModel,
                height = 128, width = 128,
                num_images_per_prompt = 1,
                device = "cpu") -> Tuple[torch.Tensor, torch.Tensor]:
    seed = 0
    batch_size = 1
    transformer.to(device)
    num_channels_latents = transformer.config.in_channels // 4
    vae_scale_factor = 8 
    adjusted_height = 2 * (int(height) // (vae_scale_factor * 2))
    adjusted_width = 2 * (int(width) // (vae_scale_factor * 2))
    num_channels_latents = transformer.config.in_channels // 4
    shape = (
        batch_size * num_images_per_prompt, 
        num_channels_latents * 4,
        adjusted_height // 2,
        adjusted_width // 2
    )

    # generator = torch.Generator(device=device).manual_seed(seed)
    # latents_raw = torch.randn(shape, generator=generator, device=device, dtype=torch.float32).to(device)
    
    generator = utils.BoxMullerGenerator(seed=seed, device=device)
    latents_raw = generator.randn(shape).to(torch.float32).to(device)

    print(f"    Latents Raw (first 20): {latents_raw.flatten().tolist()[:20]}")
    latent_ids = flux2_tools.prepare_latent_ids(latents_raw)
    latents = flux2_tools.pack_latents(latents_raw)
    return latents, latent_ids


def schedule(transformer: Flux2Transformer2DModel, 
             scheduler: FlowMatchEulerDiscreteScheduler, 
             latents: torch.Tensor,
             latent_ids: torch.Tensor,
             prompt_embeds: torch.Tensor,
             text_ids: torch.Tensor,
             num_inference_steps: int = 1, 
             device = "cpu") -> torch.Tensor:

    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    image_seq_len = latents.shape[1]
    empirical_mu = flux2_tools.compute_empirical_mu(image_seq_len=image_seq_len, num_steps=num_inference_steps)
    print(f"   Latents (first 20): {latents.flatten().tolist()[:20]}")
    timesteps, num_inference_steps = flux2_tools.retrieve_timesteps(
        scheduler=scheduler,
        num_inference_steps=num_inference_steps,
        device=device,
        sigmas=sigmas,
        mu=empirical_mu)    
    print(f"   Sigmas: {sigmas}")
    print(f"   Empirical Mu: {empirical_mu:.4f}")
    print(f"Scheduler Timesteps (first 20): {timesteps.flatten().tolist()[:20]}")
    print(f"Scheduler Sigmas (first 20): {scheduler.sigmas.flatten().tolist()[:20]}")
    print(f"   Num Inference Steps: {num_inference_steps}")
    scheduler.set_begin_index(0)
    for idx, timestep in enumerate(timesteps):
        print(f"   Step {idx+1}/{num_inference_steps} (t={timestep:.2f})")
        
        # Broadcast timestep
        timestep = timestep.expand(latents.shape[0]).to(latents.dtype)
        
        latent_model_input = latents.to(transformer.dtype)
        latent_image_ids = latent_ids
        
        noise_pred = transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=None,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_image_ids,
            joint_attention_kwargs=None,
            return_dict=False,
        )[0]
        print(f"   noise_pred (transformer output) (first 20): {noise_pred.flatten()[:20].tolist()}")

        # Scheduler Step
        latents_dtype = latents.dtype
        print(f"   Latents before step. {latents.shape} : {latents.flatten()[:20]}")
        latents = scheduler.step(noise_pred, timestep, latents, return_dict=False)[0]
        if latents.dtype != latents_dtype:
             latents = latents.to(latents_dtype)
        print(f"   Latents after step (first 20) {latents.shape} : {latents.flatten()[:20]}")
    
    print(f"   Latents after step (first 20). {latents.shape} : {latents.flatten()[:20]}")
    latents_out: torch.Tensor = flux2_tools.unpack_latents_with_ids(latents, latent_ids)
    print(f"   Latents Out (NCHW) (first 20): {latents_out.flatten()[:20].tolist()}")
    return latents_out



def variational_auto_encode(variational_auto_encoder: AutoencoderKLFlux2, latents: torch.Tensor, device="cpu") -> torch.Tensor:
    variational_auto_encoder.to(device)
    # assert latents.dtype == torch.float32, "Latents must be float32 for VAE decoding."
    # VAE Normalization (Flux2 Klein Specific)
    # assert latents.dtype == torch.float32, "Latents must be float32 for VAE decoding."
    latents_bn_mean = variational_auto_encoder.bn.running_mean.view(1, -1, 1, 1).to(device, latents.dtype)
    print(f"    VAE BN Running Mean (first 20): {latents_bn_mean.flatten()[:20]}")
    latents_bn_std = torch.sqrt(variational_auto_encoder.bn.running_var.view(1, -1, 1, 1) + variational_auto_encoder.config.batch_norm_eps).to(
        latents.device, latents.dtype
    )
    print(f"    VAE BN Running Std (first 20): {latents_bn_std.flatten()[:20]}")
    # Apply Denormalization
    latents = latents * latents_bn_std + latents_bn_mean
    print(f"    Latents after VAE Denormalization (first 20): {latents.flatten()[:20]}")
    latents = flux2_tools.unpatchify_latents(latents)
    print(f"    Unpatched Latents Shape: {latents.shape}")
    print(f"    Latents after Unpatchify (first 20): {latents.flatten()[:20]}")
    # Calculate stats
    min_val = latents.min()
    max_val = latents.max()
    mean_val = latents.mean()
    non_zero = torch.count_nonzero(latents)
    print(f"   VAE Input Latents: Min={min_val:.4f}, Max={max_val:.4f}, Mean={mean_val:.4f}, NonZero={non_zero}/{latents.numel()}")
    image = variational_auto_encoder.decode(latents, return_dict=False)[0]
    print(f"   Decoded Image Shape: {image.shape}")
    print(f"   Decoded Image (first 20): {image.flatten()[:20]}")
    print(f"   Image dtype {image.dtype}")
    return image



def convert_image_to_pil_png(image: torch.Tensor, image_name: str) -> None:
    image_magik = (image / 2 + 0.5).clamp(0, 1)
    print(f"image_magik dtype: {image_magik.dtype}")
    image_magik = image_magik.cpu().permute(0, 2, 3, 1).float().detach().numpy()
    image_magik = (image_magik * 255).round().astype("uint8")
    print(f"image 20 : {image_magik.flatten()[:20]}")
    first_image = image_magik[0]
    print(f"image_hand[0] 20 : {first_image.flatten()[:20]}")
    np.save(f"{image_name}.npy", first_image)
    first_image_pil = Image.fromarray(first_image)
    first_image_pil.save(f"{image_name}.png")


def save_encoding_embeded_from_tokens(token_encoded_embeds: torch.Tensor, text_ids: torch.Tensor, file_prefix: str) -> None:
    np.save(f"/Users/kevin/zml/{file_prefix}_embeds.npy", token_encoded_embeds.cpu().detach().numpy())
    np.save(f"/Users/kevin/zml/{file_prefix}_text_ids.npy", text_ids.cpu().detach().numpy())

def run_pipline():
    repo_id = "black-forest-labs/FLUX.2-klein-4B"
    prompt = "A flying surperman style cat"
    img_dim: int = 128

    if os.path.exists("/Users/kevin/zml/flux_klein_notebook_embeds.npy") and os.path.exists("/Users/kevin/zml/flux_klein_notebook_text_ids.npy"):
        print("\n>>> Loading Embeds...")
        token_encoded_embeds = torch.from_numpy(np.load("/Users/kevin/zml/flux_klein_notebook_embeds.npy"))
        text_ids = torch.from_numpy(np.load("/Users/kevin/zml/flux_klein_notebook_text_ids.npy"))
    else:
        print("\n>>> Tokenizing Prompt...")
        token: dict[str, torch.Tensor] = get_tokens_from_prompt(repo_id=repo_id, prompt=prompt, max_length=20)
        print(f"input_ids: {token['input_ids'].flatten()[:20]}")
        print(f"attention_mask: {token['attention_mask'].flatten()[:20]}")
        print("\n>>> Encoding Prompt...")
        token_encoded_embeds, text_ids = get_encoding_embeded_from_tokens(repo_id=repo_id, tokens=token)
        save_encoding_embeded_from_tokens(token_encoded_embeds, text_ids, "flux_klein_notebook")

    print(f"    text_ids (first 20). {text_ids.shape} : {text_ids.flatten()[:20]}")
    print(f"    token_encoded_embeds (first 20). {token_encoded_embeds.shape} : {token_encoded_embeds.flatten()[:20]}")

    print("\n>>>Preparing Latents...")
    transformer = Flux2Transformer2DModel.from_pretrained(
        repo_id,
        subfolder="transformer"
    )
    latents, latent_ids = get_latents(transformer=transformer, height=img_dim, width=img_dim)
    print(f"    Latents (first 20). {latents.shape} : {latents.flatten()[:20]}")
    print(f"    Latent_ids (first 20). {latent_ids.shape} : {latent_ids.flatten()[:20]}")
    
    print("\n>>> Preparing Timesteps...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        repo_id,
        subfolder="scheduler"
    )
    print(f"    Scheduler Sigmas (first 20): {scheduler.sigmas.flatten()[:20]}")
    latents_out: torch.Tensor = schedule(
        transformer=transformer,
        scheduler=scheduler,
        latents=latents,
        latent_ids=latent_ids,
        prompt_embeds=token_encoded_embeds,
        text_ids=text_ids,
        num_inference_steps=1)

    print(f"    Latents Out (first 20). {latents_out.shape} : {latents_out.flatten().tolist()[:20]}")

    # Stopping before VAE decoding for now

    print("\n>>> Decoding Latents...")
    variational_auto_encoder = AutoencoderKLFlux2.from_pretrained(
        repo_id,
        subfolder="vae"
    )
    image_decoded: torch.Tensor = variational_auto_encode(variational_auto_encoder=variational_auto_encoder,latents=latents_out)

    print(f"    Image Decoded (first 20). {image_decoded.shape} : {image_decoded.flatten().tolist()[:20]}")

    convert_image_to_pil_png(image_decoded, "flux_klein_notebook_result")
    print("\n>>> Pipeline Complete.")

# %%

if __name__ == "__main__":
    run_pipline()

# %%
