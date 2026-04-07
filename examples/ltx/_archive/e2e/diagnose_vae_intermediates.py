"""Diagnose VAE decoder divergence by replaying step-by-step with manual ops.

Loads the Python reference activations + checkpoint weights, then manually
applies each step (denorm, conv_in, up_blocks, etc.) using different strategies
(zero-pad vs reflect-pad, etc.) and compares against the reference intermediates.

Usage (on GPU server):
  cd /root/repos/LTX-2
  uv run /root/repos/zml/examples/ltx/e2e/diagnose_vae_intermediates.py \
      --checkpoint /root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors \
      --activations /root/e2e_demo/vae_ref_small/vae_activations.safetensors
"""

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--activations", type=Path, required=True)
    return p.parse_args()


def psnr_db(ref, test, peak=2.0):
    """PSNR in dB, default peak=2 for [-1,1] signal range."""
    mse = ((ref.float() - test.float()) ** 2).mean().item()
    if mse == 0:
        return float('inf')
    return 10 * np.log10(peak ** 2 / mse)


def compare(name, ref, test):
    """Print comparison metrics between ref and test tensors."""
    diff = (ref.float() - test.float()).abs()
    max_abs = diff.max().item()
    mean_abs = diff.mean().item()
    db = psnr_db(ref, test)
    status = "OK" if db > 40 else "FAIL"
    print(f"  {name:30s}: max_abs={max_abs:.6f} mean_abs={mean_abs:.6f} PSNR={db:.1f} dB [{status}]")
    return db


def load_conv3d_weights(ckpt, prefix):
    """Load conv3d weight + bias from checkpoint with standard CausalConv3d key pattern."""
    w = ckpt.get_tensor(f"{prefix}.conv.weight")
    b = ckpt.get_tensor(f"{prefix}.conv.bias")
    return w, b


def causal_conv3d_noncausal_zeropad(x, weight, bias):
    """Zig-style: temporal replicate-pad + spatial zero-pad via conv3d."""
    first = x[:, :, :1, :, :]
    last = x[:, :, -1:, :, :]
    padded = torch.cat([first, x, last], dim=2)
    out = F.conv3d(padded, weight, bias=None, padding=(0, 1, 1))
    out = out + bias.view(1, -1, 1, 1, 1)
    return out


def causal_conv3d_noncausal_reflect(x, weight, bias):
    """Zig-style with reflect: temporal replicate-pad + spatial reflect-pad."""
    first = x[:, :, :1, :, :]
    last = x[:, :, -1:, :, :]
    padded = torch.cat([first, x, last], dim=2)
    # Reflect pad spatial: (W_left, W_right, H_left, H_right)
    padded = F.pad(padded, (1, 1, 1, 1), mode='reflect')
    out = F.conv3d(padded, weight, bias=None, padding=0)
    out = out + bias.view(1, -1, 1, 1, 1)
    return out


def pixel_norm_f32(x, eps=1e-8):
    """PixelNorm computed in f32 (Zig impl)."""
    xf = x.float()
    rms = torch.sqrt(torch.mean(xf ** 2, dim=1, keepdim=True) + eps)
    return (xf / rms).to(x.dtype)


def pixel_norm_python(x, eps=1e-6):
    """PixelNorm as in the Python ltx_core (eps=1e-6, native dtype)."""
    return x / torch.sqrt(torch.mean(x ** 2, dim=1, keepdim=True) + eps)


@torch.inference_mode()
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load activations
    print("Loading reference activations...")
    act = {}
    with safe_open(str(args.activations), framework="pt", device=str(device)) as f:
        for key in f.keys():
            act[key] = f.get_tensor(key)
            print(f"  {key:30s}: {list(act[key].shape)}")

    # Load checkpoint
    print("\nLoading checkpoint weights...")
    ckpt = safe_open(str(args.checkpoint), framework="pt", device=str(device))

    # List VAE decoder keys
    vae_keys = [k for k in ckpt.keys() if k.startswith("vae.decoder.")]
    print(f"  Found {len(vae_keys)} vae.decoder.* keys")
    pcs_keys = [k for k in ckpt.keys() if k.startswith("vae.per_channel")]
    print(f"  Found {len(pcs_keys)} vae.per_channel* keys")
    for k in pcs_keys:
        print(f"    {k}: {list(ckpt.get_tensor(k).shape)}")

    # ========================================================================
    # Step 0: Verify input
    # ========================================================================
    print("\n=== Step 0: Input latent ===")
    input_latent = act["input_latent"]
    print(f"  shape: {list(input_latent.shape)}, dtype: {input_latent.dtype}")

    # ========================================================================
    # Step 1: Denormalization
    # ========================================================================
    print("\n=== Step 1: Denormalization ===")
    mean_of_means = ckpt.get_tensor("vae.per_channel_statistics.mean-of-means").to(device=device, dtype=dtype)
    std_of_means = ckpt.get_tensor("vae.per_channel_statistics.std-of-means").to(device=device, dtype=dtype)

    # Zig: latent * std + mean
    denorm_zig = input_latent * std_of_means.view(1, -1, 1, 1, 1) + mean_of_means.view(1, -1, 1, 1, 1)
    compare("denorm (Zig logic)", act["after_denorm"], denorm_zig)

    # ========================================================================
    # Step 2: conv_in
    # ========================================================================
    print("\n=== Step 2: conv_in ===")
    conv_in_w, conv_in_b = load_conv3d_weights(ckpt, "vae.decoder.conv_in")
    print(f"  conv_in weight: {list(conv_in_w.shape)}, bias: {list(conv_in_b.shape)}")

    # Use the REFERENCE after_denorm for this step to isolate conv_in behavior
    denorm_ref = act["after_denorm"]

    conv_in_zero = causal_conv3d_noncausal_zeropad(denorm_ref, conv_in_w, conv_in_b)
    compare("conv_in (zero-pad)", act["after_conv_in"], conv_in_zero)

    # Reflect pad for 5D requires padding size 6 in PyTorch; test via 2D reshape approach
    try:
        conv_in_refl = causal_conv3d_noncausal_reflect(denorm_ref, conv_in_w, conv_in_b)
        compare("conv_in (reflect-pad)", act["after_conv_in"], conv_in_refl)
    except Exception as e:
        print(f"  conv_in (reflect-pad): SKIPPED ({e.__class__.__name__})")

    # Use zero-pad (matches reference best)
    conv_fn = causal_conv3d_noncausal_zeropad
    pad_name = "zero"

    # ========================================================================
    # Step 3: up_blocks[0] = UNetMidBlock3D (2 ResBlocks)
    # ========================================================================
    print("\n=== Step 3: up_blocks[0] (2 ResBlocks @ 1024) ===")

    # Load ResBlock weights for up_blocks.0
    def load_resblock_weights(prefix):
        c1w = ckpt.get_tensor(f"{prefix}.conv1.conv.weight")
        c1b = ckpt.get_tensor(f"{prefix}.conv1.conv.bias")
        c2w = ckpt.get_tensor(f"{prefix}.conv2.conv.weight")
        c2b = ckpt.get_tensor(f"{prefix}.conv2.conv.bias")
        return (c1w, c1b), (c2w, c2b)

    def run_resblock(x, rb_weights, norm_fn, conv_fn):
        (c1w, c1b), (c2w, c2b) = rb_weights
        h = norm_fn(x)
        h = F.silu(h)
        h = conv_fn(h, c1w, c1b)
        h = norm_fn(h)
        h = F.silu(h)
        h = conv_fn(h, c2w, c2b)
        return h + x

    # Use the reference after_conv_in for this step
    x_ref = act["after_conv_in"]

    # Test with Zig-style PixelNorm (f32, eps=1e-8)
    rb0_w = load_resblock_weights("vae.decoder.up_blocks.0.res_blocks.0")
    rb1_w = load_resblock_weights("vae.decoder.up_blocks.0.res_blocks.1")

    x_zig = run_resblock(x_ref, rb0_w, pixel_norm_f32, conv_fn)
    x_zig = run_resblock(x_zig, rb1_w, pixel_norm_f32, conv_fn)
    compare("up0 (Zig norm, zero-pad)", act["after_up0"], x_zig)

    # Test with Python-style PixelNorm (native bf16, eps=1e-6)
    x_py = run_resblock(x_ref, rb0_w, pixel_norm_python, conv_fn)
    x_py = run_resblock(x_py, rb1_w, pixel_norm_python, conv_fn)
    compare("up0 (Python norm, zero-pad)", act["after_up0"], x_py)

    # ========================================================================
    # Step 4: up_blocks[1] (DepthToSpace)
    # ========================================================================
    print("\n=== Step 4: up_blocks[1] (DepthToSpace stride=(2,2,2)) ===")
    from einops import rearrange
    d2s1_w = ckpt.get_tensor("vae.decoder.up_blocks.1.conv.conv.weight")
    d2s1_b = ckpt.get_tensor("vae.decoder.up_blocks.1.conv.conv.bias")
    print(f"  D2S conv weight: {list(d2s1_w.shape)}, bias: {list(d2s1_b.shape)}")

    x_d2s_in = act["after_up0"]
    # Apply conv + depth-to-space
    x_d2s = conv_fn(x_d2s_in, d2s1_w, d2s1_b)
    print(f"  After conv: {list(x_d2s.shape)}")

    # Rearrange: "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)"
    x_d2s = rearrange(x_d2s, "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)",
                       p1=2, p2=2, p3=2)
    # Remove first frame (temporal upsample)
    x_d2s = x_d2s[:, :, 1:, :, :]
    compare("up1 D2S (Python rearrange)", act["after_up1"], x_d2s)

    # Also test with Zig-style D2S (explicit reshape + transpose)
    x_d2s2 = conv_fn(x_d2s_in, d2s1_w, d2s1_b)
    B, C_total, Fr, H, W = x_d2s2.shape
    p1, p2, p3 = 2, 2, 2
    C = C_total // (p1 * p2 * p3)
    x_d2s2 = x_d2s2.reshape(B, C, p1, p2, p3, Fr, H, W)
    x_d2s2 = x_d2s2.permute(0, 1, 5, 2, 6, 3, 7, 4)  # [B,C,F,p1,H,p2,W,p3]
    x_d2s2 = x_d2s2.reshape(B, C, Fr * p1, H * p2, W * p3)
    x_d2s2 = x_d2s2[:, :, 1:, :, :]
    compare("up1 D2S (Zig reshape+permute)", act["after_up1"], x_d2s2)

    # ========================================================================
    # Full pipeline replay: chain all steps using Python norm + zero-pad
    # Report cumulative error at each intermediate
    # ========================================================================
    print("\n=== Full pipeline replay (Python norm, zero-pad, chained) ===")
    from einops import rearrange

    x = act["input_latent"]

    # Denorm
    x = x * std_of_means.view(1, -1, 1, 1, 1) + mean_of_means.view(1, -1, 1, 1, 1)
    compare("after_denorm", act["after_denorm"], x)

    # conv_in
    x = causal_conv3d_noncausal_zeropad(x, conv_in_w, conv_in_b)
    compare("after_conv_in", act["after_conv_in"], x)

    # up_blocks.0: 2 ResBlocks
    for j in range(2):
        w = load_resblock_weights(f"vae.decoder.up_blocks.0.res_blocks.{j}")
        x = run_resblock(x, w, pixel_norm_python, conv_fn)
    compare("after_up0", act["after_up0"], x)

    # up_blocks.1: D2S (2,2,2)
    d2s_w = ckpt.get_tensor("vae.decoder.up_blocks.1.conv.conv.weight")
    d2s_b = ckpt.get_tensor("vae.decoder.up_blocks.1.conv.conv.bias")
    x = conv_fn(x, d2s_w, d2s_b)
    x = rearrange(x, "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)", p1=2, p2=2, p3=2)
    x = x[:, :, 1:, :, :]
    compare("after_up1", act["after_up1"], x)

    # up_blocks.2: 2 ResBlocks
    for j in range(2):
        w = load_resblock_weights(f"vae.decoder.up_blocks.2.res_blocks.{j}")
        x = run_resblock(x, w, pixel_norm_python, conv_fn)
    compare("after_up2", act["after_up2"], x)

    # up_blocks.3: D2S (2,2,2)
    d2s_w = ckpt.get_tensor("vae.decoder.up_blocks.3.conv.conv.weight")
    d2s_b = ckpt.get_tensor("vae.decoder.up_blocks.3.conv.conv.bias")
    x = conv_fn(x, d2s_w, d2s_b)
    x = rearrange(x, "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)", p1=2, p2=2, p3=2)
    x = x[:, :, 1:, :, :]
    compare("after_up3", act["after_up3"], x)

    # up_blocks.4: 4 ResBlocks
    for j in range(4):
        w = load_resblock_weights(f"vae.decoder.up_blocks.4.res_blocks.{j}")
        x = run_resblock(x, w, pixel_norm_python, conv_fn)
    compare("after_up4", act["after_up4"], x)

    # up_blocks.5: D2S (2,1,1)
    d2s_w = ckpt.get_tensor("vae.decoder.up_blocks.5.conv.conv.weight")
    d2s_b = ckpt.get_tensor("vae.decoder.up_blocks.5.conv.conv.bias")
    x = conv_fn(x, d2s_w, d2s_b)
    x = rearrange(x, "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)", p1=2, p2=1, p3=1)
    x = x[:, :, 1:, :, :]
    compare("after_up5", act["after_up5"], x)

    # up_blocks.6: 6 ResBlocks
    for j in range(6):
        w = load_resblock_weights(f"vae.decoder.up_blocks.6.res_blocks.{j}")
        x = run_resblock(x, w, pixel_norm_python, conv_fn)
    compare("after_up6", act["after_up6"], x)

    # up_blocks.7: D2S (1,2,2)
    d2s_w = ckpt.get_tensor("vae.decoder.up_blocks.7.conv.conv.weight")
    d2s_b = ckpt.get_tensor("vae.decoder.up_blocks.7.conv.conv.bias")
    x = conv_fn(x, d2s_w, d2s_b)
    x = rearrange(x, "b (c p1 p2 p3) f h w -> b c (f p1) (h p2) (w p3)", p1=1, p2=2, p3=2)
    # No frame removal for p1=1
    compare("after_up7", act["after_up7"], x)

    # up_blocks.8: 4 ResBlocks
    for j in range(4):
        w = load_resblock_weights(f"vae.decoder.up_blocks.8.res_blocks.{j}")
        x = run_resblock(x, w, pixel_norm_python, conv_fn)
    compare("after_up8", act["after_up8"], x)

    # PixelNorm + SiLU
    x = pixel_norm_python(x)
    x = F.silu(x)
    compare("after_norm_silu", act["after_norm_silu"], x)

    # conv_out
    conv_out_w = ckpt.get_tensor("vae.decoder.conv_out.conv.weight")
    conv_out_b = ckpt.get_tensor("vae.decoder.conv_out.conv.bias")
    x = conv_fn(x, conv_out_w, conv_out_b)
    compare("after_conv_out", act["after_conv_out"], x)

    # unpatchify
    x = rearrange(x, "b (c p r q) f h w -> b c (f p) (h q) (w r)", p=1, q=4, r=4)
    compare("output", act["output"], x)

    print("\nDiagnostics complete.")


if __name__ == "__main__":
    main()
