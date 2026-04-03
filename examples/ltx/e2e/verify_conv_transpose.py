#!/usr/bin/env python3
"""Verify ConvTranspose1d output by comparing manual F.conv_transpose1d with model."""
import safetensors
import torch
import torch.nn.functional as F

# Load weight
with safetensors.safe_open('/root/models/ltx-2.3/ltx-2.3-22b-distilled.safetensors', framework='pt') as f:
    w = f.get_tensor('vocoder.vocoder.ups.0.weight')
    b = f.get_tensor('vocoder.vocoder.ups.0.bias')
print(f'ups.0.weight shape: {list(w.shape)} {w.dtype}')
print(f'ups.0.bias shape: {list(b.shape)} {b.dtype}')

# Load after_conv_pre
with safetensors.safe_open('/root/e2e_demo/vocoder_ref/vocoder_stages.safetensors', framework='pt') as f:
    after_conv_pre = f.get_tensor('after_conv_pre').cuda()
print(f'after_conv_pre shape: {list(after_conv_pre.shape)}')

# PyTorch ConvTranspose1d: stride=5, padding=3, kernel=11
result_pt = F.conv_transpose1d(after_conv_pre, w.cuda().float(), b.cuda().float(), stride=5, padding=3)
print(f'PyTorch conv_transpose1d shape: {list(result_pt.shape)}')
print(f'result[:8]: {result_pt[0,0,:8].tolist()}')

# Now simulate what MLIR does: dilate input, then conv with flipped kernel
# lhs_dilation=5: insert 4 zeros between each element
# padding = k - pytorch_padding - 1 = 11 - 3 - 1 = 7
# window_reversal = True (flip kernel)
# kernel dims: output=1, input=0 (transposed)

x = after_conv_pre  # [1, 1536, 8]
# Dilate: insert 4 zeros between elements -> [1, 1536, 36]
x_dilated = torch.zeros(1, 1536, (8-1)*5 + 1, device='cuda')
x_dilated[:, :, ::5] = x
print(f'Dilated shape: {list(x_dilated.shape)}')

# Pad with 7 on each side
x_padded = F.pad(x_dilated, (7, 7))
print(f'Padded shape: {list(x_padded.shape)}')

# Flip the kernel (window_reversal=True)
w_flipped = w.cuda().float().flip(2)  # flip along spatial dim

# Regular conv1d with transposed weight
# w is [1536, 768, 11] -> for regular conv1d, we need [C_out, C_in, K] = [768, 1536, 11]
# With kernel_output_feature_dimension=1, kernel_input_feature_dimension=0:
# That means w[input=0, output=1, spatial=2] = [1536, 768, 11]
# So we permute to standard [C_out, C_in, K] = [768, 1536, 11]
w_standard = w_flipped.permute(1, 0, 2)  # [768, 1536, 11]
result_manual = F.conv1d(x_padded, w_standard)
print(f'Manual dilated conv shape: {list(result_manual.shape)}')
print(f'manual[:8]: {result_manual[0,0,:8].tolist()}')

# Add bias
result_manual = result_manual + b.cuda().float().view(1, -1, 1)
print(f'After bias[:8]: {result_manual[0,0,:8].tolist()}')

# Compare
diff = (result_pt - result_manual).abs()
print(f'Max diff: {diff.max().item():.8f}')
print(f'Match: {torch.allclose(result_pt, result_manual, atol=1e-5)}')

# Also check WITHOUT flipping (window_reversal=False would mean this)
w_noflip_standard = w.cuda().float().permute(1, 0, 2)  # [768, 1536, 11]
result_noflip = F.conv1d(x_padded, w_noflip_standard) + b.cuda().float().view(1, -1, 1)
print(f'\nWithout kernel flip:')
print(f'noflip[:8]: {result_noflip[0,0,:8].tolist()}')
diff_noflip = (result_pt - result_noflip).abs()
print(f'Max diff (no flip): {diff_noflip.max().item():.8f}')
print(f'Match (no flip): {torch.allclose(result_pt, result_noflip, atol=1e-5)}')
