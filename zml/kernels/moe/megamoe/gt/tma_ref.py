"""Run tma_ref.cu (DeepGEMM TMA copies on a 2-CTA cluster) on random data; save inputs/outputs.
Usage: python -I tma_ref.py <deep_gemm include dir> <out dir>"""
import os, sys
import numpy as np
import torch
from torch.utils.cpp_extension import load_inline
inc, out = sys.argv[1], sys.argv[2]
here = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(here, 'tma_ref.cu')).read() + r'''
#include <torch/extension.h>
void run(torch::Tensor a, torch::Tensor sf, torch::Tensor w, torch::Tensor so, torch::Tensor bs, torch::Tensor smem, torch::Tensor bo) {
    launch_tma((uint8_t*) a.data_ptr(), (uint32_t*) sf.data_ptr(), (uint8_t*) w.data_ptr(), (uint8_t*) so.data_ptr(), (uint8_t*) bs.data_ptr(), (uint8_t*) smem.data_ptr(), (uint8_t*) bo.data_ptr());
}
'''
mod = load_inline('megamoe_tma_ref', cpp_sources='void run(torch::Tensor a, torch::Tensor sf, torch::Tensor w, torch::Tensor so, torch::Tensor bs, torch::Tensor smem, torch::Tensor bo);',
                  cuda_sources=src, functions=['run'], extra_include_paths=[inc], extra_ldflags=['-lcuda'],
                  extra_cuda_cflags=['-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', '-U__CUDA_NO_BFLOAT16_CONVERSIONS__',
                                     '-U__CUDA_NO_HALF2_OPERATORS__', '-O3', '-std=c++20', '--expt-relaxed-constexpr', '-gencode=arch=compute_103a,code=sm_103a'])
rng = np.random.default_rng(7)
a = rng.integers(0, 256, 192 * 256, dtype=np.uint8)
sf = rng.integers(0, 2**32, 2 * 256, dtype=np.uint64).astype(np.uint32)
w = rng.integers(0, 256, 2 * 256 * 128, dtype=np.uint8)
bs = rng.integers(0, 256, 2048, dtype=np.uint8)
t = lambda x: torch.from_numpy(x.view(np.int8)).cuda()
so = torch.zeros(32 * 128, dtype=torch.int8, device='cuda')
smem = torch.zeros(2 * 34816, dtype=torch.int8, device='cuda')
bo = torch.zeros(2048, dtype=torch.int8, device='cuda')
mod.run(t(a), torch.from_numpy(sf.view(np.int32)).cuda(), t(w), so, t(bs), smem, bo)
torch.cuda.synchronize()
for name, x in (('tma_a', a), ('tma_sf', sf), ('tma_w', w), ('tma_bulk_src', bs)):
    x.tofile(os.path.join(out, name + '.bin'))
for name, x in (('tma_store_out', so), ('tma_smem_out', smem), ('tma_bulk_out', bo)):
    x.cpu().numpy().tofile(os.path.join(out, name + '.bin'))
sm = smem.cpu().numpy().view(np.uint8).reshape(2, 34816)
print('ok; unwritten smem words per cta:', [(sm[c].view(np.uint32) == 0xdeadbeef).sum() for c in range(2)],
      'bulk ok', bool((bo.cpu().numpy().view(np.uint8) == bs).all()))
